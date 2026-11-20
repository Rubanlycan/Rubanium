"""Local Wan 2.2 TI2V-5B studio with text and named-image conditioning."""

from __future__ import annotations

import atexit
import base64
import gc
import ctypes
import hashlib
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Iterable

import gradio as gr
import imageio_ffmpeg
import torch
from PIL import Image
from settings import configured_path, load_project_env


APP_DIR = Path(__file__).resolve().parent
load_project_env(APP_DIR / ".env")


MODELS_DIR = configured_path("RUBANIUM_MODELS_DIR", Path("models"), APP_DIR)
OFFLOAD_ROOT = configured_path("RUBANIUM_OFFLOAD_DIR", Path("offload"), APP_DIR)
OUTPUT_DIR = configured_path("RUBANIUM_OUTPUT_DIR", Path("outputs"), APP_DIR)
LOG_DIR = configured_path("RUBANIUM_LOG_DIR", Path(".runtime") / "logs", APP_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATHS = {
    "LTX-Video 2B (recommended for 8 GB VRAM)": MODELS_DIR / "ltx-video-2b" / "components",
    "Wan 2.2 TI2V-5B": MODELS_DIR / "wan2.2-ti2v-5b-diffusers",
}
DEFAULT_MODEL = "LTX-Video 2B (recommended for 8 GB VRAM)"
LTX_LABEL = "LTX-Video 2B v0.9.5"
WAN_LABEL = "Wan 2.2 TI2V-5B"
OFFLOAD_DIR = OFFLOAD_ROOT / f"session-{os.getpid()}-{uuid.uuid4().hex[:8]}"
OFFLOAD_MIN_FREE_GIB = 45
SERVER_NAME = os.environ.get("RUBANIUM_SERVER_NAME", "127.0.0.1")
SERVER_PORT = int(os.environ.get("RUBANIUM_SERVER_PORT", "7860"))
SHARE_APP = os.environ.get("RUBANIUM_SHARE", "false").strip().lower() in {"1", "true", "yes", "on"}

DEFAULT_NEGATIVE = "worst quality, inconsistent motion, blurry, jittery, distorted, watermark, logo, subtitles"
DEFAULT_PROMPT = (
    "A small fox trots through a misty pine forest at sunrise, its coat catching warm golden light, "
    "gentle handheld camera movement, natural cinematic footage."
)
MAX_REFERENCES = 6
MAX_HISTORY = 12
FPS = 24
WAN_TOTAL_GB = 34.2
WAN_TOTAL_BYTES = int(WAN_TOTAL_GB * 1_000_000_000)
LTX_TOTAL_GB = 25.6
LTX_TOTAL_BYTES = int(LTX_TOTAL_GB * 1_000_000_000)

DTYPE = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else torch.float16
)
pipeline = None
pipeline_image = None
pipeline_model: str | None = None


def flush() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def system_ram_gib() -> float | None:
    if os.name != "nt":
        return None

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return status.ullTotalPhys / (1024**3)


def system_memory_stats() -> tuple[int, float, float] | None:
    """Return physical RAM load percent, used GiB, and total GiB on Windows."""
    if os.name != "nt":
        return None

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    total = status.ullTotalPhys / (1024**3)
    used = (status.ullTotalPhys - status.ullAvailPhys) / (1024**3)
    return int(status.dwMemoryLoad), used, total


def resource_usage_html() -> str:
    ram = system_memory_stats()
    if ram is None:
        ram_display = "RAM unavailable"
    else:
        percent, used, total = ram
        ram_display = f"RAM <b>{percent}%</b><small>{used:.1f} / {total:.1f} GiB</small>"

    if not torch.cuda.is_available():
        vram_display = "VRAM unavailable"
    else:
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(torch.cuda.current_device())
            total = total_bytes / (1024**3)
            used = (total_bytes - free_bytes) / (1024**3)
            percent = int(round(100 * (total_bytes - free_bytes) / max(total_bytes, 1)))
            vram_display = f"VRAM <b>{percent}%</b><small>{used:.1f} / {total:.1f} GiB</small>"
        except Exception:
            vram_display = "VRAM unavailable"

    return (
        '<div class="resource-chip">'
        f'<span>{ram_display}</span><i></i><span>{vram_display}</span>'
        "</div>"
    )


def wan_model_ready() -> bool:
    transformer = MODEL_PATHS[WAN_LABEL] / "transformer"
    text_encoder = MODEL_PATHS[WAN_LABEL] / "text_encoder"
    vae = MODEL_PATHS[WAN_LABEL] / "vae"
    return all(
        [
            (MODEL_PATHS[WAN_LABEL] / "model_index.json").is_file(),
            (transformer / "config.json").is_file(),
            (transformer / "diffusion_pytorch_model.safetensors.index.json").is_file(),
            len(list(transformer.glob("diffusion_pytorch_model-*-of-00005.safetensors"))) >= 5,
            (text_encoder / "config.json").is_file(),
            len(list(text_encoder.glob("model-*-of-00003.safetensors"))) >= 3,
            (vae / "config.json").is_file(),
            bool(list(vae.glob("*.safetensors"))),
        ]
    )


def ltx_model_ready() -> bool:
    components = MODEL_PATHS[DEFAULT_MODEL]
    return (
        (components / "model_index.json").is_file()
        and (components / "scheduler" / "scheduler_config.json").is_file()
        and (components / "tokenizer" / "tokenizer_config.json").is_file()
        and (components / "text_encoder" / "config.json").is_file()
        and bool(list((components / "text_encoder").glob("*.safetensors")))
        and bool(list((components / "transformer").glob("*.safetensors")))
        and (components / "vae" / "config.json").is_file()
        and bool(list((components / "vae").glob("*.safetensors")))
    )


def model_download_metrics(model_name: str) -> tuple[int, float, bool]:
    """Return deduplicated downloaded bytes, percentage, and active state."""
    is_wan = "Wan" in model_name
    model_dir = MODEL_PATHS[WAN_LABEL if is_wan else DEFAULT_MODEL]
    total_bytes = WAN_TOTAL_BYTES if is_wan else LTX_TOTAL_BYTES
    marker_name = "wan-download.pid" if is_wan else "ltx-download.pid"
    cache_dir = model_dir / ".cache" / "huggingface" / "download"
    bytes_by_artifact: dict[str, int] = {}
    if model_dir.exists():
        for path in model_dir.rglob("*"):
            try:
                if path.is_file() and ".cache" not in path.parts:
                    artifact = path.relative_to(model_dir).as_posix()
                    bytes_by_artifact[artifact] = path.stat().st_size
            except OSError:
                continue

    active = False
    matched_partials: set[Path] = set()
    lock_paths = list(cache_dir.rglob("*.lock")) if cache_dir.exists() else []
    for lock_path in lock_paths:
        try:
            from filelock import FileLock

            lock_probe = FileLock(str(lock_path))
            try:
                lock_probe.acquire(timeout=0)
                lock_probe.release()
            except TimeoutError:
                active = True
        except Exception:
            pass

    pid_marker = LOG_DIR / marker_name
    try:
        pid = int(pid_marker.read_text(encoding="ascii").strip())
        os.kill(pid, 0)
        active = True
    except (OSError, ValueError):
        pid_marker.unlink(missing_ok=True)

    if cache_dir.exists():
        # Hugging Face stores each partial under a hash of its `.metadata` name.
        # Match it back to its destination, then count max(final, partial) once.
        for metadata_path in cache_dir.rglob("*.metadata"):
            try:
                metadata_lines = metadata_path.read_text(encoding="utf-8").splitlines()
                if len(metadata_lines) < 2:
                    continue
                etag = metadata_lines[1].strip()
                short_hash = base64.urlsafe_b64encode(
                    hashlib.sha1(metadata_path.name.encode("utf-8")).digest()
                ).decode("ascii")
                artifact_rel = metadata_path.relative_to(cache_dir).as_posix()[:-len(".metadata")]
                artifact_bytes = bytes_by_artifact.get(artifact_rel, 0)
                for partial in metadata_path.parent.glob(f"{short_hash}.{etag}*.incomplete"):
                    try:
                        matched_partials.add(partial.resolve())
                        artifact_bytes = max(artifact_bytes, partial.stat().st_size)
                        if partial.stat().st_size and partial.stat().st_mtime > (time.time() - 90):
                            active = True
                    except OSError:
                        continue
                if artifact_bytes:
                    bytes_by_artifact[artifact_rel] = artifact_bytes
            except OSError:
                continue

        # Preserve progress for partials without sidecar metadata, while avoiding
        # duplicate `.incomplete` variants for the same Hugging Face artifact.
        orphan_partials: dict[tuple[str, str], int] = {}
        for partial in cache_dir.rglob("*.incomplete"):
            try:
                if partial.resolve() in matched_partials:
                    continue
                relative_parent = partial.parent.relative_to(cache_dir).as_posix()
                artifact_hash = partial.name.split(".", 1)[0]
                key = (relative_parent, artifact_hash)
                orphan_partials[key] = max(orphan_partials.get(key, 0), partial.stat().st_size)
                if partial.stat().st_size and partial.stat().st_mtime > (time.time() - 90):
                    active = True
            except OSError:
                continue
        downloaded = sum(bytes_by_artifact.values()) + sum(orphan_partials.values())
    else:
        downloaded = sum(bytes_by_artifact.values())

    # A few bytes of manifest or server-side estimate variance must not make the
    # UI report more than the configured model total.
    downloaded = min(downloaded, total_bytes)
    percent = 100.0 * downloaded / max(total_bytes, 1)
    return downloaded, percent, active


def wan_download_metrics() -> tuple[int, float, bool]:
    return model_download_metrics(WAN_LABEL)


def _format_gb(byte_count: int) -> str:
    return f"{byte_count / 1_000_000_000:.1f} GB"


def model_download_status_html(model_name: str = DEFAULT_MODEL) -> str:
    is_wan = "Wan" in model_name
    ready = wan_model_ready() if is_wan else ltx_model_ready()
    total_gb = WAN_TOTAL_GB if is_wan else LTX_TOTAL_GB
    if ready:
        return '<span class="downloaded-label"><span class="download-check">✓</span>Downloaded</span>'

    downloaded, percent, active = model_download_metrics(model_name)
    if active:
        return (
            '<div class="download-progress-wrap">'
            f'<div class="download-progress-title">Downloading {model_name.split(" (")[0]} '
            f'<b>{percent:.0f}%</b><span>{_format_gb(downloaded)} / {total_gb:.1f} GB</span></div>'
            f'<progress class="download-progress" max="100" value="{percent:.1f}"></progress>'
            '</div>'
        )
    return ""


def download_button_update(model_name: str = DEFAULT_MODEL):
    is_wan = "Wan" in model_name
    ready = wan_model_ready() if is_wan else ltx_model_ready()
    downloaded, _, active = model_download_metrics(model_name) if not ready else (0, 0.0, False)
    model_label = "Wan 2.2" if is_wan else "LTX-Video 2B"
    label = f"Resume {model_label}" if downloaded else f"Download {model_label}"
    return gr.update(visible=not ready and not active, value=label, interactive=True)


def start_model_download(model_name: str):
    is_wan = "Wan" in model_name
    ready = wan_model_ready() if is_wan else ltx_model_ready()
    if ready:
        return model_download_status_html(model_name), download_button_update(model_name)
    _, _, active = model_download_metrics(model_name)
    if active:
        return model_download_status_html(model_name), download_button_update(model_name)

    log_dir = LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    slug = "wan" if is_wan else "ltx"
    stdout_path = log_dir / f"{slug}-download.stdout.log"
    marker_path = log_dir / f"{slug}-download.pid"
    try:
        with stdout_path.open("a", encoding="utf-8") as download_log:
            process = subprocess.Popen(
                [sys.executable, str(APP_DIR / "download_models.py"), "--models-dir", str(MODELS_DIR), "--model", slug],
                cwd=str(APP_DIR),
                stdin=subprocess.DEVNULL,
                stdout=download_log,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        marker_path.write_text(str(process.pid), encoding="ascii")
        return (
            f'<div class="download-progress-wrap"><div class="download-progress-title">Starting {model_name.split(" (")[0]} download…</div>'
            '<progress class="download-progress" max="100" value="0"></progress></div>',
            gr.update(visible=False),
        )
    except Exception as exc:
        model_label = model_name.split(" (")[0]
        return (
            f'<span class="download-state">Could not start download: {type(exc).__name__}: {exc}</span>',
            gr.update(visible=True, value=f"Resume {model_label}"),
        )


def refresh_status(model_name: str = DEFAULT_MODEL):
    return (
        resource_usage_html(),
        model_status(model_name),
        model_download_status_html(DEFAULT_MODEL),
        download_button_update(DEFAULT_MODEL),
        model_download_status_html(WAN_LABEL),
        download_button_update(WAN_LABEL),
    )


def select_model(model_name: str):
    return (
        gr.update(value=model_name),
        gr.update(open=False, label=model_name),
        gr.update(variant="primary" if model_name == DEFAULT_MODEL else "secondary"),
        gr.update(variant="primary" if model_name == WAN_LABEL else "secondary"),
    )


def model_status(model_name: str = DEFAULT_MODEL) -> str:
    model_dir = MODEL_PATHS.get(model_name, MODEL_PATHS[DEFAULT_MODEL])
    model_label = WAN_LABEL if "Wan" in model_name else LTX_LABEL
    config_dir = model_dir
    if not model_dir.is_dir():
        return (
            f"**Model not installed.** Run `python download_models.py` to place "
            f"{model_label} in `{model_dir}`."
        )
    if "Wan" in model_name and not wan_model_ready():
        downloaded, percent, active = wan_download_metrics()
        if active:
            return (
                f"**Wan 2.2 download in progress:** {percent:.0f}% "
                f"({_format_gb(downloaded)} / {WAN_TOTAL_GB:.1f} GB). "
                f"Files are going to `{model_dir}`."
            )
        if downloaded:
            return (
                f"**Wan 2.2 download paused:** {_format_gb(downloaded)} / "
                f"{WAN_TOTAL_GB:.1f} GB is present in `{model_dir}`. Resume it from the button."
            )
        return f"**Wan 2.2 is not downloaded yet** to `{model_dir}`."
    if "Wan" not in model_name and (
        not (config_dir / "model_index.json").exists()
        or not list((config_dir / "transformer").glob("*.safetensors"))
    ):
        return f"**{model_label} is missing or incomplete** in `{model_dir}`."
    weight_gib = sum(
        path.stat().st_size
        for path in config_dir.rglob("*.safetensors")
        if ".cache" not in path.parts
    ) / (1024**3)
    scratch_free_gib = shutil.disk_usage(OFFLOAD_ROOT.parent).free / (1024**3)
    if scratch_free_gib < OFFLOAD_MIN_FREE_GIB:
        return (
            f"**Offload space is low:** the model is installed, but the model drive has "
            f"{scratch_free_gib:.1f} GiB free; generation needs at least "
            f"{OFFLOAD_MIN_FREE_GIB} GiB for temporary staged weights."
        )
    vram_gib = (
        torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / (1024**3)
        if torch.cuda.is_available()
        else 0
    )
    ram_gib = system_ram_gib()
    if vram_gib < 24:
        if "Wan" in model_name:
            memory_advice = (
                "Wan's official single-GPU setup calls for at least 24 GiB; this app uses "
                "CPU/disk offload, but long clips or 720p may still run out of memory."
            )
        else:
            memory_advice = (
                "Disk-backed weight staging and tiled decoding are enabled. Long clips and "
                "720p still use substantial memory; 480p and fewer steps are the safer settings."
            )
        return (
            f"**{model_label} ready, low-memory mode enabled:** {weight_gib:.1f} GiB of weights; "
            f"this GPU has {vram_gib:.1f} GiB VRAM. {memory_advice} "
            f"System RAM: {ram_gib:.1f} GiB. Generation may be slow."
        )
    return f"**Ready:** {model_label} - local files in `{model_dir}`"


def _set_load_progress(progress, value: float, message: str, model_label: str = "") -> None:
    if progress is not None:
        percent = int(round(value * 100))
        prefix = f"{model_label} · " if model_label else ""
        progress(value, desc=f"{prefix}{message} - {percent}%")


def load_pipeline(progress=None, model_name: str = DEFAULT_MODEL, image_to_video: bool = False):
    global pipeline, pipeline_image, pipeline_model
    model_dir = MODEL_PATHS.get(model_name, MODEL_PATHS[DEFAULT_MODEL])
    is_wan = "Wan" in model_name
    model_label = WAN_LABEL if is_wan else LTX_LABEL
    if pipeline is not None and pipeline_model == model_name:
        return pipeline_image if image_to_video else pipeline
    if pipeline is not None or pipeline_image is not None:
        pipeline = None
        pipeline_image = None
        pipeline_model = None
        flush()
    if not (model_dir / "model_index.json").exists() or not list(
        (model_dir / "transformer").glob("*.safetensors")
    ):
        raise gr.Error(
            f"{model_label} is missing or incomplete at `{model_dir}`. "
            + ("The Wan download is still running; select LTX-Video 2B for now." if is_wan else "Verify the model download, then restart the app.")
        )

    text_encoder = transformer = vae = None
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for Wan 2.2 generation.")

        free_gib = shutil.disk_usage(OFFLOAD_ROOT.parent).free / (1024**3)
        if free_gib < OFFLOAD_MIN_FREE_GIB:
            raise RuntimeError(
                f"The offload drive needs {OFFLOAD_MIN_FREE_GIB} GiB free; "
                f"it currently has {free_gib:.1f} GiB."
            )

        device = torch.device("cuda", torch.cuda.current_device())
        OFFLOAD_DIR.mkdir(parents=True, exist_ok=True)
        text_offload_dir = OFFLOAD_DIR / "text_encoder"
        transformer_offload_dir = OFFLOAD_DIR / "transformer"
        for folder in (text_offload_dir, transformer_offload_dir):
            folder.mkdir(parents=True, exist_ok=True)
        atexit.register(shutil.rmtree, OFFLOAD_DIR, ignore_errors=True)

        _set_load_progress(progress, 0.03, "Preparing model disk offload", model_label)
        if is_wan:
            from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, WanPipeline, WanTransformer3DModel
            from transformers import T5TokenizerFast, UMT5EncoderModel

            tokenizer = T5TokenizerFast.from_pretrained(model_dir / "tokenizer", local_files_only=True)
            _set_load_progress(progress, 0.08, "Staging the UMT5 text encoder", model_label)
            text_encoder = UMT5EncoderModel.from_pretrained(
                model_dir / "text_encoder", torch_dtype=DTYPE, local_files_only=True,
                use_safetensors=True, low_cpu_mem_usage=True, offload_state_dict=True,
                device_map="auto", max_memory={device.index: "1GiB", "cpu": "4GiB"},
                offload_folder=str(text_offload_dir),
            )
            _set_load_progress(progress, 0.22, "Staging the Wan video transformer", model_label)
            transformer = WanTransformer3DModel.from_pretrained(
                model_dir / "transformer", torch_dtype=DTYPE, local_files_only=True,
                use_safetensors=True, low_cpu_mem_usage=True, offload_state_dict=True,
                device_map="auto", max_memory={device.index: "3GiB", "cpu": "4GiB"},
                offload_folder=str(transformer_offload_dir),
            )
            _set_load_progress(progress, 0.52, "Loading the Wan video decoder", model_label)
            vae = AutoencoderKLWan.from_pretrained(
                model_dir / "vae", torch_dtype=DTYPE, local_files_only=True,
                use_safetensors=True, low_cpu_mem_usage=True,
            ).to(device)
            pipe_class = WanPipeline
        else:
            from diffusers import AutoencoderKLLTXVideo, FlowMatchEulerDiscreteScheduler, LTXConditionPipeline, LTXVideoTransformer3DModel
            from transformers import T5EncoderModel, T5Tokenizer

            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_dir / "scheduler", local_files_only=True)
            tokenizer = T5Tokenizer.from_pretrained(model_dir / "tokenizer", local_files_only=True)
            _set_load_progress(progress, 0.08, "Staging the LTX text encoder", model_label)
            text_encoder = T5EncoderModel.from_pretrained(
                model_dir / "text_encoder", torch_dtype=DTYPE, local_files_only=True,
                use_safetensors=True, low_cpu_mem_usage=True, offload_state_dict=True,
                device_map="auto", max_memory={device.index: "1GiB", "cpu": "4GiB"},
                offload_folder=str(text_offload_dir),
            )
            _set_load_progress(progress, 0.22, "Staging the LTX video transformer", model_label)
            transformer = LTXVideoTransformer3DModel.from_pretrained(
                model_dir / "transformer", torch_dtype=DTYPE, local_files_only=True,
                use_safetensors=True, low_cpu_mem_usage=True,
            )
            transformer.enable_group_offload(
                onload_device=device,
                offload_device=torch.device("cpu"),
                offload_type="block_level",
                num_blocks_per_group=1,
                non_blocking=True,
                use_stream=True,
                record_stream=True,
                low_cpu_mem_usage=True,
                offload_to_disk_path=str(transformer_offload_dir),
            )
            _set_load_progress(progress, 0.52, "Loading the tiled LTX decoder", model_label)
            vae = AutoencoderKLLTXVideo.from_pretrained(
                model_dir / "vae", torch_dtype=DTYPE, local_files_only=True,
                use_safetensors=True, low_cpu_mem_usage=True,
            )
            vae.enable_tiling()
            vae.to(device)
            pipe_class = LTXConditionPipeline

        flush()
        _set_load_progress(progress, 0.8, "Assembling the local video pipeline", model_label)
        kwargs = dict(
            text_encoder=text_encoder, tokenizer=tokenizer, transformer=transformer,
            vae=vae, torch_dtype=DTYPE, local_files_only=True,
            low_cpu_mem_usage=True, use_safetensors=True,
        )
        if not is_wan:
            kwargs["scheduler"] = scheduler
        pipeline = pipe_class.from_pretrained(model_dir, **kwargs)
        if is_wan:
            pipeline_image = WanImageToVideoPipeline.from_pretrained(model_dir, **kwargs)
        else:
            pipeline_image = pipeline
        pipeline_model = model_name
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        _set_load_progress(progress, 1.0, f"{model_label} is ready")
        print(f"[INFO] {model_label} pipeline ready (dtype={DTYPE}).")
    except Exception as exc:
        pipeline = None
        pipeline_image = None
        pipeline_model = None
        text_encoder = transformer = vae = None
        flush()
        shutil.rmtree(OFFLOAD_DIR, ignore_errors=True)
        raise gr.Error(
            f"Could not load {model_label} with low-memory disk offload. Confirm the model "
            f"download completed and `{OFFLOAD_DIR}` is on a fast SSD. ({type(exc).__name__}: {exc})"
        ) from exc
    return pipeline_image if image_to_video else pipeline


def image_tag_for(path: str, used: set[str]) -> str:
    stem = Path(path).stem
    tag = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_-").lower() or "image"
    if tag[0].isdigit():
        tag = f"image_{tag}"
    base = tag
    suffix = 2
    while tag in used:
        tag = f"{base}_{suffix}"
        suffix += 1
    used.add(tag)
    return tag


def named_images(paths: Iterable[str] | None) -> dict[str, str]:
    result: dict[str, str] = {}
    used: set[str] = set()
    for path in list(paths or [])[:MAX_REFERENCES]:
        if path and Path(path).is_file():
            result[image_tag_for(path, used)] = path
    return result


def update_references(paths: list[str] | None):
    refs = named_images(paths)
    if not refs:
        return "Upload images to create named references. Example: `@bunny`", []
    tags = " · ".join(f"`@{name}`" for name in refs)
    preview = [(path, f"@{name}") for name, path in refs.items()]
    return f"**Available references:** {tags}\n\nMention a tag in the prompt to use that image.", preview


def _parse_prompt(prompt: str, refs: dict[str, str]) -> tuple[str, list[str]]:
    tags = re.findall(r"(?<!\w)@([A-Za-z0-9_-]+)", prompt or "")
    unique_tags = list(dict.fromkeys(tags))
    unknown = [tag for tag in unique_tags if tag not in refs]
    if unknown:
        raise gr.Error(
            "These image tags do not match an uploaded file: "
            + ", ".join(f"@{tag}" for tag in unknown)
        )
    selected = unique_tags
    cleaned = re.sub(
        r"(?<!\w)@([A-Za-z0-9_-]+)",
        lambda match: f"the subject matching the {match.group(1).replace('_', ' ')} reference image",
        prompt or "",
    )
    return cleaned.strip(), selected


def video_dimensions(resolution: str) -> tuple[int, int]:
    return (1280, 704) if str(resolution).startswith("720") else (832, 480)


def _validate_inputs(prompt: str, model_name: str, width: int, height: int, frames: int, steps: int) -> None:
    if not (prompt or "").strip():
        raise gr.Error("Enter a prompt describing the scene and motion.")
    if width % 32 or height % 32:
        raise gr.Error("The selected resolution must use dimensions divisible by 32.")
    modulus = 4 if "Wan" in model_name else 8
    if (frames - 1) % modulus:
        raise gr.Error(f"Frame count must be 4n + 1 for Wan or 8n + 1 for LTX.")
    if not 241 <= frames <= 721:
        raise gr.Error("Choose a duration from 10 to 30 seconds.")
    if not 1 <= steps <= 12:
        raise gr.Error("Inference steps must be between 1 and 12.")


def make_reference_image(refs: dict[str, str], selected_tags: list[str]) -> Image.Image:
    images = [Image.open(refs[tag]).convert("RGB") for tag in selected_tags]
    if len(images) == 1:
        return images[0]
    columns = 2
    rows = (len(images) + columns - 1) // columns
    cell = 512
    sheet = Image.new("RGB", (columns * cell, rows * cell), "#172033")
    for index, image in enumerate(images):
        image.thumbnail((cell - 24, cell - 24), Image.Resampling.LANCZOS)
        x = (index % columns) * cell + (cell - image.width) // 2
        y = (index // columns) * cell + (cell - image.height) // 2
        sheet.paste(image, (x, y))
    return sheet


def generate_video(
    prompt: str,
    uploaded_paths: list[str] | None,
    model_name: str,
    duration_seconds: int,
    resolution: str,
    negative_prompt: str,
    steps: int,
    guidance: float,
    seed: int,
    history: list | None,
    progress=gr.Progress(track_tqdm=True),
):
    refs = named_images(uploaded_paths)
    clean_prompt, selected_tags = _parse_prompt(prompt, refs)
    width, height = video_dimensions(resolution)
    frames = int(duration_seconds) * FPS + 1
    _validate_inputs(clean_prompt, model_name, width, height, frames, int(steps))
    is_wan = "Wan" in model_name
    model_label = WAN_LABEL if is_wan else LTX_LABEL
    progress(0, desc=f"{model_label} · Loading selected model")
    if refs and not selected_tags:
        progress(0, desc=f"{model_label} · 0% - Uploaded images are unused; add their @name tags to the prompt")

    pipe = load_pipeline(progress, model_name=model_name, image_to_video=bool(selected_tags) if is_wan else False)
    generator = torch.Generator(device="cpu")
    actual_seed = int(seed) if int(seed) >= 0 else int(torch.seed() % (2**31 - 1))
    generator.manual_seed(actual_seed)

    reference_image = make_reference_image(refs, selected_tags) if selected_tags and is_wan else None
    ltx_conditions = None
    if selected_tags and not is_wan:
        from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition

        ltx_conditions = [
            LTXVideoCondition(
                image=Image.open(refs[tag]).convert("RGB"),
                frame_index=0,
                strength=0.65,
            )
            for tag in selected_tags
        ]

    chunk_seconds = 2
    segment_count = (int(duration_seconds) + chunk_seconds - 1) // chunk_seconds
    filename = f"{'wan' if is_wan else 'ltx'}_{uuid.uuid4().hex[:10]}.mp4"
    out_path = OUTPUT_DIR / filename
    temp_path = OUTPUT_DIR / f".{filename}.partial.mp4"
    writer = None
    try:
        writer = imageio_ffmpeg.write_frames(
            str(temp_path),
            size=(int(width), int(height)),
            fps=FPS,
            quality=8,
            pix_fmt_in="rgb24",
            pix_fmt_out="yuv420p",
        )
        writer.send(None)
        elapsed_seconds = 0
        previous_frame = None
        for segment_index in range(segment_count):
            segment_duration = min(chunk_seconds, int(duration_seconds) - elapsed_seconds)
            segment_frames = segment_duration * FPS + 1
            pipe = load_pipeline(
                progress,
                model_name=model_name,
                image_to_video=bool(is_wan and (selected_tags or previous_frame is not None)),
            )

            def report_step(_pipe, step, _timestep, callback_kwargs):
                local_fraction = (step + 1) / max(int(steps), 1)
                overall = (elapsed_seconds + local_fraction * segment_duration) / int(duration_seconds)
                percent = int(round(overall * 100))
                progress(
                    overall,
                    desc=f"{model_label} · Segment {segment_index + 1}/{segment_count} - {percent}% - step {step + 1}/{int(steps)}",
                )
                return callback_kwargs

            call_args = dict(
                prompt=clean_prompt,
                negative_prompt=(negative_prompt or "").strip() or DEFAULT_NEGATIVE,
                width=int(width),
                height=int(height),
                num_frames=segment_frames,
                num_inference_steps=int(steps),
                guidance_scale=float(guidance),
                generator=generator,
                output_type="pil",
                callback_on_step_end=report_step,
            )
            if is_wan:
                if previous_frame is not None:
                    call_args["image"] = previous_frame
                elif reference_image is not None:
                    call_args["image"] = reference_image
            elif previous_frame is not None:
                from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition

                call_args["conditions"] = [
                    LTXVideoCondition(image=previous_frame, frame_index=0, strength=0.8)
                ]
            elif ltx_conditions is not None:
                call_args["conditions"] = ltx_conditions
            if not is_wan:
                call_args.update(
                    frame_rate=FPS,
                    image_cond_noise_scale=0.15,
                    decode_timestep=0.05,
                    decode_noise_scale=0.025,
                )

            with torch.inference_mode():
                result = pipe(**call_args)
            video_frames = result.frames[0]
            previous_frame = video_frames[-1].convert("RGB")
            # Keep only the previous segment's ending frame as the next start;
            # discard each repeated seam frame and stream frames straight to ffmpeg.
            for frame_index, frame in enumerate(video_frames):
                if segment_index > 0 and frame_index == 0:
                    continue
                writer.send(frame.convert("RGB").tobytes())
            elapsed_seconds += segment_duration
            del result, video_frames
            flush()
        writer.close()
        writer = None
        temp_path.replace(out_path)
    except gr.Error:
        if writer is not None:
            writer.close()
        temp_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        if writer is not None:
            writer.close()
        temp_path.unlink(missing_ok=True)
        flush()
        raise gr.Error(
            "Generation failed. Try LTX-Video 2B at 480p with 4 steps, "
            f"or fewer reference images. ({type(exc).__name__}: {exc})"
        ) from exc

    flush()
    prompt_label = (prompt or "").strip()
    image_label = ", ".join(f"@{tag}" for tag in selected_tags)
    caption = f"{prompt_label[:100]} · seed {actual_seed}"
    if image_label:
        caption = f"{image_label} · {caption}"
    new_history = [(str(out_path), caption)] + list(history or [])
    new_history = new_history[:MAX_HISTORY]
    duration = (int(frames) - 1) / FPS
    status = f"Generated **{duration:.0f}s** at {width}x{height} with {model_name} - seed `{actual_seed}`"
    return str(out_path), new_history, new_history, status


CSS = """
:root { color-scheme: dark; }
body, .gradio-container { background: #080b12 !important; }
.gradio-container { max-width: 1440px !important; }
#hero { padding: 30px 32px 24px; border: 1px solid #20293a; border-radius: 24px;
  background: radial-gradient(circle at 90% 0%, #293554 0, #121a2b 33%, #0d111b 75%); }
#hero h1 { margin: 0; font-size: clamp(2rem, 4vw, 3.4rem); letter-spacing: -0.04em; }
#hero p { color: #aebbd1; font-size: 1.05rem; margin: 10px 0 0; }
.panel { border: 1px solid #242d3d !important; border-radius: 18px !important; background: #101622 !important; }
.generate-btn { min-height: 54px !important; font-weight: 700 !important; }
#model-status { color: #c8d7ec; }
#model-picker { width: min(600px, calc(100vw - 32px)) !important;
  min-width: min(600px, calc(100vw - 32px)) !important; max-width: min(600px, calc(100vw - 32px)) !important;
  position: relative !important; z-index: 1005 !important; overflow: visible !important;
  margin-left: 0 !important; margin-right: auto !important; align-self: flex-start !important;
  border: 0 !important; border-radius: 26px !important; background: transparent !important; }
#model-picker .label-wrap { min-height: 74px !important; padding: 14px 24px !important;
  border: 2px solid #9164ff !important; border-radius: 26px !important;
  background: linear-gradient(110deg, #19162f 0%, #13182d 100%) !important;
  color: #f7f6ff !important; font-size: 20px !important; font-weight: 700 !important;
  box-shadow: 0 0 20px #7048ff35, inset 0 0 18px #7348ff12 !important; }
#model-picker .label-wrap.open { margin-bottom: 0 !important; }
#model-picker .label-wrap .icon { color: #d2c8ff !important; font-size: 22px !important; }
#model-picker > [data-testid="accordion-content"] {
  position: absolute !important; top: calc(100% + 8px) !important; left: 0 !important;
  width: 100% !important; max-width: calc(100vw - 24px) !important; z-index: 1006 !important;
  padding: 4px 0 !important; box-sizing: border-box !important; overflow: visible !important;
  border: 1px solid #414067 !important; border-radius: 24px !important;
  background: linear-gradient(145deg, #17172b 0%, #101322 100%) !important;
  box-shadow: 0 24px 60px #000c, 0 0 24px #7250ff25 !important;
}
.model-option-row { min-height: 112px !important; padding: 14px 20px !important; margin: 0 !important;
  align-items: center !important; gap: 14px !important; }
.model-option-row + .model-option-row { border-top: 1px solid #292a43 !important; }
.model-option-main { min-width: 0; flex: 1 1 auto; gap: 3px !important; }
.model-option-name { min-width: 0; width: 100% !important; flex: 1 1 auto; white-space: normal !important;
  justify-content: flex-start !important; text-align: left !important; font-size: 14px !important;
  font-weight: 700 !important; color: #f4f3ff !important; background: transparent !important;
  border: 0 !important; box-shadow: none !important; padding: 0 !important; }
.model-option-name:hover { color: #b7a0ff !important; background: transparent !important; }
.model-option-status { min-width: 200px; max-width: 200px; flex: 0 0 200px; align-items: flex-end !important;
  align-self: center; margin-left: auto !important; text-align: right; }
.model-capabilities { color: #a9a9c2; font-size: 14px; line-height: 1.25; margin: 0 !important; }
.model-option-status .download-state, .model-option-status .downloaded-label,
.model-option-status button, .model-option-status .download-progress-title { font-size: 14px !important; }
.model-option-status button { width: 200px !important; min-width: 200px !important; max-width: 200px !important;
  height: 50px !important; min-height: 50px !important; max-height: 50px !important;
  box-sizing: border-box !important; border: 0 !important; border-radius: 15px !important;
  color: white !important; font-weight: 700 !important;
  background: linear-gradient(110deg, #8952f6 0%, #3975f5 100%) !important;
  box-shadow: 0 7px 22px #604af044 !important; }
.model-option-status .downloaded-label { display: inline-flex; align-items: center; gap: 9px; padding: 0 !important; background: transparent !important;
  border: 0 !important; color: #58e3aa !important; font-size: 16px !important; }
.download-check { display: inline-flex; width: 30px; height: 30px; align-items: center;
  justify-content: center; border-radius: 50%; background: #4ee2a7; color: #10251f;
  font-size: 20px; font-weight: 900; }
.model-option-status .downloaded-label { font-size: 14px !important; }
.model-option-status .download-check { font-size: 20px !important; }
.model-option-status .download-state { color: #aaaac4 !important; font-size: 14px !important; }
.model-option-status .download-progress-wrap { min-width: 190px; padding: 0; border: 0; background: transparent; }
.model-option-status .download-progress-title { color: #f2f1ff !important; font-size: 14px !important; }
.model-option-status .download-progress-title b { color: #fff !important; }
.model-option-status .download-progress-title span { color: #a9a9c2 !important; }
.model-option-status .download-progress { width: 100%; height: 9px; margin-top: 8px; accent-color: #8c55ff; }
.downloaded-label { display: inline-flex; align-items: center; padding: 9px 13px; border-radius: 12px;
  background: #10251f; border: 1px solid #28664f; color: #69e1a8; font-weight: 750; letter-spacing: .02em; }
.download-state { color: #bac7db; font-size: .88rem; }
.download-progress-wrap { min-width: 240px; padding: 8px 10px; border: 1px solid #30405a;
  border-radius: 12px; background: #0c1320; }
.download-progress-title { display: flex; justify-content: space-between; gap: 8px; color: #dbe8fa; font-size: .78rem; }
.download-progress-title b { color: #85e3c1; }
.download-progress-title span { color: #9baac1; white-space: nowrap; }
.download-progress { width: 100%; height: 9px; margin-top: 8px; accent-color: #63d9ad; }
#resource-hud { position: fixed; top: 14px; right: 18px; z-index: 1000; width: auto !important;
  pointer-events: none; }
.resource-chip { display: flex; align-items: center; gap: 13px; padding: 9px 13px;
  border: 1px solid #36445b; border-radius: 14px; background: rgba(10, 15, 25, .92);
  color: #dce8fa; box-shadow: 0 8px 28px #0008; backdrop-filter: blur(12px); font-size: .78rem; }
.resource-chip span { display: flex; align-items: baseline; gap: 5px; white-space: nowrap; }
.resource-chip b { color: #85e3c1; font-size: .9rem; }
.resource-chip small { color: #9baac1; font-size: .68rem; }
.resource-chip i { width: 1px; height: 22px; background: #344056; }
#reference-upload button[aria-label*="remove" i],
#reference-upload button[title*="remove" i],
#reference-upload button[data-testid*="remove" i] {
  width: 40px !important; min-width: 40px !important; height: 40px !important;
}
#reference-upload button[aria-label*="remove" i] svg,
#reference-upload button[title*="remove" i] svg,
#reference-upload button[data-testid*="remove" i] svg {
  width: 32px !important; height: 32px !important;
}
#output video { border-radius: 16px; }
footer { display: none !important; }
@media (max-width: 640px) { #resource-hud { top: auto; right: 10px; bottom: 10px; }
  #model-picker { width: calc(100vw - 24px) !important; min-width: calc(100vw - 24px) !important;
    max-width: calc(100vw - 24px) !important; }
  #model-picker > [data-testid="accordion-content"] { padding: 12px !important; }
  .model-option-row { gap: 10px !important; padding: 12px !important; }
  .model-option-status { min-width: 200px; max-width: 200px; flex: 0 0 200px; }
  .resource-chip { gap: 8px; padding: 7px 9px; font-size: .7rem; } }
"""

with gr.Blocks(title="Wan 2.2 Motion Studio") as demo:
    gr.HTML(
        """
        <div id="hero">
          <h1>Motion Studio</h1>
          <p>Choose a local model, set a 10-30 second duration, and generate from text or image references.</p>
        </div>
        """
    )
    model_selector = gr.Dropdown(choices=list(MODEL_PATHS), value=DEFAULT_MODEL, visible=False)
    with gr.Accordion(DEFAULT_MODEL, open=False, elem_id="model-picker") as model_picker:
        with gr.Row(equal_height=True, elem_classes="model-option-row"):
            with gr.Column(elem_classes="model-option-main"):
                ltx_model_button = gr.Button(
                    "LTX-Video 2B · recommended for 8 GB VRAM",
                    variant="primary",
                    elem_classes="model-option-name",
                )
                gr.HTML('<div class="model-capabilities">Text to video · Image to video</div>')
            with gr.Column(elem_classes="model-option-status"):
                ltx_download_status = gr.HTML(model_download_status_html(DEFAULT_MODEL))
                ltx_download_button = gr.Button("Download LTX-Video 2B", visible=not ltx_model_ready(), variant="secondary")
        with gr.Row(equal_height=True, elem_classes="model-option-row"):
            with gr.Column(elem_classes="model-option-main"):
                wan_model_button = gr.Button("Wan 2.2 TI2V-5B", variant="secondary", elem_classes="model-option-name")
                gr.HTML('<div class="model-capabilities">Text to video · Image to video</div>')
            with gr.Column(elem_classes="model-option-status"):
                wan_download_status = gr.HTML(model_download_status_html(WAN_LABEL))
                wan_download_button = gr.Button("Download Wan 2.2", visible=not wan_model_ready(), variant="secondary")
    ltx_model_button.click(
        lambda: select_model(DEFAULT_MODEL),
        outputs=[model_selector, model_picker, ltx_model_button, wan_model_button],
        show_progress="hidden",
    )
    wan_model_button.click(
        lambda: select_model(WAN_LABEL),
        outputs=[model_selector, model_picker, ltx_model_button, wan_model_button],
        show_progress="hidden",
    )
    wan_model_value = gr.State(WAN_LABEL)
    wan_download_button.click(
        start_model_download,
        inputs=wan_model_value,
        outputs=[wan_download_status, wan_download_button],
        show_progress="hidden",
    )
    model_status_display = gr.Markdown(model_status(DEFAULT_MODEL), elem_id="model-status")
    resource_hud = gr.HTML(resource_usage_html(), elem_id="resource-hud")
    resource_timer = gr.Timer(10.0, active=True)
    resource_timer.tick(
        refresh_status,
        inputs=model_selector,
        outputs=[
            resource_hud, model_status_display,
            ltx_download_status, ltx_download_button,
            wan_download_status, wan_download_button,
        ],
        queue=False,
        show_progress="hidden",
    )
    model_selector.change(
        refresh_status,
        inputs=model_selector,
        outputs=[
            resource_hud, model_status_display,
            ltx_download_status, ltx_download_button,
            wan_download_status, wan_download_button,
        ],
        queue=False,
        show_progress="hidden",
    )
    history_state = gr.State([])

    with gr.Row(equal_height=True):
        with gr.Column(scale=6, elem_classes="panel"):
            prompt = gr.Textbox(
                label="Describe your scene",
                value=DEFAULT_PROMPT,
                placeholder="A bunny like @bunny runs down a forest road as a lion like @lion follows...",
                lines=5,
            )
            images = gr.File(
                label="Visual references · up to 6 images",
                file_count="multiple",
                file_types=["image"],
                type="filepath",
                elem_id="reference-upload",
            )
            reference_help = gr.Markdown("Upload up to 6 named references. Example: `@bunny`. Selected images are combined into one visual reference board.")
            reference_gallery = gr.Gallery(label="Attached references", columns=3, height=180)
            images.change(update_references, inputs=images, outputs=[reference_help, reference_gallery])

            with gr.Row():
                duration = gr.Slider(10, 30, value=10, step=1, label="Duration in seconds")
                resolution = gr.Dropdown(["480p (832x480)", "720p (1280x704)"], value="480p (832x480)", label="Output resolution", info="720p uses substantially more memory; use LTX and 480p for the smoothest runs.")
            with gr.Row():
                steps = gr.Slider(1, 12, value=4, step=1, label="Inference steps - lower is faster")
                seed = gr.Number(value=-1, label="Seed", precision=0)
            with gr.Accordion("Fine-tune generation", open=False):
                guidance = gr.Slider(1.0, 7.0, value=5.0, step=0.1, label="Guidance scale")
                negative_prompt = gr.Textbox(label="Avoid", value=DEFAULT_NEGATIVE, lines=2)

            generate = gr.Button("Generate video", variant="primary", elem_classes="generate-btn")

        with gr.Column(scale=5, elem_classes="panel"):
            output_video = gr.Video(label="Your video", elem_id="output")
            status = gr.Markdown("")
            history = gr.Gallery(label="Recent generations · saves the latest 12", columns=2, height=320)

    generate.click(
        generate_video,
        inputs=[prompt, images, model_selector, duration, resolution, negative_prompt, steps, guidance, seed, history_state],
        outputs=[output_video, history, history_state, status],
    )
    gr.Markdown(
        "Local generation has no API token quota. It is limited by available VRAM, RAM, "
        "generation time, disk space, and model frame/resolution constraints. This app has no NSFW filter; "
        "outputs are not guaranteed, and the model license applies."
    )


if __name__ == "__main__":
    demo.queue(max_size=1).launch(
        server_name=SERVER_NAME,
        server_port=SERVER_PORT,
        share=SHARE_APP,
        inbrowser=True,
        theme=gr.themes.Soft(primary_hue="cyan", neutral_hue="slate"),
        css=CSS,
    )
