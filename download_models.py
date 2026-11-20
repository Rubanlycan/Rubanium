"""Download a supported local video model into D:\\models."""

from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path

# The Xet transfer helper can stall on some Windows networks. Keep the normal
# Hugging Face HTTP resume path, which is easier to retry across restarts.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")

from huggingface_hub import snapshot_download


MODEL_CONFIGS = {
    "wan": {
        "repo": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "path": Path("wan2.2-ti2v-5b-diffusers"),
        "size_gb": 34.2,
        "reserve_gib": 45,
        "required": [
            "model_index.json", "transformer/config.json",
            "transformer/diffusion_pytorch_model.safetensors.index.json",
            "vae/config.json", "text_encoder/config.json",
        ],
    },
    "ltx": {
        "repo": "Lightricks/LTX-Video-0.9.5",
        "path": Path("ltx-video-2b") / "components",
        "size_gb": 25.6,
        "reserve_gib": 35,
        "required": [
            "model_index.json", "transformer/config.json",
            "scheduler/scheduler_config.json", "tokenizer/tokenizer_config.json",
            "vae/config.json", "text_encoder/config.json",
        ],
    },
}
MODEL_PATTERNS = [
    "model_index.json",
    "scheduler/**",
    "text_encoder/**",
    "tokenizer/**",
    "transformer/**",
    "vae/**",
]
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-dir",
        default=r"D:\models",
        help="Directory for local model files (default: D:\\models)",
    )
    parser.add_argument("--model", choices=MODEL_CONFIGS, default="wan")
    args = parser.parse_args()

    config = MODEL_CONFIGS[args.model]
    root = Path(args.models_dir).expanduser().resolve() / config["path"]
    root.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(root).free / (1024**3)
    if free_gib < config["reserve_gib"]:
        raise SystemExit(
            f"This model needs about {config['size_gb']:.1f} GB; keep at least {config['reserve_gib']} GiB free. "
            f"Only {free_gib:.1f} GiB is available on {root.drive}."
        )

    print(f"Downloading {config['repo']} to {root}", flush=True)
    print(f"The download is about {config['size_gb']:.1f} GB; interrupted files can be resumed.", flush=True)
    for attempt in range(1, 101):
        try:
            snapshot_download(
                repo_id=config["repo"],
                repo_type="model",
                local_dir=str(root),
                allow_patterns=MODEL_PATTERNS,
                max_workers=4,
            )
            break
        except Exception as exc:
            if attempt == 100:
                raise SystemExit(
                    "Hugging Face repeatedly interrupted the download. Partial files are "
                    "preserved; rerun this command to resume. "
                    f"Last error: {type(exc).__name__}: {exc}"
                ) from exc
            delay = min(60, attempt * 5)
            print(
                f"Download interrupted ({type(exc).__name__}); resuming in {delay}s "
                f"(retry {attempt}/100).",
                flush=True,
            )
            time.sleep(delay)

    required = [root / item for item in config["required"]]
    missing = [str(path) for path in required if not path.is_file()]
    transformer_weights = list((root / "transformer").glob("*.safetensors"))
    if missing or not transformer_weights:
        details = "\n".join(missing) if missing else "transformer safetensors shards"
        raise SystemExit("Model download is incomplete; missing:\n" + details)

    print(f"{args.model.upper()} Diffusers model is ready.", flush=True)
    print(f"Model folder: {root}", flush=True)
    print("The app loads these local files without using a remote inference API.", flush=True)


if __name__ == "__main__":
    main()
