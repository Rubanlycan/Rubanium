# Motion Studio

Local video generation with a dropdown for **LTX-Video 2B** and **Wan 2.2 TI2V-5B**. LTX-Video 2B is the default recommendation for this RTX 4060 Laptop GPU with 8 GiB VRAM. The Wan 2.2 checkpoint is still downloading into `D:\models\wan2.2-ti2v-5b-diffusers`; the selector will report when it is ready.

## Video controls

- Choose a 10–30 second clip. The app generates two-second sections and carries the last frame between sections to reduce peak memory and keep motion continuity. Frames stream to FFmpeg as each section completes.
- Choose 480p (832×480) or 720p (1280×704 model-native output).
- Start with LTX-Video 2B, 480p, and 4 inference steps. Higher resolution, longer duration, and more steps increase runtime and memory use. Wan's official single-GPU setup recommends 24 GiB VRAM; this PC has 8 GiB, so Wan and especially 720p can still run out of memory.
- Low-memory loading stages the LTX text encoder to disk, streams transformer blocks with CUDA, and tiles the LTX decoder. Wan uses device-map CPU/disk offload. CUDA SDPA and TF32 are enabled where supported.
- Clips are encoded at 24 fps. Long clips can take a long time on this GPU; offload reduces peak weight memory at the cost of transfer speed.

## Install and run

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python download_models.py
python app.py
```

Open <http://127.0.0.1:7860>. LTX files should be in `D:\models\ltx-video-2b\components`. Generated MP4 files are saved under `outputs/`.

If you put Wan or LTX in another model directory, pass its parent directory to the downloader where applicable and set `RUBANIUM_MODELS_DIR` before starting the app.

## Named image references

Upload up to six images and mention their filename tags, such as `@bunny`, in the prompt. LTX uses the selected images as conditioning; Wan combines them into a starting reference board. For Wan, the carried ending frame conditions each following two-second section.

## Practical limits

Generation is local and has no API token quota. Memory, speed, free disk space, model constraints, and the chosen duration/resolution are the practical limits. The app does not include an NSFW filter; follow the model license and applicable law.
