# Motion Studio

Motion Studio is a local Gradio app for text-to-video and image-to-video generation with **LTX-Video 2B** or **Wan 2.2 TI2V-5B**. Models run on the user's own machine; this app does not send prompts to a hosted inference API.

## Requirements

- Windows 10 or newer.
- Python supported by the installed PyTorch release. See [PyTorch's Windows install guide](https://pytorch.org/get-started/locally/).
- An NVIDIA GPU with a CUDA-enabled PyTorch build. The app requires CUDA for generation; CPU and disk offloading reduce peak VRAM use but do not provide CPU-only generation.
- Internet access for the first model download.
- Free disk space for model files and temporary offload data. LTX-Video is approximately 25.6 GB; Wan 2.2 is approximately 34.2 GB. The downloader checks for 35 GiB free for LTX and 45 GiB for Wan.

LTX-Video 2B is the recommended starting point for lower-VRAM systems. Results and performance depend on GPU memory, system RAM, disk speed, clip length, resolution, and inference steps.

## Install

Clone or download this repository, then open PowerShell in the project directory:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install a CUDA-enabled PyTorch and TorchVision build using the command for your GPU driver from the [official PyTorch selector](https://pytorch.org/get-started/locally/). Then install the remaining project requirements:

```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` if you want to change model, offload, output, or server paths. Keep `.env` private; it is ignored by Git.

## Download models and run

Start the app:

```powershell
python app.py
```

Open <http://127.0.0.1:7860>, open the model picker, and use the download button for the model you want. Downloads can also be started from PowerShell:

```powershell
python download_models.py --model ltx
python download_models.py --model wan
```

The default paths are `models/`, `offload/`, and `outputs/` under the project directory. Model weights, generated videos, logs, offload data, virtual environments, and `.env` files are excluded from Git. Set `RUBANIUM_MODELS_DIR`, `RUBANIUM_OFFLOAD_DIR`, and `RUBANIUM_OUTPUT_DIR` in `.env` to store them elsewhere.

The app binds to `127.0.0.1` and does not create a public share link by default. Do not expose it to a network without adding suitable authentication and access controls.

## Features and limits

- 10–30 second clips at 480p or 720p.
- Up to six image references, selected in prompts using filename tags such as `@bunny`.
- Local MP4 output saved in the configured output directory.
- No API token quota. Generation is limited by hardware, disk space, model constraints, and time.

See the model pages for their usage terms and limitations: [LTX-Video](https://huggingface.co/Lightricks/LTX-Video-0.9.5) and [Wan 2.2 TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers). This repository does not currently include a separate license for the application source code.
