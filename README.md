---
title: Stegosaurus
emoji: 🦕
colorFrom: green
colorTo: blue
sdk: gradio
app_file: demo/app.py
pinned: true
short_description: Hide secret messages inside generated text
---

# Stegosaurus

Hide secret messages inside naturally-generated text using a large language model. A message is encoded into cover text by steering next-token selection at each step; a recipient with the same model can recover the original message deterministically. See [docs/whitepaper.md](docs/whitepaper.md) for how it works.

**Live demo:** [huggingface.co/spaces/gperdrizet/stegosaurus](https://huggingface.co/spaces/gperdrizet/stegosaurus)

## Run locally

```bash
# CPU
docker run -p 8080:8080 gperdrizet/stegosaurus:latest

# GPU (requires NVIDIA Container Toolkit; model uses ~3.5 GB VRAM)
docker run --gpus all -p 8080:8080 gperdrizet/stegosaurus:latest
```

Open `http://localhost:8080` in a browser.


## Development setup

**Dev container (recommended):** open the repo in VS Code and choose *Reopen in Container*. VS Code will prompt you to select a configuration:

- **Python 3.12 (CPU / Codespaces)** - works everywhere, including GitHub Codespaces. Encode/decode runs on CPU (~30s per request).
- **Python 3.12 (GPU)** - passes your local GPU into the container via `--gpus all`. Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on the host. Encode/decode runs on GPU (~2s per request).

**Virtual environment:**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

The model is configured via environment variable:

| Variable | Default | Description |
|---|---|---|
| `MODEL` | `Qwen/Qwen2.5-1.5B` | HuggingFace model ID. Must be a key in `src/model_config.json`. |

Supported models:
- `google/gemma-3-1b-pt`
- `Qwen/Qwen2.5-1.5B`
- `Qwen/Qwen2.5-3B`
- `meta-llama/Llama-3.2-3B`

Per-model tokenizer settings are in `src/model_config.json`.

## Deployment

A single image runs on CPU or GPU (includes the CUDA 13 PyTorch wheel, supports Truing / sm_75 and newer).

**Build from source:**
```bash
make build
```

**Run after building locally:**
```bash
docker run -p 8080:8080 gperdrizet/stegosaurus:dev
```

**Build, tag, and push to Docker Hub** using the Makefile (reads the version from the latest git tag automatically):
```bash
make build    # build the image, tagged with the current git tag and latest
make push     # push to Docker Hub + update Docker Hub description
make release  # build + push in one step
```

Requires a `DOCKERHUB_TOKEN` in `.env`. Each build produces two tags: `gperdrizet/stegosaurus:v1.0.0` and `gperdrizet/stegosaurus:latest`. Builds on untagged commits use `dev` (e.g. `gperdrizet/stegosaurus:dev`).

**Deploy to Hugging Face Spaces:**
```bash
make deploy-hf  # force push main to the HF Space
```

Requires `HF_TOKEN` in `.env`.

See [Deploying Stegosaurus to Hugging Face Spaces](https://github.com/gperdrizet/stegosaurus/blob/main/docs/deployment_options/HF-spaces.md) for detailed set-up instructions.

Requires an `HF_TOKEN` in `.env` (generate at huggingface.co → Settings → Access tokens, with Write scope).

**Push to Google Cloud Artifact Registry:**
```bash
make push-gcp
```

Requires `GCP_PROJECT_ID` in `.env`.

See [Deploying Stegosaurus to Google Cloud Run](github.com/gperdrizet/stegosaurus/blob/main/docs/deployment_options/GCP-cloudrun.md) for detailed set-up instructions.