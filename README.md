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

Hide secret messages inside naturally-generated text using a large language model. A message is encoded into cover text by steering next-token selection at each step; a recipient with the same model can recover the original message deterministically. See the [technical whitepaper](https://github.com/gperdrizet/stegosaurus/blob/main/docs/whitepaper.md) for how it works and Ziegler et al., (2019) "[*Neural Linguistic Steganography*](https://aclanthology.org/D19-1115/)" for the inspiration.

**Live demo:** [huggingface.co/spaces/gperdrizet/stegosaurus](https://huggingface.co/spaces/gperdrizet/stegosaurus)


## 1. Run the app locally

### 1.1. Using docker

```bash
# CPU
docker run -p 8080:8080 gperdrizet/stegosaurus:latest

# GPU (requires NVIDIA Container Toolkit; model uses ~1.5 GB VRAM)
docker run --gpus all -p 8080:8080 gperdrizet/stegosaurus:latest
```

Open `http://localhost:8080` in a browser.


### 1.2. Using a clone

```bash
git clone https://github.com/gperdrizet/stegosaurus.git
cd stegosaurus
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python demo/app.py
```


## 2. Development setup


### 2.1. Using a dev container (recommended)
Open the repo in VS Code and choose *Reopen in Container*. VS Code will prompt you to select a configuration:

- **Python 3.12 (CPU / Codespaces)** - works everywhere, including GitHub Codespaces. Encode/decode runs on CPU (~30s per request).
- **Python 3.12 (GPU)** - passes your local GPU into the container via `--gpus all`. Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on the host. Encode/decode runs on GPU (~2s per request).


### 2.2. Using a virtual environment

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
```


### 2.3. CUDA version

By default, `requirements-dev.txt` installs a CUDA 12.6 wheel by for wide GPU compatibility (Pascal and later). If you have a newer GPU, you can update it to CUDA 12.8 or 13x for a slight improvement in performance.


### 2.4. Hugging Face authentication

If a `.env` file exists in the repo root, the dev container will automatically load it into the container environment. This is the recommended way to supply `HF_TOKEN` to avoid unauthenticated HuggingFace Hub requests:

```
HF_TOKEN=hf_your_token_here
```

Generate a token at huggingface.co → Settings → Access Tokens (Read scope is sufficient). The `.env` file is gitignored and never committed. On GitHub Codespaces, set `HF_TOKEN` as a Codespaces secret instead.


## 3. Configuration

Model and Gradio port are set via environment variables.

| Variable | Default | Description |
|---|---|---|
| `MODEL` | `Qwen/Qwen3-0.6B` | HuggingFace model ID. Must be a key in `src/model_config.json`. |
| `PORT` | `8080` | Listen port for Gradio app |
| `TOP_K` | `20` | Number of top tokens to consider at each generation step |
| `N_PARTITIONS` | `2` | Partitions per token; must be a power of 2 (bits per token = log₂) |
| `ROOT_PATH` | _(empty)_ | Set to the public service URL when running behind a reverse proxy (e.g. Cloud Run) |
| `LOG_LEVEL` | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

Supported models:
- `google/gemma-3-1b-pt`
- `Qwen/Qwen3-0.6B`
- `Qwen/Qwen2.5-1.5B`
- `Qwen/Qwen2.5-3B`
- `meta-llama/Llama-3.2-3B`

Per-model tokenizer settings are in `src/model_config.json`.

## 4. Deployment

### 4.1. Docker

The app is published as a single image which runs on CPU or GPU. The image includes the CUDA 12.6 PyTorch wheel for wide GPU support (Pascal sm_60 and newer). If a GPU is not avalible, the model falls back to CPU inference automatically at runtime. The `qwen3-0.6B` (default) model is included in the image for running in offline environments. Hugging Face telemetry & model update checks are disabled via environment variables.

**Build from source:**
```bash
make build
```

**Run after building locally:**
```bash
docker run -p 8080:8080 gperdrizet/stegosaurus:latest
```

**Build, tag, and push to Docker Hub** using the Makefile (reads the version from the latest git tag automatically):

```bash
make build    # build the image, tagged with the current git tag and latest
make push     # push to Docker Hub + update Docker Hub description
make release  # build + push in one step
```

Requires a `DOCKERHUB_TOKEN` in `.env`. Each build produces two tags: `gperdrizet/stegosaurus:v1.0.0` and `gperdrizet/stegosaurus:latest`. Builds on untagged commits use `dev` (e.g. `gperdrizet/stegosaurus:dev`).

### 4.2. Hugging Face Spaces

Hugging Face Space deployment uses a force push to the repo specified by `HF_SPACE_REPO` in `.env`. Requires `HF_TOKEN` with write access in `.env`.

```bash
make deploy-hf  # force push main to the HF Space
```

See [Deploying Stegosaurus to Hugging Face Spaces](https://github.com/gperdrizet/stegosaurus/blob/main/docs/deployment_options/HF-spaces.md) for detailed set-up instructions.

Requires an `HF_TOKEN` in `.env` (generate at huggingface.co → Settings → Access tokens, with Write scope).

### 4.3. Google Cloud Run

The Docker image build is configured for deployment to Google Cloud Run via and an artifact registry. To push a new image build:

```bash
make push-gcp
```

Requires `GCP_PROJECT_ID` in `.env`.

See [Deploying Stegosaurus to Google Cloud Run](github.com/gperdrizet/stegosaurus/blob/main/docs/deployment_options/GCP-cloudrun.md) for detailed set-up instructions.