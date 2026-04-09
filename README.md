---
title: Stegosaurus
emoji: 🦕
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "6.11.0"
app_file: demo/app.py
python_version: "3.12"
pinned: true
short_description: Hide secret messages inside generated text
---

# Stegosaurus

Hide secret messages inside naturally-generated text using a large language model. A message is encoded into cover text by steering next-token selection at each step; a recipient with the same model can recover the original message deterministically. See [docs/whitepaper.md](docs/whitepaper.md) for how it works.

**Live demo:** [huggingface.co/spaces/gperdrizet/stegosaurus](https://huggingface.co/spaces/gperdrizet/stegosaurus)

## Setup

**Dev container (recommended):** open the repo in VS Code and choose *Reopen in Container*. VS Code will prompt you to select a configuration:

- **Python 3.12 (CPU / Codespaces)** - works everywhere, including GitHub Codespaces. Encode/decode runs on CPU (~30s per request).
- **Python 3.12 (GPU)** - passes your local GPU into the container via `--gpus all`. Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on the host. Encode/decode runs on GPU (~2s per request).

**Virtual environment:**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Web interface

```bash
python demo/app.py
```

Open `http://localhost:8080`. Use the **Encode** tab to produce cover text from a secret message, and the **Decode** tab to recover a message from cover text.

### Command line

```bash
# Encode
python src/stegosaurus.py -e "your secret message"

# Decode (pipe cover text via stdin)
echo "<cover text>" | python src/stegosaurus.py -d
```

## Configuration

Model and dtype are controlled via environment variables:

| Variable | Default | Description |
|---|---|---|
| `MODEL` | `Qwen/Qwen2.5-1.5B` | HuggingFace model ID. Must be a key in `src/model_config.json`. |
| `TORCH_DTYPE` | `float32` | PyTorch dtype. Use `float32` for CPU, `bfloat16` for GPU. |

Supported models:
- `google/gemma-3-1b-pt`
- `Qwen/Qwen2.5-1.5B`
- `Qwen/Qwen2.5-3B`
- `meta-llama/Llama-3.2-3B`

Per-model tokenizer settings are in `src/model_config.json`.

## Docker

Two images are provided via `docker-compose.yml`: a CPU build and a GPU build.

**Build:**
```bash
docker compose build cpu   # CPU-only torch (~250 MB)
docker compose build gpu   # CUDA 12.6 torch (~2.5 GB)
docker compose build       # both
```

**Run:**
```bash
docker compose up cpu
docker compose up gpu      # requires NVIDIA Container Toolkit on the host
```

**Build, tag, and push to Docker Hub** using the Makefile (reads the version from the latest git tag automatically):
```bash
make build    # build both images, tagged with the current git tag and latest
make push     # push all four tags to Docker Hub + update Docker Hub description
make release  # build + push in one step
```

Requires a `DOCKERHUB_TOKEN` in `.env`. Each build produces two tags per service: `gperdrizet/stegosaurus:v1.0.0-cpu` and `gperdrizet/stegosaurus:latest-cpu`. Builds on untagged commits use `dev` (e.g. `gperdrizet/stegosaurus:dev-cpu`).

**Deploy to Hugging Face Spaces:**
```bash
make deploy-hf  # force push main to the HF Space
```

Requires an `HF_TOKEN` in `.env` (generate at huggingface.co → Settings → Access tokens, with Write scope).