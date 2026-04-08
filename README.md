# Stegosaurus

Hide secret messages inside naturally-generated text using a large language model. A message is encoded into cover text by steering next-token selection at each step; a recipient with the same model can recover the original message deterministically. See [docs/whitepaper.md](docs/whitepaper.md) for how it works.

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

Open `http://localhost:7860`. Use the **Encode** tab to produce cover text from a secret message, and the **Decode** tab to recover a message from cover text.

### Command line

```bash
# Encode
python src/stegosaurus.py -e "your secret message"

# Decode (pipe cover text via stdin)
echo "<cover text>" | python src/stegosaurus.py -d
```

## Configuration

Edit `src/stegosaurus.py` to change the active model:

```python
MODEL_NAME = 'Qwen/Qwen2.5-1.5B'  # or any model listed in src/model_config.json
```

Supported models:
- `google/gemma-3-1b-pt`,
- `Qwen/Qwen2.5-1.5B`,
- `Qwen/Qwen2.5-3B`,
- `meta-llama/Llama-3.2-3B`.

Per-model tokenizer settings are in `src/model_config.json`.