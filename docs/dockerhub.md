# Stegosaurus

Hide secret messages inside naturally-generated text using a large language model. A message is encoded into cover text by steering next-token selection at each step; a recipient with the same model can recover the original message deterministically.

Source: [github.com/gperdrizet/stegosaurus](https://github.com/gperdrizet/stegosaurus)

## Usage

The image includes the CUDA 12.6 PyTorch build and runs on CPU or GPU — no NVIDIA base image required, the CUDA runtime is bundled in the wheel.

```bash
docker run -p 8080:8080 gperdrizet/stegosaurus:latest
```

For GPU inference, pass `--gpus all` (requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)):

```bash
docker run --gpus all -p 8080:8080 gperdrizet/stegosaurus:latest
```

## Configuration

Override the model at runtime via environment variable:

| Variable | Default | Description |
|---|---|---|
| `MODEL` | `Qwen/Qwen2.5-1.5B` | HuggingFace model ID |

```bash
docker run -p 8080:8080 \
  -e MODEL=Qwen/Qwen2.5-3B \
  gperdrizet/stegosaurus:latest
```
