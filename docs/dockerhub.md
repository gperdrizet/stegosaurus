# Stegosaurus

Hide secret messages inside naturally-generated text using a large language model. A message is encoded into cover text by steering next-token selection at each step; a recipient with the same model can recover the original message deterministically.

Source: [github.com/gperdrizet/stegosaurus](https://github.com/gperdrizet/stegosaurus)

## Images

### CPU (`latest-cpu`)

Runs on any machine — no GPU or NVIDIA drivers required.

- **Base:** `python:3.12-slim`
- **PyTorch:** CPU wheel (~250 MB)
- **Default model:** `Qwen/Qwen2.5-1.5B` in `float32`
- **Encode latency:** ~30s

```bash
docker run -p 8080:8080 gperdrizet/stegosaurus:latest-cpu
```

### GPU (`latest-gpu`)

Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on the host.

- **Base:** `python:3.12-slim`
- **PyTorch:** CUDA 12.6 wheel (~2.5 GB)
- **Default model:** `Qwen/Qwen2.5-1.5B` in `bfloat16`
- **Encode latency:** ~2s

```bash
docker run --gpus all -p 8080:8080 gperdrizet/stegosaurus:latest-gpu
```

## Configuration

Override model or dtype at runtime via environment variables:

| Variable | Default (CPU) | Default (GPU) | Description |
|---|---|---|---|
| `MODEL` | `Qwen/Qwen2.5-1.5B` | `Qwen/Qwen2.5-1.5B` | HuggingFace model ID |
| `TORCH_DTYPE` | `float32` | `bfloat16` | PyTorch dtype |

```bash
docker run -p 8080:8080 \
  -e MODEL=Qwen/Qwen2.5-3B \
  -e TORCH_DTYPE=float32 \
  gperdrizet/stegosaurus:latest-cpu
```
