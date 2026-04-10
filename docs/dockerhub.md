# Stegosaurus

Hide secret messages inside naturally-generated text using a large language model. A message is encoded into cover text by steering next-token selection at each step; a recipient with the same model can recover the original message deterministically.

Source: [github.com/gperdrizet/stegosaurus](https://github.com/gperdrizet/stegosaurus)

## Usage

The image uses the CUDA 13 PyTorch build, which supports Turing GPUs (sm_75) and newer. Runs on CPU too; torch selects the device automatically at runtime.

```bash
docker run -p 8080:8080 gperdrizet/stegosaurus:latest
```

For GPU inference, pass `--gpus all` (requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)):

```bash
docker run --gpus all -p 8080:8080 gperdrizet/stegosaurus:latest
```
