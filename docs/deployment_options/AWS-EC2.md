# Deploying Stegosaurus to AWS ECS on EC2 (GPU)

## Architecture

```
User → Route 53 / DNS → ACM (TLS) → ALB → ECS Task on EC2 g4dn.xlarge (Gradio, port 7860)
                                               ↓
                                         NVIDIA T4 GPU (16 GB VRAM)
```

- **Compute:** ECS EC2 launch type on `g4dn.xlarge` (T4 GPU), always-on
- **Model:** `Qwen/Qwen2.5-1.5B` in bfloat16 — fits in T4's 16 GB VRAM
- **Encode latency:** ~2s (vs ~30s on CPU Fargate)
- **Cost:** ~$0.53/hr on-demand, ~$0.16/hr spot

---

## How it differs from Fargate

The Docker image, task definition structure, ECR push, and ALB setup are identical to the Fargate plan.

---

## Phase 1 - Code changes

Same as Fargate, plus update `model_config.json` to set `bfloat16` for the Qwen entry (the default is already `bfloat16`, so this is only needed if it was changed for a CPU deployment):

```json
"Qwen/Qwen2.5-1.5B": {
    "dtype": "bfloat16"
}
```

And update `demo/app.py`:
```python
demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
```

## Phase 2 - Docker image

Use a **CUDA-enabled image** instead of CPU-only:

```dockerfile
FROM python:3.12-slim
# Install torch with CUDA 12.6 support
RUN pip install torch==2.11.0 --extra-index-url https://download.pytorch.org/whl/cu126
```

Everything else (copy `src/`, `demo/`, set `HF_HOME`, `EXPOSE 7860`) is the same as the Fargate plan.

## Phase 3 - AWS setup

### ECR
Same as Fargate — create repo, authenticate, tag, push.

### IAM
Same as Fargate — ECS task execution role with `AmazonECSTaskExecutionRolePolicy`.

Add an **EC2 instance profile** with `AmazonEC2ContainerServiceforEC2Role` so the instance can register with the ECS cluster.

### ECS cluster
- Create an **EC2 cluster** (not Fargate) in the ECS console
- Add a `g4dn.xlarge` instance using the **Deep Learning Base OSS Nvidia Driver AMI** (search in AMI catalog — has CUDA drivers pre-installed)
- Attach the EC2 instance profile created above

### Task definition
- Launch type: **EC2** (not Fargate)
- GPU: request 1 GPU resource (`"resourceRequirements": [{"type": "GPU", "value": "1"}]`)
- Memory: 8192 MB (leaves headroom on the 16 GB instance RAM)
- Container image: ECR URI, port 7860
- Environment: `HF_HOME=/tmp/huggingface`
- Log driver: `awslogs` → `/ecs/stegosaurus`

### ECS service, ALB, and custom domain
Identical to the Fargate plan — create an ALB, target group on port 7860, attach to the ECS service, add ACM cert and Route 53 alias for a custom domain.

---

## Using Spot instances (cost saving)

Replace the On-Demand instance with a Spot instance to cut costs by ~70%:

- In the ECS cluster's **Capacity Provider**, configure `g4dn.xlarge` Spot
- Set a Spot interruption handler (ECS drains tasks automatically on 2-min interruption notice)
- Accept that the service may briefly go offline during Spot reclamation (~rare for g4dn)

At ~$0.16/hr Spot, the GPU option becomes cheaper than on-demand CPU Fargate.

---

## Cold start & caching

With a running EC2 instance, the model is loaded into GPU memory once and stays there — no per-request cold start. The first load after instance boot takes ~1–2 min.

To persist model weights across instance reboots, mount an EFS volume at `HF_HOME`.

---

## Smoke test

```bash
curl http://<public-ip>:7860       # without ALB
curl https://yourdomain.com        # with ALB + domain
```

Open in browser, encode a short message, decode the cover text, and verify it matches. Encode should complete in ~2s.
