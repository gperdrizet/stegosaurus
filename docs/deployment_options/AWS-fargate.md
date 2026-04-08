# Deploying Stegosaurus to AWS ECS Fargate

## Architecture

```
User → Route 53 / DNS → ACM (TLS) → ALB → ECS Fargate Task (Gradio, port 7860)
                                               ↓ (cold start)
                                         Hugging Face (model download)
```

- **Compute:** Fargate (serverless, no EC2 to manage), CPU-only
- **Model:** `Qwen/Qwen2.5-1.5B` - public (no HF token), fast on CPU, ~6 GB float32
- **Task size:** 4 vCPU / 16 GB RAM
- **Cost:** ~$0.29/hr (task) + ~$16/mo (ALB, if using a custom domain)

## Phase 1 - Code changes

| File | Change |
|---|---|
| `src/stegosaurus.py` | Set `MODEL_NAME = 'Qwen/Qwen2.5-1.5B'` |
| `src/model_config.json` | Qwen entry: `"dtype": "float32"` (bfloat16 is unreliable on Fargate's older CPUs) |
| `demo/app.py` | `demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))` |

## Phase 2 - Docker image

1. Create `deploy/requirements.txt` — CPU-only deps (no CUDA index URL, no Jupyter/matplotlib)
2. Create `Dockerfile` — `python:3.12-slim`, install CPU torch from `https://download.pytorch.org/whl/cpu`, copy `src/` and `demo/`, set `HF_HOME=/tmp/huggingface`, `EXPOSE 7860`
3. Create `.dockerignore` — exclude `models/`, `notebooks/`, `.git/`, `.vscode/`
4. Build and smoke-test locally: `docker build -t stegosaurus . && docker run -p 7860:7860 stegosaurus`

## Phase 3 - AWS setup (console / CLI)

### ECR
```bash
aws ecr create-repository --repository-name stegosaurus --region <region>
aws ecr get-login-password | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
docker tag stegosaurus <account>.dkr.ecr.<region>.amazonaws.com/stegosaurus:latest
docker push <account>.dkr.ecr.<region>.amazonaws.com/stegosaurus:latest
```

### IAM
- Create ECS task execution role with managed policy `AmazonECSTaskExecutionRolePolicy`

### CloudWatch
```bash
aws logs create-log-group --log-group-name /ecs/stegosaurus
```

### ECS cluster
- Create a new Fargate cluster in the ECS console (one click)

### Task definition
- Launch type: Fargate
- CPU: 4 vCPU, Memory: 16 GB
- Container image: ECR URI above, port 7860
- Environment: `HF_HOME=/tmp/huggingface`
- Log driver: `awslogs` → `/ecs/stegosaurus`

### Networking
- VPC: use default VPC with public subnets
- Container security group: allow TCP 7860 inbound from the ALB security group only

### ECS service
- 1 desired task, Fargate launch type
- Without domain: `assignPublicIp=ENABLED`, access via task public IP
- With domain: attach ALB target group (see below)

## Adding a custom domain (optional)

Additional resources required:

| Resource | Purpose |
|---|---|
| ACM Certificate | TLS for your domain (free) - validate via DNS |
| Application Load Balancer | Stable DNS target, TLS termination (HTTPS 443 → HTTP 7860) |
| Target Group | Routes ALB traffic to the ECS service |
| Route 53 A alias (or CNAME) | `yourdomain.com` → ALB DNS name |

ALB security group: allow TCP 443 inbound from `0.0.0.0/0`.
Container security group: allow TCP 7860 inbound from the ALB security group only (not the public internet).

## Cold start & caching

The model (~2 GB) downloads from Hugging Face on the first request after a new task starts (~2–3 min). It is cached in `/tmp/huggingface` for the lifetime of the task. To persist the cache across restarts, mount an EFS volume at `HF_HOME`.

## Smoke test

```bash
curl http://<public-ip>:7860       # without ALB
curl https://yourdomain.com        # with ALB + domain
```

Open in browser, encode a short message, decode the cover text, and verify it matches.
