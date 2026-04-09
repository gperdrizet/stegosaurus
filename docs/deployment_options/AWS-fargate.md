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

## Phase 1 - Configuration

No code changes are needed. Model and dtype are configured via environment variables in the task definition (see Phase 3). Defaults in the repository are already set for CPU Fargate:

| Variable | Value | Notes |
|---|---|
---|
| `MODEL` | `Qwen/Qwen2.5-1.5B` | Public model, no HF token required |
| `TORCH_DTYPE` | `float32` | bfloat16 is unreliable on Fargate's older CPUs |

## Phase 2 - Docker image

`requirements-deploy.txt`, `Dockerfile`, `.dockerignore`, and `docker-compose.yml` are present in the repository. Build and smoke-test locally:

```bash
docker compose build cpu
docker run --rm -p 8080:8080 stegosaurus:dev-cpu
```

Open `http://localhost:8080` and verify encode/decode works before pushing to ECR.

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
- Container image: ECR URI above, port 8080
- Environment: `HF_HOME=/tmp/huggingface`, `MODEL=Qwen/Qwen2.5-1.5B`, `TORCH_DTYPE=float32`
- Log driver: `awslogs` → `/ecs/stegosaurus`

### Networking
- VPC: use default VPC with public subnets
- Container security group: allow TCP 8080 inbound from the ALB security group only

### ECS service
- 1 desired task, Fargate launch type
- Without domain: `assignPublicIp=ENABLED`, access via task public IP on port 8080
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
Container security group: allow TCP 8080 inbound from the ALB security group only (not the public internet).

## Cold start & caching

The model (~2 GB) downloads from Hugging Face on the first request after a new task starts (~2–3 min). It is cached in `/tmp/huggingface` for the lifetime of the task. To persist the cache across restarts, mount an EFS volume at `HF_HOME`.

## Smoke test

```bash
curl http://<public-ip>:8080       # without ALB
curl https://yourdomain.com        # with ALB + domain
```

Open in browser, encode a short message, decode the cover text, and verify it matches.
