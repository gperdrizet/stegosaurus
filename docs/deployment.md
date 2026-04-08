# Deployment Strategy

This document outlines the three phases of Stegosaurus deployment, from a minimal demo to a production-grade microservices architecture. Platform-specific details are in the individual docs; this covers what we gain at each step.


## Act 1 - Monolith on HF Spaces

**What we deploy:** The Gradio app and model in a single process, pushed directly to a Hugging Face Space.

**Architecture:**
```
User → HF Spaces (Gradio + model, single process)
```

**What we gain:**
- Zero infrastructure: no servers, no containers, no config
- Public URL and TLS provided automatically
- Enough to demonstrate the algorithm end-to-end

**Constraints:**
- CPU-only on the free tier (~30s encode)
- No custom domain
- No auth, no user accounts
- Everything coupled in one process - model, UI, business logic

This is the right starting point for a demo. Ship fast, show the idea working.

## Act 2 - Monolith in a container

**What we deploy:** The same Gradio app containerized with Docker, deployed to a managed platform (GCP Cloud Run, ECS, or a VPS).

**Architecture:**
```
User → Managed platform / VPS (nginx + TLS) → Container (Gradio + model)
```

**What we gain:**
- Custom domain with HTTPS
- Reproducible, portable deployments (Docker image = exact environment)
- Choice of compute: CPU for low cost, GPU for ~2s latency
- On managed platforms: automatic scaling, health checks, rolling deploys
- On VPS: full control, lowest cost (~$5/mo + local GPU)

**Constraints:**
- Model and UI still tightly coupled in one container
- Scaling the GPU backend means scaling the whole app
- No auth, no persistent user state

The code doesn't change - just the packaging and where it runs. This is the natural next step after HF Spaces when you want a real URL and better control.

## Act 3 - Microservices

**What we deploy:** The encode/decode logic extracted into a dedicated GPU worker service, fronted by a Django web app with user authentication.

**Architecture:**
```
User → Django (auth, UI, REST API)
           ↓ submits job
       Celery task queue (Redis broker)
           ↓ picks up task
       GPU Worker (stegosaurus module)
           ↓ stores result
       PostgreSQL / Redis result backend
           ↑ polls / WebSocket
       Django → returns result to user
```

**What we gain:**
- User accounts, login, per-user history
- GPU workers scale independently of the web frontend
- Long-running encode/decode doesn't block web server threads (async via task queue)
- Frontend (Django) can run on cheap CPU instances; GPU cost is isolated to workers
- Path to Kubernetes: add GPU nodes, add worker replicas, frontend stays light
- Clean separation of concerns - UI, business logic, and inference are distinct services

**Constraints:**
- Significantly more infrastructure to manage
- Gradio replaced by a custom frontend


## Platform comparison

For Act 1 and Act 2 (monolith), the platform options are:

| | HF Spaces | GCP Cloud Run | ECS Fargate (CPU) | ECS on EC2 (GPU) | DIY VPS + WireGuard |
|---|---|---|---|---|---|
| Setup complexity | Minimal | Low | Medium | Medium | Medium |
| Custom domain | No | Yes (1 command) | Yes (ALB + ACM) | Yes (ALB + ACM) | Yes (nginx + Certbot) |
| HTTPS | Automatic | Automatic | Via ALB | Via ALB | Via Certbot |
| GPU | Paid tier only | Yes (L4, serverless) | No | Yes (T4) | Yes (local) |
| Encode latency | ~30s (CPU) | ~2s (GPU) / ~30s (CPU) | ~30s | ~2s | ~2s |
| Scales to zero | Yes | Yes (default) | Requires extra config | No | No |
| Monthly cost (idle) | Free | Free (CPU) / ~$0.70/hr (GPU, warm) | ~$0.29/hr | ~$0.53/hr | ~$5/mo |
| Infra ownership | None | None | None | EC2 instance | VPS + local machine |
