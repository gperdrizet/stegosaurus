# Deploying Stegosaurus to Google Cloud Run

## Architecture

```
User → Cloud Run URL (*.run.app) → Cloud Run Service (Gradio, port 8080)
              ↓ (automatic TLS, no ALB needed)        ↓ (cold start)
         Custom domain (optional, 1 command)    Hugging Face (model download)
```

- **Compute:** Cloud Run (serverless, scales to zero), CPU-only
- **Model:** `Qwen/Qwen2.5-1.5B` - public (no HF token), ~6 GB float32
- **Task size:** 4 vCPU / 16 GB RAM
- **Cost:** ~$0.002/request (30s encode) - $0 when idle; ~$0.17/hr for a warm instance

## Phase 1 - Install and configure gcloud

Install the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) for your OS, then initialize it:

```bash
gcloud init
```

This opens a browser to sign in with your Google account and prompts you to select or create a project. Once complete, set your active project if you haven't already:

```bash
gcloud config set project <project-id>
```

Verify the install:

```bash
gcloud --version
```

## Phase 2 - Docker image

`requirements-deploy.txt`, `Dockerfile`, and `.dockerignore` are present in the repository. Build and smoke-test locally:

```bash
make build
docker run --rm -p 8080:8080 gperdrizet/stegosaurus:latest
```

Open `http://localhost:8080` and verify encode/decode works before pushing to Artifact Registry.

## Phase 3 - GCP setup

Two options: the web console or the `gcloud` CLI. Both produce identical results - use whichever you prefer.

`<region>` appears throughout these instructions. Choose one and use it consistently for both Artifact Registry and Cloud Run - they must match. Common choices:

| Region | Location |
|---|---|
| `us-central1` | Iowa (also the only US region with Cloud Run GPU) |
| `us-east4` | Northern Virginia |
| `us-west1` | Oregon |

For the full list: `gcloud run regions list`

### Option A - Console (web UI)

**Create a project**
1. Go to [console.cloud.google.com](https://console.cloud.google.com)
2. Click the project selector at the top of the page, then **New project**
3. Enter a project name (e.g. `stegosaurus`), note the auto-generated **Project ID** - this is what you'll use as `<project>` in later steps
4. Click **Create** and wait for the project to be provisioned, then select it from the project selector

**Enable APIs**
1. Go to [console.cloud.google.com](https://console.cloud.google.com) and select your project
2. Navigate to **APIs & Services > Enable APIs and services**
3. Search for and enable: **Cloud Run Admin API**, **Artifact Registry API**

**Create an Artifact Registry repository**
1. Navigate to **Artifact Registry > Repositories > Create repository**
2. Name: `stegosaurus`, Format: `Docker`, Location: choose a region, then click **Create**

**Push the image**

The console cannot push images directly - use Docker locally:
```bash
gcloud auth configure-docker <region>-docker.pkg.dev

docker tag gperdrizet/stegosaurus \
  <region>-docker.pkg.dev/<project>/stegosaurus/app:latest

docker push \
  <region>-docker.pkg.dev/<project>/stegosaurus/app:latest
```

**Deploy the service**
1. Navigate to **Cloud Run > Create service**
2. Select **Deploy one revision from an existing container image**, click **Select** and pick the image you just pushed
3. Service name: `stegosaurus`, Region: same as your repository
4. Under **Container, networking, security**:
   - Container port: `8080`
   - Memory: `16 GiB`, CPU: `4`
   - Request timeout: `300`
   - Maximum concurrent requests per instance: `1`
5. Under **Authentication**, select **Allow unauthenticated invocations**
6. Click **Create**

Cloud Run displays the service URL (`https://*.run.app`) once the deployment completes.

**Custom domain (optional)**
1. Navigate to **Cloud Run > your service > Manage custom domains > Add mapping**
2. Enter your domain and follow the DNS record instructions shown
3. Add the CNAME or A records to your DNS provider - GCP provisions and renews the TLS certificate automatically

**Keep one instance warm (optional)**
1. Navigate to **Cloud Run > your service > Edit & deploy new revision**
2. Under **Capacity**, set **Minimum number of instances** to `1`
3. Click **Deploy** - this adds ~$0.17/hr for a continuously running instance

### Option B - CLI

**Create a project**
```bash
gcloud projects create <project-id> --name="Stegosaurus"
gcloud config set project <project-id>
```

You'll also need to link a billing account. Find your billing account ID at [console.cloud.google.com/billing](https://console.cloud.google.com/billing), then:
```bash
gcloud billing projects link <project-id> --billing-account=<billing-account-id>
```

**Enable APIs**
```bash
gcloud services enable run.googleapis.com artifactregistry.googleapis.com
```

**Artifact Registry**
```bash
gcloud artifacts repositories create stegosaurus \
  --repository-format=docker \
  --location=<region>

gcloud auth configure-docker <region>-docker.pkg.dev

docker tag stegosaurus \
  <region>-docker.pkg.dev/<project>/stegosaurus/app:latest

docker push \
  <region>-docker.pkg.dev/<project>/stegosaurus/app:latest
```

**Deploy**
```bash
gcloud run deploy stegosaurus \
  --image <region>-docker.pkg.dev/<project>/stegosaurus/app:latest \
  --region <region> \
  --memory 16Gi \
  --cpu 4 \
  --timeout 300 \
  --concurrency 1 \
  --allow-unauthenticated
```

To override the default model, add `--set-env-vars MODEL=<model-id>`.

Cloud Run immediately provides a `https://*.run.app` URL with managed TLS - no load balancer or certificate setup required.

**Custom domain (optional)**
```bash
gcloud run domain-mappings create \
  --service stegosaurus \
  --domain yourdomain.com \
  --region <region>
```

Add the CNAME or A records printed by the command to your DNS provider. GCP provisions and renews the TLS certificate automatically.

**Keep one instance warm (optional)**
```bash
gcloud run services update stegosaurus \
  --min-instances 1 \
  --region <region>
```


## Cold start & caching

The model (~2 GB) downloads from Hugging Face when a new instance starts (~2-3 min first request). It is cached in `/tmp/huggingface` for the lifetime of the instance.

To keep one instance warm and eliminate cold starts, see the "Keep one instance warm" step in phase 3 above. This adds ~$0.17/hr for a continuously running instance.

**To persist the cache across restarts**, mount a Cloud Storage FUSE volume at `HF_HOME` - avoids the 2-3 min download on every cold start.


## GPU option (Cloud Run with NVIDIA L4)

Cloud Run supports GPUs since 2024. Adding `--gpu 1` gives you an NVIDIA L4 (24 GB VRAM), dropping encode latency from ~30s to ~2s - comparable to the EC2 g4dn.xlarge plan, but fully serverless.

Build and push the image as normal:
```bash
make build
docker tag gperdrizet/stegosaurus:dev <region>-docker.pkg.dev/<project>/stegosaurus/app:latest
docker push <region>-docker.pkg.dev/<project>/stegosaurus/app:latest
```

Then deploy:
```bash
gcloud run deploy stegosaurus \
  --image <region>-docker.pkg.dev/<project>/stegosaurus/app:latest \
  --region <region> \
  --memory 16Gi \
  --cpu 4 \
  --gpu 1 \
  --gpu-type nvidia-l4 \
  --timeout 300 \
  --concurrency 1 \
  --allow-unauthenticated
```

GPU Cloud Run availability is limited to specific regions (currently `us-central1`, `asia-northeast1`, others). Check the [Cloud Run GPU docs](https://cloud.google.com/run/docs/configuring/services/gpu) for the current list.

**Cold start caveat:** GPU instances have a significantly longer cold start than CPU - container boot + GPU allocation + CUDA runtime init + model download adds up to ~5–7 min. This makes scale-to-zero impractical for a demo. Set `--min-instances 1` to keep one instance warm at all times (NVIDIA L4 costs ~$0.70/hr).


## Smoke test

```bash
curl https://<your-service>.run.app   # *.run.app URL
curl https://yourdomain.com           # custom domain
```

Open in browser, encode a short message, decode the cover text, and verify it matches.
