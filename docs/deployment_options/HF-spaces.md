# Deploying Stegosaurus to Hugging Face Spaces

## Architecture

```
User → HF Spaces URL (https://<username>-<space>.hf.space) → Gradio app + model (CPU)
```

- **Compute:** HF-managed CPU (free tier: 2 vCPU / 16 GB RAM)
- **Model:** `Qwen/Qwen2.5-1.5B` or any public model - no HF token required for public models
- **Cost:** Free (CPU); upgraded hardware available on paid tiers
- **Encode latency:** ~30s (CPU)

## Phase 1 - README header

HF Spaces requires a YAML front-matter block at the top of `README.md` to configure the Space.

```yaml
---
title: Stegosaurus
emoji: 🦕
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "6.11.0"
app_file: demo/app.py
python_version: "3.12"
pinned: false
---
```

Key fields:

| Field | Value | Notes |
|---|---|---|
| `sdk` | `gradio` | Tells Spaces which runtime to use |
| `sdk_version` | `6.11.0` | Must match the version in `requirements.txt` |
| `app_file` | `demo/app.py` | Path to the Gradio app, relative to repo root |
| `python_version` | `3.12` | Must match the Python version used in dev |
| `pinned` | `false` | Whether to pin the Space to your profile |

## Phase 2 - Create the Space

1. Go to [huggingface.co](https://huggingface.co) and sign in
2. Click your profile picture, then **New Space**
3. Fill in the form:
   - **Space name:** `stegosaurus` (or any name - this becomes part of the URL)
   - **License:** choose one (MIT matches the project LICENSE file)
   - **SDK:** `Gradio`
   - **Hardware:** `CPU Basic` (free)
   - **Visibility:** Public or Private
4. Click **Create Space**

HF creates an empty git repository for the Space at `https://huggingface.co/spaces/<username>/stegosaurus`.

## Phase 3 - Add HF token to Codespace secrets

Pushing to a HF Space requires authentication. Store your token as a GitHub Codespace secret so it is available in the terminal without being hardcoded anywhere.

1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click **New token**, name it (e.g. `CODESPACE_PUSH`), set role to **Write**, click **Generate a token**, and copy it
3. Go to your GitHub repository **Settings > Secrets and variables > Codespaces > New repository secret**
4. Name: `HF_TOKEN`, value: paste the token, click **Add secret**

The secret is now automatically injected as `$HF_TOKEN` in any Codespace opened from this repository.

## Phase 4 - Add the remote and push

From a Codespace terminal:

```bash
# Add the HF Space as a remote (token embedded in URL for auth)
git remote add space https://oauth2:${HF_TOKEN}@huggingface.co/spaces/<username>/stegosaurus

# Force-push - required because the Space repo has an initial commit not in our history
git push --force space main
```

HF Spaces begins building the container immediately after push. Watch the build log at `https://huggingface.co/spaces/<username>/stegosaurus` - the **Building** badge turns **Running** once the app is live.

To redeploy after changes, just push again (force is only needed the first time):
```bash
git push space main
```

## Cold start & model caching

The model downloads from HF Hub on the first request after the Space starts (~2-3 min). HF Spaces caches data in `/data` (persistent across restarts on upgraded hardware) or `/tmp` (ephemeral on free CPU). On the free tier, the container is paused after ~15 min of inactivity and the model cache is lost - the next visitor triggers a cold download.

## Smoke test

Open `https://huggingface.co/spaces/<username>/stegosaurus` in a browser (or use the **App** tab on the Space page), encode a short message, decode the cover text, and verify it matches.
