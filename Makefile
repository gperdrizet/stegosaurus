# =============================================================================
# Stegosaurus Makefile
# =============================================================================
# Targets:
#   build      - Build CPU and GPU Docker images
#   push       - Push images to Docker Hub + update repo description
#   release    - build + push (full release workflow)
#   deploy-hf  - Deploy to Hugging Face Spaces
#
# Required in .env:
#   DOCKERHUB_TOKEN  - Docker Hub PAT with Read/Write/Delete scope
#   HF_TOKEN         - Hugging Face token with Write scope
# =============================================================================

# Read version from the most recent git tag; fall back to 'dev' if untagged
VERSION := $(shell git describe --tags --abbrev=0 2>/dev/null || echo dev)

# Docker Hub repository name
REPO    := gperdrizet/stegosaurus

# Load secrets from .env (silently ignored if file doesn't exist)
-include .env
export

.PHONY: build push release deploy-hf

# Build both CPU and GPU images using docker compose.
# Tags each image with the current VERSION and 'latest' (e.g. v0.1.0-cpu, latest-cpu).
build:
	VERSION=$(VERSION) docker compose build

# Push all four image tags to Docker Hub, then update the repo description
# from docs/dockerhub.md via the Docker Hub API.
push:
	docker push $(REPO):$(VERSION)-cpu
	docker push $(REPO):latest-cpu
	docker push $(REPO):$(VERSION)-gpu
	docker push $(REPO):latest-gpu
	@JWT=$$(curl -s -X POST https://hub.docker.com/v2/users/login \
	  -H "Content-Type: application/json" \
	  -d '{"username":"gperdrizet","password":"$(DOCKERHUB_TOKEN)"}' \
	  | jq -r .token) && \
	curl -s -X PATCH https://hub.docker.com/v2/repositories/$(REPO)/ \
	  -H "Authorization: JWT $$JWT" \
	  -H "Content-Type: application/json" \
	  --data-binary "{\"full_description\": $$(jq -Rs . < docs/dockerhub.md)}"

# Full release: build images then push to Docker Hub
release: build push

# Deploy to Hugging Face Spaces by force-pushing main to the 'space' remote.
# Adds the remote automatically on first run.
deploy-hf:
	git remote get-url space 2>/dev/null || \
	  git remote add space https://oauth2:$(HF_TOKEN)@huggingface.co/spaces/gperdrizet/stegosaurus
	git push --force space main
