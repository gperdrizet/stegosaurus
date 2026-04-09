VERSION := $(shell git describe --tags --abbrev=0 2>/dev/null || echo dev)
REPO    := gperdrizet/stegosaurus

-include .env
export

.PHONY: build push release deploy-hf

build:
	VERSION=$(VERSION) docker compose build

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

release: build push

deploy-hf:
	git remote get-url space 2>/dev/null || \
	  git remote add space https://oauth2:$(HF_TOKEN)@huggingface.co/spaces/gperdrizet/stegosaurus
	git push --force space main
