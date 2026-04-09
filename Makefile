VERSION := $(shell git describe --tags --abbrev=0 2>/dev/null || echo dev)
REPO    := gperdrizet/stegosaurus

.PHONY: build push release

build:
	VERSION=$(VERSION) docker compose build

push:
	docker push $(REPO):$(VERSION)-cpu
	docker push $(REPO):latest-cpu
	docker push $(REPO):$(VERSION)-gpu
	docker push $(REPO):latest-gpu

release: build push
