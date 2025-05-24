# Makefile for Hafez Semantic Search Project

# Variables
docker_image = hafez-search
docker_compose = docker-compose

# Default target
default: help

#################################
.PHONY: help build-image embed compose-up compose-down clean

help:
	@echo "Makefile commands:"
	@echo "  build   Build the Docker image ($(docker_image))"
	@echo "  embed         Regenerate embeddings locally using run_embedding.py"
	@echo "  up    Build and start services with Docker Compose"
	@echo "  down  Stop and remove services with Docker Compose"
	@echo "  clean         Remove local LanceDB directory"

#################################
build:
	@echo "Building Docker image: $(docker_image)"
	docker build -t $(docker_image) .

#################################
embed:
	@echo "Regenerating embeddings in local environment"
	python run_embedding.py

#################################
up:
	@echo "Starting services with Docker Compose"
	$(docker_compose) up --build

#################################
down:
	@echo "Stopping and removing services"
	$(docker_compose) down

#################################
clean:
	@echo "Removing LanceDB directory"
	rm -rf lancedb_dir
