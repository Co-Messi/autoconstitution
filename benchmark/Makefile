# Makefile - Convenient commands for SwarmResearch Benchmark reproduction
# Version: 1.0.0

.PHONY: help setup verify run clean docker-build docker-run report reproducibility

# Default target
.DEFAULT_GOAL := help

# Variables
SEED ?= 42
OUTPUT_DIR ?= ./results
CONFIG ?= ./configs/m4_baseline.yaml
IMAGE_NAME = swarmresearch/benchmark
VERSION = 1.0.0

help: ## Show this help message
	@echo "SwarmResearch Benchmark - Available Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Set up the environment
	@echo "Setting up environment..."
	python3 -m venv venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements.txt

download: ## Download models and datasets
	@echo "Downloading assets..."
	./venv/bin/python -m swarmresearch.download --dataset --model --verify

verify: ## Verify environment and dependencies
	@echo "Verifying environment..."
	./venv/bin/python -m reproducibility.verify_deps
	./venv/bin/python -m swarmresearch.verify --full

run: ## Run the benchmark (default seed: 42)
	@echo "Running benchmark with seed $(SEED)..."
	mkdir -p $(OUTPUT_DIR)
	./venv/bin/python -m swarmresearch.benchmark \
		--config $(CONFIG) \
		--output $(OUTPUT_DIR)/run_$$(date +%Y%m%d_%H%M%S)_seed$(SEED).json \
		--seed $(SEED)

reproduce: ## Full reproduction pipeline
	@echo "Starting full reproduction..."
	bash reproduce.sh $(SEED) $(OUTPUT_DIR)

report: ## Generate HTML report from results
	@echo "Generating report..."
	./venv/bin/python -m swarmresearch.report \
		--results $(OUTPUT_DIR)/run_*.json \
		--output $(OUTPUT_DIR)/report.html

clean: ## Clean generated files
	@echo "Cleaning..."
	rm -rf venv/
	rm -rf $(OUTPUT_DIR)/
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

docker-build: ## Build Docker container
	@echo "Building Docker container..."
	docker build -t $(IMAGE_NAME):$(VERSION) -t $(IMAGE_NAME):latest .

docker-run: ## Run benchmark in Docker container
	@echo "Running benchmark in Docker..."
	docker run --rm \
		-v $(PWD)/results:/benchmark/results \
		-v $(PWD)/configs:/benchmark/configs:ro \
		-e SWARMRESEARCH_SEED=$(SEED) \
		$(IMAGE_NAME):$(VERSION) \
		--config /benchmark/configs/m4_baseline.yaml \
		--output /benchmark/results/ \
		--seed $(SEED)

docker-verify: ## Verify Docker environment
	@echo "Verifying Docker environment..."
	docker run --rm $(IMAGE_NAME):$(VERSION) python -m swarmresearch.verify --full

docker-reproducibility: ## Run reproducibility check in Docker
	@echo "Running reproducibility check in Docker..."
	docker-compose --profile reproducibility run --rm reproducibility

reproducibility: ## Run reproducibility check locally
	@echo "Running reproducibility check..."
	./venv/bin/python -m reproducibility.check_reproducibility \
		--runs 3 \
		--seed $(SEED) \
		--output $(OUTPUT_DIR)/reproducibility_test

# CI/CD targets
ci-test: setup verify ## Run CI tests
	./venv/bin/python -m pytest tests/ -v --cov=swarmresearch

ci-benchmark: setup download verify run ## Run CI benchmark
	@echo "CI benchmark complete"

# Development targets
format: ## Format code with black
	./venv/bin/black reproducibility/ tests/

lint: ## Lint code with flake8
	./venv/bin/flake8 reproducibility/ tests/

typecheck: ## Type check with mypy
	./venv/bin/mypy reproducibility/

dev-setup: setup ## Setup development environment
	./venv/bin/pip install -r requirements-dev.txt
