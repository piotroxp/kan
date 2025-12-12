# Makefile for KAN Speech Model Project
# Wraps CMake, Conan, and common operations

.PHONY: help build clean test train train-quick install-deps configure all examples checkpoints

# Default target
.DEFAULT_GOAL := help

# Configuration
BUILD_DIR := build
CMAKE_BUILD_TYPE ?= Release
CONAN_PROFILE ?= default
BATCH_SIZE ?= 128
LEARNING_RATE ?= 1e-4
EPOCHS ?= 50
CHECKPOINT_DIR ?= checkpoints
DATASET_PATH ?= ../fsd50k

# Detect number of CPU cores
NPROC := $(shell nproc 2>/dev/null || echo 4)

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)KAN Speech Model - Available Targets:$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Configuration:$(NC)"
	@echo "  BUILD_DIR: $(BUILD_DIR)"
	@echo "  CMAKE_BUILD_TYPE: $(CMAKE_BUILD_TYPE)"
	@echo "  BATCH_SIZE: $(BATCH_SIZE)"
	@echo "  LEARNING_RATE: $(LEARNING_RATE)"
	@echo "  EPOCHS: $(EPOCHS)"
	@echo "  CHECKPOINT_DIR: $(CHECKPOINT_DIR)"
	@echo "  DATASET_PATH: $(DATASET_PATH)"
	@echo ""

install-deps: ## Install Conan dependencies
	@echo "$(BLUE)Installing Conan dependencies...$(NC)"
	@cd $(BUILD_DIR) && conan install .. --build=missing -s build_type=$(CMAKE_BUILD_TYPE)
	@echo "$(GREEN)✓ Dependencies installed$(NC)"

configure: ## Configure CMake build
	@echo "$(BLUE)Configuring CMake...$(NC)"
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake .. -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE)
	@echo "$(GREEN)✓ CMake configured$(NC)"

build: configure ## Build the project
	@echo "$(BLUE)Building project...$(NC)"
	@cd $(BUILD_DIR) && cmake --build . -j$(NPROC)
	@echo "$(GREEN)✓ Build complete$(NC)"

rebuild: clean build ## Clean and rebuild from scratch

all: install-deps configure build ## Install deps, configure, and build

test: build ## Run tests
	@echo "$(BLUE)Running tests...$(NC)"
	@cd $(BUILD_DIR) && ctest --output-on-failure || ./tests/tests
	@echo "$(GREEN)✓ Tests complete$(NC)"

train: build ## Train the model (uses BATCH_SIZE, LEARNING_RATE, EPOCHS, CHECKPOINT_DIR)
	@echo "$(BLUE)Starting training...$(NC)"
	@echo "$(YELLOW)Configuration:$(NC)"
	@echo "  Batch size: $(BATCH_SIZE)"
	@echo "  Learning rate: $(LEARNING_RATE)"
	@echo "  Epochs: $(EPOCHS)"
	@echo "  Checkpoint dir: $(CHECKPOINT_DIR)"
	@echo ""
	@cd $(BUILD_DIR) && ./train $(BATCH_SIZE) $(LEARNING_RATE) $(EPOCHS) $(CHECKPOINT_DIR)

train-quick: build ## Quick training run (1 epoch, batch 16)
	@echo "$(BLUE)Starting quick training run...$(NC)"
	@cd $(BUILD_DIR) && ./train 16 1e-4 1 checkpoints

train-gpu: build ## Train with optimal GPU settings (batch 128, 50 epochs)
	@echo "$(BLUE)Starting GPU-optimized training...$(NC)"
	@cd $(BUILD_DIR) && ./train 128 1e-4 50 checkpoints

examples: build ## Build and list example executables
	@echo "$(BLUE)Example executables:$(NC)"
	@ls -1 $(BUILD_DIR)/*_example 2>/dev/null || echo "  No examples found"
	@echo ""
	@echo "$(YELLOW)Run examples with:$(NC)"
	@echo "  make run-example EXAMPLE=audio_example"
	@echo "  make run-example EXAMPLE=full_model_example"

run-example: build ## Run an example (set EXAMPLE=name)
	@if [ -z "$(EXAMPLE)" ]; then \
		echo "$(RED)Error: EXAMPLE not set$(NC)"; \
		echo "Usage: make run-example EXAMPLE=audio_example"; \
		exit 1; \
	fi
	@if [ ! -f "$(BUILD_DIR)/$(EXAMPLE)" ]; then \
		echo "$(RED)Error: Example '$(EXAMPLE)' not found$(NC)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Running $(EXAMPLE)...$(NC)"
	@cd $(BUILD_DIR) && ./$(EXAMPLE)

checkpoints: ## List checkpoints
	@echo "$(BLUE)Checkpoints in $(BUILD_DIR)/$(CHECKPOINT_DIR):$(NC)"
	@if [ -d "$(BUILD_DIR)/$(CHECKPOINT_DIR)" ]; then \
		ls -lh $(BUILD_DIR)/$(CHECKPOINT_DIR)/*.ckpt 2>/dev/null || echo "  No checkpoints found"; \
	else \
		echo "  Checkpoint directory does not exist"; \
	fi

clean: ## Clean build artifacts
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	@rm -rf $(BUILD_DIR)
	@echo "$(GREEN)✓ Clean complete$(NC)"

clean-conan: ## Clean Conan cache and build files
	@echo "$(BLUE)Cleaning Conan files...$(NC)"
	@rm -rf $(BUILD_DIR)/.conan
	@rm -f $(BUILD_DIR)/conanbuildinfo.* $(BUILD_DIR)/conan_toolchain.cmake
	@echo "$(GREEN)✓ Conan clean complete$(NC)"

clean-all: clean clean-conan ## Clean everything including Conan cache

gpu-info: ## Show GPU information
	@echo "$(BLUE)GPU Information:$(NC)"
	@rocm-smi --showuse 2>/dev/null || echo "  rocm-smi not available"
	@echo ""
	@if [ -f "$(BUILD_DIR)/train" ]; then \
		echo "$(BLUE)Running GPU detection from build...$(NC)"; \
		cd $(BUILD_DIR) && ./train 1 1e-4 0 checkpoints 2>&1 | head -15; \
	fi

status: ## Show project status
	@echo "$(BLUE)Project Status:$(NC)"
	@echo ""
	@echo "$(YELLOW)Build:$(NC)"
	@if [ -f "$(BUILD_DIR)/train" ]; then \
		echo "  $(GREEN)✓ Training executable built$(NC)"; \
	else \
		echo "  $(RED)✗ Training executable not found$(NC)"; \
	fi
	@if [ -f "$(BUILD_DIR)/tests/tests" ]; then \
		echo "  $(GREEN)✓ Tests built$(NC)"; \
	else \
		echo "  $(RED)✗ Tests not built$(NC)"; \
	fi
	@echo ""
	@echo "$(YELLOW)Checkpoints:$(NC)"
	@if [ -d "$(BUILD_DIR)/$(CHECKPOINT_DIR)" ]; then \
		COUNT=$$(find $(BUILD_DIR)/$(CHECKPOINT_DIR) -name "*.ckpt" 2>/dev/null | wc -l); \
		if [ $$COUNT -gt 0 ]; then \
			echo "  $(GREEN)✓ Found $$COUNT checkpoint(s)$(NC)"; \
		else \
			echo "  $(YELLOW)○ No checkpoints yet$(NC)"; \
		fi \
	else \
		echo "  $(YELLOW)○ Checkpoint directory not created$(NC)"; \
	fi
	@echo ""
	@echo "$(YELLOW)Dataset:$(NC)"
	@if [ -d "$(DATASET_PATH)" ]; then \
		echo "  $(GREEN)✓ Dataset found at $(DATASET_PATH)$(NC)"; \
	else \
		echo "  $(RED)✗ Dataset not found at $(DATASET_PATH)$(NC)"; \
		echo "    Set DATASET_PATH to your FSD50K dataset location"; \
	fi

watch-gpu: ## Watch GPU usage (Ctrl+C to stop)
	@echo "$(BLUE)Watching GPU usage... (Ctrl+C to stop)$(NC)"
	@watch -n 1 'rocm-smi --showuse 2>&1 | head -10'

# Development shortcuts
dev-build: CMAKE_BUILD_TYPE=Debug
dev-build: build ## Build in Debug mode

quick-test: build ## Quick test run
	@cd $(BUILD_DIR) && timeout 30 ./tests/tests 2>&1 | head -50

# Documentation
docs: ## Generate documentation (placeholder)
	@echo "$(YELLOW)Documentation generation not yet implemented$(NC)"

# Install (if needed)
install: build ## Install the project (placeholder)
	@echo "$(YELLOW)Installation not yet implemented$(NC)"

# Show this help by default
.DEFAULT_GOAL := help



