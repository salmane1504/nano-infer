# Shortcuts for common nano-infer tasks. Run `make help` for the list.
# The scripts in tasks/ hold the actual commands and can be run directly.

MODEL ?= model_files/qwen2.5_0.5B-instruct
# Extra CLI or pytest arguments, e.g. make chat ARGS="--temperature 0.7"
ARGS ?=

.PHONY: help setup test chat complete download-model clean

help:
	@echo "Targets:"
	@echo "  setup                 Install runtime and dev dependencies (uv sync)"
	@echo "  test                  Run the test suite (ARGS=... passes pytest options)"
	@echo "  chat                  Start interactive chat with MODEL (ARGS=... for options)"
	@echo "  complete              Raw completion; PROMPT='...' or pipe stdin"
	@echo "  download-model        Download a model (MODEL_REPO=..., MODEL=... for the target dir)"
	@echo "  clean                 Remove __pycache__ and .pytest_cache"
	@echo ""
	@echo "Variables: MODEL=$(MODEL)"

setup:
	tasks/setup.sh

test:
	tasks/test.sh $(ARGS)

chat:
	uv run python -m nano_infer chat $(MODEL) $(ARGS)

complete:
	uv run python -m nano_infer complete $(MODEL) $(if $(PROMPT),"$(PROMPT)") $(ARGS)

download-model:
	tasks/download-model.sh "$(or $(MODEL_REPO),Qwen/Qwen2.5-0.5B-Instruct)" "$(MODEL)"

clean:
	tasks/clean.sh
