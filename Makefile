# Every promise the README makes routes through one of these targets.
# `make verify` must pass from a fresh clone with no network access.

PY ?= python3

.PHONY: verify lint test receipts experiments report demo verdict capture-embeddings set-handle clean

verify: lint test receipts
	@echo "verify: all green"

lint:
	ruff check src tests experiments tools

test:
	$(PY) -m pytest --junitxml=junit.xml --cov=rag_stage_attribution --cov-report=json --cov-report=term-missing:skip-covered

receipts:
	$(PY) tools/collect_metrics.py
	$(PY) tools/check_numbers.py

experiments:
	$(PY) experiments/e1_attribution_accuracy.py
	$(PY) experiments/e2_confusable_class.py
	$(PY) experiments/e3_retriever_decoupling.py
	$(PY) experiments/e4_chunking_blame.py
	$(PY) experiments/e5_trace_cost.py

report:
	$(PY) -m rag_stage_attribution.cli report --out docs/experiments/report.md

# The verdict exit code is the interface. make would fold any nonzero code into
# its own exit 2, so this target prints the code and returns 0; branch on the
# console script directly in automation.
verdict:
	-$(PY) -m rag_stage_attribution.cli verdict --run docs/experiments/e1_runs.json; echo "stageattr verdict exit code: $$?"

demo:
	bash tools/demo.sh

# Needs the capture extra and network access to huggingface.co. Regenerates
# fixtures/embeddings/ from fixtures/corpus/ and records the manifest. Nothing
# else in the repository touches the network.
capture-embeddings:
	$(PY) -m rag_stage_attribution.embeddings.capture --out fixtures/embeddings

# Replace the placeholder GitHub handle everywhere it appears.
set-handle:
	@test -n "$(HANDLE)" || (echo "usage: make set-handle HANDLE=yourhandle" && exit 1)
	grep -rl "GITHUB_USERNAME" README.md docs .github 2>/dev/null | xargs -r sed -i "s/GITHUB_USERNAME/$(HANDLE)/g"
	@echo "handle set to $(HANDLE)"

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov coverage.json junit.xml
	find . -name __pycache__ -type d -not -path "./node_modules/*" -exec rm -rf {} +
