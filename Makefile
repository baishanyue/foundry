PYTHON ?= python3
export PYTHONPATH := src

.PHONY: test lint demo bench

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m compileall -q src tests benchmarks

demo:
	$(PYTHON) -m foundry stats --data examples/sft.jsonl
	$(PYTHON) -m foundry stats --data examples/dpo.jsonl
	$(PYTHON) -m foundry schedule --steps 20 --warmup 4 --every 4
	$(PYTHON) -m foundry card --data examples/sft.jsonl \
		--base-model Qwen/Qwen2.5-7B-Instruct --hidden-size 4096 \
		--out /tmp/foundry-model-card.md

bench:
	$(PYTHON) benchmarks/bench.py
