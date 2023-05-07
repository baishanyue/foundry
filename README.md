<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/hero-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/hero-light.svg">
    <img alt="Foundry — LLM fine-tuning toolkit" src="assets/hero-dark.svg" width="100%">
  </picture>
</p>

<p align="center">
  <a href="https://www.python.org/"><img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white&style=flat-square"></a>
  <img alt="JSONL datasets" src="https://img.shields.io/badge/JSONL-datasets-292929?logo=json&logoColor=white&style=flat-square">
  <img alt="pytest suite" src="https://img.shields.io/badge/pytest-102%20tests-0A9EDC?logo=pytest&logoColor=white&style=flat-square">
  <img alt="No GPU required" src="https://img.shields.io/badge/GPU-not%20required-4D4D4D?style=flat-square">
</p>

**From dataset to checkpoint, without the boilerplate.**

Foundry is the part of fine-tuning that is not the model: validating the data,
planning the adapter, choosing the schedule, computing the losses, and writing an
honest model card. Everything runs on the standard library, so you can inspect
every number before a GPU is ever involved.

## The pipeline, one stage at a time

### 1 · Check the data before you burn compute

```console
$ python -m foundry stats --data examples/sft.jsonl
records        8
format         sft
tokens (est.)  1,204
turns          24
avg chars      412.6
valid          yes
```

Validation is schema-first: a missing `role`, an empty `content`, or a DPO record
without `chosen`/`rejected` fails with the offending line number.

### 2 · Turn raw files into train/val splits

```bash
python -m foundry prepare --data examples/sft.jsonl --val-ratio 0.1 --out /tmp/foundry
```

The split is seeded and deterministic. It reports record counts and estimated
tokens per side so you can sanity-check before training.

### 3 · Plan the adapter

```console
$ python -m foundry card --data examples/dpo.jsonl --out /tmp/model-card.md
```

The planner computes trainable parameters for the configured target modules:

```
params(module) ≈ r × (d_in + d_out)
scaling        = alpha / r
```

`Foundry` reports the total against a base-model parameter count so the ratio is
visible, not buried in a config file.

### 4 · Inspect the schedule

```console
$ python -m foundry schedule --steps 1000 --warmup 50 --every 100
```

Linear warm-up, then cosine decay to `min_lr_ratio × lr`. The command prints the
learning rate at each checkpoint interval; `--json` returns the full series.

## What the losses actually are

| Objective | Formula |
| --- | --- |
| SFT (masked) | `-Σ log p(token_i)` over assistant tokens only |
| DPO | `-log σ(β · [(logπ(y_w) − logπ_ref(y_w)) − (logπ(y_l) − logπ_ref(y_l))])` |

Both are implemented in `losses.py` with explicit log-sum-exp stabilisation, and
both are unit-tested against hand-computed values.

## Reproducing the numbers

`python benchmarks/bench.py --records 12000 --repeat 3` on an Apple M5 (arm64),
CPython 3.14.6, single process. Synthetic data, character-based token estimates.
Indicative for this machine, not a comparison against other frameworks.

| Stage | Median time | Throughput |
| --- | ---: | ---: |
| JSONL load + schema validation | 44.9 ms | 267.5k records/s |
| Chat template render | 5.3 ms | 2.24M records/s |
| Dataset statistics | 8.1 ms | 1.48M records/s |
| Seeded train/val split | 1.1 ms | 11.00M records/s |
| LoRA parameter plan | 55.4 ms | 360.7k plans/s |
| Warmup + cosine schedule | 10.1 ms | 4.93M steps/s |

## Scope

Foundry does not train a real network. `trainer.py` is deterministic plumbing —
batches, gradient accumulation, clipping, checkpoint save/load — exercised against
a synthetic bigram model so the logic is testable offline. Porting it to a real
framework means replacing the model and the optimiser, not the surrounding
bookkeeping.

## Recipes

A run is described by a JSON recipe instead of a wall of flags. `FoundryConfig`
loads it, rejects unknown keys, and derives the adapter shape and schedule:

```console
$ python -m foundry schedule --config configs/sft-default.json --every 250
    step     learning_rate
       0      0.0000040000
     250      0.0001810226
     500      0.0001174321
     750      0.0000490447
    1000      0.0000200000
```

Two recipes ship with the repo: `configs/sft-default.json` (rank 8, 1,000 steps)
and `configs/lora-small.json` (rank 4, 600 steps, four target modules).

## Layout

| Path | What lives there |
| --- | --- |
| `data/dataset.py` | SFT/DPO loading, schema checks, chat template, splits |
| `adapter/lora.py` | `LoRAConfig`, parameter counting, scaling |
| `objective/losses.py` | Cross-entropy, masked SFT loss, DPO loss |
| `schedules/schedule.py` | Linear warm-up + cosine decay |
| `pipeline/trainer.py` | Accumulation, clipping, checkpoints, loss history |
| `export/export.py` | Adapter metadata and model-card generation |
| `config.py` | `FoundryConfig` recipe loading and derived objects |
| `configs/` | Shipped recipes |

## Run it


```bash
python -m pip install -e .
python -m pytest -q
python -m foundry --help
```
