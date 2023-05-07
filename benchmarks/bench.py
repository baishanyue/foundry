#!/usr/bin/env python3
"""Measure Foundry dataset and planning throughput.

The benchmark writes a deterministic synthetic SFT file, then times the
standard-library code paths used by the toolkit. It measures wall time on one
Python process and reports medians over repeated runs. Results are indicative:
they describe this machine and this workload, not other projects or real
tokenizer throughput.

Run ``python benchmarks/bench.py`` and paste the table into the README.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foundry.data.dataset import (  # noqa: E402
    dataset_stats,
    estimate_tokens,
    load_sft,
    Message,
    record_estimated_tokens,
    render_chat_template,
    split_records,
)
from foundry.adapter.lora import LoRAConfig, plan_adapter  # noqa: E402
from foundry.schedules.schedule import LRSchedule  # noqa: E402

SYSTEM = (
    "You are a precise assistant for software engineers. Keep answers short, "
    "name the tradeoff, and include one concrete next step."
)

TOPICS = (
    (
        "Why does a database index make some writes slower?",
        "Each insert or update also maintains the index, and random key order can "
        "force extra page splits. Reads get faster while writes pay that cost.",
    ),
    (
        "What should I check before raising a training batch size?",
        "Measure memory headroom, keep the effective batch size fixed, and watch "
        "for a learning-rate that was tuned for the old batch size.",
    ),
    (
        "How do I make a webhook handler safe to retry?",
        "Verify the signature, persist the event id, return 2xx after storing it, "
        "and make the downstream handler idempotent.",
    ),
    (
        "When is gradient accumulation useful?",
        "It raises the effective batch size when activations no longer fit in "
        "memory, at the cost of extra step latency and optimizer overhead.",
    ),
    (
        "What is the difference between a token and a character?",
        "A token is a unit produced by a tokenizer and can span characters; a "
        "character count is only a rough budget estimate.",
    ),
    (
        "How should a service split a large JSONL file for evaluation?",
        "Shuffle with a recorded seed, hold out a fixed fraction, and write the "
        "split manifest next to the files so the run is reproducible.",
    ),
    (
        "Why use a warmup before cosine decay?",
        "Warmup limits early parameter movement while optimizer statistics are "
        "noisy; cosine decay then lowers the rate smoothly to a floor.",
    ),
    (
        "What belongs in an adapter manifest?",
        "Base model, rank, alpha, scaling, target modules, token accounting, and "
        "a dataset fingerprint so a checkpoint can be audited later.",
    ),
)

MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
MODULE_SHAPES = {
    "q_proj": (4096, 4096),
    "k_proj": (4096, 1024),
    "v_proj": (4096, 1024),
    "o_proj": (4096, 4096),
}


def cpu_label() -> str:
    if sys.platform == "darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return platform.processor() or platform.machine() or "unknown"


def environment_line() -> str:
    return (
        f"CPython {platform.python_version()} · {platform.platform()} · "
        f"{cpu_label()} · {os.cpu_count() or 1} logical CPUs"
    )


def build_sft_file(path: Path, count: int) -> int:
    """Write a deterministic SFT dataset and return its estimated token count."""

    tokens = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for index in range(count):
            question, answer = TOPICS[index % len(TOPICS)]
            record = {
                "id": f"bench-{index:06d}",
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {
                        "role": "user",
                        "content": f"{question} (review {index:06d})",
                    },
                    {
                        "role": "assistant",
                        "content": (
                            f"{answer} Context: request {index:06d} in the "
                            "offline throughput fixture."
                        ),
                    },
                ],
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            messages = tuple(
                Message(role=item["role"], content=item["content"])
                for item in record["messages"]
            )
            tokens += estimate_tokens(render_chat_template(messages))
    return tokens


def median_seconds(function: Callable[[], Any], repeat: int) -> float:
    samples: list[float] = []
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def throughput(count: int, seconds: float) -> float:
    return count / seconds if seconds > 0 else float("inf")


def format_rate(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:,.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:,.1f}k"
    return f"{value:,.1f}"


def run(count: int, repeat: int) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="foundry-bench-") as directory:
        path = Path(directory) / "bench.jsonl"
        total_tokens = build_sft_file(path, count)
        load_seconds = median_seconds(lambda: load_sft(path), repeat)
        records = load_sft(path)
        token_total = sum(record_estimated_tokens(record) for record in records)
        render_seconds = median_seconds(
            lambda: [render_chat_template(record.messages) for record in records],
            repeat,
        )
        stats_seconds = median_seconds(lambda: dataset_stats(records), repeat)
        split_seconds = median_seconds(
            lambda: split_records(records, val_ratio=0.1, seed=7), repeat
        )

    lora_config = LoRAConfig(r=16, alpha=32, target_modules=MODULES)
    plan_iterations = 20_000
    plan_seconds = median_seconds(
        lambda: [plan_adapter(lora_config, MODULE_SHAPES) for _ in range(plan_iterations)],
        repeat,
    )

    schedule = LRSchedule(
        total_steps=50_000, base_lr=3e-4, warmup_steps=2_000, min_lr_ratio=0.1
    )
    schedule_seconds = median_seconds(
        lambda: sum(schedule(step) for step in range(schedule.total_steps + 1)),
        repeat,
    )

    rows = [
        {
            "stage": "JSONL load + schema validation",
            "workload": f"{count:,} records · {token_total:,} est. tokens",
            "seconds": load_seconds,
            "throughput": throughput(count, load_seconds),
            "unit": "records/s",
            "tokens_per_second": throughput(token_total, load_seconds),
        },
        {
            "stage": "Chat template render",
            "workload": f"{count:,} conversations",
            "seconds": render_seconds,
            "throughput": throughput(count, render_seconds),
            "unit": "records/s",
            "tokens_per_second": throughput(token_total, render_seconds),
        },
        {
            "stage": "Dataset statistics",
            "workload": f"{count:,} records · {token_total:,} est. tokens",
            "seconds": stats_seconds,
            "throughput": throughput(count, stats_seconds),
            "unit": "records/s",
            "tokens_per_second": throughput(token_total, stats_seconds),
        },
        {
            "stage": "Seeded train/val split",
            "workload": f"{count:,} records · val_ratio 0.1",
            "seconds": split_seconds,
            "throughput": throughput(count, split_seconds),
            "unit": "records/s",
            "tokens_per_second": None,
        },
        {
            "stage": "LoRA parameter plan",
            "workload": f"{plan_iterations:,} plans · 4 target modules",
            "seconds": plan_seconds,
            "throughput": throughput(plan_iterations, plan_seconds),
            "unit": "plans/s",
            "tokens_per_second": None,
        },
        {
            "stage": "Warmup + cosine schedule",
            "workload": f"{schedule.total_steps + 1:,} LR lookups",
            "seconds": schedule_seconds,
            "throughput": throughput(schedule.total_steps + 1, schedule_seconds),
            "unit": "steps/s",
            "tokens_per_second": None,
        },
    ]
    return {
        "environment": environment_line(),
        "count": count,
        "repeat": repeat,
        "estimated_tokens": total_tokens,
        "rows": rows,
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        f"Environment: {result['environment']}",
        "",
        "| Stage | Workload | Median time | Throughput |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in result["rows"]:
        milliseconds = row["seconds"] * 1_000
        rate = f"{format_rate(row['throughput'])} {row['unit']}"
        if row["tokens_per_second"]:
            rate += f" · {format_rate(row['tokens_per_second'])} est. tokens/s"
        lines.append(
            f"| {row['stage']} | {row['workload']} | "
            f"{milliseconds:,.1f} ms | {rate} |"
        )
    lines.extend(
        [
            "",
            "Median of "
            f"{result['repeat']} runs on one Python process. Numbers are indicative: "
            "synthetic data, character-based token estimates, no GPU, and no "
            "claims about other projects.",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=12_000, help="synthetic records")
    parser.add_argument("--repeat", type=int, default=3, help="timed repetitions")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of Markdown")
    parser.add_argument("--out", default=None, help="optional Markdown output path")
    options = parser.parse_args(argv)
    if options.records < 1 or options.repeat < 1:
        parser.error("--records and --repeat must be positive")
    result = run(options.records, options.repeat)
    output = (
        json.dumps(result, indent=2)
        if options.json
        else render_markdown(result)
    )
    print(output)
    if options.out:
        Path(options.out).write_text(output + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
