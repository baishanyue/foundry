"""Command-line entry points for Foundry.

Subcommands:

* ``stats``    -- validate a JSONL file and report length statistics
* ``prepare``  -- write train/validation JSONL plus a stats manifest
* ``card``     -- finalize adapter metadata and render a model card
* ``schedule`` -- print a warmup + cosine learning-rate table
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import __version__
from .config import ConfigError, FoundryConfig
from .data.dataset import (
    DatasetError,
    DatasetStats,
    SFTRecord,
    dataset_fingerprint,
    dataset_stats,
    estimate_tokens,
    load_dataset,
    record_estimated_tokens,
    render_chat_template,
    split_records,
    write_jsonl,
)
from .export.export import (
    AdapterMetadata,
    render_model_card,
    write_adapter_metadata,
)
from .adapter.lora import LoRAConfig
from .schedules.schedule import LRSchedule


def _number(value: int | float) -> str:
    if isinstance(value, float) and not value.is_integer():
        return f"{value:,.2f}"
    return f"{int(value):,}"


def _resolve_data(args: argparse.Namespace) -> str:
    path = getattr(args, "data_option", None) or args.data
    if not path:
        raise DatasetError("a dataset path is required (positional or --data)")
    return path


def _load(args: argparse.Namespace) -> tuple[str, list[Any]]:
    args.data = _resolve_data(args)
    return load_dataset(args.data, args.kind)


def _stats_payload(path: str, kind: str, stats: DatasetStats, records: Sequence[Any]) -> dict[str, Any]:
    return {
        "file": str(Path(path)),
        "kind": kind,
        "fingerprint": dataset_fingerprint(records),
        "stats": stats.to_dict(),
    }


def _run_stats(args: argparse.Namespace) -> int:
    kind, records = _load(args)
    if not records:
        raise DatasetError(f"{args.data}: dataset contains no records")
    stats = dataset_stats(records)
    payload = _stats_payload(args.data, kind, stats, records)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    print(f"{'dataset':<12}{Path(args.data)}")
    print(f"{'kind':<12}{kind}")
    print(f"{'records':<12}{_number(stats.records)}")
    label = "turns" if kind == "sft" else "segments"
    print(f"{label:<12}{_number(stats.messages)}")
    print(f"{'characters':<12}{_number(stats.characters)}")
    print(
        f"{'est. tokens':<12}{_number(stats.estimated_tokens_total)} total  "
        f"mean {stats.mean_tokens:.1f}  median {stats.median_tokens:.1f}  "
        f"p95 {stats.p95_tokens:.1f}  max {_number(stats.max_tokens)}"
    )
    print(f"{'estimate':<12}characters / 4 (no tokenizer; approximate)")
    roles = "  ".join(f"{name}={count}" for name, count in sorted(stats.roles.items()))
    print(f"{'roles':<12}{roles}")
    print(f"{'fingerprint':<12}{payload['fingerprint'][:16]}…")
    return 0


def _prepared_row(record: Any, kind: str, index: int) -> dict[str, Any]:
    record_id = record.record_id or f"{kind}-{index:05d}"
    if isinstance(record, SFTRecord):
        text = render_chat_template(record.messages)
        return {
            "id": record_id,
            "text": text,
            "turns": len(record.messages),
            "estimated_tokens": estimate_tokens(text),
        }
    return {
        "id": record_id,
        "prompt": record.prompt,
        "chosen": record.chosen,
        "rejected": record.rejected,
        "estimated_tokens": record_estimated_tokens(record),
    }


def _run_prepare(args: argparse.Namespace) -> int:
    kind, records = _load(args)
    if not records:
        raise DatasetError(f"{args.data}: dataset contains no records")
    stats = dataset_stats(records)
    split = split_records(records, val_ratio=args.val_ratio, seed=args.seed)
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_rows = [
        _prepared_row(record, kind, index)
        for index, record in enumerate(split.train)
    ]
    val_rows = [
        _prepared_row(record, kind, index)
        for index, record in enumerate(split.val)
    ]
    train_path = output / "train.jsonl"
    val_path = output / "val.jsonl"
    stats_path = output / "dataset-stats.json"
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)
    manifest = {
        "source": str(Path(args.data)),
        "kind": kind,
        "fingerprint": dataset_fingerprint(records),
        "split": {
            "seed": args.seed,
            "val_ratio": args.val_ratio,
            "train": len(train_rows),
            "val": len(val_rows),
        },
        "stats": stats.to_dict(),
    }
    stats_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {train_path} ({_number(len(train_rows))} records)")
    print(f"wrote {val_path} ({_number(len(val_rows))} records)")
    print(f"wrote {stats_path}")
    return 0


def _run_card(args: argparse.Namespace) -> int:
    kind, records = _load(args)
    if not records:
        raise DatasetError(f"{args.data}: dataset contains no records")
    stats = dataset_stats(records)
    targets = tuple(
        name.strip() for name in args.target_modules.split(",") if name.strip()
    )
    lora = LoRAConfig(
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=targets,
    )
    shapes: dict[str, tuple[int, int]] = {}
    if args.hidden_size:
        if args.hidden_size < 1:
            raise ValueError("--hidden-size must be a positive integer")
        shapes = {name: (args.hidden_size, args.hidden_size) for name in lora.target_modules}
    training = {
        "backend": "reference loop (no neural network)",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "optimizer": "sgd",
        "schedule": "linear warmup + cosine decay",
        "gradient_accumulation_steps": 1,
        "max_grad_norm": 1.0,
        "note": "declared configuration; no run metrics recorded",
    }
    metadata = AdapterMetadata(
        name=args.name,
        base_model=args.base_model,
        lora=lora,
        dataset=stats,
        training=training,
        module_shapes=shapes,
        dataset_files=(str(Path(args.data)),),
        dataset_fingerprint=dataset_fingerprint(records),
        created=args.created,
        notes=(
            f"Dataset format: {kind.upper()}.",
            "Generated offline by the Foundry CLI.",
        ),
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_model_card(metadata), encoding="utf-8")
    print(f"wrote model card {out_path}")
    if args.adapter_out:
        adapter_path = write_adapter_metadata(metadata, args.adapter_out)
        print(f"wrote adapter metadata {adapter_path}")
    print(
        f"base model {args.base_model} · {_number(stats.records)} records · "
        f"LoRA r={lora.r} alpha={_number(lora.alpha)} scaling={lora.scaling:.4f}"
    )
    return 0


def _run_schedule(args: argparse.Namespace) -> int:
    if args.config:
        config = FoundryConfig.load(args.config)
        schedule = config.to_schedule()
    else:
        schedule = LRSchedule(
            total_steps=args.steps,
            base_lr=args.lr,
            warmup_steps=args.warmup,
            min_lr_ratio=args.min_lr_ratio,
        )
    if args.every < 1:
        raise ValueError("--every must be at least 1")
    total = schedule.total_steps
    sample_steps = sorted(set(range(0, total + 1, args.every)) | {total})
    samples = [(step, schedule(step)) for step in sample_steps]
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write("step,learning_rate\n")
            for step, lr in enumerate(schedule.values()):
                handle.write(f"{step},{lr:.10f}\n")
        stream = sys.stderr if args.json else sys.stdout
        print(f"wrote schedule {out_path}", file=stream)
    if args.json:
        print(
            json.dumps(
                {
                    "schedule": schedule.as_dict(),
                    "samples": [{"step": step, "lr": lr} for step, lr in samples],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    print(f"{'step':>8}  {'learning_rate':>16}")
    for step, lr in samples:
        print(f"{step:>8}  {lr:>16.10f}")
    return 0


def _add_data_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("data", nargs="?", help="path to a JSONL dataset")
    parser.add_argument(
        "--data",
        dest="data_option",
        default=None,
        metavar="PATH",
        help="path to a JSONL dataset (alias for the positional form)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="foundry",
        description=(
            "Validate datasets, plan LoRA adapters, and inspect training "
            "schedules. Pure Python, offline."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    stats_parser = subparsers.add_parser(
        "stats", help="validate a JSONL dataset and print length statistics"
    )
    _add_data_argument(stats_parser)
    stats_parser.add_argument(
        "--kind", choices=("auto", "sft", "dpo"), default="auto", help="dataset format"
    )
    stats_parser.add_argument("--json", action="store_true", help="emit JSON")
    stats_parser.set_defaults(handler=_run_stats)

    prepare_parser = subparsers.add_parser(
        "prepare", help="write train/validation JSONL and a stats manifest"
    )
    _add_data_argument(prepare_parser)
    prepare_parser.add_argument(
        "--kind", choices=("auto", "sft", "dpo"), default="auto", help="dataset format"
    )
    prepare_parser.add_argument(
        "--out-dir", default="prepared", help="directory for prepared files"
    )
    prepare_parser.add_argument(
        "--val-ratio", type=float, default=0.1, help="validation fraction"
    )
    prepare_parser.add_argument(
        "--seed", type=int, default=42, help="shuffle seed for the split"
    )
    prepare_parser.set_defaults(handler=_run_prepare)

    card_parser = subparsers.add_parser(
        "card", help="finalize adapter metadata and render a model card"
    )
    _add_data_argument(card_parser)
    card_parser.add_argument(
        "--kind", choices=("auto", "sft", "dpo"), default="auto", help="dataset format"
    )
    card_parser.add_argument("--base-model", required=True, help="base model id")
    card_parser.add_argument("--name", default="foundry-adapter", help="adapter name")
    card_parser.add_argument("--out", default="model-card.md", help="model card path")
    card_parser.add_argument(
        "--adapter-out", default=None, help="optional adapter manifest path"
    )
    card_parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    card_parser.add_argument(
        "--lora-alpha", type=float, default=32.0, help="LoRA alpha"
    )
    card_parser.add_argument(
        "--lora-dropout", type=float, default=0.05, help="LoRA dropout"
    )
    card_parser.add_argument(
        "--target-modules",
        default="q_proj,v_proj",
        help="comma-separated projection names",
    )
    card_parser.add_argument(
        "--hidden-size",
        type=int,
        default=0,
        help="optional square layer size for parameter estimates",
    )
    card_parser.add_argument("--epochs", type=int, default=1, help="declared epochs")
    card_parser.add_argument(
        "--batch-size", type=int, default=8, help="declared batch size"
    )
    card_parser.add_argument(
        "--learning-rate", type=float, default=2e-4, help="declared learning rate"
    )
    card_parser.add_argument(
        "--created", default=None, help="creation date in ISO format (YYYY-MM-DD)"
    )
    card_parser.set_defaults(handler=_run_card)

    schedule_parser = subparsers.add_parser(
        "schedule", help="print a linear-warmup cosine learning-rate table"
    )
    schedule_parser.add_argument(
        "--config", default=None,
        help="load steps/lr/warmup from a JSON recipe (overrides the flags)",
    )
    schedule_parser.add_argument("--steps", type=int, default=100, help="total steps")
    schedule_parser.add_argument("--warmup", type=int, default=10, help="warmup steps")
    schedule_parser.add_argument("--lr", type=float, default=3e-4, help="peak learning rate")
    schedule_parser.add_argument(
        "--min-lr-ratio", type=float, default=0.1, help="final LR as a fraction of peak"
    )
    schedule_parser.add_argument(
        "--every", type=int, default=10, help="print every Nth step"
    )
    schedule_parser.add_argument("--json", action="store_true", help="emit JSON")
    schedule_parser.add_argument("--out", default=None, help="optional CSV path")
    schedule_parser.set_defaults(handler=_run_schedule)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args) or 0)
    except DatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
