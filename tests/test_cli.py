import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from foundry.cli import main

ROOT = Path(__file__).resolve().parents[1]
SFT_PATH = ROOT / "examples" / "sft.jsonl"
DPO_PATH = ROOT / "examples" / "dpo.jsonl"


def run_module(*args: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ROOT / "src") + (os.pathsep + existing if existing else "")
    return subprocess.run(
        [sys.executable, "-m", "foundry", *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_module_help_works():
    result = run_module("--help")
    assert result.returncode == 0
    for command in ("stats", "prepare", "card", "schedule"):
        assert command in result.stdout
    version = run_module("--version")
    assert version.returncode == 0
    assert "0.1.0" in version.stdout


@pytest.mark.parametrize("command", ["stats", "prepare", "card", "schedule"])
def test_subcommand_help_works(command):
    result = run_module(command, "--help")
    assert result.returncode == 0
    assert "usage" in result.stdout.lower()


def test_stats_text_and_json(capsys):
    assert main(["stats", str(SFT_PATH)]) == 0
    text = capsys.readouterr().out
    assert "records" in text
    assert "8" in text
    assert "est. tokens" in text
    assert "characters / 4" in text

    assert main(["stats", str(SFT_PATH), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["kind"] == "sft"
    assert payload["stats"]["records"] == 8
    assert len(payload["fingerprint"]) == 64

    assert main(["stats", "--data", str(DPO_PATH)]) == 0
    assert "dpo" in capsys.readouterr().out


def test_prepare_sft(tmp_path):
    output = tmp_path / "prepared"
    assert (
        main(
            [
                "prepare",
                str(SFT_PATH),
                "--out-dir",
                str(output),
                "--val-ratio",
                "0.25",
                "--seed",
                "7",
            ]
        )
        == 0
    )
    train_lines = (output / "train.jsonl").read_text(encoding="utf-8").splitlines()
    val_lines = (output / "val.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(train_lines) == 6
    assert len(val_lines) == 2
    row = json.loads(train_lines[0])
    assert row["text"].startswith("<|bos|>")
    assert row["estimated_tokens"] > 0
    manifest = json.loads((output / "dataset-stats.json").read_text(encoding="utf-8"))
    assert manifest["kind"] == "sft"
    assert manifest["split"] == {"seed": 7, "val_ratio": 0.25, "train": 6, "val": 2}
    assert manifest["stats"]["records"] == 8


def test_prepare_dpo(tmp_path):
    output = tmp_path / "dpo"
    assert (
        main(["prepare", str(DPO_PATH), "--out-dir", str(output), "--seed", "3"])
        == 0
    )
    lines = (output / "train.jsonl").read_text(encoding="utf-8").splitlines()
    assert lines
    row = json.loads(lines[0])
    assert {"prompt", "chosen", "rejected", "estimated_tokens"} <= set(row)
    manifest = json.loads((output / "dataset-stats.json").read_text(encoding="utf-8"))
    assert manifest["kind"] == "dpo"
    assert manifest["split"]["train"] + manifest["split"]["val"] == 8


def test_card_writes_markdown_and_manifest(tmp_path):
    card = tmp_path / "model-card.md"
    adapter = tmp_path / "adapter.json"
    assert (
        main(
            [
                "card",
                str(SFT_PATH),
                "--base-model",
                "Qwen/Qwen2.5-7B-Instruct",
                "--out",
                str(card),
                "--adapter-out",
                str(adapter),
                "--hidden-size",
                "64",
                "--created",
                "2026-11-06",
            ]
        )
        == 0
    )
    text = card.read_text(encoding="utf-8")
    assert "# foundry-adapter" in text
    assert "Qwen/Qwen2.5-7B-Instruct" in text
    assert "LoRA" in text
    assert "64" in text
    payload = json.loads(adapter.read_text(encoding="utf-8"))
    assert payload["method"]["trainable_parameters"] == 16 * (64 + 64) * 2
    assert payload["created"] == "2026-11-06"
    assert payload["dataset"]["records"] == 8


def test_schedule_text_json_and_csv(tmp_path, capsys):
    assert main(["schedule", "--steps", "100", "--warmup", "10", "--lr", "3e-4"]) == 0
    text = capsys.readouterr().out
    assert "learning_rate" in text
    assert "0" in text

    assert (
        main(
            [
                "schedule",
                "--steps",
                "100",
                "--warmup",
                "10",
                "--lr",
                "3e-4",
                "--min-lr-ratio",
                "0.1",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["schedule"]["name"] == "linear-warmup-cosine"
    assert len(payload["samples"]) == 11
    assert payload["samples"][0]["step"] == 0
    assert payload["samples"][0]["lr"] == pytest.approx(3e-5)
    assert payload["samples"][1]["lr"] == pytest.approx(3e-4)
    assert payload["samples"][-1]["step"] == 100
    assert payload["samples"][-1]["lr"] == pytest.approx(3e-5)

    csv_path = tmp_path / "lr.csv"
    assert main(["schedule", "--steps", "10", "--out", str(csv_path)]) == 0
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "step,learning_rate"
    assert len(lines) == 12


def test_errors_return_exit_code_two(tmp_path, capsys):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"messages": "not-a-list"}\n', encoding="utf-8")
    assert main(["stats", str(bad)]) == 2
    captured = capsys.readouterr()
    assert "error:" in captured.err
    assert ":1:" in captured.err
    assert main(["stats", str(tmp_path / "missing.jsonl")]) == 2
    assert "file not found" in capsys.readouterr().err
    assert main(["schedule", "--steps", "10", "--warmup", "20"]) == 2
    assert "warmup_steps" in capsys.readouterr().err
