import json
from pathlib import Path

import pytest

from foundry.data.dataset import (
    ChatTemplate,
    DatasetError,
    DPORecord,
    Message,
    SFTRecord,
    dataset_fingerprint,
    dataset_stats,
    detect_kind,
    estimate_tokens,
    load_dataset,
    load_dpo,
    load_sft,
    record_estimated_tokens,
    render_chat_template,
    split_records,
)

ROOT = Path(__file__).resolve().parents[1]
SFT_PATH = ROOT / "examples" / "sft.jsonl"
DPO_PATH = ROOT / "examples" / "dpo.jsonl"


def write_lines(path: Path, rows) -> Path:
    path.write_text(
        "\n".join(
            row if isinstance(row, str) else json.dumps(row) for row in rows
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_load_sft_examples():
    records = load_sft(SFT_PATH)
    assert len(records) == 8
    assert all(isinstance(record, SFTRecord) for record in records)
    assert all(record.messages[-1].role == "assistant" for record in records)
    assert len({record.record_id for record in records}) == 8


def test_load_dpo_examples():
    records = load_dpo(DPO_PATH)
    assert len(records) == 8
    assert all(isinstance(record, DPORecord) for record in records)
    assert all(record.chosen != record.rejected for record in records)


def test_detect_and_load_dataset():
    assert detect_kind(SFT_PATH) == "sft"
    assert detect_kind(DPO_PATH) == "dpo"
    kind, records = load_dataset(SFT_PATH)
    assert kind == "sft"
    assert len(records) == 8
    kind, records = load_dataset(DPO_PATH, kind="dpo")
    assert kind == "dpo"
    with pytest.raises(DatasetError, match="unknown dataset kind"):
        load_dataset(SFT_PATH, kind="preference")


def test_render_chat_template_uses_special_tokens():
    messages = (
        Message(role="system", content="Be brief."),
        Message(role="user", content="Hi"),
        Message(role="assistant", content="Hello."),
    )
    rendered = render_chat_template(messages)
    assert rendered.startswith("<|bos|>")
    assert rendered.endswith("<|eos|>")
    assert "<|system|>Be brief.<|end|>\n" in rendered
    assert "<|assistant|>Hello.<|end|>" in rendered
    generation = render_chat_template(messages, add_generation_prompt=True)
    assert generation.endswith("<|assistant|>\n")
    assert not generation.endswith("<|eos|>")


def test_custom_chat_template():
    template = ChatTemplate(bos_token="<s>", eos_token="</s>", separator="")
    rendered = template.render((Message(role="user", content="hi"),))
    assert rendered == "<s><|user|>hi<|end|></s>"
    assert template.role_token("tool") == "<|tool|>"
    with pytest.raises(DatasetError):
        template.role_token("narrator")


def test_render_rejects_empty_conversation():
    with pytest.raises(DatasetError):
        render_chat_template(())


def test_estimate_tokens_is_character_based():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2
    assert estimate_tokens("abcdefgh", chars_per_token=2) == 4
    with pytest.raises(ValueError):
        estimate_tokens("abc", chars_per_token=0)


def test_sft_stats_match_definition():
    records = load_sft(SFT_PATH)
    stats = dataset_stats(records)
    assert stats.kind == "sft"
    assert stats.records == 8
    assert stats.messages == sum(len(record.messages) for record in records)
    assert stats.messages == sum(stats.roles.values())
    assert stats.characters == sum(
        len(message.content)
        for record in records
        for message in record.messages
    )
    expected_tokens = sum(
        estimate_tokens(render_chat_template(record.messages)) for record in records
    )
    assert stats.estimated_tokens_total == expected_tokens
    assert stats.mean_tokens == pytest.approx(expected_tokens / 8)
    assert 0 < stats.median_tokens <= stats.p95_tokens <= stats.max_tokens
    payload = stats.to_dict()
    assert payload["chars_per_token"] == 4.0
    assert "estimate" in payload["token_estimate"]


def test_dpo_stats_and_record_estimate():
    records = load_dpo(DPO_PATH)
    stats = dataset_stats(records)
    assert stats.kind == "dpo"
    assert stats.records == 8
    assert stats.messages == 24
    assert stats.roles == {"prompt": 8, "chosen": 8, "rejected": 8}
    expected = sum(
        estimate_tokens(record.prompt)
        + estimate_tokens(record.chosen)
        + estimate_tokens(record.rejected)
        for record in records
    )
    assert stats.estimated_tokens_total == expected
    assert stats.estimated_tokens_total == sum(
        record_estimated_tokens(record) for record in records
    )


def test_stats_reject_empty_and_mixed_data():
    with pytest.raises(DatasetError, match="no records"):
        dataset_stats([])
    with pytest.raises(DatasetError, match="mix"):
        dataset_stats([load_sft(SFT_PATH)[0], load_dpo(DPO_PATH)[0]])


def test_split_is_deterministic_and_sized():
    records = load_sft(SFT_PATH)
    first = split_records(records, val_ratio=0.25, seed=7)
    second = split_records(records, val_ratio=0.25, seed=7)
    assert first.train == second.train
    assert first.val == second.val
    assert len(first.train) == 6
    assert len(first.val) == 2
    ids = {record.record_id for record in first.train + first.val}
    assert ids == {record.record_id for record in records}
    assert first.total == 8


def test_split_edge_cases():
    records = load_sft(SFT_PATH)
    empty = split_records([], val_ratio=0.2, seed=1)
    assert empty.train == () and empty.val == ()
    no_val = split_records(records, val_ratio=0.0)
    assert no_val.val == ()
    assert len(no_val.train) == 8
    single = split_records(records[:1], val_ratio=0.5)
    assert single.val == ()
    assert len(single.train) == 1
    with pytest.raises(DatasetError):
        split_records(records, val_ratio=1.0)
    with pytest.raises(DatasetError):
        split_records(records, val_ratio=-0.1)


def test_fingerprint_is_stable_and_content_sensitive():
    records = load_sft(SFT_PATH)
    first = dataset_fingerprint(records)
    second = dataset_fingerprint(load_sft(SFT_PATH))
    assert first == second
    assert len(first) == 64
    assert dataset_fingerprint(records[:4]) != first


@pytest.mark.parametrize(
    "record, expected",
    [
        ({"messages": []}, "non-empty array"),
        ({"messages": [{"role": "user", "content": "hi"}]}, "assistant"),
        ({"messages": [{"role": "narrator", "content": "hi"}]}, "role"),
        ({"messages": [{"role": "assistant", "content": ""}]}, "content"),
        (
            {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "ok"},
                    {"role": "user", "content": "again"},
                ]
            },
            "final",
        ),
        ({"messages": "nope"}, "non-empty array"),
        ({"messages": [{"role": "user"}]}, "content"),
    ],
)
def test_invalid_sft_records_raise(tmp_path, record, expected):
    path = write_lines(tmp_path / "bad.jsonl", [record])
    with pytest.raises(DatasetError, match=expected):
        load_sft(path)


def test_sft_error_reports_file_and_line(tmp_path):
    path = write_lines(
        tmp_path / "bad.jsonl",
        [load_sft(SFT_PATH)[0].to_dict(), {"messages": "broken"}],
    )
    with pytest.raises(DatasetError) as excinfo:
        load_sft(path)
    message = str(excinfo.value)
    assert ":2:" in message
    assert "non-empty array" in message


def test_malformed_json_reports_line(tmp_path):
    path = write_lines(tmp_path / "bad.jsonl", ["{not json}"])
    with pytest.raises(DatasetError, match="invalid JSON"):
        load_sft(path)


@pytest.mark.parametrize(
    "record, expected",
    [
        ({"prompt": "p", "chosen": "c"}, "rejected"),
        ({"prompt": "p", "chosen": 3, "rejected": "r"}, "chosen"),
        ({"prompt": "  ", "chosen": "c", "rejected": "r"}, "prompt"),
    ],
)
def test_invalid_dpo_records_raise(tmp_path, record, expected):
    path = write_lines(tmp_path / "bad.jsonl", [record])
    with pytest.raises(DatasetError, match=expected):
        load_dpo(path)


def test_missing_and_empty_files(tmp_path):
    with pytest.raises(DatasetError, match="file not found"):
        load_sft(tmp_path / "absent.jsonl")
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    assert load_sft(empty) == []
    with pytest.raises(DatasetError, match="empty"):
        detect_kind(empty)


def test_message_and_record_validation():
    with pytest.raises(DatasetError):
        Message(role="narrator", content="hi")
    with pytest.raises(DatasetError):
        Message(role="user", content="  ")
    with pytest.raises(DatasetError):
        SFTRecord(messages=())
    with pytest.raises(DatasetError):
        SFTRecord(messages=(Message(role="user", content="hi"),))
    with pytest.raises(DatasetError):
        DPORecord(prompt="p", chosen="", rejected="r")
