"""Batching behaviour of the local fine-tuned corrector — no GPU, no model load.

Fakes the tokenizer/model so the length-sorted batching, the per-batch
max_new_tokens cap and the cut-off guard are all exercised on CPU in CI.
"""

from types import SimpleNamespace

import torch

from app.services.llm_processing import local_client


STOP, PAD = 106, 0          # <end_of_turn>, <pad> — the real gemma-3 ids
PROMPT_BASE, OUT_BASE = 1000, 2000


class _Enc(dict):
    """Stands in for BatchEncoding: unpacks as **kwargs, indexes, .to(device)."""

    def to(self, device):
        return self


class FakeTokenizer:
    """One id per prompt/output string; word counts for the length probe."""

    pad_token_id = PAD
    padding_side = "right"

    def __init__(self):
        self.prompts: list[str] = []
        self.outputs: list[str] = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return messages[-1]["content"]

    def __call__(self, text, return_tensors=None, padding=False, add_special_tokens=True):
        if return_tensors is None:
            # Length probe: token count == word count.
            return SimpleNamespace(input_ids=[t.split() for t in text])
        ids = []
        for t in text:
            ids.append(PROMPT_BASE + len(self.prompts))
            self.prompts.append(t)
        return _Enc(
            input_ids=torch.tensor(ids).unsqueeze(1),
            attention_mask=torch.ones(len(ids), 1, dtype=torch.long),
        )

    def batch_decode(self, rows, skip_special_tokens=True):
        return [
            " ".join(self.outputs[i - OUT_BASE] for i in row.tolist() if i >= OUT_BASE)
            for row in rows
        ]


class FakeModel:
    generation_config = SimpleNamespace(eos_token_id=[1, STOP])

    def __init__(self, tok, correct, no_stop=()):
        self.tok, self.correct, self.no_stop = tok, correct, set(no_stop)
        self.caps: list[int] = []
        self.batches: list[list[str]] = []

    def generate(self, input_ids=None, attention_mask=None, max_new_tokens=None,
                 do_sample=None, pad_token_id=None):
        self.caps.append(max_new_tokens)
        prompts = [self.tok.prompts[i - PROMPT_BASE] for i in input_ids[:, 0].tolist()]
        self.batches.append(prompts)
        rows = []
        for p in prompts:
            oid = OUT_BASE + len(self.tok.outputs)
            self.tok.outputs.append(self.correct(p))
            # No stop token == generation hit the cap mid-line.
            rows.append([oid] if p in self.no_stop else [oid, STOP])
        width = max(len(r) for r in rows)
        padded = [r + [PAD] * (width - len(r)) for r in rows]
        return torch.cat([input_ids, torch.tensor(padded)], dim=1)


def _run(monkeypatch, tmp_path, text, correct=str.upper, no_stop=(), batch_size=2):
    tok = FakeTokenizer()
    model = FakeModel(tok, correct, no_stop)

    import app.utils.torch_device as td
    monkeypatch.setattr(td, "select_torch_device", lambda: "cuda")
    monkeypatch.setattr(local_client, "_adapter_dir", lambda: str(tmp_path))
    monkeypatch.setattr(local_client, "_load", lambda *a, **k: (tok, model, "cpu"))

    out = local_client.post_process_text_finetuned(text, batch_size=batch_size)
    return out, model


def test_batches_group_by_length_and_cap_scales(monkeypatch, tmp_path):
    long_a, long_b = " ".join("ab" * 1 for _ in range(10)), " ".join("cd" for _ in range(10))
    text = "\n".join(["x", long_a, "y", long_b])

    out, model = _run(monkeypatch, tmp_path, text)

    # Short lines batch with short, long with long — never one of each.
    assert [len(b) for b in model.batches] == [2, 2]
    assert {len(p.split()) for p in model.batches[0]} == {1}
    assert {len(p.split()) for p in model.batches[1]} == {10}

    # Cap is derived per batch (1.5x + 16), not the flat 160 it used to be.
    assert sorted(model.caps) == [17, 31]

    # Original line order survives the sort — the frontend maps back by index.
    assert out.split("\n") == ["X", long_a.upper(), "Y", long_b.upper()]


def test_blank_lines_are_preserved(monkeypatch, tmp_path):
    out, _ = _run(monkeypatch, tmp_path, "one\n\ntwo")
    assert out.split("\n") == ["ONE", "", "TWO"]


def test_cut_off_line_keeps_the_original(monkeypatch, tmp_path):
    # A row with no stop token was truncated at the cap; writing that back would
    # silently chop the line, so the raw line must survive instead.
    out, _ = _run(monkeypatch, tmp_path, "keep me\ncorrect me", no_stop=["keep me"])
    assert out.split("\n") == ["keep me", "CORRECT ME"]
