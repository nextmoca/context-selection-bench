"""(d) CI smoke for the RULER fetch path.

Executes the fetch path far enough to catch API drift without downloading
corpora or calling a paid API. The expensive parts (clone, corpora, generate)
are not exercised; what IS exercised is the contract that broke silently:
the shim's method names, the pinned-upstream assertion, and fail-loud on a
generator that produces nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import fetch_ruler  # noqa: E402


def test_shim_defines_the_api_ruler_actually_calls():
    """The original defect: shim named text2tokens, RULER calls text_to_tokens."""
    for method in fetch_ruler.REQUIRED_TOKENIZER_METHODS:
        assert f"def {method}(" in fetch_ruler._TOKENIZER_PATCH, (
            f"shim is missing {method}; RULER's generators call it and every task "
            "will fail with AttributeError"
        )


def test_shim_does_not_use_the_old_broken_names():
    for dead in ("def text2tokens(", "def tokens2text("):
        assert dead not in fetch_ruler._TOKENIZER_PATCH


def test_required_methods_match_upstream_naming_convention():
    assert fetch_ruler.REQUIRED_TOKENIZER_METHODS == {"text_to_tokens", "tokens_to_text"}


def test_patch_tokenizer_rejects_upstream_drift(tmp_path):
    """If the pinned upstream class stops defining the API, fail loudly."""
    tok = tmp_path / "scripts" / "data" / "tokenizer.py"
    tok.parent.mkdir(parents=True)
    tok.write_text("class GeminiTokenizer:\n    def text2tokens(self, t):\n        return []\n")
    with pytest.raises(SystemExit, match="drifted"):
        fetch_ruler.patch_tokenizer(tmp_path)


def test_patch_tokenizer_rejects_missing_class(tmp_path):
    tok = tmp_path / "scripts" / "data" / "tokenizer.py"
    tok.parent.mkdir(parents=True)
    tok.write_text("class SomethingElse:\n    pass\n")
    with pytest.raises(SystemExit, match="Could not find GeminiTokenizer"):
        fetch_ruler.patch_tokenizer(tmp_path)


def test_patch_tokenizer_writes_working_shim(tmp_path):
    tok = tmp_path / "scripts" / "data" / "tokenizer.py"
    tok.parent.mkdir(parents=True)
    tok.write_text(
        "class GeminiTokenizer:\n"
        "    def text_to_tokens(self, t):\n        return []\n"
        "    def tokens_to_text(self, x):\n        return ''\n"
    )
    fetch_ruler.patch_tokenizer(tmp_path)
    out = tok.read_text()
    assert "def text_to_tokens(" in out and "def tokens_to_text(" in out
    assert "def text2tokens(" not in out


def test_generate_fails_loudly_when_no_file_is_produced(tmp_path, monkeypatch):
    """The exact silent failure: prepare.py 'succeeds' and writes nothing."""
    monkeypatch.setattr(fetch_ruler, "_run", lambda *a, **k: None)
    monkeypatch.setattr(fetch_ruler, "LENGTHS", ((8192, "8k"),))
    monkeypatch.setattr(fetch_ruler, "RULER_TASKS", ("niah_single_1",))
    with pytest.raises(SystemExit, match="produced no"):
        fetch_ruler.generate(tmp_path, tmp_path / "out", num_samples=100, seed=42)


def test_generate_fails_loudly_on_short_output(tmp_path, monkeypatch):
    out = tmp_path / "out" / "8k" / "niah_single_1"
    out.mkdir(parents=True)

    def fake_run(*a, **k):
        (out / "test.jsonl").write_text('{"index": 1}\n' * 7)

    monkeypatch.setattr(fetch_ruler, "_run", fake_run)
    monkeypatch.setattr(fetch_ruler, "LENGTHS", ((8192, "8k"),))
    monkeypatch.setattr(fetch_ruler, "RULER_TASKS", ("niah_single_1",))
    with pytest.raises(SystemExit, match="produced 7 rows, expected 100"):
        fetch_ruler.generate(tmp_path, tmp_path / "out", num_samples=100, seed=42)


def test_pins_are_unchanged():
    """These pins are what make the dataset reproducible; changing them is a
    scientific decision, not a refactor."""
    assert fetch_ruler.RULER_PINNED_COMMIT == "38da79d79519ef87aa46ae804f838e1eab7f86d7"
    assert fetch_ruler.RANDOM_SEED == 42
    assert fetch_ruler.NUM_SAMPLES == 100


def test_stale_broken_patch_is_detected_not_skipped(tmp_path):
    """The pre-1.1 shim carried the same marker with the wrong method names."""
    tok = tmp_path / "scripts" / "data" / "tokenizer.py"
    tok.parent.mkdir(parents=True)
    tok.write_text(
        "# --- patched by fetch_ruler.py: use the unified google-genai SDK ---\n"
        "class GeminiTokenizer:\n"
        "    def text2tokens(self, t):\n        return []\n"
        "    def tokens2text(self, x):\n        return ''\n"
    )
    with pytest.raises(SystemExit, match="older fetch_ruler.py patch"):
        fetch_ruler.patch_tokenizer(tmp_path)


def test_current_patch_is_left_alone(tmp_path):
    tok = tmp_path / "scripts" / "data" / "tokenizer.py"
    tok.parent.mkdir(parents=True)
    tok.write_text(
        "# --- patched by fetch_ruler.py: use the unified google-genai SDK ---\n"
        "class GeminiTokenizer:\n"
        "    def text_to_tokens(self, t):\n        return []\n"
        "    def tokens_to_text(self, x):\n        return ''\n"
    )
    before = tok.read_text()
    fetch_ruler.patch_tokenizer(tmp_path)
    assert tok.read_text() == before
