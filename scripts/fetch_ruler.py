#!/usr/bin/env python3
"""Fetch/generate the official RULER v1 data this harness runs on.

This does NOT reimplement RULER. It clones NVIDIA's official RULER generator at a
pinned commit and runs it as-is, so the resulting ``test.jsonl`` files are
genuine RULER v1 task constructions. Nothing here redistributes RULER or its
source corpora: this script downloads them from upstream at run time.

LICENSING: READ BEFORE USE
---------------------------
- RULER generator code: Apache-2.0 (NVIDIA/RULER).
- Source corpora the generator reads carry their OWN licenses, which flow into
  the generated rows:
    * SQuAD v2, HotpotQA: CC BY-SA 4.0 (share-alike).
    * Paul Graham essays: no open license, all rights reserved (© Paul Graham).
  Generated data is therefore NOT redistributed by this project. You generate it
  locally, under those upstream terms, for your own evaluation.

REQUIREMENTS
------------
- ``git`` (with ``git-lfs``), ``wget``, and a Python 3.11+ interpreter.
- A ``GEMINI_API_KEY`` in the environment: the pinned run used the Gemini
  tokenizer for exact per-model token counts (Gemini has no offline tokenizer,
  so RULER's haystack sizing calls the ``countTokens`` API). Matching the
  published run requires the same tokenizer.

USAGE
-----
    export GEMINI_API_KEY=...
    python datasets/fetch_ruler.py --out data/ruler --work ./_ruler_build

Output layout (what the harness reads):
    data/ruler/<length>/<task>/test.jsonl        # length in {8k, 16k}

The generator is idempotent: a (task, length) whose ``test.jsonl`` already has
``--num-samples`` lines is skipped, so a re-run only fills gaps.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Pinned upstream so a reproduction is deterministic.
RULER_REPO_URL = "https://github.com/NVIDIA/RULER.git"
RULER_PINNED_COMMIT = "38da79d79519ef87aa46ae804f838e1eab7f86d7"

RULER_TASKS = (
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery",
    "vt", "cwe", "fwe", "qa_1", "qa_2",
)
# (max_seq_length, harness length label)
LENGTHS = ((8192, "8k"), (16384, "16k"))
NUM_SAMPLES = 100
RANDOM_SEED = 42
TOKENIZER_TYPE = "gemini"
TOKENIZER_PATH = "gemini-3.1-pro-preview"
MODEL_TEMPLATE_TYPE = "base"  # identity template: no chat wrapper

# The generator's Gemini tokenizer uses the deprecated google.generativeai SDK.
# Swap it for the unified google-genai SDK (behavior identical: an exact
# server-side token count driving RULER's haystack-size search).
_TOKENIZER_PATCH = '''
# --- patched by fetch_ruler.py: use the unified google-genai SDK ---
import os as _os
from google import genai as _genai


class GeminiTokenizer:
    def __init__(self, model_path="gemini-3.1-pro-preview"):
        self.model_path = model_path
        self.client = _genai.Client(api_key=_os.environ["GEMINI_API_KEY"])

    def text2tokens(self, text):
        result = self.client.models.count_tokens(model=self.model_path, contents=text)
        return list(range(result.total_tokens))

    def tokens2text(self, tokens):
        raise NotImplementedError("Gemini has no offline detokenizer")
# --- end patch ---
'''


def _run(cmd: list[str], *, cwd: Path | None = None, env: dict | None = None) -> None:
    print("+", " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def clone_ruler(work: Path) -> Path:
    ruler = work / "RULER"
    if not (ruler / ".git").exists():
        work.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", RULER_REPO_URL, str(ruler)])
    _run(["git", "checkout", RULER_PINNED_COMMIT], cwd=ruler)
    _run(["git", "lfs", "install", "--local"], cwd=ruler)
    _run(["git", "lfs", "pull"], cwd=ruler)  # english_words.json is LFS
    return ruler


def download_corpora(ruler: Path) -> None:
    json_dir = ruler / "scripts" / "data" / "synthetic" / "json"
    _run([sys.executable, "download_paulgraham_essay.py"], cwd=json_dir)
    if not (json_dir / "squad.json").exists():
        _run(["wget", "-O", "squad.json",
              "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"], cwd=json_dir)
    if not (json_dir / "hotpotqa.json").exists():
        # RULER's primary hotpotqa mirror is frequently down; use its documented
        # secondary HuggingFace mirror, pinned to a revision.
        _run(["wget", "-O", "hotpotqa.json",
              "https://huggingface.co/datasets/namlh2004/hotpotqa/resolve/"
              "7e54db4656209750ff487f6fdf8e39a66dba136b/hotpot_dev_distractor_v1.json"], cwd=json_dir)


def patch_tokenizer(ruler: Path) -> None:
    tok = ruler / "scripts" / "data" / "tokenizer.py"
    text = tok.read_text(encoding="utf-8")
    if "fetch_ruler.py: use the unified google-genai SDK" in text:
        return  # already patched
    marker = "class GeminiTokenizer"
    idx = text.find(marker)
    if idx == -1:
        raise SystemExit(
            f"Could not find GeminiTokenizer in {tok}; upstream may have drifted "
            f"from the pinned commit {RULER_PINNED_COMMIT}. Apply the SDK swap by hand."
        )
    # Replace from the class definition to the end of file (it is the last class).
    tok.write_text(text[:idx] + _TOKENIZER_PATCH.lstrip("\n"), encoding="utf-8")
    print(f"patched {tok}", file=sys.stderr)


def generate(ruler: Path, out_root: Path, *, num_samples: int, seed: int) -> None:
    prepare = ruler / "scripts" / "data" / "prepare.py"
    for max_len, label in LENGTHS:
        save_dir = out_root / label
        for task in RULER_TASKS:
            test_jsonl = save_dir / task / "test.jsonl"
            if test_jsonl.exists() and _line_count(test_jsonl) >= num_samples:
                print(f"skip {label}/{task} (already {num_samples} lines)", file=sys.stderr)
                continue
            _run([
                sys.executable, str(prepare),
                "--save_dir", str(save_dir),
                "--benchmark", "synthetic",
                "--task", task,
                "--tokenizer_path", TOKENIZER_PATH,
                "--tokenizer_type", TOKENIZER_TYPE,
                "--max_seq_length", str(max_len),
                "--model_template_type", MODEL_TEMPLATE_TYPE,
                "--num_samples", str(num_samples),
                "--random_seed", str(seed),
                "--subset", "test",
            ], cwd=ruler / "scripts" / "data")


def _line_count(path: Path) -> int:
    with path.open("rb") as fh:
        return sum(1 for line in fh if line.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch/generate official RULER v1 data.")
    parser.add_argument("--out", type=Path, default=Path("data/ruler"))
    parser.add_argument("--work", type=Path, default=Path("_ruler_build"),
                        help="scratch dir for the RULER clone (kept out of the repo)")
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--skip-clone", action="store_true", help="reuse an existing --work/RULER clone")
    args = parser.parse_args()

    if "GEMINI_API_KEY" not in os.environ:
        return _fail("GEMINI_API_KEY is required (RULER's Gemini tokenizer calls the countTokens API).")

    print(
        "NOTE: generated RULER data embeds upstream corpora under their own licenses "
        "(SQuAD/HotpotQA CC BY-SA 4.0; Paul Graham essays all rights reserved). "
        "It is generated locally for your own evaluation and is not redistributed.",
        file=sys.stderr,
    )

    ruler = (args.work / "RULER") if args.skip_clone else clone_ruler(args.work)
    download_corpora(ruler)
    patch_tokenizer(ruler)
    generate(ruler, args.out, num_samples=args.num_samples, seed=args.seed)
    print(f"done -> {args.out}/<length>/<task>/test.jsonl", file=sys.stderr)
    return 0


def _fail(msg: str) -> int:
    print(f"error: {msg}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
