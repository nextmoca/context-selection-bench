"""GSM-IC (GSM with Irrelevant Context) data surface for the GSM boundary suite.

BOUNDARY-SUITE FRAMING (read this first). This module belongs to a DIAGNOSTIC
suite, NOT a flagship result. GSM-IC injects exactly ONE irrelevant sentence
INLINE, inside the single question record a solver must keep. That distractor
therefore lives WITHIN one record, which places it OUTSIDE the record-level
selection regime this benchmark studies: a selector that operates on whole
records cannot remove a distractor sentence embedded in the one record it has
to preserve. GSM-IC is included precisely to disclose that boundary honestly,
alongside GSM8K's complementary boundary (clean single-document math where
there is nothing to select away, so selection is a no-op by design). Flagship
claims live with the RULER and the tool/document suites, not here. No
headline/marketing numbers appear in this module by design.

This file is the data-loading foundation only: it turns the real GSM-IC 2-step
dataset into typed items, reconstructs and locates the injected distractor
sentence, segments each question into ordered sentence-units, assembles prompts
using the shared GSM few-shot convention, and draws a stratified sample. Turning
that output into per-record selection inputs is the job of a separate request
builder (example + condition -> ``ContextRequest``), which is intentionally not
implemented here.

Design notes (preserved from the reference implementation so numbers reproduce):

- **Data source**: real GSM-IC, specifically the already-2-step-filtered public
  file ``GSM-IC_2step.json`` published by ``google-research-datasets/GSM-IC`` on
  GitHub (see ``GSMIC_URL`` / ``GSMIC_REVISION`` below). This dataset is not
  distributed through the Hugging Face ``datasets`` hub, so it is fetched over
  plain HTTP via the standard library (``urllib.request``, no third-party HTTP
  dependency and no optional ``datasets`` extra) rather than
  ``datasets.load_dataset``. The file has 34,220 rows, ALL with ``n_steps == 2``
  -- no filtering needed on our side. The revision is factored into a single
  constant so the fetched snapshot is recorded in one place; freeze it to a
  commit SHA to pin a run.
- **Local cache, never committed**: the raw file is cached under a ``.cache``
  directory at the package root. On every call, if the cache file exists it is
  used as-is (no re-fetch) unless ``force_refetch=True`` is passed. If the
  network fetch fails for any reason (offline, 404, etc.), ``fetch_gsmic_raw``
  raises ``RuntimeError`` loudly -- there is no synthetic or fabricated
  fallback, per this benchmark's "real data only" ethos.
- **``item_id``**: the row's position (0-indexed) in the cached raw JSON array.
  This is deterministic and stable AS LONG AS the cached file's content does not
  change -- which is why the file is cached locally (pins the row order/content
  for the lifetime of the cache) rather than re-fetched on every run at the risk
  of upstream drift. If the cache is ever force-refreshed, ``item_id``s should
  be treated as re-derived from whatever the file looks like at that time, and
  any previously-written sample manifest should be regenerated against the same
  cache snapshot it was built from.
- **``base_problem_id``**: an integer ``0..59`` assigned by FIRST-APPEARANCE
  order of each row's ``original_question`` while scanning the raw file in its
  fixed ``item_id`` order (i.e. ``base_problem_id=0`` is the base problem whose
  first variant row appears earliest in the file, etc.). This is the fixed,
  deterministic group iteration order the stratified sampler
  (``sample_gsmic_2step``) uses -- never Python's arbitrary dict/set iteration
  order, never hash-seed-dependent.
- **Gold answer parsing**: GSM-IC's ``answer`` field is already a plain numeric
  string (e.g. ``"72"``), not a ``"#### <n>"``-marker format, so
  ``parse_gsmic_answer`` is simpler than the GSM8K gold parser -- but keeps the
  same defensive tolerance for comma-thousands separators and negative numbers.
- **Distractor sentence reconstruction**: GSM-IC provides ``sentence_template``
  + ``role`` + ``number``, and ``sentence_template.format(role=role,
  number=number)`` reconstructs the EXACT injected sentence as it appears in
  ``new_question`` -- verified against real sample rows. This is deliberately
  used INSTEAD OF diffing ``original_question`` vs. ``new_question`` textually,
  which is fragile: GSM-IC's insertion can involve minor paraphrase/whitespace
  drift elsewhere in the sentence (e.g. a double space in ``original_question``
  that collapses to a single space in ``new_question``, unrelated to the
  distractor insertion). ``distractor_sentence()`` reconstructs the sentence and
  locates it in ``new_question`` using WHITESPACE-TOLERANT matching
  (token-by-token, with a whitespace run allowed between tokens) rather than a
  byte-exact substring check -- a full scan of all 34,220 real rows found 80
  rows (~0.23%) where ``sentence_template`` itself contains a double space that
  collapses to a single space in the actual inserted text (a real upstream data
  quirk, not something introduced here). The function still returns the ACTUAL
  verbatim substring as it appears in ``new_question`` (never a
  fabricated/reconstructed string) and raises ``ValueError`` loudly if no match
  is found at all.
- **Sentence-unit segmentation** (``segment_distractor_question``): splits
  ``new_question`` into ordered sentence-units via ``_split_sentences`` (an
  abbreviation-aware variant of the shared ``[.!?]``-followed-by-whitespace
  convention) and identifies which unit is the injected distractor by matching
  (after whitespace normalization AND collapsing duplicated trailing terminal
  punctuation) against ``distractor_sentence(item)``. Two real upstream data
  quirks required this beyond a naive split, both found by scanning all 34,220
  real rows:
  1. **Duplicated terminal punctuation** (352 rows, ~1.03%): the inserted
     sentence's own period sits immediately adjacent to the next sentence's
     leading text with no space, e.g. literal text ``"...is 9.. If Jose..."`` --
     the naive splitter (which only splits after ``[.!?]`` followed by
     whitespace) folds both periods into one sentence-unit (``"...is 9.."``),
     which no longer string-equals the reconstructed distractor sentence
     (``"...is 9."``) without collapsing the duplicate. Matching (not the stored
     ``sentences`` content) collapses a run of repeated terminal ``.``/``!``/``?``
     to one for comparison purposes only.
  2. **Abbreviation false sentence-boundaries** (240 rows, ~0.70%, entirely
     within ONE base problem): a naive ``[.!?]``-followed-by-whitespace splitter
     treats ``"Mr."`` as a sentence end, corrupting the segmentation for every
     row of that base problem. ``_split_sentences`` merges a split back together
     when the preceding fragment ends in a period directly after a short list of
     common abbreviations (Mr, Mrs, Ms, Dr, Jr, Sr, St, Mt, Prof, vs).
  With both fixes, 0 of 34,220 real rows fail segmentation (verified by a
  full-dataset scan during development). See ``SentenceSegmentation``'s docstring
  for the exact return shape a downstream request builder consumes.
- **Prompt assembly reuse**: per the "few-shot prefix/task prompt must be
  identical plain text across all methods" rule, GSM-IC prompts are assembled
  with the exact SAME few-shot prefix (``build_fewshot_prompt``, imported and
  reused directly from the shared GSM8K data module -- not reimplemented) and
  the exact same ``f"{fewshot_prompt}\\n\\nQuestion: {question}\\nAnswer:"``
  convention (``_assemble_prompt``, likewise imported). There is exactly ONE
  definition of "how a prompt is assembled" in this suite, not two
  independently-maintained copies.
- **Stratified 2,400-item sample**: per GSM-IC's convention of sampling
  uniformly PER BASE PROBLEM (not one global uniform sample over all 34,220
  rows, which could under/over-represent a base problem), ``2400 / 60 = 40``
  items per base problem. ``sample_gsmic_2step`` iterates the 60 base-problem
  groups in the fixed first-appearance order described above, seeded-shuffles
  each group's rows independently (seed offset by ``base_problem_id`` so groups
  do not all shuffle identically), and takes the first ``n / len(groups)`` from
  each. Raises ``ValueError`` if ``n`` does not divide evenly by the group count
  (2400 does, into 40s) rather than silently rounding.
- **Sample manifest**: ``build_sample_manifest`` produces a small,
  git-committable JSON document recording everything needed to verify/regenerate
  the exact same sample (source URL, ``n``, ``seed``, per-group count, ordered
  sampled ``item_id``s, and the represented base-problem ids) without re-running
  network code.
"""
from __future__ import annotations

import json
import re
import random
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .gsm8k_data import Condition, _assemble_prompt, build_fewshot_prompt  # noqa: F401 (reused/re-exported, see module docstring)


# GSM-IC is served as a plain JSON file from the ``google-research-datasets/GSM-IC``
# GitHub repository. The revision is factored into a single constant so the
# fetched snapshot is recorded in one place; freeze it to a commit SHA to pin a run.
GSMIC_REPO = "google-research-datasets/GSM-IC"
GSMIC_REVISION = "main"
GSMIC_URL = (
    f"https://raw.githubusercontent.com/{GSMIC_REPO}/{GSMIC_REVISION}/GSM-IC_2step.json"
)

DEFAULT_SEED = 42
DEFAULT_SAMPLE_N = 2400
EXPECTED_BASE_PROBLEMS = 60
EXPECTED_TOTAL_ROWS = 34220

_CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache" / "gsm_ic"
_CACHE_FILENAME = "gsmic_2step.json"

_NUMERIC_RE = re.compile(r"-?[\d,]+(?:\.\d+)?")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")  # same convention as the shared GSM8K splitter
_TRAILING_DUP_PUNCT_RE = re.compile(r"([.!?])\1+$")  # e.g. "..." / "??" -> single char
# Common abbreviations that must NOT be treated as a sentence boundary (see
# module docstring, "Abbreviation false sentence-boundaries"). Checked
# case-insensitively against the last word of a fragment ending in ".".
_ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "jr", "sr", "st", "mt", "prof", "vs"}


@dataclass(frozen=True)
class GSMICItem:
    """A single GSM-IC (2-step) row with parsed gold answer and a stable base-problem grouping."""

    item_id: int  # position in the cached raw JSON array (see module docstring)
    base_problem_id: int  # 0..59, assigned by first-appearance order of `original_question`
    original_question: str
    new_question: str
    answer: str  # raw plain numeric string, e.g. "72" (NOT a "#### <n>" format)
    gold_numeric: float
    role: str
    number: str
    sentence_template: str
    role_label: str
    number_label: str
    sentence_label: str


@dataclass(frozen=True)
class GSMICPromptExample:
    """One (item, condition) prompt instance -- the unit a request builder consumes.

    Mirrors the GSM8K prompt-example shape: ``condition`` is the only thing that
    differs in content between the CLEAN and DISTRACTOR rows for the same
    ``item_id`` (``question``/``prompt``); ``gold_numeric`` and the few-shot
    prefix are identical across both.
    """

    item_id: int
    base_problem_id: int
    condition: Condition
    question: str  # original_question for "clean", new_question for "distractor"
    gold_numeric: float
    prompt: str  # fully assembled prompt: fixed few-shot prefix + question


@dataclass(frozen=True)
class SentenceSegmentation:
    """Ordered sentence-unit decomposition of a GSM-IC item's ``new_question``.

    This is the shape a downstream request builder consumes to build one
    ``ContextRecord`` per sentence-unit. Note the boundary-suite point: even
    with a perfect segmentation, the distractor is one sentence-unit INSIDE the
    single question record, so record-level selection cannot drop it without
    dropping the record it must keep -- this decomposition exists to make that
    boundary explicit and auditable, not to enable removing the distractor at
    the record level.

    - ``sentences``: the ordered list of sentence-unit strings as split from
      ``new_question`` (original order preserved, whitespace-stripped, empty
      units dropped). A builder may attach one record per entry, marking every
      unit high-importance EXCEPT the one at ``distractor_index``.
    - ``distractor_index``: the index into ``sentences`` of the ONE sentence-unit
      that is the GSM-IC-injected distractor. A builder should tag
      ``sentences[distractor_index]`` as the low-importance/distractor unit -- the
      sentence-granularity analog of an appended irrelevant-context block, except
      here the distractor is inline rather than a separate trailing block.
    - ``clean_minus_distractor`` (property): reassembles all sentence-units
      EXCEPT the distractor one, in their original order, single-space joined --
      i.e. what the question looks like if the distractor sentence is fully
      dropped. Useful for builders and scoring/sanity checks that want "clean
      text reconstructed by removal" without re-deriving the join logic.
    """

    sentences: List[str]
    distractor_index: int

    @property
    def clean_minus_distractor(self) -> str:
        return " ".join(s for i, s in enumerate(self.sentences) if i != self.distractor_index)


def parse_gsmic_answer(raw_answer: str) -> float:
    """Parse GSM-IC's plain numeric ``answer`` field (e.g. "72", "-3", "1,600").

    Simpler than the GSM8K gold parser (no "#### <n>" marker to locate) but keeps
    the same defensive tolerance for comma-thousands separators and negative
    numbers.
    """
    match = _NUMERIC_RE.search(raw_answer.strip())
    if not match:
        raise ValueError(f"Could not parse a numeric GSM-IC answer from: {raw_answer!r}")
    return float(match.group(0).replace(",", ""))


def _cache_path() -> Path:
    return _CACHE_DIR / _CACHE_FILENAME


def fetch_gsmic_raw(force_refetch: bool = False, timeout: int = 60) -> List[dict]:
    """Fetch the real GSM-IC 2-step JSON file, using a local cache when available.

    - If ``force_refetch`` is False (default) and the cache file already exists,
      the cache is read and returned WITHOUT any network call.
    - Otherwise, fetches ``GSMIC_URL`` over plain HTTP. On ANY failure (offline,
      404, malformed payload, etc.) raises ``RuntimeError`` with a clear message
      -- there is no synthetic/fabricated fallback, per this benchmark's "real
      data only" policy.
    - On a successful fetch, writes the raw JSON to the cache path (creating
      parent directories as needed) atomically (write to a temp file, then
      rename) so a crash mid-write can't corrupt the cache.
    """
    cache_path = _cache_path()
    if not force_refetch and cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    request = urllib.request.Request(GSMIC_URL, headers={"User-Agent": "csbench/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw_bytes = response.read()
    except Exception as exc:  # noqa: BLE001 - re-raised with context, not swallowed
        raise RuntimeError(
            f"Failed to fetch GSM-IC dataset from {GSMIC_URL}: {exc}. "
            "Real data only -- no fabricated/synthetic fallback available."
        ) from exc

    try:
        data = json.loads(raw_bytes)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"GSM-IC payload from {GSMIC_URL} was not valid JSON: {exc}") from exc

    if not isinstance(data, list) or not data:
        raise RuntimeError(
            f"Unexpected GSM-IC payload shape from {GSMIC_URL}: expected a non-empty JSON array, "
            f"got {type(data).__name__}"
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f)
    tmp_path.replace(cache_path)
    return data


def _build_items(raw_rows: Sequence[dict]) -> List[GSMICItem]:
    base_id_by_question: Dict[str, int] = {}
    items: List[GSMICItem] = []
    for idx, row in enumerate(raw_rows):
        original_question = row["original_question"]
        if original_question not in base_id_by_question:
            base_id_by_question[original_question] = len(base_id_by_question)
        items.append(
            GSMICItem(
                item_id=idx,
                base_problem_id=base_id_by_question[original_question],
                original_question=original_question,
                new_question=row["new_question"],
                answer=row["answer"],
                gold_numeric=parse_gsmic_answer(row["answer"]),
                role=row["role"],
                number=row["number"],
                sentence_template=row["sentence_template"],
                role_label=row["role_label"],
                number_label=row["number_label"],
                sentence_label=row["sentence_label"],
            )
        )
    return items


def load_gsmic_2step_items(force_refetch: bool = False) -> List[GSMICItem]:
    """Load the real GSM-IC 2-step file (34,220 rows) as ``GSMICItem``s.

    Uses ``fetch_gsmic_raw`` (local cache by default; see its docstring). No
    Hugging Face ``datasets`` dependency is required -- GSM-IC 2-step is a plain
    JSON file fetched over the standard library.
    """
    raw_rows = fetch_gsmic_raw(force_refetch=force_refetch)
    return _build_items(raw_rows)


def _normalize_for_match(s: str) -> str:
    """Whitespace-collapse + trailing-duplicate-punctuation-collapse, for MATCHING only.

    Never used to alter what is actually stored/returned as sentence text -- only
    to compare two strings that may differ in ways that are real, benign upstream
    data quirks (see module docstring) rather than substantive content
    differences.
    """
    collapsed_ws = " ".join(s.split())
    return _TRAILING_DUP_PUNCT_RE.sub(r"\1", collapsed_ws)


def _split_sentences(text: str) -> List[str]:
    """Abbreviation-aware sentence split (see module docstring for why).

    Same base convention as the shared GSM8K sentence splitter (split after
    ``[.!?]`` followed by whitespace), but re-merges a split that landed right
    after a common abbreviation (Mr., Dr., St., etc.) -- a naive split treats
    those as sentence ends, which corrupts segmentation for any GSM-IC base
    problem whose text contains one.
    """
    raw_parts = _SENTENCE_SPLIT_RE.split(text.strip())
    merged: List[str] = []
    for part in raw_parts:
        if merged and merged[-1].endswith("."):
            prior_words = merged[-1][:-1].split()
            last_word = prior_words[-1].lower() if prior_words else ""
            if last_word in _ABBREVIATIONS:
                merged[-1] = f"{merged[-1]} {part}"
                continue
        merged.append(part)
    return merged


def distractor_sentence(item: GSMICItem) -> str:
    """Reconstruct the exact injected distractor sentence for ``item``.

    Uses ``item.sentence_template.format(role=item.role, number=item.number)`` and
    locates it in ``item.new_question`` with whitespace-TOLERANT matching (see
    module docstring -- ~0.23% of real rows have a template with a double space
    that collapses to one in the actual text). Returns the ACTUAL substring as it
    appears in ``new_question`` (never the fabricated/reconstructed string), and
    raises ``ValueError`` loudly (never silently proceeding) if no match is found
    at all.
    """
    reconstructed = item.sentence_template.format(role=item.role, number=item.number)
    tokens = reconstructed.split()
    if not tokens:
        raise ValueError(f"item_id={item.item_id}: empty reconstructed distractor sentence")
    pattern = re.compile(r"\s+".join(re.escape(tok) for tok in tokens))
    match = pattern.search(item.new_question)
    if not match:
        raise ValueError(
            f"item_id={item.item_id}: reconstructed distractor sentence {reconstructed!r} "
            f"could not be located (even with whitespace-tolerant matching) in "
            f"new_question {item.new_question!r}"
        )
    return match.group(0)


def segment_distractor_question(item: GSMICItem) -> SentenceSegmentation:
    """Split ``item.new_question`` into sentence-units and locate the distractor one.

    Splits via ``_split_sentences`` (abbreviation-aware; see module docstring),
    then locates the ONE sentence-unit matching ``distractor_sentence(item)`` via
    ``_normalize_for_match`` (whitespace + trailing-duplicate-punctuation
    tolerant). Raises ``ValueError`` if zero or more than one sentence-unit
    matches -- a segmentation ambiguity/failure is surfaced loudly, never
    silently dropped or guessed at. A full scan of all 34,220 real rows during
    development found zero such failures with this matching logic.
    """
    target = distractor_sentence(item)
    target_normalized = _normalize_for_match(target)

    raw_sentences = [s.strip() for s in _split_sentences(item.new_question) if s.strip()]
    matches = [
        i for i, s in enumerate(raw_sentences) if _normalize_for_match(s) == target_normalized
    ]
    if len(matches) != 1:
        raise ValueError(
            f"item_id={item.item_id}: expected exactly one sentence-unit matching the "
            f"reconstructed distractor sentence {target!r}, found {len(matches)} in "
            f"{raw_sentences!r}"
        )
    return SentenceSegmentation(sentences=raw_sentences, distractor_index=matches[0])


def segment_clean_question(item: GSMICItem) -> List[str]:
    """Split ``item.original_question`` into ordered sentence-units.

    The CLEAN-condition analog of ``segment_distractor_question``: same
    abbreviation-aware ``_split_sentences`` convention, but there is no
    GSM-IC-provided distractor to locate/flag for the clean question (by
    construction -- ``original_question`` has no injected sentence). Provided so a
    downstream request builder can segment BOTH conditions at the same sentence
    granularity and build one record per sentence-unit for either condition
    uniformly.
    """
    return [s.strip() for s in _split_sentences(item.original_question) if s.strip()]


def build_prompt_examples(
    items: Sequence[GSMICItem], fewshot_prompt: str
) -> List[GSMICPromptExample]:
    """Build CLEAN + DISTRACTOR prompt examples for each item.

    Every item yields exactly two ``GSMICPromptExample`` rows sharing ``item_id``
    and ``gold_numeric``: "clean" uses ``original_question``, "distractor" uses
    ``new_question`` verbatim (GSM-IC's distractor is already inline in
    ``new_question`` -- no separate block to append, unlike GSM8K's appended
    distractor block). Both are assembled with the exact same few-shot prefix and
    prompt-assembly convention as the shared GSM8K data module (see module
    docstring).
    """
    examples: List[GSMICPromptExample] = []
    for item in items:
        examples.append(
            GSMICPromptExample(
                item_id=item.item_id,
                base_problem_id=item.base_problem_id,
                condition="clean",
                question=item.original_question,
                gold_numeric=item.gold_numeric,
                prompt=_assemble_prompt(fewshot_prompt, item.original_question),
            )
        )
        examples.append(
            GSMICPromptExample(
                item_id=item.item_id,
                base_problem_id=item.base_problem_id,
                condition="distractor",
                question=item.new_question,
                gold_numeric=item.gold_numeric,
                prompt=_assemble_prompt(fewshot_prompt, item.new_question),
            )
        )
    return examples


def verify_all_base_problems_represented(
    sample: Sequence[GSMICItem],
    expected_base_problems: int = EXPECTED_BASE_PROBLEMS,
    expected_per_group: Optional[int] = None,
) -> None:
    """Assert every expected base problem appears in ``sample`` (and, optionally, at an exact count).

    Raises ``AssertionError`` (with counts) if any base problem is missing, if an
    unexpected extra base problem shows up, or (when ``expected_per_group`` is
    given) if any group's count doesn't match it exactly. Called internally by
    ``sample_gsmic_2step``, and also exposed standalone so tests can call it
    directly against an arbitrary sample.
    """
    counts = Counter(item.base_problem_id for item in sample)
    if len(counts) != expected_base_problems:
        raise AssertionError(
            f"Expected {expected_base_problems} distinct base problems in sample, "
            f"found {len(counts)}: {sorted(counts)}"
        )
    if expected_per_group is not None:
        bad_groups = {bpid: n for bpid, n in counts.items() if n != expected_per_group}
        if bad_groups:
            raise AssertionError(
                f"Expected exactly {expected_per_group} items per base problem, "
                f"but these groups differ: {bad_groups}"
            )


def sample_gsmic_2step(
    items: Optional[Sequence[GSMICItem]] = None,
    n: int = DEFAULT_SAMPLE_N,
    seed: int = DEFAULT_SEED,
) -> List[GSMICItem]:
    """Stratified sample of ``n`` items, ``n / <num base problems>`` per base problem.

    Per GSM-IC's methodology (restricted to the 2-step subset here): samples
    uniformly PER BASE PROBLEM rather than one global uniform sample, so the
    benchmark subset is balanced across all base problems. Groups are iterated in
    a FIXED, deterministic order -- first-appearance order of each base problem's
    ``original_question`` in the loaded item list (i.e. ``base_problem_id``
    ascending, since ``base_problem_id`` is itself assigned by first-appearance
    order; see module docstring) -- never Python's arbitrary dict/set iteration
    order.

    For each group, a ``random.Random(seed + base_problem_id)`` seeded shuffle is
    applied to that group's rows (in their ``item_id`` order beforehand), and the
    first ``n / <num groups>`` are taken. Raises ``ValueError`` if ``n`` does not
    divide evenly by the number of base-problem groups (2400 does, into exactly
    40 per group for the real 60-group dataset), rather than silently rounding.
    Calls ``verify_all_base_problems_represented`` on the result before returning
    (fails loudly on any violation).
    """
    if items is None:
        items = load_gsmic_2step_items()

    groups: Dict[int, List[GSMICItem]] = {}
    group_order: List[int] = []
    for item in items:
        if item.base_problem_id not in groups:
            groups[item.base_problem_id] = []
            group_order.append(item.base_problem_id)
        groups[item.base_problem_id].append(item)

    num_groups = len(group_order)
    if num_groups == 0:
        raise ValueError("No items provided/loaded -- cannot sample.")
    if n % num_groups != 0:
        raise ValueError(
            f"n={n} does not divide evenly across {num_groups} base-problem groups "
            f"({n} / {num_groups} = {n / num_groups})"
        )
    per_group = n // num_groups

    sample: List[GSMICItem] = []
    for base_problem_id in group_order:
        group_items = sorted(groups[base_problem_id], key=lambda it: it.item_id)
        if len(group_items) < per_group:
            raise ValueError(
                f"base_problem_id={base_problem_id} has only {len(group_items)} rows, "
                f"fewer than the requested {per_group} per group"
            )
        rng = random.Random(seed + base_problem_id)
        shuffled = list(group_items)
        rng.shuffle(shuffled)
        sample.extend(shuffled[:per_group])

    verify_all_base_problems_represented(
        sample, expected_base_problems=num_groups, expected_per_group=per_group
    )
    return sample


def build_sample_manifest(
    sample: Sequence[GSMICItem], n: int, seed: int
) -> dict:
    """Build the small, git-committable JSON manifest documenting an exact sample.

    Contains everything needed to verify/regenerate the exact same sample
    deterministically: the source URL, ``n``, ``seed``, per-group count, the
    ordered ``item_id``s actually sampled, and the sorted set of base problem ids
    represented (sanity-checkable against ``EXPECTED_BASE_PROBLEMS``).
    """
    base_ids = sorted(set(item.base_problem_id for item in sample))
    per_group = len(sample) // len(base_ids) if base_ids else 0
    return {
        "source_url": GSMIC_URL,
        "n": n,
        "seed": seed,
        "n_base_problems": len(base_ids),
        "per_group": per_group,
        "base_problem_ids_represented": base_ids,
        "item_ids": [item.item_id for item in sample],
    }
