import json

from csbench.suites.ruler.data import (
    OfficialRulerExample,
    chunk_text,
    load_examples,
    match_type_for_task,
    score_ruler_answer,
    selection_keywords_for_example,
    split_official_prompt,
)


def test_match_type_for_task():
    assert match_type_for_task("qa_1") == "part"
    assert match_type_for_task("niah_single_1") == "all"


def test_score_ruler_answer_part_and_all():
    # "part": correct if ANY reference is a substring of the prediction
    assert score_ruler_answer("the answer is Lima!", ["lima"], "part") == 1.0
    assert score_ruler_answer("nothing here", ["lima"], "part") == 0.0
    # "all": fraction of references present
    assert score_ruler_answer("a and c", ["a", "b", "c"], "all") == 2 / 3
    # control chars normalized, case-insensitive
    assert score_ruler_answer("ALPHA\x01beta", ["alpha", "beta"], "all") == 1.0
    assert score_ruler_answer("x", [], "all") == 0.0


def test_split_official_prompt_on_question_marker():
    ctx, q = split_official_prompt("some long context here\nQuestion: what is X?")
    assert ctx == "some long context here"
    assert q.startswith("Question:")


def test_selection_keywords_drops_stopwords_and_gold():
    ex = OfficialRulerExample(
        task="qa_1", index=0, input_text="", answer_prefix="", expected_answer=["lima"], length=8192
    )
    kws = selection_keywords_for_example(ex, "What is the capital city Peru?")
    assert "lima" not in kws  # gold excluded
    assert "the" not in kws  # stopword excluded
    assert "capital" in kws and "peru" in kws


def test_chunk_text_covers_all_words():
    text = " ".join(f"w{i}" for i in range(500))
    chunks = chunk_text(text, target_tokens=260, overlap_tokens=30)
    assert len(chunks) >= 2
    assert chunks[0].split()[0] == "w0"
    assert chunks[-1].split()[-1] == "w499"


def test_load_examples_reads_jsonl(tmp_path):
    task_dir = tmp_path / "qa_1"
    task_dir.mkdir()
    (task_dir / "test.jsonl").write_text(
        json.dumps({"index": 0, "input": "ctx\nQuestion: q?", "answer_prefix": "A:", "outputs": ["lima"], "length": 8192})
        + "\n"
    )
    examples = load_examples(tmp_path, "qa_1", limit=10)
    assert len(examples) == 1
    assert examples[0].expected_answer == ("lima",)
    assert examples[0].index == 0
