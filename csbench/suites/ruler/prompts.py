"""Model-prompt assembly for the RULER suite.

Each arm feeds its rendered context into the arm-appropriate, byte-faithful
template below. The reference harness used three distinct shapes (a bare
full-context baseline, a scaffolded selected-context template, and a compressed
context template) and they are reproduced here exactly (same headers,
whitespace, ordering) so an outsider reproduces the published numbers.

- ``baseline_prompt``, the full-context control arm: the raw input text with the
  answer prefix appended, no scaffolding at all.
- ``needlepath_prompt``, the selection arm: an instruction header + a
  ``SELECTED OFFICIAL RULER CONTEXT:`` label + the (split) question + answer
  prefix; on a fallback it degrades to the bare baseline, exactly as the
  reference did.
- ``compression_prompt``, a compression arm (LLMLingua-2, CPC): the compressed
  context, the (split) question, and the answer prefix, with no instruction
  header.
"""

from __future__ import annotations


def baseline_prompt(input_text: str, answer_prefix: str) -> str:
    """Full-context control arm: the raw input text plus the answer prefix.

    Byte-faithful to the reference baseline: ``input_text + answer_prefix``. No
    instruction header, no context/question split: the whole original input is
    handed to the model unchanged.
    """
    return input_text + answer_prefix


def needlepath_prompt(
    selected_context: str,
    question: str,
    answer_prefix: str,
    *,
    fallback_used: bool,
    input_text: str,
) -> str:
    """Selection arm: scaffolded selected-context template.

    On a fallback the reference degraded to the bare baseline
    (``input_text + answer_prefix``); otherwise it wraps the selected fragments
    in the instruction header + ``SELECTED OFFICIAL RULER CONTEXT:`` label +
    ``{question}\\n{answer_prefix}`` tail. Both branches are reproduced exactly.
    """
    if fallback_used:
        return baseline_prompt(input_text, answer_prefix)
    return (
        "Use the selected official RULER context fragments below to answer the question. "
        "Return only the requested continuation.\n\n"
        f"SELECTED OFFICIAL RULER CONTEXT:\n{selected_context}\n\n"
        f"{question}\n{answer_prefix}"
    )


def compression_prompt(compressed_context: str, question: str, answer_prefix: str) -> str:
    """Compression arm: ``{compressed_context}\\n\\n{question}\\n{answer_prefix}``.

    Byte-faithful to the reference compression template: the compressed context,
    a blank line, the split question, and the answer prefix, with no instruction
    header.
    """
    return f"{compressed_context}\n\n{question}\n{answer_prefix}"
