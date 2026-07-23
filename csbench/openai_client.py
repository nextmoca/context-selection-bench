"""Provider-clean OpenAI chat client for benchmark runs.

Reads the API key from the environment only. The ``openai`` package is an
OPTIONAL dependency: it is imported lazily inside ``build_openai_client`` so
this module imports cleanly even when the package is not installed.

Environment variables:
    OPENAI_API_KEY  -- API key (the only accepted source)

The suite runners use two entry points:

* ``call_openai`` for answer generation over a prompt.
* ``judge_openai`` for LLM-as-judge scoring of a prediction against a known
  ground truth on a 1-5 scale.
"""

from __future__ import annotations

import os
import re
from typing import Any

JUDGE_PROMPT = """You are evaluating a memory-based question answering system.

Given a question, the ground truth answer, and the system's predicted answer,
score the prediction on a scale of 1-5:

5 = Perfect: The predicted answer is semantically equivalent to the ground truth
4 = Mostly correct: The prediction captures the main point with minor differences
3 = Partially correct: The prediction has some correct information but is incomplete or has errors
2 = Mostly incorrect: The prediction has minimal relevant information or significant errors
1 = Completely wrong: The prediction is irrelevant or contradicts the ground truth

Question: {question}

Ground Truth Answer: {ground_truth}

Predicted Answer: {prediction}

First, provide a brief reasoning (1-2 sentences), then give your score.

Format your response EXACTLY as:
Reasoning: <your reasoning>
Score: <number 1-5>"""


def build_openai_client() -> Any:
    """Build an OpenAI client, importing ``openai`` lazily.

    The API key is read from ``OPENAI_API_KEY`` in the environment and from no
    other source. The import happens here (not at module load) so that importing
    this module does not require the optional ``openai`` dependency.
    """
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required.")
    return OpenAI(api_key=api_key)


def call_openai(client: Any, *, model: str, prompt: str, max_tokens: int) -> str:
    """Send a single-prompt chat completion and return the text content.

    Temperature is pinned to 0.0 for reproducibility.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return str(response.choices[0].message.content or "")


def judge_openai(
    client: Any,
    *,
    model: str,
    question: str,
    ground_truth: str,
    prediction: str,
) -> tuple[float, str]:
    """Score ``prediction`` against ``ground_truth`` on a 1-5 scale.

    Returns ``(score, reasoning)``. The score is clamped to ``[1.0, 5.0]`` and
    defaults to ``0.0`` when the judge response omits a parseable score.
    """
    prompt = JUDGE_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        prediction=prediction,
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=200,
    )
    text = str(response.choices[0].message.content or "")
    score = 0.0
    reasoning = text.strip()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("reasoning:"):
            reasoning = stripped[len("reasoning:") :].strip()
        if stripped.lower().startswith("score:"):
            match = re.search(r"(\d+(?:\.\d+)?)", stripped)
            if match:
                score = max(1.0, min(5.0, float(match.group(1))))
    return score, reasoning
