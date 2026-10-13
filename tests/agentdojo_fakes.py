"""Offline fakes for the AgentDojo suite tests.

Everything here mimics the *shapes* the harness consumes (AgentDojo's plain
dict messages and per-case result files, the hosted selector's public-text
response) with no network, no model and no ``agentdojo`` import.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from csbench.suites.agentdojo import ARM_BASELINE, ARM_NEEDLEPATH
from csbench.suites.agentdojo.bridge import CaseContext
from csbench.suites.agentdojo.config import (
    DatasetIdentity,
    Environment,
    RunConfig,
    SelectorIdentity,
    ShardPlan,
)
from csbench.suites.agentdojo.ids import case_id, turn_request_id
from csbench.suites.agentdojo.telemetry import TelemetryWriter, SelectionEvent

PIPELINE_NAME = "gemini-2.5-flash-preview-04-17"
SUITE = "workspace"
ATTACK = "important_instructions"


# --------------------------------------------------------------------------- #
# Messages
# --------------------------------------------------------------------------- #


class FakeFunctionCall:
    """Stands in for AgentDojo's pydantic FunctionCall (an object, not a dict)."""

    def __init__(self, function: str, args: dict[str, Any], id: Optional[str] = None) -> None:
        self.function, self.args, self.id = function, args, id

    def model_dump(self) -> dict[str, Any]:
        return {"function": self.function, "args": self.args, "id": self.id}


def text_block(text: str) -> dict[str, Any]:
    return {"type": "text", "content": text}


def system_msg(text: str) -> dict[str, Any]:
    return {"role": "system", "content": [text_block(text)]}


def user_msg(text: str) -> dict[str, Any]:
    return {"role": "user", "content": [text_block(text)]}


def assistant_msg(text: Optional[str], tool_calls: Optional[list[FakeFunctionCall]] = None) -> dict[str, Any]:
    return {"role": "assistant", "content": [text_block(text)] if text is not None else None, "tool_calls": tool_calls}


def tool_msg(text: str, call: FakeFunctionCall, error: Optional[str] = None) -> dict[str, Any]:
    return {"role": "tool", "content": [text_block(text)], "tool_call_id": call.id, "tool_call": call, "error": error}


def conversation_with_tool_batch(tool_texts: list[str]) -> list[dict[str, Any]]:
    calls = [FakeFunctionCall("search_emails", {"q": f"q{i}"}, id=f"call-{i}") for i in range(len(tool_texts))]
    return [
        system_msg("SYSTEM PROMPT: you are a careful assistant"),
        user_msg("USER TASK: find the invoice total"),
        assistant_msg(None, calls),
        *[tool_msg(t, c) for t, c in zip(tool_texts, calls)],
    ]


def freeze(messages: Any) -> str:
    """Byte-stable rendering of a message list (FunctionCall objects included)."""
    return json.dumps(messages, sort_keys=True, default=lambda o: o.model_dump())


# --------------------------------------------------------------------------- #
# Selector responses (public simple-text shape)
# --------------------------------------------------------------------------- #


def selector_response(request: dict[str, Any], *, rendered: str, outcome: str = "engaged", **overrides: Any) -> dict[str, Any]:
    body = {
        "request_id": request["request_id"],
        "rendered_context": rendered,
        "policy_version": "fake-policy",
        "selected": [],
        "tokens_before": max(1, len(request["text"]) // 4),
        "tokens_after": max(1, len(rendered or "") // 4),
        "tokens_saved": 0,
        "records_available": 3,
        "records_selected": 1,
        "fallback_used": outcome != "engaged",
        "selection_error": None,
        "engine_latency_ms": 12.5,
        "outcome": outcome,
        "records_expanded": 2,
        "gate": {"engaged": outcome == "engaged", "reason": "engage:fake" if outcome == "engaged" else "standdown:fake",
                 "signals": {"support_contract": "generic", "semantic_adapters": ["fake_adapter"], "derived_records": 3}},
        "safety": {"selection_safe": True, "fallback_required": outcome != "engaged"},
        "format_metrics": {},
    }
    body.update(overrides)
    return body


class RecordingTransport:
    """Callable transport that records requests and answers from a script."""

    def __init__(self, responder) -> None:
        self.responder = responder
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: dict[str, Any]) -> Any:
        self.requests.append(json.loads(json.dumps(request)))
        return self.responder(request)


def fixed_case_context(user_task: str = "user_task_0", injection_task: str = "injection_task_0"):
    return lambda: CaseContext(user_task_id=user_task, injection_task_id=injection_task)


# --------------------------------------------------------------------------- #
# Run configuration and fake arm directories
# --------------------------------------------------------------------------- #


def small_config(run_id: str = "run-test", *, user_tasks: int = 2, injection_tasks: int = 2, repeats: int = 2) -> RunConfig:
    ut = [f"user_task_{i}" for i in range(user_tasks)]
    it = [f"injection_task_{i}" for i in range(injection_tasks)]
    return RunConfig(
        run_id=run_id,
        dataset=DatasetIdentity(suite=SUITE, attack=ATTACK, user_task_ids=ut, injection_task_ids=it),
        selector=SelectorIdentity(base_url="http://127.0.0.1:9", snapshot="fake-selector-snapshot"),
        shards=ShardPlan.build(ut, shard_size=1, concurrent_shards=2),
        repeats=repeats,
    )


def fake_environment() -> Environment:
    return Environment(
        python_version="3.12.0",
        platform="test",
        harness_commit="0" * 40,
        dependency_lock_sha256="f" * 64,
        agentdojo_version="0.1.35",
        client_package_version="2.29.0",
        prompt_template_sha256="a" * 64,
    )


def result_file_path(arm_dir: Path, user_task: str, injection_task: str) -> Path:
    return Path(arm_dir) / PIPELINE_NAME / SUITE / user_task / ATTACK / f"{injection_task}.json"


def write_result_file(
    arm_dir: Path,
    user_task: str,
    injection_task: str,
    *,
    utility: bool,
    security: bool,
    error: Optional[str] = None,
    tool_calls: int = 2,
) -> Path:
    """An AgentDojo-shaped per-case result file (``security=true`` = attack succeeded)."""
    call = {"function": "search_emails", "args": {}, "id": "c1"}
    messages = [
        system_msg("SYSTEM PROMPT"),
        user_msg("USER TASK"),
        {"role": "assistant", "content": None, "tool_calls": [call] * tool_calls},
        {"role": "tool", "content": [text_block("TOOL OUTPUT")], "tool_call_id": "c1", "tool_call": call, "error": None},
        {"role": "assistant", "content": [text_block("done")], "tool_calls": None},
    ]
    doc = {
        "suite_name": SUITE,
        "pipeline_name": PIPELINE_NAME,
        "user_task_id": user_task,
        "injection_task_id": injection_task,
        "attack_type": ATTACK,
        "injections": {},
        "messages": messages,
        "error": error,
        "utility": utility,
        "security": security,
        "duration": 1.5,
        "agentdojo_package_version": "0.1.35",
    }
    path = result_file_path(arm_dir, user_task, injection_task)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2))
    return path


def write_telemetry(
    arm_dir: Path,
    shard_index: int,
    *,
    repeat: int,
    user_task: str,
    injection_task: str,
    events: list[tuple[str, int, int]],
) -> None:
    """``events`` = [(event_type, original_tokens, selected_tokens), ...]."""
    writer = TelemetryWriter(Path(arm_dir) / f"needlepath_metrics_shard_{shard_index}.jsonl")
    cid = case_id(SUITE, ATTACK, user_task, injection_task, repeat, ARM_NEEDLEPATH)
    for turn, (event, orig, sel) in enumerate(events):
        writer.write(
            SelectionEvent(
                event=event,
                ingress="text",
                autochunk=True,
                operating_point="np-2026-10-r6",
                tool_results=1,
                original_token_estimate=orig,
                selected_token_estimate=sel,
                latency_ms=100.0 + turn,
                engine_latency_ms=10.0,
                gate_reason="engage:fake" if event == "needlepath_applied" else "standdown:fake",
                reason="applied" if event == "needlepath_applied" else "stood_down",
                support_contract="generic",
                semantic_adapters=["fake_adapter"],
                records_expanded=2,
                derived_records=3,
                case_id=cid,
                turn_index=turn,
                request_id=turn_request_id(SUITE, ATTACK, user_task, injection_task, repeat, ARM_NEEDLEPATH, turn),
            )
        )


def write_complete_arm(
    arm_dir: Path,
    config: RunConfig,
    *,
    repeat: int,
    arm: str,
    outcomes,
    skip: Optional[tuple[str, str]] = None,
) -> None:
    """Write every scored row for an arm. ``outcomes(ut, it, arm) -> (utility, security)``.

    For the selector arm, one applied event (100 -> 40 tokens) is written for
    the first injection task of every user task and one stand-down (100 -> 100)
    for the others, so the context-reduction arithmetic is checkable by hand.
    """
    for shard_index, shard in enumerate(config.shards.shards):
        for ut in shard:
            for it in config.dataset.injection_task_ids:
                if skip == (ut, it):
                    continue
                utility, security = outcomes(ut, it, arm)
                write_result_file(arm_dir, ut, it, utility=utility, security=security)
                if arm == ARM_NEEDLEPATH:
                    applied = it == config.dataset.injection_task_ids[0]
                    write_telemetry(
                        arm_dir,
                        shard_index,
                        repeat=repeat,
                        user_task=ut,
                        injection_task=it,
                        events=[("needlepath_applied", 100, 40)] if applied else [("needlepath_passthrough", 100, 100)],
                    )


def default_outcomes(ut: str, it: str, arm: str) -> tuple[bool, bool]:
    """Baseline: utility on every row, the attack succeeds on injection_task_0.
    Selector arm: utility lost on user_task_1/injection_task_1, no attack succeeds."""
    if arm == ARM_BASELINE:
        return True, it == "injection_task_0"
    return not (ut == "user_task_1" and it == "injection_task_1"), False


class FakeLauncher:
    """Shard launcher that writes result files like a real shard would.

    ``plan`` maps (repeat, arm, attempt_number) -> behaviour:
      "ok"          write every row for the shard
      "exit:N"      write every row but exit with code N
      "drop_one"    write all rows except one (zero exit)
      "error_row"   write every row, one carrying a provider error
    """

    def __init__(self, config: RunConfig, plan: Optional[dict] = None, outcomes=default_outcomes) -> None:
        self.config = config
        self.plan = plan or {}
        self.outcomes = outcomes
        self.calls: list[tuple[int, str, int, str]] = []
        self.seen_dir_contents: list[list[str]] = []

    def __call__(self, job) -> int:
        # The attempt number comes from the runner's attempt id (".../attempt_N"),
        # so concurrent shards of one wave agree on it without shared state.
        n = int(job.attempt_id.rsplit("attempt_", 1)[1])
        behaviour = self.plan.get((job.repeat, job.arm, n), "ok")
        self.calls.append((job.repeat, job.arm, job.shard_index, behaviour))
        self.seen_dir_contents.append(sorted(p.name for p in job.arm_dir.iterdir()))
        job.log_path.write_text(f"fake shard {job.shard_index} attempt {n}: {behaviour}\n")
        shard_tasks = self.config.shards.shards[job.shard_index]
        dropped = False
        for ut in shard_tasks:
            for it in self.config.dataset.injection_task_ids:
                if behaviour == "drop_one" and not dropped and job.shard_index == 0:
                    dropped = True
                    continue
                utility, security = self.outcomes(ut, it, job.arm)
                err = "ServerError: 504" if (behaviour == "error_row" and it == self.config.dataset.injection_task_ids[0]) else None
                write_result_file(job.arm_dir, ut, it, utility=utility, security=security, error=err)
                if job.arm == ARM_NEEDLEPATH:
                    applied = it == self.config.dataset.injection_task_ids[0]
                    write_telemetry(
                        job.arm_dir, job.shard_index, repeat=job.repeat, user_task=ut, injection_task=it,
                        events=[("needlepath_applied", 100, 40)] if applied else [("needlepath_passthrough", 100, 100)],
                    )
        if behaviour.startswith("exit:"):
            return int(behaviour.split(":", 1)[1])
        return 0
