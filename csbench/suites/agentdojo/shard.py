"""Per-shard entry point: run one arm's slice of user tasks in one process.

The runner launches ``python -m csbench.suites.agentdojo.shard`` once per
shard, two shards concurrently per wave. Each process:

1. loads and validates the run configuration;
2. builds the LLM element through the API-key provider;
3. builds AgentDojo's standard pipeline around it (no defense);
4. installs the neutral preflight (both arms);
5. for the selector arm, inserts the bridge after the tool executor and opens
   this shard's telemetry file;
6. runs AgentDojo's scored loop over the shard's user tasks and every
   injection task, writing results under the arm directory.

Requires the ``agentdojo`` extra. Nothing here is imported by the offline tests
except ``build_pipeline``'s helpers, which are exercised with fakes.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

from . import ARM_NEEDLEPATH
from .bridge import (
    BridgeIdentity,
    HttpTextTransport,
    NeedlepathToolResultBridge,
    insert_after_tool_executor,
)
from .config import RunConfig, validate_config
from .preflight import install_preflight
from .provider import build_llm_element
from .telemetry import TelemetryWriter


def telemetry_path(arm_dir: Path, shard_index: int) -> Path:
    return Path(arm_dir) / f"needlepath_metrics_shard_{int(shard_index)}.jsonl"


def shard_done_path(arm_dir: Path, shard_index: int) -> Path:
    return Path(arm_dir) / f"shard_{int(shard_index)}.done.json"


def build_pipeline(config: RunConfig, arm: str, *, llm: Any, bridge: Optional[Any]) -> Any:
    """AgentDojo's default pipeline around ``llm``; the bridge goes after the
    tool executor on the selector arm. The pipeline name is the LLM's name in
    both arms, so the attack text and result layout are identical."""
    from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, PipelineConfig

    pipeline = AgentPipeline.from_config(
        PipelineConfig(llm=llm, model_id=None, defense=None, system_message_name=None, system_message=None)
    )
    if arm == ARM_NEEDLEPATH:
        if bridge is None:
            raise ValueError("the selector arm requires a bridge")
        insert_after_tool_executor(pipeline, bridge)
    return pipeline


def run_shard(
    config: RunConfig,
    *,
    repeat: int,
    arm: str,
    shard_index: int,
    arm_dir: Path,
    attempt_id: Optional[str] = None,
    environ: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    import agentdojo.benchmark as agentdojo_benchmark
    from agentdojo.attacks.attack_registry import load_attack
    from agentdojo.benchmark import benchmark_suite_with_injections
    from agentdojo.logging import OutputLogger
    from agentdojo.task_suite.load_suites import get_suite

    validate_config(config)
    env = os.environ if environ is None else environ
    arm_dir = Path(arm_dir)
    arm_dir.mkdir(parents=True, exist_ok=True)
    user_tasks = list(config.shards.shards[int(shard_index)])

    install_preflight(agentdojo_benchmark)
    llm = build_llm_element(config.model, environ=env)

    bridge = None
    if arm == ARM_NEEDLEPATH:
        bridge = NeedlepathToolResultBridge(
            identity=BridgeIdentity(
                suite=config.dataset.suite, attack=config.dataset.attack, repeat=int(repeat), arm=arm
            ),
            selector=config.selector,
            budget=config.budget,
            transport=HttpTextTransport(config.selector, environ=env),
            telemetry=TelemetryWriter(telemetry_path(arm_dir, shard_index)),
        )
    pipeline = build_pipeline(config, arm, llm=llm, bridge=bridge)

    suite = get_suite(config.dataset.benchmark_version, config.dataset.suite)
    attack = load_attack(config.dataset.attack, suite, pipeline)

    with OutputLogger(str(arm_dir)):
        results = benchmark_suite_with_injections(
            pipeline,
            suite,
            attack,
            logdir=arm_dir,
            force_rerun=False,
            user_tasks=user_tasks,
            injection_tasks=list(config.dataset.injection_task_ids),
            benchmark_version=config.dataset.benchmark_version,
        )

    done = {
        "shard_index": int(shard_index),
        "repeat": int(repeat),
        "arm": arm,
        "attempt_id": attempt_id,
        "user_tasks": user_tasks,
        "scored_rows": len(results["utility_results"]),
        "pipeline_name": pipeline.name,
    }
    shard_done_path(arm_dir, shard_index).write_text(json.dumps(done, indent=2, sort_keys=True) + "\n")
    return done


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="csbench.suites.agentdojo.shard")
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--run-root", required=True, type=Path)
    ap.add_argument("--repeat", required=True, type=int)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--shard", required=True, type=int)
    ap.add_argument("--attempt-id", default=None)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    config = RunConfig.load(args.config)
    arm_dir = config.arm_dir(args.run_root, args.repeat, args.arm)
    done = run_shard(
        config, repeat=args.repeat, arm=args.arm, shard_index=args.shard, arm_dir=arm_dir, attempt_id=args.attempt_id
    )
    print(json.dumps(done, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
