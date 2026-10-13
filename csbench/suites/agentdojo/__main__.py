"""CLI: ``python -m csbench.suites.agentdojo <command>``.

    init-config   write a run configuration from the installed AgentDojo suite
    run           execute every repeat and arm under a run root (paid model calls)
    summarize     produce the gated summary for a run root (no model calls)
    validate      validate a run configuration or a run manifest file

``run`` requires the ``agentdojo`` extra, the model key in ``GEMINI_API_KEY``
(or ``NM_GEMINI_API_KEY``) and the selector key in ``NEEDLEPATH_API_KEY``. The
other commands run offline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .config import (
    ConfigError,
    DatasetIdentity,
    Environment,
    ManifestError,
    RunConfig,
    RunManifest,
    SelectorIdentity,
    ShardPlan,
    validate_config,
    validate_manifest,
)
from .runner import Runner
from .summarize import SummaryGateError, summarize_run


def _numeric_suffix(task_id: str) -> int:
    try:
        return int(task_id.rsplit("_", 1)[-1])
    except ValueError:
        return 1 << 30


def suite_task_ids(benchmark_version: str, suite: str) -> tuple[list[str], list[str]]:
    """User and injection task ids from the installed AgentDojo package, in
    numeric order (ids are not contiguous in every suite version)."""
    from agentdojo.task_suite.load_suites import get_suite

    s = get_suite(benchmark_version, suite)
    return (
        sorted(s.user_tasks, key=_numeric_suffix),
        sorted(s.injection_tasks, key=_numeric_suffix),
    )


def cmd_init_config(args: argparse.Namespace) -> int:
    user_tasks, injection_tasks = suite_task_ids(args.benchmark_version, args.suite)
    config = RunConfig(
        run_id=args.run_id,
        dataset=DatasetIdentity(
            benchmark_version=args.benchmark_version,
            suite=args.suite,
            attack=args.attack,
            user_task_ids=user_tasks,
            injection_task_ids=injection_tasks,
        ),
        selector=SelectorIdentity(base_url=args.selector_url, snapshot=args.selector_snapshot),
        shards=ShardPlan.build(user_tasks, shard_size=args.shard_size, concurrent_shards=args.concurrent_shards),
        repeats=args.repeats,
    )
    validate_config(config)
    config.save(args.out)
    print(f"wrote {args.out}: {len(user_tasks)} user tasks x {len(injection_tasks)} injection tasks "
          f"= {config.dataset.expected_rows} rows per arm; {len(config.shards.shards)} shards")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    config = RunConfig.load(args.config)
    prompt_template: Optional[str] = None
    try:
        from agentdojo.agent_pipeline.agent_pipeline import load_system_message

        prompt_template = load_system_message(None)
    except Exception:
        prompt_template = None
    env = Environment.probe(lockfile=args.lockfile, prompt_template=prompt_template)
    runner = Runner(config, args.out, environment=env)
    manifest = runner.run()
    accepted = len(manifest.accepted)
    print(f"run {manifest.run_id}: {accepted} accepted arm(s), {len(manifest.attempts)} attempt(s)")
    return 0


def cmd_summarize(args: argparse.Namespace) -> int:
    try:
        summary = summarize_run(args.run_root, allow_directional=args.allow_directional)
    except SummaryGateError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({k: summary[k] for k in ("run_id", "evidence_class", "statement")}, indent=2))
    print(f"wrote {Path(args.run_root) / 'summary'}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    with Path(args.path).open(encoding="utf-8") as fh:
        d = json.load(fh)
    try:
        if "schema" in d and str(d["schema"]).endswith("run_manifest/1"):
            validate_manifest(d)
            RunManifest.from_wire(d)
            print("manifest valid")
        else:
            validate_config(RunConfig.from_wire(d))
            print("config valid")
    except (ConfigError, ManifestError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="csbench.suites.agentdojo", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init-config", help="write a run configuration from the installed AgentDojo suite")
    p.add_argument("--run-id", required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--benchmark-version", default="v1.2.2")
    p.add_argument("--suite", default="workspace")
    p.add_argument("--attack", default="important_instructions")
    p.add_argument("--selector-url", required=True, help="hosted selector base URL")
    p.add_argument("--selector-snapshot", required=True, help="selector source commit or image identity")
    p.add_argument("--shard-size", type=int, default=5)
    p.add_argument("--concurrent-shards", type=int, default=2)
    p.add_argument("--repeats", type=int, default=5)
    p.set_defaults(func=cmd_init_config)

    p = sub.add_parser("run", help="execute the configured repeats and arms (paid model calls)")
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path, help="run root")
    p.add_argument("--lockfile", type=Path, default=None, help="dependency lockfile to hash into the manifest")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("summarize", help="gated summary of a run root")
    p.add_argument("--run-root", required=True, type=Path)
    p.add_argument("--allow-directional", action="store_true",
                   help="permit a directional-only summary of >= 2 complete repeats")
    p.set_defaults(func=cmd_summarize)

    p = sub.add_parser("validate", help="validate a run configuration or manifest JSON file")
    p.add_argument("path", type=Path)
    p.set_defaults(func=cmd_validate)
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
