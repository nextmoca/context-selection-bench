"""Matched-pair runner: shards, waves, whole-arm validity and whole-arm retry.

An arm is a complete execution of every scored case for one repeat. It is
accepted only when every shard exited 0, exactly the expected rows exist with
no duplicate or missing ids, and (for the selector arm) selection telemetry is
present. Anything else invalidates the whole arm: the directory is archived to
a timestamped ``<arm>.failed_attempt_<stamp>`` sibling and the retry starts
from an empty directory. Up to ``max_arm_attempts`` attempts are made. Every
attempt, valid or not, is recorded in the manifest with its id, exit codes,
row count and invalidation reason.

Run order: ``repeat 1: baseline -> selector; repeat 2: ...``. After both arms of
a repeat are accepted, the pair gate checks they cover the same case set.

The shard launcher is injectable so the runner's control flow is tested
offline with a fake that writes result files; the default launcher runs
``python -m csbench.suites.agentdojo.shard`` as a subprocess with its console
captured to ``console-shard-<n>.log``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from . import ARM_BASELINE, ARM_NEEDLEPATH
from .collect import collect_arm, validate_arm, validate_pair
from .config import (
    ArmAttempt,
    Environment,
    RunConfig,
    RunManifest,
    utc_now,
    validate_config,
    validate_manifest,
)
from .ids import expected_case_ids

MANIFEST_NAME = "manifest.json"
CONFIG_NAME = "run_config.json"
TIMING_NAME = "timing.json"


class ArmFailed(RuntimeError):
    """Every allowed attempt of an arm was invalid."""


@dataclass(frozen=True)
class ShardJob:
    config_path: Path
    run_root: Path
    repeat: int
    arm: str
    shard_index: int
    arm_dir: Path

    @property
    def log_path(self) -> Path:
        return self.arm_dir / f"console-shard-{self.shard_index}.log"


ShardLauncher = Callable[[ShardJob], int]


def default_shard_launcher(job: ShardJob) -> int:
    cmd = [
        sys.executable,
        "-m",
        "csbench.suites.agentdojo.shard",
        "--config",
        str(job.config_path),
        "--run-root",
        str(job.run_root),
        "--repeat",
        str(job.repeat),
        "--arm",
        job.arm,
        "--shard",
        str(job.shard_index),
    ]
    with job.log_path.open("ab") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        return proc.wait()


def archive_stamp(now: Optional[datetime] = None) -> str:
    now = now or datetime.now(timezone.utc)
    return now.strftime("%Y%m%dT%H%M%SZ")


def archive_failed_attempt(arm_dir: Path, stamp: str) -> Path:
    """Move an invalid arm directory to a sibling the collector never reads."""
    arm_dir = Path(arm_dir)
    target = arm_dir.with_name(f"{arm_dir.name}.failed_attempt_{stamp}")
    n = 1
    while target.exists():
        n += 1
        target = arm_dir.with_name(f"{arm_dir.name}.failed_attempt_{stamp}_{n}")
    arm_dir.rename(target)
    return target


class Runner:
    def __init__(
        self,
        config: RunConfig,
        run_root: Path,
        *,
        launcher: ShardLauncher = default_shard_launcher,
        environment: Optional[Environment] = None,
        stamp: Callable[[], str] = archive_stamp,
    ) -> None:
        validate_config(config)
        self.config = config
        self.run_root = Path(run_root)
        self.launcher = launcher
        self.stamp = stamp
        self.environment = environment or Environment.probe(prompt_template=None)
        self.manifest: Optional[RunManifest] = None

    # -- paths ----------------------------------------------------------------- #

    @property
    def manifest_path(self) -> Path:
        return self.run_root / MANIFEST_NAME

    @property
    def config_path(self) -> Path:
        return self.run_root / CONFIG_NAME

    # -- lifecycle ------------------------------------------------------------- #

    def start(self) -> RunManifest:
        """Write config and manifest; the manifest is validated before any shard runs."""
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.config.save(self.config_path)
        if self.manifest_path.exists():
            manifest = RunManifest.load(self.manifest_path)
            if manifest.run_id != self.config.run_id:
                raise RuntimeError(f"run root already holds run {manifest.run_id!r}, not {self.config.run_id!r}")
        else:
            manifest = RunManifest.create(self.config, self.environment)
            manifest.started_at = utc_now()
        validate_manifest(manifest.to_wire())
        manifest.save(self.manifest_path)
        self.manifest = manifest
        return manifest

    def run(self) -> RunManifest:
        manifest = self.start()
        arms = [a for a in (ARM_BASELINE, ARM_NEEDLEPATH) if a in self.config.arms]
        for repeat in range(1, self.config.repeats + 1):
            rows_by_arm = {}
            for arm in arms:
                if manifest.accepted_attempt(repeat, arm) is None:
                    self.run_arm(repeat, arm)
                arm_dir = self.config.arm_dir(self.run_root, repeat, arm)
                rows_by_arm[arm], _ = collect_arm(arm_dir, repeat=repeat, arm=arm)
            if len(rows_by_arm) == 2:
                validate_pair(rows_by_arm[ARM_BASELINE], rows_by_arm[ARM_NEEDLEPATH])
        manifest.ended_at = utc_now()
        manifest.save(self.manifest_path)
        return manifest

    # -- one arm, with whole-arm retry ----------------------------------------- #

    def run_arm(self, repeat: int, arm: str) -> ArmAttempt:
        assert self.manifest is not None, "call start() first"
        manifest = self.manifest
        cfg = self.config
        arm_dir = cfg.arm_dir(self.run_root, repeat, arm)
        expected = expected_case_ids(
            cfg.dataset.suite, cfg.dataset.attack, cfg.dataset.user_task_ids, cfg.dataset.injection_task_ids, repeat, arm
        )
        prior = sum(1 for a in manifest.attempts if a.repeat == repeat and a.arm == arm)

        for n in range(prior + 1, prior + cfg.max_arm_attempts + 1):
            if arm_dir.exists():
                # A leftover directory (interrupted process, manual run) is never
                # reused: archive it so the attempt starts empty.
                archive_failed_attempt(arm_dir, self.stamp())
            arm_dir.mkdir(parents=True, exist_ok=False)

            attempt = ArmAttempt(
                repeat=repeat,
                arm=arm,
                attempt_id=f"{cfg.run_id}/repeat_{repeat}/{arm}/attempt_{n}",
                parent_run_id=cfg.run_id,
                started_at=utc_now(),
                worker_count=cfg.shards.concurrent_shards,
            )
            manifest.attempts.append(attempt)
            manifest.save(self.manifest_path)

            attempt.shard_exit_codes = self._run_waves(repeat, arm, arm_dir)
            attempt.ended_at = utc_now()

            rows, events = collect_arm(arm_dir, repeat=repeat, arm=arm)
            validation = validate_arm(
                rows,
                expected_ids=expected,
                shard_exit_codes=attempt.shard_exit_codes,
                require_telemetry=cfg.uses_selector(arm),
                events=events,
            )
            attempt.row_count = validation.row_count
            attempt.valid = validation.valid
            attempt.invalidation_reason = validation.reason
            (arm_dir / TIMING_NAME).write_text(
                json.dumps(
                    {
                        "attempt_id": attempt.attempt_id,
                        "started_at": attempt.started_at,
                        "ended_at": attempt.ended_at,
                        "shard_exit_codes": attempt.shard_exit_codes,
                        "validation": validation.to_wire(),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            if validation.valid:
                manifest.accepted[RunManifest.accepted_key(repeat, arm)] = attempt.attempt_id
                manifest.save(self.manifest_path)
                return attempt

            archived = archive_failed_attempt(arm_dir, self.stamp())
            attempt.archived_to = archived.name
            manifest.save(self.manifest_path)

        raise ArmFailed(
            f"repeat {repeat} arm {arm}: {cfg.max_arm_attempts} attempt(s) invalid; "
            f"last reason: {manifest.attempts[-1].invalidation_reason}"
        )

    def _run_waves(self, repeat: int, arm: str, arm_dir: Path) -> dict[str, int]:
        exit_codes: dict[str, int] = {}
        for wave in self.config.shards.waves():
            jobs = [
                ShardJob(
                    config_path=self.config_path,
                    run_root=self.run_root,
                    repeat=repeat,
                    arm=arm,
                    shard_index=i,
                    arm_dir=arm_dir,
                )
                for i in wave
            ]
            with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
                for job, code in zip(jobs, pool.map(self._launch, jobs)):
                    exit_codes[str(job.shard_index)] = int(code)
        return exit_codes

    def _launch(self, job: ShardJob) -> int:
        try:
            return int(self.launcher(job))
        except Exception as exc:  # a launcher crash is a failed shard, not a crashed run
            job.log_path.parent.mkdir(parents=True, exist_ok=True)
            with job.log_path.open("a") as log:
                log.write(f"launcher raised {type(exc).__name__}: {exc}\n")
            return 1
