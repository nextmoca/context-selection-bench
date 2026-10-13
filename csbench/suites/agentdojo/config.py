"""Typed run configuration and the ``RunManifest`` schema.

Everything a run depends on is declared here, in plain dataclasses that
round-trip through JSON, so a result directory is self-describing and a reader
can see exactly what produced it. ``validate_manifest`` is called by the runner
before the first model call and by the summarizer before any aggregation; both
refuse to proceed on a manifest that does not describe the run completely.

No credential is ever part of a configuration or a manifest. API keys are
named by environment variable and read at call time only.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from . import ARMS, ARM_BASELINE, ARM_NEEDLEPATH

MANIFEST_SCHEMA = "csbench.agentdojo.run_manifest/1"
CONFIG_SCHEMA = "csbench.agentdojo.run_config/1"

DEFAULT_OPERATING_POINT = "np-2026-10-r6"
DEFAULT_AGENTDOJO_MODEL = "gemini-2.5-flash-preview-04-17"
DEFAULT_PROVIDER_MODEL = "gemini-2.5-flash"
DEFAULT_SELECTOR_KEY_ENV = ("NEEDLEPATH_API_KEY",)
DEFAULT_MODEL_KEY_ENV = ("GEMINI_API_KEY", "NM_GEMINI_API_KEY")

# Protocol invariants. These are not tunables: a configuration outside them is
# a different protocol and is refused.
MAX_CONCURRENT_SHARDS = 2
MAX_ARM_ATTEMPTS = 3
MAX_MODEL_REQUEST_TIMEOUT_S = 60.0


class ConfigError(ValueError):
    """A run configuration is incomplete or inconsistent."""


class ManifestError(ValueError):
    """A run manifest does not describe its run completely."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass
class DatasetIdentity:
    """Which AgentDojo population is scored."""

    benchmark_version: str = "v1.2.2"
    suite: str = "workspace"
    attack: str = "important_instructions"
    user_task_ids: list[str] = field(default_factory=list)
    injection_task_ids: list[str] = field(default_factory=list)

    @property
    def expected_rows(self) -> int:
        return len(self.user_task_ids) * len(self.injection_task_ids)


@dataclass
class ModelIdentity:
    """Both the identifier AgentDojo accepts and the model actually requested."""

    provider: str = "google"
    agentdojo_model: str = DEFAULT_AGENTDOJO_MODEL
    provider_model: str = DEFAULT_PROVIDER_MODEL
    client_package: str = "google-genai"
    temperature: float = 0.0
    max_output_tokens: int = 65535
    request_timeout_s: float = 60.0
    sdk_attempts: int = 1
    api_key_env: list[str] = field(default_factory=lambda: list(DEFAULT_MODEL_KEY_ENV))


@dataclass
class SelectorIdentity:
    """The hosted selector the R6 arm submits to. ``snapshot`` names the
    selector source or image the run was taken against (recorded, not derived)."""

    base_url: Optional[str] = None
    endpoint_path: str = "/v1/context/select"
    operating_point: str = DEFAULT_OPERATING_POINT
    snapshot: Optional[str] = None
    entry_point: str = "text"
    support_contract: Optional[str] = None
    adapter_route: Optional[str] = None
    timeout_s: float = 20.0
    api_key_env: list[str] = field(default_factory=lambda: list(DEFAULT_SELECTOR_KEY_ENV))


@dataclass
class BudgetPolicy:
    max_context_tokens: int = 1200
    max_records: int = 8
    capacity_cap_tokens: int = 120000
    mode: str = "fixed"
    autochunk: bool = True
    fallback_policy: str = "fail_open"


@dataclass
class ShardPlan:
    """User tasks partitioned into shards; ``concurrent_shards`` run per wave."""

    shards: list[list[str]] = field(default_factory=list)
    concurrent_shards: int = 2

    @classmethod
    def build(cls, user_task_ids: list[str], shard_size: int = 5, concurrent_shards: int = 2) -> "ShardPlan":
        if shard_size < 1:
            raise ConfigError("shard_size must be >= 1")
        shards = [list(user_task_ids[i : i + shard_size]) for i in range(0, len(user_task_ids), shard_size)]
        return cls(shards=shards, concurrent_shards=concurrent_shards)

    def waves(self) -> list[list[int]]:
        n = max(1, int(self.concurrent_shards))
        idx = list(range(len(self.shards)))
        return [idx[i : i + n] for i in range(0, len(idx), n)]


@dataclass
class RunConfig:
    run_id: str
    dataset: DatasetIdentity = field(default_factory=DatasetIdentity)
    model: ModelIdentity = field(default_factory=ModelIdentity)
    selector: SelectorIdentity = field(default_factory=SelectorIdentity)
    budget: BudgetPolicy = field(default_factory=BudgetPolicy)
    shards: ShardPlan = field(default_factory=ShardPlan)
    repeats: int = 5
    arms: list[str] = field(default_factory=lambda: list(ARMS))
    max_arm_attempts: int = 3
    region: Optional[str] = None
    schema: str = CONFIG_SCHEMA

    # ----- serialization -------------------------------------------------- #

    def to_wire(self) -> dict[str, Any]:
        d = asdict(self)
        d["dataset"]["expected_rows"] = self.dataset.expected_rows
        return d

    @classmethod
    def from_wire(cls, d: dict[str, Any]) -> "RunConfig":
        ds = dict(d.get("dataset", {}))
        ds.pop("expected_rows", None)
        return cls(
            run_id=d["run_id"],
            dataset=DatasetIdentity(**ds),
            model=ModelIdentity(**d.get("model", {})),
            selector=SelectorIdentity(**d.get("selector", {})),
            budget=BudgetPolicy(**d.get("budget", {})),
            shards=ShardPlan(**d.get("shards", {})),
            repeats=int(d.get("repeats", 5)),
            arms=list(d.get("arms", ARMS)),
            max_arm_attempts=int(d.get("max_arm_attempts", 3)),
            region=d.get("region"),
            schema=d.get("schema", CONFIG_SCHEMA),
        )

    @classmethod
    def load(cls, path: Path) -> "RunConfig":
        with Path(path).open(encoding="utf-8") as fh:
            return cls.from_wire(json.load(fh))

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_wire(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # ----- derived -------------------------------------------------------- #

    def arm_dir(self, run_root: Path, repeat: int, arm: str) -> Path:
        return Path(run_root) / f"repeat_{int(repeat)}" / arm

    def uses_selector(self, arm: str) -> bool:
        return arm == ARM_NEEDLEPATH


def validate_config(config: RunConfig) -> None:
    """Refuse a configuration that cannot produce a complete, matched run."""
    problems: list[str] = []
    ds = config.dataset
    if not ds.user_task_ids:
        problems.append("dataset.user_task_ids is empty")
    if not ds.injection_task_ids:
        problems.append("dataset.injection_task_ids is empty")
    if len(set(ds.user_task_ids)) != len(ds.user_task_ids):
        problems.append("dataset.user_task_ids has duplicates")
    if len(set(ds.injection_task_ids)) != len(ds.injection_task_ids):
        problems.append("dataset.injection_task_ids has duplicates")
    if not ds.suite or not ds.attack or not ds.benchmark_version:
        problems.append("dataset.suite, dataset.attack and dataset.benchmark_version are required")

    flat = [t for shard in config.shards.shards for t in shard]
    if sorted(flat) != sorted(ds.user_task_ids):
        problems.append("shards must cover every user task exactly once")
    if not 1 <= config.shards.concurrent_shards <= MAX_CONCURRENT_SHARDS:
        problems.append(f"shards.concurrent_shards must be in 1..{MAX_CONCURRENT_SHARDS} (the protocol's provider-pressure cap)")

    if config.repeats < 1:
        problems.append("repeats must be >= 1")
    if not 1 <= config.max_arm_attempts <= MAX_ARM_ATTEMPTS:
        problems.append(f"max_arm_attempts must be in 1..{MAX_ARM_ATTEMPTS}")
    unknown = [a for a in config.arms if a not in ARMS]
    if unknown:
        problems.append(f"unknown arms {unknown}; valid arms are {list(ARMS)}")
    if ARM_BASELINE not in config.arms:
        problems.append("the baseline arm is required for a paired run")

    m = config.model
    if not m.provider or not m.agentdojo_model or not m.provider_model or not m.client_package:
        problems.append("model.provider, model.agentdojo_model, model.provider_model, model.client_package are required")
    if not 0 < m.request_timeout_s <= MAX_MODEL_REQUEST_TIMEOUT_S:
        problems.append(f"model.request_timeout_s must be in (0, {MAX_MODEL_REQUEST_TIMEOUT_S}]")
    if m.sdk_attempts != 1:
        problems.append("model.sdk_attempts must be 1: the benchmark's own retry wrapper owns per-call retries")
    if not m.api_key_env:
        problems.append("model.api_key_env must name at least one environment variable")

    s = config.selector
    if config.uses_selector(ARM_NEEDLEPATH) and ARM_NEEDLEPATH in config.arms:
        if not s.base_url:
            problems.append("selector.base_url is required when the selector arm is configured")
        if not s.operating_point:
            problems.append("selector.operating_point is required")
        if not s.snapshot:
            problems.append("selector.snapshot (the selector source/image identity) is required")
        if s.entry_point != "text":
            problems.append("selector.entry_point must be 'text' (the public simple-text ingress)")
        if s.timeout_s <= 0:
            problems.append("selector.timeout_s must be > 0")

    b = config.budget
    if b.max_context_tokens < 1 or b.max_records < 1 or b.capacity_cap_tokens < b.max_context_tokens:
        problems.append("budget: max_context_tokens >= 1, max_records >= 1, capacity_cap_tokens >= max_context_tokens")
    if b.fallback_policy != "fail_open":
        problems.append("budget.fallback_policy must be 'fail_open'")
    if b.autochunk is not True:
        problems.append("budget.autochunk must be true for the public-text arm")

    if problems:
        raise ConfigError("invalid run configuration:\n  " + "\n  ".join(problems))


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #


@dataclass
class Environment:
    python_version: str
    platform: str
    harness_commit: Optional[str]
    dependency_lock_sha256: Optional[str]
    agentdojo_version: Optional[str]
    client_package_version: Optional[str]
    prompt_template_sha256: Optional[str]

    @classmethod
    def probe(
        cls,
        *,
        lockfile: Optional[Path] = None,
        prompt_template: Optional[str] = None,
        repo_root: Optional[Path] = None,
    ) -> "Environment":
        return cls(
            python_version=sys.version.split()[0],
            platform=platform.platform(),
            harness_commit=_git_head(repo_root),
            dependency_lock_sha256=_sha256_file(lockfile) if lockfile else None,
            agentdojo_version=_dist_version("agentdojo"),
            client_package_version=_dist_version("google-genai"),
            prompt_template_sha256=(
                hashlib.sha256(prompt_template.encode("utf-8")).hexdigest() if prompt_template is not None else None
            ),
        )


@dataclass
class ArmAttempt:
    """One whole-arm attempt. Invalid attempts stay in the manifest, archived."""

    repeat: int
    arm: str
    attempt_id: str
    parent_run_id: str
    started_at: str
    ended_at: Optional[str] = None
    worker_count: int = 2
    shard_exit_codes: dict[str, int] = field(default_factory=dict)
    row_count: Optional[int] = None
    valid: Optional[bool] = None
    invalidation_reason: Optional[str] = None
    archived_to: Optional[str] = None


@dataclass
class RunManifest:
    schema: str
    run_id: str
    created_at: str
    config: RunConfig
    environment: Environment
    attempts: list[ArmAttempt] = field(default_factory=list)
    accepted: dict[str, str] = field(default_factory=dict)  # "repeat_1/baseline" -> attempt_id
    started_at: Optional[str] = None
    ended_at: Optional[str] = None

    @classmethod
    def create(cls, config: RunConfig, environment: Environment) -> "RunManifest":
        return cls(
            schema=MANIFEST_SCHEMA,
            run_id=config.run_id,
            created_at=utc_now(),
            config=config,
            environment=environment,
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "config": self.config.to_wire(),
            "environment": asdict(self.environment),
            "attempts": [asdict(a) for a in self.attempts],
            "accepted": dict(self.accepted),
        }

    @classmethod
    def from_wire(cls, d: dict[str, Any]) -> "RunManifest":
        return cls(
            schema=d["schema"],
            run_id=d["run_id"],
            created_at=d["created_at"],
            config=RunConfig.from_wire(d["config"]),
            environment=Environment(**d["environment"]),
            attempts=[ArmAttempt(**a) for a in d.get("attempts", [])],
            accepted=dict(d.get("accepted", {})),
            started_at=d.get("started_at"),
            ended_at=d.get("ended_at"),
        )

    @classmethod
    def load(cls, path: Path) -> "RunManifest":
        with Path(path).open(encoding="utf-8") as fh:
            return cls.from_wire(json.load(fh))

    def save(self, path: Path) -> None:
        wire = self.to_wire()
        validate_manifest(wire)
        tmp = Path(str(path) + ".tmp")
        tmp.write_text(json.dumps(wire, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(path)

    @staticmethod
    def accepted_key(repeat: int, arm: str) -> str:
        return f"repeat_{int(repeat)}/{arm}"

    def accepted_attempt(self, repeat: int, arm: str) -> Optional[ArmAttempt]:
        aid = self.accepted.get(self.accepted_key(repeat, arm))
        if aid is None:
            return None
        for a in self.attempts:
            if a.attempt_id == aid:
                return a
        return None


# The protocol's "Required manifest fields" table, as (category, dotted path).
REQUIRED_MANIFEST_PATHS: tuple[tuple[str, str], ...] = (
    ("dataset identity", "environment.agentdojo_version"),
    ("dataset identity", "config.dataset.benchmark_version"),
    ("dataset identity", "config.dataset.suite"),
    ("dataset identity", "config.dataset.attack"),
    ("dataset identity", "config.dataset.user_task_ids"),
    ("dataset identity", "config.dataset.injection_task_ids"),
    ("dataset identity", "config.dataset.expected_rows"),
    ("selector identity", "config.selector.snapshot"),
    ("selector identity", "config.selector.operating_point"),
    ("selector identity", "config.selector.entry_point"),
    ("model identity", "config.model.provider"),
    ("model identity", "config.model.provider_model"),
    ("model identity", "config.model.agentdojo_model"),
    ("model identity", "environment.client_package_version"),
    ("model identity", "config.model.temperature"),
    ("model identity", "config.model.max_output_tokens"),
    ("model identity", "environment.prompt_template_sha256"),
    ("budget policy", "config.budget.max_context_tokens"),
    ("budget policy", "config.budget.max_records"),
    ("budget policy", "config.budget.capacity_cap_tokens"),
    ("budget policy", "config.budget.fallback_policy"),
    ("budget policy", "config.budget.autochunk"),
    ("execution", "config.repeats"),
    ("execution", "config.arms"),
    ("execution", "config.shards.shards"),
    ("execution", "config.shards.concurrent_shards"),
    ("environment", "environment.python_version"),
    ("environment", "environment.platform"),
    ("environment", "environment.dependency_lock_sha256"),
    ("environment", "environment.harness_commit"),
)

_SECRET_ENV_NAMES = tuple(DEFAULT_MODEL_KEY_ENV) + tuple(DEFAULT_SELECTOR_KEY_ENV)


def _lookup(d: dict[str, Any], dotted: str) -> Any:
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _walk_strings(value: Any):
    if isinstance(value, dict):
        for v in value.values():
            yield from _walk_strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _walk_strings(v)
    elif isinstance(value, str):
        yield value


def validate_manifest(manifest: dict[str, Any], *, environ: Optional[dict[str, str]] = None) -> None:
    """Refuse a manifest missing any protocol-required field, or carrying a secret.

    Called before execution (so a run cannot start under-described) and before
    aggregation (so a summary cannot be produced from an under-described run).
    """
    problems: list[str] = []
    if manifest.get("schema") != MANIFEST_SCHEMA:
        problems.append(f"schema must be {MANIFEST_SCHEMA!r}, got {manifest.get('schema')!r}")
    for category, path in REQUIRED_MANIFEST_PATHS:
        value = _lookup(manifest, path)
        if value is None or value == "" or value == []:
            problems.append(f"[{category}] {path} is missing")

    cfg = manifest.get("config", {})
    ds = cfg.get("dataset", {})
    ut, it = ds.get("user_task_ids") or [], ds.get("injection_task_ids") or []
    if ut and it and ds.get("expected_rows") != len(ut) * len(it):
        problems.append(
            f"[dataset identity] expected_rows {ds.get('expected_rows')} != "
            f"{len(ut)} user tasks x {len(it)} injection tasks"
        )
    if cfg.get("budget", {}).get("fallback_policy") not in (None, "fail_open"):
        problems.append("[budget policy] fallback_policy must be 'fail_open'")

    env = os.environ if environ is None else environ
    secrets = {env[n] for n in _SECRET_ENV_NAMES if env.get(n)}
    for s in _walk_strings(manifest):
        if s in secrets:
            problems.append("manifest carries a credential value; manifests record key *names* only")
            break

    for attempt in manifest.get("attempts", []):
        for key in ("repeat", "arm", "attempt_id", "parent_run_id", "started_at"):
            if attempt.get(key) in (None, ""):
                problems.append(f"[execution] attempt missing {key}")
        if attempt.get("valid") is False and not attempt.get("invalidation_reason"):
            problems.append(f"[execution] invalid attempt {attempt.get('attempt_id')} has no invalidation_reason")

    if problems:
        raise ManifestError("manifest incomplete:\n  " + "\n  ".join(problems))


# --------------------------------------------------------------------------- #
# Probes (standard library only)
# --------------------------------------------------------------------------- #


def _git_head(repo_root: Optional[Path]) -> Optional[str]:
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[3]
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _sha256_file(path: Path) -> Optional[str]:
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _dist_version(name: str) -> Optional[str]:
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version(name)
    except Exception:  # PackageNotFoundError and any metadata backend error
        return None
