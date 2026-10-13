"""Configuration, manifest schema, model-alias recording, provider and preflight."""

from __future__ import annotations

import json
import types

import pytest

from agentdojo_fakes import fake_environment, small_config
from csbench.suites.agentdojo import ARM_NEEDLEPATH
from csbench.suites.agentdojo.config import (
    REQUIRED_MANIFEST_PATHS,
    ArmAttempt,
    ConfigError,
    ManifestError,
    ModelIdentity,
    RunConfig,
    RunManifest,
    ShardPlan,
    validate_config,
    validate_manifest,
)
from csbench.suites.agentdojo.preflight import (
    NEUTRAL_SECURITY,
    NEUTRAL_UTILITY,
    install_preflight,
    neutral_run_task_without_injection_tasks,
    preflight_installed,
)
from csbench.suites.agentdojo.provider import (
    MODEL_ALIASES,
    ProviderConfigError,
    build_llm_element,
    describe_model,
    read_api_key,
    resolve_provider_model,
)

SECRET = "fake-gemini-key-value-0123456789abcdef"


# ------------------------------------------------------------------ config


def test_config_round_trips_through_json(tmp_path):
    cfg = small_config()
    cfg.save(tmp_path / "c.json")
    back = RunConfig.load(tmp_path / "c.json")
    assert back == cfg
    assert json.loads((tmp_path / "c.json").read_text())["dataset"]["expected_rows"] == 4


def test_shard_plan_covers_tasks_in_waves():
    plan = ShardPlan.build([f"user_task_{i}" for i in range(40)], shard_size=5, concurrent_shards=2)
    assert len(plan.shards) == 8 and all(len(s) == 5 for s in plan.shards)
    assert plan.waves() == [[0, 1], [2, 3], [4, 5], [6, 7]]


@pytest.mark.parametrize(
    "mutate, fragment",
    [
        (lambda c: setattr(c.selector, "base_url", None), "selector.base_url"),
        (lambda c: setattr(c.selector, "snapshot", None), "selector.snapshot"),
        (lambda c: setattr(c.budget, "autochunk", False), "autochunk"),
        (lambda c: setattr(c.budget, "fallback_policy", "fail_closed"), "fallback_policy"),
        (lambda c: setattr(c.shards, "shards", [["user_task_0"]]), "every user task exactly once"),
        (lambda c: setattr(c, "arms", [ARM_NEEDLEPATH]), "baseline arm is required"),
        (lambda c: setattr(c, "repeats", 0), "repeats"),
        (lambda c: setattr(c.selector, "entry_point", "records"), "entry_point"),
    ],
)
def test_config_validation_refuses(mutate, fragment):
    cfg = small_config()
    mutate(cfg)
    with pytest.raises(ConfigError, match=fragment):
        validate_config(cfg)


# ---------------------------------------------------------------- manifest


def _manifest() -> RunManifest:
    return RunManifest.create(small_config(), fake_environment())


def test_manifest_valid_and_round_trips(tmp_path):
    m = _manifest()
    validate_manifest(m.to_wire())
    m.save(tmp_path / "manifest.json")
    back = RunManifest.load(tmp_path / "manifest.json")
    assert back.to_wire() == m.to_wire()


def _unset(d: dict, dotted: str) -> None:
    parts = dotted.split(".")
    cur = d
    for p in parts[:-1]:
        cur = cur[p]
    cur[parts[-1]] = None


@pytest.mark.parametrize("category, path", REQUIRED_MANIFEST_PATHS)
def test_every_required_manifest_field_is_enforced(category, path):
    wire = _manifest().to_wire()
    _unset(wire, path)
    with pytest.raises(ManifestError, match=path.replace(".", r"\.")):
        validate_manifest(wire)


def test_manifest_expected_rows_must_match_population():
    wire = _manifest().to_wire()
    wire["config"]["dataset"]["expected_rows"] = 3
    with pytest.raises(ManifestError, match="expected_rows"):
        validate_manifest(wire)


def test_manifest_refuses_a_credential_value():
    wire = _manifest().to_wire()
    wire["environment"]["platform"] = SECRET
    with pytest.raises(ManifestError, match="credential"):
        validate_manifest(wire, environ={"GEMINI_API_KEY": SECRET})
    validate_manifest(wire, environ={})  # the same string is fine when it is not a live key


def test_manifest_invalid_attempt_needs_a_reason():
    m = _manifest()
    m.attempts.append(ArmAttempt(repeat=1, arm="baseline", attempt_id="a1", parent_run_id="run-test", started_at="t", valid=False))
    with pytest.raises(ManifestError, match="invalidation_reason"):
        validate_manifest(m.to_wire())


def test_model_alias_recording_keeps_both_identifiers():
    """The manifest retains the identifier AgentDojo accepts AND the model requested."""
    wire = _manifest().to_wire()
    model = wire["config"]["model"]
    assert model["agentdojo_model"] == "gemini-2.5-flash-preview-04-17"
    assert model["provider_model"] == "gemini-2.5-flash"
    assert model["agentdojo_model"] != model["provider_model"]
    assert wire["environment"]["client_package_version"] == "2.29.0"
    assert model["api_key_env"] == ["GEMINI_API_KEY", "NM_GEMINI_API_KEY"]


# ---------------------------------------------------------------- provider


def test_alias_maps_legacy_enum_to_stable_model():
    assert MODEL_ALIASES["gemini-2.5-flash-preview-04-17"] == "gemini-2.5-flash"
    assert resolve_provider_model("gemini-2.5-flash-preview-04-17") == "gemini-2.5-flash"
    assert resolve_provider_model("anything", "explicit") == "explicit"
    with pytest.raises(ProviderConfigError):
        resolve_provider_model("gemini-9-unknown")


def test_api_key_read_order_and_absence():
    assert read_api_key(("GEMINI_API_KEY", "NM_GEMINI_API_KEY"), {"NM_GEMINI_API_KEY": "b"}) == "b"
    assert read_api_key(("GEMINI_API_KEY", "NM_GEMINI_API_KEY"), {"GEMINI_API_KEY": "a", "NM_GEMINI_API_KEY": "b"}) == "a"
    with pytest.raises(ProviderConfigError):
        read_api_key(("GEMINI_API_KEY", "NM_GEMINI_API_KEY"), {})


def test_build_llm_element_with_injected_factories(capsys):
    seen: dict = {}

    def client_factory(api_key, *, timeout_s, attempts):
        seen["timeout_s"], seen["attempts"] = timeout_s, attempts
        seen["key_len"] = len(api_key)
        return types.SimpleNamespace(kind="fake-client")

    def llm_factory(provider_model, client, *, temperature, max_tokens):
        return types.SimpleNamespace(model=provider_model, client=client, temperature=temperature, max_tokens=max_tokens)

    llm = build_llm_element(
        ModelIdentity(),
        environ={"GEMINI_API_KEY": SECRET},
        client_factory=client_factory,
        llm_factory=llm_factory,
    )
    assert seen == {"timeout_s": 60.0, "attempts": 1, "key_len": len(SECRET)}
    assert llm.model == "gemini-2.5-flash"
    assert llm.name == "gemini-2.5-flash-preview-04-17", "pipeline name stays the legacy id"
    assert SECRET not in repr(vars(llm)), "the key is not retained on the element"
    assert SECRET not in capsys.readouterr().out
    assert SECRET not in json.dumps(describe_model(ModelIdentity()))


def test_build_llm_element_refuses_other_providers():
    with pytest.raises(ProviderConfigError):
        build_llm_element(ModelIdentity(provider="openai"), environ={"GEMINI_API_KEY": "k"})


def test_default_client_factory_sets_timeout_and_single_attempt():
    pytest.importorskip("google.genai", reason="google-genai not installed")
    from csbench.suites.agentdojo.provider import default_client_factory

    client = default_client_factory("not-a-real-key", timeout_s=60.0, attempts=1)
    opts = client._api_client._http_options
    assert opts.timeout == 60000
    assert opts.retry_options.attempts == 1
    assert client.vertexai is False


# --------------------------------------------------------------- preflight


def test_preflight_is_neutral_and_writes_nothing(tmp_path):
    calls: list = []

    def original(*a, **k):
        calls.append(a)
        (tmp_path / "none.json").write_text("{}")
        return True, True

    fake = types.ModuleType("fake_benchmark")
    fake.run_task_without_injection_tasks = original
    install_preflight(fake)
    assert preflight_installed(fake)
    install_preflight(fake)  # idempotent
    assert fake._original_run_task_without_injection_tasks is original
    result = fake.run_task_without_injection_tasks("suite", "pipeline", "task", tmp_path, False, "v1.2.2")
    assert result == (NEUTRAL_UTILITY, NEUTRAL_SECURITY) == (False, True)
    assert calls == [] and not (tmp_path / "none.json").exists()
    assert neutral_run_task_without_injection_tasks() == (False, True)


def test_preflight_refuses_a_module_without_the_hook():
    with pytest.raises(AttributeError):
        install_preflight(types.ModuleType("empty"))


def test_preflight_installs_on_the_real_module():
    bm = pytest.importorskip("agentdojo.benchmark", reason="agentdojo extra not installed")
    saved = bm.run_task_without_injection_tasks
    try:
        install_preflight(bm)
        assert preflight_installed(bm)
    finally:
        bm.run_task_without_injection_tasks = saved
