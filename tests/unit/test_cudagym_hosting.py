# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the recipe-declared CudaGym hosting resolver."""

import sys
import types
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from slurm.cudagym_hosting import (
    HostingError,
    ResolvedEntry,
    check_registry_against_solswarm,
    ensure_vendored_cudagym,
    load_endpoints,
    load_recipe_merged,
    probe_endpoint,
    resolve_hosting,
    verify_health_payload,
)

CLUSTER_H100 = {"host": "aws-iad-cs-002", "sku": "h100"}
CLUSTER_GB200 = {"host": "cluster-gb200", "sku": "gb200"}


def _endpoints_dir(tmp_path):
    d = tmp_path / "endpoints"
    d.mkdir(exist_ok=True)
    (d / "modal.yaml").write_text(
        "b200:\n  sku: B200\n  url: https://b200.modal.run\n"
        "h100:\n  sku: H100\n  url: https://h100.modal.run\n"
    )
    # The entry key is a familiar shorthand; `sku:` carries the canonical
    # SupportedHardware value, which for a DGX Spark is not "GB10".
    (d / "astra.yaml").write_text(
        "gb10:\n  sku: DGX_SPARK\n  url: https://titan/dgx-spark\n  disabled_reason: pod outage\n"
    )
    return d


def _recipe(entries: dict):
    return OmegaConf.create({"env": {"cudagym": entries}})


def _resolve(entries, cluster=CLUSTER_H100, num_nodes=4, agentic=False, ep_dir=None):
    return resolve_hosting(
        _recipe(entries), cluster, num_nodes, agentic, endpoints_dir=ep_dir
    )


@pytest.fixture()
def modal_env(monkeypatch):
    monkeypatch.setenv("MODAL_PROXY_TOKEN_ID", "id")
    monkeypatch.setenv("MODAL_PROXY_TOKEN_SECRET", "secret")
    monkeypatch.delenv("CUDAGYM_UNIFIED_SERVER_URL", raising=False)
    monkeypatch.delenv("CUDAGYM_URL", raising=False)


# --------------------------------------------------------------------------
# Recipe loading (defaults inheritance)
# --------------------------------------------------------------------------


def test_load_recipe_merged_follows_defaults(tmp_path):
    (tmp_path / "parent.yaml").write_text(
        "env:\n  cudagym:\n    b200:\n      sku: B200\n      hosting: {kind: colocated}\n"
    )
    child = tmp_path / "child.yaml"
    child.write_text('defaults: "parent.yaml"\ngrpo:\n  num_prompts_per_step: 4\n')
    cfg = load_recipe_merged(child)
    assert OmegaConf.select(cfg, "env.cudagym.b200.hosting.kind") == "colocated"
    assert OmegaConf.select(cfg, "grpo.num_prompts_per_step") == 4


def test_load_recipe_merged_tolerates_parent_mandatory_values(tmp_path):
    """A parent may declare top-level ``???`` (mandatory) values; composing the
    chain must carry them through unresolved rather than raising. The child
    fills one here; the other stays missing for a later override to provide."""
    (tmp_path / "parent.yaml").write_text(
        "cluster: ???\nrun_name: ???\n"
        "env:\n  cudagym:\n    b200:\n      sku: B200\n      hosting: {kind: colocated}\n"
    )
    child = tmp_path / "child.yaml"
    child.write_text('defaults: "parent.yaml"\ncluster:\n  num_nodes: 2\n')
    cfg = load_recipe_merged(child)
    assert OmegaConf.select(cfg, "cluster.num_nodes") == 2
    assert OmegaConf.is_missing(cfg, "run_name")
    assert OmegaConf.select(cfg, "env.cudagym.b200.sku") == "B200"


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


def test_load_endpoints_names_and_fields(tmp_path):
    entries = load_endpoints(_endpoints_dir(tmp_path))
    assert entries["modal/b200"].provider == "modal"
    assert entries["modal/b200"].url == "https://b200.modal.run"
    assert entries["astra/gb10"].disabled_reason == "pod outage"
    assert entries["astra/gb10"].sku == "DGX_SPARK"


def test_load_endpoints_rejects_a_non_canonical_sku(tmp_path):
    d = _endpoints_dir(tmp_path)
    (d / "static.yaml").write_text("local:\n  sku: b200\n  url: http://srv:8000\n")
    with pytest.raises(HostingError, match="not spelled the way CudaGym spells it"):
        load_endpoints(d)


def test_load_endpoints_rejects_incomplete_entry(tmp_path):
    d = tmp_path / "endpoints"
    d.mkdir()
    (d / "static.yaml").write_text("broken:\n  url: http://x\n")
    with pytest.raises(HostingError, match="sku"):
        load_endpoints(d)


# --------------------------------------------------------------------------
# Resolution + validation matrix
# --------------------------------------------------------------------------


def test_endpoint_via_registry_resolves_url_and_opts(tmp_path, modal_env):
    res = _resolve(
        {
            "b200": {
                "sku": "B200",
                "hosting": {"kind": "endpoint", "endpoint": "modal/b200"},
            }
        },
        ep_dir=_endpoints_dir(tmp_path),
    )
    assert res.cudagym_mode == ""
    assert res.extra_config_opts == [
        "++env.cudagym.b200.server_url=https://b200.modal.run"
    ]
    assert res.sku_endpoints == {"B200": "https://b200.modal.run"}
    assert res.unified_server_url == "https://b200.modal.run"


def test_endpoint_inline_url_no_auth_requirement(tmp_path):
    res = _resolve(
        {
            "b200": {
                "sku": "B200",
                "hosting": {"kind": "endpoint", "url": "http://my-server:8000/"},
            }
        },
        ep_dir=_endpoints_dir(tmp_path),
    )
    assert res.endpoints[0].url == "http://my-server:8000"


def test_endpoint_and_url_mutually_exclusive(tmp_path, modal_env):
    with pytest.raises(HostingError, match="mutually exclusive"):
        _resolve(
            {
                "b200": {
                    "sku": "B200",
                    "hosting": {
                        "kind": "endpoint",
                        "endpoint": "modal/b200",
                        "url": "http://x",
                    },
                }
            },
            ep_dir=_endpoints_dir(tmp_path),
        )


def test_unknown_and_disabled_registry_refs(tmp_path, modal_env):
    with pytest.raises(HostingError, match="not found"):
        _resolve(
            {
                "x": {
                    "sku": "B200",
                    "hosting": {"kind": "endpoint", "endpoint": "modal/nope"},
                }
            },
            ep_dir=_endpoints_dir(tmp_path),
        )
    with pytest.raises(HostingError, match="disabled: pod outage"):
        _resolve(
            {
                "gb10": {
                    "sku": "DGX_SPARK",
                    "hosting": {"kind": "endpoint", "endpoint": "astra/gb10"},
                }
            },
            ep_dir=_endpoints_dir(tmp_path),
        )


def test_registry_sku_must_match_entry_sku(tmp_path, modal_env):
    with pytest.raises(HostingError, match="serves H100"):
        _resolve(
            {
                "b200": {
                    "sku": "B200",
                    "hosting": {"kind": "endpoint", "endpoint": "modal/h100"},
                }
            },
            ep_dir=_endpoints_dir(tmp_path),
        )


def test_missing_modal_auth_env_fails(tmp_path, monkeypatch):
    monkeypatch.delenv("MODAL_PROXY_TOKEN_ID", raising=False)
    monkeypatch.setenv("MODAL_PROXY_TOKEN_SECRET", "s")
    with pytest.raises(HostingError, match="MODAL_PROXY_TOKEN_ID"):
        _resolve(
            {
                "b200": {
                    "sku": "B200",
                    "hosting": {"kind": "endpoint", "endpoint": "modal/b200"},
                }
            },
            ep_dir=_endpoints_dir(tmp_path),
        )


def test_bare_server_url_is_not_honored(tmp_path):
    """A bare `server_url:` without `hosting:` errors, and the error names the
    unhonored field so the fix is one obvious edit."""
    with pytest.raises(HostingError, match="server_url.*not honored"):
        _resolve(
            {"b200": {"sku": "B200", "server_url": "http://legacy:8000"}},
            ep_dir=_endpoints_dir(tmp_path),
        )


def test_entry_without_hosting_errors(tmp_path):
    with pytest.raises(HostingError, match="no `hosting:` block"):
        _resolve({"b200": {"sku": "B200"}}, ep_dir=_endpoints_dir(tmp_path))


def test_escape_hatch_env_url(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDAGYM_UNIFIED_SERVER_URL", "http://hatch:8000")
    res = _resolve(
        {"b200": {"sku": "B200", "hosting": {"kind": "endpoint"}}},
        ep_dir=_endpoints_dir(tmp_path),
    )
    assert res.endpoints[0].url == "http://hatch:8000"
    assert any("environment" in w for w in res.warnings)
    monkeypatch.delenv("CUDAGYM_UNIFIED_SERVER_URL")
    with pytest.raises(HostingError, match="escape hatch"):
        _resolve(
            {"b200": {"sku": "B200", "hosting": {"kind": "endpoint"}}},
            ep_dir=_endpoints_dir(tmp_path),
        )


def test_single_in_allocation_entry_enforced(tmp_path):
    with pytest.raises(HostingError, match="at most one"):
        _resolve(
            {
                "a": {"sku": "H100", "hosting": {"kind": "colocated"}},
                "b": {"sku": "H100", "hosting": {"kind": "disjoint", "num_nodes": 1}},
            },
            ep_dir=_endpoints_dir(tmp_path),
        )


def test_duplicate_sku_across_entries_is_refused(tmp_path, modal_env):
    """Every consumer downstream keys on the SKU rather than the entry name, so
    two entries for the same GPU are refused, and the message names both."""
    with pytest.raises(HostingError, match="distinct sku") as excinfo:
        _resolve(
            {
                "b200": {
                    "sku": "B200",
                    "hosting": {"kind": "endpoint", "endpoint": "modal/b200"},
                },
                "b200_spare": {
                    "sku": "B200",
                    "hosting": {"kind": "endpoint", "url": "http://spare:8000"},
                },
            },
            ep_dir=_endpoints_dir(tmp_path),
        )
    assert "b200" in str(excinfo.value) and "b200_spare" in str(excinfo.value)


def test_in_allocation_entry_maps_to_an_empty_endpoint_url(tmp_path, modal_env):
    """An in-allocation server has no address until ray.sub brings its load
    balancer up, so its SKU maps to the empty string — the Gym resources
    server's signal to read CUDAGYM_UNIFIED_SERVER_URL at runtime."""
    res = _resolve(
        {
            "h100": {"sku": "H100", "hosting": {"kind": "colocated"}},
            "b200": {
                "sku": "B200",
                "hosting": {"kind": "endpoint", "endpoint": "modal/b200"},
            },
        },
        cluster=CLUSTER_H100,
        agentic=True,
        ep_dir=_endpoints_dir(tmp_path),
    )
    assert res.sku_endpoints == {"H100": "", "B200": "https://b200.modal.run"}


def test_in_allocation_sku_must_match_cluster_silicon(tmp_path):
    with pytest.raises(HostingError, match="h100 silicon"):
        _resolve(
            {"b200": {"sku": "B200", "hosting": {"kind": "colocated"}}},
            cluster=CLUSTER_H100,
            ep_dir=_endpoints_dir(tmp_path),
        )
    # Each silicon serves only its own kernels. B200 and GB200 share an SM
    # class but not their clock locking, so a B200 entry on GB200 silicon is
    # refused here rather than passing submit and failing the runtime check
    # against the server's reported GPU.
    with pytest.raises(HostingError, match="gb200 silicon"):
        _resolve(
            {"b200": {"sku": "B200", "hosting": {"kind": "colocated"}}},
            cluster=CLUSTER_GB200,
            ep_dir=_endpoints_dir(tmp_path),
        )
    res = _resolve(
        {"gb200": {"sku": "GB200", "hosting": {"kind": "colocated"}}},
        cluster=CLUSTER_GB200,
        ep_dir=_endpoints_dir(tmp_path),
    )
    assert res.cudagym_mode == "colocated"
    assert res.unified_server_url == ""


def test_disjoint_bounds_and_mode(tmp_path):
    res = _resolve(
        {"h100": {"sku": "H100", "hosting": {"kind": "disjoint", "num_nodes": 2}}},
        cluster=CLUSTER_H100,
        num_nodes=4,
        ep_dir=_endpoints_dir(tmp_path),
    )
    assert res.cudagym_mode == "disjoint"
    assert res.cudagym_num_nodes == 2
    with pytest.raises(HostingError, match=r"\[1, --num-nodes\)"):
        _resolve(
            {"h100": {"sku": "H100", "hosting": {"kind": "disjoint", "num_nodes": 4}}},
            cluster=CLUSTER_H100,
            num_nodes=4,
            ep_dir=_endpoints_dir(tmp_path),
        )


def test_registry_drift_check_against_gpu_skus_toml(tmp_path):
    """check_registry_against_solswarm parses the actual gpu-skus.toml format:
    [[gpu_skus]] tables with `id` and `cudagym_url`."""
    root = tmp_path / "solswarm"
    (root / "deployments" / "files").mkdir(parents=True)
    (root / "deployments" / "files" / "gpu-skus.toml").write_text(
        '[[gpu_skus]]\nid = "b200"\ncudagym_url = "https://fleet-v9-b200-web.modal.run"\n'
        '[[gpu_skus]]\nid = "h100"\ncudagym_url = "https://fleet-v9-h100-web.modal.run"\n'
    )
    ep_dir = tmp_path / "endpoints"
    ep_dir.mkdir()
    (ep_dir / "modal.yaml").write_text(
        "b200: {sku: B200, url: https://fleet-v9-b200-web.modal.run}\n"
        "h100: {sku: H100, url: https://STALE-h100-web.modal.run}\n"
    )
    lines = check_registry_against_solswarm(load_endpoints(ep_dir), root)
    assert len(lines) == 1 and "modal/h100" in lines[0] and "fleet-v9-h100" in lines[0]
    # Absent toml (no solswarm checkout) is silent, never fatal.
    assert (
        check_registry_against_solswarm(load_endpoints(ep_dir), tmp_path / "nope") == []
    )


def test_registry_drift_check_matches_on_key_not_sku(tmp_path):
    """The upstream id is the registry KEY, not the sku: an entry whose SDK enum
    name differs from the fleet id (gb10 -> DGX_SPARK) must still be checked,
    and entries sharing a sku (A100 memory variants) must not cross-compare."""
    root = tmp_path / "solswarm"
    (root / "deployments" / "files").mkdir(parents=True)
    (root / "deployments" / "files" / "gpu-skus.toml").write_text(
        '[[gpu_skus]]\nid = "gb10"\ncudagym_url = "https://titan/dgx-spark"\n'
        '[[gpu_skus]]\nid = "a100"\ncudagym_url = "https://fleet-a100-web.modal.run"\n'
        '[[gpu_skus]]\nid = "a100-40gb"\ncudagym_url = "https://fleet-a100-40gb-web.modal.run"\n'
    )
    ep_dir = tmp_path / "endpoints"
    ep_dir.mkdir()
    (ep_dir / "astra.yaml").write_text(
        "gb10: {sku: DGX_SPARK, url: https://STALE/dgx-spark}\n"
    )
    (ep_dir / "modal.yaml").write_text(
        "a100: {sku: A100, url: https://fleet-a100-web.modal.run}\n"
        "a100-40gb: {sku: A100, url: https://fleet-a100-40gb-web.modal.run}\n"
    )
    lines = check_registry_against_solswarm(load_endpoints(ep_dir), root)
    # gb10 is flagged via its key despite the DGX_SPARK sku; the A100 variants
    # each match their own upstream id, so no false drift between them.
    assert (
        len(lines) == 1 and "astra/gb10" in lines[0] and "titan/dgx-spark" in lines[0]
    )


def test_shipped_registry_skus_are_exact_supported_hardware_values():
    """Guard the shipped endpoints/*.yaml files: a registry sku is compared for
    string equality with the recipe entry's sku, and the recipe sku must be an
    exact (case-sensitive) SupportedHardware value at env init — so a registry
    sku that is only a name alias (e.g. GB10 for DGX_SPARK) would force a recipe
    that passes submit-time validation and then fails inside the job."""
    ensure_vendored_cudagym()
    from cudagym.contracts.solution import SupportedHardware

    for name, entry in load_endpoints().items():
        assert entry.sku in {h.value for h in SupportedHardware}, (
            f"{name}: sku {entry.sku!r} is not an exact SupportedHardware value"
        )


def test_slurm_service_fields_and_warning(tmp_path):
    with pytest.raises(HostingError, match="missing fields"):
        _resolve(
            {
                "svc": {
                    "sku": "H100",
                    "hosting": {"kind": "slurm-service", "service_cluster": "x"},
                }
            },
            ep_dir=_endpoints_dir(tmp_path),
        )
    res = _resolve(
        {
            "svc": {
                "sku": "H100",
                "hosting": {
                    "kind": "slurm-service",
                    "service_cluster": "cw-dfw-cs-001",
                    "num_service_nodes": 2,
                    "endpoint_port": 9100,
                },
            }
        },
        ep_dir=_endpoints_dir(tmp_path),
    )
    assert res.slurm_services[0].service["service_login_port"] == 8998
    assert any("experimental" in w for w in res.warnings)
    with pytest.raises(HostingError, match="submit cluster itself"):
        _resolve(
            {
                "svc": {
                    "sku": "H100",
                    "hosting": {
                        "kind": "slurm-service",
                        "service_cluster": "aws-iad-cs-002",
                        "num_service_nodes": 2,
                        "endpoint_port": 9100,
                    },
                }
            },
            ep_dir=_endpoints_dir(tmp_path),
        )


def test_agentic_refuses_slurm_service_hosting(tmp_path):
    """slurm-service hosting delivers its URL only as a ++server_url override,
    which the NeMo-Gym servers never read (they evaluate against the per-SKU
    endpoint map, which carries no URL for this kind), so an agentic recipe
    declaring it must be refused with the working alternatives named."""
    entries = {
        "svc": {
            "sku": "H100",
            "hosting": {
                "kind": "slurm-service",
                "service_cluster": "cw-dfw-cs-001",
                "num_service_nodes": 2,
                "endpoint_port": 9100,
            },
        }
    }
    with pytest.raises(HostingError, match="kind: endpoint.*colocated"):
        _resolve(entries, agentic=True, ep_dir=_endpoints_dir(tmp_path))
    # The same declaration stays valid for single-turn (non-agentic) recipes.
    res = _resolve(entries, agentic=False, ep_dir=_endpoints_dir(tmp_path))
    assert res.slurm_services[0].name == "svc"


def test_agentic_resolves_multiple_entries(tmp_path, modal_env):
    """The agentic resources server holds one endpoint per GPU, so a recipe may
    declare several entries; each lands in the SKU -> URL map. Two endpoints
    name no single ambient URL, so unified_server_url stays empty."""
    res = _resolve(
        {
            "a": {
                "sku": "B200",
                "hosting": {"kind": "endpoint", "endpoint": "modal/b200"},
            },
            "b": {
                "sku": "H100",
                "hosting": {"kind": "endpoint", "endpoint": "modal/h100"},
            },
        },
        agentic=True,
        ep_dir=_endpoints_dir(tmp_path),
    )
    assert res.sku_endpoints == {
        "B200": "https://b200.modal.run",
        "H100": "https://h100.modal.run",
    }
    assert res.unified_server_url == ""
    # In-allocation hosting stays a warning rather than an error.
    res = _resolve(
        {"h100": {"sku": "H100", "hosting": {"kind": "colocated"}}},
        cluster=CLUSTER_H100,
        agentic=True,
        ep_dir=_endpoints_dir(tmp_path),
    )
    assert any("time-shares" in w for w in res.warnings)
    # Zero entries leaves the resources server with no endpoint to evaluate on.
    with pytest.raises(HostingError, match="at least one env.cudagym entry"):
        _resolve({}, agentic=True, ep_dir=_endpoints_dir(tmp_path))


def test_multiple_endpoints_no_unified_url(tmp_path, modal_env):
    res = _resolve(
        {
            "b200": {
                "sku": "B200",
                "hosting": {"kind": "endpoint", "endpoint": "modal/b200"},
            },
            "h100": {
                "sku": "H100",
                "hosting": {"kind": "endpoint", "endpoint": "modal/h100"},
            },
        },
        ep_dir=_endpoints_dir(tmp_path),
    )
    assert res.unified_server_url == ""
    assert len(res.extra_config_opts) == 2


# --------------------------------------------------------------------------
# /health payload verification + probe
# --------------------------------------------------------------------------


def test_verify_health_payload_is_the_sdk_helper_and_stays_three_valued():
    # The name this module exports is the SDK's function, not a copy of it: the
    # payload table it is checked against lives with the implementation, in
    # cudagym's own tests. What submit_grpo depends on is the three-valued
    # verdict -- it fails the submit on False, and prints "NOT VERIFIED" on
    # None rather than treating an unchecked endpoint as a pass.
    import cudagym.rl

    assert verify_health_payload is cudagym.rl.verify_health_payload
    match = {"gpu_model": "NVIDIA B200", "sm_version": "sm_100"}
    mismatch = {"gpu_model": "NVIDIA GB200", "sm_version": "sm_100a"}
    assert verify_health_payload(match, "B200")[0] is True
    assert verify_health_payload(mismatch, "B200")[0] is False
    assert verify_health_payload({}, "B200")[0] is None


class _FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text

    def json(self):
        return self._payload


def _install_fake_requests(monkeypatch, get):
    fake = types.ModuleType("requests")
    fake.get = get
    monkeypatch.setitem(sys.modules, "requests", fake)


def test_probe_endpoint_success_and_headers(monkeypatch):
    seen = {}

    def get(url, headers=None, timeout=None):
        seen["url"] = url
        seen["headers"] = headers
        return _FakeResponse(
            payload={"gpu_model": "NVIDIA B200", "sm_version": "sm_100"}
        )

    _install_fake_requests(monkeypatch, get)
    monkeypatch.setenv("CUDAGYM_AUTH_TOKEN", "tok")
    entry = ResolvedEntry(
        name="b200", sku="B200", kind="endpoint", url="http://srv:8000"
    )
    payload = probe_endpoint(entry)
    assert seen["url"] == "http://srv:8000/health"
    assert seen["headers"]["Authorization"] == "Bearer tok"
    assert verify_health_payload(payload, "B200")[0]


def test_probe_endpoint_unreachable_raises_after_retries(monkeypatch):
    calls = {"n": 0}

    def get(url, headers=None, timeout=None):
        calls["n"] += 1
        raise OSError("connection refused")

    _install_fake_requests(monkeypatch, get)
    entry = ResolvedEntry(
        name="b200", sku="B200", kind="endpoint", url="http://down:8000"
    )
    with pytest.raises(HostingError, match="unreachable"):
        probe_endpoint(entry, retries=2)
    assert calls["n"] == 3


def test_probe_endpoint_http_error_raises(monkeypatch):
    _install_fake_requests(
        monkeypatch,
        lambda url, headers=None, timeout=None: _FakeResponse(401, text="denied"),
    )
    entry = ResolvedEntry(
        name="b200", sku="B200", kind="endpoint", url="http://srv:8000"
    )
    with pytest.raises(HostingError, match="HTTP 401"):
        probe_endpoint(entry, retries=0)


# ---------------------------------------------------------------------------
# ensure_vendored_cudagym
# ---------------------------------------------------------------------------


def test_ensure_vendored_cudagym_noop_when_importable(monkeypatch):
    import slurm.cudagym_hosting as hosting_mod

    monkeypatch.setattr(hosting_mod, "_cudagym_import_error", lambda: None)
    before = list(sys.path)
    ensure_vendored_cudagym("/nonexistent")
    assert sys.path == before


def test_ensure_vendored_cudagym_errors_on_missing_checkout(tmp_path, monkeypatch):
    import slurm.cudagym_hosting as hosting_mod

    monkeypatch.setattr(
        hosting_mod, "_cudagym_import_error", lambda: "No module named 'cudagym'"
    )
    with pytest.raises(HostingError, match="git submodule update --init"):
        ensure_vendored_cudagym(tmp_path)


def test_ensure_vendored_cudagym_errors_on_missing_deps(tmp_path, monkeypatch):
    import slurm.cudagym_hosting as hosting_mod

    src = tmp_path / "3rdparty" / "solswarm" / "cudagym" / "src"
    (src / "cudagym").mkdir(parents=True)
    (src / "cudagym" / "__init__.py").write_text("")
    monkeypatch.setattr(
        hosting_mod, "_cudagym_import_error", lambda: "No module named 'pydantic'"
    )
    with pytest.raises(HostingError, match="loguru pydantic"):
        ensure_vendored_cudagym(tmp_path)
    # The failed bootstrap must not leave its path entry behind.
    assert str(src) not in sys.path


def test_ensure_vendored_cudagym_bootstraps_sys_path(tmp_path, monkeypatch):
    import slurm.cudagym_hosting as hosting_mod

    src = tmp_path / "3rdparty" / "solswarm" / "cudagym" / "src"
    (src / "cudagym").mkdir(parents=True)
    (src / "cudagym" / "__init__.py").write_text("")
    # The first import check fails (no venv); the retry after the path insert succeeds.
    outcomes = iter(["No module named 'cudagym'", None])
    monkeypatch.setattr(hosting_mod, "_cudagym_import_error", lambda: next(outcomes))
    try:
        ensure_vendored_cudagym(tmp_path)
        assert sys.path[0] == str(src)
    finally:
        while str(src) in sys.path:
            sys.path.remove(str(src))


def test_ensure_vendored_cudagym_displaces_a_stale_installed_package(
    tmp_path, monkeypatch
):
    """An installed cudagym without cudagym.rl must not shadow the vendored checkout.

    The failed probe leaves the stale parent in sys.modules, so a retry that
    did not clear it would resolve `rl` against that same package's __path__
    and fail again — with a message blaming missing dependencies.
    """
    import slurm.cudagym_hosting as hosting_mod

    src = tmp_path / "3rdparty" / "solswarm" / "cudagym" / "src"
    (src / "cudagym").mkdir(parents=True)
    (src / "cudagym" / "__init__.py").write_text("")
    stale = types.ModuleType("cudagym")
    monkeypatch.setitem(sys.modules, "cudagym", stale)
    # Fails while the stale package is cached, succeeds once it is gone.
    monkeypatch.setattr(
        hosting_mod,
        "_cudagym_import_error",
        lambda: "No module named 'cudagym.rl'" if "cudagym" in sys.modules else None,
    )
    ensure_vendored_cudagym(tmp_path)
    assert str(src) in sys.path
    sys.path.remove(str(src))


def test_ensure_vendored_cudagym_names_the_stale_package_when_that_is_the_cause(
    tmp_path, monkeypatch
):
    """A persistent cudagym.rl failure points at the installed copy, not at pip."""
    import slurm.cudagym_hosting as hosting_mod

    src = tmp_path / "3rdparty" / "solswarm" / "cudagym" / "src"
    (src / "cudagym").mkdir(parents=True)
    (src / "cudagym" / "__init__.py").write_text("")
    monkeypatch.setitem(sys.modules, "cudagym", types.ModuleType("cudagym"))
    monkeypatch.setattr(
        hosting_mod, "_cudagym_import_error", lambda: "No module named 'cudagym.rl'"
    )
    with pytest.raises(HostingError, match="predates the cudagym.rl helpers"):
        ensure_vendored_cudagym(tmp_path)
    # Both edits are undone: the path entry and the module the purge removed.
    assert str(src) not in sys.path
    assert "cudagym" in sys.modules


def test_two_skus_may_not_share_one_endpoint(tmp_path):
    """One address cannot serve two GPUs: the rows for one would be timed on the other.

    The /health preflight cannot catch this — it probes the shared URL once per
    declared SKU, and one of those probes passes.
    """
    entries = {
        "b200": {
            "sku": "B200",
            "hosting": {"kind": "endpoint", "url": "http://shared:8000"},
        },
        "h100": {
            "sku": "H100",
            "hosting": {"kind": "endpoint", "url": "http://shared:8000"},
        },
    }
    with pytest.raises(HostingError, match="same endpoint"):
        _resolve(entries, agentic=True, ep_dir=tmp_path)
    # Distinct addresses are the normal multi-SKU case.
    entries["h100"]["hosting"]["url"] = "http://h100:8000"
    res = _resolve(entries, agentic=True, ep_dir=tmp_path)
    assert res.sku_endpoints == {
        "B200": "http://shared:8000",
        "H100": "http://h100:8000",
    }


# --- shipped agentic recipes must produce a Gym config the launcher can start --

ATLAS_RECIPES_DIR = (
    Path(__file__).parents[2] / "examples" / "configs" / "recipes" / "atlas"
)


def test_agentic_recipes_declare_no_empty_gym_mappings():
    """NeMo-Gym's launcher indexes into every top-level mapping of the global
    config as a candidate server entry, so an empty mapping (for example a
    ``cudagym_endpoints: {}`` placeholder) aborts spin-up with an IndexError.
    A shipped recipe must therefore either fill such a key or omit it."""
    recipes = sorted(ATLAS_RECIPES_DIR.glob("grpo_cuda_agentic*.yaml"))
    assert recipes, f"no agentic recipes found under {ATLAS_RECIPES_DIR}"
    for path in recipes:
        gym_block = load_recipe_merged(path).env.nemo_gym
        empty = [
            key
            for key in gym_block
            if OmegaConf.is_dict(gym_block[key]) and len(gym_block[key]) == 0
        ]
        assert not empty, (
            f"{path.name}: env.nemo_gym key(s) {empty} are empty mappings, "
            "which NeMo-Gym's launcher cannot start as servers"
        )


def test_endpoint_override_creates_the_gym_key_from_scratch():
    """The base agentic recipe deliberately does not declare
    ``env.nemo_gym.cudagym_endpoints`` (see the test above), so the submit-time
    ``++`` override must create the whole path itself. Apply it exactly the way
    ``examples/nemo_gym/run_grpo_nemo_gym.py`` does."""
    from nemo_rl.utils.config import parse_hydra_overrides

    recipe = load_recipe_merged(ATLAS_RECIPES_DIR / "grpo_cuda_agentic_qwen3-8b.yaml")
    assert "cudagym_endpoints" not in recipe.env.nemo_gym
    merged = parse_hydra_overrides(
        recipe, ["++env.nemo_gym.cudagym_endpoints.B200=https://b200.test"]
    )
    assert OmegaConf.to_container(merged.env.nemo_gym.cudagym_endpoints) == {
        "B200": "https://b200.test"
    }
