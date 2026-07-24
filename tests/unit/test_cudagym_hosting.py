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

import pytest
from omegaconf import OmegaConf

from slurm.cudagym_hosting import (
    HostingError,
    ResolvedEntry,
    load_endpoints,
    load_recipe_merged,
    probe_endpoint,
    resolve_hosting,
    verify_health_payload,
)

CLUSTER_H100 = {"host": "aws-iad-cs-002", "sku": "h100"}
CLUSTER_GB200 = {"host": "aws-dfw-cs-001", "sku": "gb200"}


def _endpoints_dir(tmp_path):
    d = tmp_path / "endpoints"
    d.mkdir(exist_ok=True)
    (d / "modal.yaml").write_text(
        "b200:\n  sku: B200\n  url: https://b200.modal.run\n"
        "h100:\n  sku: H100\n  url: https://h100.modal.run\n"
    )
    (d / "astra.yaml").write_text(
        "gb10:\n  sku: GB10\n  url: https://titan/dgx-spark\n  disabled_reason: pod outage\n"
    )
    return d


def _recipe(entries: dict):
    return OmegaConf.create({"env": {"cudagym": entries}})


def _resolve(entries, cluster=CLUSTER_H100, num_nodes=4, agentic=False, ep_dir=None):
    return resolve_hosting(_recipe(entries), cluster, num_nodes, agentic, endpoints_dir=ep_dir)


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


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


def test_load_endpoints_names_and_fields(tmp_path):
    entries = load_endpoints(_endpoints_dir(tmp_path))
    assert entries["modal/b200"].provider == "modal"
    assert entries["modal/b200"].url == "https://b200.modal.run"
    assert entries["astra/gb10"].disabled_reason == "pod outage"


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
        {"b200": {"sku": "B200", "hosting": {"kind": "endpoint", "endpoint": "modal/b200"}}},
        ep_dir=_endpoints_dir(tmp_path),
    )
    assert res.cudagym_mode == ""
    assert res.extra_config_opts == ["++env.cudagym.b200.server_url=https://b200.modal.run"]
    assert res.unified_server_url == "https://b200.modal.run"


def test_endpoint_inline_url_no_auth_requirement(tmp_path):
    res = _resolve(
        {"b200": {"sku": "B200", "hosting": {"kind": "endpoint", "url": "http://my-server:8000/"}}},
        ep_dir=_endpoints_dir(tmp_path),
    )
    assert res.endpoints[0].url == "http://my-server:8000"


def test_endpoint_and_url_mutually_exclusive(tmp_path, modal_env):
    with pytest.raises(HostingError, match="mutually exclusive"):
        _resolve(
            {"b200": {"sku": "B200", "hosting": {"kind": "endpoint", "endpoint": "modal/b200", "url": "http://x"}}},
            ep_dir=_endpoints_dir(tmp_path),
        )


def test_unknown_and_disabled_registry_refs(tmp_path, modal_env):
    with pytest.raises(HostingError, match="not found"):
        _resolve(
            {"x": {"sku": "B200", "hosting": {"kind": "endpoint", "endpoint": "modal/nope"}}},
            ep_dir=_endpoints_dir(tmp_path),
        )
    with pytest.raises(HostingError, match="disabled: pod outage"):
        _resolve(
            {"gb10": {"sku": "GB10", "hosting": {"kind": "endpoint", "endpoint": "astra/gb10"}}},
            ep_dir=_endpoints_dir(tmp_path),
        )


def test_registry_sku_must_match_entry_sku(tmp_path, modal_env):
    with pytest.raises(HostingError, match="serves H100"):
        _resolve(
            {"b200": {"sku": "B200", "hosting": {"kind": "endpoint", "endpoint": "modal/h100"}}},
            ep_dir=_endpoints_dir(tmp_path),
        )


def test_missing_modal_auth_env_fails(tmp_path, monkeypatch):
    monkeypatch.delenv("MODAL_PROXY_TOKEN_ID", raising=False)
    monkeypatch.setenv("MODAL_PROXY_TOKEN_SECRET", "s")
    with pytest.raises(HostingError, match="MODAL_PROXY_TOKEN_ID"):
        _resolve(
            {"b200": {"sku": "B200", "hosting": {"kind": "endpoint", "endpoint": "modal/b200"}}},
            ep_dir=_endpoints_dir(tmp_path),
        )


def test_legacy_server_url_shim_warns(tmp_path):
    res = _resolve(
        {"b200": {"sku": "B200", "server_url": "http://legacy:8000"}},
        ep_dir=_endpoints_dir(tmp_path),
    )
    assert res.endpoints[0].url == "http://legacy:8000"
    assert any("deprecated" in w for w in res.warnings)


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


def test_in_allocation_sku_must_match_cluster_silicon(tmp_path):
    with pytest.raises(HostingError, match="h100 silicon"):
        _resolve(
            {"b200": {"sku": "B200", "hosting": {"kind": "colocated"}}},
            cluster=CLUSTER_H100,
            ep_dir=_endpoints_dir(tmp_path),
        )
    # GB200-silicon clusters serve B200 kernels.
    res = _resolve(
        {"b200": {"sku": "B200", "hosting": {"kind": "colocated"}}},
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


def test_slurm_service_fields_and_warning(tmp_path):
    with pytest.raises(HostingError, match="missing fields"):
        _resolve(
            {"svc": {"sku": "H100", "hosting": {"kind": "slurm-service", "service_cluster": "x"}}},
            ep_dir=_endpoints_dir(tmp_path),
        )
    res = _resolve(
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
    assert res.slurm_services[0].service["service_login_port"] == 8998
    assert any("experimental" in w for w in res.warnings)


def test_agentic_requires_exactly_one_entry(tmp_path, modal_env):
    with pytest.raises(HostingError, match="exactly one"):
        _resolve(
            {
                "a": {"sku": "B200", "hosting": {"kind": "endpoint", "endpoint": "modal/b200"}},
                "b": {"sku": "H100", "hosting": {"kind": "endpoint", "endpoint": "modal/h100"}},
            },
            agentic=True,
            ep_dir=_endpoints_dir(tmp_path),
        )
    res = _resolve(
        {"h100": {"sku": "H100", "hosting": {"kind": "colocated"}}},
        cluster=CLUSTER_H100,
        agentic=True,
        ep_dir=_endpoints_dir(tmp_path),
    )
    assert any("time-shares" in w for w in res.warnings)


def test_multiple_endpoints_no_unified_url(tmp_path, modal_env):
    res = _resolve(
        {
            "b200": {"sku": "B200", "hosting": {"kind": "endpoint", "endpoint": "modal/b200"}},
            "h100": {"sku": "H100", "hosting": {"kind": "endpoint", "endpoint": "modal/h100"}},
        },
        ep_dir=_endpoints_dir(tmp_path),
    )
    assert res.unified_server_url == ""
    assert len(res.extra_config_opts) == 2


# --------------------------------------------------------------------------
# /health payload verification + probe
# --------------------------------------------------------------------------


def test_verify_health_payload_table():
    ok, _ = verify_health_payload({"gpu_model": "NVIDIA B200", "sm_version": "sm_100"}, "B200")
    assert ok
    ok, _ = verify_health_payload({"gpu_model": "NVIDIA GB200", "sm_version": "sm_100a"}, "B200")
    assert ok  # GB200 silicon serves B200 kernels
    ok, detail = verify_health_payload({"gpu_model": "NVIDIA H100 80GB HBM3"}, "B200")
    assert not ok and "gpu_model" in detail
    ok, detail = verify_health_payload({"gpu_model": "NVIDIA B200", "sm_version": "sm_90"}, "B200")
    assert not ok and "sm_version" in detail
    ok, detail = verify_health_payload({}, "B200")
    assert ok and detail.startswith("unverifiable")
    ok, detail = verify_health_payload({"gpu_model": "Whatever"}, "RTX_5090")
    assert ok and "no expectations" in detail


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
        return _FakeResponse(payload={"gpu_model": "NVIDIA B200", "sm_version": "sm_100"})

    _install_fake_requests(monkeypatch, get)
    monkeypatch.setenv("CUDAGYM_AUTH_TOKEN", "tok")
    entry = ResolvedEntry(name="b200", sku="B200", kind="endpoint", url="http://srv:8000")
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
    entry = ResolvedEntry(name="b200", sku="B200", kind="endpoint", url="http://down:8000")
    with pytest.raises(HostingError, match="unreachable"):
        probe_endpoint(entry, retries=2)
    assert calls["n"] == 3


def test_probe_endpoint_http_error_raises(monkeypatch):
    _install_fake_requests(
        monkeypatch, lambda url, headers=None, timeout=None: _FakeResponse(401, text="denied")
    )
    entry = ResolvedEntry(name="b200", sku="B200", kind="endpoint", url="http://srv:8000")
    with pytest.raises(HostingError, match="HTTP 401"):
        probe_endpoint(entry, retries=0)
