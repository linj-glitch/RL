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

"""Resolve and validate recipe-declared CudaGym hosting.

Every ``env.cudagym.<name>`` recipe entry declares where its eval servers live:

    env:
      cudagym:
        b200:
          sku: "B200"
          hosting:
            kind: endpoint          # colocated | disjoint | endpoint | slurm-service
            endpoint: modal/b200    # kind=endpoint: registry ref "<provider>/<key>" ...
            # url: https://...      #   ... or an inline URL (mutually exclusive)
            # num_nodes: 1          # kind=disjoint only
            # service_cluster: ...  # kind=slurm-service (experimental) + num_service_nodes,
            #                       #   endpoint_port, [service_login_port]

``kind`` is topology; the provider (modal/astra/static) is a property of the
registry entry (``endpoints/<provider>.yaml``). ``submit_grpo.py`` resolves and
validates every entry at submit time — hosting is declared in the recipe, not
on the command line. Remote endpoints are pinged at submit time and their
reported GPU is checked against the declared SKU; in-allocation servers (which
don't exist yet at submit) get the same check at runtime init
(``verify_endpoint_sku``). ``slurm-service`` entries stand up a CudaGym service
job on ANOTHER Slurm cluster at submit time and chain login-node proxies back
to this one (``slurm.deploy_remote_cudagym``).

Back-compat: a legacy ``server_url:`` with no ``hosting:`` block is treated as
``hosting: {kind: endpoint, url: <server_url>}`` with a deprecation warning, and
``hosting: {kind: endpoint}`` with neither ``endpoint`` nor ``url`` resolves from
``CUDAGYM_UNIFIED_SERVER_URL`` / ``CUDAGYM_URL`` (the environment-variable
escape hatch).

Kept dependency-light on purpose (omegaconf + stdlib; ``requests`` imported
lazily in ``probe_endpoint``): the recipe loader is the hydra-free
``nemo_rl.utils.config_inheritance``, shared with the training-side loader.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from omegaconf import DictConfig, OmegaConf

from nemo_rl.environments.atlas.cuda_kernel_utils import (  # noqa: F401  (re-exported for submit-time callers)
    verify_health_payload,
)
from nemo_rl.utils.config_inheritance import load_config_with_inheritance

REPO_ROOT = Path(__file__).parent.parent
ENDPOINTS_DIR = REPO_ROOT / "endpoints"

IN_ALLOCATION_KINDS = ("colocated", "disjoint")
KINDS = ("colocated", "disjoint", "endpoint", "slurm-service")

# provider (endpoints/<provider>.yaml stem) -> env vars that MUST be set at submit.
# Modal's edge proxy rejects requests without the workspace proxy-token headers.
PROVIDER_REQUIRED_ENV: dict[str, tuple[str, ...]] = {
    "modal": ("MODAL_PROXY_TOKEN_ID", "MODAL_PROXY_TOKEN_SECRET"),
}


# cluster yaml `sku:` (lowercase) -> the kernel-target SKU that silicon serves.
# A small hand-maintained table so this check works on machines without the
# cudagym SDK installed (the SDK's _hardware_match_keys encodes the same
# aliasing). The one fact worth restating is that GB200 superchips serve B200
# kernels; add a row per new cluster silicon.
CLUSTER_SILICON: dict[str, str] = {
    "h100": "H100",
    "h200": "H200",
    "gb200": "B200",
    "b200": "B200",
}

EXAMPLE_HOSTING_BLOCK = (
    "      hosting:\n"
    "        kind: endpoint            # colocated | disjoint | endpoint | slurm-service\n"
    "        endpoint: modal/b200      # or `url: https://...`; see endpoints/*.yaml"
)


class HostingError(ValueError):
    """A hosting declaration failed validation (message is user-facing)."""


def load_recipe_merged(
    config_path: Union[str, Path], base_dir: Optional[Union[str, Path]] = None
) -> DictConfig:
    """Load a recipe YAML, following its ``defaults:`` inheritance chain."""
    return load_config_with_inheritance(config_path, base_dir)


# --------------------------------------------------------------------------
# Endpoint registry (endpoints/<provider>.yaml).
# --------------------------------------------------------------------------


@dataclass
class EndpointEntry:
    """One eval endpoint from the ``endpoints/<provider>.yaml`` registry."""

    name: str  # "<provider>/<key>", e.g. "modal/b200"
    provider: str
    sku: str
    url: str
    disabled_reason: Optional[str] = None
    auth_token_env: Optional[str] = None


def load_endpoints(endpoints_dir: Path = ENDPOINTS_DIR) -> dict[str, EndpointEntry]:
    """Read every ``endpoints/<provider>.yaml`` into a ``provider/key`` map."""
    entries: dict[str, EndpointEntry] = {}
    for path in sorted(endpoints_dir.glob("*.yaml")):
        provider = path.stem
        data = OmegaConf.to_container(OmegaConf.load(path), resolve=True) or {}
        if not isinstance(data, dict):
            raise HostingError(f"{path} must be a mapping of endpoint entries")
        for key, val in data.items():
            if not isinstance(val, dict) or "url" not in val or "sku" not in val:
                raise HostingError(
                    f"{path}: entry '{key}' must be a mapping with at least 'sku' and 'url'"
                )
            name = f"{provider}/{key}"
            entries[name] = EndpointEntry(
                name=name,
                provider=provider,
                sku=str(val["sku"]),
                url=str(val["url"]).rstrip("/"),
                disabled_reason=val.get("disabled_reason"),
                auth_token_env=val.get("auth_token_env"),
            )
    return entries


# --------------------------------------------------------------------------
# Resolution + validation.
# --------------------------------------------------------------------------


@dataclass
class ResolvedEntry:
    """One validated ``env.cudagym.<name>`` entry with its hosting resolved."""

    name: str  # the env.cudagym key
    sku: str  # normalized upper-case
    kind: str
    url: str = ""  # kind=endpoint: the resolved URL
    endpoint: Optional[EndpointEntry] = None  # kind=endpoint via registry
    num_nodes: int = 0  # kind=disjoint
    service: dict[str, Any] = field(default_factory=dict)  # kind=slurm-service


@dataclass
class HostingResolution:
    """All validated ``env.cudagym`` entries of a recipe, grouped by hosting kind."""

    entries: list[ResolvedEntry]
    in_allocation: Optional[ResolvedEntry]
    endpoints: list[ResolvedEntry]
    slurm_services: list[ResolvedEntry]
    extra_config_opts: list[str]
    unified_server_url: str  # single-endpoint jobs: the URL; else ""
    warnings: list[str]

    @property
    def cudagym_mode(self) -> str:
        """The in-allocation hosting kind, filled into ray.sub's CUDAGYM_MODE ('' when none)."""
        return self.in_allocation.kind if self.in_allocation else ""

    @property
    def cudagym_num_nodes(self) -> int:
        """Node count for ray.sub's CUDAGYM_NUM_NODES (0 unless kind is disjoint)."""
        return self.in_allocation.num_nodes if self.in_allocation else 0


def _auth_env_names(entry: ResolvedEntry) -> tuple[str, ...]:
    """Return the env vars the entry's provider requires in the submitting shell."""
    if entry.endpoint is None:
        return ()
    return PROVIDER_REQUIRED_ENV.get(entry.endpoint.provider, ())


def resolve_hosting(
    recipe_cfg: DictConfig,
    cluster_cfg: Any,
    num_nodes: int,
    uses_nemo_gym: bool,
    endpoints_dir: Path = ENDPOINTS_DIR,
) -> HostingResolution:
    """Validate every ``env.cudagym.<name>.hosting`` declaration and resolve URLs.

    Raises ``HostingError`` with a user-facing message on any invalid
    combination; see the module docstring for the schema.
    """
    warnings: list[str] = []
    selected = OmegaConf.select(recipe_cfg, "env.cudagym")
    raw = {} if selected is None else OmegaConf.to_container(selected, resolve=True)
    if not isinstance(raw, dict):
        raise HostingError("env.cudagym must be a mapping of per-SKU entries")

    registry: Optional[dict[str, EndpointEntry]] = None
    resolved: list[ResolvedEntry] = []
    for name, entry in raw.items():
        if not isinstance(entry, dict):
            raise HostingError(f"env.cudagym.{name} must be a mapping")
        sku = str(entry.get("sku") or name).upper()

        hosting = entry.get("hosting")
        if hosting is None:
            if entry.get("server_url"):
                warnings.append(
                    f"env.cudagym.{name}: bare `server_url:` is deprecated — declare "
                    f"`hosting: {{kind: endpoint, url: ...}}` instead."
                )
                hosting = {"kind": "endpoint", "url": entry["server_url"]}
            else:
                raise HostingError(
                    f"env.cudagym.{name} declares no `hosting:` block. Every entry must "
                    f"say where its eval servers live, e.g.:\n{EXAMPLE_HOSTING_BLOCK}"
                )
        if not isinstance(hosting, dict):
            raise HostingError(f"env.cudagym.{name}.hosting must be a mapping")
        kind = hosting.get("kind")
        if kind not in KINDS:
            raise HostingError(
                f"env.cudagym.{name}.hosting.kind={kind!r} — expected one of {list(KINDS)}"
            )

        if kind == "colocated":
            resolved.append(ResolvedEntry(name=name, sku=sku, kind=kind))
        elif kind == "disjoint":
            n = int(hosting.get("num_nodes") or 0)
            if not (1 <= n < num_nodes):
                raise HostingError(
                    f"env.cudagym.{name}.hosting.num_nodes must be in [1, --num-nodes) "
                    f"(got {n} with --num-nodes={num_nodes})"
                )
            resolved.append(ResolvedEntry(name=name, sku=sku, kind=kind, num_nodes=n))
        elif kind == "endpoint":
            ref = hosting.get("endpoint")
            url = hosting.get("url")
            if ref and url:
                raise HostingError(
                    f"env.cudagym.{name}.hosting: `endpoint` and `url` are mutually exclusive"
                )
            ep: Optional[EndpointEntry] = None
            if ref:
                if registry is None:
                    registry = load_endpoints(endpoints_dir)
                ep = registry.get(str(ref))
                if ep is None:
                    raise HostingError(
                        f"env.cudagym.{name}.hosting.endpoint={ref!r} not found in "
                        f"{endpoints_dir}/*.yaml (known: {', '.join(sorted(registry)) or 'none'})"
                    )
                if ep.disabled_reason:
                    raise HostingError(
                        f"endpoint {ep.name} is disabled: {ep.disabled_reason} "
                        f"(use an inline `url:` to override deliberately)"
                    )
                if ep.sku.upper() != sku:
                    raise HostingError(
                        f"env.cudagym.{name} declares sku {sku} but endpoint {ep.name} "
                        f"serves {ep.sku}"
                    )
                url = ep.url
            if not url:
                url = os.environ.get("CUDAGYM_UNIFIED_SERVER_URL") or os.environ.get(
                    "CUDAGYM_URL"
                )
                if not url:
                    raise HostingError(
                        f"env.cudagym.{name}.hosting has neither `endpoint` nor `url`, and "
                        f"CUDAGYM_UNIFIED_SERVER_URL is not set (the escape hatch)."
                    )
                warnings.append(
                    f"env.cudagym.{name}: endpoint URL taken from the environment ({url})."
                )
            entry_resolved = ResolvedEntry(
                name=name, sku=sku, kind=kind, url=str(url).rstrip("/"), endpoint=ep
            )
            missing = [
                v for v in _auth_env_names(entry_resolved) if not os.environ.get(v)
            ]
            if missing:
                raise HostingError(
                    f"endpoint {ep.name if ep else url} requires env var(s) "
                    f"{', '.join(missing)} to be set in the submitting shell "
                    f"(provider '{ep.provider if ep else '?'}')."
                )
            resolved.append(entry_resolved)
        else:  # slurm-service
            required = ("service_cluster", "num_service_nodes", "endpoint_port")
            missing_fields = [f for f in required if not hosting.get(f)]
            if missing_fields:
                raise HostingError(
                    f"env.cudagym.{name}.hosting (slurm-service) missing fields: "
                    f"{', '.join(missing_fields)}"
                )
            service = {
                "service_cluster": hosting["service_cluster"],
                "num_service_nodes": int(hosting["num_service_nodes"]),
                "endpoint_port": int(hosting["endpoint_port"]),
                "service_login_port": int(hosting.get("service_login_port") or 8998),
            }
            if str(service["service_cluster"]) == str(cluster_cfg.get("host") or ""):
                raise HostingError(
                    f"env.cudagym.{name}.hosting (slurm-service) points at the submit "
                    f"cluster itself — use hosting kind 'colocated' or 'disjoint' instead."
                )
            warnings.append(
                f"env.cudagym.{name}: hosting kind 'slurm-service' is experimental "
                f"(pending live validation of the refreshed cudagym slurm deployment)."
            )
            resolved.append(
                ResolvedEntry(name=name, sku=sku, kind=kind, service=service)
            )

    in_alloc = [e for e in resolved if e.kind in IN_ALLOCATION_KINDS]
    if len(in_alloc) > 1:
        names = ", ".join(f"{e.name}({e.kind})" for e in in_alloc)
        raise HostingError(
            f"at most one env.cudagym entry may be hosted in-allocation per job; got: {names}"
        )
    if in_alloc:
        cluster_sku = str(cluster_cfg.get("sku") or "").lower()
        silicon = CLUSTER_SILICON.get(cluster_sku, cluster_sku.upper())
        if silicon != in_alloc[0].sku:
            raise HostingError(
                f"env.cudagym.{in_alloc[0].name} wants {in_alloc[0].kind} hosting for sku "
                f"{in_alloc[0].sku}, but cluster '{cluster_cfg.get('host', '?')}' has "
                f"{cluster_sku or 'unknown'} silicon (serves {silicon or 'unknown'})."
            )

    endpoints = [e for e in resolved if e.kind == "endpoint"]
    slurm_services = [e for e in resolved if e.kind == "slurm-service"]

    if uses_nemo_gym:
        if len(resolved) != 1:
            raise HostingError(
                f"agentic (NeMo-Gym) recipes must declare exactly one env.cudagym entry "
                f"(the resources server speaks one endpoint); got {len(resolved)}."
            )
        if in_alloc:
            warnings.append(
                "Agentic recipe with in-allocation hosting: kernel eval time-shares the "
                "training GPUs, so timing (the performance reward) is unreliable and clocks "
                "can't be locked. OK for smoke tests; use an endpoint for real runs."
            )

    extra_opts = [f"++env.cudagym.{e.name}.server_url={e.url}" for e in endpoints]
    unified = endpoints[0].url if len(endpoints) == 1 else ""

    return HostingResolution(
        entries=resolved,
        in_allocation=in_alloc[0] if in_alloc else None,
        endpoints=endpoints,
        slurm_services=slurm_services,
        extra_config_opts=extra_opts,
        unified_server_url=unified,
        warnings=warnings,
    )


# --------------------------------------------------------------------------
# Preflight probe + SKU verification.
# --------------------------------------------------------------------------


def _probe_headers(entry: ResolvedEntry) -> dict[str, str]:
    """Return auth headers for a /health request, mirroring what the cudagym SDK sends."""
    headers: dict[str, str] = {}
    token_env = (
        entry.endpoint.auth_token_env if entry.endpoint else None
    ) or "CUDAGYM_AUTH_TOKEN"
    token = os.environ.get(token_env)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if entry.endpoint and entry.endpoint.provider == "modal":
        headers["Modal-Key"] = os.environ.get("MODAL_PROXY_TOKEN_ID", "")
        headers["Modal-Secret"] = os.environ.get("MODAL_PROXY_TOKEN_SECRET", "")
    return headers


def probe_endpoint(
    entry: ResolvedEntry, timeout: float = 10.0, retries: int = 2
) -> dict[str, Any]:
    """GET ``<url>/health`` with provider auth headers.

    Returns the decoded JSON payload. Raises ``HostingError`` when the endpoint
    stays unreachable across retries or answers with an HTTP error.
    """
    try:
        import requests
    except ImportError as e:  # pragma: no cover - environment guard
        raise HostingError(
            "the `requests` package is required for endpoint preflight "
            "(pip install requests, or pass --skip-endpoint-check)"
        ) from e

    url = f"{entry.url}/health"
    headers = _probe_headers(entry)
    last_error = ""
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            if resp.status_code == 200:
                payload = resp.json()
                if not isinstance(payload, dict):
                    raise HostingError(
                        f"{url} returned non-object JSON: {str(payload)[:120]}"
                    )
                return payload
            last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
        except HostingError:
            raise
        except Exception as e:  # noqa: BLE001 - report the transport failure verbatim
            last_error = str(e)
    raise HostingError(f"endpoint {entry.name} unreachable at {url}: {last_error}")


def check_registry_against_solswarm(
    endpoints: dict[str, EndpointEntry], solswarm_root: Union[str, Path]
) -> list[str]:
    """Compare the endpoint registry against SolSwarm's ``gpu-skus.toml``.

    Both files describe the same managed eval fleets but are maintained
    separately, so a fleet redeploy can update one without the other. The toml
    is an array of tables: ``[[gpu_skus]]`` with ``id`` (the lowercase SKU) and
    ``cudagym_url`` (the unified endpoint our registry stores). Returns
    human-readable difference lines, empty when the two agree. Callers warn
    rather than fail, since a locally-overridden URL can be a deliberate choice
    worth flagging, not blocking.
    """
    toml_path = Path(solswarm_root) / "deployments" / "files" / "gpu-skus.toml"
    if not toml_path.is_file():
        return []
    try:
        import tomllib

        upstream = tomllib.loads(toml_path.read_text())
    except Exception:  # noqa: BLE001 - a drift check must never break a submit
        return []

    upstream_urls: dict[str, str] = {}
    for sku_entry in upstream.get("gpu_skus") or []:
        if (
            isinstance(sku_entry, dict)
            and sku_entry.get("id")
            and sku_entry.get("cudagym_url")
        ):
            upstream_urls[str(sku_entry["id"]).lower()] = str(
                sku_entry["cudagym_url"]
            ).rstrip("/")

    lines = []
    for name, entry in (endpoints or {}).items():
        url = (getattr(entry, "url", "") or "").rstrip("/")
        key = str(getattr(entry, "sku", name)).lower()
        if key in upstream_urls and url and url != upstream_urls[key]:
            lines.append(f"{name}: ours={url} solswarm={upstream_urls[key]}")
    return lines
