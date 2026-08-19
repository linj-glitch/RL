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

``submit_grpo.py`` calls ``resolve_hosting`` at submit time: every entry is
validated, registry refs (``endpoints/<provider>.yaml``) resolve to URLs, and
remote endpoints get a /health GPU preflight. ``hosting: {kind: endpoint}``
with neither ``endpoint`` nor ``url`` falls back to the
CUDAGYM_UNIFIED_SERVER_URL / CUDAGYM_URL environment variables.

The module must import without the training venv: it uses omegaconf, the
stdlib, and the CudaGym SDK, which ``ensure_vendored_cudagym`` bootstraps from
the ``3rdparty/solswarm/cudagym`` tree at import time.
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from omegaconf import DictConfig, OmegaConf

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


# cluster yaml `sku:` (lowercase, this repo's own spelling) -> the kernel-target
# SKU that silicon serves; add a row per new cluster silicon. B200 and GB200
# stay distinct: CudaGym locks B200 clocks and leaves GB200 unlocked, so the
# same kernel times differently on the two.
CLUSTER_SILICON: dict[str, str] = {
    "h100": "H100",
    "h200": "H200",
    "gb200": "GB200",
    "b200": "B200",
}

EXAMPLE_HOSTING_BLOCK = (
    "      hosting:\n"
    "        kind: endpoint            # colocated | disjoint | endpoint | slurm-service\n"
    "        endpoint: modal/b200      # or `url: https://...`; see endpoints/*.yaml"
)


class HostingError(ValueError):
    """A hosting declaration failed validation (message is user-facing)."""


# --------------------------------------------------------------------------
# CudaGym SDK bootstrap. Both SKU checks below come from the SDK, so it has to
# be importable before ``from cudagym.rl import ...`` runs — hence the call at
# module scope rather than at the first use.
# --------------------------------------------------------------------------


def _cudagym_import_error() -> Optional[str]:
    """Try to import ``cudagym.rl``; return the ImportError message, or None on success.

    ``cudagym.rl`` (not the bare namespace) is what this module goes on to import.
    """
    try:
        import cudagym.rl  # noqa: F401
    except ImportError as e:
        return str(e)
    return None


def ensure_vendored_cudagym(repo_root: Union[str, Path] = REPO_ROOT) -> None:
    """Make the cudagym SDK importable, falling back to the vendored checkout.

    The SKU checks (``canonical_sku``, ``verify_health_payload``) are the SDK's
    own ``cudagym.rl`` helpers, so the SDK must import before this module can
    validate anything. Submit hosts usually lack the training venv; the
    ``3rdparty/solswarm/cudagym`` tree needs only ``loguru`` and ``pydantic``.
    Raises ``HostingError`` naming the fix when even that import fails.
    """
    # Already importable (e.g. the training venv) — nothing to do.
    if _cudagym_import_error() is None:
        return
    # Fall back to the vendored submodule checkout, which must be initialized.
    src = Path(repo_root) / "3rdparty" / "solswarm" / "cudagym" / "src"
    if not (src / "cudagym" / "__init__.py").is_file():
        raise HostingError(
            "the cudagym SDK is not importable and the vendored checkout is missing "
            f"({src}); initialize it with `git submodule update --init 3rdparty/solswarm`"
        )
    # An installed cudagym that predates the cudagym.rl helpers imports fine as
    # a parent package, so the probe leaves it bound in sys.modules and the
    # retry below would resolve `rl` against that same stale package's __path__
    # and fail again. Drop the cached modules so the vendored checkout is what
    # the retry actually imports.
    shadowed = "cudagym" in sys.modules
    stale = [
        name for name in sys.modules if name == "cudagym" or name.startswith("cudagym.")
    ]
    cached = {name: sys.modules.pop(name) for name in stale}
    # Put the checkout on sys.path and try the import again.
    sys.path.insert(0, str(src))
    error = _cudagym_import_error()
    if error is not None:
        # Still failing: undo both edits so a failed preflight leaves the
        # process as it found it, then name the fix.
        sys.path.remove(str(src))
        sys.modules.update(cached)
        # Two different causes, two different fixes: an installed cudagym that
        # shadows the checkout without carrying cudagym.rl, versus a checkout
        # whose own import chain has nothing to import from.
        missing_dep = "cudagym" not in str(error)
        remedy = (
            "install the import chain's two non-stdlib dependencies: `pip install loguru pydantic`"
            if missing_dep or not shadowed
            else "the installed cudagym predates the cudagym.rl helpers; update or remove it"
        )
        raise HostingError(
            f"the cudagym SDK is not importable even from the vendored checkout ({src}): "
            f"{error}. The SKU checks need the SDK's device table; {remedy}"
        )


ensure_vendored_cudagym()

# Deliberately below the bootstrap call, which is what makes this import work on
# a host without the SDK installed. verify_health_payload is re-exported for
# submit-time callers.
from cudagym.rl import canonical_sku, verify_health_payload  # noqa: E402, F401


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


def load_endpoints(endpoints_dir: Path = ENDPOINTS_DIR) -> dict[str, EndpointEntry]:
    """Read every ``endpoints/<provider>.yaml`` into a ``provider/key`` map."""
    entries: dict[str, EndpointEntry] = {}
    # One yaml per provider; the file stem becomes the "<provider>/" name prefix.
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
            try:
                sku = canonical_sku(val["sku"], f"{path}: entry '{key}' sku")
            except ValueError as e:
                raise HostingError(str(e)) from e
            entries[name] = EndpointEntry(
                name=name,
                provider=provider,
                sku=sku,
                url=str(val["url"]).rstrip("/"),
                disabled_reason=val.get("disabled_reason"),
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
    # SKU -> endpoint URL, one key per entry; in-allocation entries map to ""
    # (the Gym server then reads CUDAGYM_UNIFIED_SERVER_URL at runtime).
    sku_endpoints: dict[str, str]
    # The one resolved URL when the job has exactly one endpoint, else "";
    # exported into the job as the ambient CUDAGYM_UNIFIED_SERVER_URL.
    unified_server_url: str
    warnings: list[str]

    @property
    def cudagym_mode(self) -> str:
        """The in-allocation hosting kind, filled into ray.sub's CUDAGYM_MODE ('' when none)."""
        return self.in_allocation.kind if self.in_allocation else ""

    @property
    def cudagym_num_nodes(self) -> int:
        """Node count for ray.sub's CUDAGYM_NUM_NODES (0 unless kind is disjoint)."""
        return self.in_allocation.num_nodes if self.in_allocation else 0


def _resolve_disjoint(
    name: str, sku: str, hosting: dict, num_nodes: int
) -> ResolvedEntry:
    """Validate a ``kind: disjoint`` entry: reserve eval nodes out of the allocation.

    Requires ``num_nodes`` in ``[1, --num-nodes)`` so at least one node remains
    for training; ray.sub carves the trailing nodes off the Ray cluster.
    """
    n = int(hosting.get("num_nodes") or 0)
    if not (1 <= n < num_nodes):
        raise HostingError(
            f"env.cudagym.{name}.hosting.num_nodes must be in [1, --num-nodes) "
            f"(got {n} with --num-nodes={num_nodes})"
        )
    return ResolvedEntry(name=name, sku=sku, kind="disjoint", num_nodes=n)


def _resolve_endpoint(
    name: str,
    sku: str,
    hosting: dict,
    registry: dict[str, EndpointEntry],
    warnings: list[str],
) -> ResolvedEntry:
    """Validate a ``kind: endpoint`` entry: an already-running eval server.

    The server is named by exactly one of a registry ref (``endpoint:
    <provider>/<key>``, resolved through ``endpoints/*.yaml`` and refused when
    disabled or SKU-mismatched) or an inline ``url:``; with neither, the URL
    comes from the ``CUDAGYM_UNIFIED_SERVER_URL`` / ``CUDAGYM_URL`` escape
    hatch (recorded as a warning). Providers with required auth must have their
    env vars set in the submitting shell — the /health preflight and the job
    both send them.
    """
    ref = hosting.get("endpoint")
    url = hosting.get("url")
    if ref and url:
        raise HostingError(
            f"env.cudagym.{name}.hosting: `endpoint` and `url` are mutually exclusive"
        )
    ep: Optional[EndpointEntry] = None
    if ref:
        # Registry ref: take that entry's URL; refuse disabled or SKU-mismatched entries.
        ep = registry.get(str(ref))
        if ep is None:
            raise HostingError(
                f"env.cudagym.{name}.hosting.endpoint={ref!r} not found in "
                f"endpoints/*.yaml (known: {', '.join(sorted(registry)) or 'none'})"
            )
        if ep.disabled_reason:
            raise HostingError(
                f"endpoint {ep.name} is disabled: {ep.disabled_reason} "
                f"(use an inline `url:` to override deliberately)"
            )
        if ep.sku.upper() != sku:
            raise HostingError(
                f"env.cudagym.{name} declares sku {sku} but endpoint {ep.name} serves {ep.sku}"
            )
        url = ep.url
    if not url:
        # Neither ref nor url: fall back to the environment-variable escape hatch, and say so.
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
    entry = ResolvedEntry(
        name=name, sku=sku, kind="endpoint", url=str(url).rstrip("/"), endpoint=ep
    )
    required_env = PROVIDER_REQUIRED_ENV.get(ep.provider, ()) if ep else ()
    missing = [v for v in required_env if not os.environ.get(v)]
    if missing:
        raise HostingError(
            f"endpoint {ep.name if ep else url} requires env var(s) "
            f"{', '.join(missing)} to be set in the submitting shell "
            f"(provider '{ep.provider if ep else '?'}')."
        )
    return entry


def _resolve_slurm_service(
    name: str, sku: str, hosting: dict, cluster_cfg: Any, warnings: list[str]
) -> ResolvedEntry:
    """Validate a ``kind: slurm-service`` entry: a service job on ANOTHER cluster.

    Requires the target cluster, its node count, and the submit-side proxy port
    (``endpoint_port``); ``service_login_port`` is the proxy port on the
    SERVICE cluster's login node. Deployment happens at submit time
    (``slurm/deploy_remote_cudagym.py``). Pointing at the submit cluster itself
    is refused — that is what the in-allocation kinds are for.
    """
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
    return ResolvedEntry(name=name, sku=sku, kind="slurm-service", service=service)


def resolve_hosting(
    recipe_cfg: DictConfig,
    cluster_cfg: Any,
    num_nodes: int,
    uses_nemo_gym: bool,
    endpoints_dir: Path = ENDPOINTS_DIR,
) -> HostingResolution:
    """Validate every ``env.cudagym.<name>.hosting`` declaration and resolve URLs.

    Dispatches each entry to its kind's validator (``_resolve_disjoint`` /
    ``_resolve_endpoint`` / ``_resolve_slurm_service``; colocated has nothing
    to validate), then applies the cross-entry rules. Raises ``HostingError``
    with a user-facing message on any invalid combination; see the module
    docstring for the schema.
    """
    warnings: list[str] = []
    # Pull the recipe's env.cudagym mapping; a recipe without one has no entries.
    selected = OmegaConf.select(recipe_cfg, "env.cudagym")
    raw = {} if selected is None else OmegaConf.to_container(selected, resolve=True)
    if not isinstance(raw, dict):
        raise HostingError("env.cudagym must be a mapping of per-SKU entries")

    # The endpoint registry is only read once some entry references it by name.
    registry: Optional[dict[str, EndpointEntry]] = None
    resolved: list[ResolvedEntry] = []
    for name, entry in raw.items():
        if not isinstance(entry, dict):
            raise HostingError(f"env.cudagym.{name} must be a mapping")
        # Declared, never inferred from the entry name, and spelled exactly as
        # CudaGym spells it: this string becomes Solution.spec.target_hardware.
        try:
            sku = canonical_sku(entry.get("sku"), f"env.cudagym.{name}.sku")
        except ValueError as e:
            raise HostingError(str(e)) from e

        hosting = entry.get("hosting")
        if hosting is None:
            # A bare `server_url:` looks like an endpoint declaration but is
            # not honored; when one is present, name it in the error so the
            # author learns where the URL belongs.
            legacy_hint = (
                " (a bare `server_url:` is not honored; put the URL in the hosting block)"
                if entry.get("server_url")
                else ""
            )
            raise HostingError(
                f"env.cudagym.{name} declares no `hosting:` block{legacy_hint}. Every entry "
                f"must say where its eval servers live, e.g.:\n{EXAMPLE_HOSTING_BLOCK}"
            )
        if not isinstance(hosting, dict):
            raise HostingError(f"env.cudagym.{name}.hosting must be a mapping")
        kind = hosting.get("kind")
        if kind not in KINDS:
            raise HostingError(
                f"env.cudagym.{name}.hosting.kind={kind!r} — expected one of {list(KINDS)}"
            )

        if kind == "colocated":
            # colocated: eval servers share the training nodes; nothing more to declare.
            resolved.append(ResolvedEntry(name=name, sku=sku, kind=kind))
        elif kind == "disjoint":
            resolved.append(_resolve_disjoint(name, sku, hosting, num_nodes))
        elif kind == "endpoint":
            if registry is None:
                registry = load_endpoints(endpoints_dir)
            resolved.append(_resolve_endpoint(name, sku, hosting, registry, warnings))
        else:  # slurm-service
            resolved.append(
                _resolve_slurm_service(name, sku, hosting, cluster_cfg, warnings)
            )

    # Cross-entry rule: ray.sub can stand up servers for at most one in-allocation
    # entry (CUDAGYM_MODE / CUDAGYM_NUM_NODES describe a single deployment).
    in_alloc = [e for e in resolved if e.kind in IN_ALLOCATION_KINDS]
    if len(in_alloc) > 1:
        names = ", ".join(f"{e.name}({e.kind})" for e in in_alloc)
        raise HostingError(
            f"at most one env.cudagym entry may be hosted in-allocation per job; got: {names}"
        )
    # In-allocation servers run on the cluster's own GPUs, so the declared SKU
    # must match the cluster silicon (per CLUSTER_SILICON).
    if in_alloc:
        cluster_sku = str(cluster_cfg.get("sku") or "").lower()
        silicon = CLUSTER_SILICON.get(cluster_sku, cluster_sku.upper())
        if silicon != in_alloc[0].sku:
            raise HostingError(
                f"env.cudagym.{in_alloc[0].name} wants {in_alloc[0].kind} hosting for sku "
                f"{in_alloc[0].sku}, but cluster '{cluster_cfg.get('host', '?')}' has "
                f"{cluster_sku or 'unknown'} silicon (serves {silicon or 'unknown'})."
            )

    # Cross-entry rule: downstream consumers key on the SKU, not on the entry
    # name — sku_endpoints below is a SKU-keyed map, and the agentic resources
    # server picks an endpoint by a task row's target hardware. Two entries for
    # the same GPU therefore have no defined winner, so refuse them by name.
    names_by_sku: dict[str, list[str]] = {}
    for e in resolved:
        names_by_sku.setdefault(e.sku, []).append(e.name)
    duplicated = {s: n for s, n in names_by_sku.items() if len(n) > 1}
    if duplicated:
        detail = "; ".join(
            f"{s} declared by {', '.join(n)}" for s, n in sorted(duplicated.items())
        )
        raise HostingError(
            f"every env.cudagym entry must declare a distinct sku; got {detail}"
        )

    # Distinct SKUs must also resolve to distinct addresses; the /health
    # preflight cannot catch a shared URL (each per-SKU probe of it passes).
    # Entries with no URL are exempt (at most one exists, per the rule above).
    names_by_url: dict[str, list[str]] = {}
    for e in resolved:
        if e.url:
            names_by_url.setdefault(e.url, []).append(f"{e.name} ({e.sku})")
    shared = {u: n for u, n in names_by_url.items() if len(n) > 1}
    if shared:
        detail = "; ".join(
            f"{u} shared by {', '.join(n)}" for u, n in sorted(shared.items())
        )
        raise HostingError(
            f"env.cudagym entries for different GPUs resolve to the same endpoint; got {detail}. "
            "One address cannot serve two SKUs: rows for one GPU would be timed on the other."
        )

    # Group by kind for the submit-side callers: endpoints get the /health
    # preflight, slurm-service entries get deployed.
    endpoints = [e for e in resolved if e.kind == "endpoint"]
    slurm_services = [e for e in resolved if e.kind == "slurm-service"]

    # Agentic (NeMo-Gym) rules: any number of `endpoint` entries (one per SKU)
    # is fine; slurm-service is refused (its URL never reaches the Gym servers);
    # in-allocation hosting only warns (fine for smoke tests, wrong for timing).
    if uses_nemo_gym:
        if not resolved:
            raise HostingError(
                "an agentic (NeMo-Gym) recipe must declare at least one env.cudagym "
                "entry: the resources server evaluates kernels on the endpoints those "
                "entries resolve to, and needs at least one of them."
            )
        if slurm_services:
            raise HostingError(
                f"env.cudagym.{slurm_services[0].name}: hosting kind 'slurm-service' cannot "
                f"serve an agentic (NeMo-Gym) recipe — the deployed service URL is resolved "
                f"after hosting validation and never reaches the Gym servers' per-SKU "
                f"endpoint map, so kernel evaluation would silently point at nothing. Use "
                f"`kind: endpoint` (an already-running eval server) or in-allocation hosting "
                f"(`kind: colocated` / `kind: disjoint`) instead."
            )
        if in_alloc:
            warnings.append(
                "Agentic recipe with in-allocation hosting: kernel eval time-shares the "
                "training GPUs, so timing (the performance reward) is unreliable and clocks "
                "can't be locked. OK for smoke tests; use an endpoint for real runs."
            )

    # Endpoint URLs ride into the training config as ++server_url overrides
    # (read by the single-turn env actors).
    extra_opts = [f"++env.cudagym.{e.name}.server_url={e.url}" for e in endpoints]
    # SKU -> URL map for the agentic path; entries with no submit-time URL
    # (in-allocation) map to "", resolved from CUDAGYM_UNIFIED_SERVER_URL at runtime.
    sku_endpoints = {e.sku: e.url for e in resolved}
    # Only a job with exactly one endpoint has a single URL to name, so only it
    # fills the ambient CUDAGYM_UNIFIED_SERVER_URL.
    unified = endpoints[0].url if len(endpoints) == 1 else ""

    return HostingResolution(
        entries=resolved,
        in_allocation=in_alloc[0] if in_alloc else None,
        endpoints=endpoints,
        slurm_services=slurm_services,
        extra_config_opts=extra_opts,
        sku_endpoints=sku_endpoints,
        unified_server_url=unified,
        warnings=warnings,
    )


# --------------------------------------------------------------------------
# Submit-time /health preflight. The payload it fetches is judged by the SDK's
# ``verify_health_payload``, imported above.
# --------------------------------------------------------------------------


def _probe_headers(entry: ResolvedEntry) -> dict[str, str]:
    """Return auth headers for a /health request, mirroring what the cudagym SDK sends."""
    headers: dict[str, str] = {}
    # CUDAGYM_AUTH_TOKEN is the same bearer token the job itself sends, so the
    # preflight exercises the same auth the rollouts will.
    token = os.environ.get("CUDAGYM_AUTH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    # Modal's edge proxy wants its own header pair on top of any bearer token.
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
    # Retry transport errors and non-200 answers alike, remembering the last
    # failure for the final message.
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
            # A malformed 200 payload is a server bug, not transient — no retry.
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
    # Without a solswarm checkout there is nothing to compare against.
    if not toml_path.is_file():
        return []
    try:
        import tomllib

        upstream = tomllib.loads(toml_path.read_text())
    except Exception:  # noqa: BLE001 - a drift check must never break a submit
        return []

    # Collect the upstream map: lowercase fleet id -> unified endpoint URL.
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

    # Report every registry entry whose URL disagrees with its upstream row.
    lines = []
    for name, entry in endpoints.items():
        url = entry.url.rstrip("/")
        # Registry keys mirror the upstream fleet ids (modal/a100-40gb <-> id
        # "a100-40gb"): match on the key, falling back to the sku for custom names.
        key = name.split("/", 1)[-1].lower()
        if key not in upstream_urls:
            key = entry.sku.lower()
        if key in upstream_urls and url and url != upstream_urls[key]:
            lines.append(f"{name}: ours={url} solswarm={upstream_urls[key]}")
    return lines
