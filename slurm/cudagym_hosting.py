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
to this one (``slurm.deploy_remote_cudagym``); agentic (NeMo-Gym) recipes
refuse this kind because the deployed URL is not plumbed to the Gym servers.

``hosting: {kind: endpoint}`` with neither ``endpoint`` nor ``url`` resolves
from ``CUDAGYM_UNIFIED_SERVER_URL`` / ``CUDAGYM_URL`` (the environment-variable
escape hatch).

Kept dependency-light on purpose (omegaconf + stdlib; ``requests`` imported
lazily in ``probe_endpoint``): the recipe loader is the hydra-free
``nemo_rl.utils.config_inheritance``, shared with the training-side loader.
The one exception is the GPU-identity preflight, which needs the cudagym SDK's
device table: ``ensure_vendored_cudagym`` imports it from the repo's own
``3rdparty/cudagym`` checkout (adding only ``loguru`` + ``pydantic`` to the
requirements) and fails with instructions rather than skipping the check.
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from omegaconf import DictConfig, OmegaConf

from nemo_rl.environments.atlas.cuda_kernel_utils import (  # noqa: F401  (re-exported for submit-time callers)
    canonical_sku,
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
# cudagym SDK installed; add a row per new cluster silicon. Each silicon maps
# to itself: CudaGym models B200 and GB200 as different hardware because it
# locks B200 clocks to 1500 MHz and leaves GB200 unlocked, so the same kernel
# times differently on the two. Mapping one onto the other here would also
# contradict the runtime check against the server's reported GPU, which would
# accept the job at submit and then refuse it at environment init.
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
    missing = [v for v in _auth_env_names(entry) if not os.environ.get(v)]
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

    # Group by kind for the submit-side callers: endpoints get the /health
    # preflight, slurm-service entries get deployed.
    endpoints = [e for e in resolved if e.kind == "endpoint"]
    slurm_services = [e for e in resolved if e.kind == "slurm-service"]

    # Agentic (NeMo-Gym) rules: exactly one entry, slurm-service hosting is
    # refused (its URL never reaches the Gym servers), and in-allocation
    # hosting only warns — usable for smoke tests, wrong for timed rewards.
    if uses_nemo_gym:
        if len(resolved) != 1:
            raise HostingError(
                f"agentic (NeMo-Gym) recipes must declare exactly one env.cudagym entry "
                f"(the resources server speaks one endpoint); got {len(resolved)}."
            )
        if slurm_services:
            # The deployed service's URL travels only as a
            # ++env.cudagym.<name>.server_url training-config override, which
            # only the single-turn env actor reads; the NeMo-Gym servers take
            # their endpoint from CUDAGYM_UNIFIED_SERVER_URL, which only
            # kind=endpoint entries fill. Allowing the combination would bring
            # the job up with a dead evaluation endpoint.
            raise HostingError(
                f"env.cudagym.{slurm_services[0].name}: hosting kind 'slurm-service' cannot "
                f"serve an agentic (NeMo-Gym) recipe — the deployed service URL is not "
                f"plumbed to the Gym servers (they read CUDAGYM_UNIFIED_SERVER_URL, which "
                f"only kind=endpoint fills), so kernel evaluation would silently point at "
                f"nothing. Use `kind: endpoint` (an already-running eval server) or "
                f"in-allocation hosting (`kind: colocated` / `kind: disjoint`) instead."
            )
        if in_alloc:
            warnings.append(
                "Agentic recipe with in-allocation hosting: kernel eval time-shares the "
                "training GPUs, so timing (the performance reward) is unreliable and clocks "
                "can't be locked. OK for smoke tests; use an endpoint for real runs."
            )

    # Each endpoint URL rides into the training config as a ++server_url override;
    # only a single-endpoint job also gets the ambient CUDAGYM_UNIFIED_SERVER_URL.
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
# Submit-time /health preflight + SKU verification.
# --------------------------------------------------------------------------


def _cudagym_import_error() -> Optional[str]:
    """Try to import the cudagym SDK; return the ImportError message, or None on success."""
    try:
        import cudagym  # noqa: F401
    except ImportError as e:
        return str(e)
    return None


def ensure_vendored_cudagym(repo_root: Union[str, Path] = REPO_ROOT) -> None:
    """Make the cudagym SDK importable, falling back to the vendored checkout.

    The SKU preflight derives its expectations from the SDK's device table
    (``sku_expectations`` in ``cuda_kernel_utils``); without the import it
    could only report "unverifiable", and an unverified GPU identity is how a
    silicon mismatch stays silent. The SDK ships in this repo at
    ``3rdparty/cudagym`` — the same checkout the job's ``uv sync`` installs, so
    a submit without it would fail at job start anyway — and its import chain
    needs only ``loguru`` and ``pydantic`` beyond the stdlib. Machines without
    the training venv therefore import it straight from the checkout; when even
    that is impossible, this raises ``HostingError`` naming the fix instead of
    letting the check degrade.
    """
    # Already importable (e.g. the training venv) — nothing to do.
    if _cudagym_import_error() is None:
        return
    # Fall back to the vendored submodule checkout, which must be initialized.
    src = Path(repo_root) / "3rdparty" / "cudagym" / "src"
    if not (src / "cudagym" / "__init__.py").is_file():
        raise HostingError(
            "the cudagym SDK is not importable and the vendored checkout is missing "
            f"({src}); initialize it with `git submodule update --init 3rdparty/cudagym`"
        )
    # Put the checkout on sys.path and try the import again.
    sys.path.insert(0, str(src))
    error = _cudagym_import_error()
    if error is not None:
        # Still failing (the SDK's own deps are missing): undo the path edit so
        # a failed preflight leaves sys.path untouched, then name the fix.
        sys.path.remove(str(src))
        raise HostingError(
            f"the cudagym SDK is not importable even from the vendored checkout ({src}): "
            f"{error}. The GPU-identity preflight needs the SDK's device table; install "
            "the import chain's two non-stdlib dependencies: `pip install loguru pydantic`"
        )


def _probe_headers(entry: ResolvedEntry) -> dict[str, str]:
    """Return auth headers for a /health request, mirroring what the cudagym SDK sends."""
    headers: dict[str, str] = {}
    token_env = (
        entry.endpoint.auth_token_env if entry.endpoint else None
    ) or "CUDAGYM_AUTH_TOKEN"
    token = os.environ.get(token_env)
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
        # "a100-40gb"), so match on the key and fall back to the sku only for
        # custom-named entries. Keying on the sku would miss entries whose SDK
        # enum name differs from the fleet id (astra/gb10 carries DGX_SPARK)
        # and cross-compare entries that share a sku (the three A100 variants).
        key = name.split("/", 1)[-1].lower()
        if key not in upstream_urls:
            key = entry.sku.lower()
        if key in upstream_urls and url and url != upstream_urls[key]:
            lines.append(f"{name}: ours={url} solswarm={upstream_urls[key]}")
    return lines
