# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""Run GRPO on a cluster.

Usage:
    python submit_grpo.py --exp-name <exp-name> --config <config-path> --cluster <cluster-name>

    # atlas cuda 8b (hosting comes from the recipe's env.cudagym.<sku>.hosting blocks;
    # see slurm/cudagym_hosting.py and endpoints/*.yaml)
    python submit_grpo.py --exp-name cuda_qwen3_8b --config grpo_cuda_qwen3-8b.yaml --cluster aws-iad-cs-002 --num-nodes 1
"""

import subprocess
import argparse
import os
import re
from pathlib import Path

from omegaconf import OmegaConf

from remote_utils import (
    SSHTunnel,
    package_code,
    get_available_clusters,
    get_available_configs,
    load_cluster_config,
    validate_cluster_paths,
    fill_template,
    upload_text_as_file,
)
from slurm.cudagym_hosting import (
    HostingError,
    check_registry_against_solswarm,
    ensure_vendored_cudagym,
    load_endpoints,
    load_recipe_merged,
    probe_endpoint,
    resolve_hosting,
    verify_health_payload,
)
from slurm.deploy_remote_cudagym import deploy_remote_cudagym

CONFIG_PATH = Path(__file__).parent / "examples" / "configs" / "recipes" / "atlas"
CLUSTER_CONFIG_PATH = Path(__file__).parent / "slurm" / "clusters"
SBATCH_TEMPLATE_PATH = Path(__file__).parent / "slurm" / "grpo" / "grpo.sh"


def _vendored_cudagym_version() -> str:
    """Return a PEP 440 version for the vendored cudagym, from its git metadata.

    setuptools-scm cannot derive a version from the uploaded tree (no .git), and
    a hand-maintained literal drifts silently on every submodule bump.
    """
    root = Path(__file__).parent / "3rdparty" / "cudagym"
    try:
        described = subprocess.run(
            ["git", "-C", str(root), "describe", "--tags", "--dirty"],
            capture_output=True, text=True, timeout=30,
        )
        raw = described.stdout.strip().lstrip("v") if described.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        raw = ""
    if not raw:
        print("⚠️  could not derive the vendored cudagym version; falling back to 0.0.0")
        return "0.0.0"
    # v2.2.3-19-g87aa3f6[-dirty] -> 2.2.3.post19+g87aa3f6 (setuptools-scm shape)
    parts = raw.split("-")
    if len(parts) >= 3:
        return f"{parts[0]}.post{parts[1]}+{parts[2]}"
    return parts[0]


def launch_jobs(
    ssh_tunnel, code_upload_path, interactive: bool = False, num_jobs: int = 1
):
    """Run the uploaded ``run.sh`` sbatch wrapper on the cluster, once per job."""
    for i in range(num_jobs):
        launch_cmd = f"cd {code_upload_path} && bash ../run.sh"
        if interactive:
            launch_cmd += " -i"

        print(f"🚀 Running sbatch script ({i + 1}/{num_jobs}): {launch_cmd}")
        rc, stdout, stderr = ssh_tunnel.run_command(launch_cmd)
        print(stdout)
        if stderr:
            # Submit-plugin advisories (e.g. the stale-data quota notice on
            # cw-dfw) arrive on stderr even when sbatch succeeds, so stderr
            # alone is not a failure signal.
            print(f"⚠️  sbatch stderr: {stderr.strip()}")
        # grpo.sh ends with `Submitted batch job <id>`; an empty id means
        # sbatch itself failed even if the wrapper exited 0.
        if rc != 0 or not re.search(r"Submitted batch job \d+", stdout):
            raise RuntimeError(
                f"Error running sbatch script (rc={rc}): {stderr.strip() or stdout.strip()}"
            )


def main():
    """Validate CudaGym hosting, upload the code and sbatch script, and submit."""
    parser = argparse.ArgumentParser()
    # exp-name and cluster deliberately have no defaults: a forgotten flag should
    # fail fast rather than silently submit to an unintended cluster.
    parser.add_argument("--exp-name", "-e", required=True, type=str)
    parser.add_argument(
        "--config",
        type=str,
        default="grpo_cuda_qwen3-8b.yaml",
        choices=get_available_configs(CONFIG_PATH, "grpo*.yaml", return_stems=False),
    )
    parser.add_argument(
        "--cluster",
        "-c",
        type=str,
        required=True,
        choices=get_available_clusters(CLUSTER_CONFIG_PATH),
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Launch the command interactively",
    )
    parser.add_argument(
        "--extra-config-opts", type=str, default="", help="Extra config options"
    )
    parser.add_argument(
        "--time", type=str, default="04:00:00", help="Time limit for the job"
    )
    parser.add_argument(
        "--num-nodes", type=int, default=1, help="Number of nodes for the job"
    )
    parser.add_argument(
        "--num-jobs",
        type=int,
        default=1,
        help="Number of jobs to run sequentially. This is useful for GRPO runs that take longer than 4 hours to run.",
    )
    parser.add_argument(
        "--only-upload",
        action="store_true",
        help="Only upload the code and sbatch script",
    )
    parser.add_argument(
        "--skip-commit-check",
        action="store_true",
        help="Skip the commit check",
    )
    parser.add_argument(
        "--proxy-timeout",
        type=int,
        default=1800,
        help="Seconds to wait for a slurm-service CudaGym deployment's proxies to report ready",
    )
    # CudaGym hosting is declared per SKU in the recipe (env.cudagym.<name>.hosting;
    # see slurm/cudagym_hosting.py). The removed flags are kept as hidden stubs so
    # passing one fails fast with a migration hint instead of argparse's
    # "unrecognized arguments" error.
    parser.add_argument("--cudagym-mode", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--cudagym-num-nodes", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--cudagym-url", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--enroot-agent-image",
        default="",
        help=(
            "Cluster path of a SolSwarm agent-image squashfs. Required by container-mode "
            "recipes (cudagym_cuda_agent_solswarm.yaml in the Gym config paths): each "
            "rollout runs inside a real instance of this image. Enables the enroot "
            "plumbing in grpo.sh (host enroot bind-mounted into the training container, "
            "CUDA_AGENT_ENROOT_IMAGE exported)."
        ),
    )
    parser.add_argument(
        "--skip-endpoint-check",
        action="store_true",
        help=(
            "Tolerate unreachable remote CudaGym endpoints at submit time (the "
            "/health preflight normally hard-fails). A reachable endpoint that "
            "reports the WRONG GPU still fails, and the in-job verify_endpoint_sku "
            "handshake stays active either way."
        ),
    )
    args = parser.parse_args()

    # A removed hosting flag was passed: fail with the migration hint.
    if any(v is not None for v in (args.cudagym_mode, args.cudagym_num_nodes, args.cudagym_url)):
        parser.error(
            "--cudagym-mode/--cudagym-url/--cudagym-num-nodes were removed: hosting is now "
            "declared per SKU in the recipe, e.g.\n"
            "  env.cudagym.b200.hosting: {kind: endpoint, endpoint: modal/b200}\n"
            "kinds: colocated | disjoint (num_nodes: N) | endpoint; "
            "registry: endpoints/*.yaml; escape hatch: export CUDAGYM_UNIFIED_SERVER_URL "
            "and declare hosting: {kind: endpoint}."
        )

    # Load cluster config with env overrides applied and resolved
    cluster_config = load_cluster_config(CLUSTER_CONFIG_PATH, args.cluster)

    # Guard against mounting home directories which can slow down clusters
    validate_cluster_paths(cluster_config["paths"])

    # Resolve + validate the recipe's per-SKU CudaGym hosting declarations BEFORE
    # any code upload, so misconfigurations fail in seconds. The recipe is loaded
    # with its `defaults:` chain merged, so inherited env.cudagym blocks count too,
    # and the user's --extra-config-opts key=value overrides are applied so e.g.
    # `++env.cudagym.b200.hosting.kind=colocated` changes the RESOLVED hosting,
    # not just the training-time config.
    recipe_cfg = load_recipe_merged(CONFIG_PATH / args.config)

    # Container mode needs BOTH sides: the SolSwarm overlay in the recipe and
    # the image-path flag (the grpo.sh enroot plumbing). A mismatch otherwise
    # surfaces only at agent-server startup, minutes into the job. The overlay
    # is recognized by its config-path name because the Gym-side YAML is
    # merged by Gym, not here.
    gym_config_paths = [str(p) for p in (OmegaConf.select(recipe_cfg, "env.nemo_gym.config_paths") or [])]
    is_container_mode = any("cudagym_cuda_agent_solswarm" in p for p in gym_config_paths)
    if args.enroot_agent_image and not is_container_mode:
        raise SystemExit(
            "❌ --enroot-agent-image is set, but the recipe's env.nemo_gym.config_paths does not "
            "include cudagym_cuda_agent_solswarm.yaml — nothing in this job runs agent containers."
        )
    if is_container_mode and not args.enroot_agent_image:
        raise SystemExit(
            "❌ the recipe is container mode (cudagym_cuda_agent_solswarm.yaml): each rollout runs "
            "inside a real agent-image instance, so --enroot-agent-image <cluster .sqsh path> is "
            "required — without it the agent server fails its startup validation."
        )
    cli_overrides = [
        opt.lstrip("+") for opt in args.extra_config_opts.split() if "=" in opt
    ]
    if cli_overrides:
        recipe_cfg = OmegaConf.merge(recipe_cfg, OmegaConf.from_dotlist(cli_overrides))
    # Whether the recipe drives the agentic (NeMo-Gym) path: it changes the
    # hosting rules here and the runner + uv extras below.
    uses_nemo_gym = bool(
        OmegaConf.select(recipe_cfg, "env.should_use_nemo_gym", default=False)
    )
    try:
        hosting = resolve_hosting(
            recipe_cfg, cluster_config, args.num_nodes, uses_nemo_gym
        )
    except HostingError as e:
        raise SystemExit(f"❌ {e}") from e
    # Echo the resolved hosting table plus any validation warnings.
    print("🔎 CudaGym hosting:")
    for entry in hosting.entries:
        detail = f"kind={entry.kind}"
        if entry.url:
            detail += f" url={entry.url}"
        if entry.kind == "disjoint":
            detail += f" num_nodes={entry.num_nodes}"
        print(f"   - {entry.name}: sku={entry.sku} {detail}")
    for warning in hosting.warnings:
        print(f"⚠️  {warning}")

    # The endpoints/ registry was seeded from solswarm's gpu-skus.toml; the two
    # are maintained separately, so flag (never fail) when they disagree — a
    # mismatch usually means upstream moved the managed fleet and our URLs are stale.
    for line in check_registry_against_solswarm(
        load_endpoints(), Path(__file__).parent / "3rdparty" / "solswarm"
    ):
        print(f"⚠️  endpoints registry differs from solswarm gpu-skus.toml: {line}")

    # Preflight: ping every remote endpoint's /health and check the reported GPU
    # against the declared SKU. In-allocation servers don't exist yet — they get
    # the same check at runtime init (verify_endpoint_sku).
    if hosting.endpoints:
        # The GPU check needs the cudagym SDK's device table; without it the
        # verdict would silently degrade to "unverifiable". Hard failure with
        # the fix spelled out, not skippable: --skip-endpoint-check is for
        # unreachable endpoints, not missing tooling.
        try:
            ensure_vendored_cudagym()
        except HostingError as e:
            raise SystemExit(f"❌ {e}") from e
    for entry in hosting.endpoints:
        try:
            payload = probe_endpoint(entry)
        except HostingError as e:
            # Unreachable endpoint: the one failure --skip-endpoint-check may waive.
            if args.skip_endpoint_check:
                print(f"⚠️  {e} (continuing: --skip-endpoint-check)")
                continue
            raise SystemExit(
                f"❌ {e}\n   (pass --skip-endpoint-check to submit anyway)"
            ) from e
        ok, detail = verify_health_payload(payload, entry.sku)
        if ok is False:
            # Never skippable: a reachable endpoint with the WRONG silicon would
            # silently mistime Triton kernels (JIT compiles on whatever GPU serves).
            raise SystemExit(f"❌ endpoint {entry.name}: {detail}")
        # ok is None means nothing was checked — report that as its own state
        # rather than a pass, so an unverified SKU stays visible in the output.
        print(f"{'⚠️  NOT VERIFIED' if ok is None else '✅'} endpoint {entry.name}: {detail}")

    # Upload the nemorl codebase to the cluster
    output_dir = Path(cluster_config["paths"]["output"]) / args.exp_name
    code_upload_path = output_dir / "code"
    ssh_tunnel = SSHTunnel(cluster_config["hostname"])
    package_code(ssh_tunnel, code_upload_path, skip_commit_check=args.skip_commit_check)

    # Per-entry resolved endpoint URLs ride into the training config as ++overrides
    # (the env actor's pinned-server_url path takes precedence over ambient env).
    remote_env_extra_opts: list[str] = list(hosting.extra_config_opts)

    # Experimental slurm-service hosting: stand up a CudaGym service job on another
    # Slurm cluster and chain login-node proxies (+ an SSH tunnel when required).
    for entry in hosting.slurm_services:
        service_url = deploy_remote_cudagym(
            entry,
            submit_cluster_config=cluster_config,
            submit_ssh=ssh_tunnel,
            submit_code_upload_path=code_upload_path,
            cluster_config_path=CLUSTER_CONFIG_PATH,
            exp_name=args.exp_name,
            skip_commit_check=args.skip_commit_check,
            proxy_timeout=args.proxy_timeout,
        )
        remote_env_extra_opts.append(
            f"++env.cudagym.{entry.name}.server_url={service_url}"
        )

    # Upload sbatch script with custom variables
    print(
        f"⬆️  Uploading sbatch script to {cluster_config['hostname']} at {output_dir / 'run.sh'}..."
    )
    sbatch_script = SBATCH_TEMPLATE_PATH.read_text()
    # Some clusters have different gpu hw and node topology so include those overrides in the config
    extra_config_opts = (
        (
            args.extra_config_opts
            + " "
            + cluster_config["extra_config_opts"]
            + f" +cluster.host={args.cluster}"  # add information about the name of the current job's cluster
            + f" +cluster.sku={cluster_config['sku']}"  # add information about the GPU SKU of the current job's cluster
            + f" +cluster.endpoint_hostname={cluster_config['hostname']}"  # add information about the hostname of the current job's login node for the CudaGym environment
            + " "
            + " ".join(
                remote_env_extra_opts
            )  # add information about the remote environment sku
        ).strip()
    )

    # Pick the runner + uv extras from the recipe: the NeMo-Gym agentic path
    # uses a different driver script and needs the nemo_gym extra on top of atlas.
    # (uses_nemo_gym was computed from the defaults-merged recipe above.)
    if uses_nemo_gym:
        run_script = "examples/nemo_gym/run_grpo_nemo_gym.py"
        uv_extras = "--extra atlas --extra nemo_gym"
    else:
        run_script = "./examples/run_grpo_cuda.py"
        uv_extras = "--extra atlas"

    # Unset local secrets must not leak into the job as the literal string "None";
    # fill with "" and tell the user (the template's ${VAR:-...} then sees empty).
    # MODAL_PROXY_TOKEN_ID/SECRET: the cudagym SDK sends them as Modal-Key/
    # Modal-Secret headers to .modal.run eval endpoints.
    secrets = {}
    for name in (
        "HF_TOKEN",
        "WANDB_API_KEY",
        "CUDAGYM_AUTH_TOKEN",
        "MODAL_PROXY_TOKEN_ID",
        "MODAL_PROXY_TOKEN_SECRET",
    ):
        val = os.getenv(name)
        if not val:
            print(f"⚠️  {name} is not set locally — the job will run without it.")
        secrets[name] = val or ""

    # The uploaded tree has no .git, so the venvs need a version handed to them.
    # Deriving it from the submodule keeps it correct across bumps; a hand-typed
    # literal in the template would drift.
    cudagym_version = _vendored_cudagym_version()

    sbatch_vars = {
        "EXP_NAME": args.exp_name,
        "CONFIG_NAME": args.config,
        "EXTRA_CONFIG_OPTS": extra_config_opts,
        "RUN_SCRIPT": run_script,
        "UV_EXTRAS": uv_extras,
        "TIME": args.time,
        "NUM_NODES": args.num_nodes,
        **secrets,
        "GPUS_PER_NODE": cluster_config["gpus_per_node"],
        # Cluster facts come from the cluster yaml, not from name-matching here.
        "SKIP_GRES_ARG": "1" if cluster_config.get("skip_gres") else "",
        "SLURM_ACCOUNT": cluster_config["account"],
        "SLURM_PARTITION": cluster_config["partition"],
        "SLURM_QOS": cluster_config.get("qos", ""),  # empty = no --qos flag
        # Always present so no DEFAULT_* token leaks into the job env when no
        # hosting kind sets them (ray.sub tests CUDAGYM_ENABLED == "1").
        "CUDAGYM_ENABLED": "0",
        "CUDAGYM_CONTAINER": "",
        "ARTIFACTS_DIR": "",
        "CCACHE_DIR": "",
        # Single-endpoint jobs also carry the resolved URL in the ambient env —
        # the agentic (NeMo-Gym) path and any entry resolved from the
        # CUDAGYM_UNIFIED_SERVER_URL escape hatch read it. Per-entry ++server_url
        # overrides (above) take precedence for the single-turn env actors.
        # In-allocation runs overwrite it with the load-balancer URL inside ray.sub.
        "CUDAGYM_UNIFIED_SERVER_URL": hosting.unified_server_url
        or os.getenv("CUDAGYM_UNIFIED_SERVER_URL")
        or "",
        # From the recipe's hosting declarations; ray.sub reads both variables.
        "CUDAGYM_MODE": hosting.cudagym_mode,
        "CUDAGYM_NUM_NODES": hosting.cudagym_num_nodes,
        "CUDAGYM_VERSION": cudagym_version,
        # Empty (the default) leaves the namespace sandbox runtime untouched;
        # a path switches grpo.sh's enroot plumbing on.
        "CUDA_AGENT_ENROOT_IMAGE": args.enroot_agent_image or "",
        # cudagym_container is excluded: CUDAGYM_CONTAINER is set explicitly
        # (above/below), and both names fill the same DEFAULT_CUDAGYM_CONTAINER
        # template token.
    } | {k: v for k, v in cluster_config["paths"].items() if k != "cudagym_container"}

    # If an entry is hosted in-allocation (colocated or disjoint), pass the server
    # sbatch vars consumed by ray.sub. Endpoint kinds launch no servers in the
    # allocation; the token env baked in above (like HF_TOKEN) is inherited by the
    # ray head/worker containers so the driver can reach the remote endpoint.
    if hosting.in_allocation is not None:
        paths = cluster_config["paths"]
        cudagym_container = paths.get("cudagym_container", paths.get("container"))
        if not cudagym_container:
            raise ValueError(
                "Cluster config missing container path(s); need 'container' or 'cudagym_container'."
            )
        # The image must carry the server runtime deps MATCHING the vendored SDK
        # (the checkout is served via PYTHONPATH; its deps come from the image) —
        # warn when the sqsh name doesn't carry the SDK's major.minor.
        major_minor = ".".join(cudagym_version.split(".")[:2])
        if major_minor != "0.0" and major_minor not in Path(cudagym_container).name:
            print(
                f"⚠️  cudagym container {Path(cudagym_container).name} does not carry the "
                f"vendored SDK version {major_minor}.x — server runtime deps may not match "
                f"(import a matching sqsh and update the cluster yaml)."
            )
        sbatch_vars |= {
            "CUDAGYM_ENABLED": "1",
            "CUDAGYM_CONTAINER": cudagym_container,
            "ARTIFACTS_DIR": paths.get("artifacts"),
            "CCACHE_DIR": paths.get("ccache"),
        }

    for k, v in sbatch_vars.items():
        sbatch_script = fill_template(
            sbatch_script, k, v
        )  # replace every DEFAULT_<VAR> token in the script

    upload_text_as_file(ssh_tunnel, sbatch_script, output_dir / "run.sh")
    print("✅ Sbatch script uploaded")

    if args.only_upload:
        print("✅ Skipping launch")
        return

    # Submit the sbatch script
    launch_jobs(
        ssh_tunnel,
        code_upload_path,
        interactive=args.interactive,
        num_jobs=args.num_jobs,
    )


if __name__ == "__main__":
    main()
