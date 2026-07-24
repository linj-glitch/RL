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

import argparse
import os
import time
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
    load_recipe_merged,
    probe_endpoint,
    resolve_hosting,
    verify_health_payload,
)

CONFIG_PATH = Path(__file__).parent / "examples" / "configs" / "recipes" / "atlas"
CLUSTER_CONFIG_PATH = Path(__file__).parent / "slurm" / "clusters"
SBATCH_TEMPLATE_PATH = Path(__file__).parent / "slurm" / "grpo" / "grpo.sh"


def generate_cudagym_script(
    ssh_tunnel: SSHTunnel,
    code_upload_path: Path,
    service_cluster: str,
    num_service_nodes: int,
) -> str:
    """Generate CudaGym cluster script using generate_slurm_scripts.py.

    Returns:
        Script filename that was generated.
    """
    script_name = f"cudagym_cluster_{service_cluster}_nodes_{num_service_nodes}.sh"

    # Run generate_slurm_scripts.py on the remote cluster
    generate_cmd = (
        f"cd {code_upload_path}/3rdparty/cudagym/deployments/multi_node/slurm && "
        f"python generate_slurm_scripts.py --clusters {service_cluster} --nodes {num_service_nodes}"
    )

    print(f"🔧 Generating CudaGym service script: {script_name}")
    rc, out, err = ssh_tunnel.run_command(generate_cmd)
    if rc != 0:
        raise RuntimeError(
            f"Failed to generate CudaGym script for {service_cluster} with {num_service_nodes} nodes:\n"
            f"stdout: {out}\nstderr: {err}"
        )

    print("✅ Generated script")
    return script_name


def run_proxy(
    ssh_tunnel: SSHTunnel,
    code_upload_path: Path,
    mode: str,
    service_url_or_script: str,
    port: int,
    timeout: int,
) -> None:
    """Run proxy.sh on a cluster."""
    # If using a service script, check that it exists
    if mode == "service":
        script_remote_path = f"{code_upload_path}/3rdparty/cudagym/deployments/multi_node/slurm/scripts/{service_url_or_script}"
        rc_chk, _, _ = ssh_tunnel.run_command(f"test -f {script_remote_path}")
        if rc_chk != 0:
            raise ValueError(
                f"Service script not found on destination cluster: {service_url_or_script}. "
                f"Generate it via generate_slurm_scripts.py or adjust num_service_nodes/service_cluster."
            )

    # proxy.sh start
    print(
        f"🌐 Starting proxy at http://{ssh_tunnel.host}:{port}, forwarding requests to {service_url_or_script} service"
    )
    base_cmd = f"cd {code_upload_path}/3rdparty/cudagym && ./deployments/multi_cluster/proxy.sh start"
    if mode == "service-url":
        cmd = f"{base_cmd} --service-url {service_url_or_script} --port {port} --force"
    elif mode == "service":
        cmd = f"{base_cmd} --service {service_url_or_script} --port {port} --force"
    else:
        raise ValueError(f"Unknown proxy mode: {mode}")
    rc, out, err = ssh_tunnel.run_command(cmd)
    if rc != 0:
        raise RuntimeError(f"Failed to start proxy: {err or out}")

    # Wait until proxy.sh discovers the service url and service is ready and returns /status 200
    wait_proxy_ready(ssh_tunnel, port=port, timeout=timeout)
    print("✅ Service proxy is ready")


def wait_proxy_ready(ssh: SSHTunnel, port: int, timeout: float = 1800.0) -> None:
    """Wait until the proxy's /status returns 200, implying the upstream service is ready."""
    start = time.time()
    while time.time() - start < timeout:
        cmd = (
            f'/bin/bash -lc "curl -sf --connect-timeout 1 --max-time 3 '
            f'http://127.0.0.1:{port}/status >/dev/null 2>&1"'
        )
        rc, _, _ = ssh.run_command(cmd)
        if rc == 0:
            return
        time.sleep(2)
    raise TimeoutError(f"Proxy on {ssh.host}:{port} not ready within {int(timeout)}s")


def _find_next_free_port(
    ssh: SSHTunnel, start_port: int = 8999, max_steps: int = 100
) -> int:
    """Return first available TCP port >= start_port not currently listening on the login node."""
    port = start_port
    for _ in range(max_steps):
        rc, _, _ = ssh.run_command(
            f"/bin/bash -lc \"ss -tln 2>/dev/null | grep -q ':{port} ' \""
        )
        if rc != 0:
            return port
        port += 1
    return port


def start_ssh_tunnel(
    ssh: SSHTunnel,
    tunnel_port: int,
    remote_host: str,
    remote_port: int,
    timeout: float = 30.0,
) -> None:
    """Ensure a persistent SSH -L tunnel is running on the cluster login node. Auto-reconnects if the SSH link drops."""
    print(
        f"🔗 Starting SSH tunnel on {ssh.host}:127.0.0.1:{tunnel_port} -> {remote_host}:{remote_port}"
    )

    # If already listening, nothing to do
    rc_chk, _, _ = ssh.run_command(
        f"/bin/bash -lc \"ss -tln 2>/dev/null | grep -q ':{tunnel_port} ' \""
    )
    if rc_chk == 0:
        print(
            f"✅ SSH tunnel already exists. Point the proxy at http://127.0.0.1:{tunnel_port}"
        )
        return

    # Start a persistent loop with nohup
    loop_cmd = (
        "nohup bash -lc '"
        "while true; do "
        f"ssh -N -L 127.0.0.1:{tunnel_port}:localhost:{remote_port} "
        "-o ExitOnForwardFailure=yes "
        "-o ServerAliveInterval=30 "
        "-o ServerAliveCountMax=3 "
        "-o StrictHostKeyChecking=accept-new "
        f'"$USER@{remote_host}" || true; '
        "sleep 2; "
        "done' >/dev/null 2>&1 &"
    )
    rc, out, err = ssh.run_command(f'/bin/bash -lc "{loop_cmd}"')
    if rc != 0:
        raise RuntimeError(f"Failed to start SSH tunnel: {err or out}")

    # Wait for listener to appear
    wait_tunnel_ready(ssh, tunnel_port, timeout)
    print(f"✅ SSH tunnel is ready. Point the proxy at http://127.0.0.1:{tunnel_port}")
    return


def wait_tunnel_ready(ssh: SSHTunnel, port: int, timeout: float = 30.0) -> None:
    """Wait for an SSH -L tunnel listener."""
    start = time.time()
    while time.time() - start < timeout:
        rc2, _, _ = ssh.run_command(
            f"/bin/bash -lc \"ss -tln 2>/dev/null | grep -q ':{port} ' \""
        )
        if rc2 == 0:
            return
        time.sleep(2)
    raise TimeoutError(
        f"SSH tunnel on {ssh.host}:127.0.0.1:{port} not up within {int(timeout)}s"
    )


def launch_jobs(
    ssh_tunnel, code_upload_path, interactive: bool = False, num_jobs: int = 1
):
    for i in range(num_jobs):
        launch_cmd = f"cd {code_upload_path} && bash ../run.sh"
        if interactive:
            launch_cmd += " -i"

        print(f"🚀 Running sbatch script ({i + 1}/{num_jobs}): {launch_cmd}")
        _, stdout, stderr = ssh_tunnel.run_command(launch_cmd)
        print(stdout)
        if stderr:
            raise RuntimeError("Error running sbatch script: " + stderr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", "-e", default="grpo_32b_sft_dfw", type=str)
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
        default="dfw",
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
        "--num-nodes", type=int, default=16, help="Number of nodes for the job"
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
        help="Timeout in seconds for the remote environment to be ready",
    )
    # CudaGym hosting is declared per SKU in the recipe (env.cudagym.<name>.hosting;
    # see slurm/cudagym_hosting.py). The old flags survive only as hidden stubs so
    # passing them fails fast with a migration hint instead of "unrecognized argument".
    parser.add_argument("--cudagym-mode", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--cudagym-num-nodes", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--cudagym-url", default=None, help=argparse.SUPPRESS)
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

    if any(v is not None for v in (args.cudagym_mode, args.cudagym_num_nodes, args.cudagym_url)):
        parser.error(
            "--cudagym-mode/--cudagym-url/--cudagym-num-nodes were removed: hosting is now "
            "declared per SKU in the recipe, e.g.\n"
            "  env.cudagym.b200.hosting: {kind: endpoint, endpoint: modal/b200}\n"
            "kinds: colocated | disjoint (num_nodes: N) | endpoint | slurm-service; "
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
    cli_overrides = [
        opt.lstrip("+") for opt in args.extra_config_opts.split() if "=" in opt
    ]
    if cli_overrides:
        recipe_cfg = OmegaConf.merge(recipe_cfg, OmegaConf.from_dotlist(cli_overrides))
    uses_nemo_gym = bool(
        OmegaConf.select(recipe_cfg, "env.should_use_nemo_gym", default=False)
    )
    try:
        hosting = resolve_hosting(
            recipe_cfg, cluster_config, args.num_nodes, uses_nemo_gym
        )
    except HostingError as e:
        raise SystemExit(f"❌ {e}") from e
    print("🔎 CudaGym hosting:")
    for entry in hosting.entries:
        detail = f"kind={entry.kind}"
        if entry.url:
            detail += f" url={entry.url}"
        if entry.kind == "disjoint":
            detail += f" num_nodes={entry.num_nodes}"
        if entry.kind == "slurm-service":
            detail += f" service_cluster={entry.service['service_cluster']}"
        print(f"   - {entry.name}: sku={entry.sku} {detail}")
    for warning in hosting.warnings:
        print(f"⚠️  {warning}")

    # Preflight: ping every remote endpoint's /health and check the reported GPU
    # against the declared SKU. In-allocation servers don't exist yet — they get
    # the same check at runtime init (verify_endpoint_sku).
    for entry in hosting.endpoints:
        try:
            payload = probe_endpoint(entry)
        except HostingError as e:
            if args.skip_endpoint_check:
                print(f"⚠️  {e} (continuing: --skip-endpoint-check)")
                continue
            raise SystemExit(
                f"❌ {e}\n   (pass --skip-endpoint-check to submit anyway)"
            ) from e
        ok, detail = verify_health_payload(payload, entry.sku)
        if not ok:
            # Never skippable: a reachable endpoint with the WRONG silicon would
            # silently mistime Triton kernels (JIT compiles on whatever GPU serves).
            raise SystemExit(f"❌ endpoint {entry.name}: {detail}")
        icon = "⚠️ " if detail.startswith("unverifiable") else "✅"
        print(f"{icon} endpoint {entry.name}: {detail}")

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
        service = entry.service
        service_cluster = service["service_cluster"]
        if service_cluster == args.cluster:
            raise SystemExit(
                f"❌ env.cudagym.{entry.name}: slurm-service pointing at the submit "
                f"cluster itself makes no sense — use hosting kind 'colocated' or "
                f"'disjoint' instead."
            )
        endpoint_port = service["endpoint_port"]
        service_login_port = service["service_login_port"]
        print(f"🌐 Readying remote CudaGym environment on {service_cluster}")
        service_cluster_config = load_cluster_config(
            CLUSTER_CONFIG_PATH, service_cluster
        )

        # Upload the nemorl codebase to the remote cluster
        service_output_dir = (
            Path(service_cluster_config["paths"]["output"]) / args.exp_name
        )
        service_code_upload_path = service_output_dir / "code"
        service_ssh = SSHTunnel(service_cluster_config["hostname"])
        package_code(
            service_ssh,
            service_code_upload_path,
            skip_commit_check=args.skip_commit_check,
        )

        # Start proxy+service on remote cluster
        service_script = generate_cudagym_script(
            ssh_tunnel=service_ssh,
            code_upload_path=service_code_upload_path,
            service_cluster=service_cluster,
            num_service_nodes=service["num_service_nodes"],
        )
        run_proxy(
            ssh_tunnel=service_ssh,
            code_upload_path=service_code_upload_path,
            mode="service",
            service_url_or_script=service_script,
            port=service_login_port,
            timeout=args.proxy_timeout,
        )

        # Start proxy on current cluster
        service_url = f"http://{service_ssh.host}:{service_login_port}"

        # If the cluster requires a tunnel, start a persistent SSH tunnel to the
        # remote cluster's login node and point the proxy at it
        if cluster_config.get("requires_proxy_tunnel"):
            tunnel_port = _find_next_free_port(
                ssh_tunnel, start_port=endpoint_port + 1
            )
            start_ssh_tunnel(
                ssh=ssh_tunnel,
                tunnel_port=tunnel_port,
                remote_host=service_ssh.host,
                remote_port=service_login_port,
            )
            service_url = f"http://127.0.0.1:{tunnel_port}"

        run_proxy(
            ssh_tunnel=ssh_tunnel,
            code_upload_path=code_upload_path,
            mode="service-url",
            service_url_or_script=service_url,
            port=endpoint_port,
            timeout=args.proxy_timeout,
        )

        # Compute nodes reach the service through the submit cluster's login-node
        # proxy; pin this env entry's endpoint at it.
        remote_env_extra_opts.append(
            f"++env.cudagym.{entry.name}.server_url="
            f"http://{cluster_config['hostname']}:{endpoint_port}"
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

    sbatch_vars = {
        "EXP_NAME": args.exp_name,
        "CONFIG_NAME": args.config,
        "EXTRA_CONFIG_OPTS": extra_config_opts,
        "RUN_SCRIPT": run_script,
        "UV_EXTRAS": uv_extras,
        "TIME": args.time,
        "NUM_NODES": args.num_nodes,
        **secrets,
        "OUTPUT_DIR": output_dir,
        "GPUS_PER_NODE": cluster_config["gpus_per_node"],
        "SKIP_GRES_ARG": "1" if args.cluster == "eos" else "",
        # account/partition/qos come from the cluster yaml; qos empty = no --qos flag.
        "SLURM_ACCOUNT": cluster_config.get("account", "coreai_nvfm_cupilot"),
        "SLURM_PARTITION": cluster_config.get("partition", "batch"),
        "SLURM_QOS": cluster_config.get("qos", ""),
        # Always present so no DEFAULT_* token leaks into the job env when no
        # hosting kind sets them (ray.sub tests CUDAGYM_ENABLED == "1").
        "CUDAGYM_ENABLED": "0",
        "CUDAGYM_CONTAINER": "",
        "ARTIFACTS_DIR": "",
        "CCACHE_DIR": "",
        # Single-endpoint jobs also carry the resolved URL in the ambient env —
        # the agentic (NeMo-Gym) path and any entry using the escape hatch read
        # it. Per-entry ++server_url overrides (above) take precedence for the
        # single-turn env actors. In-allocation runs overwrite this with the LB
        # URL inside ray.sub, exactly as before.
        "CUDAGYM_UNIFIED_SERVER_URL": hosting.unified_server_url
        or os.getenv("CUDAGYM_UNIFIED_SERVER_URL")
        or "",
        # From the recipe's hosting declarations (ray.sub contract unchanged).
        "CUDAGYM_MODE": hosting.cudagym_mode,
        "CUDAGYM_NUM_NODES": hosting.cudagym_num_nodes,
    } | {**cluster_config["paths"]}

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
