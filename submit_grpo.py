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

    # atlas cpp->cuda 32b
    python submit_grpo.py --exp-name cuda_b200 --config grpo_cuda_b200.yaml --cluster dfw --num-nodes 16 --cudagym-mode colocated
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

CONFIG_PATH = Path(__file__).parent / "examples" / "configs" / "recipes" / "atlas"
CLUSTER_CONFIG_PATH = Path(__file__).parent / "slurm" / "clusters"
SBATCH_TEMPLATE_PATH = Path(__file__).parent / "slurm" / "grpo" / "grpo.sh"


def parse_remote_environments(config_path: Path) -> list[dict]:
    """Return the SSH-tunneled remote CudaGym services declared in a nemo-rl config.

    Only ``env.cudagym.<name>`` entries that specify the remote-service fields
    ({endpoint_port, service_cluster, num_service_nodes[, service_login_port]}) are
    returned. Thin-client per-arch env configs (colocated/disjoint, or a Modal
    ``server_url``) carry none of those fields and are skipped here — their hosting
    is driven by ``--cudagym-mode`` + ray.sub (or the recipe's ``server_url``).
    """
    selected = OmegaConf.select(OmegaConf.load(config_path), "env.cudagym")
    remote_envs = (
        {} if selected is None else OmegaConf.to_container(selected, resolve=True)
    )

    environments: dict[str, dict] = {}
    for env_name, cluster_config in remote_envs.items():
        if not isinstance(cluster_config, dict):
            # Invalid cluster config
            continue

        required_fields = [
            "endpoint_port",
            "service_cluster",
            "num_service_nodes",
        ]
        # Entries lacking the SSH-remote service fields are NOT SSH-tunneled remote
        # services — they are thin-client per-arch env configs (colocated/disjoint, or
        # a Modal endpoint via ``server_url``). Skip them; their hosting is driven by
        # ``--cudagym-mode`` + ray.sub (or the recipe's ``server_url`` for remote).
        missing = [f for f in required_fields if not cluster_config.get(f)]
        if missing:
            continue

        environments[env_name] = {
            "endpoint_port": cluster_config.get("endpoint_port"),
            "service_cluster": cluster_config.get("service_cluster"),
            "num_service_nodes": cluster_config.get("num_service_nodes"),
            "service_login_port": cluster_config.get(
                "service_login_port", 8998
            ),  # unused when colocated
        }
    return environments


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
        f"cd {code_upload_path}/3rdparty/cudagym/examples/multi_node/slurm && "
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
        script_remote_path = f"{code_upload_path}/3rdparty/cudagym/examples/multi_node/slurm/scripts/{service_url_or_script}"
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
    base_cmd = f"cd {code_upload_path}/3rdparty/cudagym && ./examples/multi_cluster/proxy.sh start"
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
        default="grpo_cuda_b200.yaml",
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
    parser.add_argument(
        "--cudagym-mode",
        type=str,
        default=None,
        choices=["colocated", "disjoint", "remote"],
        help=(
            "How CudaGym compile/GPU servers are hosted for this run. "
            "'colocated': servers on every node, load balancer on the ray head "
            "(eval time-shares training GPUs; M0 single-turn). "
            "'disjoint': the trailing --cudagym-num-nodes nodes are carved out of the "
            "ray cluster and dedicated to CudaGym (M1 in-cluster). "
            "'remote': no in-allocation servers; the driver talks to a remote endpoint "
            "carried by the recipe (env.cudagym.<arch>.server_url; M1 default). "
            "If omitted, falls back to the recipe-derived behavior (colocated when the "
            "recipe declares an in-cluster env, disabled otherwise)."
        ),
    )
    parser.add_argument(
        "--cudagym-num-nodes",
        type=int,
        default=0,
        help="Number of trailing nodes reserved for CudaGym when --cudagym-mode=disjoint.",
    )
    args = parser.parse_args()

    if args.cudagym_mode == "disjoint" and not (
        1 <= args.cudagym_num_nodes < args.num_nodes
    ):
        parser.error(
            "--cudagym-num-nodes must be in [1, --num-nodes) when "
            f"--cudagym-mode=disjoint (got {args.cudagym_num_nodes} with "
            f"--num-nodes={args.num_nodes})"
        )

    # Load cluster config with env overrides applied and resolved
    cluster_config = load_cluster_config(CLUSTER_CONFIG_PATH, args.cluster)

    # Guard against mounting home directories which can slow down clusters
    validate_cluster_paths(cluster_config["paths"])

    # Upload the nemorl codebase to the cluster
    output_dir = Path(cluster_config["paths"]["output"]) / args.exp_name
    code_upload_path = output_dir / "code"
    ssh_tunnel = SSHTunnel(cluster_config["hostname"])
    package_code(ssh_tunnel, code_upload_path, skip_commit_check=args.skip_commit_check)

    # Parse training config for remote environments
    remote_environments = parse_remote_environments(CONFIG_PATH / args.config)
    print(f"🔎 Found {len(remote_environments)} CudaGym environment(s) in config")
    # Track cluster information for the remote environments
    remote_env_extra_opts: list[str] = []
    # Track if a colocated CudaGym has been found
    colocated_cudagym_found = False
    for env_name, service in remote_environments.items():
        endpoint_port = service["endpoint_port"]
        service_cluster = service["service_cluster"]
        num_service_nodes = service["num_service_nodes"]
        service_login_port = service["service_login_port"]

        if service_cluster == args.cluster:
            # Found CudaGym on the current cluster, so it will be colocated on the same nodes as nemorl
            # Do not create proxies, the service will be launched inside ray.sub
            print("🔎 Found colocated CudaGym environment")

            if colocated_cudagym_found:
                print(
                    f"⚠️ Multiple colocated CudaGym environments detected; "
                    f"only the first will be started. Skipping '{env_name}'."
                )
                continue
            colocated_cudagym_found = True

            # Add env_name.arch to the remote environment config
            remote_env_extra_opts.append(
                f"+env.cudagym.{env_name}.arch={cluster_config['arch']}"
            )

        else:
            print(f"🌐 Readying remote CudaGym environment on {service_cluster}")
            service_cluster_config = load_cluster_config(
                CLUSTER_CONFIG_PATH, service["service_cluster"]
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
                num_service_nodes=num_service_nodes,
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

            # If the cluster requires a tunnel, start a persistent SSH tunnel to the remote cluster's login node and point the proxy at it
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

            # Add env_name.arch to the remote environment config
            remote_env_extra_opts.append(
                f"+env.cudagym.{env_name}.arch={service_cluster_config['arch']}"
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
            + f" +cluster.arch={cluster_config['arch']}"  # add information about the GPU architecture of the current job's cluster
            + f" +cluster.endpoint_hostname={cluster_config['hostname']}"  # add information about the hostname of the current job's login node for the CudaGym environment
            + " "
            + " ".join(
                remote_env_extra_opts
            )  # add information about the remote environment arch
        ).strip()
    )

    sbatch_vars = {
        "EXP_NAME": args.exp_name,
        "CONFIG_NAME": args.config,
        "EXTRA_CONFIG_OPTS": extra_config_opts,
        "TIME": args.time,
        "NUM_NODES": args.num_nodes,
        "HF_TOKEN": os.getenv("HF_TOKEN"),
        "WANDB_API_KEY": os.getenv("WANDB_API_KEY"),
        "CUDAGYM_AUTH_TOKEN": os.getenv("CUDAGYM_AUTH_TOKEN"),
        "OUTPUT_DIR": output_dir,
        "GPUS_PER_NODE": cluster_config["gpus_per_node"],
        "SKIP_GRES_ARG": "1" if args.cluster == "eos" else "",
    } | {**cluster_config["paths"]}

    # Resolve the effective CudaGym hosting mode. An explicit --cudagym-mode wins;
    # otherwise fall back to the original recipe-derived behavior (colocated when the
    # recipe declares an in-cluster env, disabled otherwise).
    cudagym_mode = args.cudagym_mode
    if cudagym_mode is None:
        cudagym_mode = "colocated" if colocated_cudagym_found else ""
    sbatch_vars["CUDAGYM_MODE"] = cudagym_mode
    sbatch_vars["CUDAGYM_NUM_NODES"] = args.cudagym_num_nodes

    # If we are hosting CudaGym in-allocation (colocated or disjoint), pass the
    # server sbatch vars consumed by ray.sub. Remote mode launches no servers in
    # the allocation; the token env baked in above (like HF_TOKEN) is inherited by
    # the ray head/worker containers so the driver can reach the remote endpoint.
    if cudagym_mode in ("colocated", "disjoint"):
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
