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

"""Stand up a CudaGym eval service on ANOTHER Slurm cluster (hosting kind ``slurm-service``).

At submit time this module, driven by ``submit_grpo.py``:

  1. uploads the codebase to the service cluster and runs cudagym's
     ``deployments/multi_node/slurm/generate_slurm_scripts.py`` there;
  2. starts cudagym's ``deployments/multi_cluster/proxy.sh`` on the service
     cluster's login node — the proxy sbatches the generated service job and
     forwards to its load balancer once healthy;
  3. starts a second ``proxy.sh`` on the SUBMIT cluster's login node pointing
     at the service cluster's proxy, so compute nodes (which usually cannot
     reach other clusters directly) talk to their own login node. When the
     submit cluster's login node cannot reach the service cluster either
     (cluster yaml ``requires_proxy_tunnel: true``), a persistent
     ``ssh -L`` tunnel bridges the two login nodes first.

The training config then pins ``env.cudagym.<name>.server_url`` at
``http://<submit-login-node>:<endpoint_port>``.

Experimental: restored from git history (deleted in the audit sweep as
then-unreachable; re-wired behind ``hosting: {kind: slurm-service}``) and not
yet re-validated live — inspect job/proxy logs on first use.
"""

import time
from pathlib import Path
from typing import Any

from remote_utils import SSHTunnel, load_cluster_config, package_code
from slurm.cudagym_hosting import ResolvedEntry


def generate_cudagym_script(
    ssh_tunnel: SSHTunnel,
    code_upload_path: Path,
    service_cluster: str,
    num_service_nodes: int,
) -> str:
    """Generate CudaGym cluster script using generate_slurm_scripts.py.

    Returns:
        Script filename that was generated (the generator's own
        ``cudagym_cluster_<host>_nodes_<n>.sh`` naming).
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
    raise RuntimeError(
        f"No free TCP port on {ssh.host} in [{start_port}, {start_port + max_steps})"
    )


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


def deploy_remote_cudagym(
    entry: ResolvedEntry,
    *,
    submit_cluster: str,
    submit_cluster_config: dict[str, Any],
    submit_ssh: SSHTunnel,
    submit_code_upload_path: Path,
    cluster_config_path: Path,
    exp_name: str,
    skip_commit_check: bool,
    proxy_timeout: int,
) -> str:
    """Deploy one ``slurm-service`` entry's CudaGym service and chain the proxies.

    Returns the URL the submit cluster's COMPUTE nodes reach the service at
    (their own login node's proxy), for pinning as the entry's ``server_url``.
    """
    service = entry.service
    service_cluster = service["service_cluster"]
    if service_cluster == submit_cluster:
        raise SystemExit(
            f"❌ env.cudagym.{entry.name}: slurm-service pointing at the submit "
            f"cluster itself makes no sense — use hosting kind 'colocated' or "
            f"'disjoint' instead."
        )
    endpoint_port = service["endpoint_port"]
    service_login_port = service["service_login_port"]
    print(f"🌐 Readying remote CudaGym environment on {service_cluster}")
    service_cluster_config = load_cluster_config(cluster_config_path, service_cluster)

    # Upload the nemorl codebase to the remote cluster
    service_output_dir = Path(service_cluster_config["paths"]["output"]) / exp_name
    service_code_upload_path = service_output_dir / "code"
    service_ssh = SSHTunnel(service_cluster_config["hostname"])
    package_code(
        service_ssh,
        service_code_upload_path,
        skip_commit_check=skip_commit_check,
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
        timeout=proxy_timeout,
    )

    # Start proxy on current cluster
    service_url = f"http://{service_ssh.host}:{service_login_port}"

    # If the cluster requires a tunnel, start a persistent SSH tunnel to the
    # remote cluster's login node and point the proxy at it
    if submit_cluster_config.get("requires_proxy_tunnel"):
        tunnel_port = _find_next_free_port(submit_ssh, start_port=endpoint_port + 1)
        start_ssh_tunnel(
            ssh=submit_ssh,
            tunnel_port=tunnel_port,
            remote_host=service_ssh.host,
            remote_port=service_login_port,
        )
        service_url = f"http://127.0.0.1:{tunnel_port}"

    run_proxy(
        ssh_tunnel=submit_ssh,
        code_upload_path=submit_code_upload_path,
        mode="service-url",
        service_url_or_script=service_url,
        port=endpoint_port,
        timeout=proxy_timeout,
    )

    # Compute nodes reach the service through the submit cluster's login-node proxy.
    return f"http://{submit_cluster_config['hostname']}:{endpoint_port}"
