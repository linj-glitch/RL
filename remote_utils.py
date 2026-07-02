#!/usr/bin/env python3
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
# this script is responsible for uploading the code to the cluster and submitting jobs from a remote machine

import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Sequence

from omegaconf import OmegaConf


def _which_or_raise(binary: str) -> str:
    path = shutil.which(binary)
    if path is None:
        raise RuntimeError(f"Required binary '{binary}' not found in PATH")
    return path


def _run(cmd: Sequence[str], cwd: Optional[str] = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, check=True, text=True, capture_output=True)


def _list_submodule_paths(repo_root: str) -> list:
    """Return a list of submodule relative paths (recursive)."""
    try:
        sub_status = _run(
            ["git", "submodule", "status", "--recursive"], cwd=repo_root
        ).stdout
    except subprocess.CalledProcessError:
        return []
    sub_paths = []
    for line in sub_status.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 2:
            sub_paths.append(parts[1])
    return sub_paths


class SSHTunnel:
    def __init__(
        self,
        host: str,
        *,
        user: Optional[str] = None,
        port: int = 22,
        identity_file: Optional[str] = None,
        strict_host_key_checking: bool = False,
        compress: bool = True,
    ):
        self.host = host
        self.user = user
        self.port = port
        self.identity_file = identity_file
        self.strict_host_key_checking = strict_host_key_checking
        self.compress = compress

        _which_or_raise("ssh")
        _which_or_raise("scp")

    def _dest(self) -> str:
        return f"{self.user}@{self.host}" if self.user else self.host

    def _ssh_base(self) -> list:
        cmd = ["ssh", "-p", str(self.port)]
        if self.compress:
            cmd.append("-C")
        if self.identity_file:
            cmd.extend(["-i", self.identity_file])
        if self.strict_host_key_checking:
            cmd.extend(["-o", "StrictHostKeyChecking=yes"])
        else:
            cmd.extend(
                ["-o", "StrictHostKeyChecking=accept-new"]
            )  # supported on modern macOS
        return cmd

    def _scp_base(self) -> list:
        cmd = ["scp", "-P", str(self.port)]
        if self.compress:
            cmd.append("-C")
        if self.identity_file:
            cmd.extend(["-i", self.identity_file])
        if self.strict_host_key_checking:
            cmd.extend(["-o", "StrictHostKeyChecking=yes"])
        else:
            cmd.extend(["-o", "StrictHostKeyChecking=accept-new"])
        return cmd

    def run_command(self, command: str):
        proc = subprocess.run(
            self._ssh_base() + [self._dest(), command], text=True, capture_output=True
        )
        return proc.returncode, proc.stdout, proc.stderr

    def put_file(self, local_path: str, remote_path: str) -> None:
        _run(self._scp_base() + [local_path, f"{self._dest()}:{remote_path}"])

    def get_file(self, remote_path: str, local_path: str) -> None:
        _run(self._scp_base() + [f"{self._dest()}:{remote_path}", local_path])


def check_for_uncommitted_changes():
    # Ensure we're inside a git repo
    try:
        repo_root = _run(["git", "rev-parse", "--show-toplevel"]).stdout.strip()
    except subprocess.CalledProcessError as e:
        raise RuntimeError("Not inside a git repository; cannot check changes") from e

    # Check working tree in main repo
    porcelain = _run(["git", "status", "--porcelain"], cwd=repo_root).stdout.strip()
    if porcelain:
        raise RuntimeError(
            "Uncommitted changes detected in repository. Commit, stash, or clean before packaging."
        )

    # Check submodules recursively
    for rel_path in _list_submodule_paths(repo_root):
        subdir_abs = os.path.join(repo_root, rel_path)
        if not os.path.isdir(subdir_abs):
            continue
        sub_porcelain = _run(
            ["git", "status", "--porcelain"], cwd=subdir_abs
        ).stdout.strip()
        if sub_porcelain:
            raise RuntimeError(
                f"Uncommitted changes detected in submodule '{rel_path}'. Commit, stash, or clean before packaging."
            )


def package_code(
    ssh_tunnel: SSHTunnel,
    upload_path: str,
    delete: bool = True,
    skip_commit_check: bool = False,
) -> str:
    """Rsync project to remote host using SSH, syncing only git-tracked files.

    Strategy: Use git ls-files to get all tracked files (including submodules)
    and rsync only those files. This respects git's view of what should be synced.
    """
    print(f"⬆️  Uploading code to {ssh_tunnel.host} at {upload_path}...")

    _which_or_raise("git")
    _which_or_raise("rsync")

    # Ensure git working trees are clean (root + submodules)
    if not skip_commit_check:
        check_for_uncommitted_changes()

    # Resolve repo root
    repo_root = _run(["git", "rev-parse", "--show-toplevel"]).stdout.strip()
    if not repo_root:
        raise RuntimeError("Not inside a git repository; cannot rsync code")

    # Build SSH transport for rsync
    ssh_cmd: list[str] = ["ssh", "-p", str(ssh_tunnel.port)]
    if ssh_tunnel.compress:
        ssh_cmd.append("-C")
    if ssh_tunnel.identity_file:
        ssh_cmd.extend(["-i", ssh_tunnel.identity_file])
    if ssh_tunnel.strict_host_key_checking:
        ssh_cmd.extend(["-o", "StrictHostKeyChecking=yes"])
    else:
        ssh_cmd.extend(["-o", "StrictHostKeyChecking=accept-new"])  # modern macOS
    rsync_rsh = " ".join(shlex.quote(x) for x in ssh_cmd)

    # Get all git-tracked files including submodules
    # --recurse-submodules ensures we get files from all submodules
    # --cached gets files from the index (staged/tracked files)
    tracked_files_result = _run(
        ["git", "ls-files", "--recurse-submodules", "--cached"], cwd=repo_root
    )

    if not tracked_files_result.stdout.strip():
        raise RuntimeError("No git-tracked files found")

    tracked_files = [
        f.strip() for f in tracked_files_result.stdout.strip().split("\n") if f.strip()
    ]

    # Create a temporary file list for rsync --files-from
    with tempfile.NamedTemporaryFile(mode="w", suffix=".rsync-files", delete=True) as f:
        for file_path in tracked_files:
            # Ensure we write relative paths
            f.write(file_path + "\n")
        f.flush()  # Ensure data is written before rsync reads it

        # Base rsync arguments for files-from mode
        rsync_base: list[str] = [
            "rsync",
            "-a",  # archive mode (preserves perms, symlinks, times, etc.)
            "-z",  # compress over the wire
            "--files-from",
            f.name,
            "--relative",  # preserve relative path structure
            "-e",
            rsync_rsh,
        ]
        if delete:
            rsync_base.append("--delete")

        # Ensure remote destination directory exists
        rc, _, err = ssh_tunnel.run_command(f"mkdir -p {upload_path}")
        if rc != 0:
            raise RuntimeError(f"Failed to create remote directory: {err}")

        # Source is repo root, destination includes the upload path
        src_root = repo_root + "/"  # trailing slash important for rsync
        dest_root = f"{ssh_tunnel._dest()}:{upload_path}/"

        # Execute rsync with the file list
        _run(rsync_base + [src_root, dest_root])

    print("✅ Code uploaded")
    return upload_path


def get_available_clusters(cluster_config_dir: Path) -> list[str]:
    """Return cluster names discovered in the cluster config directory."""
    return [p.stem for p in cluster_config_dir.glob("*.yaml")]


def get_available_configs(
    config_dir: Path, glob_pattern: str, return_stems: bool = False
) -> list[str]:
    """List config files under a directory by a glob pattern."""
    return [p.stem if return_stems else p.name for p in config_dir.glob(glob_pattern)]


def load_cluster_config(cluster_config_dir: Path, cluster_name: str) -> dict:
    """Load and resolve cluster YAML, applying env overrides for paths.* keys.

    The environment can override any key under paths.* by setting an env var with
    the uppercased key name. For example, paths.workspace can be overridden by WORKSPACE.
    """
    cfg = OmegaConf.load(cluster_config_dir / f"{cluster_name}.yaml")
    if "paths" in cfg and cfg.paths is not None:
        for key in list(cfg.paths.keys()):
            env_val = os.getenv(key.upper(), None)
            if env_val:
                cfg.paths[key] = env_val
                print(f'- Overriding "{key}" from env, new value: "{env_val}"')
    return OmegaConf.to_container(cfg, resolve=True)


def validate_cluster_paths(paths: dict) -> None:
    """The home directory should never be mounted even if you're just mounting a symlink.

    This can slow down the cluster https://nvidia.slack.com/archives/C08TRFE6EJ0/p1758840246385429?thread_ts=1753981727.190459&cid=C08TRFE6EJ0.
    """
    for key, path in paths.items():
        if isinstance(path, str) and (
            path.startswith("~/") or path.startswith("/home")
        ):
            raise ValueError(
                f"Path {key}={path} should not start with ~/ or /home. Please use an absolute /lustre/... path instead."
            )


def fill_template(sbatch_script: str, var_name: str, value: str | int) -> str:
    """Replace DEFAULT_<VAR> with a value in the sbatch script text."""
    var = var_name.upper()
    value_str = f'"{value}"' if isinstance(value, str) else str(value)
    return sbatch_script.replace(f"DEFAULT_{var}", value_str)


def upload_text_as_file(ssh_tunnel: SSHTunnel, text: str, remote_path: Path) -> None:
    """Upload the provided text to the remote path using a temporary local file."""
    with tempfile.NamedTemporaryFile() as temp_file:
        temp_file.write(text.encode())
        temp_file.flush()
        ssh_tunnel.put_file(temp_file.name, remote_path)
