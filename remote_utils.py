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
"""Helpers for submitting jobs to Slurm clusters from a local machine.

Used by ``submit_grpo.py`` and ``submit_sft.py``: run commands and copy files
over SSH, rsync the git-tracked tree to the cluster, load
``slurm/clusters/<name>.yaml``, fill the ``DEFAULT_<VAR>`` tokens in the
sbatch template before uploading it, and run the uploaded wrapper.
"""

import os
import re
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Sequence, Union

from omegaconf import OmegaConf


def _which_or_raise(binary: str) -> str:
    """Return the path to a required binary, raising if it is not on PATH."""
    path = shutil.which(binary)
    if path is None:
        raise RuntimeError(f"Required binary '{binary}' not found in PATH")
    return path


def _run(cmd: Sequence[str], cwd: Optional[str] = None) -> subprocess.CompletedProcess:
    """Run a local command with captured text output, raising on a non-zero exit."""
    return subprocess.run(cmd, cwd=cwd, check=True, text=True, capture_output=True)


def _list_submodule_paths(repo_root: str) -> list[str]:
    """Return a list of submodule relative paths (recursive)."""
    try:
        sub_status = _run(
            ["git", "submodule", "status", "--recursive"], cwd=repo_root
        ).stdout
    except subprocess.CalledProcessError:
        return []
    sub_paths = []
    # Each status line is "<flag><sha> <path> (<describe>)"; take the path field.
    for line in sub_status.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 2:
            sub_paths.append(parts[1])
    return sub_paths


class SSHTunnel:
    """Access to one remote host through the ``ssh`` and ``scp`` command-line tools.

    Holds the connection options (port, identity file, host-key policy,
    compression) and applies them to every remote command and file copy.
    """

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
        """Return the ``user@host`` destination string (or bare host)."""
        return f"{self.user}@{self.host}" if self.user else self.host

    def _ssh_base(self) -> list[str]:
        """Return the ``ssh`` argv prefix with the shared connection options."""
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
            )  # accept-new needs OpenSSH >= 7.6
        return cmd

    def _scp_base(self) -> list[str]:
        """Return the ``scp`` argv prefix with the shared connection options."""
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
        """Run a shell command on the remote host; return ``(rc, stdout, stderr)``."""
        proc = subprocess.run(
            self._ssh_base() + [self._dest(), command], text=True, capture_output=True
        )
        return proc.returncode, proc.stdout, proc.stderr

    def put_file(self, local_path: str, remote_path: str) -> None:
        """Copy a local file to the remote host with ``scp``."""
        _run(self._scp_base() + [local_path, f"{self._dest()}:{remote_path}"])


def check_for_uncommitted_changes():
    """Raise if the repository or any submodule has uncommitted changes."""
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
    upload_path: Union[str, Path],
    delete: bool = True,
    skip_commit_check: bool = False,
) -> Union[str, Path]:
    """Rsync the project to the remote host, syncing only git-tracked files.

    ``git ls-files --recurse-submodules`` supplies the file list, so submodule
    files are included while untracked build artifacts are not. rsync then
    copies exactly those paths over SSH.
    """
    print(f"⬆️  Uploading code to {ssh_tunnel.host} at {upload_path}...")

    _which_or_raise("git")
    _which_or_raise("rsync")

    # Ensure git working trees are clean (root + submodules)
    if not skip_commit_check:
        check_for_uncommitted_changes()

    # Resolve repo root (_run raises if this is not a git work tree).
    repo_root = _run(["git", "rev-parse", "--show-toplevel"]).stdout.strip()

    # rsync's -e option takes one shell-parsed command string, so the tunnel's
    # ssh argv is quoted per token and joined.
    rsync_rsh = " ".join(shlex.quote(x) for x in ssh_tunnel._ssh_base())

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


def launch_jobs(
    ssh_tunnel: SSHTunnel,
    code_upload_path: Path,
    interactive: bool = False,
    num_jobs: int = 1,
    env_prefix: str = "",
) -> None:
    """Run the uploaded ``run.sh`` sbatch wrapper on the cluster, once per job.

    ``env_prefix`` is prepended verbatim to the remote command line (for
    example ``"CONVERT_STEP=100 CONVERT_HF_MODEL=Qwen/Qwen3-8B "``, which
    makes the SFT wrapper submit a checkpoint-conversion job instead of a
    training job).
    """
    for i in range(num_jobs):
        launch_cmd = f"cd {code_upload_path} && {env_prefix}bash ../run.sh"
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
        # The wrapper ends with `Submitted batch job <id>`; an empty id means
        # sbatch itself failed even if the wrapper exited 0.
        if rc != 0 or not re.search(r"Submitted batch job \d+", stdout):
            raise RuntimeError(
                f"Error running sbatch script (rc={rc}): {stderr.strip() or stdout.strip()}"
            )


def get_available_clusters(cluster_config_dir: Path) -> list[str]:
    """Return cluster names discovered in the cluster config directory."""
    return [p.stem for p in cluster_config_dir.glob("*.yaml")]


def get_available_configs(config_dir: Path, glob_pattern: str) -> list[str]:
    """List config file names under a directory by a glob pattern."""
    return [p.name for p in config_dir.glob(glob_pattern)]


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


def fill_template(sbatch_script: str, var_name: str, value) -> str:
    r"""Replace every ``DEFAULT_<VAR>`` token in the sbatch script text with a value.

    Matching is token-exact: a trailing negative lookahead keeps
    ``DEFAULT_ARTIFACTS`` from also matching the prefix of
    ``DEFAULT_ARTIFACTS_DIR``. ``None`` renders as an empty quoted string
    rather than the literal ``None``; ints and floats are inserted bare; every
    other value (str, Path, ...) is single-quoted, with embedded single quotes
    escaped as ``'\''``, so the job shell reads the value byte-for-byte — a
    ``$``, backtick, or ``"`` in e.g. a secret is never expanded. No template
    slot relies on job-shell expansion: cluster/container paths arrive fully
    resolved from the cluster YAML, and EXTRA_CONFIG_OPTS was already expanded
    by the submitting shell.
    """
    var = var_name.upper()
    if value is None:
        value_str = "''"
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        value_str = str(value)
    else:
        escaped = str(value).replace("'", "'\\''")
        value_str = f"'{escaped}'"
    return re.sub(
        rf"DEFAULT_{re.escape(var)}(?![A-Za-z0-9_])",
        lambda _m: value_str,
        sbatch_script,
    )


def upload_text_as_file(ssh_tunnel: SSHTunnel, text: str, remote_path: Path) -> None:
    """Upload the provided text to the remote path using a temporary local file."""
    with tempfile.NamedTemporaryFile() as temp_file:
        temp_file.write(text.encode())
        temp_file.flush()
        ssh_tunnel.put_file(temp_file.name, remote_path)
