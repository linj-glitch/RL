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

"""Unit tests for the Slurm submit tooling (submit_grpo.py + remote_utils.py)."""

import subprocess
import sys

import pytest

import submit_grpo
from remote_utils import fill_template
from submit_grpo import parse_extra_config_opts

SOLSWARM_OVERLAY = "resources_servers/cudagym/configs/cudagym_cuda_agent_solswarm.yaml"
BASE_AGENT_CONFIG = "resources_servers/cudagym/configs/cudagym_cuda_agent.yaml"


# --------------------------------------------------------------------------
# --extra-config-opts parsing (submit-time merge)
# --------------------------------------------------------------------------


def test_parse_extra_config_opts_honors_shell_quoting():
    """A quoted value containing spaces must stay one override; before shlex the
    whitespace split silently dropped everything after the first space."""
    opts = parse_extra_config_opts('++policy.notes="two words" ++env.x=1 loose-flag')
    assert opts == ["policy.notes=two words", "env.x=1"]
    assert parse_extra_config_opts("") == []


# --------------------------------------------------------------------------
# Container-mode gate vs. --extra-config-opts overrides
# --------------------------------------------------------------------------


@pytest.fixture()
def hermetic_submit_env(monkeypatch):
    """Drop ambient env that would change what the submit reads: cluster yaml
    paths.* keys are env-overridable by their uppercased names, and set Modal
    proxy tokens would let a mis-ordered gate reach the network preflight."""
    for name in (
        "WORKSPACE",
        "CACHE",
        "CONTAINER",
        "CUDAGYM_CONTAINER",
        "MODELS",
        "OUTPUT",
        "DATASETS",
        "ARTIFACTS",
        "CCACHE",
        "MODAL_PROXY_TOKEN_ID",
        "MODAL_PROXY_TOKEN_SECRET",
    ):
        monkeypatch.delenv(name, raising=False)


def _run_main(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["submit_grpo.py", *argv])
    submit_grpo.main()


def test_container_gate_sees_overlay_added_by_extra_opts(
    monkeypatch, hermetic_submit_env
):
    """An override that appends the solswarm overlay makes the job container
    mode, so submitting without --enroot-agent-image must fail at the gate
    (before hosting resolution, upload, or allocation)."""
    with pytest.raises(SystemExit, match="container mode"):
        _run_main(
            monkeypatch,
            [
                "--exp-name",
                "t",
                "--cluster",
                "cw-dfw-cs-001",
                "--config",
                "grpo_cuda_agentic_qwen3-8b.yaml",
                "--extra-config-opts",
                f"++env.nemo_gym.config_paths=[{BASE_AGENT_CONFIG},{SOLSWARM_OVERLAY}]",
            ],
        )


def test_container_gate_sees_overlay_removed_by_extra_opts(
    monkeypatch, hermetic_submit_env
):
    """The symmetric direction: an override that drops the overlay from a
    container-mode recipe makes --enroot-agent-image an error."""
    with pytest.raises(SystemExit, match="nothing in this job runs agent containers"):
        _run_main(
            monkeypatch,
            [
                "--exp-name",
                "t",
                "--cluster",
                "cw-dfw-cs-001",
                "--config",
                "grpo_cuda_agentic_solswarm_qwen3-8b.yaml",
                "--enroot-agent-image",
                "/some/agent-image.sqsh",
                "--extra-config-opts",
                f"++env.nemo_gym.config_paths=[{BASE_AGENT_CONFIG}]",
            ],
        )


# --------------------------------------------------------------------------
# fill_template quoting
# --------------------------------------------------------------------------


def test_fill_template_value_with_spaces_stays_one_assignment():
    line = "export UV_EXTRAS=${UV_EXTRAS:-DEFAULT_UV_EXTRAS}\n"
    filled = fill_template(line, "UV_EXTRAS", "--extra atlas --extra nemo_gym")
    assert "'--extra atlas --extra nemo_gym'" in filled
    assert "DEFAULT_UV_EXTRAS" not in filled


@pytest.mark.parametrize(
    "secret",
    [
        "with space",
        "pa$$word`whoami`",
        'has"double-quote',
        "has'single-quote and $HOME",
    ],
)
def test_fill_template_round_trips_literally_through_bash(secret):
    """The filled run.sh line must hand the job shell the value byte-for-byte:
    no expansion of $, backticks, or quotes (double quotes used to let bash
    expand and corrupt e.g. secrets)."""
    line = "export HF_TOKEN=${HF_TOKEN:-DEFAULT_HF_TOKEN}\n"
    filled = fill_template(line, "HF_TOKEN", secret)
    script = "unset HF_TOKEN\n" + filled + 'printf %s "$HF_TOKEN"\n'
    out = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, check=True
    )
    assert out.stdout == secret


def test_fill_template_none_and_numbers():
    assert fill_template("X=DEFAULT_FOO", "FOO", None) == "X=''"
    assert fill_template("N=DEFAULT_NUM_NODES", "NUM_NODES", 4) == "N=4"
