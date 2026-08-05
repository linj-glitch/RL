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
"""Tests for the KernelFactory problem row normalization.

Row fields the builders always write are hard-indexed by design
(``definition``, ``workloads``, ``language``, ``destination_passing_style``,
``sol_anchors``): a silent default would evaluate a row as something its
builder never declared. ``target_hardware`` alone is deliberately optional —
it falls back to the environment's SKU.
"""

import json
from types import SimpleNamespace

import pytest

from nemo_rl.data.atlas_datasets.grpo_cuda_dataset import (
    format_cuda_problem,
    kfb_problem_to_gym_seed,
    kfb_problem_to_row,
)

ENVS = {"b200": SimpleNamespace(weight=1.0, sku="B200")}


def _row(**overrides):
    row = {
        "definition": "{}",
        "workloads": "[]",
        "language": "triton",
        "target_hardware": "B200",
        "destination_passing_style": True,
        "sol_anchors": "{}",
    }
    row.update(overrides)
    return row


def test_declared_fields_pass_through():
    out = format_cuda_problem(_row(), ENVS)
    assert out["task_name"] == "b200"
    assert out["language"] == "triton"
    assert out["destination_passing_style"] is True


def test_missing_language_is_a_dataset_bug():
    row = _row()
    del row["language"]
    with pytest.raises(KeyError):
        format_cuda_problem(row, ENVS)


def test_missing_destination_passing_style_is_a_dataset_bug():
    row = _row()
    del row["destination_passing_style"]
    with pytest.raises(KeyError):
        format_cuda_problem(row, ENVS)


def test_missing_sol_anchors_is_a_dataset_bug():
    row = _row()
    del row["sol_anchors"]
    with pytest.raises(KeyError):
        format_cuda_problem(row, ENVS)


def test_missing_target_hardware_falls_back_to_the_env_sku():
    row = _row()
    del row["target_hardware"]
    assert format_cuda_problem(row, ENVS)["target_hardware"] == "B200"


# --- anchors are hardware-specific: a row must not inherit another GPU's ---


def _kfb_problem(tmp_path):
    """A KFB problem dir carrying the published (B200-measured) solution latencies."""
    (tmp_path / "definition.json").write_text(json.dumps({"name": "p1", "description": "add"}))
    (tmp_path / "workload.jsonl").write_text(json.dumps({"uuid": "w0", "axes": {"N": 16}}) + "\n")
    (tmp_path / "kernel_factory_solution.json").write_text(
        json.dumps(
            {
                "kernel_factory_result": {
                    "is_correct": True,
                    "per_workload": [{"axes": {"N": 16}, "latency_ms": 2.5}],
                }
            }
        )
    )
    return str(tmp_path)


def test_published_anchors_only_reach_rows_for_the_gpu_they_were_measured_on(tmp_path):
    """kernel_factory_solution.json is a B200 measurement, so only B200 rows may use it.

    Scoring a kernel timed on one GPU against a human-best timed on another
    yields a plausible-looking but meaningless performance reward, and nothing
    downstream can detect it: the anchor does not record its hardware.
    """
    pdir = _kfb_problem(tmp_path)
    b200 = kfb_problem_to_gym_seed(pdir, "triton", "B200")["verifier_metadata"]["sol_anchors"]
    assert b200["w0"]["human_best_latency_ms"] == 2.5
    # H100 rows get no anchors and fall back to speedup-over-reference.
    assert kfb_problem_to_gym_seed(pdir, "triton", "H100")["verifier_metadata"]["sol_anchors"] == {}
    assert json.loads(kfb_problem_to_row(pdir, "triton", "H100")["sol_anchors"]) == {}


def test_a_latency_csv_measured_elsewhere_is_refused(tmp_path):
    """A CSV records no hardware, so the caller declares it and a mismatch is refused."""
    pdir = _kfb_problem(tmp_path)
    csv_path = tmp_path / "latencies_b200.csv"
    csv_path.write_text(
        "artifact_id,workload_uuid,sol_latency_ms,human_best_latency_ms\np1,w0,1.0,2.0\n"
    )
    with pytest.raises(ValueError, match="measured on B200.*target H100"):
        kfb_problem_to_gym_seed(pdir, "triton", "H100", sol_latencies_csv=str(csv_path))
    # Declared to match, it is used.
    anchors = kfb_problem_to_gym_seed(
        pdir, "triton", "H100", sol_latencies_csv=str(csv_path), sol_latencies_sku="H100"
    )["verifier_metadata"]["sol_anchors"]
    assert anchors["w0"] == {"human_best_latency_ms": 2.0, "sol_latency_ms": 1.0}
