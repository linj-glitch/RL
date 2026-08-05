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

from types import SimpleNamespace

import pytest

from nemo_rl.data.atlas_datasets.grpo_cuda_dataset import format_cuda_problem

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
