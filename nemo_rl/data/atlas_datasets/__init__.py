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

"""Datasets for the CUDA kernel-optimization tasks (the ``atlas`` recipe family).

``grpo_cuda_dataset`` loads pre-built KernelFactory-problem row JSONLs for the
single-turn GRPO recipe (``examples/run_grpo_cuda.py``); its module docstring
defines the row schema, including the contract that a row's ``sol_anchors``
were measured on the row's own ``target_hardware``.
"""

from .grpo_cuda_dataset import (
    format_cuda_problem,
    prepare_cuda_dataset,
)

__all__ = [
    "format_cuda_problem",
    "prepare_cuda_dataset",
]
