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

"""Parse a single-turn (M0) model completion into kernel source code.

The M0 prompt asks the policy for ``<think>...</think>`` reasoning followed by a
single fenced code block holding the kernel (one file). ``check_inline_format``
validates that shape and ``get_code`` extracts the block. ``fence_lang`` is the
markdown fence tag the model writes (e.g. ``python``, ``cpp``), which is *not*
the cudagym ``SupportedLanguages`` value (e.g. ``triton``, ``cuda_cpp``) — the
mapping from a cudagym language to its fence tag lives in ``cudagym_client``.

Multi-file C++/CUDA solutions (kernel.cu + main.cpp pybind wrapper) are not
expressible in one fenced block; those are produced by the agentic path (M1),
where the policy writes files directly in its sandbox.
"""

import re

__all__ = ["check_inline_format", "get_code"]


def check_inline_format(completion: str, fence_lang: str = "python") -> bool:
    r"""Validate a single-turn completion's structure.

    Expects: a ``<think>\\n ... \\n</think>`` block, then anything, then a
    ```` ```<fence_lang>\\n ... \\n``` ```` code fence.

    The fence sub-pattern is kept identical to ``get_code`` (newline right after the
    fence tag and before the closing fence) so that a completion which passes this
    check is guaranteed to be extractable by ``get_code`` -- otherwise a well-formed
    -looking kernel could earn the format reward yet fail extraction and score 0.

    Args:
        completion: the model's raw completion text.
        fence_lang: the markdown fence tag expected after the think block.

    Returns:
        True iff the completion matches the expected think-then-fence shape.
    """
    pattern = rf"^<think>\n.*?\n</think>\n.*?```{fence_lang}\n.*?\n```.*?$"
    return bool(re.search(pattern, completion, re.DOTALL | re.MULTILINE))


def get_code(completion: str, fence_lang: str = "python") -> str:
    """Return the content of the first ```` ```<fence_lang> ... ``` ```` block.

    Args:
        completion: the model's raw completion text.
        fence_lang: the markdown fence tag whose block content to extract.

    Returns:
        The (stripped) code inside the first matching fence.

    Raises:
        ValueError: if no fence with ``fence_lang`` is present — the caller
            records this as a format error (no reward beyond 0).
    """
    match = re.search(rf"```{fence_lang}\n(.*?)\n```", completion, re.DOTALL)
    if match:
        return match.group(1).strip()
    raise ValueError(f"No ```{fence_lang} ... ``` code block found in the completion")
