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
"""Tests for the single-turn completion parser.

Key invariant: ``check_inline_format`` must agree with ``get_code`` -- a
completion that passes the format gate must be extractable, otherwise a
well-formed-looking kernel earns the format reward yet scores 0 on extraction.
"""

import pytest

from nemo_rl.environments.atlas.llm_response_parsing import (
    check_inline_format,
    get_code,
)

_WELL_FORMED = (
    "<think>\nreason about the kernel\n</think>\n"
    "Here is the kernel:\n```python\nimport torch\n\n\ndef run(x):\n    return x\n```\n"
)


def test_well_formed_passes_and_extracts():
    assert check_inline_format(_WELL_FORMED) is True
    assert get_code(_WELL_FORMED).startswith("import torch")


def test_no_newline_fence_fails_both():
    # A fence without the surrounding newlines used to pass check_inline_format
    # (lenient) while get_code (strict) raised -- the format/extract mismatch.
    completion = "<think>\nr\n</think>\nintro ```python code```"
    assert check_inline_format(completion) is False
    with pytest.raises(ValueError):
        get_code(completion)


def test_missing_think_block_fails():
    completion = "```python\nx = 1\n```"
    assert check_inline_format(completion) is False


def test_cpp_fence():
    completion = (
        "<think>\nx\n</think>\nblah\n```cpp\nint main() { return 0; }\n```\ntrail"
    )
    assert check_inline_format(completion, "cpp") is True
    assert get_code(completion, "cpp") == "int main() { return 0; }"


def test_draft_fence_inside_think_is_not_extracted():
    # A draft kernel fenced INSIDE <think> must never be the one evaluated:
    # get_code anchors extraction after the </think> close (same anchor as
    # check_inline_format), so the FINAL kernel is compiled/rewarded.
    completion = (
        "<think>\nlet me sketch:\n```python\ndraft = 'WRONG KERNEL'\n```\n"
        "no, better plan\n</think>\n"
        "Final:\n```python\nfinal = 'RIGHT KERNEL'\n```\n"
    )
    assert check_inline_format(completion) is True
    assert get_code(completion) == "final = 'RIGHT KERNEL'"


def test_fence_only_inside_think_fails_extraction():
    # If the only fence is inside the think block, the format check fails and
    # extraction must raise (never silently evaluate a draft).
    completion = "<think>\n```python\ndraft\n```\nstill thinking\n</think>\ndone."
    assert check_inline_format(completion) is False
    with pytest.raises(ValueError):
        get_code(completion)


@pytest.mark.parametrize(
    "completion",
    [
        _WELL_FORMED,
        "<think>\nr\n</think>\nintro ```python code```",  # no-newline fence
        "```python\nx = 1\n```",  # no think block
        "<think>\nr\n</think>\nno fence at all here",
        "plain text, nothing structured",
    ],
)
def test_format_check_implies_extractable(completion):
    """If check_inline_format passes, get_code must succeed (no format/extract gap)."""
    if check_inline_format(completion):
        # Must not raise.
        assert isinstance(get_code(completion), str)
