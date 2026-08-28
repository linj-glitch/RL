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

"""Build cuda_smoke{,_async} train/val JSONL from solswarm kernel_factory examples.

Row format mirrors Gym's resources_servers/cudagym/data/example.jsonl:
  responses_create_params.input  — the user prompt
  verifier_metadata              — language, target_hardware,
                                   destination_passing_style, definition
                                   (definition.json), workloads
                                   (workload.jsonl rows). sol_anchors omitted:
                                   optional per app.py, and anchors must be
                                   measured on the row's own hardware — none
                                   exist for this cluster yet, so scoring
                                   falls back to speedup vs reference.

Single-GPU problems only (multi_gpu_* excluded from the smoke set), Triton
rows for every problem except cpp_cublas_matmul (a C++/CUBLAS problem, kept
language=cuda).

Usage: python build_cuda_smoke_dataset.py <kernel_factory_dir> <out_root> <target_hardware>
"""

import json
import sys
from pathlib import Path

SKIP = {"multi_gpu_all_reduce", "multi_gpu_ddp_gradient", "multi_gpu_rank_sharded_inputs"}
LANG_OVERRIDES = {"cpp_cublas_matmul": "cublas"}


def build_row(problem_dir: Path, target_hardware: str) -> dict:
    definition = json.loads((problem_dir / "definition.json").read_text())
    workloads = [
        json.loads(line)
        for line in (problem_dir / "workload.jsonl").read_text().splitlines()
        if line.strip()
    ]
    name = definition["name"]
    desc = (definition.get("description") or definition.get("op_type") or name).strip().rstrip(".")
    inputs = ", ".join(definition.get("inputs", {}).keys())
    outputs = list(definition.get("outputs", {}).keys())
    outs = outputs[0] if len(outputs) == 1 else "(" + ", ".join(outputs) + ")"
    prompt = (
        f"Optimize a fast GPU kernel for the `{name}` problem: {desc}. "
        f"The full spec (definition + workloads) is in ./problem/. "
        f"Implement `run({inputs})` returning {outs}."
    )
    return {
        "responses_create_params": {"input": [{"role": "user", "content": prompt}]},
        # Routes the row to the cudagym_cuda_agent config's agent at dispatch
        # (nemo_gym.py keys rollouts on row["agent_ref"]["name"]; README: rows
        # carrying agent_ref are directly usable without ng_prepare_data).
        "agent_ref": {"type": "responses_api_agents", "name": "cudagym_cuda_agent"},
        "verifier_metadata": {
            "language": LANG_OVERRIDES.get(problem_dir.name, "triton"),
            "target_hardware": target_hardware,
            "destination_passing_style": False,
            "definition": definition,
            "workloads": workloads,
        },
    }


def cycle(rows: list[dict], n: int) -> list[dict]:
    return [rows[i % len(rows)] for i in range(n)]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{path}: {len(rows)} rows")


def main() -> None:
    kf_dir, out_root, hw = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    problems = sorted(
        d for d in kf_dir.iterdir()
        if d.is_dir() and d.name not in SKIP and (d / "definition.json").exists()
    )
    rows = [build_row(p, hw) for p in problems]
    print(f"{len(rows)} problems: {', '.join(p.name for p in problems)}")
    # Sync smoke: 12-row train (problems cycled), val = one pass over the set.
    write_jsonl(out_root / "cuda_smoke" / "train.jsonl", cycle(rows, 12))
    write_jsonl(out_root / "cuda_smoke" / "val.jsonl", rows)
    # Async smoke: 48 rows = 4 steps x 8 prompts + one 8-prompt lookahead + headroom.
    write_jsonl(out_root / "cuda_smoke_async" / "train.jsonl", cycle(rows, 48))
    write_jsonl(out_root / "cuda_smoke_async" / "val.jsonl", rows)


if __name__ == "__main__":
    main()
