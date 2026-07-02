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
"""Run SFT on a cluster.

Usage:
    python submit_sft.py --exp-name <exp-name> --config <config-path> --cluster <cluster-name>

    # atlas c1 32b
    python submit_sft.py --exp-name atlas_c1_32b --config sft_megatron_qwen3-32b.yaml --cluster dfw --num-nodes 16

"""

import argparse
import os
from pathlib import Path

from remote_utils import (
    SSHTunnel,
    package_code,
    get_available_clusters,
    get_available_configs,
    load_cluster_config,
    validate_cluster_paths,
    upload_text_as_file,
    fill_template,
)

CONFIG_PATH = Path(__file__).parent / "examples" / "configs" / "recipes" / "atlas"
CLUSTER_CONFIG_PATH = Path(__file__).parent / "slurm" / "clusters"
SBATCH_TEMPLATE_PATH = Path(__file__).parent / "slurm" / "sft" / "sft.sh"


def launch_jobs(
    ssh_tunnel,
    code_upload_path,
    convert: int = None,
    interactive: bool = False,
    num_jobs: int = 1,
):
    for i in range(num_jobs):
        launch_cmd = f"cd {code_upload_path} && "
        if convert is not None:
            launch_cmd += f"CONVERT_STEP={convert} "
        launch_cmd += "bash ../run.sh"
        if interactive:
            launch_cmd += " -i"

        print(f"🚀 Running sbatch script ({i + 1}/{num_jobs}): {launch_cmd}")
        _, stdout, stderr = ssh_tunnel.run_command(launch_cmd)
        print(stdout)
        if stderr:
            raise RuntimeError("Error running sbatch script: " + stderr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", "-e", default="debug", type=str)
    parser.add_argument(
        "--config",
        type=str,
        default="sft_megatron_qwen3-32b.yaml",
        choices=get_available_configs(CONFIG_PATH, "sft*.yaml", return_stems=False),
    )
    parser.add_argument(
        "--cluster",
        "-c",
        type=str,
        default="hsg",
        choices=get_available_clusters(CLUSTER_CONFIG_PATH),
    )
    parser.add_argument(
        "--convert",
        type=int,
        default=None,
        help="Launch a conversion job to convert the checkpoint to HF. Please specify the step number to convert.",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Launch the command interactively",
    )
    parser.add_argument(
        "--extra-config-opts", type=str, default="", help="Extra config options"
    )
    parser.add_argument(
        "--time", type=str, default="04:00:00", help="Time limit for the job"
    )
    parser.add_argument(
        "--num-nodes", type=int, default=16, help="Number of nodes for the job"
    )
    parser.add_argument(
        "--num-jobs",
        type=int,
        default=1,
        help="Number of sequential jobs to run. This is useful for SFT runs that take longer than 4 hours to run.",
    )
    parser.add_argument(
        "--only-upload",
        action="store_true",
        help="Only upload the code and sbatch script",
    )
    parser.add_argument(
        "--skip-commit-check",
        action="store_true",
        help="Skip the commit check",
    )
    args = parser.parse_args()

    if args.convert:
        if args.num_jobs > 1:
            parser.error("Cannot specify --num-jobs when --convert is specified")
        if args.time != "04:00:00":
            parser.error("Cannot specify --time when --convert is specified")
        args.time = "00:30:00"

    # Load cluster config with env overrides applied and resolved
    cluster_config = load_cluster_config(CLUSTER_CONFIG_PATH, args.cluster)

    # Guard against mounting home directories which can slow down clusters
    validate_cluster_paths(cluster_config["paths"])

    output_dir = Path(cluster_config["paths"]["output"]) / args.exp_name
    code_upload_path = output_dir / "code"
    ssh_tunnel = SSHTunnel(cluster_config["hostname"])
    if args.convert is not None:
        print("🔄 Skipping code upload for conversion job")
    else:
        package_code(
            ssh_tunnel, code_upload_path, skip_commit_check=args.skip_commit_check
        )

    # Upload sbatch script with custom variables
    print(
        f"⬆️  Uploading sbatch script to {cluster_config['hostname']} at {output_dir / 'run.sh'}..."
    )
    sbatch_script = SBATCH_TEMPLATE_PATH.read_text()
    # Some clusters have different gpu hw and node topology so include those overrides in the config
    extra_config_opts = (
        args.extra_config_opts + " " + cluster_config["extra_config_opts"]
    ).strip()

    sbatch_vars = {
        "EXP_NAME": args.exp_name,
        "CONFIG_NAME": Path(args.config).resolve().relative_to(Path(__file__).parent),
        "EXTRA_CONFIG_OPTS": extra_config_opts,
        "TIME": args.time,
        "NUM_NODES": args.num_nodes,
        "HF_TOKEN": os.getenv("HF_TOKEN"),
        "WANDB_API_KEY": os.getenv("WANDB_API_KEY"),
        "OUTPUT_DIR": output_dir,
        "GPUS_PER_NODE": cluster_config["gpus_per_node"],
        "SKIP_GRES_ARG": "1" if args.cluster == "eos" else "",
    } | {**cluster_config["paths"]}

    for k, v in sbatch_vars.items():
        sbatch_script = fill_template(
            sbatch_script, k, v
        )  # replace every DEFAULT_<VAR> token in the script

    upload_text_as_file(ssh_tunnel, sbatch_script, output_dir / "run.sh")
    print("✅ Sbatch script uploaded")

    if args.only_upload:
        print("✅ Skipping launch")
        return

    # Submit the sbatch script
    launch_jobs(
        ssh_tunnel, code_upload_path, args.convert, args.interactive, args.num_jobs
    )


if __name__ == "__main__":
    main()
