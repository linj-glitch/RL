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

"""`defaults:`-inheritance YAML loading, dependency-light (omegaconf + stdlib).

Split out of ``nemo_rl.utils.config`` (which imports hydra at module scope) so
submit-time tooling — ``slurm/cudagym_hosting.py`` runs on login nodes without
the training venv — can share the ONE loader instead of carrying a hand-synced
copy.
"""

from pathlib import Path
from typing import Optional, Union, cast

from omegaconf import DictConfig, ListConfig, OmegaConf


def resolve_path(base_path: Path, path: str) -> Path:
    """Resolve a path relative to the base path."""
    if path.startswith("/"):
        return Path(path)
    return base_path / path


def merge_with_override(
    base_config: DictConfig, override_config: DictConfig
) -> DictConfig:
    """Merge configs with support for _override_ marker to completely override sections."""
    for key in list(override_config.keys()):
        if isinstance(override_config[key], DictConfig):
            if override_config[key].get("_override_", False):
                # remove the _override_ marker
                override_config[key].pop("_override_")
                # remove the key from base_config so it won't be merged
                if key in base_config:
                    base_config.pop(key)

    merged_config = cast(DictConfig, OmegaConf.merge(base_config, override_config))
    return merged_config


def load_config_with_inheritance(
    config_path: Union[str, Path],
    base_dir: Optional[Union[str, Path]] = None,
) -> DictConfig:
    """Load a config file with inheritance support.

    Args:
        config_path: Path to the config file
        base_dir: Base directory for resolving relative paths. If None, uses config_path's directory

    Returns:
        Merged config dictionary
    """
    config_path = Path(config_path)
    if base_dir is None:
        base_dir = config_path.parent
    base_dir = Path(base_dir)

    config = OmegaConf.load(config_path)
    assert isinstance(config, DictConfig), (
        "Config must be a Dictionary Config (List Config not supported)"
    )

    # Handle inheritance
    if "defaults" in config:
        defaults = config.pop("defaults")
        if isinstance(defaults, (str, Path)):
            defaults = [defaults]
        elif isinstance(defaults, ListConfig):
            defaults = [str(d) for d in defaults]

        # Load and merge all parent configs
        base_config = OmegaConf.create({})
        for default in defaults:
            parent_path = resolve_path(base_dir, str(default))
            # Use parent's directory as base_dir for resolving its own defaults
            parent_config = load_config_with_inheritance(
                parent_path, parent_path.parent
            )
            base_config = cast(
                DictConfig, merge_with_override(base_config, parent_config)
            )

        # Merge with current config
        config = cast(DictConfig, merge_with_override(base_config, config))

    return config
