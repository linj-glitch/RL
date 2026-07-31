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

"""YAML config loading with ``defaults:``-based inheritance.

A config file may name one or more parent configs under a top-level
``defaults:`` key (a single path or a list of paths, each absolute or relative
to the file's own directory). ``load_config_with_inheritance`` loads each
parent recursively, merges the parents left to right, and merges the child
file last, so on any key conflict a later parent overrides an earlier one and
the child overrides all parents. A mapping section marked ``_override_: true``
replaces the inherited section wholesale instead of being deep-merged.

This module depends only on ``omegaconf`` and the standard library. It is a
standalone mirror of the loader in ``nemo_rl.utils.config`` (which imports
hydra at module scope) so tooling that runs outside the training virtualenv
(``slurm/cudagym_hosting.py`` on cluster login nodes) can import it without
touching that upstream file. If the upstream loader's semantics change, update
this copy to match.
"""

from pathlib import Path
from typing import Optional, Union, cast

from omegaconf import DictConfig, ListConfig, OmegaConf


def resolve_path(base_path: Path, path: str) -> Path:
    """Resolve ``path`` against ``base_path``, returning absolute paths unchanged."""
    if path.startswith("/"):
        return Path(path)
    return base_path / path


def merge_with_override(
    base_config: DictConfig, override_config: DictConfig
) -> DictConfig:
    """Merge two configs, honoring the ``_override_`` section-replacement marker.

    Performs a standard ``OmegaConf.merge(base_config, override_config)``:
    nested mappings are merged recursively, and on key conflicts the value
    from ``override_config`` wins. Before merging, any top-level mapping in
    ``override_config`` that carries ``_override_: true`` has the marker
    stripped and the same-named section removed from ``base_config``, so that
    section is taken wholesale from ``override_config`` instead of being
    deep-merged with inherited keys.

    Args:
        base_config: The config being inherited from.
        override_config: The config whose values take precedence.

    Returns:
        The merged config.
    """
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
    """Load a YAML config file, resolving its ``defaults:`` inheritance chain.

    If the file has a top-level ``defaults:`` key (a single path or a list of
    paths), each listed parent is loaded recursively with this same function;
    a parent's own ``defaults:`` entries resolve relative to that parent's
    directory. The parents are merged left to right and the current file is
    merged on top, so on any key conflict a later parent overrides an earlier
    one and the current file overrides all parents. Sections marked
    ``_override_: true`` replace the inherited section instead of deep-merging
    (see ``merge_with_override``).

    Args:
        config_path: Path to the config file.
        base_dir: Base directory for resolving relative ``defaults:`` entries.
            If None, uses ``config_path``'s directory.

    Returns:
        The merged config, with the ``defaults`` key consumed.
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
