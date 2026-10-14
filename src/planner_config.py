from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "v0.yaml"


def load_planner_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    if not path.is_absolute():
        path = Path.cwd() / path
        if not path.exists():
            path = DEFAULT_CONFIG_PATH.parent.parent / path
    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}

    planning = config.get("planning_node")
    perception = config.get("perception_node")
    if not isinstance(planning, dict) or not isinstance(perception, dict):
        raise ValueError("Config must contain planning_node and perception_node mappings")

    class_names = perception.get("class_names")
    class_costs = perception.get("class_costs")
    if not isinstance(class_names, list) or not isinstance(class_costs, list):
        raise ValueError("perception_node.class_names and class_costs must be lists")
    if len(class_names) != len(class_costs):
        raise ValueError("perception_node.class_names and class_costs must have equal length")

    for key in ("curr_loc", "goal_loc"):
        value = planning.get(key)
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"planning_node.{key} must contain two coordinates")

    return config
