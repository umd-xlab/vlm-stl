import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

@dataclass
class RTAConfig:
    config_description: str
    offline_mode: bool
    enable_logging: bool
    stl_rule_file: str
    logging_output_folder: Optional[str] = None

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "RTAConfig":
        with open(path, mode="r", encoding="utf-8") as file:
            data = json.load(file)

        return cls(**data)