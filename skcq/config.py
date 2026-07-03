from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from skcq.clustering import CodebookParams, LayerOverride

__all__ = ["CodebookParams", "LayerOverride", "ExperimentConfig"]


class ExperimentConfig(BaseModel):
    model_id: str = Field(default="Qwen/Qwen3.6-35B-A3B")
    defaults: CodebookParams = Field(default_factory=CodebookParams)
    layer_overrides: dict[int, LayerOverride] = Field(default_factory=dict)
    eval_samples: int = Field(
        default=1000, description="Number of C4 validation samples for perplexity"
    )
    output_dir: Path = Field(default=Path("codebooks"))

    def params_for_layer(self, layer_idx: int) -> CodebookParams:
        """Get effective params for a layer, merging defaults with overrides."""
        if layer_idx not in self.layer_overrides:
            return self.defaults

        override = self.layer_overrides[layer_idx]
        return self.defaults.model_copy(
            update={k: v for k, v in override.model_dump().items() if v is not None}
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        with open(path) as f:
            data: dict[str, Any] = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)
