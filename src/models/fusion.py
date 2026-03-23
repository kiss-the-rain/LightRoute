from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class FusionOutput:
    scores: list[float]
    route: str


class LightweightFusionHead:
    """Placeholder lightweight fusion/routing head."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
