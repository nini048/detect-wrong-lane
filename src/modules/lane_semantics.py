from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class LaneSemantic:
    lane_index: int
    label: str
    allowed_classes: List[str]
    rule_source: str
    description: str


class LaneSemanticsInferer:
    """
    Lane semantics are assigned in image / BEV order from left to right.

    Default assumption:
    - lane 0 is the inner lane
    - last lane is the outer lane

    If the dominant flow direction of the camera is reversed, semantics are
    reversed by setting reverse_order=True.
    """

    def _build_templates(self, lane_count: int) -> List[Tuple[str, List[str], str]]:
        if lane_count <= 0:
            return []

        if lane_count == 1:
            return [
                (
                    "mixed",
                    ["motorbike", "car", "truck", "bus"],
                    "Single detected lane. Treat as mixed traffic.",
                )
            ]

        if lane_count == 2:
            return [
                (
                    "preferred_motorbike",
                    ["motorbike", "car"],
                    "Prefer motorbike / smaller vehicles.",
                ),
                (
                    "preferred_car_heavy",
                    ["car", "truck", "bus"],
                    "Prefer larger / faster vehicles.",
                ),
            ]

        templates = [
            (
                "preferred_motorbike",
                ["motorbike", "car"],
                "Inner lane. Prefer motorbike / smaller vehicles.",
            )
        ]

        for _ in range(lane_count - 2):
            templates.append(
                (
                    "mixed_motor_vehicle",
                    ["car", "truck", "bus","motorbike"],
                    "Middle lane. Treat as mixed motor-vehicle lane.",
                )
            )

        templates.append(
            (
                "preferred_car_heavy",
                ["car", "truck", "bus"],
                "Outer lane. Prefer larger / faster vehicles.",
            )
        )
        return templates

    def infer(self, lane_count: int, reverse_order: bool = False) -> List[Dict[str, object]]:
        templates = self._build_templates(lane_count)
        if not templates:
            return []

        if reverse_order:
            templates = list(reversed(templates))

        semantics: List[LaneSemantic] = []
        for lane_index, (label, allowed_classes, description) in enumerate(templates):
            semantics.append(
                LaneSemantic(
                    lane_index=lane_index,
                    label=label,
                    allowed_classes=allowed_classes,
                    rule_source="inferred",
                    description=description,
                )
            )

        return [s.__dict__ for s in semantics]
