from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Union

Point = Tuple[int, int]
LanePolygons = Dict[str, List[Point]]


@dataclass
class AppConfig:
    video_source: Union[int, str] = 0
    model_path: str = "yolov8n.pt"
    vehicle_classes: set[str] = field(default_factory=lambda: {"car", "bus", "truck", "motorbike"})
    count_line_y: int = 400
    pixel_per_meter: float = 8.0
    speed_threshold_crash: float = 5.0
    crash_distance_pixels: int = 60
    ocr_min_width: int = 60
    lane_polygons: LanePolygons = field(
        default_factory=lambda: {
            "lane_1": [(0, 480), (320, 300), (380, 300), (0, 480)],
            "lane_2": [(320, 300), (1280, 300), (1280, 480), (380, 300)],
        }
    )
    display_window_name: str = "Vehicle Monitor - LPR & Lane Violation"


DEFAULT_CONFIG = AppConfig()


def _coerce_video_source(value: Union[int, str]) -> Union[int, str]:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return value


def load_config(config_path: str | None = None) -> AppConfig:
    if not config_path:
        return AppConfig()

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    cfg = AppConfig()

    if "video_source" in payload:
        cfg.video_source = _coerce_video_source(payload["video_source"])
    if "model_path" in payload:
        cfg.model_path = str(payload["model_path"])
    if "vehicle_classes" in payload:
        cfg.vehicle_classes = set(payload["vehicle_classes"])
    if "count_line_y" in payload:
        cfg.count_line_y = int(payload["count_line_y"])
    if "pixel_per_meter" in payload:
        cfg.pixel_per_meter = float(payload["pixel_per_meter"])
    if "speed_threshold_crash" in payload:
        cfg.speed_threshold_crash = float(payload["speed_threshold_crash"])
    if "crash_distance_pixels" in payload:
        cfg.crash_distance_pixels = int(payload["crash_distance_pixels"])
    if "ocr_min_width" in payload:
        cfg.ocr_min_width = int(payload["ocr_min_width"])
    if "lane_polygons" in payload:
        lane_polygons: LanePolygons = {}
        for lane_name, points in payload["lane_polygons"].items():
            lane_polygons[lane_name] = [(int(x), int(y)) for x, y in points]
        cfg.lane_polygons = lane_polygons

    return cfg
