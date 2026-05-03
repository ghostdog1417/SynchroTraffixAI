from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class EventStore:
    events: List[Dict[str, Any]]

    def __init__(self) -> None:
        self.events = []

    def append(self, event: Dict[str, Any]) -> None:
        self.events.append(event)

    def dump_json(self, output_path: str) -> None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.events, indent=2), encoding="utf-8")
