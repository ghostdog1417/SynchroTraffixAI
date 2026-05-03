from __future__ import annotations

import argparse

from .config import load_config
from .pipeline import TrafficMonitor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SynchroTraffixAI traffic monitoring")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    parser.add_argument("--video-source", type=str, default=None, help="Override video source (0, file path, or URL)")
    parser.add_argument("--model-path", type=str, default=None, help="Override YOLO model path")
    parser.add_argument("--save-events", type=str, default=None, help="Write events to JSON file")
    parser.add_argument("--headless", action="store_true", help="Disable preview window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.video_source is not None:
        cfg.video_source = int(args.video_source) if args.video_source.isdigit() else args.video_source
    if args.model_path is not None:
        cfg.model_path = args.model_path

    monitor = TrafficMonitor(cfg)
    events = monitor.run(headless=args.headless)

    if args.save_events:
        events.dump_json(args.save_events)


if __name__ == "__main__":
    main()
