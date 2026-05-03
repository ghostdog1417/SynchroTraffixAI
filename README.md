# SynchroTraffixAI

Real-time traffic monitoring and analytics using computer vision.

## Features

- Vehicle detection with YOLOv8
- Multi-object tracking with DeepSORT
- Real-time speed estimation (pixel-to-meter calibrated)
- Vehicle counting via configurable counting line
- Lane violation detection using polygonal lane regions
- Near-collision crash heuristic detection
- License plate OCR with EasyOCR
- Event export to JSON

## Tech Stack

### Core Dependencies

| Component | Library | Version | Purpose |
|-----------|---------|---------|---------|

| **Computer Vision** | OpenCV | 4.10+ | Frame capture, image processing, lane polygon detection |
| **Object Detection** | YOLOv8 | via Ultralytics | Real-time vehicle detection (nano model for speed) |
| **Multi-Object Tracking** | DeepSORT | 1.3.2+ | Persistent vehicle ID tracking across frames |
| **OCR** | EasyOCR | 1.7.1+ | License plate text recognition |
| **Deep Learning** | PyTorch | 2.3.0+ | Neural network inference backend |
| **Vision Models** | TorchVision | 0.18.0+ | Pre-trained model utilities |
| **Numerical Computing** | NumPy | 1.26.0+ | Array operations, speed calculations |
| **Image Processing** | Pillow | 7.1.2+ | Image manipulation utilities |
| **Scientific Computing** | SciPy | 1.4.1+ | Advanced numerical algorithms |
| **Data Processing** | Polars | 0.20.0+ | High-performance data handling |

### Architecture

```text
Capture → Detection → Tracking → Analysis → Visualization → Logging
  (OpenCV) (YOLOv8) (DeepSORT) (Custom) (OpenCV) (JSON)
```

**Pipeline Flow:**

1. **Capture**: Read frames from camera/video using OpenCV (DirectShow backend on Windows)
2. **Detection**: Run YOLOv8 nano model on each frame to find vehicles
3. **Tracking**: DeepSORT maintains stable IDs across frames
4. **Analysis**:
   - Speed estimation using centroid motion and calibrated pixel-to-meter ratio
   - Lane assignment via point-in-polygon tests (OpenCV)
   - Crash detection based on proximity + low speed heuristic
   - Plate OCR on vehicle bounding box regions
5. **Visualization**: Overlay detections, speeds, and alerts on frame
6. **Logging**: Export events (plates, violations, crashes) to JSON

### Development Stack

- **Language**: Python 3.10+
- **Package Management**: pip
- **Packaging**: setuptools, pyproject.toml
- **Build**: No compilation required (pure Python + pre-built wheels)
- **Virtual Environment**: venv/virtualenv
- **Configuration**: JSON-based config files
- **CLI**: argparse

## Project Structure

```text
SynchroTraffixAI/
 config/
  config.example.json
 src/
  synchrotraffixai/
   __init__.py
   __main__.py
   config.py
   events.py
   main.py
   pipeline.py
 .gitignore
 LICENSE
 pyproject.toml
 README.md
 requirements.txt
 tracking.py
```

## Requirements

- Python 3.10+
- Webcam, video file, or IP camera stream
- Optional GPU for faster inference

## Quick Start

1. Create and activate a virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

1. Install dependencies.

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

1. Copy and edit config.

```powershell
copy config\config.example.json config\config.json
```

Edit `config/config.json` to match your camera and lane geometry.

1. Run the app.

```powershell
python tracking.py --config config/config.json
```

or

```powershell
python -m synchrotraffixai --config config/config.json
```

## CLI Options

```text
--config <path>         JSON configuration file
--video-source <value>  Camera index, video file path, or stream URL
--model-path <path>     YOLO model file path (for example yolov8n.pt)
--save-events <path>    Write detected events to JSON
--headless              Disable preview window
```

## Configuration Notes

- `pixel_per_meter` must be calibrated for your scene.
- `lane_polygons` must be updated to your camera perspective.
- Default `vehicle_classes` are COCO classes: `car`, `bus`, `truck`, `motorbike`.

## Event Output

Events can include:

- `plate` – License plate detected and recognized
- `lane_violation` – Vehicle crossed from assigned lane to another
- `crash` – Potential collision detected (proximity + low speed)

Use:

```powershell
python tracking.py --config config/config.json --save-events events/output.json
```

Output example (`events/output.json`):

```json
[
  {
    "time": 1714815234.567,
    "type": "plate",
    "id": 1,
    "plate": "ABC1234"
  },
  {
    "time": 1714815245.123,
    "type": "lane_violation",
    "id": 2,
    "lane_from": "lane_1",
    "pos": [320, 400]
  },
  {
    "time": 1714815250.890,
    "type": "crash",
    "id": 1,
    "other_id": 3,
    "pos": [400, 350]
  }
]
```

## Performance Tips

- **Reduce frame size** for faster processing (adjust camera resolution)
- **Use YOLOv8-nano** (default) for CPU; upgrade to `yolov8s.pt` or `yolov8m.pt` for GPU
- **Enable GPU** by building PyTorch with CUDA support
- **Increase `max_age` in DeepSORT** (default 30) for longer tracking history
- **Adjust `imgsz`** in detection: smaller = faster, larger = more accurate

## Calibration Guide

### Pixel-to-Meter Ratio

1. Identify a known distance in your camera view (e.g., lane width = 3.5 meters)
2. Measure the pixel distance in the frame
3. Set `pixel_per_meter = pixel_distance / meter_distance` in config

### Lane Polygons

1. Run the app and view the preview window
2. Identify lane boundaries
3. Update `lane_polygons` coordinates in `config/config.json` (format: `[[x1, y1], [x2, y2], ...]`)
4. Restart and verify polygons overlay correctly

### Counting Line

- `count_line_y`: horizontal line position in pixels where vehicles are counted
- Set this to the center of your target detection zone

## Troubleshooting

| Issue | Solution |
|-------|----------|

| Camera not opening | Try different `--video-source` value (0, 1, 2...) or file path |
| No frames detected | Check camera permissions; restart camera driver |
| Slow inference | Reduce frame resolution; switch to CPU-optimized YOLOv8-nano |
| Inaccurate speeds | Calibrate `pixel_per_meter` with known distance |
| Poor tracking | Increase `max_age` in tracker config; improve lighting |
| Missed plates | Ensure license plate area is in vehicle bounding box; adjust OCR thresholds |

## Next Steps & Improvements

**Current Roadmap:**

- [ ] Add unit tests for geometry and speed estimation
- [ ] Implement tracker persistence (save/load tracking state)
- [ ] Add duplicate event suppression logic
- [ ] Build REST API for dashboard integration
- [ ] Add support for multi-camera streams
- [ ] Implement zone-based analytics (entry/exit counts per region)
- [ ] Add alerting system (email/webhook on violations)
- [ ] Export statistics and heatmaps

**Performance Enhancements:**

- [ ] Multi-threading for parallel frame processing
- [ ] GPU acceleration for YOLOv8 and DeepSORT
- [ ] Model quantization for edge deployment
- [ ] Frame skipping and adaptive inference

**Extensibility:**

- [ ] Pluggable detector/tracker backends
- [ ] Custom event handlers
- [ ] Database integration for event persistence

## API Reference

### TrafficMonitor Class

```python
from synchrotraffixai.config import AppConfig
from synchrotraffixai.pipeline import TrafficMonitor

# Initialize
config = AppConfig(video_source=0, model_path="yolov8n.pt")
monitor = TrafficMonitor(config)

# Run and collect events
events = monitor.run(headless=False)

# Export events
events.dump_json("output/events.json")
```

### Configuration Object

See [src/synchrotraffixai/config.py](src/synchrotraffixai/config.py) for `AppConfig` dataclass with all configurable parameters.

## License

See [LICENSE](LICENSE) file.

## Contributing

Pull requests and issue reports welcome. Please test changes locally before submitting.

## Acknowledgments

- **YOLOv8**: Ultralytics for state-of-the-art detection
- **DeepSORT**: Nicolai Wojke et al. for multi-object tracking
- **EasyOCR**: Jaided AI for accessible OCR
- **OpenCV**: Computer Vision Library
