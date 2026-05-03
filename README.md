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

## Project Structure

```
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

2. Install dependencies.

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

3. Copy and edit config.

```powershell
copy config\config.example.json config\config.json
```

Edit `config/config.json` to match your camera and lane geometry.

4. Run the app.

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

- `plate`
- `lane_violation`
- `crash`

Use:

```powershell
python tracking.py --config config/config.json --save-events events/output.json
```

## Next Improvements

- Add unit tests for config parsing and geometry helpers
- Add tracker persistence and duplicate event suppression logic
- Add optional REST API for integration with dashboards

