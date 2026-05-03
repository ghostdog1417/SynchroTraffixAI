from __future__ import annotations

import math
import time
from typing import Dict, Tuple

import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from easyocr import Reader
from ultralytics import YOLO

from .config import AppConfig
from .events import EventStore


class TrafficMonitor:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.model = YOLO(config.model_path)
        self.tracker = DeepSort(max_age=30)
        self.ocr_reader = Reader(["en"], gpu=False)

        self.vehicle_ids: set[int] = set()
        self.vehicle_speeds: Dict[int, float] = {}
        self.frame_positions: Dict[int, Tuple[int, int, float]] = {}
        self.first_lane_assignment: Dict[int, str] = {}
        self.lane_violations: set[int] = set()
        self.vehicle_plate_cache: Dict[int, str] = {}
        self.events = EventStore()

    def point_in_poly(self, pt: Tuple[int, int], poly: list[Tuple[int, int]]) -> bool:
        contour = np.array(poly, dtype=np.int32)
        return cv2.pointPolygonTest(contour, pt, False) >= 0

    def estimate_speed(
        self,
        prev_pos: Tuple[int, int],
        cur_pos: Tuple[int, int],
        dt_seconds: float,
    ) -> float:
        dx = cur_pos[0] - prev_pos[0]
        dy = cur_pos[1] - prev_pos[1]
        dist_pixels = math.hypot(dx, dy)
        dist_meters = dist_pixels / self.config.pixel_per_meter
        if dt_seconds <= 0:
            return 0.0
        speed_m_s = dist_meters / dt_seconds
        return round(speed_m_s * 3.6, 2)

    def extract_plate_image(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> np.ndarray | None:
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1

        plate_y1 = y1 + int(h * 0.55)
        plate_y2 = y2
        plate_x1 = x1
        plate_x2 = x2

        pad_x = int(0.05 * w)
        pad_y = int(0.05 * h)
        plate_x1 = max(0, plate_x1 - pad_x)
        plate_y1 = max(0, plate_y1 - pad_y)
        plate_x2 = min(frame.shape[1] - 1, plate_x2 + pad_x)
        plate_y2 = min(frame.shape[0] - 1, plate_y2 + pad_y)

        crop = frame[plate_y1:plate_y2, plate_x1:plate_x2]
        if crop.size == 0 or crop.shape[1] < self.config.ocr_min_width:
            return None
        return crop

    def run_ocr_on_plate(self, crop: np.ndarray) -> str:
        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        except cv2.error:
            gray = crop

        img = cv2.resize(gray, (gray.shape[1] * 2, gray.shape[0] * 2))
        results = self.ocr_reader.readtext(img)

        best = ""
        for _, text, conf in results:
            cleaned = "".join(ch for ch in text if ch.isalnum())
            if len(cleaned) >= 4 and conf > 0.3 and len(cleaned) > len(best):
                best = cleaned
        return best

    def draw_lane_polygons(self, frame: np.ndarray) -> None:
        for lane_name, poly in self.config.lane_polygons.items():
            pts = np.array(poly, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, (200, 200, 0), 2)
            moments = cv2.moments(pts)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                cv2.putText(
                    frame,
                    lane_name,
                    (cx - 30, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 0),
                    2,
                )

    def run(self, headless: bool = False) -> EventStore:
        # On Windows, try DirectShow backend first (more reliable than MSMF)
        if isinstance(self.config.video_source, int):
            cap = cv2.VideoCapture(self.config.video_source, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(self.config.video_source)
        
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {self.config.video_source}")

        try:
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    if frame_count == 0:
                        print("WARNING: Unable to read frames from camera. Check camera permissions or try a different camera index.")
                    break
                
                frame_count += 1

                h_frame, w_frame = frame.shape[:2]
                results = self.model(frame, imgsz=640, verbose=False)
                detections = []

                for result in results:
                    for box in result.boxes:
                        cls = int(box.cls[0])
                        label = self.model.names[cls]
                        if label in self.config.vehicle_classes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            conf = float(box.conf[0])
                            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, label))

                tracks = self.tracker.update_tracks(detections, frame=frame)

                self.draw_lane_polygons(frame)
                cv2.line(
                    frame,
                    (0, self.config.count_line_y),
                    (w_frame, self.config.count_line_y),
                    (0, 0, 255),
                    2,
                )

                for track in tracks:
                    if not track.is_confirmed():
                        continue

                    tid = int(track.track_id)
                    x1, y1, x2, y2 = map(int, track.to_ltrb())
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    now = time.time()
                    if tid in self.frame_positions:
                        prev_cx, prev_cy, prev_t = self.frame_positions[tid]
                        dt = now - prev_t
                        self.vehicle_speeds[tid] = self.estimate_speed((prev_cx, prev_cy), (cx, cy), dt)
                    else:
                        self.vehicle_speeds[tid] = 0.0
                    self.frame_positions[tid] = (cx, cy, now)

                    if abs(cy - self.config.count_line_y) < 8:
                        self.vehicle_ids.add(tid)

                    if tid not in self.first_lane_assignment:
                        for lane_name, poly in self.config.lane_polygons.items():
                            if self.point_in_poly((cx, cy), poly):
                                self.first_lane_assignment[tid] = lane_name
                                break

                    violation_flag = False
                    assigned_lane = self.first_lane_assignment.get(tid)
                    if assigned_lane:
                        current_lane = None
                        for lane_name, poly in self.config.lane_polygons.items():
                            if self.point_in_poly((cx, cy), poly):
                                current_lane = lane_name
                                break
                        if current_lane and current_lane != assigned_lane:
                            violation_flag = True
                            self.lane_violations.add(tid)

                    crash_flag = False
                    my_speed = self.vehicle_speeds.get(tid, 0.0)
                    if my_speed < self.config.speed_threshold_crash:
                        for other_tid, (ox, oy, _) in self.frame_positions.items():
                            if other_tid == tid:
                                continue
                            if math.hypot(cx - ox, cy - oy) < self.config.crash_distance_pixels:
                                crash_flag = True
                                self.events.append(
                                    {
                                        "time": now,
                                        "type": "crash",
                                        "id": tid,
                                        "other_id": other_tid,
                                        "pos": [cx, cy],
                                    }
                                )
                                break

                    plate_text = self.vehicle_plate_cache.get(tid, "")
                    if not plate_text:
                        plate_crop = self.extract_plate_image(frame, (x1, y1, x2, y2))
                        if plate_crop is not None:
                            plate_text = self.run_ocr_on_plate(plate_crop)
                            if plate_text:
                                self.vehicle_plate_cache[tid] = plate_text
                                self.events.append(
                                    {
                                        "time": now,
                                        "type": "plate",
                                        "id": tid,
                                        "plate": plate_text,
                                    }
                                )

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"ID {tid} {self.vehicle_speeds.get(tid, 0.0)} km/h",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
                    if plate_text:
                        cv2.putText(
                            frame,
                            f"Plate: {plate_text}",
                            (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 0),
                            2,
                        )
                    if violation_flag:
                        cv2.putText(
                            frame,
                            "LANE VIOLATION",
                            (x1, y2 + 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2,
                        )
                        self.events.append(
                            {
                                "time": now,
                                "type": "lane_violation",
                                "id": tid,
                                "lane_from": assigned_lane,
                                "pos": [cx, cy],
                            }
                        )
                    if crash_flag:
                        cv2.putText(
                            frame,
                            "CRASH!",
                            (x1, y2 + 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            3,
                        )

                cv2.putText(
                    frame,
                    f"Count: {len(self.vehicle_ids)}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 0, 0),
                    3,
                )
                cv2.putText(
                    frame,
                    f"Violations: {len(self.lane_violations)}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                )

                if not headless:
                    cv2.imshow(self.config.display_window_name, frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            cap.release()
            if not headless:
                cv2.destroyAllWindows()

        return self.events
