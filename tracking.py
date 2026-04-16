import cv2
import math
import time
import numpy as np
import imutils
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import easyocr
from collections import defaultdict

# ---------- CONFIG ----------
VIDEO_SOURCE = 0   # or 0 for webcam, or IP camera URL
MODEL_PATH = "yolov8n.pt"         # yolov8n or yolov8s
VEHICLE_CLASSES = {"car", "bus", "truck", "motorbike"}  # COCO names
COUNT_LINE_Y = 400                # adjust to your video
PIXEL_PER_METER = 8.0            # calibrate: pixels per meter
SPEED_THRESHOLD_CRASH = 5.0      # km/h threshold for near-zero speed
CRASH_DISTANCE_PIXELS = 60       # proximity in pixels to consider a collision
OCR_MIN_WIDTH = 60               # min width of plate crop for OCR
# Define lane polygons (example for 2 lanes). Coordinates are (x,y).
# You MUST edit these polygons to match your camera view.
LANE_POLYGONS = {
    "lane_1": [(0,480),(320,300),(380,300),(0,480)],    # sample poly - fix it
    "lane_2": [(320,300),(1280,300),(1280,480),(380,300)]
}
# ----------------------------

# Load models and utilities
model = YOLO(MODEL_PATH)
tracker = DeepSort(max_age=30)
ocr_reader = easyocr.Reader(["en"], gpu=False)  # set gpu=True if you have GPU and built EasyOCR with CUDA

# Video capture
cap = cv2.VideoCapture(VIDEO_SOURCE)
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

# State containers
vehicle_ids = set()
vehicle_speeds = {}          # id -> speed km/h
frame_positions = {}         # id -> (cx, cy, timestamp)
first_lane_assignment = {}   # id -> lane_id
lane_violations = set()      # track ids that violated
vehicle_plate_cache = {}     # id -> plate text (cache to avoid repeated OCR)
events_log = []              # optional: list of event dicts for saving/logging

# Helper functions
def point_in_poly(pt, poly):
    """pt=(x,y), poly=list of (x,y). Uses cv2.pointPolygonTest."""
    contour = np.array(poly, dtype=np.int32)
    return cv2.pointPolygonTest(contour, pt, False) >= 0

def estimate_speed(prev_pos, cur_pos, dt_seconds):
    """Estimate speed in km/h given pixel positions and elapsed seconds.
       Uses PIXEL_PER_METER for pixel->meter conversion."""
    dx = cur_pos[0] - prev_pos[0]
    dy = cur_pos[1] - prev_pos[1]
    dist_pixels = math.hypot(dx, dy)
    dist_meters = dist_pixels / PIXEL_PER_METER
    if dt_seconds <= 0:
        return 0.0
    speed_m_s = dist_meters / dt_seconds
    return round(speed_m_s * 3.6, 2)

def extract_plate_image(frame, bbox):
    """Try to extract plausible plate area from vehicle bbox.
       bbox = (x1,y1,x2,y2). Returns cropped image or None."""
    x1,y1,x2,y2 = bbox
    w = x2 - x1
    h = y2 - y1
    # Plates usually near bottom half of vehicle bbox — crop bottom 40% region
    plate_y1 = y1 + int(h * 0.55)
    plate_y2 = y2
    plate_x1 = x1
    plate_x2 = x2
    # Expand slightly
    pad_x = int(0.05 * w)
    pad_y = int(0.05 * h)
    plate_x1 = max(0, plate_x1 - pad_x)
    plate_y1 = max(0, plate_y1 - pad_y)
    plate_x2 = min(frame.shape[1]-1, plate_x2 + pad_x)
    plate_y2 = min(frame.shape[0]-1, plate_y2 + pad_y)
    crop = frame[plate_y1:plate_y2, plate_x1:plate_x2]
    if crop.size == 0 or crop.shape[1] < OCR_MIN_WIDTH:
        return None
    return crop

def run_ocr_on_plate(crop):
    """Run EasyOCR on the crop and try to return the best text (alphanumeric)."""
    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    except:
        gray = crop
    # Optional preproc: increase contrast, threshold
    img = cv2.resize(gray, (gray.shape[1]*2, gray.shape[0]*2))
    # run ocr
    results = ocr_reader.readtext(img)
    # choose the longest-looking alphanumeric result
    best = ""
    for (bbox, text, conf) in results:
        # simple filter: keep alnum+ - length >= 4
        cleaned = "".join(ch for ch in text if ch.isalnum())
        if len(cleaned) >= 4 and conf > 0.3:
            if len(cleaned) > len(best):
                best = cleaned
    return best

def draw_lane_polygons(frame):
    for lname, poly in LANE_POLYGONS.items():
        pts = np.array(poly, np.int32).reshape((-1,1,2))
        cv2.polylines(frame, [pts], True, (200,200,0), 2)
        # label at centroid
        M = cv2.moments(pts)
        if M["m00"] != 0:
            cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
            cv2.putText(frame, lname, (cx-30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,0),2)

# Main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h_frame, w_frame = frame.shape[:2]
    # Run detection
    results = model(frame, imgsz=640, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            if label in VEHICLE_CLASSES:
                x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, label))

    # Tracker update
    tracks = tracker.update_tracks(detections, frame=frame)

    # Draw lane polygons
    draw_lane_polygons(frame)
    # Draw count line
    cv2.line(frame, (0, COUNT_LINE_Y), (w_frame, COUNT_LINE_Y), (0, 0, 255), 2)

    # For quick lookup of current positions for crash check
    current_positions = {}

    for track in tracks:
        if not track.is_confirmed():
            continue
        tid = track.track_id
        ltrb = track.to_ltrb()
        x1,y1,x2,y2 = map(int, ltrb)
        cx, cy = (x1 + x2)//2, (y1 + y2)//2
        current_positions[tid] = (cx, cy)

        # Speed estimation
        now = time.time()
        if tid in frame_positions:
            prev_cx, prev_cy, prev_t = frame_positions[tid]
            dt = now - prev_t
            speed_kmph = estimate_speed((prev_cx, prev_cy), (cx, cy), dt)
            vehicle_speeds[tid] = speed_kmph
        else:
            vehicle_speeds[tid] = 0.0
        frame_positions[tid] = (cx, cy, now)

        # Counting
        if abs(cy - COUNT_LINE_Y) < 8:
            vehicle_ids.add(tid)

        # Lane assignment (first time)
        if tid not in first_lane_assignment:
            assigned = None
            for lname, poly in LANE_POLYGONS.items():
                if point_in_poly((cx, cy), poly):
                    assigned = lname
                    first_lane_assignment[tid] = assigned
                    break

        # Lane violation detection:
        # If vehicle was first assigned to lane A and later its centroid is inside lane B != A => violation
        violation_flag = False
        if tid in first_lane_assignment:
            assigned_lane = first_lane_assignment[tid]
            # check which lane it's currently in
            current_lane = None
            for lname, poly in LANE_POLYGONS.items():
                if point_in_poly((cx, cy), poly):
                    current_lane = lname
                    break
            if current_lane is not None and current_lane != assigned_lane:
                violation_flag = True
                lane_violations.add(tid)

        # Crash detection (nearby vehicle + low speed)
        crash_flag = False
        my_speed = vehicle_speeds.get(tid, 0)
        if my_speed < SPEED_THRESHOLD_CRASH:
            for other_tid, (ox, oy, ot) in frame_positions.items():
                if other_tid == tid: continue
                if math.hypot(cx - ox, cy - oy) < CRASH_DISTANCE_PIXELS:
                    crash_flag = True
                    # log crash event
                    events_log.append({
                        "time": now, "type": "crash", "id": tid, "other_id": other_tid,
                        "pos": (cx, cy)})
                    break

        # License plate recognition (attempt once per vehicle or if not found)
        plate_text = vehicle_plate_cache.get(tid, "")
        if plate_text == "":
            plate_crop = extract_plate_image(frame, (x1,y1,x2,y2))
            if plate_crop is not None:
                plate_text = run_ocr_on_plate(plate_crop)
                if plate_text:
                    vehicle_plate_cache[tid] = plate_text
                    events_log.append({"time": now, "type": "plate", "id": tid, "plate": plate_text})

        # Draw bounding box and labels
        color = (0,255,0)
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        label = f"ID {tid} {vehicle_speeds.get(tid,0)} km/h"
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color,2)
        if plate_text:
            cv2.putText(frame, f"Plate: {plate_text}", (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0),2)
        if violation_flag:
            cv2.putText(frame, "LANE VIOLATION", (x1, y2+40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
            events_log.append({"time": now, "type":"lane_violation","id":tid,"lane_from": first_lane_assignment.get(tid), "pos":(cx,cy)})
        if crash_flag:
            cv2.putText(frame, "CRASH!", (x1, y2+60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255),3)

    # Draw counts and stats
    cv2.putText(frame, f"Count: {len(vehicle_ids)}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0),3)
    cv2.putText(frame, f"Violations: {len(lane_violations)}", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255),2)

    cv2.imshow("Vehicle Monitor - LPR & Lane Violation", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# optionally save events_log to json or push to Firebase here
cap.release()
cv2.destroyAllWindows()
