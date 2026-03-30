import cv2
import numpy as np
from ultralytics import YOLO
import os
import subprocess

model = YOLO("yolov8n.pt")

video_path = "test_video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Error: Could not open video.")
    exit()

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps

print(f"📹 Video: {total_frames} frames | {fps} fps | {duration:.1f} seconds")

# Event tracking
prev_positions   = {}
prev_directions  = {}
last_event_frame = {}

# Thresholds
SPEED_THRESHOLD     = 15
DIRECTION_THRESHOLD = 90
CLUSTER_DISTANCE    = 80

# Highlight scoring
frame_scores = {}  # frame -> excitement score
frame_count  = 0

def get_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def get_speed(p1, p2):
    return np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)

def get_direction(p1, p2):
    return np.degrees(np.arctan2(p2[1]-p1[1], p2[0]-p1[0]))

def angle_diff(a1, a2):
    diff = abs(a1 - a2)
    return min(diff, 360 - diff)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, verbose=False,
                          tracker="bytetrack.yaml", conf=0.3, iou=0.5)

    score = 0  # excitement score for this frame

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids   = results[0].boxes.id.cpu().numpy().astype(int)

        # More players detected = more exciting
        score += len(ids) * 0.5

        centers = {}
        for box, tid in zip(boxes, ids):
            centers[tid] = get_center(box)

        for tid, center in centers.items():
            if tid in prev_positions:
                speed     = get_speed(prev_positions[tid], center)
                direction = get_direction(prev_positions[tid], center)

                # Fast movement = high score
                if speed > SPEED_THRESHOLD:
                    score += 3

                # Direction change = high score
                if tid in prev_directions:
                    if angle_diff(prev_directions[tid], direction) > DIRECTION_THRESHOLD and speed > 5:
                        score += 4

                prev_directions[tid] = direction
            prev_positions[tid] = center

        # Player clustering = high score
        id_list = list(centers.keys())
        for i in range(len(id_list)):
            for j in range(i+1, len(id_list)):
                tid1, tid2 = id_list[i], id_list[j]
                dist = get_speed(centers[tid1], centers[tid2])
                if dist < CLUSTER_DISTANCE:
                    score += 2

    frame_scores[frame_count] = score
    frame_count += 1

cap.release()

# --- Find highlight moments ---
print(f"\n🔍 Analyzing excitement scores...")

# Smooth scores using rolling average
window = int(fps)  # 1 second window
smoothed = {}
frames = sorted(frame_scores.keys())

for i, f in enumerate(frames):
    start = max(0, i - window // 2)
    end   = min(len(frames), i + window // 2)
    smoothed[f] = np.mean([frame_scores[frames[j]] for j in range(start, end)])

# Find top highlight moments (peaks in excitement)
highlight_frames = []
min_gap = int(fps * 5)  # minimum 5 seconds between highlights
threshold = np.percentile(list(smoothed.values()), 75)  # top 25% exciting moments

last_highlight = -min_gap
for f in frames:
    if smoothed[f] >= threshold and f - last_highlight >= min_gap:
        highlight_frames.append(f)
        last_highlight = f

print(f"🎯 Found {len(highlight_frames)} highlight moments!")

# --- Extract highlight clips ---
os.makedirs = __import__('os').makedirs
__import__('os').makedirs("highlights", exist_ok=True)

highlight_log = []
clip_duration = 3  # seconds per highlight clip

for i, hf in enumerate(highlight_frames):
    start_frame = max(0, hf - int(fps * 1))  # 1 sec before
    end_frame   = min(total_frames, hf + int(fps * clip_duration))
    start_time  = start_frame / fps
    end_time    = end_frame / fps

    output_file = f"highlights/highlight_{i+1}.mp4"

    # Use ffmpeg to extract clip
    import subprocess
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-ss", str(start_time),
        "-to", str(end_time),
        "-c:v", "libx264",
        "-c:a", "aac",
        output_file
    ]
    subprocess.run(cmd, capture_output=True)

    highlight_log.append({
        "clip": i + 1,
        "frame": hf,
        "timestamp": hf / fps,
        "start": start_time,
        "end": end_time,
        "score": smoothed[hf],
        "file": output_file
    })

    print(f"   🎬 Highlight {i+1}: {start_time:.1f}s - {end_time:.1f}s | Score: {smoothed[hf]:.1f} | Saved: {output_file}")

print(f"\n🎉 Done!")
print(f"🎬 Total highlights: {len(highlight_log)}")
print(f"📁 Check your highlights/ folder!")

# Save highlight log for Streamlit UI later
import json
with open("highlight_log.json", "w") as f:
    json.dump(highlight_log, f, indent=2)
print(f"💾 Highlight log saved: highlight_log.json")