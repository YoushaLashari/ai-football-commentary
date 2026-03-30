import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

video_path = "test_video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Error: Could not open video.")
    exit()

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(
    "phase4_events.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

# Store previous positions of each tracked ID
prev_positions = {}
prev_directions = {}

# Store all detected events
# Store all detected events
event_log = []
last_event_frame = {}  # prevent duplicate events

# Thresholds
SPEED_THRESHOLD     = 15   # pixels/frame = fast movement
DIRECTION_THRESHOLD = 90   # degrees = sudden direction change
CLUSTER_DISTANCE    = 80   # pixels = players too close = clustering

frame_count = 0

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

    annotated_frame = results[0].plot()
    events_this_frame = []

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids   = results[0].boxes.id.cpu().numpy().astype(int)

        centers = {}
        for box, tid in zip(boxes, ids):
            centers[tid] = get_center(box)

        # --- Event 1: Fast movement & Direction change ---
        for tid, center in centers.items():
            if tid in prev_positions:
                speed = get_speed(prev_positions[tid], center)
                direction = get_direction(prev_positions[tid], center)

                if speed > SPEED_THRESHOLD:
                    events_this_frame.append(f"⚡ Player {tid} sprinting!")

                if tid in prev_directions:
                    if angle_diff(prev_directions[tid], direction) > DIRECTION_THRESHOLD and speed > 5:
                        events_this_frame.append(f"🔄 Player {tid} changed direction!")

                prev_directions[tid] = direction
            prev_positions[tid] = center

        # --- Event 2: Player clustering ---
        id_list = list(centers.keys())
        for i in range(len(id_list)):
            for j in range(i+1, len(id_list)):
                tid1, tid2 = id_list[i], id_list[j]
                dist = get_speed(centers[tid1], centers[tid2])
                if dist < CLUSTER_DISTANCE:
                    events_this_frame.append(f"👥 Players {tid1} & {tid2} clustering!")

    # --- Log & display events ---
        if events_this_frame:
                print(f"\n🎯 Frame {frame_count}:")
                for e in events_this_frame:
                    # Only log if this event hasn't fired in last 30 frames
                    if last_event_frame.get(e, -30) + 30 <= frame_count:
                        print(f"   {e}")
                        event_log.append({"frame": frame_count, "event": e})
                        last_event_frame[e] = frame_count

        # Draw events on video frame
        y = 30
        for e in events_this_frame[:3]:  # show max 3 on screen
            cv2.putText(annotated_frame, e, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y += 25

    out.write(annotated_frame)
    frame_count += 1

cap.release()
out.release()

print(f"\n🎉 Done! Total frames: {frame_count}")
print(f"📋 Total events detected: {len(event_log)}")
print(f"📁 Output saved as: phase4_events.mp4")