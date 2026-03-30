import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Load video
video_path = "test_video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Error: Could not open video.")
    exit()

# Get video properties for output video
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = int(cap.get(cv2.CAP_PROP_FPS))

# Save output as a new video (not just frames)
out = cv2.VideoWriter(
    "phase3_tracked.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

frame_count = 0
track_history = {}  # store movement path of each ID

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Run YOLO tracking (botsort is built-in, no install needed)
    results = model.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml", conf=0.3, iou=0.5)

    annotated_frame = results[0].plot()

    # Log tracking info every 30 frames
    if frame_count % 30 == 0:
        if results[0].boxes.id is not None:
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            print(f"✅ Frame {frame_count} - Tracking IDs: {ids.tolist()}")
        else:
            print(f"⚠️ Frame {frame_count} - No tracked objects")

    # Write annotated frame to output video
    out.write(annotated_frame)
    frame_count += 1

cap.release()
out.release()
print(f"\n🎉 Done! Total frames: {frame_count}")
print("📁 Output saved as: phase3_tracked.mp4")