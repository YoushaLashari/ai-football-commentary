import cv2
from ultralytics import YOLO

# Load YOLOv8 model (downloads automatically first time, ~6MB)
model = YOLO("yolov8n.pt")  # 'n' = nano, fastest version

# Load video
video_path = "test_video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Error: Could not open video.")
    exit()

frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Run YOLO detection on every frame
    results = model(frame, verbose=False)

    # Draw bounding boxes on frame
    annotated_frame = results[0].plot()

    # Save every 30th annotated frame
    if frame_count % 30 == 0:
        filename = f"frames/detected_frame_{frame_count}.jpg"
        cv2.imwrite(filename, annotated_frame)
        print(f"✅ Frame {frame_count} - Objects detected: {len(results[0].boxes)}")

    frame_count += 1

cap.release()
print(f"\n🎉 Done! Total frames processed: {frame_count}")
print("📁 Check your frames/ folder for detected images!")