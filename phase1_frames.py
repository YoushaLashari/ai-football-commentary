import cv2
import os

# Create folder to save frames
os.makedirs("frames", exist_ok=True)

# Load video - put any video file in your project folder
video_path = "test_video.mp4"  # your video file name here
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Error: Could not open video.")
    exit()

frame_count = 0

while True:
    ret, frame = cap.read()
    
    if not ret:
        break  # No more frames
    
    # Save every 30th frame (1 per second if 30fps)
    if frame_count % 30 == 0:
        filename = f"frames/frame_{frame_count}.jpg"
        cv2.imwrite(filename, frame)
        print(f"✅ Saved {filename}")
    
    frame_count += 1

cap.release()
print(f"\n🎉 Done! Total frames processed: {frame_count}")