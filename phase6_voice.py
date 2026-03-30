import cv2
import numpy as np
from ultralytics import YOLO
from groq import Groq
from gtts import gTTS
from dotenv import load_dotenv
import os
import re
import time
import subprocess

# Load API key from .env
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

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
    "phase6_no_audio.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

# Event tracking
prev_positions   = {}
prev_directions  = {}
event_log        = []
last_event_frame = {}
commentary_log   = []

# Thresholds
SPEED_THRESHOLD     = 15
DIRECTION_THRESHOLD = 90
CLUSTER_DISTANCE    = 80

frame_count           = 0
last_commentary_frame = -90
audio_files           = []

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

def generate_commentary(events):
    events_text = "\n".join(events)
    prompt = f"""You are an exciting football (soccer) TV commentator.
Based on these detected events, give ONE short punchy commentary line (max 15 words).
Rules:
- Do NOT mention any player numbers or IDs
- Use natural football language like "the striker", "the midfielder", "the defender"
- Be dramatic and exciting like a real TV commentator
- No quotes, no punctuation except exclamation mark

Events detected:
{events_text}

Reply with ONLY the commentary line, nothing else."""
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=40,
            temperature=0.9
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"   ⚠️ Groq error: {e}")
        return "Exciting action on the pitch!"

def text_to_speech(text, filename):
    try:
        clean_text = text.replace('"', '').replace("'", "")
        clean_text = re.sub(r'\d+', '', clean_text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        tts = gTTS(text=clean_text, lang='en', slow=False)
        tts.save(filename)
        print(f"   🔊 Audio saved: {filename}")
        return True
    except Exception as e:
        print(f"   ⚠️ TTS error: {e}")
        return False

os.makedirs("audio", exist_ok=True)
recent_events = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, verbose=False,
                          tracker="bytetrack.yaml", conf=0.3, iou=0.5)

    annotated_frame   = results[0].plot()
    events_this_frame = []

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids   = results[0].boxes.id.cpu().numpy().astype(int)

        centers = {}
        for box, tid in zip(boxes, ids):
            centers[tid] = get_center(box)

        for tid, center in centers.items():
            if tid in prev_positions:
                speed     = get_speed(prev_positions[tid], center)
                direction = get_direction(prev_positions[tid], center)

                if speed > SPEED_THRESHOLD:
                    events_this_frame.append(f"Player {tid} is sprinting fast")
                if tid in prev_directions:
                    if angle_diff(prev_directions[tid], direction) > DIRECTION_THRESHOLD and speed > 5:
                        events_this_frame.append(f"Player {tid} changed direction suddenly")

                prev_directions[tid] = direction
            prev_positions[tid] = center

        id_list = list(centers.keys())
        for i in range(len(id_list)):
            for j in range(i+1, len(id_list)):
                tid1, tid2 = id_list[i], id_list[j]
                dist = get_speed(centers[tid1], centers[tid2])
                if dist < CLUSTER_DISTANCE:
                    events_this_frame.append(f"Players {tid1} and {tid2} are close together")

    for e in events_this_frame:
        if last_event_frame.get(e, -30) + 30 <= frame_count:
            recent_events.append(e)
            event_log.append({"frame": frame_count, "event": e})
            last_event_frame[e] = frame_count

    if (recent_events and
            frame_count - last_commentary_frame >= 90):

        print(f"\n🎙️ Generating commentary at frame {frame_count}...")
        commentary_text = generate_commentary(recent_events[-5:])
        timestamp = frame_count / fps

        commentary_log.append({
            "frame": frame_count,
            "timestamp": timestamp,
            "commentary": commentary_text
        })
        print(f"   💬 {commentary_text}")

        audio_file = f"audio/commentary_{frame_count}.mp3"
        if text_to_speech(commentary_text, audio_file):
            audio_files.append({
                "file": audio_file,
                "timestamp": timestamp
            })

        recent_events = []
        last_commentary_frame = frame_count
        time.sleep(0.5)

    # Draw commentary on frame
    if commentary_log:
        last_comment  = commentary_log[-1]["commentary"]
        clean_comment = re.sub(r'\d+', '', last_comment).strip()
        cv2.rectangle(annotated_frame, (0, height-60), (width, height), (0, 0, 0), -1)
        cv2.putText(annotated_frame, clean_comment,
                    (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 255, 100), 2)

    out.write(annotated_frame)
    frame_count += 1

cap.release()
out.release()

print(f"\n🎉 Video processing done!")
print(f"🎙️ Total commentary lines: {len(commentary_log)}")
print(f"🔊 Total audio files: {len(audio_files)}")

# --- Merge audio into video using ffmpeg ---
print(f"\n🔧 Merging audio into video...")

if audio_files:
    video_duration = frame_count / fps
    inputs         = ["-i", "phase6_no_audio.mp4"]
    filter_parts   = []

    for i, af in enumerate(audio_files):
        inputs += ["-i", af["file"]]
        delay_ms = int(af["timestamp"] * 1000)
        filter_parts.append(
            f"[{i+1}:a]adelay={delay_ms}|{delay_ms}[a{i}]"
        )

    mix_labels = "".join([f"[a{i}]" for i in range(len(audio_files))])
    filter_parts.append(
        f"{mix_labels}amix=inputs={len(audio_files)}:normalize=0:dropout_transition=0[aout]"
    )

    filter_str = ";".join(filter_parts)

    cmd = (["ffmpeg", "-y"] + inputs +
           ["-filter_complex", filter_str,
            "-map", "0:v",
            "-map", "[aout]",
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            "phase6_final.mp4"])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ Final video saved as: phase6_final.mp4")
    else:
        print(f"⚠️ ffmpeg error: {result.stderr[-300:]}")
else:
    print("⚠️ No audio files generated")

print("\n📁 Check phase6_final.mp4 for video with voice commentary!")