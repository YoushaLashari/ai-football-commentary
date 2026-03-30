import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import re
import time
import subprocess
from ultralytics import YOLO
from groq import Groq
from gtts import gTTS
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Page Config ---
st.set_page_config(
    page_title="AI Football Commentary",
    page_icon="⚽",
    layout="wide"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
        color: #ffffff;
    }
    .hero {
        background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
        border: 1px solid #00ff88;
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 0 40px rgba(0, 255, 136, 0.15);
    }
    .hero h1 {
        font-size: 3em;
        color: #00ff88;
        margin-bottom: 10px;
        text-shadow: 0 0 20px rgba(0, 255, 136, 0.5);
    }
    .hero p {
        font-size: 1.2em;
        color: #aaaaaa;
        margin-bottom: 20px;
    }
    .badges {
        display: flex;
        justify-content: center;
        gap: 15px;
        flex-wrap: wrap;
        margin-top: 15px;
    }
    .badge {
        background: rgba(0, 255, 136, 0.1);
        border: 1px solid #00ff88;
        border-radius: 20px;
        padding: 6px 16px;
        font-size: 0.85em;
        color: #00ff88;
    }
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #00ff88;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 0 15px rgba(0, 255, 136, 0.1);
    }
    div[data-testid="metric-container"] label {
        color: #aaaaaa !important;
    }
    div[data-testid="metric-container"] div {
        color: #00ff88 !important;
        font-size: 1.8em !important;
        font-weight: bold !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #00ff88, #00cc6a) !important;
        color: #000000 !important;
        font-weight: bold !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 15px !important;
        font-size: 1.1em !important;
    }
    .stDownloadButton > button {
        background: linear-gradient(135deg, #0f3460, #16213e) !important;
        color: #00ff88 !important;
        border: 1px solid #00ff88 !important;
        border-radius: 8px !important;
        font-weight: bold !important;
    }
    .commentary-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-left: 4px solid #00ff88;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        color: #ffffff;
    }
    .commentary-time {
        color: #00ff88;
        font-weight: bold;
        font-size: 0.85em;
    }
    .highlight-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #ff6b35;
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        margin-bottom: 10px;
    }
    .section-header {
        color: #00ff88;
        font-size: 1.5em;
        font-weight: bold;
        border-bottom: 2px solid #00ff88;
        padding-bottom: 8px;
        margin: 25px 0 15px 0;
    }
    .stProgress > div > div {
        background: linear-gradient(90deg, #00ff88, #00cc6a) !important;
    }
    hr {
        border-color: #00ff88 !important;
        opacity: 0.3 !important;
    }
    [data-testid="stFileUploader"] {
        background: rgba(0, 255, 136, 0.05) !important;
        border: 2px dashed #00ff88 !important;
        border-radius: 12px !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Hero Section ---
st.markdown("""
<div class="hero">
    <h1>⚽ AI Football Commentary</h1>
    <p>Upload any football video and get real-time AI-powered player tracking,<br>
    event detection, commentary generation, and highlight extraction!</p>
    <div class="badges">
        <span class="badge">🤖 YOLOv8 Detection</span>
        <span class="badge">🎯 Player Tracking</span>
        <span class="badge">🎙️ Llama Commentary</span>
        <span class="badge">🔊 Voice Output</span>
        <span class="badge">🎬 Highlight Detection</span>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.divider()
    commentary_interval = st.slider("🎙️ Commentary Interval (frames)", 30, 120, 60)
    speed_threshold     = st.slider("⚡ Sprint Speed Threshold", 5, 30, 15)
    cluster_distance    = st.slider("👥 Cluster Distance (px)", 40, 150, 80)
    st.divider()
    st.markdown("### 🛠️ Powered By")
    st.markdown("""
    - 🤖 **YOLOv8** — Object Detection
    - 🎯 **ByteTrack** — Player Tracking
    - 🧠 **Llama 3.1** — Commentary AI
    - 🔊 **gTTS** — Voice Synthesis
    - 🎬 **FFmpeg** — Video Processing
    """)
    st.divider()
    st.markdown("Built for **Upwork Portfolio** 🚀")

# --- Check API Key ---
if not GROQ_API_KEY:
    st.error("❌ Groq API key not found! Please add GROQ_API_KEY to your .env file.")
    st.stop()

# --- Upload Section ---
st.markdown('<div class="section-header">📁 Upload Video</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Drop your football video here",
    type=["mp4", "avi", "mov"],
    help="Recommended: 30-120 second football match footage"
)

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<div class="section-header">📹 Original Video</div>', unsafe_allow_html=True)
        st.video(video_path)
    with col2:
        st.markdown('<div class="section-header">ℹ️ Video Info</div>', unsafe_allow_html=True)
        cap_info = cv2.VideoCapture(video_path)
        total_f  = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_info = int(cap_info.get(cv2.CAP_PROP_FPS))
        w_info   = int(cap_info.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_info   = int(cap_info.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_info.release()
        dur_info = total_f / fps_info if fps_info > 0 else 0
        st.metric("📹 Frames", total_f)
        st.metric("🎞️ FPS", fps_info)
        st.metric("📐 Resolution", f"{w_info}x{h_info}")
        st.metric("⏱️ Duration", f"{dur_info:.1f}s")

    st.markdown("")
    analyze = st.button("🚀 Analyze Video", type="primary", use_container_width=True)

    if analyze:
        client = Groq(api_key=GROQ_API_KEY)
        model  = YOLO("yolov8n.pt")

        cap = cv2.VideoCapture(video_path)
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps          = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration     = total_frames / fps

        # Fixed output paths
        output_path  = "processed_no_audio.mp4"
        final_output = "final_output.mp4"

        out = cv2.VideoWriter(output_path,
                              cv2.VideoWriter_fourcc(*"mp4v"),
                              fps, (width, height))

        prev_positions        = {}
        prev_directions       = {}
        last_event_frame      = {}
        commentary_log        = []
        event_log             = []
        frame_scores          = {}
        recent_events         = []
        audio_files           = []
        last_commentary_frame = -commentary_interval

        os.makedirs("audio", exist_ok=True)

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
            except:
                return "Exciting action on the pitch!"

        def text_to_speech(text, filename):
            try:
                clean_text = text.replace('"', '').replace("'", "")
                clean_text = re.sub(r'\d+', '', clean_text)
                clean_text = re.sub(r'\s+', ' ', clean_text).strip()
                tts = gTTS(text=clean_text, lang='en', slow=False)
                tts.save(filename)
                return True
            except:
                return False

        # --- Processing UI ---
        st.markdown('<div class="section-header">⚙️ Processing</div>',
                    unsafe_allow_html=True)

        progress_bar    = st.progress(0)
        col_s1, col_s2, col_s3 = st.columns(3)
        status_frame    = col_s1.empty()
        status_events   = col_s2.empty()
        status_comment  = col_s3.empty()
        live_commentary = st.empty()

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.track(frame, persist=True, verbose=False,
                                  tracker="bytetrack.yaml", conf=0.3, iou=0.5)

            annotated_frame   = results[0].plot()
            events_this_frame = []
            score = 0

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids   = results[0].boxes.id.cpu().numpy().astype(int)
                score += len(ids) * 0.5

                centers = {}
                for box, tid in zip(boxes, ids):
                    centers[tid] = get_center(box)

                for tid, center in centers.items():
                    if tid in prev_positions:
                        speed     = get_speed(prev_positions[tid], center)
                        direction = get_direction(prev_positions[tid], center)

                        if speed > speed_threshold:
                            events_this_frame.append(f"Player {tid} is sprinting fast")
                            score += 3
                        if tid in prev_directions:
                            if angle_diff(prev_directions[tid], direction) > 90 and speed > 5:
                                events_this_frame.append(f"Player {tid} changed direction suddenly")
                                score += 4

                        prev_directions[tid] = direction
                    prev_positions[tid] = center

                id_list = list(centers.keys())
                for i in range(len(id_list)):
                    for j in range(i+1, len(id_list)):
                        tid1, tid2 = id_list[i], id_list[j]
                        dist = get_speed(centers[tid1], centers[tid2])
                        if dist < cluster_distance:
                            events_this_frame.append(
                                f"Players {tid1} and {tid2} are close together")
                            score += 2

            frame_scores[frame_count] = score

            for e in events_this_frame:
                if last_event_frame.get(e, -30) + 30 <= frame_count:
                    recent_events.append(e)
                    event_log.append({"frame": frame_count, "event": e})
                    last_event_frame[e] = frame_count

            if (recent_events and
                    frame_count - last_commentary_frame >= commentary_interval):

                commentary_text = generate_commentary(recent_events[-5:])
                timestamp = frame_count / fps

                commentary_log.append({
                    "frame": frame_count,
                    "timestamp": round(timestamp, 2),
                    "commentary": commentary_text
                })

                audio_file = f"audio/commentary_{frame_count}.mp3"
                if text_to_speech(commentary_text, audio_file):
                    audio_files.append({
                        "file": audio_file,
                        "timestamp": timestamp
                    })

                clean_c = re.sub(r'\d+', '', commentary_text).strip()
                live_commentary.markdown(f"""
                <div class="commentary-card">
                    🎙️ <b>{clean_c}</b>
                </div>
                """, unsafe_allow_html=True)

                recent_events         = []
                last_commentary_frame = frame_count
                time.sleep(0.5)

            if commentary_log:
                last_comment  = commentary_log[-1]["commentary"]
                clean_comment = re.sub(r'\d+', '', last_comment).strip()
                cv2.rectangle(annotated_frame, (0, height-60),
                              (width, height), (0, 0, 0), -1)
                cv2.putText(annotated_frame, clean_comment,
                            (10, height-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (0, 255, 100), 2)

            out.write(annotated_frame)
            frame_count += 1

            progress = int((frame_count / total_frames) * 100)
            progress_bar.progress(progress)
            status_frame.metric("🎞️ Frame",   f"{frame_count}/{total_frames}")
            status_events.metric("🎯 Events",  len(event_log))
            status_comment.metric("🎙️ Lines",  len(commentary_log))

        cap.release()
        out.release()

        # --- Merge Audio ---
        live_commentary.info("🔧 Merging audio into video...")

        if audio_files:
            inputs       = ["-i", output_path]
            filter_parts = []

            for i, af in enumerate(audio_files):
                inputs += ["-i", af["file"]]
                delay_ms = int(af["timestamp"] * 1000)
                filter_parts.append(
                    f"[{i+1}:a]adelay={delay_ms}|{delay_ms}[a{i}]"
                )

            mix_labels = "".join([f"[a{i}]" for i in range(len(audio_files))])
            filter_parts.append(
                f"{mix_labels}amix=inputs={len(audio_files)}"
                f":normalize=0:dropout_transition=0[aout]"
            )

            cmd = (["ffmpeg", "-y"] + inputs +
                   ["-filter_complex", ";".join(filter_parts),
                    "-map", "0:v",
                    "-map", "[aout]",
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-shortest",
                    final_output])
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                st.warning(f"⚠️ Audio merge issue: {result.stderr[-200:]}")
                final_output = output_path
        else:
            final_output = output_path

        # --- Highlight Detection ---
        live_commentary.info("🎬 Detecting highlights...")

        frames   = sorted(frame_scores.keys())
        window   = int(fps)
        smoothed = {}

        for i, f in enumerate(frames):
            start = max(0, i - window // 2)
            end   = min(len(frames), i + window // 2)
            smoothed[f] = np.mean(
                [frame_scores[frames[j]] for j in range(start, end)])

        highlight_frames = []
        min_gap   = int(fps * 5)
        threshold = np.percentile(list(smoothed.values()), 75)
        last_hl   = -min_gap

        for f in frames:
            if smoothed[f] >= threshold and f - last_hl >= min_gap:
                highlight_frames.append(f)
                last_hl = f

        highlight_log = []
        os.makedirs("highlights", exist_ok=True)

        for i, hf in enumerate(highlight_frames):
            start_time  = max(0, hf - fps) / fps
            end_time    = min(total_frames, hf + fps * 3) / fps
            output_file = f"highlights/highlight_{i+1}.mp4"

            subprocess.run([
                "ffmpeg", "-y", "-i", video_path,
                "-ss", str(start_time),
                "-to", str(end_time),
                "-c:v", "libx264", "-c:a", "aac",
                output_file
            ], capture_output=True)

            highlight_log.append({
                "clip":      i + 1,
                "timestamp": round(hf / fps, 2),
                "start":     round(start_time, 2),
                "end":       round(end_time, 2),
                "score":     round(smoothed[hf], 2),
                "file":      output_file
            })

        progress_bar.progress(100)
        live_commentary.success("✅ Analysis Complete!")

        # --- Results ---
        st.markdown('<div class="section-header">📊 Results</div>',
                    unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🎯 Events Detected",  len(event_log))
        col2.metric("🎙️ Commentary Lines", len(commentary_log))
        col3.metric("🎬 Highlights Found", len(highlight_frames))
        col4.metric("⚡ Events/Min",
                    round(len(event_log) / max(duration / 60, 1), 1))

        # --- Processed Video ---
        st.markdown('<div class="section-header">🎥 Processed Video with Commentary</div>',
                    unsafe_allow_html=True)

        if os.path.exists(final_output):
            with open(final_output, "rb") as f:
                video_bytes = f.read()
            st.video(video_bytes)
            st.download_button(
                label="⬇️ Download Processed Video",
                data=video_bytes,
                file_name="ai_commentary_video.mp4",
                mime="video/mp4",
                use_container_width=True
            )
        else:
            st.warning("⚠️ Processed video not found.")

        # --- Commentary Log ---
        st.markdown('<div class="section-header">🎙️ Commentary Log</div>',
                    unsafe_allow_html=True)
        for c in commentary_log:
            clean = re.sub(r'\d+', '', c['commentary']).strip()
            st.markdown(f"""
            <div class="commentary-card">
                <span class="commentary-time">⏱️ {c['timestamp']}s</span>
                <span style="margin-left:10px">{clean}</span>
            </div>
            """, unsafe_allow_html=True)

        # --- Highlights ---
        if highlight_log:
            st.markdown('<div class="section-header">🎬 Highlight Clips</div>',
                        unsafe_allow_html=True)
            cols = st.columns(min(3, len(highlight_log)))
            for i, hl in enumerate(highlight_log):
                with cols[i % 3]:
                    st.markdown(f"""
                    <div class="highlight-card">
                        🎬 <b>Highlight {hl['clip']}</b><br>
                        ⏱️ {hl['timestamp']}s &nbsp;|&nbsp; 🔥 Score: {hl['score']}
                    </div>
                    """, unsafe_allow_html=True)
                    if os.path.exists(hl['file']):
                        with open(hl['file'], "rb") as f:
                            hl_bytes = f.read()
                        st.video(hl_bytes)
                        st.download_button(
                            label=f"⬇️ Download Highlight {hl['clip']}",
                            data=hl_bytes,
                            file_name=f"highlight_{hl['clip']}.mp4",
                            mime="video/mp4",
                            use_container_width=True,
                            key=f"hl_{i}"
                        )

        # --- Performance Metrics ---
        st.markdown('<div class="section-header">📈 Performance Metrics</div>',
                    unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("📹 Total Frames", total_frames)
        m2.metric("🎯 Avg Score",    round(np.mean(list(frame_scores.values())), 2))
        m3.metric("⏱️ Duration",     f"{duration:.1f}s")
        m4.metric("🎞️ FPS",          fps)

else:
    st.markdown("""
    <div style="text-align:center; padding:60px; color:#555555;">
        <div style="font-size:5em;">⚽</div>
        <h3 style="color:#555555;">Upload a football video to get started</h3>
        <p>Supported formats: MP4, AVI, MOV • Max size: 200MB</p>
    </div>
    """, unsafe_allow_html=True)