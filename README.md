# ⚽ AI Football Video Analysis & Commentary System

> **Automatically analyze football videos with AI-powered player tracking, 
> real-time commentary generation, voice output, and highlight detection.**


---

## 🎬 Demo

> Upload any football video → Get AI commentary, player tracking & highlights instantly!

---

## ✨ Features

| Feature | Description |
|--------|-------------|
| 🤖 **Object Detection** | YOLOv8 detects players and objects in every frame |
| 🎯 **Player Tracking** | ByteTrack assigns persistent IDs across frames |
| ⚡ **Event Detection** | Detects sprints, direction changes, and clustering |
| 🎙️ **AI Commentary** | Groq + Llama 3.1 generates real football commentary |
| 🔊 **Voice Output** | gTTS converts commentary to speech and merges into video |
| 🎬 **Highlight Detection** | Automatically extracts the most exciting moments |
| 📊 **Performance Metrics** | FPS, events/min, detection scores and more |
| 🖥️ **Streamlit UI** | Clean, dark-themed web interface with download buttons |

---

## 🛠️ Tech Stack

- **Computer Vision:** OpenCV, YOLOv8 (Ultralytics)
- **Object Tracking:** ByteTrack (built into YOLOv8)
- **AI Commentary:** Groq API + Llama 3.1 8B Instant
- **Text to Speech:** gTTS (Google Text-to-Speech)
- **Video Processing:** FFmpeg
- **UI:** Streamlit
- **Language:** Python 3.10+

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/ai-football-commentary.git
cd ai-football-commentary
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up API keys
```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your Groq API key
GROQ_API_KEY=your_groq_api_key_here
```

> Get your free Groq API key at: https://console.groq.com

### 5. Run the app
```bash
streamlit run app.py
```

---

## 📁 Project Structure
```
ai-football-commentary/
│
├── app.py                  # Main Streamlit UI
├── phase1_frames.py        # Video frame extraction
├── phase2_detection.py     # YOLOv8 object detection
├── phase3_tracking.py      # Player tracking
├── phase4_events.py        # Event detection
├── phase5_commentary.py    # AI commentary generation
├── phase6_voice.py         # Voice output
├── phase7_highlights.py    # Highlight detection
│
├── .env.example            # API key template
├── .gitignore              # Git ignore rules
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## ⚙️ How It Works
```
Video Input
    ↓
YOLOv8 Detection (detect players per frame)
    ↓
ByteTrack Tracking (assign persistent IDs)
    ↓
Event Detection (sprint / direction change / clustering)
    ↓
Groq + Llama 3.1 (generate commentary from events)
    ↓
gTTS (convert commentary to voice)
    ↓
FFmpeg (merge voice into video)
    ↓
Highlight Detection (extract exciting moments)
    ↓
Streamlit UI (display results + download)
```

---

## 📦 Requirements

Create a `requirements.txt` file:
```
opencv-python
numpy
ultralytics
groq
gtts
streamlit
python-dotenv
pydub
```

---

## 🔑 Environment Variables

Create a `.env` file based on `.env.example`:
```env
GROQ_API_KEY=your_groq_api_key_here
```

---

## 📝 License

MIT License — free to use and modify.

---

## 👨‍💻 Author

Built as a portfolio project demonstrating:
- Computer Vision with YOLOv8
- AI/LLM integration with Groq
- Real-time video processing
- Full-stack AI application development

---

⭐ **If you found this useful, please star the repo!**
```

---

Now also create `requirements.txt`:
```
opencv-python
numpy
ultralytics
groq
gtts
streamlit
python-dotenv
pydub
google-genai