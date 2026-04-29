# Voice-Based Conversational AI — Speech LLM Pipeline

An end-to-end speech-to-speech conversational AI system integrating Automatic Speech Recognition (ASR), Large Language Model (LLM) reasoning, and Text-to-Speech (TTS) synthesis — built as an experimental pipeline to study **information loss at modality boundaries** in Speech LLM systems.

![Python](https://img.shields.io/badge/Python-3.12-blue) ![Whisper](https://img.shields.io/badge/ASR-Whisper-green) ![LLaMA](https://img.shields.io/badge/LLM-LLaMA3-orange) ![gTTS](https://img.shields.io/badge/TTS-gTTS-red) ![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 🔍 Research Motivation

This pipeline was built not just as a working demo, but to investigate a concrete question:

> **Where does information get lost at modality boundaries in a modular Speech LLM pipeline?**

When a speaker says *"he was livid"* in a furious tone, Whisper transcribes the words — but the prosody (pitch, tempo, intensity) is discarded. The LLM receives `"he was livid"`, not the emotional signal. Does the LLM's response change? Does gTTS carry any coloring back in its synthesized output?

This project builds the infrastructure to test these questions and runs preliminary experiments on them (see [Observations](#-observations--findings)).

---

## 🔁 Pipeline

```
User Voice / Audio File
        │
        ▼
┌──────────────────┐
│  Whisper (ASR)   │  ← Converts speech to text
└──────────────────┘
        │  [Boundary 1: Prosody lost here]
        ▼
┌──────────────────┐
│  LLaMA 3 (LLM)  │  ← Generates intelligent response
│   via Groq API   │
└──────────────────┘
        │  [Boundary 2: Semantic intent may shift in synthesis]
        ▼
┌──────────────────┐
│   gTTS  (TTS)    │  ← Converts response to voice
└──────────────────┘
        │
        ▼
   Spoken Audio Output
```

---

## 🧪 Observations & Findings

### Boundary 1: ASR → LLM (Prosody Loss)

Whisper operates on acoustic features but outputs only text. To test how much prosodic information matters, the same sentence was recorded in three emotional tones — neutral, angry, and anxious — and fed through the pipeline.

| Input Tone | Transcript | LLM Response Tone |
|------------|------------|-------------------|
| Neutral    | "I need help with this" | Informational |
| Angry      | "I need help with this" | Informational (identical) |
| Anxious    | "I need help with this" | Informational (identical) |

**Finding:** The LLM response was nearly identical across all three. The entire affective signal — carried acoustically — was erased at the ASR boundary. This is the central limitation of modular pipelines vs. end-to-end Speech LLMs, where audio tokens flow directly into the model without text as an intermediary.

### Boundary 2: LLM → TTS (Expressive Flattening)

gTTS synthesizes audio at a fixed neutral pitch and tempo regardless of the semantic content of the text. A response containing urgency ("Call emergency services immediately") is synthesized in the same flat tone as "The capital of France is Paris."

**Finding:** gTTS has no prosody control. Neural TTS systems (e.g., Bark, XTTS) support emotional tone injection — this is a meaningful upgrade path.

---

## ⚡ Latency Analysis

| Stage | Model | Latency (CPU) | Bottleneck? |
|-------|-------|---------------|-------------|
| ASR | Whisper `base` | ~3.5 sec | ✅ Yes |
| LLM | LLaMA 3.3-70B via Groq | ~2.0 sec | ❌ No |
| TTS | gTTS | ~1.5 sec | ❌ No |
| **Total** | — | **~5–8 sec** | — |

**Why Whisper is the bottleneck:**
Whisper's encoder-decoder transformer runs on CPU for this setup. The `base` model (74M parameters) runs noticeably faster than `small` (244M) or `medium` (769M), with a real trade-off:

| Model | Latency (CPU) | WER (English) | Recommended For |
|-------|--------------|---------------|-----------------|
| `tiny` | ~1.5 sec | Higher | Real-time prototyping |
| `base` | ~3.5 sec | Moderate | This project |
| `small` | ~7 sec | Lower | Accuracy-critical tasks |
| `medium` | ~15 sec | Lowest | Offline batch processing |

For real-time use, two practical paths exist: (1) use `tiny` and accept lower accuracy, or (2) run `base` on GPU (reduces latency to ~0.4 sec). Groq's API already handles LLM inference at very low latency (~400 tokens/sec), so LLM is not a meaningful optimization target.

---

## 🎯 What This Project Does

You speak (or provide an audio file) → the system transcribes it → generates a contextual reply → speaks the reply back.

| Component | Technology | Purpose |
|-----------|------------|---------|
| ASR | OpenAI Whisper (`base`) | Converts spoken audio → text |
| LLM | LLaMA 3.3-70B via Groq | Generates intelligent text response |
| TTS | Google gTTS | Converts response text → spoken audio |
| Audio Backend | ffmpeg | Decodes WAV/MP3/MP4 audio formats |
| Language | Python 3.12 | Core development language |

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/parkhi-12-code/voice-ai-assistant.git
cd voice-ai-assistant
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install ffmpeg (Required for Whisper)

**Windows:**
```
Download: https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip
Extract → add bin/ folder to Windows System PATH
Verify: ffmpeg -version
```

**Mac:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt install ffmpeg
```

### 4. Configure Your API Key (Secure)

This project uses `python-dotenv` for secure API key management. **Never hardcode keys in source files.**

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_actual_key_here
```

Get your free Groq API key at: https://console.groq.com → API Keys → Create Key

`.env` is listed in `.gitignore` and will never be committed to version control.

### 5. Run the Pipeline

```bash
python voice_assistant.py
```

---

## 📁 Project Structure

```
voice-ai-assistant/
│
├── voice_assistant.py      ← Main pipeline (run this)
├── transcribe.py           ← ASR module (Whisper)
├── get_reply.py            ← LLM module (LLaMA 3 via Groq)
├── speak.py                ← TTS module (gTTS)
├── record.py               ← Audio recording module
├── requirements.txt        ← All dependencies
├── .env                    ← API keys (not committed — see .gitignore)
├── .env.example            ← Template for environment setup
├── .gitignore              ← Excludes .env, cache, audio files
└── README.md               ← This file
```

---

## ⚙️ How Each File Works

### `voice_assistant.py` — Main Pipeline

Orchestrates all stages in sequence:

```python
text   = transcribe(audio_file)   # Stage 1: Voice → Text
reply  = get_reply(text)          # Stage 2: Text → LLM Response
speak(reply)                      # Stage 3: Response → Voice
```

### `transcribe.py` — ASR Module

```python
import whisper
model = whisper.load_model("base")
result = model.transcribe(r"path\to\audio.wav")
print(result["text"])
```

### `get_reply.py` — LLM Module

```python
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
completion = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": user_text}]
)
```

### `speak.py` — TTS Module

```python
from gtts import gTTS
tts = gTTS(text=reply, lang='en')
tts.save("response.mp3")
```

---

## 📦 Requirements

```
openai-whisper
groq
gtts
sounddevice
scipy
playsound==1.2.2
python-dotenv
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---------|-----|
| `FileNotFoundError: ffmpeg` | Install ffmpeg and add to PATH (see Step 3) |
| `SyntaxWarning: invalid escape sequence` | Add `r` before Windows paths: `r"E:\speech\file.wav"` |
| `model decommissioned error` | Use `llama-3.3-70b-versatile` (not `llama3-8b-8192`) |
| `playsound install fails` | Use `pip install playsound==1.2.2` (not 1.3.0) |
| No audio plays on Linux | `sudo apt install mpg321` |
| FP16 warning on CPU | Harmless — Whisper auto-switches to FP32 |
| `GROQ_API_KEY not found` | Ensure `.env` file exists in project root with correct key |

---

## 🌍 Real-World Use Cases

- **Accessibility** — Voice interface for visually impaired users
- **Education** — Spoken Q&A tutoring assistant
- **Healthcare** — Hands-free medical query assistant
- **Smart Devices** — Open-source voice control (alternative to Alexa)
- **Multilingual Service** — Foundation for regional language voice bots
- **Language Learning** — Speak, get corrected, hear correct pronunciation

---

## 🔮 Future Improvements

- [ ] Add streaming ASR + LLM for real-time responses
- [ ] Multilingual support (Hindi, Telugu, Tamil)
- [ ] Replace gTTS with neural TTS (XTTS / Bark) for prosody control
- [ ] Pass Whisper word-level timestamps + confidence scores to LLM as context (partial prosody proxy)
- [ ] Build Streamlit web UI with microphone widget
- [ ] Add multi-turn conversation memory
- [ ] Fine-tune Whisper on Indian accents
- [ ] Add wake word detection (Porcupine)
- [ ] Docker containerisation for cross-platform deployment
- [ ] Benchmark `tiny` vs `base` vs `small` Whisper on Indian English to find optimal latency/accuracy tradeoff

---

## 🧠 Key Concepts Demonstrated

- **Automatic Speech Recognition (ASR)** — Whisper transformer encoder-decoder
- **Large Language Model Integration** — LLaMA 3 via Groq low-latency API
- **Text-to-Speech Synthesis** — gTTS cloud-based neural synthesis
- **Speech LLM Paradigm** — Modular pipeline and its boundary limitations vs. end-to-end architectures
- **Secure API Key Management** — Environment variable injection with `python-dotenv`
- **Real-World Engineering** — ffmpeg setup, API integration, Windows audio I/O

---

## 👤 Author

**PARKHI YADAV** — Undergraduate Student, IIT Madras

---

## 📄 License

This project is open-source under the MIT License.

---

## 🙏 Acknowledgements

- [OpenAI Whisper](https://github.com/openai/whisper) — ASR model
- [Meta LLaMA 3](https://llama.meta.com/) — Open-source LLM
- [Groq](https://groq.com/) — Ultra-fast LLM inference API
- [gTTS](https://gtts.readthedocs.io/) — Text-to-Speech library
