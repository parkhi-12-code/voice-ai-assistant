# 🎙️ Voice-Based Conversational AI — Speech LLM Pipeline

> An end-to-end **speech-to-speech conversational AI system** that integrates Automatic Speech Recognition (ASR), Large Language Model (LLM) reasoning, and Text-to-Speech (TTS) synthesis into a unified pipeline — aligned with the **Speech LLM paradigm**.

<br>

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![Whisper](https://img.shields.io/badge/ASR-OpenAI%20Whisper-orange)](https://github.com/openai/whisper)
[![LLaMA](https://img.shields.io/badge/LLM-LLaMA%203%20via%20Groq-purple)](https://console.groq.com)
[![gTTS](https://img.shields.io/badge/TTS-gTTS-green)](https://gtts.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## 🔁 Pipeline

```
User Voice / Audio File
        │
        ▼
┌──────────────────┐
│  Whisper (ASR)   │  ← Converts speech to text
└──────────────────┘
        │
        ▼
┌──────────────────┐
│  LLaMA 3 (LLM)  │  ← Generates intelligent response
│   via Groq API   │
└──────────────────┘
        │
        ▼
┌──────────────────┐
│   gTTS  (TTS)    │  ← Converts response to voice
└──────────────────┘
        │
        ▼
   Spoken Audio Output
```

---

## 🎯 What This Project Does

You speak (or provide an audio file) → the system **hears** you → **thinks** of a smart reply → **speaks back** to you.

| Component | Technology | Purpose |
|-----------|-----------|---------|
| ASR | OpenAI Whisper (base) | Converts spoken audio → text |
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
- Download from: https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip
- Extract and add the `bin` folder to your Windows System PATH
- Verify: `ffmpeg -version`

**Mac:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt install ffmpeg
```

### 4. Get Your FREE Groq API Key

- Go to: https://console.groq.com
- Sign up (free) → API Keys → Create Key
- Copy your key

### 5. Add Your API Key

Open `get_reply.py` and `voice_assistant.py` and replace:
```python
api_key = "YOUR_GROQ_KEY_HERE"
```
with your actual Groq API key.

### 6. Run the Pipeline

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
├── .gitignore              ← Excludes keys, cache, audio files
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
from groq import Groq
client = Groq(api_key="YOUR_KEY")
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
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---------|-----|
| `FileNotFoundError: ffmpeg` | Install ffmpeg and add to PATH (see Step 3 above) |
| `SyntaxWarning: invalid escape sequence` | Add `r` before Windows file paths: `r"E:\speech\file.wav"` |
| `model decommissioned` error | Use `llama-3.3-70b-versatile` instead of `llama3-8b-8192` |
| `playsound` install fails | Use `pip install playsound==1.2.2` (not 1.3.0) |
| No audio plays on Linux | Install: `sudo apt install mpg321` |
| `FP16 not supported` warning | Harmless — Whisper auto-switches to FP32 on CPU |

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
- [ ] Replace gTTS with neural TTS (XTTS / Bark)
- [ ] Build Streamlit web UI with microphone widget
- [ ] Add multi-turn conversation memory
- [ ] Fine-tune Whisper on Indian accents
- [ ] Add wake word detection (Porcupine)
- [ ] Docker containerisation for cross-platform deployment

---

## 🧠 Key Concepts Demonstrated

- **Automatic Speech Recognition (ASR)** — Whisper transformer encoder-decoder
- **Large Language Model Integration** — LLaMA 3 via Groq low-latency API
- **Text-to-Speech Synthesis** — gTTS cloud-based neural synthesis
- **Speech LLM Paradigm** — Unified speech + language pipeline
- **Real-World Engineering** — ffmpeg setup, API integration, Windows audio I/O

---

## 📊 Performance (CPU Hardware)

| Stage | Latency |
|-------|---------|
| Whisper ASR | ~3.5 sec |
| LLaMA 3 LLM | ~2.0 sec |
| gTTS TTS | ~1.5 sec |
| **Total Pipeline** | **~5–8 sec** |

---

## 👤 Author

**PARKHI YADAV**
Undergraduate Student — IIT Madras


---

## 📄 License

This project is open-source under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- [OpenAI Whisper](https://github.com/openai/whisper) — ASR model
- [Meta LLaMA 3](https://ai.meta.com/llama/) — Open-source LLM
- [Groq](https://console.groq.com) — Ultra-fast LLM inference API
- [gTTS](https://gtts.readthedocs.io) — Text-to-Speech library
- [LTRC, IIIT Hyderabad](https://ltrc.iiit.ac.in) — IASNLP 2026
