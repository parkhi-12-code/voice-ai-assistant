from gtts import gTTS
import os
import time

def speak(text, filename="response.mp3"):
    print("🔊 Converting reply to voice...")
    tts = gTTS(text=text, lang='en', slow=False)
    tts.save(filename)
    print(" Audio saved as", filename)
    os.system(f"start {filename}")
    time.sleep(3)

# Test it
speak("Hello! I am your AI voice assistant. I can answer any question you ask me.")