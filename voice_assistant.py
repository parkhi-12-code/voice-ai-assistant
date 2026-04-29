import whisper
from groq import Groq
from gtts import gTTS
import os
import time


# my GROQ KEY 
API_KEY = "YOUR_GROQ_KEY_HERE"


AUDIO_FILE = r"E:\speech\demo_speech.wav"

# ---- STEP 1: Voice → Text ----
def transcribe(filename):
    print("\n🔄 Step 1: Understanding the audio...")
    model = whisper.load_model("base")
    result = model.transcribe(filename)
    text = result["text"].strip()
    print(f"📝 Heard: '{text}'")
    return text

# ---- STEP 2: Text → Smart Reply ----
def get_reply(user_text):
    print("\n🤔 Step 2: Thinking of a reply...")
    client = Groq(api_key=API_KEY)
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful AI assistant. Give clear, concise answers in 3-4 sentences."
            },
            {
                "role": "user",
                "content": user_text
            }
        ],
        model="llama-3.3-70b-versatile",
    )
    reply = chat_completion.choices[0].message.content
    print(f"🤖 Reply: '{reply}'")
    return reply

# ---- STEP 3: Reply → Voice ----
def speak(text, filename="response.mp3"):
    print("\n🔊 Step 3: Speaking the reply...")
    tts = gTTS(text=text, lang='en', slow=False)
    tts.save(filename)
    os.system(f"start {filename}")
    time.sleep(5)
    print(" Done!")

# ---- MAIN PIPELINE ----
def main():
    print("="*50)
    print("🚀 VOICE AI ASSISTANT — Speech LLM Pipeline")
    print("="*50)
    print("Pipeline: Audio → Whisper → LLM → gTTS → Voice")
    print("="*50)

    # Running the pipeline
    text = transcribe(AUDIO_FILE)

    if not text:
        print("❌ No speech detected in audio file.")
        return

    reply = get_reply(text)
    speak(reply)

    print("\n" + "="*50)
    print(" Pipeline Complete!")
    print("="*50)

if __name__ == "__main__":
    main()