import whisper

def transcribe(filename=r"E:\speech\demo_speech.wav"):
    print("🔄 Converting your voice to text...")
    model = whisper.load_model("base")
    result = model.transcribe(filename)
    text = result["text"]
    print(" You said:", text)
    return text

transcribe()