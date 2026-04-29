import whisper

model = whisper.load_model("base")


result = model.transcribe("E:\speech\demo_speech.wav")

text = result["text"]
print("Whisper understood:", text)