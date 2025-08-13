import whisper

model = whisper.load_model("large-v2")

def transcribe_audio(file_path):
    result = model.transcribe(file_path)
    return result["text"]