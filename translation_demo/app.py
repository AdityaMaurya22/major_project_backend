import streamlit as st
import tempfile
import whisper
from deep_translator import GoogleTranslator
from gtts import gTTS
import pyttsx3
import os
import ffmpeg
import torch

# Auto-detect GPU
USE_GPU = torch.cuda.is_available()
model = whisper.load_model("large-v2" if USE_GPU else "small")

# Translation function
def translate(text, lang="hi"):
    try:
        return GoogleTranslator(source="auto", target=lang).translate(text)
    except Exception as e:
        return f"Translation failed: {e}"

# Clean audio with aggressive filters
def clean_audio(path):
    cleaned = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    (
        ffmpeg
        .input(path)
        .output(cleaned, af="highpass=f=200, lowpass=f=3000, dynaudnorm", ar='16000', ac=1)
        .overwrite_output()
        .run(quiet=True)
    )
    return cleaned

# Extract audio from video
def extract_audio(path):
    audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    (
        ffmpeg
        .input(path)
        .output(audio_path, ar="16000", ac=1, format="wav")
        .overwrite_output()
        .run(quiet=True)
    )
    return audio_path

# Transcribe with Whisper
def transcribe(path):
    return model.transcribe(path, fp16=USE_GPU)["text"]

# TTS with gTTS or fallback to pyttsx3
def tts(text, lang="hi"):
    speech_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    try:
        gTTS(text=text, lang=lang).save(speech_path)
    except:
        engine = pyttsx3.init()
        engine.save_to_file(text, speech_path)
        engine.runAndWait()
    return speech_path

# Match TTS to video duration (smart sync)
def match_audio_to_video(audio_path, video_duration):
    try:
        audio_duration = float(ffmpeg.probe(audio_path)["format"]["duration"])
    except:
        return audio_path  # fallback

    if abs(audio_duration - video_duration) < 0.5:
        return audio_path

    tempo = audio_duration / video_duration
    tempo = max(0.5, min(2.0, tempo))  # FFmpeg atempo supports 0.5–2.0

    adjusted_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    (
        ffmpeg
        .input(audio_path)
        .filter("atempo", tempo)
        .output(adjusted_path)
        .overwrite_output()
        .run(quiet=True)
    )
    return adjusted_path

# Get duration
def get_duration(path):
    try:
        return float(ffmpeg.probe(path)["format"]["duration"])
    except:
        return 0

# Merge audio & video
def merge_audio_video(video_path, audio_path):
    output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    video = ffmpeg.input(video_path).video
    audio = ffmpeg.input(audio_path).audio
    (
        ffmpeg
        .output(video, audio, output, c='copy')
        .global_args("-shortest")
        .overwrite_output()
        .run(quiet=True)
    )
    return output

# Language options
lang_options = {
    "English": "en", "Hindi": "hi", "Bengali": "bn", "Tamil": "ta", "Telugu": "te",
    "Marathi": "mr", "Gujarati": "gu", "Kannada": "kn", "Malayalam": "ml", "Punjabi": "pa",
    "Urdu": "ur", "Spanish": "es", "French": "fr", "German": "de", "Japanese": "ja", "Arabic": "ar"
}

# Streamlit UI
st.set_page_config(page_title="🎙️ Fast Translator", layout="centered")
st.title("🎬 Translate & Dub Media")

file = st.file_uploader("📤 Upload video/audio", type=["mp4", "mov", "mkv", "mp3", "wav", "m4a"])
lang_name = st.selectbox("🌍 Choose Target Language", list(lang_options.keys()))
lang_code = lang_options[lang_name]

if file:
    st.success(f"✅ Uploaded: {file.name}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[-1]) as temp:
        temp.write(file.read())
        input_path = temp.name

    is_video = file.name.lower().endswith((".mp4", ".mov", ".mkv"))
    st.info("🔄 Processing, please wait...")

    try:
        # Step 1: Extract audio if video
        raw_audio = extract_audio(input_path) if is_video else input_path

        # Step 2: Clean audio
        cleaned_audio = clean_audio(raw_audio)

        # Step 3: Transcribe
        transcript = transcribe(cleaned_audio)

        # Step 4: Heuristic check for poor transcript
        if len(transcript.strip()) < 10 or "[INAUDIBLE]" in transcript or transcript.count(" ") < 5:
            st.warning("⚠️ Audio may be unclear. Re-cleaning and retrying transcription...")
            cleaned_audio = clean_audio(cleaned_audio)
            transcript = transcribe(cleaned_audio)

        # Step 5: Translate
        translation = translate(transcript, lang=lang_code)

        # Step 6: TTS
        tts_path = tts(translation, lang=lang_code)

        # Step 7: Display results
        st.text_area("📝 Transcript", transcript, height=150)
        st.text_area("🌐 Translation", translation, height=150)
        st.audio(tts_path)

        with open(tts_path, "rb") as f:
            st.download_button("🔊 Download Translated Audio", data=f, file_name="translated_audio.mp3", mime="audio/mp3")

        # Step 8: Merge with video
        if is_video:
            duration = get_duration(input_path)
            synced_audio = match_audio_to_video(tts_path, duration)
            final_video = merge_audio_video(input_path, synced_audio)
            st.video(final_video)
            with open(final_video, "rb") as f:
                st.download_button("🎞️ Download Dubbed Video", data=f, file_name="translated_video.mp4", mime="video/mp4")

    except Exception as e:
        st.error(f"💥 Error: {e}")