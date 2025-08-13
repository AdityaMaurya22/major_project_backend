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
MODEL_NAME = "large-v3" if USE_GPU else "small"
model = whisper.load_model(MODEL_NAME)

# Translation function (Google)
def translate_google(text, lang="hi"):
    try:
        sentences = text.split(". ")
        translated_sentences = [
            GoogleTranslator(source="en", target=lang).translate(s)
            for s in sentences if s.strip()
        ]
        return ". ".join(translated_sentences)
    except Exception as e:
        return f"Translation failed: {e}"

# Clean audio
def clean_audio(path):
    cleaned = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    (
        ffmpeg
        .input(path)
        .output(cleaned, af="highpass=f=100, lowpass=f=8000, dynaudnorm", ar='16000', ac=1)
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

# Whisper: transcribe in same language
def whisper_transcribe(path):
    return model.transcribe(path, task="transcribe", fp16=USE_GPU)["text"]

# Whisper: translate to English
def whisper_translate(path):
    return model.transcribe(path, task="translate", fp16=USE_GPU)["text"]

# TTS
def tts(text, lang="hi"):
    speech_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    try:
        gTTS(text=text, lang=lang, slow=False).save(speech_path)
    except:
        engine = pyttsx3.init()
        engine.save_to_file(text, speech_path)
        engine.runAndWait()
    return speech_path

# Match audio duration to video
def match_audio_to_video(audio_path, video_duration):
    try:
        audio_duration = float(ffmpeg.probe(audio_path)["format"]["duration"])
    except:
        return audio_path

    if abs(audio_duration - video_duration) < 0.5:
        return audio_path

    tempo = audio_duration / video_duration
    tempo = max(0.5, min(2.0, tempo))

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

# Merge audio with video
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
    "Urdu": "ur", "Spanish": "es", "French": "fr", "German": "de", "Japanese": "ja", "Arabic": "ar","Italian": "it","Nepali": "ne",
    "Portuguese": "pt","Russian": "ru","Tamil": "ta","Telugu": "te"

}

# Streamlit UI
st.set_page_config(page_title="🎙️ Fast Translator", layout="centered")
st.title("🎬 Translate & Dub Media (Dual Whisper Mode)")

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

        # Step 3: Whisper transcription (same language)
        original_transcript = whisper_transcribe(cleaned_audio)

        # Step 4: Whisper English translation
        whisper_english = whisper_translate(cleaned_audio)

        # Step 5: Google translation from English to target
        final_translation = translate_google(whisper_english, lang=lang_code)

        # Step 6: TTS
        tts_path = tts(final_translation, lang=lang_code)

        # Step 7: Display results
        st.text_area("📝 Original Transcript", original_transcript, height=150)
        st.text_area("🇬🇧 Whisper English Translation", whisper_english, height=150)
        st.text_area("🌐 Final Target Translation", final_translation, height=150)
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
