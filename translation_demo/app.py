import streamlit as st
import tempfile
import whisper
from googletrans import Translator
from gtts import gTTS
import os
import ffmpeg

# Load Whisper base model (faster)
model = whisper.load_model("base")
translator = Translator()

# Optional: Clean noisy audio
def clean_audio(path):
    cleaned = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    (
        ffmpeg
        .input(path)
        .output(cleaned, af="highpass=f=200, lowpass=f=3000", ar='16000', ac=1)
        .overwrite_output()
        .run()
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
        .run()
    )
    return audio_path

# Transcribe with Whisper
def transcribe(path):
    return model.transcribe(path)["text"]

# Translate with Google Translate
def translate(text, lang="hi"):
    return translator.translate(text, dest=lang).text

# TTS with gTTS
def tts(text, lang="hi"):
    speech_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    gTTS(text=text, lang=lang).save(speech_path)
    return speech_path

# Match TTS to video duration
def match_audio_to_video(audio, video_duration):
    fitted = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    (
        ffmpeg
        .input(audio)
        .filter("atempo", 1.25)  # Adjust tempo slightly
        .output(fitted, t=video_duration)
        .overwrite_output()
        .run()
    )
    return fitted

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
        .run()
    )
    return output

# Language options (Indian + global)
lang_options = {
    "English": "en",
    "Hindi": "hi",
    "Bengali": "bn",
    "Tamil": "ta",
    "Telugu": "te",
    "Marathi": "mr",
    "Gujarati": "gu",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Punjabi": "pa",
    "Urdu": "ur",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Japanese": "ja",
    "Arabic": "ar"
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
        # Step 1: Extract or clean audio
        raw_audio = extract_audio(input_path) if is_video else input_path
        processed_audio = clean_audio(raw_audio)

        # Step 2: Transcription
        transcript = transcribe(processed_audio)

        # Step 3: Translation
        translation = translate(transcript, lang=lang_code)

        # Step 4: TTS
        tts_path = tts(translation, lang=lang_code)

        # Step 5: Sync audio to video if needed
        st.text_area("📝 Transcript", transcript, height=150)
        st.text_area("🌐 Translation", translation, height=150)
        st.audio(tts_path)

        with open(tts_path, "rb") as f:
            st.download_button("🔊 Download Translated Audio", data=f, file_name="translated_audio.mp3", mime="audio/mp3")

        if is_video:
            duration = get_duration(input_path)
            fitted_audio = match_audio_to_video(tts_path, duration)
            final_video = merge_audio_video(input_path, fitted_audio)
            st.video(final_video)
            with open(final_video, "rb") as f:
                st.download_button("🎞️ Download Dubbed Video", data=f, file_name="translated_video.mp4", mime="video/mp4")

    except Exception as e:
        st.error(f"💥 Error: {e}")