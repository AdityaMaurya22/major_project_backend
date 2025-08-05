Required Libraries
Library 
streamlit - Web app interface — used to build interactive UI
ffmpeg-python - Python wrapper for FFmpeg — handles audio/video conversion & merging
whisper - OpenAI’s speech-to-text model — used for transcription of audio 
googletrans==4.0.0-rc1 - Translation of transcribed text into target language  
gTTS - Google Text-to-Speech — generates voice audio from translated text 
tempfile - Python standard lib — creates secure temp files for processing  
os - Python standard lib — used to manage file paths  

Installation Command
pip install streamlit ffmpeg-python openai-whisper googletrans==4.0.0-rc1 gTTS

Setup Instructions
Step 1: Create a virtual environment
python -m venv dubbing-env

Step 2: Activate the virtual environment
dubbing-env\Scripts\activate

