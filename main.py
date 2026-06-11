# server/main.py
# Free speech recognition server using faster-whisper
# Deploy this to Render.com for free

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import tempfile
import os

app = FastAPI()

# Allow requests from any Chrome extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Whisper model once at startup (free, runs on server)
from faster_whisper import WhisperModel
print("Loading Whisper model...")
model = WhisperModel("small", device="cpu", compute_type="int8")
print("Model ready!")

class AudioRequest(BaseModel):
    audio: str      # base64 encoded audio
    mimeType: str   # e.g. "audio/webm;codecs=opus"

@app.get("/")
def root():
    return {"status": "CDrama Subtitle Server running!"}

@app.post("/transcribe")
async def transcribe(req: AudioRequest):
    try:
        # Decode base64 audio
        audio_bytes = base64.b64decode(req.audio)

        # Save to temp file
        suffix = ".webm"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(audio_bytes)
            tmp_path = f.name

        # Transcribe with faster-whisper (Mandarin)
        segments, info = model.transcribe(
            tmp_path,
            language="zh",
            beam_size=5,
            vad_filter=True,  # skip silence automatically
            vad_parameters={"min_silence_duration_ms": 500}
        )

        # Collect all segments
        mandarin = " ".join([seg.text.strip() for seg in segments]).strip()

        os.unlink(tmp_path)  # clean up temp file

        if not mandarin:
            return {"mandarin": None}

        return {"mandarin": mandarin}

    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e), "mandarin": None}
