# AI Promo Plug Generator

A local, GPU-accelerated video trailer/promo plug generator with AI-powered scene selection, voiceover, music, and sound effects. Built with OpenCV, PySceneDetect, FFmpeg, and local AI models.

## Features

- **AI Scene Detection** — PySceneDetect finds scene boundaries; OpenCV scores quality (sharpness, brightness, face detection)
- **AI Vision Scoring** — Optional Ollama vision models rank scenes by content quality
- **Genre Presets** — 15 genre-specific presets (action, horror, comedy, etc.) with automatic transitions, music, and SFX
- **AI Music Generation** — ACE-Step generates genre-matched background music
- **AI SFX** — Sony Woosh generates sound effects; stamped at every scene cut
- **TTS Voiceover** — Fish Audio S2 generates Tagalog/multilingual voiceovers (83 languages)
- **Speech-to-Text** — Faster-Whisper large-v3 for transcription/subtitles
- **J/L Cuts** — Professional audio bleed across cut points
- **Beat Matching** — Librosa-based tempo sync between video and music
- **Web UI** — Flask-based DaVinci-style interface with live preview

## Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA support (tested on RTX 4000 Ada 20GB)
- **RAM**: 16GB+ recommended
- **Storage**: ~50GB for all models
- **OS**: Windows (tested on Windows Server 2022)

## Installation

### 1. Main Application

```bash
cd C:\opencv-pyscenedetect
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Fish Audio S2 (TTS)

```bash
cd C:
git clone https://github.com/fishaudio/fish-speech.git
cd fish-speech
uv venv --python 3.12
uv pip install --python .venv torch torchaudio --index-url https://download.pytorch.org/whl/cu128
uv sync --python 3.12 --extra cu128 --no-default-groups
```

Download the S2-Pro model:
```bash
# Models auto-download on first use, or manually:
uv run huggingface-cli download fishaudio/fish-s2-pro --local-dir checkpoints/s2-pro
```

**VRAM Optimization** (for 20GB GPUs): Edit `checkpoints/s2-pro/config.json` and set `"max_seq_len": 8192` (default 32768).

### 3. Woosh (SFX)

```bash
cd C:
git clone https://github.com/SonyResearch/Woosh.git
cd Woosh
uv sync --extra cuda
```

Weights download automatically on first run.

### 4. Faster-Whisper (STT)

```bash
cd C:
mkdir faster-whisper && cd faster-whisper
uv venv --python 3.12
uv pip install --python .venv faster-whisper
uv pip install --python .venv torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

The `large-v3` model downloads automatically on first use (~3GB).

### 5. FFmpeg

Download from https://ffmpeg.org/download.html and add to `C:\ffmpeg\bin\` or your PATH.

### 6. Ollama (Optional, for AI Vision)

Install from https://ollama.com and pull a vision model:
```bash
ollama pull qwen3-vl:8b
```

### 7. ACE-Step (Optional, for AI Music)

Run the ACE-Step server on `http://localhost:8001` for AI music/SFX generation.

## Usage

```bash
cd C:\opencv-pyscenedetect
venv\Scripts\activate
python app.py
```

Open http://localhost:5000 in your browser.

### Generate a Promo Plug

1. Upload a video (or paste a URL)
2. Select **Generate Promo Plug** tab
3. Choose genre (auto-configures transitions, music, SFX)
4. Select trailer length (15s / 30s / 45s / 60s)
5. Enable voiceover (Fish Audio S2) and enter script
6. Click **Generate**

## Running AI Services

Start each service in a separate terminal:

```bash
# Fish Audio S2 TTS server
cd C:\fish-speech
.\.venv\Scripts\python.exe -m fish_speech.api_server --port 8080

# Woosh SFX server
cd C:\Woosh
uv run python woosh/api_server.py --port 8030

# ACE-Step music server (if using)
# Follow ACE-Step docs to start on port 8001
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/trailer/generate` | POST | Generate promo plug |
| `/api/opencv/info` | POST | Get video info |
| `/api/opencv/analyze` | POST | Analyze video frames |
| `/api/scenedetect/detect` | POST | Detect scenes |
| `/api/vision/analyze` | POST | AI frame analysis |
| `/api/vision/models` | GET | List vision models |
| `/api/playback/start` | POST | Start video player |
| `/api/playback/stream/<mode>` | GET | Stream filtered video |
| `/api/playback/pause` | GET | Pause playback |
| `/api/playback/resume` | GET | Resume playback |
| `/api/playback/stop` | GET | Stop playback |

## Project Structure

```
C:\opencv-pyscenedetect\
├── app.py              # Main application (simpler version)
├── app_17_(3).py       # Full version with TTS, job queue, progress tracking
├── requirements.txt    # Python dependencies
└── venv/               # Virtual environment

C:\fish-speech\         # Fish Audio S2 (TTS)
C:\Woosh\               # Sony Woosh (SFX)
C:\faster-whisper\      # Faster-Whisper (STT)
```

## Genre Presets

| Genre | Transition | SFX | Music Style |
|-------|-----------|-----|-------------|
| Action | Zoom In | Yes | Epic orchestral |
| Drama | Fade | No | Emotional piano |
| Horror | Wipe Left | Yes | Dark ambient |
| Comedy | Squeeze V | Yes | Upbeat playful |
| Documentary | Fade | No | Cinematic inspiring |
| Thriller | Radial | Yes | Suspenseful pulsing |
| Sci-Fi | Pixelize | Yes | Futuristic electronic |
| Fantasy | Dissolve | Yes | Magical orchestral |
| Romance | Fade | No | Romantic soft |
| Adventure | Smooth Right | Yes | Epic heroic |
| Mystery | Fade Black | No | Intriguing atmospheric |
| Western | Diag BR | Yes | Acoustic guitar |
| Sports | Slide Up | Yes | Energetic driving |
| Noir | Circle Close | No | Dark jazz |
| War | Distance | Yes | Dramatic somber |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `FISH_AUDIO_URL` | `http://localhost:8080/v1/tts` | Fish Audio S2 endpoint |
| `FISH_AUDIO_API_KEY` | (empty) | API key for hosted Fish Audio |
| `FISH_AUDIO_MODEL` | `s2.1-pro-free` | Fish Audio model |
| `MAX_CONCURRENT_JOBS` | `2` | Max parallel trailer jobs |

## License

Internal use only.
