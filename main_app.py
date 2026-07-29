import os, cv2, numpy as np, tempfile, threading, time, pathlib, base64, json, requests, subprocess, shutil
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from flask import Flask, render_template_string, request, send_from_directory, jsonify, Response
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 * 1024  # 5GB
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'}
ACE_STEP_URL = 'http://localhost:8001'
WOOSH_URL = 'http://localhost:8030'  # local API server for Sony AI's Woosh SFX foundation model (github.com/SonyResearch/Woosh); tried first for genre SFX, falls back to ACE-Step then a procedural synth
# Local speech-to-text service (e.g. fedirz/faster-whisper-server or any server exposing
# an OpenAI-compatible POST /v1/audio/transcriptions endpoint). Dialogue transcription
# calls out to this service instead of loading faster-whisper in-process, same as the
# other engines (Fish Audio, Ollama, ACE-Step, Woosh) — no local Python whisper
# dependency required. Override WHISPER_MODEL if your server needs a model name passed.
WHISPER_URL = os.environ.get('WHISPER_URL', 'http://localhost:8000')
WHISPER_MODEL = os.environ.get('WHISPER_MODEL', 'large-v2')
# Self-hosted Fish Audio S2 (SGLang-based server you run yourself) — no API key needed.
# To use Fish Audio's hosted cloud API instead, set FISH_AUDIO_URL=https://api.fish.audio/v1/tts
# and FISH_AUDIO_API_KEY to your key.
FISH_AUDIO_URL = os.environ.get('FISH_AUDIO_URL', 'http://10.0.1.213:8080/v1/tts')
FISH_AUDIO_API_KEY = os.environ.get('FISH_AUDIO_API_KEY', '')  # leave blank for a self-hosted server
FISH_AUDIO_MODEL = os.environ.get('FISH_AUDIO_MODEL', 's2.1-pro-free')  # ignored by most self-hosted servers, which just run whichever checkpoint they were started with
# Reference WAV used for Fish Audio voice cloning (zero-shot): the narration voice is
# cloned from this sample on every request rather than requiring a pre-registered
# reference_id. Looked up next to this script by default; override with the env var.
# Cozy Voice 3 — alternate narration TTS engine, offered as a second choice
# alongside Fish Audio. Point this at your local Cozy Voice 3 server.
COZY_VOICE_URL = os.environ.get('COZY_VOICE_URL', 'http://localhost:8040')

# ---- Fish Audio S2 (fish.audio) — primary voiceover engine (self-hosted or cloud REST API) ----
def fish_audio_tts(text, output_wav_path, voice_id=None, rate=175, reference_audio_path=None, language=None):
    """Generate a voiceover WAV using a Fish Audio S2-compatible TTS server (POST /v1/tts) —
    either your own self-hosted instance or Fish Audio's hosted cloud API. Returns
    (ok, error_message). `voice_id`, if set, is passed as `reference_id` to reuse a
    pre-registered voice. Otherwise, if `reference_audio_path` points at an existing WAV,
    that sample is base64-encoded and sent as a `references` entry so the server clones
    the voice from it directly (zero-shot cloning) — no pre-registration needed. `rate` is
    the same wpm-style value used by the UI (default 175); mapped onto Fish Audio's
    prosody.speed multiplier (1.0 = normal) so faster/slower selections keep the same
    behavior as before. Language does not need to be specified — S2 auto-detects it
    from the text (83 languages, including Tagalog)."""
    speed = max(0.5, min(2.0, (rate or 175) / 175.0))
    body = {
        'text': text,
        'format': 'wav',
        'prosody': {'speed': speed, 'volume': 0, 'normalize_loudness': True},
    }
    if voice_id:
        body['reference_id'] = voice_id
    elif reference_audio_path and os.path.exists(reference_audio_path):
        try:
            with open(reference_audio_path, 'rb') as rf:
                ref_b64 = base64.b64encode(rf.read()).decode('ascii')
            body['references'] = [{'audio': ref_b64, 'text': ''}]
        except Exception as e:
            return False, f'Failed to read voice reference WAV ({reference_audio_path}): {e}'
    if language and language != 'auto':
        # Best-effort hint only: S2 normally auto-detects language from the text
        # itself, but passing it explicitly helps disambiguate short/ambiguous
        # scripts. A server that doesn't recognize this field just ignores it.
        body['language'] = language
    headers = {'Content-Type': 'application/json'}
    if FISH_AUDIO_API_KEY:
        # Only the hosted api.fish.audio endpoint needs these; a self-hosted server
        # with no key configured just ignores their absence.
        headers['Authorization'] = f'Bearer {FISH_AUDIO_API_KEY}'
        headers['model'] = FISH_AUDIO_MODEL
    try:
        r = requests.post(FISH_AUDIO_URL, headers=headers, json=body, timeout=30)
        if r.ok and r.content:
            with open(output_wav_path, 'wb') as f:
                f.write(r.content)
            ok = os.path.exists(output_wav_path) and os.path.getsize(output_wav_path) > 0
            return ok, (None if ok else 'Fish Audio returned an empty response')
        return False, f'Fish Audio API error {r.status_code}: {r.text[:200]}'
    except Exception as e:
        return False, f'Fish Audio request failed: {e}'

# ---- Cozy Voice 3 — alternate narration engine ----
def cozy_voice_tts(text, output_wav_path, voice_id=None, rate=175, reference_audio_path=None, language=None):
    """Generate a voiceover WAV using a local Cozy Voice 3 server (POST /tts).
    Mirrors fish_audio_tts's signature/behavior so the two engines are
    interchangeable from generate_tts()'s point of view: `voice_id` selects a
    preset/registered voice, `reference_audio_path` (if set and no voice_id)
    is sent as a base64 reference sample for zero-shot cloning, and `rate` is
    mapped onto a speed multiplier the same way. Adjust the payload/endpoint
    below if your Cozy Voice 3 server's actual API differs. Returns
    (ok, error_message)."""
    speed = max(0.5, min(2.0, (rate or 175) / 175.0))
    body = {'text': text, 'format': 'wav', 'speed': speed}
    if voice_id:
        body['voice_id'] = voice_id
    elif reference_audio_path and os.path.exists(reference_audio_path):
        try:
            with open(reference_audio_path, 'rb') as rf:
                body['reference_audio'] = base64.b64encode(rf.read()).decode('ascii')
        except Exception as e:
            return False, f'Failed to read voice reference WAV ({reference_audio_path}): {e}'
    if language and language != 'auto':
        body['language'] = language
    try:
        r = requests.post(f'{COZY_VOICE_URL}/tts', json=body, timeout=30)
        if r.ok and r.content:
            with open(output_wav_path, 'wb') as f:
                f.write(r.content)
            ok = os.path.exists(output_wav_path) and os.path.getsize(output_wav_path) > 0
            return ok, (None if ok else 'Cozy Voice 3 returned an empty response')
        return False, f'Cozy Voice 3 API error {r.status_code}: {r.text[:200]}'
    except Exception as e:
        return False, f'Cozy Voice 3 request failed: {e}'

# Curated subset of languages Fish Audio S2 auto-detects/supports, offered as an
# optional override in the UI dropdown. 'auto' (S2's normal auto-detect behavior)
# is always the default — this list isn't exhaustive of all ~83 supported
# languages, just the ones most likely to be picked explicitly.
FISH_AUDIO_LANGUAGES = [
    {'code': 'auto', 'label': 'Auto-detect (recommended)'},
    {'code': 'en', 'label': 'English'},
    {'code': 'tl', 'label': 'Tagalog / Filipino'},
    {'code': 'zh', 'label': 'Chinese (Mandarin)'},
    {'code': 'yue', 'label': 'Chinese (Cantonese)'},
    {'code': 'ja', 'label': 'Japanese'},
    {'code': 'ko', 'label': 'Korean'},
    {'code': 'es', 'label': 'Spanish'},
    {'code': 'fr', 'label': 'French'},
    {'code': 'de', 'label': 'German'},
    {'code': 'it', 'label': 'Italian'},
    {'code': 'pt', 'label': 'Portuguese'},
    {'code': 'ru', 'label': 'Russian'},
    {'code': 'ar', 'label': 'Arabic'},
    {'code': 'hi', 'label': 'Hindi'},
    {'code': 'id', 'label': 'Indonesian'},
    {'code': 'vi', 'label': 'Vietnamese'},
    {'code': 'th', 'label': 'Thai'},
    {'code': 'nl', 'label': 'Dutch'},
    {'code': 'pl', 'label': 'Polish'},
    {'code': 'tr', 'label': 'Turkish'},
]

_VOICES_CACHE = {
    'fish_audio': {'voices': None, 'source': None, 'error': None, 'fetched_at': 0},
    'cozy_voice': {'voices': None, 'source': None, 'error': None, 'fetched_at': 0},
}
_VOICES_CACHE_TTL = 60  # seconds; avoids hammering the API every time the dropdown opens

def fish_audio_list_voices(force=False):
    """List voices registered for narration via Fish Audio: registered voice
    models from Fish Audio's cloud API (if FISH_AUDIO_API_KEY is set), or a
    best-effort probe of a self-hosted server's own model-listing endpoint if
    it has one. There's no fallback "default" entry — if nothing is
    registered/listable, this returns an empty voices list, and the caller is
    expected to fall back to "upload a reference sample" for zero-shot
    cloning instead. Returns (voices, source, error) where source is
    'cloud' | 'self_hosted' | 'none' | 'error', and each voice is
    {'id': <reference_id>, 'title': <display name>, 'languages': [...]}."""
    now = time.time()
    cache = _VOICES_CACHE['fish_audio']
    if not force and cache['voices'] is not None and now - cache['fetched_at'] < _VOICES_CACHE_TTL:
        return cache['voices'], cache['source'], cache['error']

    voices = []
    source = 'none'
    error = None

    if FISH_AUDIO_API_KEY:
        # Hosted cloud API: list voice models registered under this account.
        try:
            r = requests.get('https://api.fish.audio/model',
                              headers={'Authorization': f'Bearer {FISH_AUDIO_API_KEY}'},
                              params={'self_only': 'true', 'page_size': 100}, timeout=8)
            if r.ok:
                data = r.json()
                items = data.get('items', data if isinstance(data, list) else [])
                for it in items:
                    vid = it.get('_id') or it.get('id')
                    if not vid:
                        continue
                    voices.append({'id': vid, 'title': it.get('title') or vid,
                                    'languages': it.get('languages') or []})
                source = 'cloud'
            else:
                error = f'Fish Audio model list error {r.status_code}: {r.text[:200]}'
        except Exception as e:
            error = f'Fish Audio model list request failed: {e}'
    else:
        # Best-effort probe: some self-hosted servers do expose a listing
        # endpoint even though it's not part of the base S2 spec. If it's not
        # there (404/refused/anything else), we just end up with an empty
        # list — expected for a plain self-hosted instance, not an error.
        base = FISH_AUDIO_URL.rsplit('/v1/', 1)[0] if '/v1/' in FISH_AUDIO_URL else FISH_AUDIO_URL
        try:
            r = requests.get(base.rstrip('/') + '/v1/models', timeout=3)
            if r.ok:
                data = r.json()
                items = data.get('items', data if isinstance(data, list) else [])
                for it in items:
                    vid = it.get('id') or it.get('_id')
                    if not vid:
                        continue
                    voices.append({'id': vid, 'title': it.get('title') or vid,
                                    'languages': it.get('languages') or []})
                    source = 'self_hosted'
        except Exception:
            pass  # no listing endpoint on this server — empty list, not an error

    _VOICES_CACHE['fish_audio'].update(voices=voices, source=source, error=error, fetched_at=now)
    return voices, source, error

def cozy_voice_list_voices(force=False):
    """List voices registered on a local Cozy Voice 3 server, via a best-effort
    probe of COZY_VOICE_URL + '/voices' (adjust the path if your server's
    actual listing endpoint differs). Same return shape as
    fish_audio_list_voices(): (voices, source, error), source is
    'self_hosted' | 'none' | 'error'. No fallback default entry here either —
    an empty list means "upload a reference sample instead."""
    now = time.time()
    cache = _VOICES_CACHE['cozy_voice']
    if not force and cache['voices'] is not None and now - cache['fetched_at'] < _VOICES_CACHE_TTL:
        return cache['voices'], cache['source'], cache['error']

    voices = []
    source = 'none'
    error = None
    try:
        r = requests.get(COZY_VOICE_URL.rstrip('/') + '/voices', timeout=3)
        if r.ok:
            data = r.json()
            items = data.get('items', data if isinstance(data, list) else [])
            for it in items:
                vid = it.get('id') or it.get('voice_id')
                if not vid:
                    continue
                voices.append({'id': vid, 'title': it.get('title') or it.get('name') or vid,
                                'languages': it.get('languages') or []})
            source = 'self_hosted'
    except Exception:
        pass  # no listing endpoint reachable — empty list, not an error

    _VOICES_CACHE['cozy_voice'].update(voices=voices, source=source, error=error, fetched_at=now)
    return voices, source, error

def list_voices_for_engine(engine, force=False):
    """Dispatches to whichever engine's voice list the caller asked for."""
    if engine == 'cozy_voice':
        return cozy_voice_list_voices(force=force)
    return fish_audio_list_voices(force=force)

# ---- Background job tracking (progress reporting for long-running trailer jobs) ----
JOBS = {}
JOBS_LOCK = threading.Lock()
JOB_TTL = 60 * 60  # drop finished jobs after an hour so JOBS doesn't grow forever

class JobCancelled(Exception):
    pass

def job_new():
    jid = f'{int(time.time()*1000)}_{threading.get_ident()}'
    with JOBS_LOCK:
        JOBS[jid] = {'percent': 0, 'step': 'Queued', 'done': False, 'error': None,
                     'result': None, 'created': time.time(), 'cancel_requested': False,
                     'status': 'queued'}
        stale = [k for k, v in JOBS.items() if v.get('done') and time.time() - v.get('created', 0) > JOB_TTL]
        for k in stale:
            JOBS.pop(k, None)
    return jid

def job_set(jid, percent=None, step=None, error=None, done=None, result=None, status=None):
    cancel_now = False
    with JOBS_LOCK:
        j = JOBS.get(jid)
        if not j:
            return
        if percent is not None:
            j['percent'] = percent
        if step is not None:
            j['step'] = step
        if status is not None:
            j['status'] = status
        if error is not None:
            j['error'] = error
            j['done'] = True
            j['status'] = 'error'
        if done is not None:
            j['done'] = done
        if result is not None:
            j['result'] = result
        # Any progress update after a cancellation request raises, so the running
        # job unwinds at its next checkpoint — this call itself (reporting the
        # cancellation) is exempt so it doesn't recursively raise.
        if j.get('cancel_requested') and error is None and not done:
            cancel_now = True
    if cancel_now:
        raise JobCancelled(jid)

def job_cancel(jid):
    """Request cancellation of a queued or running job. Queued jobs are removed
    from the wait line immediately; running jobs unwind at their next progress
    checkpoint (best-effort — an in-flight ffmpeg/API call still finishes first)."""
    with JOBS_LOCK:
        j = JOBS.get(jid)
        if not j or j.get('done'):
            return False
        j['cancel_requested'] = True
    with JOB_QUEUE_LOCK:
        if jid in JOB_QUEUE:
            JOB_QUEUE.remove(jid)
            job_set(jid, error='Cancelled', status='cancelled')
    return True

def job_get(jid):
    with JOBS_LOCK:
        j = JOBS.get(jid)
        return dict(j) if j else None

class JobGate:
    """Caps how many trailer jobs run at once. Limit is adjustable at runtime
    (e.g. via the /api/queue/limit endpoint) without needing to restart anything —
    waiting jobs just re-check the current limit each time they're woken up."""
    def __init__(self, limit):
        self.limit = max(1, limit)
        self.running = 0
        self.cond = threading.Condition()

    def set_limit(self, new_limit):
        with self.cond:
            self.limit = max(1, int(new_limit))
            self.cond.notify_all()

    def status(self):
        with self.cond:
            return {'running': self.running, 'limit': self.limit}

MAX_CONCURRENT_JOBS = int(os.environ.get('MAX_CONCURRENT_JOBS', 2))
GATE = JobGate(MAX_CONCURRENT_JOBS)
JOB_QUEUE = []  # job_ids waiting for a free slot, in submission order
JOB_QUEUE_LOCK = threading.Lock()

def run_trailer_job_gated(jid, params):
    """Entry point used for every submitted job: waits for a free concurrency
    slot (reporting queue position while it waits), then runs the job, then
    frees the slot for the next one in line."""
    with JOB_QUEUE_LOCK:
        JOB_QUEUE.append(jid)
    try:
        with GATE.cond:
            while GATE.running >= GATE.limit:
                with JOB_QUEUE_LOCK:
                    if jid not in JOB_QUEUE:  # cancelled while waiting
                        return
                    ahead = JOB_QUEUE.index(jid)
                job_set(jid, step=(f'Queued — {ahead} job(s) ahead' if ahead else 'Queued — starting shortly'),
                        percent=0, status='queued')
                GATE.cond.wait(timeout=2)
            GATE.running += 1
        with JOB_QUEUE_LOCK:
            if jid in JOB_QUEUE:
                JOB_QUEUE.remove(jid)
        job_set(jid, step='Starting', percent=1, status='running')
        run_trailer_job(jid, params)
    finally:
        with GATE.cond:
            GATE.running = max(0, GATE.running - 1)
            GATE.cond.notify_all()
        with JOB_QUEUE_LOCK:
            if jid in JOB_QUEUE:
                JOB_QUEUE.remove(jid)

GENRE_PROMPTS = {
    'action': 'Epic orchestral action trailer music, dramatic powerful percussion, bold brass, cinematic tension building, instrumental, no vocals',
    'drama': 'Emotional dramatic piano piece with warm strings, gentle build, cinematic ambient soundscape, instrumental, no vocals',
    'horror': 'Dark ambient horror soundtrack, eerie drones, tension, suspenseful creeping atmosphere, instrumental, no vocals',
    'comedy': 'Upbeat cheerful funny comedy background music, lighthearted, playful bright melody, instrumental, no vocals',
    'documentary': 'Cinematic documentary background music, inspiring emotional, ambient thoughtful, soft piano and strings, instrumental, no vocals',
    'thriller': 'Suspenseful thriller background music, tense pulsing rhythm, dark atmospheric pads, building anxiety, instrumental, no vocals',
    'scifi': 'Futuristic sci-fi background music, electronic synthesizers, ethereal pads, cosmic atmospheric, instrumental, no vocals',
    'fantasy': 'Magical fantasy orchestral music, enchanting strings and woodwinds, mysterious yet uplifting, instrumental, no vocals',
    'romance': 'Romantic soft background music, gentle piano, warm strings, tender intimate atmosphere, instrumental, no vocals',
    'adventure': 'Epic adventure orchestral music, heroic brass, sweeping strings, triumphant uplifting, instrumental, no vocals',
    'mystery': 'Intriguing mystery background music, subtle tension, curious piano motif, atmospheric suspense, instrumental, no vocals',
    'western': 'Western style background music, acoustic guitar, harmonica, dusty lonely atmosphere, instrumental, no vocals',
    'sports': 'Energetic sports background music, driving beat, triumphant brass, motivational energetic, instrumental, no vocals',
    'noir': 'Film noir dark jazz background music, smoky saxophone, moody double bass, melancholic detective atmosphere, instrumental, no vocals',
    'war': 'Dramatic war epic music, somber strings, military drums, tragic yet heroic, instrumental, no vocals',
}

GENRE_PRESETS = {
    # transition choices are deliberately genre-signature — 'fade' is only reused
    # across drama/documentary/romance, the tonally-similar "soft/emotional" cluster,
    # which stay differentiated from each other via xfade_dur instead.
    'action': {'transition': 'zoomin', 'xfade_dur': 0.2, 'sfx': True},
    'drama': {'transition': 'fade', 'xfade_dur': 0.6, 'sfx': False},
    'horror': {'transition': 'wipeleft', 'xfade_dur': 0.3, 'sfx': True},
    'comedy': {'transition': 'squeezev', 'xfade_dur': 0.25, 'sfx': True},
    'documentary': {'transition': 'fade', 'xfade_dur': 0.5, 'sfx': False},
    'thriller': {'transition': 'radial', 'xfade_dur': 0.2, 'sfx': True},
    'scifi': {'transition': 'pixelize', 'xfade_dur': 0.3, 'sfx': True},
    'fantasy': {'transition': 'dissolve', 'xfade_dur': 0.5, 'sfx': True},
    'romance': {'transition': 'fade', 'xfade_dur': 0.8, 'sfx': False},
    'adventure': {'transition': 'smoothright', 'xfade_dur': 0.2, 'sfx': True},
    'mystery': {'transition': 'fadeblack', 'xfade_dur': 0.4, 'sfx': False},
    'western': {'transition': 'diagbr', 'xfade_dur': 0.3, 'sfx': True},
    'sports': {'transition': 'slideup', 'xfade_dur': 0.15, 'sfx': True},
    'noir': {'transition': 'circleclose', 'xfade_dur': 0.6, 'sfx': False},
    'war': {'transition': 'distance', 'xfade_dur': 0.3, 'sfx': True},
}

GENRE_NAMES = list(GENRE_PRESETS.keys())

GENRE_SFX_PROMPTS = {
    'action': 'Explosion impact sound effect, cinematic boom, dramatic hit, short burst, no music, sound effect only',
    'horror': 'Eerie horror sound effect, creepy whoosh, tension sting, dark impact, short burst, no music, sound effect only',
    'comedy': 'Funny cartoon boing sound effect, lighthearted pop, comedic sting, short burst, no music, sound effect only',
    'thriller': 'Suspenseful tension hit sound effect, dramatic sting, pulse, short burst, no music, sound effect only',
    'scifi': 'Futuristic sci-fi whoosh sound effect, electronic glitch, cybernetic hit, short burst, no music, sound effect only',
    'adventure': 'Epic orchestral hit sound effect, heroic brass stab, cinematic impact, short burst, no music, sound effect only',
    'western': 'Western gunshot or whip crack sound effect, dusty impact, short burst, no music, sound effect only',
    'sports': 'Stadium crowd hit sound effect, whistle blow, energetic impact, short burst, no music, sound effect only',
    'war': 'Explosion blast sound effect, gunfire burst, military hit, short burst, no music, sound effect only',
    'fantasy': 'Magical sparkle chime sound effect, ethereal shimmer, enchanting twinkle, short burst, no music, sound effect only',
}

GENRE_LAVFI = {
    'default': 'sin(261.63*t)*0.25+sin(329.63*t)*0.18+sin(392.00*t)*0.14+sin(523.25*t)*0.1+sin(130.81*t)*0.12',
    'action': 'sin(55*t)*(1+0.3*sin(4*t))+sin(110*t)*0.4+sin(220*t)*0.2+sin(440*t)*0.1+sin(880*t)*0.05',
    'drama': 'sin(130.81*t)*0.3+sin(196*t)*0.2+sin(261.63*t)*0.15+sin(392*t)*0.08',
    'horror': 'sin(30*t)*0.5+sin(35*t)*0.3+sin(2000*t)*0.05+sin(2100*t)*0.04+random(t)*0.02',
    'comedy': 'sin(523.25*t)*0.3+sin(659.25*t)*0.25+sin(783.99*t)*0.2+sin(1046.5*t)*0.1+sin(1318.5*t)*0.05',
    'documentary': 'sin(261.63*t)*0.2+sin(329.63*t)*0.15+sin(392*t)*0.12+sin(523.25*t)*0.08',
    'thriller': 'sin(50*t)*0.4+sin(100*t)*0.2+sin(150*t)*0.1+sin(800*t)*0.05+sin(1200*t)*0.03',
    'scifi': 'sin(220*t)*0.2+sin(440*t)*0.15+sin(880*t)*0.1+sin(1760*t)*0.05+sin(200*t*(1+0.1*sin(0.5*t)))*0.15',
    'fantasy': 'sin(261.63*t)*0.2+sin(392*t)*0.15+sin(523.25*t)*0.12+sin(659.25*t)*0.08+sin(783.99*t)*0.05',
    'romance': 'sin(261.63*t)*0.25+sin(329.63*t)*0.2+sin(392*t)*0.15+sin(523.25*t)*0.08',
    'adventure': 'sin(65.41*t)*0.3+sin(130.81*t)*0.2+sin(261.63*t)*0.15+sin(392*t)*0.1+sin(523.25*t)*0.08',
    'mystery': 'sin(100*t)*0.3+sin(150*t)*0.15+sin(1200*t)*0.05+sin(1800*t)*0.03',
    'western': 'sin(196*t)*0.25+sin(220*t)*0.15+sin(261.63*t)*0.12+sin(329.63*t)*0.08+sin(392*t)*0.05',
    'sports': 'sin(110*t)*(0.5+0.3*lt(sin(2*t),0))+sin(220*t)*0.2+sin(440*t)*0.1+sin(880*t)*0.05',
    'noir': 'sin(98*t)*0.25+sin(130.81*t)*0.2+sin(196*t)*0.15+sin(246.94*t)*0.1',
    'war': 'sin(55*t)*0.3+sin(65.41*t)*0.2+sin(110*t)*0.15+sin(200*t*(1+0.2*sin(2*t)))*0.1+sin(440*t)*0.05',
}

FFMPEG = shutil.which('ffmpeg') or 'C:\\ffmpeg\\bin\\ffmpeg.exe'
FFPROBE = shutil.which('ffprobe') or 'C:\\ffmpeg\\bin\\ffprobe.exe'

# Precomputed once at import time for the Docs tab's genre reference table —
# pulls straight from GENRE_PRESETS/GENRE_PROMPTS/GENRE_SFX_PROMPTS so the docs
# can never drift out of sync with the actual per-genre behavior.
GENRE_DOCS_ROWS = [{
    'genre': g,
    'transition': GENRE_PRESETS[g]['transition'],
    'xfade_dur': GENRE_PRESETS[g]['xfade_dur'],
    'sfx': GENRE_PRESETS[g]['sfx'],
    'music_theme': GENRE_PROMPTS.get(g, ''),
    'sfx_theme': GENRE_SFX_PROMPTS.get(g, '—'),
} for g in GENRE_NAMES]

# ---- Download/export format options ----
# 'mp4_high' is a genuine, standard H.264 High Profile MP4 — no caveats.
# The two ProRes options use ffmpeg's real prores_ks encoder, which is a
# legitimate, widely-used open implementation (not an approximation).
# 'avci100i' is NOT a certified Panasonic AVC-Intra stream — ffmpeg has no
# actual AVC-Intra encoder. It's a best-effort approximation built from
# all-intra libx264 at a similar spec (1080i, 4:2:2 10-bit, ~100Mb/s CBR),
# meant to be visually/structurally similar, not a guaranteed match for
# equipment that specifically validates the AVC-Intra codec ID.
EXPORT_FORMATS = {
    'mp4_high':        {'ext': 'mp4', 'label': 'MP4 (H.264 High Profile)'},
    'prores_hq_2997':  {'ext': 'mov', 'label': 'Apple ProRes 422 HQ — 29.97fps'},
    'prores_hq_2398':  {'ext': 'mov', 'label': 'Apple ProRes 422 HQ — 23.976fps'},
    'avci100i':        {'ext': 'mov', 'label': 'AVC-Intra 100i (H.264 Intra approximation)'},
}

def build_export_cmd(src, dst, fmt_key):
    """Build the ffmpeg command that transcodes a finished trailer (src) into the
    requested delivery format (dst). Returns None for an unknown fmt_key."""
    if fmt_key == 'mp4_high':
        # A distinct higher-quality delivery pass vs. the crf22/fast preset used
        # internally while assembling the trailer.
        return [FFMPEG, '-y', '-i', src,
                '-c:v', 'libx264', '-profile:v', 'high', '-pix_fmt', 'yuv420p',
                '-preset', 'slow', '-crf', '16',
                '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', dst]
    if fmt_key == 'prores_hq_2997':
        return [FFMPEG, '-y', '-i', src, '-vf', 'fps=30000/1001',
                '-c:v', 'prores_ks', '-profile:v', '3', '-vendor', 'apl0',
                '-pix_fmt', 'yuv422p10le', '-c:a', 'pcm_s16le', dst]
    if fmt_key == 'prores_hq_2398':
        return [FFMPEG, '-y', '-i', src, '-vf', 'fps=24000/1001',
                '-c:v', 'prores_ks', '-profile:v', '3', '-vendor', 'apl0',
                '-pix_fmt', 'yuv422p10le', '-c:a', 'pcm_s16le', dst]
    if fmt_key == 'avci100i':
        return [FFMPEG, '-y', '-i', src, '-vf', 'fps=30000/1001,format=yuv422p10le',
                '-c:v', 'libx264', '-profile:v', 'high422',
                '-x264-params', 'keyint=1:bframes=0:cabac=1:interlaced=1',
                '-b:v', '100M', '-minrate', '100M', '-maxrate', '100M', '-bufsize', '100M',
                '-flags', '+ildct+ilme', '-pix_fmt', 'yuv422p10le',
                '-c:a', 'pcm_s16le', dst]
    return None

# Find ONNX model for face detection
ONNX_PATH = next((p for p in [
    os.path.join(os.environ.get('TEMP', '/tmp'), 'face_detection_yunet.onnx'),
    os.path.join(os.environ.get('TMP', '/tmp'), 'face_detection_yunet.onnx'),
    os.path.join(str(pathlib.Path.home()), 'face_detection_yunet.onnx'),
    os.path.join(os.path.dirname(cv2.__file__), 'face_detection_yunet.onnx'),
    os.path.join(os.path.dirname(cv2.__file__), 'data', 'face_detection_yunet.onnx'),
] if os.path.exists(p)), None)

# Cache the face detector (reused across requests)
_fd_lock = threading.Lock()
_fd = None

def get_fd(w, h):
    global _fd
    if _fd is None:
        with _fd_lock:
            if _fd is None:
                _fd = cv2.FaceDetectorYN.create(model=ONNX_PATH, config='', input_size=(320, 320),
                                                score_threshold=0.8, nms_threshold=0.3, top_k=5000)
    _fd.setInputSize((w, h))
    return _fd

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_video(req):
    if 'file' in req.files and req.files['file'].filename != '':
        f = req.files['file']
        if not allowed_file(f.filename):
            return None, 'File type not allowed'
        fn = secure_filename(f.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], fn)
        f.save(path)
        return path, fn
    return None, 'No video provided'

def _scene_desc(s):
    ai = s.get('ai_desc', '') or ''
    if ai:
        return ai
    tags = []
    if s.get('has_face'): tags.append('talking heads')
    edge = s.get('edge_ratio', 0)
    if edge > 0.15: tags.append('high detail')
    elif edge > 0.06: tags.append('detailed')
    else: tags.append('smooth')
    hue = s.get('mean_hue', 0)
    sat = s.get('mean_sat', 0)
    val = s.get('mean_val', 0)
    if sat < 30: tags.append('desaturated')
    elif sat > 100: tags.append('vibrant')
    if val < 40: tags.append('dark scene')
    elif val > 200: tags.append('bright scene')
    else: tags.append('daylight')
    if 90 < hue < 150: tags.append('outdoor/greens')
    elif 0 < hue < 30 or 160 < hue < 180: tags.append('warm tones')
    elif 90 < hue < 150: tags.append('cool tones')
    dur = s.get('duration', 0)
    if dur > 5: tags.append('long take')
    return ' | '.join(tags)

def beat_match_audio(video_path, bgm_path, target_dur, output_path):
    try:
        import librosa
        # Extract audio from source video, resample to consistent rate
        audio_tmp = os.path.join(app.config['UPLOAD_FOLDER'], f'beat_video_{int(time.time())}.wav')
        subprocess.run([FFMPEG, '-y', '-i', video_path, '-vn', '-ar', '22050', '-ac', '1', audio_tmp],
                       capture_output=True, text=True, timeout=60)
        if not os.path.exists(audio_tmp) or os.path.getsize(audio_tmp) == 0:
            return False
        y_vid, sr = librosa.load(audio_tmp, sr=22050)
        os.remove(audio_tmp)
        tempo_vid, _ = librosa.beat.beat_track(y=y_vid, sr=sr)
        tempo_vid = float(tempo_vid)
        if tempo_vid < 30 or tempo_vid > 300:
            tempo_vid = 120
    except Exception as e:
        print(f'Beat detection error: {e}')
        return False

    try:
        y_bgm, sr_bgm = librosa.load(bgm_path, sr=22050)
        orig_len = len(y_bgm)
        # Detect BGM tempo
        tempo_bgm, _ = librosa.beat.beat_track(y=y_bgm, sr=sr_bgm)
        tempo_bgm = float(tempo_bgm)
        if tempo_bgm < 30 or tempo_bgm > 300:
            tempo_bgm = tempo_vid

        # Time-stretch to match video tempo (preserves pitch)
        stretch = tempo_vid / tempo_bgm
        if abs(stretch - 1.0) > 0.01:
            y_bgm = librosa.effects.time_stretch(y=y_bgm, rate=stretch)

        # Loop or trim to fill target_dur seconds
        target_samples = int(target_dur * sr_bgm)
        if len(y_bgm) < target_samples:
            repeats = int(np.ceil(target_samples / len(y_bgm)))
            y_bgm = np.tile(y_bgm, repeats)
        y_bgm = y_bgm[:target_samples]

        # Write processed BGM with proper sample rate
        import soundfile as sf
        sf.write(output_path, y_bgm, sr_bgm)
        return os.path.exists(output_path) and os.path.getsize(output_path) > 0
    except Exception as e:
        print(f'Beat match processing error: {e}')
        return False

def apply_filter(frame, mode, prev_gray=None):
    if mode == 'gray':
        return cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR), prev_gray
    if mode == 'edges':
        return cv2.cvtColor(cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 100, 200), cv2.COLOR_GRAY2BGR), prev_gray
    if mode == 'hsv':
        return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), prev_gray
    if mode == 'blur':
        return cv2.GaussianBlur(frame, (15, 15), 0), prev_gray
    if mode == 'face':
        if ONNX_PATH is None:
            return frame, prev_gray
        h, w = frame.shape[:2]
        _, faces = get_fd(w, h).detect(frame)
        if faces is not None:
            for f in faces:
                x, y, fw, fh = map(int, f[:4])
                cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 3)
        return frame, prev_gray
    if mode == 'motion':
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)
        if prev_gray is None:
            return frame, gray
        diff = cv2.absdiff(prev_gray, gray)
        thresh = cv2.dilate(cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1], None, iterations=2)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        n = 0
        for c in cnts:
            if cv2.contourArea(c) > 500:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                n += 1
        if n:
            cv2.putText(frame, f'Motion: {n}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame, gray
    return frame, prev_gray

def synth_sfx_waveform(genre, sample_rate=22050, sfx_dur=0.6):
    """Synthesize a single procedural 'hit' waveform for a genre. Returns a
    1-D float array in [-1, 1], or None if the genre has no synth SFX defined."""
    n_sfx = int(sfx_dur * sample_rate)
    t = np.arange(n_sfx) / sample_rate
    if genre == 'action' or genre == 'war':
        # low sine "thump" for body + short filtered-feeling noise burst for texture,
        # instead of pure white noise (which reads as hiss, not boom)
        thump = np.sin(2*np.pi * 65 * t) * np.exp(-t * 14)
        crackle = np.random.randn(n_sfx) * np.exp(-t * 22) * 0.35
        sfx = thump * 0.8 + crackle
    elif genre == 'horror':
        # lower, narrower sweep than before (was up to 8kHz — shrill/painful);
        # add a sub-bass rumble underneath for dread rather than a piercing shriek
        sweep = np.sin(2*np.pi * (1200 + t * 3000) * t) * np.exp(-t * 8) * 0.25
        rumble = np.sin(2*np.pi * 42 * t) * np.exp(-t * 6) * 0.3
        sfx = sweep + rumble
    elif genre == 'comedy':
        sfx = np.sin(2*np.pi * (600 - t * 1200) * t) * np.exp(-t * 5) * 0.4
    elif genre in ('thriller', 'adventure', 'scifi'):
        sfx = np.sin(2*np.pi * (100 + t * 3000) * t) * np.exp(-t * 6) * 0.3
    elif genre == 'western':
        # sharp transient crack plus a touch of low-mid body so it reads as a
        # gunshot/whip-crack rather than a thin, bodyless tick
        crack = np.random.randn(n_sfx) * np.exp(-t * 35) * 0.55
        body = np.sin(2*np.pi * 180 * t) * np.exp(-t * 16) * 0.3
        sfx = crack + body
    elif genre == 'sports':
        # sine-based whistle tone in real whistle range (~2.8-3.1kHz) instead of a
        # raw square wave, which aliases into a harsh electronic buzz
        sfx = (np.sin(2*np.pi * 2800 * t) * 0.35 + np.sin(2*np.pi * 3100 * t) * 0.2) * np.exp(-t * 4)
    elif genre == 'fantasy':
        sfx = (np.sin(2*np.pi * 528 * t) * 0.3 + np.sin(2*np.pi * 1056 * t) * 0.15 +
               np.sin(2*np.pi * 1584 * t) * 0.08) * np.exp(-t * 4)
    else:
        return None
    peak = np.max(np.abs(sfx))
    if peak > 0:
        sfx = sfx / peak * 0.85
    return sfx

def load_hit_waveform(path, sample_rate=22050, max_dur=1.2):
    """Load an uploaded or AI-generated one-shot SFX file (any format ffmpeg/librosa
    can read) as a short mono waveform, trimmed and fade-tailed so it behaves like
    a 'hit' when stamped at multiple cut points. Returns None on failure."""
    try:
        import librosa
        y, sr = librosa.load(path, sr=sample_rate, mono=True, duration=max_dur)
        if y is None or len(y) == 0:
            return None
        # short fade-out so repeated stamping never clicks at the tail
        fade_n = min(len(y), int(0.05 * sample_rate))
        if fade_n > 0:
            y[-fade_n:] *= np.linspace(1, 0, fade_n)
        peak = np.max(np.abs(y))
        if peak > 0:
            y = y / peak * 0.85
        return y
    except Exception as e:
        print(f'SFX load error: {e}')
        return None

def write_wav_pcm16(track, output_path, sample_rate=22050):
    track = np.clip(track, -1, 1)
    track_int = (track * 32767).astype(np.int16)
    import struct
    with open(output_path, 'wb') as f:
        data_len = len(track_int) * 2
        f.write(b'RIFF')
        f.write(struct.pack('<I', 36 + data_len))
        f.write(b'WAVE')
        f.write(b'fmt ')
        f.write(struct.pack('<I', 16))
        f.write(struct.pack('<H', 1))
        f.write(struct.pack('<H', 1))
        f.write(struct.pack('<I', sample_rate))
        f.write(struct.pack('<I', sample_rate * 2))
        f.write(struct.pack('<H', 2))
        f.write(struct.pack('<H', 16))
        f.write(b'data')
        f.write(struct.pack('<I', data_len))
        f.write(track_int.tobytes())
    return os.path.exists(output_path) and os.path.getsize(output_path) > 0

def stamp_hits(hit_wave, timestamps, output_path, sample_rate=22050):
    """Place a copy of `hit_wave` at every timestamp (seconds) into one continuous
    track and write it as a WAV. This is what makes SFX land on *every* cut,
    regardless of whether the hit sound came from synth, upload, or ACE-Step."""
    try:
        if hit_wave is None or len(hit_wave) == 0 or not timestamps:
            return False
        total_dur = max(timestamps) + (len(hit_wave) / sample_rate) + 0.5
        total_samples = int(total_dur * sample_rate)
        track = np.zeros(total_samples)
        for i, ts in enumerate(timestamps):
            # slight per-hit variation (pitch + level) so repeated cuts on a long
            # trailer don't sound like an obviously copy-pasted stock sound
            rng = np.random.RandomState(int(ts * 1000) & 0xffffffff)
            pitch = 1.0 + rng.uniform(-0.08, 0.08)
            amp = 0.9 + rng.uniform(-0.1, 0.1)
            if abs(pitch - 1.0) > 1e-6:
                idx = np.clip((np.arange(len(hit_wave)) * pitch).astype(int), 0, len(hit_wave) - 1)
                hit = hit_wave[idx] * amp
            else:
                hit = hit_wave * amp
            start = int(ts * sample_rate)
            end = min(start + len(hit), total_samples)
            if end > start:
                track[start:end] += hit[:end - start]
        return write_wav_pcm16(track, output_path, sample_rate)
    except Exception as e:
        print(f'SFX stamping error: {e}')
        return False

def woosh_sfx(genre, output_path, duration=0.8):
    """Generate a one-shot SFX using Sony AI's Woosh text-to-audio model via its local
    API server (see github.com/SonyResearch/Woosh — 'Woosh models can be served via
    our API server'). This assumes a simple synchronous POST that returns raw audio
    bytes; adjust the path/payload below to match however your Woosh server is set up.
    Returns True/False — failures fall through silently so the caller can try
    ACE-Step next, then the procedural synth fallback."""
    try:
        prompt = GENRE_SFX_PROMPTS.get(genre)
        if not prompt:
            return False
        r = requests.post(f'{WOOSH_URL}/generate', json={
            'prompt': prompt,
            'duration': duration,
        }, timeout=15)
        if r.ok and r.content:
            with open(output_path, 'wb') as f:
                f.write(r.content)
            return os.path.getsize(output_path) > 0
    except Exception as e:
        print(f'Woosh SFX error: {e}')
    return False

def acestep_sfx(genre, output_path, duration=0.8):
    try:
        prompt = GENRE_SFX_PROMPTS.get(genre)
        if not prompt:
            return False
        r = requests.post(f'{ACE_STEP_URL}/release_task', json={
            'prompt': prompt,
            'audio_duration': duration,
            'thinking': False,
            'inference_steps': 8,
            'batch_size': 1,
        }, timeout=5)
        data = r.json()
        task_id = data.get('data', {}).get('task_id')
        if not task_id:
            return False
        for _ in range(30):
            time.sleep(2)
            q = requests.post(f'{ACE_STEP_URL}/query_result', json={
                'task_id_list': [task_id]
            }, timeout=5)
            qd = q.json()
            items = qd.get('data', [])
            if items and items[0].get('status') == 1:
                result = json.loads(items[0]['result'])
                audio_path = result[0]['file'] if isinstance(result, list) else result.get('file', '')
                if audio_path:
                    dl_url = f'{ACE_STEP_URL}{audio_path}'
                    resp = requests.get(dl_url, timeout=30)
                    with open(output_path, 'wb') as f:
                        f.write(resp.content)
                    return os.path.getsize(output_path) > 0
            elif items and items[0].get('status') == 2:
                break
    except Exception as e:
        print(f'ACE-Step SFX error: {e}')
    return False

def generate_tts(text, output_wav_path, rate=175, voice_id=None, reference_audio_path=None, language=None, engine='fish_audio'):
    """Generate a narration WAV from text, via whichever engine the user picked:
    'fish_audio' (Fish Audio S2, voice cloning, auto-detects language including
    Tagalog) or 'cozy_voice' (Cozy Voice 3). Returns (ok, error_message). There's
    no bundled default voice — the caller must pass either `voice_id` (a voice
    picked from list_voices_for_engine()) or `reference_audio_path` (an uploaded
    sample to clone zero-shot); if neither is given, this returns an error
    rather than silently falling back to some fixed voice file."""
    text = (text or '').strip()
    if not text:
        return False, 'No text provided'
    if not voice_id and not (reference_audio_path and os.path.exists(reference_audio_path)):
        return False, 'No voice selected — choose a voice from the list or upload a reference sample to clone.'
    engine_fn, engine_label = {
        'cozy_voice': (cozy_voice_tts, 'Cozy Voice 3'),
        'fish_audio': (fish_audio_tts, 'Fish Audio'),
    }.get(engine, (fish_audio_tts, 'Fish Audio'))
    try:
        ok, err = engine_fn(text, output_wav_path, voice_id=voice_id, rate=rate,
                             reference_audio_path=reference_audio_path, language=language)
        if ok:
            return True, None
        print(f'{engine_label} TTS error: {err}')
        return False, f'{engine_label}: {err}'
    except Exception as e:
        print(f'{engine_label} unavailable: {e}')
        return False, f'{engine_label} unavailable: {e}'

def prepare_bgm_track(genre, scoring_mode, scoring_audio_path, duration, base_ts, fade_in=2.0, fade_out=3.0):
    """Produce a ready-to-mix BGM track (AAC .m4a, faded, trimmed to `duration`).
    Shared by the early beat-sync pass (approximate target duration) and the
    final mix pass (reused as-is if already prepared, else generated fresh).
    Returns (path_or_None, source) where source is 'uploaded' | 'ai_generated' | 'synth_fallback' | 'none'."""
    bgm_source = 'none'
    if not scoring_audio_path:
        return None, bgm_source
    if scoring_audio_path == 'GENERATE':
        gen_audio = os.path.join(app.config['UPLOAD_FOLDER'], f'gen_{base_ts}_{int(time.time()*1000)%100000}.m4a')
        acestep_ok = False
        try:
            prompt = GENRE_PROMPTS.get(genre, 'Cinematic background music, instrumental, no vocals')
            r = requests.post(f'{ACE_STEP_URL}/release_task', json={
                'prompt': prompt, 'audio_duration': duration, 'thinking': False,
                'inference_steps': 8, 'batch_size': 1,
            }, timeout=5)
            data = r.json()
            task_id = data.get('data', {}).get('task_id')
            if task_id:
                for _ in range(60):
                    time.sleep(2)
                    q = requests.post(f'{ACE_STEP_URL}/query_result', json={'task_id_list': [task_id]}, timeout=5)
                    qd = q.json()
                    items = qd.get('data', [])
                    if items and items[0].get('status') == 1:
                        result = json.loads(items[0]['result'])
                        audio_path = result[0]['file'] if isinstance(result, list) else result.get('file', '')
                        if audio_path:
                            dl_url = f'{ACE_STEP_URL}{audio_path}'
                            resp = requests.get(dl_url, timeout=60)
                            with open(gen_audio, 'wb') as f:
                                f.write(resp.content)
                            if os.path.getsize(gen_audio) > 0:
                                acestep_ok = True
                        break
                    elif items and items[0].get('status') == 2:
                        break
        except Exception as e:
            print(f'ACE-Step error: {e}')
        if acestep_ok:
            bgm_source = 'ai_generated'
        else:
            lavfi_src = GENRE_LAVFI.get(genre, GENRE_LAVFI['default'])
            subprocess.run([FFMPEG, '-y', '-f', 'lavfi', '-i',
                            f'aevalsrc=exprs=\'{lavfi_src}\':d={duration}:s=44100:c=stereo',
                            '-af',
                            f'lowpass=f=6000,tremolo=f=0.15:d=0.4,volume=1.5,'
                            f'aecho=0.8:0.7:60:0.25,'
                            f'afade=t=in:d={fade_in},'
                            f'afade=t=out:st={max(duration - fade_out, 0)}:d={min(fade_out, duration)}',
                            '-c:a', 'aac', '-b:a', '192k', gen_audio],
                           capture_output=True, text=True, timeout=30)
            bgm_source = 'synth_fallback'
        if os.path.exists(gen_audio) and os.path.getsize(gen_audio) > 0:
            return gen_audio, bgm_source
        return None, 'none'
    else:
        processed_audio = os.path.join(app.config['UPLOAD_FOLDER'], f'score_{base_ts}_{int(time.time()*1000)%100000}.m4a')
        r = subprocess.run([FFMPEG, '-y', '-i', scoring_audio_path,
                            '-af', (f'atrim=duration={duration},'
                                    f'afade=t=in:d={fade_in},'
                                    f'afade=t=out:st={max(duration - fade_out, 0)}:d={min(fade_out, duration)}'),
                            '-c:a', 'aac', '-b:a', '192k', '-vn', processed_audio],
                           capture_output=True, text=True, timeout=60)
        if os.path.exists(processed_audio) and os.path.getsize(processed_audio) > 0:
            return processed_audio, 'uploaded'
        return None, 'none'

def finalize_bgm_duration(src_path, duration, base_ts, fade_in=2.0, fade_out=3.0):
    """Re-trim/pad + re-fade an already-generated BGM track to an exact final
    duration (used when the early beat-sync pass generated it against an
    estimate that ended up slightly off from the final trailer length)."""
    out = os.path.join(app.config['UPLOAD_FOLDER'], f'bgmfit_{base_ts}_{int(time.time()*1000)%100000}.m4a')
    r = subprocess.run([FFMPEG, '-y', '-i', src_path,
                        '-af', (f'atrim=duration={duration},apad=whole_dur={duration},'
                                f'afade=t=in:d={fade_in},'
                                f'afade=t=out:st={max(duration - fade_out, 0)}:d={min(fade_out, duration)}'),
                        '-c:a', 'aac', '-b:a', '192k', out],
                       capture_output=True, text=True, timeout=30)
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return out
    return None

def mux_card_vo(video_path, vo_path, trim_start, trim_end, output_path):
    """Replace a title/end card video's audio with a trimmed window of an uploaded
    VO file: [trim_start, trim_end) seconds of vo_path (trim_end=None means to the
    end of the file). The VO is padded with silence if shorter than the card video
    so the card keeps its full original length either way. Returns output_path on
    success, or None if the mux failed (caller should keep the card's original
    audio/video untouched in that case)."""
    cmd = [FFMPEG, '-y', '-i', video_path, '-ss', str(trim_start)]
    if trim_end is not None:
        cmd.extend(['-to', str(trim_end)])
    cmd.extend(['-i', vo_path,
                '-map', '0:v', '-map', '1:a',
                '-af', 'apad', '-shortest',
                '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', output_path])
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return output_path
    print(f'Card VO mux error ({video_path}): {r.stderr[:500]}')
    return None

def detect_beat_times(audio_path, duration):
    """Return a sorted list of beat timestamps (seconds) within an audio file, or
    an empty list if librosa/beat detection isn't available. Used to snap cut
    points onto the music so edits land 'on the beat'."""
    try:
        import librosa
        y, sr = librosa.load(audio_path, sr=22050, duration=duration)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        times = librosa.frames_to_time(beat_frames, sr=sr)
        return sorted(float(t) for t in times)
    except Exception as e:
        print(f'Beat detection error: {e}')
        return []

def nearest_beat(target, beats, lo, hi):
    """Nearest beat time to `target` within [lo, hi]; falls back to `target` if none in range."""
    candidates = [b for b in beats if lo <= b <= hi]
    if not candidates:
        return target
    return min(candidates, key=lambda b: abs(b - target))

# ---- whisper service — dialogue transcription to improve scene selection ----
def transcribe_video(path):
    """Transcribe the source video's dialogue via the local whisper service
    (WHISPER_URL, an OpenAI-compatible /v1/audio/transcriptions endpoint), with
    word-level timestamps. Returns (words, segments):
      words:    [{'start','end','word'}, ...]
      segments: [{'start','end','text'}, ...]
    Returns ([], []) if the service is unreachable or transcription fails —
    callers should treat that as 'feature unavailable' and continue without it."""
    try:
        with open(path, 'rb') as f:
            r = requests.post(
                f'{WHISPER_URL}/v1/audio/transcriptions',
                files={'file': (os.path.basename(path), f, 'application/octet-stream')},
                data={
                    'model': WHISPER_MODEL,
                    'response_format': 'verbose_json',
                    'timestamp_granularities[]': 'word',
                },
                timeout=600,
            )
        r.raise_for_status()
        data = r.json()
        words, segments = [], []
        for seg in data.get('segments', []):
            text = (seg.get('text') or '').strip()
            if text:
                segments.append({'start': seg['start'], 'end': seg['end'], 'text': text})
        for w in data.get('words', []):
            word = (w.get('word') or '').strip()
            if word:
                words.append({'start': w['start'], 'end': w['end'], 'word': word})
        return words, segments
    except Exception as e:
        print(f'Whisper transcription error (service at {WHISPER_URL}): {e}')
        return [], []

def nearest_word_boundary(target, boundaries, max_snap=0.35):
    """Nearest timestamp in `boundaries` to `target`, but only if within
    `max_snap` seconds — otherwise returns `target` unchanged (no nearby word
    to snap to, e.g. a silent B-roll clip, so leave the cut point as-is)."""
    if not boundaries:
        return target
    candidates = [b for b in boundaries if abs(b - target) <= max_snap]
    if not candidates:
        return target
    return min(candidates, key=lambda b: abs(b - target))

HOOK_KEYWORDS = {'never', 'always', 'everyone', 'no one', 'everything', 'nothing',
                  'must', 'only', 'last', 'first', 'why', 'how', 'secret', 'truth',
                  'promise', 'forever', 'impossible', 'anymore', 'again', 'stop', 'run'}

def score_hook_line(text):
    """Lightweight heuristic score for how 'quotable' a line of dialogue is —
    used to surface a suggested hook line, not to auto-insert anything.
    Higher = more likely to work as a trailer pull-quote."""
    t = (text or '').strip()
    words_n = len(t.split())
    if words_n < 3:
        return -1
    score = min(words_n, 14)
    if '?' in t:
        score += 4
    if '!' in t:
        score += 3
    low = t.lower()
    score += sum(2 for kw in HOOK_KEYWORDS if kw in low)
    return score

def detect_scenes(path, threshold=30.0, min_scene_len_sec=0.5, downscale=None):
    """Run PySceneDetect's ContentDetector over `path` and return the raw
    scene list [(start_timecode, end_timecode), ...].

    threshold and min_scene_len_sec are parameters (not hardcoded) so every
    caller — the live preview endpoint, the vision-analyze endpoint, and the
    actual trailer-generation job — can agree on the same detection settings
    instead of the job silently using its own fixed threshold=30.0 regardless
    of what a user picked in the preview.

    min_scene_len_sec filters out sub-fragment "scenes" (whip-pans, motion
    blur, flash cuts) that PySceneDetect's frame-count default would otherwise
    let through; it's converted to frames using the source's actual frame rate.

    downscale, if given, scales frames down by that factor during detection
    only (does not affect the returned timecodes) — a cheap way to speed up
    detection on large source files with negligible accuracy loss.
    """
    video = open_video(path)
    if downscale:
        try:
            video.set_downscale_factor(downscale)
        except Exception:
            pass
    fps = video.frame_rate or 30.0
    min_scene_len = max(1, int(round(min_scene_len_sec * fps)))
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_scene_len))
    sm.detect_scenes(video)
    return sm.get_scene_list()

def get_video_info(path):
    cap = cv2.VideoCapture(path)
    info = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': round(cap.get(cv2.CAP_PROP_FPS), 2),
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    info['duration_sec'] = round(info['total_frames'] / info['fps'], 2) if info['fps'] > 0 else 0
    cap.release()
    return info

# ---- API ----

@app.route('/api/opencv/info', methods=['POST'])
def api_info():
    path, err = load_video(request)
    if not path:
        return jsonify(error=err), 400
    return jsonify(video_info=get_video_info(path))

@app.route('/api/opencv/analyze', methods=['POST'])
def api_analyze():
    path, err = load_video(request)
    if not path:
        return jsonify(error=err), 400
    n = min(int(request.form.get('num_frames', 10)), 100)
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total // n, 1)
    frames = []
    for i in range(0, total, step):
        if len(frames) >= n:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, f = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=20, qualityLevel=0.01, minDistance=10)
        frames.append({'idx': i, 'shape': list(f.shape),
                       'mean_bgr': [round(float(c), 1) for c in cv2.mean(f)[:3]],
                       'brightness': round(float(np.mean(gray)), 1),
                       'edge_pixels': int(np.sum(edges > 0)),
                       'corners': len(corners) if corners is not None else 0})
    cap.release()
    return jsonify(frames=frames)

@app.route('/api/scenedetect/detect', methods=['POST'])
def api_sd():
    path, err = load_video(request)
    if not path:
        return jsonify(error=err), 400
    th = float(request.form.get('threshold', 30.0))
    try:
        min_len = max(0.1, min(5.0, float(request.form.get('min_scene_len', 0.5))))
    except ValueError:
        min_len = 0.5
    scene_list = detect_scenes(path, threshold=th, min_scene_len_sec=min_len)
    scenes = [{'scene': i+1, 'start': s.get_timecode(), 'end': e.get_timecode(),
               'start_sec': round(s.get_seconds(), 2), 'end_sec': round(e.get_seconds(), 2),
               'duration': round(e.get_seconds() - s.get_seconds(), 2)}
              for i, (s, e) in enumerate(scene_list)]
    return jsonify(scenes=scenes)

# ---- Playback ----

_pb_lock = threading.Lock()
_pb_cap = None
_pb_paused = False

def _ensure_readable(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.mov', '.mkv', '.flv', '.wmv', '.webm'):
        cap = cv2.VideoCapture(path)
        ret, _ = cap.read()
        cap.release()
        if not ret:
            mp4_path = os.path.splitext(path)[0] + '_converted.mp4'
            r = subprocess.run([FFMPEG, '-y', '-i', path, '-c:v', 'libx264', '-preset', 'ultrafast',
                                '-crf', '28', '-pix_fmt', 'yuv420p', '-an', mp4_path],
                               capture_output=True, text=True)
            if r.returncode == 0 and os.path.exists(mp4_path):
                return mp4_path
    return path

@app.route('/api/playback/start', methods=['POST'])
def pb_start():
    global _pb_cap, _pb_paused
    path, err = load_video(request)
    if not path:
        return jsonify(error=err), 400
    path = _ensure_readable(path)
    with _pb_lock:
        if _pb_cap:
            _pb_cap.release()
        _pb_cap = cv2.VideoCapture(path)
        _pb_paused = False
        info = get_video_info(path)
    return jsonify(status='ok', **info)

@app.route('/api/playback/stream/<mode>')
def pb_stream(mode):
    global _pb_cap
    if mode not in ('raw', 'gray', 'edges', 'hsv', 'blur', 'face', 'motion'):
        mode = 'raw'
    def gen():
        global _pb_paused
        prev = None
        cap = _pb_cap
        if cap is None:
            return
        while True:
            with _pb_lock:
                if _pb_paused:
                    time.sleep(0.05)
                    continue
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop
                    continue
                frame, prev = apply_filter(frame, mode, prev)
                _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n'
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/playback/pause')
def pb_pause():
    global _pb_paused
    _pb_paused = True
    return jsonify(status='paused')

@app.route('/api/playback/resume')
def pb_resume():
    global _pb_paused
    _pb_paused = False
    return jsonify(status='resumed')

@app.route('/api/playback/seek')
def pb_seek():
    with _pb_lock:
        if _pb_cap:
            t = float(request.args.get('t', 0))
            _pb_cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * _pb_cap.get(cv2.CAP_PROP_FPS)))
    return jsonify(status='ok')

@app.route('/api/playback/stop')
def pb_stop():
    global _pb_cap
    with _pb_lock:
        if _pb_cap:
            _pb_cap.release()
            _pb_cap = None
    return '', 204

# ---- Ollama Vision ----

OLLAMA_URL = os.environ.get('OLLAMA_URL', 'http://localhost:11434')

@app.route('/api/vision/analyze', methods=['POST'])
def api_vision():
    path, err = load_video(request)
    if not path:
        return jsonify(error=err), 400
    prompt = request.form.get('prompt', 'Describe what is happening in this video frame in 1-2 sentences.')
    num_frames = min(int(request.form.get('num_frames', 5)), 20)
    model = request.form.get('model', 'llama3.2-vision:11b')

    path = _ensure_readable(path)

    scene_list = detect_scenes(path, threshold=30.0)
    scenes = [{'scene': i+1, 'start': s.get_seconds(), 'end': e.get_seconds(),
               'start_tc': s.get_timecode(), 'end_tc': e.get_timecode(),
               'duration': round(e.get_seconds() - s.get_seconds(), 2)}
              for i, (s, e) in enumerate(scene_list)]

    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    frames_to_analyze = []
    if scenes:
        for sc in scenes:
            mid_sec = (sc['start'] + sc['end']) / 2
            mid_frame = int(mid_sec * fps) if fps else 0
            frames_to_analyze.append({'frame_idx': mid_frame, 'time_sec': round(mid_sec, 2), 'scene': sc})
    else:
        step = max(total // num_frames, 1) if total > num_frames else 1
        for i in range(0, total, step):
            if len(frames_to_analyze) >= num_frames:
                break
            ts = round(i / fps, 2) if fps > 0 else 0
            frames_to_analyze.append({'frame_idx': i, 'time_sec': ts, 'scene': None})

    results = []
    for fa in frames_to_analyze[:num_frames]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fa['frame_idx'])
        ret, frame = cap.read()
        if not ret:
            continue
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64 = base64.b64encode(buf.tobytes()).decode()

        scene_ctx = ''
        if fa['scene']:
            s = fa['scene']
            scene_ctx = f' (Scene {s["scene"]}, {s["start_tc"]}-{s["end_tc"]}, {s["duration"]}s)'
        full_prompt = prompt + scene_ctx

        try:
            r = requests.post(f'{OLLAMA_URL}/api/generate', json={
                'model': model, 'prompt': full_prompt, 'stream': False,
                'images': [b64]
            }, timeout=300)
            data = r.json()
            resp = data.get('response', '')
        except Exception as e:
            resp = f'Error: {e}'

        entry = {'frame_idx': fa['frame_idx'], 'time_sec': fa['time_sec'], 'ollama_response': resp}
        if fa['scene']:
            entry['scene'] = fa['scene']['scene']
            entry['scene_start'] = fa['scene']['start_tc']
            entry['scene_end'] = fa['scene']['end_tc']
            entry['scene_duration'] = fa['scene']['duration']
        results.append(entry)

    cap.release()
    return jsonify(frames_analyzed=len(results), total_scenes=len(scenes), results=results)

@app.errorhandler(413)
def too_large(e):
    return jsonify(error='File too large (max 2GB).'), 413

def _model_supports_vision(name):
    # /api/tags does not include per-model capabilities, so we have to ask
    # /api/show for each model individually to find out if it's a vision model.
    try:
        r = requests.post(f'{OLLAMA_URL}/api/show', json={'model': name}, timeout=10)
        data = r.json()
        caps = data.get('capabilities', [])
        if caps:
            return 'vision' in caps
        # Older Ollama versions don't return `capabilities` at all - fall back
        # to checking the projector/family info that vision models expose.
        details = data.get('details', {}) or {}
        families = (details.get('families') or []) + [details.get('family', '')]
        if any('clip' in f.lower() or 'mllama' in f.lower() or 'vision' in f.lower() for f in families if f):
            return True
        return 'vision' in name.lower() or 'vl' in name.lower()
    except Exception:
        # If we can't introspect the model, guess from its name rather than
        # silently dropping it from the list.
        return 'vision' in name.lower() or 'vl' in name.lower()

@app.route('/api/vision/models')
def api_vision_models():
    try:
        r = requests.get(f'{OLLAMA_URL}/api/tags', timeout=10)
        names = [m['name'] for m in r.json().get('models', [])]
    except Exception:
        return jsonify(models=[])
    if not names:
        return jsonify(models=[])
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(8, len(names))) as ex:
        flags = list(ex.map(_model_supports_vision, names))
    models = [n for n, ok in zip(names, flags) if ok]
    return jsonify(models=models)

# ---- Trailer Generator (ffmpeg) ----

@app.route('/api/trailer/generate', methods=['POST'])
def api_trailer():
    path, orig_name = load_video(request)
    if not path:
        return jsonify(error=orig_name), 400

    # Scoring mode: 'ai' (OpenCV + Ollama Vision) or 'ai_stt' (adds faster-whisper
    # dialogue transcription on top of Vision scoring). 'ai_stt' is normalized down to
    # 'ai' for the scoring logic below, with whisper_enhance derived from it.
    mode = request.form.get('mode', 'ai')
    if mode not in ('ai', 'ai_stt'):
        mode = 'ai'
    mode_includes_stt = (mode == 'ai_stt')
    if mode_includes_stt:
        mode = 'ai'
    genre = request.form.get('genre', '').strip()
    scoring_mode = request.form.get('scoring_mode', 'generate')
    trailer_length = int(request.form.get('trailer_length', 15))
    if trailer_length not in (15, 30, 45, 60):
        trailer_length = 30
    try:
        max_scene_dur = float(request.form.get('max_scene_dur', '') or 0)
        max_scene_dur = max_scene_dur if max_scene_dur > 0 else None
    except ValueError:
        max_scene_dur = None
    try:
        scene_threshold = max(1.0, min(100.0, float(request.form.get('scene_threshold', 30.0))))
    except ValueError:
        scene_threshold = 30.0
    try:
        min_scene_len_sec = max(0.1, min(5.0, float(request.form.get('min_scene_len', 0.5))))
    except ValueError:
        min_scene_len_sec = 0.5
    transition = request.form.get('transition', 'fade')
    VALID_TRANSITIONS = {'fade','fadeblack','fadewhite','fadefast','fadegrays',
        'wipeleft','wiperight','wipeup','wipedown',
        'slideleft','slideright','slideup','slidedown',
        'smoothleft','smoothright','smoothup','smoothdown',
        'circlecrop','rectcrop','circleopen','circleclose',
        'distance','pixelize','diagtl','diagtr','diagbl','diagbr',
        'hlslice','hrslice','vuslice','vdslice',
        'radial','zoomin','dissolve','hblur','squeezev','squeezeh',
        'horzopen','horzclose','vertopen','vertclose','custom_matte'}
    transition_matte_path = None
    if genre in GENRE_PRESETS:
        preset = GENRE_PRESETS[genre]
        transition = preset['transition']
        xfade_dur = preset['xfade_dur']
        if scoring_mode not in ('upload', 'generate'):
            scoring_mode = 'generate'
    else:
        if transition not in VALID_TRANSITIONS:
            transition = 'fade'
        xfade_dur = float(request.form.get('xfade_dur', 0.3))
        xfade_dur = max(0.1, min(2.0, xfade_dur))
        if transition == 'custom_matte':
            if 'transition_matte' in request.files and request.files['transition_matte'].filename:
                f = request.files['transition_matte']
                fn = secure_filename(f.filename)
                if fn:
                    transition_matte_path = os.path.join(
                        app.config['UPLOAD_FOLDER'], f'transmatte_{int(time.time())}{os.path.splitext(fn)[1]}')
                    f.save(transition_matte_path)
            if not transition_matte_path:
                # Selected "Custom" but didn't upload anything — fall back rather
                # than fail the whole job over a missing optional asset.
                transition = 'fade'
    target_loudness = float(request.form.get('target_loudness', -14))
    true_peak = float(request.form.get('true_peak', -1.5))
    music_duck_db = float(request.form.get('music_duck_db', -3))
    beat_match = request.form.get('beat_match') == 'on'
    broadcast_stereo = request.form.get('broadcast_stereo') == 'on'
    model = request.form.get('model', 'qwen3-vl:8b')

    # SFX source selection: 'genre' (AI-generate/synth from the genre preset),
    # 'upload' (stamp a user-supplied one-shot at every cut), or 'none'.
    default_sfx_mode = 'genre' if (genre in GENRE_PRESETS and GENRE_PRESETS[genre].get('sfx')) else 'none'
    sfx_mode = request.form.get('sfx_mode', default_sfx_mode)
    if sfx_mode not in ('genre', 'upload', 'none'):
        sfx_mode = 'none'
    sfx_upload_path = None
    if sfx_mode == 'upload' and 'sfx_upload' in request.files and request.files['sfx_upload'].filename:
        f = request.files['sfx_upload']
        fn = secure_filename(f.filename)
        if fn:
            sfx_upload_path = os.path.join(app.config['UPLOAD_FOLDER'], f'sfxsrc_{int(time.time())}{os.path.splitext(fn)[1]}')
            f.save(sfx_upload_path)
    if sfx_mode == 'upload' and not sfx_upload_path:
        sfx_mode = 'none'  # nothing usable was uploaded, don't silently fall back to genre SFX


    # Voiceover: 'none' (skip), 'upload' (use a supplied VO track as-is), or
    # 'tts' (generate from typed text via a local TTS engine). Whichever
    # source, music/SFX/original audio get ducked underneath it, same as
    # scoring audio ducks under the original dialogue.
    vo_mode = request.form.get('vo_mode', 'none')
    if vo_mode not in ('none', 'upload', 'tts'):
        vo_mode = 'none'
    vo_upload_path = None
    if vo_mode == 'upload' and 'vo_upload' in request.files and request.files['vo_upload'].filename:
        f = request.files['vo_upload']
        fn = secure_filename(f.filename)
        if fn:
            vo_upload_path = os.path.join(app.config['UPLOAD_FOLDER'], f'vosrc_{int(time.time())}{os.path.splitext(fn)[1]}')
            f.save(vo_upload_path)
    if vo_mode == 'upload' and not vo_upload_path:
        vo_mode = 'none'
    vo_text = request.form.get('vo_text', '').strip()
    if vo_mode == 'tts' and not vo_text:
        vo_mode = 'none'
    vo_voice = request.form.get('vo_voice', '').strip() or None
    vo_language = request.form.get('vo_language', '').strip() or None
    vo_engine = request.form.get('vo_engine', 'fish_audio').strip()
    if vo_engine not in ('fish_audio', 'cozy_voice'):
        vo_engine = 'fish_audio'
    vo_ref_upload_path = None
    if vo_mode == 'tts' and 'vo_ref_upload' in request.files and request.files['vo_ref_upload'].filename:
        f = request.files['vo_ref_upload']
        fn = secure_filename(f.filename)
        if fn:
            vo_ref_upload_path = os.path.join(app.config['UPLOAD_FOLDER'], f'vorefsrc_{int(time.time())}{os.path.splitext(fn)[1]}')
            f.save(vo_ref_upload_path)
            vo_voice = None  # an uploaded reference clone takes priority over a picked registered voice
    try:
        vo_rate = int(request.form.get('vo_rate', 175))
    except ValueError:
        vo_rate = 175
    try:
        vo_start = max(0.0, float(request.form.get('vo_start', 0)))
    except ValueError:
        vo_start = 0.0
    try:
        vo_volume = max(0.3, min(3.0, float(request.form.get('vo_volume', 1.15))))
    except ValueError:
        vo_volume = 1.15
    # Trim points *within the uploaded VO file itself* (which portion of that
    # source clip to use) — distinct from vo_start above, which places the
    # (already-trimmed) narration on the trailer's own timeline.
    try:
        vo_trim_start = max(0.0, float(request.form.get('vo_trim_start', 0) or 0))
    except ValueError:
        vo_trim_start = 0.0
    vo_trim_end_raw = request.form.get('vo_trim_end', '').strip()
    try:
        vo_trim_end = float(vo_trim_end_raw) if vo_trim_end_raw else None
    except ValueError:
        vo_trim_end = None
    if vo_trim_end is not None and vo_trim_end <= vo_trim_start:
        vo_trim_end = None

    # Sync cuts to the beat of the background music (only meaningful when a
    # music track is actually used). Requires prepping the BGM before scene
    # selection instead of after, so cut points can be nudged onto beats.
    sync_beats = request.form.get('sync_beats') == 'on' and scoring_mode != 'none'
    whisper_enhance = mode_includes_stt

    end_card_path = None
    schedule_card_path = None
    scoring_audio_path = None
    if scoring_mode == 'upload' and 'scoring_audio' in request.files and request.files['scoring_audio'].filename:
        f = request.files['scoring_audio']
        fn = secure_filename(f.filename)
        if fn:
            scoring_audio_path = os.path.join(app.config['UPLOAD_FOLDER'], f'audio_{int(time.time())}{os.path.splitext(fn)[1]}')
            f.save(scoring_audio_path)
    if scoring_mode == 'generate':
        scoring_audio_path = 'GENERATE'  # flag to generate ambient
    if 'end_card_video' in request.files and request.files['end_card_video'].filename:
        f = request.files['end_card_video']
        if allowed_file(f.filename):
            end_card_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
            f.save(end_card_path)
    if 'schedule_video' in request.files and request.files['schedule_video'].filename:
        f = request.files['schedule_video']
        if allowed_file(f.filename):
            schedule_card_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
            f.save(schedule_card_path)

    # Optional VO tracks for the title card ("end_card_video" field, despite the
    # name) and end card ("schedule_video" field) — each can have its own
    # uploaded narration audio, muxed on in place of whatever audio the card
    # video already has, trimmed to a chosen [start, end) window of the source file.
    def _parse_card_vo(file_key, start_key, end_key):
        path = None
        if file_key in request.files and request.files[file_key].filename:
            f = request.files[file_key]
            fn = secure_filename(f.filename)
            if fn:
                path = os.path.join(app.config['UPLOAD_FOLDER'], f'cardvo_{int(time.time()*1000)}{os.path.splitext(fn)[1]}')
                f.save(path)
        try:
            start = max(0.0, float(request.form.get(start_key, 0) or 0))
        except ValueError:
            start = 0.0
        end_raw = request.form.get(end_key, '').strip()
        try:
            end = float(end_raw) if end_raw else None
        except ValueError:
            end = None
        if end is not None and end <= start:
            end = None
        return path, start, end

    title_card_vo_path, title_card_vo_start, title_card_vo_end = _parse_card_vo(
        'title_card_vo', 'title_card_vo_start', 'title_card_vo_end')
    end_card_vo_path, end_card_vo_start, end_card_vo_end = _parse_card_vo(
        'end_card_vo', 'end_card_vo_start', 'end_card_vo_end')

    prompt = request.form.get('prompt',
        'Describe this scene in 3-5 words, then rate it 1-5 for a movie trailer. Format: DESC: <words> | SCORE: <digit>')

    jid = job_new()
    with JOBS_LOCK:
        if jid in JOBS:
            JOBS[jid]['orig_name'] = orig_name
    params = dict(path=path, orig_name=orig_name, mode=mode, genre=genre, scoring_mode=scoring_mode,
                  trailer_length=trailer_length, max_scene_dur=max_scene_dur,
                  scene_threshold=scene_threshold, min_scene_len_sec=min_scene_len_sec,
                  transition=transition, xfade_dur=xfade_dur, transition_matte_path=transition_matte_path,
                  target_loudness=target_loudness, true_peak=true_peak, music_duck_db=music_duck_db, beat_match=beat_match, broadcast_stereo=broadcast_stereo, model=model,
                  sfx_mode=sfx_mode, sfx_upload_path=sfx_upload_path,
                  vo_mode=vo_mode, vo_upload_path=vo_upload_path, vo_text=vo_text, vo_voice=vo_voice,
                  vo_language=vo_language, vo_engine=vo_engine, vo_ref_upload_path=vo_ref_upload_path,
                  vo_rate=vo_rate, vo_start=vo_start, vo_volume=vo_volume, sync_beats=sync_beats, whisper_enhance=whisper_enhance,
                  vo_trim_start=vo_trim_start, vo_trim_end=vo_trim_end,
                  end_card_path=end_card_path, schedule_card_path=schedule_card_path,
                  title_card_vo_path=title_card_vo_path, title_card_vo_start=title_card_vo_start, title_card_vo_end=title_card_vo_end,
                  end_card_vo_path=end_card_vo_path, end_card_vo_start=end_card_vo_start, end_card_vo_end=end_card_vo_end,
                  scoring_audio_path=scoring_audio_path, prompt=prompt)
    threading.Thread(target=run_trailer_job_gated, args=(jid, params), daemon=True).start()
    return jsonify(job_id=jid)

@app.route('/api/trailer/progress/<job_id>')
def api_trailer_progress(job_id):
    j = job_get(job_id)
    if not j:
        return jsonify(error='Unknown job id'), 404
    j.pop('created', None)
    return jsonify(**j)

@app.route('/api/trailer/cancel/<job_id>', methods=['POST'])
def api_trailer_cancel(job_id):
    """Cancel a queued or in-flight trailer job. Queued jobs stop immediately;
    running jobs unwind at their next progress checkpoint (best-effort)."""
    ok = job_cancel(job_id)
    if not ok:
        j = job_get(job_id)
        if not j:
            return jsonify(error='Unknown job id'), 404
        return jsonify(error='Job already finished'), 409
    return jsonify(cancelled=True, job_id=job_id)

@app.route('/api/queue/status')
def api_queue_status():
    """Overall queue state: how many jobs are running vs the current concurrency
    limit, plus every currently-queued job with its wait position."""
    gate_status = GATE.status()
    with JOB_QUEUE_LOCK:
        queued = list(JOB_QUEUE)
    queued_info = []
    for i, jid in enumerate(queued):
        j = job_get(jid)
        queued_info.append({'job_id': jid, 'position': i, 'orig_name': (j or {}).get('orig_name')})
    return jsonify(running=gate_status['running'], limit=gate_status['limit'], queued=queued_info)

@app.route('/api/queue/limit', methods=['GET', 'POST'])
def api_queue_limit():
    """View or change how many trailer jobs are allowed to run at once. Changing
    this takes effect immediately — queued jobs re-check the limit as soon as a
    slot frees up or the limit itself changes."""
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        try:
            new_limit = int(data.get('limit', request.form.get('limit')))
        except (TypeError, ValueError):
            return jsonify(error='limit must be an integer'), 400
        if new_limit < 1:
            return jsonify(error='limit must be at least 1'), 400
        GATE.set_limit(new_limit)
    return jsonify(**GATE.status())

# ---- Health check: pings every external service this app depends on ----
# For each one we just check that *something* is listening and answers HTTP —
# we deliberately avoid hitting generation endpoints (Fish Audio /v1/tts,
# ACE-Step /release_task, Woosh /generate, Whisper /v1/audio/transcriptions)
# so a health check never costs GPU time or produces real output. Ollama is the
# only service with a cheap, purpose-built status endpoint (/api/tags), so that
# one's checked precisely; the rest are checked for bare reachability at their
# base URL, which is enough to tell "service is up" from "connection refused".
def _check_service(name, base_url, path='/', timeout=3):
    t0 = time.time()
    try:
        r = requests.get(base_url.rstrip('/') + path, timeout=timeout)
        latency_ms = round((time.time() - t0) * 1000)
        # Any HTTP response at all means something is listening and answering,
        # even a 404/405 for a path that server doesn't implement.
        return {'name': name, 'url': base_url, 'status': 'up',
                'http_status': r.status_code, 'latency_ms': latency_ms}
    except requests.exceptions.Timeout:
        return {'name': name, 'url': base_url, 'status': 'down', 'error': 'timeout'}
    except requests.exceptions.ConnectionError:
        return {'name': name, 'url': base_url, 'status': 'down', 'error': 'connection refused'}
    except Exception as e:
        return {'name': name, 'url': base_url, 'status': 'down', 'error': str(e)}

@app.route('/api/health')
def api_health():
    """Reachability check for every local model/media service the app talks to
    (Ollama, Fish Audio S2, Cozy Voice 3, faster-whisper, ACE-Step, Woosh), checked in parallel
    so one slow/dead service doesn't stall the others. Returns per-service status
    plus an overall ok flag."""
    checks = [
        ('ollama', OLLAMA_URL, '/api/tags'),
        ('fish_audio', FISH_AUDIO_URL.rsplit('/v1/', 1)[0] if '/v1/' in FISH_AUDIO_URL else FISH_AUDIO_URL, '/'),
        ('cozy_voice', COZY_VOICE_URL, '/'),
        ('whisper', WHISPER_URL, '/'),
        ('ace_step', ACE_STEP_URL, '/'),
        ('woosh', WOOSH_URL, '/'),
    ]
    results = {}
    threads = []
    def worker(name, url, path):
        results[name] = _check_service(name, url, path)
    for name, url, path in checks:
        th = threading.Thread(target=worker, args=(name, url, path))
        th.start()
        threads.append(th)
    for th in threads:
        th.join(timeout=5)
    ordered = [results.get(name, {'name': name, 'url': url, 'status': 'down', 'error': 'no response'})
               for name, url, path in checks]
    overall_ok = all(c['status'] == 'up' for c in ordered)
    return jsonify(ok=overall_ok, checked_at=time.time(), services=ordered)

@app.route('/api/voices')
def api_voices():
    """Lists narration voices and languages for the Narration dropdowns, for
    whichever engine is asked for via ?engine=fish_audio|cozy_voice (defaults
    to fish_audio). There's no bundled default voice — if the list comes back
    empty, the UI should fall back to "upload a reference sample" for
    zero-shot cloning."""
    force = request.args.get('refresh') == '1'
    engine = request.args.get('engine', 'fish_audio')
    if engine not in ('fish_audio', 'cozy_voice'):
        engine = 'fish_audio'
    voices, source, error = list_voices_for_engine(engine, force=force)
    return jsonify(ok=error is None, voices=voices, languages=FISH_AUDIO_LANGUAGES,
                    source=source, error=error, engine=engine)

@app.route('/api/vo/preview', methods=['POST'])
def api_vo_preview():
    """Generates a short narration clip so the script/voice/rate/language can be
    checked by ear before it's used in an actual trailer job. Runs synchronously
    (previews are short) and returns a URL to the generated WAV, served from the
    same /uploads/<filename> route as everything else in UPLOAD_FOLDER."""
    text = (request.form.get('text') or '').strip()
    if not text:
        return jsonify(ok=False, error='No narration text to preview'), 400
    if len(text) > 800:
        text = text[:800]  # previews are for checking voice/tone, not full scripts
    try:
        rate = int(request.form.get('rate', 175))
    except ValueError:
        rate = 175
    voice_id = (request.form.get('voice') or '').strip() or None
    language = (request.form.get('language') or '').strip() or None
    engine = (request.form.get('engine') or 'fish_audio').strip()
    if engine not in ('fish_audio', 'cozy_voice'):
        engine = 'fish_audio'

    ref_path = None
    if 'ref_upload' in request.files and request.files['ref_upload'].filename:
        f = request.files['ref_upload']
        fn = secure_filename(f.filename)
        if fn:
            ref_path = os.path.join(app.config['UPLOAD_FOLDER'], f'voprevref_{int(time.time()*1000)}{os.path.splitext(fn)[1]}')
            f.save(ref_path)
            voice_id = None  # an uploaded reference clone takes priority over a picked registered voice

    out_name = f'vopreview_{int(time.time()*1000)}.wav'
    out_path = os.path.join(app.config['UPLOAD_FOLDER'], out_name)
    ok, err = generate_tts(text, out_path, rate=rate, voice_id=voice_id,
                            reference_audio_path=ref_path, language=language, engine=engine)
    if not ok:
        return jsonify(ok=False, error=err or 'Preview generation failed'), 502
    return jsonify(ok=True, url=f'/uploads/{out_name}')

def run_trailer_job(jid, params):
    try:
        _run_trailer_job(jid, params)
    except JobCancelled:
        print(f'Trailer job {jid} cancelled')
        job_set(jid, error='Cancelled', status='cancelled')
    except Exception as e:
        print(f'Trailer job {jid} crashed: {e}')
        job_set(jid, error=f'Unexpected error: {e}')

def _run_trailer_job(jid, params):
    path = params['path']; orig_name = params['orig_name']; mode = params['mode']
    genre = params['genre']; scoring_mode = params['scoring_mode']; trailer_length = params['trailer_length']
    max_scene_dur = params.get('max_scene_dur')
    scene_threshold = params.get('scene_threshold', 30.0)
    min_scene_len_sec = params.get('min_scene_len_sec', 0.5)
    transition = params['transition']; xfade_dur = params['xfade_dur']
    transition_matte_path = params.get('transition_matte_path')
    target_loudness = params['target_loudness']; true_peak = params['true_peak']
    music_duck_db = params.get('music_duck_db', -3)
    broadcast_stereo = params.get('broadcast_stereo', False)
    beat_match = params['beat_match']; model = params['model']
    sfx_mode = params['sfx_mode']; sfx_upload_path = params['sfx_upload_path']
    vo_mode = params['vo_mode']; vo_upload_path = params['vo_upload_path']; vo_text = params['vo_text']
    vo_voice = params['vo_voice']; vo_rate = params['vo_rate']; vo_start = params['vo_start']
    vo_language = params.get('vo_language'); vo_ref_upload_path = params.get('vo_ref_upload_path')
    vo_engine = params.get('vo_engine', 'fish_audio')
    vo_volume = params.get('vo_volume', 1.15)
    vo_trim_start = params.get('vo_trim_start', 0.0); vo_trim_end = params.get('vo_trim_end')
    sync_beats = params['sync_beats']
    whisper_enhance = params.get('whisper_enhance', False)
    end_card_path = params['end_card_path']; schedule_card_path = params['schedule_card_path']
    title_card_vo_path = params.get('title_card_vo_path'); title_card_vo_start = params.get('title_card_vo_start', 0.0); title_card_vo_end = params.get('title_card_vo_end')
    end_card_vo_path = params.get('end_card_vo_path'); end_card_vo_start = params.get('end_card_vo_start', 0.0); end_card_vo_end = params.get('end_card_vo_end')
    scoring_audio_path = params['scoring_audio_path']; prompt = params['prompt']

    job_set(jid, percent=2, step='Reading video info')
    last_ffmpeg_stderr = None
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps if fps else 0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    src_fps = fps if fps and fps > 0 else 30
    cap.release()

    trailer_duration = trailer_length
    min_required = trailer_length * 1.5
    if video_duration < min_required:
        job_set(jid, error=f'Video is only {video_duration:.1f}s long, but a {trailer_length}s episodic promo plug requires at least {min_required:.1f}s of raw video. Upload a longer video or select a shorter length.')
        return

    # Measure card durations before selecting scenes
    card_files = []
    card_durations = []
    _card_vo_ts = int(time.time() * 1000)
    if end_card_path and os.path.exists(end_card_path):
        if title_card_vo_path and os.path.exists(title_card_vo_path):
            muxed = os.path.join(app.config['UPLOAD_FOLDER'], f'titlecard_vo_{_card_vo_ts}.mp4')
            result = mux_card_vo(end_card_path, title_card_vo_path, title_card_vo_start, title_card_vo_end, muxed)
            if result:
                end_card_path = result
        card_files.append(end_card_path)
    if schedule_card_path and os.path.exists(schedule_card_path):
        if end_card_vo_path and os.path.exists(end_card_vo_path):
            muxed = os.path.join(app.config['UPLOAD_FOLDER'], f'endcard_vo_{_card_vo_ts}.mp4')
            result = mux_card_vo(schedule_card_path, end_card_vo_path, end_card_vo_start, end_card_vo_end, muxed)
            if result:
                schedule_card_path = result
        card_files.append(schedule_card_path)
    for cf in card_files:
        r = subprocess.run([FFPROBE, '-v', 'error', '-show_entries', 'format=duration',
                            '-of', 'default=noprint_wrappers=1:nokey=1', cf],
                           capture_output=True, text=True)
        try:
            card_durations.append(float(r.stdout.strip()))
        except:
            card_durations.append(5)
    total_card_dur = sum(card_durations)

    # Scene target starts at trailer_length, minus cards duration
    base_target = max(5, trailer_length - total_card_dur)

    job_set(jid, percent=8, step='Detecting scene cuts')
    # Detect scenes via PySceneDetect. downscale=2 speeds up detection on large
    # source files (frames are only scaled down for the detector's own
    # analysis; returned timecodes are unaffected).
    scene_list = detect_scenes(path, threshold=scene_threshold,
                                min_scene_len_sec=min_scene_len_sec, downscale=2)
    if not scene_list:
        job_set(jid, error='No scene changes detected. Try a video with clear cuts, or lower the detection threshold.')
        return
    if len(scene_list) == 1 and (scene_list[0][1].get_seconds() - scene_list[0][0].get_seconds()) > video_duration * 0.95:
        # PySceneDetect's own fallback: no real cuts found, so it returned one
        # scene spanning the whole video. Selecting from a single "scene" isn't
        # meaningful — surface this clearly instead of silently treating the
        # entire source as one giant clip.
        job_set(jid, error='No distinct scene cuts were found — PySceneDetect sees this video as one continuous shot. Try lowering the detection threshold or upload footage with visible cuts.')
        return

    job_set(jid, percent=15, step=f'Scoring {len(scene_list)} scenes (sharpness/brightness)')
    # Score scenes
    from statistics import median
    scenes_data = []
    cap = cv2.VideoCapture(path)
    for start, end in scene_list:
        mid_f = int((start.get_frames() + end.get_frames()) / 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_f)
        ret, frame = cap.read()
        if not ret:
            continue
        dur = end.get_seconds() - start.get_seconds()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F).var()
        bri = float(np.mean(gray))
        h, w = gray.shape
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = float(np.count_nonzero(edges)) / (h * w)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mean_hue = float(np.mean(hsv[:,:,0]))
        mean_sat = float(np.mean(hsv[:,:,1]))
        mean_val = float(np.mean(hsv[:,:,2]))
        has_face = False
        if ONNX_PATH is not None:
            _, faces = get_fd(w, h).detect(frame)
            has_face = faces is not None and len(faces) > 0
        scenes_data.append({
            'start': start.get_seconds(), 'end': end.get_seconds(),
            'start_f': start.get_frames(), 'end_f': end.get_frames(),
            'duration': dur, 'laplacian': round(lap, 2), 'brightness': round(bri, 1),
            'edge_ratio': round(edge_ratio, 3), 'mean_hue': round(mean_hue, 1),
            'mean_sat': round(mean_sat, 1), 'mean_val': round(mean_val, 1),
            'has_face': has_face, 'frame': frame, 'frame_idx': mid_f,
        })
    cap.release()
    if not scenes_data:
        job_set(jid, error='No frames could be read.')
        return

    med_lap = median([s['laplacian'] for s in scenes_data])
    med_bri = median([s['brightness'] for s in scenes_data])
    for s in scenes_data:
        score = 0
        if s['laplacian'] > med_lap * 1.2: score += 2
        elif s['laplacian'] > med_lap * 0.8: score += 1
        if 80 < s['brightness'] < 180: score += 2
        elif s['brightness'] > 30: score += 1
        if 1 < s['duration'] < 8: score += 2
        elif s['duration'] > 8: score += 1
        elif s['duration'] < 0.7:
            # Sub-fragment scenes (whip-pans, flash cuts) can still win on raw
            # sharpness/brightness alone with no duration credit at all —
            # penalize them explicitly so they don't out-rank a real scene of
            # similar visual quality and end up as a flicker cut in the output.
            score -= 1
        if s['has_face']:
            # A face/reaction shot is generally more useful in a trailer than
            # empty B-roll of similar sharpness/brightness.
            score += 1
        s['quality_score'] = score

    if mode == 'ai':
        n_scenes_ai = len(scenes_data)
        for ai_i, s in enumerate(scenes_data):
            job_set(jid, percent=18 + int(12 * ai_i / max(n_scenes_ai, 1)),
                    step=f'AI-scoring scene {ai_i+1}/{n_scenes_ai}')
            _, buf = cv2.imencode('.jpg', s['frame'], [cv2.IMWRITE_JPEG_QUALITY, 85])
            b64 = base64.b64encode(buf.tobytes()).decode()
            try:
                r = requests.post(f'{OLLAMA_URL}/api/generate', json={
                    'model': model, 'prompt': prompt, 'stream': False, 'images': [b64]
                }, timeout=120)
                txt = r.json().get('response', '')
                import re
                desc_m = re.search(r'DESC:\s*(.+?)(?:\s*\||$)', txt)
                score_m = re.search(r'SCORE:\s*([1-5])', txt)
                if not score_m:
                    print(f'AI vision score parse failed for scene {ai_i+1}, defaulting to 3. Raw response: {txt[:300]!r}')
                s['ai_desc'] = desc_m.group(1).strip() if desc_m else ''
                s['total_score'] = s['quality_score'] + (int(score_m.group(1)) if score_m else 3)
            except Exception as e:
                print(f'AI vision request failed for scene {ai_i+1}, defaulting to 3: {e}')
                s['total_score'] = s['quality_score'] + 3
    else:
        for s in scenes_data:
            s['total_score'] = s['quality_score']

    # Dialogue transcription (faster-whisper) — improves scene selection three ways:
    # 1. Scenes with actual quotable dialogue get a small scoring boost, so
    #    selection isn't purely based on visual sharpness/brightness/AI framing.
    # 2. Word-level timestamps let cut in/out points snap to word boundaries
    #    later, instead of landing mid-word.
    # 3. The single best-scoring line across the video is surfaced as a
    #    suggested hook line/pull-quote (informational only — not auto-inserted
    #    anywhere).
    word_starts, word_ends = [], []
    hook_line = None
    if whisper_enhance:
        job_set(jid, percent=22, step='Transcribing dialogue (faster-whisper)')
        words, segments = transcribe_video(path)
        if words or segments:
            word_starts = [w['start'] for w in words]
            word_ends = [w['end'] for w in words]
            for s in scenes_data:
                overlap_text = ' '.join(
                    sg['text'] for sg in segments if sg['start'] < s['end'] and sg['end'] > s['start']
                ).strip()
                s['dialogue'] = overlap_text
                if overlap_text:
                    bonus = 1
                    if '?' in overlap_text or '!' in overlap_text:
                        bonus += 1
                    s['total_score'] += bonus
            if segments:
                best = max(segments, key=lambda sg: score_hook_line(sg['text']))
                if score_hook_line(best['text']) > 0:
                    hook_line = {'text': best['text'], 'start': round(best['start'], 1), 'end': round(best['end'], 1)}
        else:
            whisper_enhance = False  # transcription unavailable/failed — skip the snapping logic below too

    # "Edit to music": prep the BGM *before* picking scenes so cut points can be
    # snapped onto its beat grid. Only worth the extra generation pass when the
    # user actually asked for it — otherwise BGM is prepared later as before.
    base_ts = int(time.time())
    early_bgm_path = None
    early_bgm_source = 'none'
    beat_times = []
    if sync_beats:
        job_set(jid, percent=20, step='Preparing music for beat-synced cuts')
        early_bgm_path, early_bgm_source = prepare_bgm_track(genre, scoring_mode, scoring_audio_path,
                                                              base_target, base_ts)
        if early_bgm_path:
            beat_times = detect_beat_times(early_bgm_path, base_target)
            if not beat_times:
                sync_beats = False  # detection failed (e.g. librosa missing) — fall back silently
        else:
            sync_beats = False

    job_set(jid, percent=28, step='Selecting best scenes')
    # Pick top scenes by score to fill target, then sort by timecode
    # Iterative: xfade transitions shorten output, so compensate
    # Floor for how short a *budget-truncated* clip is allowed to be. Without this,
    # whichever scene happens to land last (in score order, not timeline order) just
    # gets clipped to "whatever duration is left" — which can be a fraction of a
    # second. That sliver then lands wherever its timecode falls once we re-sort by
    # start time, often producing a jarring near-invisible cut right before the end.
    # A scene's own *natural* PySceneDetect duration can still be shorter than this
    # (that's a legitimate quick cut in the source) — this floor only stops us from
    # truncating a longer scene down below it just to hit the target duration exactly.
    min_seg_dur = max(0.8, xfade_dur * 2.5)
    if max_scene_dur:
        min_seg_dur = min(min_seg_dur, max_scene_dur)
    # Minimum spacing (in source timeline seconds) required between two
    # selected scenes' start points. Without this, greedy score-based
    # selection can pick several scenes from the same high-scoring stretch of
    # the video (e.g. one well-lit dialogue scene) and leave the rest of the
    # source entirely unrepresented. Scaled to the source length but bounded
    # so it's meaningful on both short and long videos.
    base_min_gap = max(2.0, min(8.0, video_duration * 0.03))
    trailer_duration = base_target
    for pass_attempt in range(4):
        # Relax the gap requirement on later passes: if spacing is preventing
        # us from filling the duration budget, it's better to allow some
        # clustering than to ship a trailer that's noticeably short.
        min_gap = max(1.0, base_min_gap - pass_attempt * (base_min_gap / 4))
        scenes_data.sort(key=lambda x: x['total_score'], reverse=True)
        selected = []
        total_sel = 0
        for s in scenes_data:
            remaining = trailer_duration - total_sel
            if remaining < min_seg_dur:
                # Not enough budget left for a decent-length clip — stop selecting
                # rather than truncating the next scene into a sliver. The
                # shortfall gets absorbed by nudging trailer_duration up on the
                # next pass_attempt below.
                break
            if any(abs(s['start'] - c['start']) < min_gap for c in selected):
                # Too close in the source timeline to an already-selected scene
                # — skip it in favor of spreading selections across the video,
                # rather than over-sampling one stretch of it.
                continue
            seg_dur = min(s['duration'], remaining)
            if max_scene_dur:
                seg_dur = min(seg_dur, max_scene_dur)
            seg_start = s['start']
            if whisper_enhance and word_starts:
                # Don't start playback mid-word — nudge the in-point forward to
                # the start of the nearest word within this scene (capped so we
                # never drift far from the original visual cut point).
                snapped_start = nearest_word_boundary(seg_start, word_starts, max_snap=0.35)
                if seg_start < snapped_start < seg_start + seg_dur:
                    drift = snapped_start - seg_start
                    seg_start = snapped_start
                    seg_dur = max(0.3, seg_dur - drift)
            scene_end = s['start'] + s['duration']
            if sync_beats and beat_times:
                # Nudge this segment's end so the *cumulative* cut point lands on
                # the nearest beat, within what this scene can actually supply.
                target_cut = total_sel + seg_dur
                snapped_cut = nearest_beat(target_cut, beat_times, total_sel + 0.3, total_sel + (scene_end - seg_start))
                seg_dur = max(0.3, min(scene_end - seg_start, snapped_cut - total_sel))
            elif whisper_enhance and word_ends and seg_dur < (scene_end - seg_start):
                # This clip is being truncated to fit the duration budget — snap
                # the actual out-point to the end of the nearest word so we don't
                # cut off mid-word.
                target_end = seg_start + seg_dur
                snapped_end = nearest_word_boundary(target_end, word_ends, max_snap=0.35)
                if seg_start < snapped_end <= scene_end:
                    seg_dur = max(0.3, snapped_end - seg_start)
            s['trim_start'] = seg_start
            s['selected_dur'] = seg_dur
            selected.append(s)
            total_sel += seg_dur
        selected.sort(key=lambda x: x['start'])

        n_seg = len(selected) + len(card_files)
        xfade_loss = max(0, (n_seg - 1)) * xfade_dur
        expected_total = total_sel + total_card_dur - xfade_loss
        shortfall = trailer_length - expected_total
        if abs(shortfall) <= 0.5 or pass_attempt == 3:
            break
        trailer_duration = total_sel + shortfall * 1.15

    if not selected:
        job_set(jid, error='No scenes selected.')
        return

    job_set(jid, percent=38, step=f'Extracting {len(selected)} selected clips')
    # Extract selected segments + card videos as temp files. `extracted`
    # tracks which of `selected` actually produced a usable clip, so stats
    # reported to the user (selected_scenes, trailer_duration) reflect what's
    # really in the output rather than what was merely picked.
    seg_files = []
    extracted = []
    for seg_i, seg in enumerate(selected):
        out_seg = os.path.join(app.config['UPLOAD_FOLDER'], f'seg_{base_ts}_{seg_i}.mp4')
        trim_start = seg.get('trim_start', seg['start'])
        cmd = [FFMPEG, '-y', '-ss', str(trim_start), '-i', path,
               '-t', str(seg['selected_dur']),
               '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '28', '-pix_fmt', 'yuv420p',
               '-c:a', 'aac', '-b:a', '128k', out_seg]
        r = subprocess.run(cmd, capture_output=True, text=True)
        ok = os.path.exists(out_seg) and os.path.getsize(out_seg) > 0
        if not ok:
            print(f'FFMPEG seg extraction error (scene at {trim_start}s): {r.stderr[:500]}')
            # Retry once with -ss placed after -i: slower (full decode from
            # the start) but more robust for seek points that land in an
            # awkward spot relative to keyframes/container index — worth one
            # extra attempt before dropping the clip outright.
            r2 = subprocess.run([FFMPEG, '-y', '-i', path, '-ss', str(trim_start),
                                  '-t', str(seg['selected_dur']),
                                  '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '28', '-pix_fmt', 'yuv420p',
                                  '-c:a', 'aac', '-b:a', '128k', out_seg], capture_output=True, text=True)
            ok = os.path.exists(out_seg) and os.path.getsize(out_seg) > 0
            if not ok:
                print(f'FFMPEG seg extraction retry also failed (scene at {trim_start}s): {r2.stderr[:500]}')
                last_ffmpeg_stderr = r2.stderr[-800:]
        if ok:
            seg_files.append(out_seg)
            extracted.append(seg)

    if len(extracted) < len(selected):
        dropped = len(selected) - len(extracted)
        print(f'{dropped} selected scene(s) failed extraction and were dropped from the trailer.')
    selected = extracted
    total_sel = sum(s['selected_dur'] for s in selected)

    if not selected:
        job_set(jid, error='All selected scenes failed to extract.' + (f' Last ffmpeg error: {last_ffmpeg_stderr}' if last_ffmpeg_stderr else ''))
        return

    all_inputs = seg_files + card_files
    n_total = len(all_inputs)

    out_path = os.path.join(app.config['UPLOAD_FOLDER'], f'trailer_{base_ts}.mp4')
    sfx_timestamps = []

    norm = (f'scale={src_w}:{src_h}:force_original_aspect_ratio=decrease,'
            f'pad={src_w}:{src_h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={src_fps}')

    job_set(jid, percent=50, step='Building transitions & crossfades')
    if n_total == 1:
        r = subprocess.run([FFMPEG, '-y', '-i', all_inputs[0], '-vf', norm,
                            '-c:v', 'libx264', '-preset', 'fast', '-crf', '22',
                            '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '128k', out_path],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f'FFMPEG single concat error: {r.stderr[:500]}')
            last_ffmpeg_stderr = r.stderr[-800:]
    else:
        # Get durations for xfade offset calculation
        durations = []
        for f in all_inputs:
            r = subprocess.run([FFPROBE, '-v', 'error', '-show_entries', 'format=duration',
                                '-of', 'default=noprint_wrappers=1:nokey=1', f],
                               capture_output=True, text=True)
            try:
                durations.append(float(r.stdout.strip()))
            except:
                durations.append(5)

        # Normalize every input to ensure consistent video/audio before xfade
        # Only re-encode if audio is missing (add silent audio as fallback)
        normed_inputs = []
        for i, inp in enumerate(all_inputs):
            check = subprocess.run([FFPROBE, '-v', 'error', '-select_streams', 'a',
                                    '-show_entries', 'stream=index', '-of', 'csv=p=0', inp],
                                   capture_output=True, text=True, timeout=10)
            has_audio = bool(check.stdout.strip())
            if has_audio:
                normed_inputs.append(inp)
            else:
                normed = os.path.join(app.config['UPLOAD_FOLDER'], f'norm_{base_ts}_{i}.mp4')
                r = subprocess.run([FFMPEG, '-y', '-i', inp,
                                    '-f', 'lavfi', '-i', 'anullsrc=channel_layout=stereo:sample_rate=44100',
                                    '-c:v', 'copy', '-c:a', 'aac', '-b:a', '128k',
                                    '-map', '0:v:0', '-map', '1:a:0', '-shortest',
                                    normed], capture_output=True, text=True, timeout=60)
                if os.path.exists(normed) and os.path.getsize(normed) > 0:
                    normed_inputs.append(normed)
                else:
                    normed_inputs.append(inp)

        # Re-measure durations after normalization
        durations = []
        for f in normed_inputs:
            d = subprocess.run([FFPROBE, '-v', 'error', '-show_entries', 'format=duration',
                                '-of', 'default=noprint_wrappers=1:nokey=1', f],
                               capture_output=True, text=True)
            try:
                durations.append(float(d.stdout.strip()))
            except:
                durations.append(5)

        all_inputs = normed_inputs
        n_total = len(all_inputs)

        norm = (f'scale={src_w}:{src_h}:force_original_aspect_ratio=decrease,'
                f'pad={src_w}:{src_h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={src_fps}')
        filter_parts = [f'[{i}:v]{norm}[n{i}]' for i in range(n_total)]
        use_matte = transition == 'custom_matte' and transition_matte_path and os.path.exists(transition_matte_path)
        matte_input_args = []

        if use_matte:
            # Custom transition: blend each cut using a user-uploaded matte's luma as
            # an opacity mask (maskedmerge) instead of one of ffmpeg's built-in xfade
            # wipe shapes. xfade computes its own overlap windows internally from a
            # single offset per cut; maskedmerge has no such concept, so each clip is
            # split by hand into a unique middle portion plus the tail/head slivers
            # (xfade_dur long) that feed the transition either side of it, and the
            # whole thing is reassembled with one concat filter at the end.
            matte_ext = os.path.splitext(transition_matte_path)[1].lower()
            is_image = matte_ext in {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
            matte_input_idx = n_total  # the matte is appended as one extra -i after all_inputs
            matte_input_args = (['-loop', '1', '-i', transition_matte_path] if is_image
                                 else ['-stream_loop', '-1', '-i', transition_matte_path])
            filter_parts.append(
                f'[{matte_input_idx}:v]{norm},format=gray,trim=0:{xfade_dur},setpts=PTS-STARTPTS[mattebase]')
            n_trans = n_total - 1
            filter_parts.append('[mattebase]split=' + str(n_trans) + ''.join(f'[mt{i}]' for i in range(n_trans)))

            seg_labels = []
            for i in range(n_total):
                start = xfade_dur if i > 0 else 0
                end = durations[i] - (xfade_dur if i < n_total - 1 else 0)
                if end <= start:
                    # Clip too short to give a full xfade_dur to both neighboring
                    # transitions — keep a thin sliver instead of an empty/negative one.
                    end = start + 0.05
                filter_parts.append(f'[n{i}]trim=start={start}:end={end},setpts=PTS-STARTPTS[seg{i}]')
                seg_labels.append(f'seg{i}')
                if i < n_total - 1:
                    tail_start = max(0, durations[i] - xfade_dur)
                    filter_parts.append(f'[n{i}]trim=start={tail_start}:end={durations[i]},setpts=PTS-STARTPTS[tailA{i}]')
                    filter_parts.append(f'[n{i+1}]trim=start=0:end={xfade_dur},setpts=PTS-STARTPTS[headB{i}]')
                    filter_parts.append(f'[tailA{i}][headB{i}][mt{i}]maskedmerge[trans{i}]')
                    seg_labels.append(f'trans{i}')
                    offset = sum(durations[:i + 1]) - (i + 1) * xfade_dur
                    sfx_timestamps.append(max(offset, 0) + xfade_dur * 0.5)
            concat_inputs = ''.join(f'[{lbl}]' for lbl in seg_labels)
            filter_parts.append(f'{concat_inputs}concat=n={len(seg_labels)}:v=1:a=0[vout]')
            prev_label = 'vout'
        else:
            prev_label = 'n0'
            for i in range(n_total - 1):
                offset = sum(durations[:i + 1]) - (i + 1) * xfade_dur
                sfx_timestamps.append(max(offset, 0) + xfade_dur * 0.5)
                out_label = f'v{i+1}'
                filter_parts.append(
                    f'[{prev_label}][n{i+1}]xfade=transition={transition}:duration={xfade_dur}:offset={max(offset, 0)}[{out_label}]')
                prev_label = out_label

        # Audio acrossfade chain — matches video transition timing exactly either way
        audio_parts = []
        for i in range(n_total):
            audio_parts.append(f'[{i}:a]atrim=0:{durations[i]}[a{i}]')
        for i in range(1, n_total):
            prev = f'af{i-1}' if i > 1 else 'a0'
            audio_parts.append(f'[{prev}][a{i}]acrossfade=d={xfade_dur}:c1=tri[af{i}]')
        last_audio_label = f'af{n_total-1}'
        filter_parts.extend(audio_parts)

        cmd = [FFMPEG, '-y']
        for f in all_inputs:
            cmd.extend(['-i', f])
        cmd.extend(matte_input_args)
        cmd.extend(['-filter_complex', ';'.join(filter_parts)])
        last_vlabel = f'[{prev_label}]'
        cmd.extend(['-map', last_vlabel, '-map', f'[{last_audio_label}]'])
        cmd.extend(['-c:v', 'libx264', '-preset', 'fast', '-crf', '22', '-pix_fmt', 'yuv420p',
                     '-c:a', 'aac', '-b:a', '128k', out_path])
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            print(f'FFMPEG xfade error: {r.stderr[:1000]}')
            last_ffmpeg_stderr = r.stderr[-800:]

    # Cleanup segment files
    for f in seg_files:
        if os.path.exists(f):
            os.remove(f)

    filename = os.path.basename(out_path)
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        detail = ''
        if not seg_files:
            detail = ' No scene clips were successfully extracted from the source video — check that ffmpeg/ffprobe are installed and on PATH, and that the uploaded video isn\'t corrupt.'
        elif last_ffmpeg_stderr:
            detail = f' Last ffmpeg error: {last_ffmpeg_stderr.strip()}'
        job_set(jid, error=f'Episodic Promo Plug generation failed (ffmpeg output empty).{detail}')
        return
    # Verify it's a valid video
    v = subprocess.run([FFPROBE, '-v', 'error', '-show_entries', 'format=duration',
                        '-of', 'default=noprint_wrappers=1:nokey=1', out_path],
                       capture_output=True, text=True)
    if v.returncode != 0 or float(v.stdout.strip() or 0) <= 0:
        job_set(jid, error='Episodic Promo Plug generation failed (invalid/corrupt output).')
        return

    job_set(jid, percent=58, step='Normalizing audio levels')
    # Normalize the raw SOT (original dialogue/nat sound baked into the source clips)
    # to a consistent loudness up front. Source footage can come in recorded at very
    # different levels — this keeps everything downstream (SFX ducking, BGM
    # sidechain detection, VO ducking) working off a predictable baseline instead of
    # being thrown off by an unusually quiet or hot original recording.
    sot_norm = os.path.join(app.config['UPLOAD_FOLDER'], f'sotnorm_{base_ts}.mp4')
    r = subprocess.run([FFMPEG, '-y', '-i', out_path,
                        '-af', f'loudnorm=I={target_loudness}:TP={true_peak}:LRA=7',
                        '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', sot_norm],
                       capture_output=True, text=True, timeout=120)
    if os.path.exists(sot_norm) and os.path.getsize(sot_norm) > 0:
        os.replace(sot_norm, out_path)
    else:
        print(f'SOT normalization error: {r.stderr[:500]}')

    job_set(jid, percent=62, step='Adding sound effects at cuts')
    # Generate SFX and mix into trailer audio. Whatever the source (AI-generated,
    # uploaded, or synthesized), the *same* hit gets stamped at every cut via
    # stamp_hits() so it's never just a single one-shot lost at the front.
    sfx_ok = False
    sfx_source = 'none'  # 'woosh' | 'ai_generated' | 'uploaded' | 'synth_fallback' | 'none'
    sfx_path = os.path.join(app.config['UPLOAD_FOLDER'], f'sfx_{base_ts}.wav')
    if sfx_mode != 'none' and sfx_timestamps:
        hit_wave = None
        if sfx_mode == 'upload' and sfx_upload_path:
            hit_wave = load_hit_waveform(sfx_upload_path)
            sfx_source = 'uploaded' if hit_wave is not None else 'none'
        elif sfx_mode == 'genre' and genre:
            woosh_sfx_path = os.path.join(app.config['UPLOAD_FOLDER'], f'woosh_sfx_{base_ts}.wav')
            if woosh_sfx(genre, woosh_sfx_path, duration=0.8) and os.path.getsize(woosh_sfx_path) > 0:
                hit_wave = load_hit_waveform(woosh_sfx_path)
                if os.path.exists(woosh_sfx_path):
                    os.remove(woosh_sfx_path)
                sfx_source = 'woosh' if hit_wave is not None else 'none'
            if hit_wave is None:
                acestep_sfx_path = os.path.join(app.config['UPLOAD_FOLDER'], f'acestep_sfx_{base_ts}.wav')
                if acestep_sfx(genre, acestep_sfx_path, duration=0.8) and os.path.getsize(acestep_sfx_path) > 0:
                    hit_wave = load_hit_waveform(acestep_sfx_path)
                    if os.path.exists(acestep_sfx_path):
                        os.remove(acestep_sfx_path)
                    sfx_source = 'ai_generated' if hit_wave is not None else 'none'
            if hit_wave is None:
                hit_wave = synth_sfx_waveform(genre)
                sfx_source = 'synth_fallback' if hit_wave is not None else 'none'
        if hit_wave is not None:
            sfx_ok = stamp_hits(hit_wave, sfx_timestamps, sfx_path)
            if not sfx_ok:
                sfx_source = 'none'
        if sfx_ok:
            sfx_m4a = os.path.join(app.config['UPLOAD_FOLDER'], f'sfx_{base_ts}.m4a')
            sfx_cmd = [FFMPEG, '-y', '-i', sfx_path]
            if sfx_source == 'synth_fallback':
                # Light production polish so the procedurally-synthesized fallback
                # doesn't sound as bare/synthetic next to AI-generated SFX.
                sfx_cmd += ['-af', 'aecho=0.6:0.5:35:0.25,alimiter=limit=0.9']
            sfx_cmd += ['-c:a', 'aac', '-b:a', '192k', sfx_m4a]
            subprocess.run(sfx_cmd, capture_output=True, text=True, timeout=30)
            if os.path.exists(sfx_m4a) and os.path.getsize(sfx_m4a) > 0:
                # Mix SFX into the trailer audio
                with_sfx = os.path.join(app.config['UPLOAD_FOLDER'], f'with_sfx_{base_ts}.mp4')
                r = subprocess.run([FFMPEG, '-y', '-i', out_path, '-i', sfx_m4a,
                                    '-filter_complex',
                                    '[0:a]volume=1.0[a0];[1:a]volume=0.85[a1];'
                                    '[a0][a1]amix=inputs=2:duration=first:dropout_transition=2:normalize=0[outa]',
                                    '-map', '0:v', '-map', '[outa]',
                                    '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
                                    '-shortest', with_sfx],
                                   capture_output=True, text=True, timeout=120)
                if os.path.exists(with_sfx) and os.path.getsize(with_sfx) > 0:
                    os.replace(with_sfx, out_path)
            if os.path.exists(sfx_path):
                os.remove(sfx_path)
            if os.path.exists(sfx_m4a):
                os.remove(sfx_m4a)
    if sfx_upload_path and os.path.exists(sfx_upload_path):
        os.remove(sfx_upload_path)
    if title_card_vo_path and os.path.exists(title_card_vo_path):
        os.remove(title_card_vo_path)
    if end_card_vo_path and os.path.exists(end_card_vo_path):
        os.remove(end_card_vo_path)

    job_set(jid, percent=80, step='Generating/mixing background music')
    # Prepare background music (ducked under SOT) as its own stem — the actual
    # merge into out_path happens later in the unified final mix below, once we
    # also know whether a voiceover is being added.
    bgm_source = 'none'  # 'uploaded' | 'ai_generated' | 'synth_fallback' | 'none'
    bgm_ready_path = None
    if scoring_audio_path and total_sel > 0:
        scenes_dur = total_sel
        prepared_bgm = None

        if early_bgm_path and os.path.exists(early_bgm_path):
            # Reuse the track generated for beat-sync instead of paying for a
            # second ACE-Step/generation pass; just re-fit it to the exact length.
            prepared_bgm = finalize_bgm_duration(early_bgm_path, scenes_dur, base_ts)
            bgm_source = early_bgm_source if prepared_bgm else 'none'
            if os.path.exists(early_bgm_path):
                os.remove(early_bgm_path)
        else:
            prepared_bgm, bgm_source = prepare_bgm_track(genre, scoring_mode, scoring_audio_path, scenes_dur, base_ts)

        if prepared_bgm and beat_match:
            job_set(jid, step='Beat-matching music to video tempo')
            matched = os.path.join(app.config['UPLOAD_FOLDER'], f'matched_{base_ts}.wav')
            if beat_match_audio(path, prepared_bgm, scenes_dur, matched):
                prepared_bgm_m4a = os.path.join(app.config['UPLOAD_FOLDER'], f'matched_{base_ts}.m4a')
                subprocess.run([FFMPEG, '-y', '-i', matched,
                                '-c:a', 'aac', '-b:a', '192k', prepared_bgm_m4a],
                               capture_output=True, text=True, timeout=30)
                if os.path.exists(prepared_bgm_m4a) and os.path.getsize(prepared_bgm_m4a) > 0:
                    prepared_bgm = prepared_bgm_m4a
                if os.path.exists(matched):
                    os.remove(matched)

        if prepared_bgm:
            # Normalize the BGM track's own loudness first — an uploaded/generated
            # music track can come in much hotter or quieter than the SOT, and a
            # flat volume multiplier alone won't correct for that. Normalize to
            # music_duck_db under overall loudness, *then* duck it under SOT on top.
            bgm_target = target_loudness + music_duck_db
            bgm_ready_path = os.path.join(app.config['UPLOAD_FOLDER'], f'bgmready_{base_ts}.m4a')
            r = subprocess.run([FFMPEG, '-y', '-i', out_path, '-i', prepared_bgm,
                                '-filter_complex',
                                f'[1:a]loudnorm=I={bgm_target}:TP={true_peak}:LRA=7[bgmnorm];'
                                '[bgmnorm][0:a]sidechaincompress=threshold=0.02:ratio=12:attack=5:release=350:makeup=1[bgm_ducked]',
                                '-map', '[bgm_ducked]', '-c:a', 'aac', '-b:a', '192k', bgm_ready_path],
                               capture_output=True, text=True, timeout=120)
            if not (os.path.exists(bgm_ready_path) and os.path.getsize(bgm_ready_path) > 0):
                print(f'BGM prep error: {r.stderr[:500]}')
                bgm_ready_path = None
            if os.path.exists(prepared_bgm):
                os.remove(prepared_bgm)

    # Voiceover: upload or TTS. Prepared as its own stem here too — mixed in below,
    # in the same unified final mix, so it can duck SOT and BGM by different amounts.
    vo_source = 'none'  # 'uploaded' | 'tts' | 'none'
    vo_error = None
    vo_ready_path = None
    if vo_mode != 'none':
        if vo_mode == 'tts':
            engine_label = 'Cozy Voice 3' if vo_engine == 'cozy_voice' else 'Fish Audio'
            if vo_ref_upload_path:
                vo_step_note = f' (cloning voice from uploaded reference sample via {engine_label})'
            elif vo_voice:
                vo_step_note = f' (voice: {vo_voice}, {engine_label})'
            else:
                vo_step_note = f' ({engine_label}, no voice selected)'
        else:
            vo_step_note = ''
        job_set(jid, percent=90, step='Adding narration' + vo_step_note)
        vo_raw_path = None
        if vo_mode == 'upload' and vo_upload_path and os.path.exists(vo_upload_path):
            vo_raw_path = vo_upload_path
            vo_source = 'uploaded'
        elif vo_mode == 'tts':
            tts_wav = os.path.join(app.config['UPLOAD_FOLDER'], f'tts_{base_ts}.wav')
            tts_kwargs = {'rate': vo_rate, 'voice_id': vo_voice, 'language': vo_language, 'engine': vo_engine}
            if vo_ref_upload_path and os.path.exists(vo_ref_upload_path):
                tts_kwargs['reference_audio_path'] = vo_ref_upload_path
            ok, err = generate_tts(vo_text, tts_wav, **tts_kwargs)
            if ok:
                vo_raw_path = tts_wav
                vo_source = 'tts'
            else:
                vo_error = err
                print(f'TTS error: {err}')

        if vo_raw_path:
            # Normalize the VO's own loudness first — an uploaded voiceover can be
            # recorded much quieter or hotter than expected, which would otherwise
            # throw off both how audible it ends up and how reliably it triggers
            # the ducking below. If this is an uploaded VO, trim it to
            # [vo_trim_start, vo_trim_end) first — that's a window within the
            # source file itself, separate from vo_start below (which places the
            # already-trimmed narration on the trailer's own timeline).
            ms = max(0, int(vo_start * 1000))
            vo_ready_path = os.path.join(app.config['UPLOAD_FOLDER'], f'voready_{base_ts}.m4a')
            cmd = [FFMPEG, '-y']
            if vo_source == 'uploaded' and (vo_trim_start > 0 or vo_trim_end is not None):
                cmd.extend(['-ss', str(vo_trim_start)])
                if vo_trim_end is not None:
                    cmd.extend(['-to', str(vo_trim_end)])
            cmd.extend(['-i', vo_raw_path,
                        '-af', f'loudnorm=I={target_loudness}:TP={true_peak}:LRA=7,adelay={ms}|{ms},volume={vo_volume}',
                        '-c:a', 'aac', '-b:a', '192k', vo_ready_path])
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if not (os.path.exists(vo_ready_path) and os.path.getsize(vo_ready_path) > 0):
                print(f'VO prep error: {r.stderr[:500]}')
                vo_ready_path = None
        if vo_upload_path and os.path.exists(vo_upload_path):
            os.remove(vo_upload_path)
        tts_wav = os.path.join(app.config['UPLOAD_FOLDER'], f'tts_{base_ts}.wav')
        if os.path.exists(tts_wav):
            os.remove(tts_wav)

    # Unified final mix: combine SOT (already in out_path) with whichever of
    # BGM/VO are present. If VO is playing, SOT gets ducked to near-silence (two
    # people talking over each other is unusable) while BGM only ducks a
    # moderate amount (music can sit under narration, it doesn't compete for
    # intelligibility the way dialogue does). BGM still ducks under SOT too,
    # for the stretches where VO isn't playing but original dialogue is.
    #
    # Broadcast delivery toggle: if enabled, every element is forced to be
    # identically audible in both channels (true center / dual-mono), so nothing
    # is lost if played back or checked on a single channel. This is applied as
    # the very last step, after all ducking/mixing — it only changes the stereo
    # image (L/R placement), never the levels or ducking decided above it.
    CENTER_FILTER = 'aformat=channel_layouts=stereo,pan=stereo|c0=0.5*c0+0.5*c1|c1=0.5*c0+0.5*c1'
    if bgm_ready_path or vo_ready_path or broadcast_stereo:
        job_set(jid, step='Finalizing audio mix')
        final_mixed = os.path.join(app.config['UPLOAD_FOLDER'], f'finalmix_{base_ts}.mp4')
        cmd = [FFMPEG, '-y', '-i', out_path]
        inputs = []
        if bgm_ready_path:
            cmd += ['-i', bgm_ready_path]
            inputs.append('bgm')
        if vo_ready_path:
            cmd += ['-i', vo_ready_path]
            inputs.append('vo')

        tail = (CENTER_FILTER + ',' if broadcast_stereo else '') + f'loudnorm=I={target_loudness}:TP={true_peak}:LRA=7'

        if vo_ready_path:
            vo_idx = 1 + inputs.index('vo')
            fc = []
            if bgm_ready_path:
                # Need 3 copies of VO: one to actually mix in, and one each to key
                # the SOT duck and the BGM duck off of.
                fc.append(f'[{vo_idx}:a]asplit=3[vo_out][vokey1][vokey2]')
            else:
                fc.append(f'[{vo_idx}:a]asplit=2[vo_out][vokey1]')
            # SOT ducked to near-silence under VO (fast attack, low threshold, high
            # ratio, no makeup — this is meant to sit well below the VO, not just
            # lower than before).
            fc.append('[0:a][vokey1]sidechaincompress=threshold=0.01:ratio=20:attack=5:release=250:makeup=1[sot_ducked]')
            mix_labels = ['sot_ducked']
            if bgm_ready_path:
                bgm_idx = 1 + inputs.index('bgm')
                # BGM (already ducked under SOT) gets a second, moderate duck under VO.
                fc.append(f'[{bgm_idx}:a][vokey2]sidechaincompress=threshold=0.05:ratio=4:attack=20:release=400:makeup=1[bgm_ducked2]')
                mix_labels.append('bgm_ducked2')
            mix_labels.append('vo_out')
            fc.append('[' + ']['.join(mix_labels) + f']amix=inputs={len(mix_labels)}:duration=first:dropout_transition=2:normalize=0[premix]')
            fc.append(f'[premix]{tail}[outa]')
            filter_complex = ';'.join(fc)
        elif bgm_ready_path:
            # BGM only, no VO — same moderate BGM-under-SOT duck as before.
            bgm_idx = 1
            filter_complex = (f'[0:a][{bgm_idx}:a]amix=inputs=2:duration=first:dropout_transition=2:normalize=0[premix];'
                               f'[premix]{tail}[outa]')
        else:
            # Neither BGM nor VO — only reached when the broadcast toggle needs
            # to center the SOT on its own.
            filter_complex = f'[0:a]{tail}[outa]'

        cmd += ['-filter_complex', filter_complex, '-map', '0:v', '-map', '[outa]',
                '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-shortest', final_mixed]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if os.path.exists(final_mixed) and os.path.getsize(final_mixed) > 0:
            os.replace(final_mixed, out_path)
        else:
            print(f'Final audio mix error: {r.stderr[:500]}')
        if bgm_ready_path and os.path.exists(bgm_ready_path):
            os.remove(bgm_ready_path)
        if vo_ready_path and os.path.exists(vo_ready_path):
            os.remove(vo_ready_path)

    job_set(jid, percent=100, step='Done', done=True, result=dict(
        status='ok', trailer_url=f'/uploads/{filename}',
        orig_name=orig_name,
        total_scenes=len(scene_list), selected_scenes=len(selected),
        trailer_duration=round(total_sel, 1),
        video_duration=round(video_duration, 1),
        trailer_length=trailer_length,
        bgm_source=bgm_source, sfx_source=sfx_source,
        vo_source=vo_source, vo_error=vo_error, sync_beats=sync_beats,
        whisper_enhance=whisper_enhance, hook_line=hook_line,
        scenes=[{
            'scene': i+1, 'start': round(s['start'], 1), 'end': round(s['end'], 1),
            'quality': s['total_score'], 'duration': round(s['selected_dur'], 1),
            'description': _scene_desc(s)
        } for i, s in enumerate(selected)]))

# ---- UI ----

UI = '''
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PromoPlug X (PPX) - OpenCV, PySceneDetect, AI Vision & FFmpeg</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='7' fill='%230b1220'/%3E%3Cpath d='M9 10h9l3 3v9H9z' fill='none' stroke='%2334e6c5' stroke-width='2'/%3E%3Ccircle cx='13' cy='16' r='2.4' fill='%2334e6c5'/%3E%3C/svg%3E">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0b1220;
  --panel:#121a2b;
  --elevated:#1a2436;
  --sunken:#03070d;
  --line:#263149;
  --ink:#e7edf6;
  --ink-dim:#8b98ad;
  --phosphor:#34e6c5;
  --phosphor-dim:#1d8f7c;
  --tally:#ff5470;
  --amber:#ffb545;
  --radius:10px;
}
html{scroll-behavior:smooth}
body{
  background:
    radial-gradient(ellipse 900px 480px at 12% -12%, rgba(52,230,197,.07), transparent 60%),
    radial-gradient(ellipse 700px 400px at 100% 0%, rgba(255,181,69,.05), transparent 55%),
    var(--bg);
  color:var(--ink);
  font-family:'IBM Plex Sans',system-ui,-apple-system,sans-serif;
  padding-bottom:92px;
  min-height:100vh;
  -webkit-font-smoothing:antialiased;
}
.container{max-width:1080px;margin:0 auto;padding:28px 24px 20px}

/* ---- header ---- */
.hdr{position:sticky;top:0;z-index:20;background:rgba(11,18,32,.88);backdrop-filter:blur(10px);border-bottom:1px solid var(--line)}
.hdr-inner{max-width:1080px;margin:0 auto;padding:16px 24px;display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap}
h1{font-family:'JetBrains Mono',monospace;font-size:16px;font-weight:700;letter-spacing:.04em;text-transform:uppercase;display:flex;align-items:center;gap:9px}
h1::before{content:'▚';color:var(--phosphor);font-size:14px}
h1 small{font-family:'JetBrains Mono',monospace;font-weight:400;text-transform:none;letter-spacing:0;font-size:11px;color:var(--ink-dim);display:block;margin-top:4px}
.hdr-engines{font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--ink-dim);letter-spacing:.09em;text-transform:uppercase;white-space:nowrap}

/* ---- bottom page selector (DaVinci-style tool dock) ---- */
.tabs{position:fixed;left:0;right:0;bottom:0;z-index:30;display:flex;justify-content:center;gap:2px;background:rgba(13,19,32,.96);backdrop-filter:blur(10px);border-top:1px solid var(--line);padding:7px 10px;overflow-x:auto}
.tab{font-family:'JetBrains Mono',monospace;padding:8px 16px;cursor:pointer;background:transparent;border:1px solid transparent;color:var(--ink-dim);font-size:11px;letter-spacing:.03em;text-transform:uppercase;border-radius:8px;white-space:nowrap;user-select:none;text-align:center;transition:color .15s,background .15s,border-color .15s}
.tab:hover{color:var(--ink);background:rgba(255,255,255,.04)}
.tab.active{color:var(--phosphor);background:rgba(52,230,197,.09);border-color:rgba(52,230,197,.28)}
.view-toggle-btn{font-family:'JetBrains Mono',monospace;padding:7px 16px;cursor:pointer;background:transparent;border:none;color:var(--ink-dim);font-size:11px;letter-spacing:.03em;text-transform:uppercase}
.view-toggle-btn+.view-toggle-btn{border-left:1px solid var(--line)}
.view-toggle-btn.active{color:var(--bg);background:var(--phosphor)}
.tab-icon{display:block;font-size:15px;margin-bottom:3px}
.tab-sub{font-size:9px;color:var(--ink-dim);display:block;font-weight:400;letter-spacing:.06em;margin-top:3px;text-transform:none}
.tab.active .tab-sub{color:var(--phosphor-dim)}

.panel{display:none;background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);padding:24px;margin-bottom:20px}
.panel.active{display:block;animation:fade-in .2s ease}
@keyframes fade-in{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}

h2{font-family:'JetBrains Mono',monospace;font-size:13px;text-transform:uppercase;letter-spacing:.05em;color:var(--ink);margin-bottom:16px;padding-bottom:11px;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:9px}
h2::before{content:'';width:6px;height:6px;background:var(--phosphor);border-radius:1px;flex:none}
h2 small{font-family:'IBM Plex Sans',sans-serif;text-transform:none;letter-spacing:0;font-weight:400;color:var(--ink-dim);font-size:12px}

p{color:var(--ink-dim);font-size:13px;line-height:1.65;margin:8px 0}

.btn{font-family:'JetBrains Mono',monospace;background:transparent;color:var(--phosphor);border:1px solid var(--phosphor-dim);padding:9px 18px;border-radius:7px;cursor:pointer;font-size:12px;letter-spacing:.04em;text-transform:uppercase;display:inline-block;text-decoration:none;transition:all .15s}
.btn:hover{background:rgba(52,230,197,.1);border-color:var(--phosphor)}
.btn:active{transform:translateY(1px)}
.btn:focus-visible,input:focus-visible,select:focus-visible{outline:2px solid var(--phosphor);outline-offset:2px}
.btn.danger{color:var(--tally);border-color:rgba(255,84,112,.5)}
.btn.danger:hover{background:rgba(255,84,112,.1);border-color:var(--tally)}
.btn.small{padding:6px 13px;font-size:11px}
.btn.active-filter{background:var(--phosphor);color:#04140f;border-color:var(--phosphor);font-weight:600}

label{display:block;margin:16px 0 6px;font-family:'JetBrains Mono',monospace;font-size:10px;text-transform:uppercase;letter-spacing:.06em;color:var(--ink-dim)}
input[type=file]{display:block;margin:10px 0;color:var(--ink-dim);font-size:13px;font-family:'IBM Plex Sans',sans-serif}
input[type=url],input[type=number],input[type=text],select{width:100%;padding:10px 12px;background:var(--elevated);border:1px solid var(--line);border-radius:7px;margin:6px 0;font-size:14px;color:var(--ink);font-family:'IBM Plex Sans',sans-serif}
select{appearance:none;background-image:linear-gradient(45deg,transparent 50%,var(--ink-dim) 50%),linear-gradient(135deg,var(--ink-dim) 50%,transparent 50%);background-position:calc(100% - 18px) center,calc(100% - 13px) center;background-size:5px 5px,5px 5px;background-repeat:no-repeat}

.card{background:var(--elevated);border:1px solid var(--line);border-radius:8px;padding:16px;margin:12px 0}
table{width:100%;border-collapse:collapse;margin:14px 0;font-size:13px}
th,td{border-bottom:1px solid var(--line);padding:9px 10px;text-align:left;font-variant-numeric:tabular-nums}
th{color:var(--ink-dim);font-family:'JetBrains Mono',monospace;font-size:10px;text-transform:uppercase;letter-spacing:.05em;font-weight:500}
td{font-size:13px}
tr:hover td{background:rgba(255,255,255,.02)}
code{font-family:'JetBrains Mono',monospace;background:var(--elevated);padding:2px 6px;border-radius:4px;font-size:12px;color:var(--phosphor);border:1px solid var(--line)}
pre{font-family:'JetBrains Mono',monospace;background:var(--sunken);border:1px solid var(--line);padding:14px;border-radius:8px;font-size:12px;overflow-x:auto;line-height:1.75;color:var(--ink-dim)}

.filters{display:flex;gap:6px;margin:14px 0;flex-wrap:wrap;align-items:center}
.rec-dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--tally);margin-right:4px;animation:pulse 1.4s infinite ease-in-out;vertical-align:middle}
.rec-label{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--tally);letter-spacing:.04em;text-transform:uppercase;display:none;align-items:center}
.rec-label.live{display:inline-flex}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.25}}

.stream-wrap{text-align:center;position:relative;padding:16px;background:var(--sunken);border-radius:10px;border:1px solid var(--line)}
.stream-wrap img,.stream-wrap video,.stream-wrap canvas{max-width:100%;border-radius:4px;max-height:550px;display:inline-block}
.corner{position:absolute;width:16px;height:16px;border-color:var(--phosphor-dim);opacity:.65;pointer-events:none}
.corner.tl{top:10px;left:10px;border-top:2px solid;border-left:2px solid;border-radius:3px 0 0 0}
.corner.tr{top:10px;right:10px;border-top:2px solid;border-right:2px solid;border-radius:0 3px 0 0}
.corner.bl{bottom:10px;left:10px;border-bottom:2px solid;border-left:2px solid;border-radius:0 0 0 3px}
.corner.br{bottom:10px;right:10px;border-bottom:2px solid;border-right:2px solid;border-radius:0 0 3px 0}

.info{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px;margin:12px 0}
.info-item{background:var(--elevated);border:1px solid var(--line);padding:8px 12px;border-radius:6px;font-size:12px;font-family:'JetBrains Mono',monospace;color:var(--ink-dim)}
.info-item strong{color:var(--phosphor);font-weight:600;margin-right:5px}
.no-data{text-align:center;padding:50px 20px;color:var(--ink-dim);font-size:12.5px;font-family:'JetBrains Mono',monospace;border:1px dashed var(--line);border-radius:8px;margin-top:14px;letter-spacing:.02em}

input[type=range]{-webkit-appearance:none;appearance:none;width:100%;height:4px;background:var(--line);border-radius:2px;margin:16px 0 8px;cursor:pointer}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;width:14px;height:14px;border-radius:50%;background:var(--phosphor);cursor:pointer;box-shadow:0 0 0 4px rgba(52,230,197,.15)}
input[type=range]::-moz-range-thumb{width:14px;height:14px;border-radius:50%;background:var(--phosphor);border:none;cursor:pointer}

@media (prefers-reduced-motion:reduce){.rec-dot{animation:none}.panel.active{animation:none}html{scroll-behavior:auto}}
@media (max-width:640px){.hdr-engines{display:none}.panel{padding:18px}}
.sub-tabs{display:flex;gap:4px;margin:0 0 16px;border-bottom:1px solid var(--line);padding:0 0 8px}
.sub-tab{padding:6px 14px;border-radius:6px;cursor:pointer;font-size:12px;font-weight:500;color:var(--ink-dim);transition:all .15s;user-select:none}
.sub-tab:hover{color:var(--ink);background:var(--elevated)}
.sub-tab.active{color:var(--phosphor);background:rgba(52,230,197,.1)}
.sub-panel{display:none}
.sub-panel.active{display:block}
</style>
</head>
<body>
<div class="hdr"><div class="hdr-inner">
  <h1>PromoPlug X <span style="font-size:.55em;opacity:.65;letter-spacing:.04em">(PPX)</span><small>Create Compelling Episodic Promos in Minutes</small></h1>
  <div class="hdr-engines">scene detection &middot; ai vision &middot; ffmpeg &middot; ai music generation &middot; text-to-speech &middot; speech-to-text</div>
</div></div>
<div class="container">
<div class="tabs">
  <div class="tab active" onclick="switchTab('p-trailer',this)" role="button" tabindex="0"><span class=tab-icon>&#9636;</span>Generate Promo Plug<div class=tab-sub>ai+ffmpeg</div></div>
  <div class="tab" onclick="switchTab('p-tools',this)" role="button" tabindex="0"><span class=tab-icon>&#9881;</span>Tools<div class=tab-sub>player+upload+vision</div></div>
  <div class="tab" onclick="switchTab('p-api',this)" role="button" tabindex="0"><span class=tab-icon>{ }</span>API<div class=tab-sub>reference</div></div>
  <div class="tab" onclick="switchTab('p-docs',this)" role="button" tabindex="0"><span class=tab-icon>&#9776;</span>Docs<div class=tab-sub>workflow+genres</div></div>
</div>

<script>function switchTab(id,btn){document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));document.querySelectorAll('.sub-panel').forEach(p=>p.classList.remove('active'));document.querySelectorAll('.sub-tab').forEach(t=>t.classList.remove('active'));btn.classList.add('active');document.getElementById(id).classList.add('active');if(id==='p-tools'){var fst=document.querySelector('#p-tools .sub-tab');if(fst)fst.click()}}function switchSubTab(id,btn){document.querySelectorAll('.sub-tab').forEach(t=>t.classList.remove('active'));document.querySelectorAll('.sub-panel').forEach(p=>p.classList.remove('active'));btn.classList.add('active');document.getElementById(id).classList.add('active')}</script>
<script>
// Universal "Clear" button for every file input on the page: shows the chosen
// filename next to the picker and a small Clear button that resets the input
// and fires a change event (so any per-field wiring, like the card VO preview
// player, updates too). Runs once at load; every <input type=file> is present
// in the initial markup, none are created dynamically later.
document.addEventListener("DOMContentLoaded", function(){
  document.querySelectorAll("input[type=file]").forEach(function(inp){
    if(inp.dataset.clearWired) return
    inp.dataset.clearWired = "1"
    var status = document.createElement("span")
    status.style.fontSize = "11px"
    status.style.opacity = "0.75"
    status.style.margin = "0 8px"
    status.style.fontFamily = "JetBrains Mono, monospace"
    var clearBtn = document.createElement("button")
    clearBtn.type = "button"
    clearBtn.textContent = "\u2715 Clear"
    clearBtn.style.display = "inline-block"
    clearBtn.style.fontFamily = "JetBrains Mono, monospace"
    clearBtn.style.background = "transparent"
    clearBtn.style.color = "var(--ink-dim)"
    clearBtn.style.border = "1px solid var(--line)"
    clearBtn.style.padding = "2px 8px"
    clearBtn.style.borderRadius = "5px"
    clearBtn.style.fontSize = "11px"
    clearBtn.style.verticalAlign = "middle"
    clearBtn.addEventListener("click", function(){
      inp.value = ""
      inp.dispatchEvent(new Event("change", {bubbles:true}))
    })
    function refresh(){
      if(inp.files && inp.files[0]){
        status.textContent = inp.files[0].name
        clearBtn.disabled = false
        clearBtn.style.opacity = "1"
        clearBtn.style.cursor = "pointer"
      } else {
        status.textContent = "No file chosen"
        clearBtn.disabled = true
        clearBtn.style.opacity = "0.35"
        clearBtn.style.cursor = "default"
      }
    }
    inp.addEventListener("change", refresh)
    inp.insertAdjacentElement("afterend", status)
    status.insertAdjacentElement("afterend", clearBtn)
    refresh()
  })
})
</script>

<!-- Tools -->
<div id="p-tools" class="panel">
<div class="sub-tabs">
  <div class="sub-tab active" onclick="switchSubTab('p-player',this)" role="button" tabindex="0">&#9654; Player</div>
  <div class="sub-tab" onclick="switchSubTab('p-upload',this)" role="button" tabindex="0">&#8682; Upload</div>
  <div class="sub-tab" onclick="switchSubTab('p-vision',this)" role="button" tabindex="0">&#9673; Vision AI</div>
</div>

<!-- Player -->
<div id="p-player" class="sub-panel active">
<h2>Video Player</h2>
<form id=pf method=POST action=/api/playback/start enctype=multipart/form-data>
  <input type=file name=file accept=video/*>
  <button class=btn type=submit>Load</button>
</form>
<div id=pb-area style=display:none>
  <div class=info id=pb-info></div>
  <div class=stream-wrap>
    <i class="corner tl"></i><i class="corner tr"></i><i class="corner bl"></i><i class="corner br"></i>
    <img id=pb-feed>
  </div>
  <input type=range class=progress id=pb-progress min=0 max=100 value=0>
  <div class=filters>
    <button class="btn small active-filter" id="pb-raw">Raw</button>
    <button class="btn small" id="pb-gray">Gray</button>
    <button class="btn small" id="pb-edges">Edges</button>
    <button class="btn small" id="pb-hsv">HSV</button>
    <button class="btn small" id="pb-blur">Blur</button>
    <button class="btn small" id="pb-face">Face</button>
    <button class="btn small" id="pb-motion">Motion</button>
    <button class="btn small" id="pb-playbtn">Pause</button>
    <button class="btn small danger" id="pb-stopbtn">Stop</button>
  </div>
</div>
<div id=pb-prompt class=no-data>Load a video file to play.</div>
</div>

<!-- Upload -->
<div id="p-upload" class="sub-panel">
<h2>Upload &amp; Analyze</h2>
<form method=POST action=/upload enctype=multipart/form-data onsubmit="return true">
  <input type=file name=file accept=video/*>
  <button class=btn type=submit>Analyze</button>
</form>
{% if r %}
<div class=card>
<div class=info>
  <div class=info-item><strong>W:</strong> {{ r.info.width }}</div>
  <div class=info-item><strong>H:</strong> {{ r.info.height }}</div>
  <div class=info-item><strong>FPS:</strong> {{ r.info.fps }}</div>
  <div class=info-item><strong>Dur:</strong> {{ r.info.duration_sec }}s</div>
  <div class=info-item><strong>Scenes:</strong> {{ r.scenes|length }}</div>
</div>
</div>
{% if r.scenes %}
<table><tr><th>#</th><th>Start</th><th>End</th><th>Dur</th></tr>
{% for s in r.scenes %}<tr><td>{{ s.scene }}</td><td>{{ s.start }}</td><td>{{ s.end }}</td><td>{{ s.duration }}s</td></tr>{% endfor %}
</table>
{% endif %}
{% if r.frames %}
<table><tr><th>Frame</th><th>Shape</th><th>BGR</th><th>Bright</th><th>Edges</th><th>Corners</th></tr>
{% for f in r.frames %}<tr><td>{{ f.idx }}</td><td>{{ f.shape }}</td><td>{{ f.mean_bgr }}</td><td>{{ f.brightness }}</td><td>{{ f.edge_pixels }}</td><td>{{ f.corners }}</td></tr>{% endfor %}
</table>
{% endif %}
{% endif %}
</div>

<!-- Vision -->
<div id="p-vision" class="sub-panel">
<h2>AI Vision</h2>
<form id=vf method=POST action=/api/vision/analyze enctype=multipart/form-data>
  <input type=file name=file accept=video/*>
  <label>Custom prompt:</label>
  <input type=text name=prompt value="Describe the quality and content of this video frame. Note any blur, color issues, or anomalies.">
  <label>Frames to analyze:</label>
  <input type=number name=num_frames value=5 min=1 max=20>
  <label>Model:</label>
  <select name=model id=vision-model><option value="">Loading...</option></select>
  <button class=btn type=submit>Analyze with AI</button>
</form>
<div id=vr-area style=display:none>
  <div class=card id=vr-result></div>
</div>
<div id=vr-prompt class=no-data>Upload a video to analyze frames with the AI Vision model.</div>
</div>
</div>

<!-- Episodic Promo Plug -->
<div id="p-trailer" class="panel active">
<h2>Episodic Promo Plug Generator</h2>
<div id=tr-area style=display:none>
  <div class=card id=tr-stats></div>
  <div style=margin:10px 0 id=tr-video></div>
  <table id=tr-table><tr><th>Scene</th><th>Start</th><th>End</th><th>Quality</th><th>Used (s)</th></tr></table>
</div>
<div id=tr-prompt class=no-data>Upload a video to generate an episodic promo plug from the best scenes.</div>
<div id=tr-progress-area class=no-data style="display:none;text-align:left;padding:20px 24px">
  <div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:4px;color:var(--ink)">
    <span id=tr-progress-step>Working...</span>
    <span id=tr-progress-pct>0%</span>
  </div>
  <div style="background:var(--panel-2,#1f232b);border-radius:6px;overflow:hidden;height:10px">
    <div id=tr-progress-bar style="background:var(--accent,#4f8cff);height:100%;width:0%;transition:width .3s ease"></div>
  </div>
</div>
<hr style="border:none;border-top:1px solid var(--line);margin:20px 0">
<form id=tf method=POST action=/api/trailer/generate enctype=multipart/form-data>
  <div class=card style="padding:10px 14px;margin:0 0 16px;display:flex;align-items:center;gap:12px;flex-wrap:wrap">
    <span style="font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--ink-dim)">View:</span>
    <div style="display:inline-flex;border:1px solid var(--line);border-radius:8px;overflow:hidden">
      <button type=button id=view-easy-btn class="view-toggle-btn active" onclick="setViewMode('easy')">Quick</button>
      <button type=button id=view-adv-btn class="view-toggle-btn" onclick="setViewMode('advanced')">Advanced</button>
    </div>
    <span style="font-size:11px;opacity:.7">Quick shows just the essentials — switch to Advanced for transitions, SFX, narration, and manual tuning.</span>
  </div>
  <div id=ollama-down-banner class=card style="display:none;border-color:var(--tally);background:rgba(255,84,112,.08);margin:0 0 16px">
    <strong style="color:var(--tally)">&#9888; AI Vision is unreachable.</strong>
    <span style="font-size:13px">Scene scoring needs it, so generation is disabled until it's back. Start Ollama, then click "Check services" on the API tab to retry.</span>
  </div>
  <input type=file name=file accept=video/*>
  <label>Scoring mode:</label>
  <select name=mode id=scoring-mode-select>
    <option value=ai selected>VISION (OpenCV + AI Vision scoring)</option>
    <option value=ai_stt data-requires="whisper">VISION + STT (adds faster-whisper dialogue transcription)</option>
  </select>
  <p id=mode-gating-note style="display:none;margin-top:-4px;margin-bottom:8px;font-size:12px;color:var(--amber)">VISION + STT is unavailable — faster-whisper isn't reachable.</p>
  <label>Episodic Promo Plug length:</label>
  <select name=trailer_length>
    <option value=15 selected>15 sec (needs 22.5s raw video)</option>
    <option value=30>30 sec (needs 45s raw video)</option>
    <option value=45>45 sec (needs 67.5s raw video)</option>
    <option value=60>60 sec (needs 90s raw video)</option>
  </select>
  <div class="adv-only">
  <label>Max clip length (s):</label>
  <input type=number name=max_scene_dur placeholder="no limit" min=0.5 step=0.5 style="width:100px">
  <p style="margin-top:-4px;margin-bottom:8px;font-size:12px;opacity:.75">Caps how long any single selected scene can run, even if the duration budget would allow more. Leave blank for no limit.</p>
  <label>Scene cut sensitivity (lower = more cuts detected):</label>
  <input type=number name=scene_threshold value=30 min=1 max=100 step=1 style="width:100px">
  <p style="margin-top:-4px;margin-bottom:8px;font-size:12px;opacity:.75">Same threshold used by the "Preview scene cuts" tool in the API tab — matches what you see there, so preview and generation agree on the same cut list. Lower values detect more/subtler cuts; higher values only catch hard cuts.</p>
  <label>Minimum scene length (s):</label>
  <input type=number name=min_scene_len value=0.5 min=0.1 max=5 step=0.1 style="width:100px">
  <p style="margin-top:-4px;margin-bottom:8px;font-size:12px;opacity:.75">Scene changes shorter than this are merged into the surrounding scene instead of becoming their own tiny fragment (avoids flicker cuts from whip-pans/motion blur).</p>
  </div>
  <label>Genre (presets transition, music, sfx):</label>
  <select name=genre>
    <option value="">Custom (manual settings below)</option>
    <option value=action>Action</option>
    <option value=adventure>Adventure</option>
    <option value=comedy>Comedy</option>
    <option value=documentary>Documentary</option>
    <option value=drama>Drama</option>
    <option value=fantasy>Fantasy</option>
    <option value=horror>Horror</option>
    <option value=mystery>Mystery</option>
    <option value=noir>Noir</option>
    <option value=romance>Romance</option>
    <option value=scifi>Sci-Fi</option>
    <option value=sports>Sports</option>
    <option value=thriller>Thriller</option>
    <option value=war>War</option>
    <option value=western>Western</option>
  </select>
  <div class="genre-manual-transition" style="display:none">
  <label>Transition:</label>
  <select name=transition>
    <option value=fade selected>Fade</option>
    <option value=dissolve>Dissolve</option>
    <option value=fadeblack>Fade to Black</option>
    <option value=fadewhite>Fade to White</option>
    <option value=slideleft>Slide Left</option>
    <option value=slideright>Slide Right</option>
    <option value=wipeleft>Wipe Left</option>
    <option value=wiperight>Wipe Right</option>
    <option value=wipeup>Wipe Up</option>
    <option value=wipedown>Wipe Down</option>
    <option value=circleopen>Circle Open</option>
    <option value=circleclose>Circle Close</option>
    <option value=radial>Radial</option>
    <option value=zoomin>Zoom In</option>
    <option value=pixelize>Pixelize</option>
    <option value=smoothleft>Smooth Left</option>
    <option value=smoothright>Smooth Right</option>
    <option value=horzopen>Horizontal Open</option>
    <option value=horzclose>Horizontal Close</option>
    <option value=squeezeh>Squeeze Horizontal</option>
    <option value=squeezev>Squeeze Vertical</option>
    <option value=custom_matte>Custom (upload matte)</option>
  </select>
  <div id=transition-matte-area style="display:none;margin-top:8px">
    <input type=file name=transition_matte accept="video/*,image/*">
    <p style="margin-top:4px;margin-bottom:0;font-size:12px;opacity:.75">A short video (or a single image) whose brightness drives the wipe: black keeps the outgoing scene, white reveals the incoming one, and whatever animates in between defines the transition's shape. Trimmed to the transition duration below and reused at every cut. Falls back to Fade if nothing is uploaded.</p>
  </div>
  </div>
  <div class="adv-only" style="display:none">
  <label>Transition duration (s):</label>
  <input type=number name=xfade_dur value=0.3 min=0.1 max=2 step=0.1 style="width:100px">
  <label>Audio normalisation:</label>
  <div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:8px 0">
    <span style="font-size:12px">Target&nbsp;LUFS:</span>
    <input type=number name=target_loudness value=-14 min=-30 max=-10 step=0.5 style="width:70px">
    <span style="font-size:12px">True&nbsp;peak&nbsp;(dB):</span>
    <input type=number name=true_peak value=-1.5 min=-6 max=0 step=0.5 style="width:70px">
    <span style="font-size:12px">Music&nbsp;ducking&nbsp;(dB):</span>
    <input type=number name=music_duck_db value=-3 min=-24 max=0 step=0.5 style="width:70px">
    <label style="font-size:13px;text-transform:none;letter-spacing:0;display:flex;align-items:center;gap:4px;margin:0 0 0 12px;cursor:pointer">
      <input type=checkbox name=beat_match checked> Beat match (librosa)
    </label>
    <label style="font-size:13px;text-transform:none;letter-spacing:0;display:flex;align-items:center;gap:4px;margin:0 0 0 12px;cursor:pointer">
      <input type=checkbox name=sync_beats checked> Sync cuts to beat
    </label>
  </div>
  <p style="margin-top:-4px;margin-bottom:8px;font-size:12px;opacity:.75">Selecting "VISION + STT" above transcribes dialogue locally to boost scoring for scenes with quotable lines, snap cut points to word boundaries instead of mid-word, and surface a suggested hook line. Uses the large-v2 Whisper model by default (needed for reliable Tagalog — smaller models are noticeably worse at it), so the first run may take a bit to load.</p>
  <div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:8px 0">
    <label style="font-size:13px;text-transform:none;letter-spacing:0;display:flex;align-items:center;gap:4px;margin:0;cursor:pointer">
      <input type=checkbox name=broadcast_stereo> Broadcast dual-mono (force all audio identical in L/R)
    </label>
  </div>
  <p style="margin-top:-4px;margin-bottom:8px;font-size:12px;opacity:.75">Applied last, after all ducking and level normalization — collapses the stereo image so every element (SOT, music, voiceover) is heard equally in both channels, per broadcast delivery requirements. Levels and ducking are unaffected.</p>
  </div>
  <div class="adv-only">
  <label>AI Vision model:</label>
  <select name=model id=trailer-model><option value="">Loading...</option></select>
  </div>
  <label>Title card video (optional):</label>
  <input type=file name=end_card_video accept=video/*>
  <div class="adv-only">
  <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:6px 0 4px">
    <span style="font-size:12px">VO for this card (optional, replaces its audio):</span>
    <input type=file name=title_card_vo accept="audio/*,.mp3,.wav,.m4a,.flac,.ogg" style="width:auto" onchange="cardVoFileChosen(this,'title_card_vo')">
  </div>
  <audio id="title_card_vo_player" controls style="display:none;width:100%;max-width:420px;height:32px;margin-bottom:6px"></audio>
  <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:0 0 12px">
    <button type=button class=btn style="padding:4px 10px;font-size:11px" onclick="cardVoSetPoint('title_card_vo','start')">Set in (player)</button>
    <button type=button class=btn style="padding:4px 10px;font-size:11px" onclick="cardVoSetPoint('title_card_vo','end')">Set out (player)</button>
    <span style="font-size:12px">or manually - start (s):</span>
    <input type=number id="title_card_vo_start" name=title_card_vo_start value=0 min=0 step=0.1 style="width:70px">
    <span style="font-size:12px">end (s, blank=to end):</span>
    <input type=number id="title_card_vo_end" name=title_card_vo_end min=0 step=0.1 style="width:70px">
    <span id="title_card_vo_preview_note" style="font-size:11px;opacity:.7"></span>
  </div>
  </div>
  <label>End card video (optional):</label>
  <input type=file name=schedule_video accept=video/*>
  <div class="adv-only">
  <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:6px 0 4px">
    <span style="font-size:12px">VO for this card (optional, replaces its audio):</span>
    <input type=file name=end_card_vo accept="audio/*,.mp3,.wav,.m4a,.flac,.ogg" style="width:auto" onchange="cardVoFileChosen(this,'end_card_vo')">
  </div>
  <audio id="end_card_vo_player" controls style="display:none;width:100%;max-width:420px;height:32px;margin-bottom:6px"></audio>
  <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:0 0 12px">
    <button type=button class=btn style="padding:4px 10px;font-size:11px" onclick="cardVoSetPoint('end_card_vo','start')">Set in (player)</button>
    <button type=button class=btn style="padding:4px 10px;font-size:11px" onclick="cardVoSetPoint('end_card_vo','end')">Set out (player)</button>
    <span style="font-size:12px">or manually - start (s):</span>
    <input type=number id="end_card_vo_start" name=end_card_vo_start value=0 min=0 step=0.1 style="width:70px">
    <span style="font-size:12px">end (s, blank=to end):</span>
    <input type=number id="end_card_vo_end" name=end_card_vo_end min=0 step=0.1 style="width:70px">
    <span id="end_card_vo_preview_note" style="font-size:11px;opacity:.7"></span>
  </div>
  </div>
  <label>Background music:</label>

  <div style="display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:8px 0">
    <label style="font-size:13px;text-transform:none;letter-spacing:0;display:flex;align-items:center;gap:4px;margin:0;cursor:pointer">
      <input type=radio name=scoring_mode value=generate data-requires="ace_step" checked> Generate music
    </label>
    <label style="font-size:13px;text-transform:none;letter-spacing:0;display:flex;align-items:center;gap:4px;margin:0;cursor:pointer">
      <input type=radio name=scoring_mode value=none> None
    </label>
    <label style="font-size:13px;text-transform:none;letter-spacing:0;display:flex;align-items:center;gap:4px;margin:0;cursor:pointer">
      <input type=radio name=scoring_mode value=upload> Upload
    </label>
  </div>
  <p id=bgm-gating-note style="display:none;margin-top:-4px;margin-bottom:8px;font-size:12px;color:var(--amber)">"Generate music" is unavailable — ACE-Step isn't reachable.</p>
  <div id=scoring-upload-area style="display:none">
    <input type=file name=scoring_audio accept="audio/*,.mp3,.wav,.m4a,.flac,.ogg">
  </div>

  <div class="adv-only">
  <label>Sound effects (stamped at every scene cut):</label>
  <div style="display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:8px 0">
    <label style="font-size:13px;text-transform:none;letter-spacing:0;display:flex;align-items:center;gap:4px;margin:0;cursor:pointer">
      <input type=radio name=sfx_mode value=genre data-requires="woosh,ace_step"> From genre
    </label>
    <label style="font-size:13px;text-transform:none;letter-spacing:0;display:flex;align-items:center;gap:4px;margin:0;cursor:pointer">
      <input type=radio name=sfx_mode value=upload> Upload one-shot
    </label>
    <label style="font-size:13px;text-transform:none;letter-spacing:0;display:flex;align-items:center;gap:4px;margin:0;cursor:pointer">
      <input type=radio name=sfx_mode value=none checked> None
    </label>
  </div>
  <p id=sfx-gating-note style="display:none;margin-top:-4px;margin-bottom:8px;font-size:12px;color:var(--amber)">"From genre" is unavailable — Woosh and ACE-Step aren't reachable.</p>
  <div id=sfx-upload-area style="display:none">
    <input type=file name=sfx_upload accept="audio/*,.mp3,.wav,.m4a,.flac,.ogg">
    <p style="margin-top:4px">A single hit/whoosh/impact sound — it gets stamped at every cut, not looped as music.</p>
  </div>
  </div>

  <div class="adv-only">
  <label>Narration:</label>
  <div style="display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:8px 0">
    <label style="font-size:13px;text-transform:none;letter-spacing:0;display:flex;align-items:center;gap:4px;margin:0;cursor:pointer">
      <input type=radio name=vo_mode value=none checked> None
    </label>
    <label style="font-size:13px;text-transform:none;letter-spacing:0;display:flex;align-items:center;gap:4px;margin:0;cursor:pointer">
      <input type=radio name=vo_mode value=upload> Upload audio
    </label>
    <label style="font-size:13px;text-transform:none;letter-spacing:0;display:flex;align-items:center;gap:4px;margin:0;cursor:pointer">
      <input type=radio name=vo_mode value=tts data-requires="fish_audio,cozy_voice"> Narration (AI voice)
    </label>
  </div>
  <p id=vo-gating-note style="display:none;margin-top:-4px;margin-bottom:8px;font-size:12px;color:var(--amber)">"Narration (AI voice)" is unavailable — neither Fish Audio nor Cozy Voice 3 is reachable.</p>
  <div id=vo-upload-area style="display:none">
    <input type=file name=vo_upload accept="audio/*,.mp3,.wav,.m4a,.flac,.ogg">
    <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:6px 0">
      <span style="font-size:12px">Trim uploaded audio — start (s):</span>
      <input type=number name=vo_trim_start value=0 min=0 step=0.5 style="width:80px">
      <span style="font-size:12px">end (s, blank=to end):</span>
      <input type=number name=vo_trim_end min=0 step=0.5 style="width:80px">
    </div>
    <p style="margin-top:-2px;margin-bottom:8px;font-size:12px;opacity:.75">Selects which portion of the uploaded file to use as narration. This is separate from "Start at" below, which places the (already-trimmed) narration on the trailer's own timeline.</p>
  </div>
  <div id=vo-tts-area style="display:none">
    <textarea name=vo_text id=vo-text-input rows=3 style="width:100%;box-sizing:border-box" placeholder="Type the narration script here..."></textarea>

    <div style="display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:8px 0 4px">
      <span style="font-size:13px">Engine:</span>
      <label style="font-size:13px;text-transform:none;letter-spacing:0;display:flex;align-items:center;gap:4px;margin:0;cursor:pointer">
        <input type=radio name=vo_engine value=fish_audio checked> Fish Audio
      </label>
      <label style="font-size:13px;text-transform:none;letter-spacing:0;display:flex;align-items:center;gap:4px;margin:0;cursor:pointer">
        <input type=radio name=vo_engine value=cozy_voice> Cozy Voice 3
      </label>
    </div>

    <div style="display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:8px 0 4px">
      <label style="font-size:13px;text-transform:none;letter-spacing:0;display:flex;align-items:center;gap:4px;margin:0;cursor:pointer">
        <input type=radio name=vo_voice_source value=registered checked> Choose a voice
      </label>
      <label style="font-size:13px;text-transform:none;letter-spacing:0;display:flex;align-items:center;gap:4px;margin:0;cursor:pointer">
        <input type=radio name=vo_voice_source value=upload> Upload a reference sample
      </label>
    </div>
    <div id=vo-ref-upload-area style="display:none;margin-bottom:6px">
      <input type=file name=vo_ref_upload accept="audio/*,.mp3,.wav,.m4a,.flac,.ogg">
      <p style="margin-top:4px;margin-bottom:0;font-size:12px;opacity:.75">A short clean sample of the voice to clone (zero-shot — no pre-registration needed).</p>
    </div>
    <div id=vo-registered-voice-area style="margin-bottom:6px">
      <select name=vo_voice id=vo-voice-select style="max-width:100%"><option value="">Loading voices…</option></select>
      <span id=vo-voice-note style="font-size:12px;opacity:.75;margin-left:6px"></span>
    </div>

    <div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:8px 0">
      <span style="font-size:12px">Language:</span>
      <select name=vo_language id=vo-language-select style="max-width:220px"><option value="auto">Auto-detect (recommended)</option></select>
      <span style="font-size:12px">Rate (wpm):</span>
      <input type=number name=vo_rate id=vo-rate-input value=175 min=80 max=300 step=5 style="width:80px">
    </div>
    <p style="margin-top:4px;font-size:12px;opacity:.75">There's no built-in default voice — the "Choose a voice" list is fetched live from whichever engine is selected above (<code>/api/voices?engine=...</code>) and reloads automatically when you switch engines. Fish Audio's list is only populated when <code>FISH_AUDIO_API_KEY</code> is set (a plain self-hosted server generally has nothing to list) or if your self-hosted server happens to expose its own model-listing endpoint; Cozy Voice 3's list depends on whether your local server exposes one at <code>COZY_VOICE_URL</code>/voices. If the list comes back empty for either engine, switch to "Upload a reference sample" to clone a voice zero-shot instead. Fish Audio's S2 also auto-detects the script's language (including Tagalog) unless you override it above.</p>

    <div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:10px 0 4px">
      <button type=button id=vo-preview-btn onclick="previewVoiceover()">Generate &amp; preview</button>
      <span id=vo-preview-status style="font-size:12px;opacity:.75"></span>
    </div>
    <audio id=vo-preview-audio controls style="display:none;width:100%;margin-top:4px"></audio>
    <p style="margin-top:4px;font-size:12px;opacity:.75">Renders just this script through the settings above so you can check the voice, rate, and language before running the full trailer job.</p>
  </div>
  <div id=vo-common-area style="display:none;margin:8px 0">
    <span style="font-size:12px">Start at (s into promo plug):</span>
    <input type=number name=vo_start value=0 min=0 step=0.5 style="width:80px">
    <span style="font-size:12px">Audio level:</span>
    <input type=number name=vo_volume value=1.15 min=0.3 max=3.0 step=0.05 style="width:80px">
    <p style="margin-top:4px;font-size:12px;opacity:.75">Music, SFX, and original dialogue automatically duck under the voiceover wherever it plays. Audio level is a gain multiplier applied to the voiceover track before mixing (1.0 = unchanged, higher = louder).</p>
  </div>
  </div>

  <script>
  document.querySelectorAll('input[name=scoring_mode]').forEach(r=>{
    r.addEventListener('change',()=>{
      document.getElementById('scoring-upload-area').style.display=
        document.querySelector('input[name=scoring_mode]:checked').value==='upload'?'':'none'
    })
  })
  document.querySelectorAll('input[name=sfx_mode]').forEach(r=>{
    r.addEventListener('change',()=>{
      document.getElementById('sfx-upload-area').style.display=
        document.querySelector('input[name=sfx_mode]:checked').value==='upload'?'':'none'
    })
  })
  document.querySelectorAll('input[name=vo_mode]').forEach(r=>{
    r.addEventListener('change',()=>{
      var v=document.querySelector('input[name=vo_mode]:checked').value
      document.getElementById('vo-upload-area').style.display = v==='upload'?'':'none'
      document.getElementById('vo-tts-area').style.display = v==='tts'?'':'none'
      document.getElementById('vo-common-area').style.display = v==='none'?'none':''
      if(v==='tts') loadVoices()
    })
  })
  document.querySelectorAll('input[name=vo_voice_source]').forEach(r=>{
    r.addEventListener('change',()=>{
      var v=document.querySelector('input[name=vo_voice_source]:checked').value
      document.getElementById('vo-ref-upload-area').style.display = v==='upload'?'':'none'
      document.getElementById('vo-registered-voice-area').style.display = v==='registered'?'':'none'
    })
  })
  document.querySelectorAll('input[name=vo_engine]').forEach(r=>{
    r.addEventListener('change',()=>{ loadVoices(true) })
  })
  // Populate the "Choose a voice" and "Language" dropdowns from whichever engine
  // is currently selected. Cached client-side per engine per page load; pass
  // force=true (e.g. on engine switch, or the refresh button) to refetch.
  var _voicesLoadedFor = null
  function loadVoices(force){
    var engine = document.querySelector('input[name=vo_engine]:checked').value
    if(_voicesLoadedFor === engine && !force) return
    _voicesLoadedFor = engine
    var voiceSel = document.getElementById('vo-voice-select')
    voiceSel.innerHTML = '<option value="">Loading voices…</option>'
    fetch('/api/voices?engine=' + engine + (force ? '&refresh=1' : '')).then(r=>r.json()).then(function(d){
      voiceSel.innerHTML = ''
      ;(d.voices || []).filter(function(v){ return v.id }).forEach(function(v){
        var opt = document.createElement('option')
        opt.value = v.id
        opt.textContent = v.title + (v.languages && v.languages.length ? ' (' + v.languages.join(', ') + ')' : '')
        voiceSel.appendChild(opt)
      })
      var note = document.getElementById('vo-voice-note')
      var engineLabel = engine === 'cozy_voice' ? 'Cozy Voice 3' : 'Fish Audio'
      if(voiceSel.options.length === 0){
        var opt = document.createElement('option')
        opt.value = ''; opt.textContent = 'No voices found for ' + engineLabel
        voiceSel.appendChild(opt)
        note.textContent = d.error || (engine === 'fish_audio'
          ? 'Set FISH_AUDIO_API_KEY to list voices registered on the Fish Audio cloud API, or switch to "Upload a reference sample".'
          : 'No voices listed at COZY_VOICE_URL — switch to "Upload a reference sample" instead.')
      } else {
        note.textContent = d.source === 'cloud' ? (voiceSel.options.length + ' voice(s) from your Fish Audio account') : (voiceSel.options.length + ' voice(s) found')
      }
      var langSel = document.getElementById('vo-language-select')
      langSel.innerHTML = ''
      ;(d.languages || [{code:'auto', label:'Auto-detect (recommended)'}]).forEach(function(l){
        var opt = document.createElement('option')
        opt.value = l.code; opt.textContent = l.label
        langSel.appendChild(opt)
      })
    }).catch(function(e){
      var note = document.getElementById('vo-voice-note')
      if(note) note.textContent = 'Could not reach ' + (engine === 'cozy_voice' ? 'Cozy Voice 3' : 'Fish Audio') + ' to list voices: ' + e
      _voicesLoadedFor = null
    })
  }
  function previewVoiceover(){
    var text = document.getElementById('vo-text-input').value.trim()
    var status = document.getElementById('vo-preview-status')
    var audioEl = document.getElementById('vo-preview-audio')
    var btn = document.getElementById('vo-preview-btn')
    if(!text){ status.textContent = 'Type a narration script first.'; status.style.color = 'var(--amber)'; return }
    var source = document.querySelector('input[name=vo_voice_source]:checked').value
    var engine = document.querySelector('input[name=vo_engine]:checked').value
    var fd = new FormData()
    fd.append('text', text)
    fd.append('rate', document.getElementById('vo-rate-input').value || 175)
    fd.append('language', document.getElementById('vo-language-select').value || 'auto')
    fd.append('engine', engine)
    if(source === 'upload'){
      var f = document.querySelector('#vo-ref-upload-area input[type=file]').files[0]
      if(f) fd.append('ref_upload', f)
    } else if(source === 'registered'){
      fd.append('voice', document.getElementById('vo-voice-select').value || '')
    }
    btn.disabled = true; status.style.color = ''; status.textContent = 'Generating preview…'
    audioEl.style.display = 'none'
    fetch('/api/vo/preview', {method:'POST', body: fd}).then(r=>r.json()).then(function(d){
      if(d.ok){
        status.textContent = 'Preview ready.'
        audioEl.src = d.url
        audioEl.style.display = ''
        audioEl.play().catch(function(){})
      } else {
        status.textContent = 'Preview failed: ' + (d.error || 'unknown error')
        status.style.color = 'var(--amber)'
      }
    }).catch(function(e){
      status.textContent = 'Preview request failed: ' + e
      status.style.color = 'var(--amber)'
    }).finally(function(){ btn.disabled = false })
  }
  document.querySelector('select[name=genre]').addEventListener('change', updateGenreManualVisibility)
  document.querySelector('select[name=transition]').addEventListener('change', updateTransitionMatteVisibility)
  function updateTransitionMatteVisibility(){
    var el = document.getElementById('transition-matte-area')
    el.style.display = (document.querySelector('select[name=transition]').value === 'custom_matte') ? '' : 'none'
  }
  function updateGenreManualVisibility(){
    var custom = document.querySelector('select[name=genre]').value === ''
    var advanced = document.body.dataset.viewMode !== 'easy'
    document.querySelectorAll('.genre-manual').forEach(function(el){
      el.style.display = (custom && advanced) ? '' : 'none'
    })
    document.querySelectorAll('.genre-manual-transition').forEach(function(el){
      el.style.display = custom ? '' : 'none'
    })
    updateTransitionMatteVisibility()
  }
  function setViewMode(mode){
    document.body.dataset.viewMode = mode
    document.getElementById('view-easy-btn').classList.toggle('active', mode === 'easy')
    document.getElementById('view-adv-btn').classList.toggle('active', mode === 'advanced')
    document.querySelectorAll('#tf .adv-only').forEach(function(el){
      if(!el.classList.contains('genre-manual')) el.style.display = (mode === 'easy') ? 'none' : ''
    })
    updateGenreManualVisibility()
  }
  setViewMode('easy')
  // Grays out and disables generate-music / from-genre-sfx / tts-narration options
  // when the AI service(s) they need aren't reachable, falling back the selection
  // to "None" if it was the one currently chosen. Wired into runHealthCheck() below.
  function applyServiceGating(services){
    var status = {}
    services.forEach(function(s){ status[s.name] = s.status })
    // AI Vision (Ollama) is a hard dependency for every scoring mode (no fallback
    // exists) — if it's down, block submission outright rather than just
    // greying out one option.
    var ollamaUp = status.ollama === 'up'
    var banner = document.getElementById('ollama-down-banner')
    var submitBtn = document.getElementById('trailer-submit-btn')
    if(banner) banner.style.display = ollamaUp ? 'none' : ''
    if(submitBtn){
      submitBtn.disabled = !ollamaUp
      submitBtn.style.opacity = ollamaUp ? '1' : '.5'
      submitBtn.style.cursor = ollamaUp ? 'pointer' : 'not-allowed'
      submitBtn.title = ollamaUp ? '' : 'AI Vision is unreachable — start Ollama and re-check before generating.'
    }
    document.querySelectorAll('#tf input[data-requires]').forEach(function(inp){
      var reqs = inp.dataset.requires.split(',')
      var available = reqs.some(function(r){ return status[r] === 'up' })
      var label = inp.closest('label')
      inp.disabled = !available
      if(label){
        label.style.opacity = available ? '1' : '.4'
        label.style.cursor = available ? 'pointer' : 'not-allowed'
        label.title = available ? '' : 'Unreachable: ' + reqs.join(' / ') + '. Start that service, then hit "Check services" on the API tab.'
      }
      if(!available && inp.checked){
        inp.checked = false
        var noneRadio = document.querySelector('#tf input[name="' + inp.name + '"][value="none"]')
        if(noneRadio){ noneRadio.checked = true; noneRadio.dispatchEvent(new Event('change', {bubbles:true})) }
      }
      var noteId = {scoring_mode:'bgm-gating-note', sfx_mode:'sfx-gating-note', vo_mode:'vo-gating-note'}[inp.name]
      if(noteId){
        var note = document.getElementById(noteId)
        if(note) note.style.display = available ? 'none' : ''
      }
    })
    // Same idea, but for <select><option> pairs (e.g. VISION + STT needs whisper).
    document.querySelectorAll('#tf option[data-requires]').forEach(function(opt){
      var reqs = opt.dataset.requires.split(',')
      var available = reqs.some(function(r){ return status[r] === 'up' })
      opt.disabled = !available
      var select = opt.parentElement
      if(!available && opt.selected){
        opt.selected = false
        var fallback = Array.prototype.find.call(select.options, function(o){ return !o.disabled })
        if(fallback){ fallback.selected = true }
        select.dispatchEvent(new Event('change', {bubbles:true}))
      }
      if(select.id === 'scoring-mode-select'){
        var note = document.getElementById('mode-gating-note')
        if(note) note.style.display = available ? 'none' : ''
      }
    })
  }
  // Title/end card VO: preview player + set-in/set-out from playhead (or type the
  // seconds manually in the number inputs next to it - both write to the same fields).
  var cardVoObjectUrls = {}
  function cardVoFileChosen(input, prefix){
    var player = document.getElementById(prefix + '_player')
    var note = document.getElementById(prefix + '_preview_note')
    if(cardVoObjectUrls[prefix]) URL.revokeObjectURL(cardVoObjectUrls[prefix])
    if(input.files && input.files[0]){
      var url = URL.createObjectURL(input.files[0])
      cardVoObjectUrls[prefix] = url
      player.src = url
      player.style.display = ''
      note.textContent = 'Play the file, pause where you want the cut, then click Set in / Set out.'
    } else {
      player.removeAttribute('src')
      player.style.display = 'none'
      note.textContent = ''
    }
  }
  function cardVoSetPoint(prefix, which){
    var player = document.getElementById(prefix + '_player')
    var note = document.getElementById(prefix + '_preview_note')
    if(!player.src){ note.textContent = 'Choose an audio file first.'; return }
    var t = Math.round(player.currentTime * 10) / 10
    document.getElementById(prefix + (which === 'start' ? '_start' : '_end')).value = t
    note.textContent = (which === 'start' ? 'In' : 'Out') + ' point set to ' + t + 's from the player.'
  }
  </script>
  <div style="display:flex;gap:10px;align-items:center;margin-top:22px">
    <button class=btn type=submit id=trailer-submit-btn>Generate Episodic Promo Plug</button>
    <button type=button id=tr-cancel-btn class=btn style="display:none;background:var(--danger,#c94f4f)">Cancel</button>
  </div>
</form>
</div>

<script>
document.getElementById('tf').addEventListener('submit', async function(e){
  e.preventDefault()
  document.getElementById('tr-area').style.display='none'
  document.getElementById('tr-prompt').style.display='none'
  var progArea=document.getElementById('tr-progress-area')
  var progBar=document.getElementById('tr-progress-bar')
  var progPct=document.getElementById('tr-progress-pct')
  var progStep=document.getElementById('tr-progress-step')
  var cancelBtn = document.getElementById('tr-cancel-btn')
  progArea.style.display='block'
  progBar.style.width='0%'; progPct.textContent='0%'; progStep.textContent='Starting...'
  cancelBtn.style.display='inline-block'; cancelBtn.disabled=false

  let startResp
  try{
    startResp = await fetch('/api/trailer/generate',{method:'POST',body:new FormData(this)})
  }catch(err){
    progArea.style.display='none'
    cancelBtn.style.display='none'
    document.getElementById('tr-prompt').style.display='block'
    document.getElementById('tr-prompt').textContent='Upload failed: '+err
    return
  }
  let startData = await startResp.json()
  if(startData.error){
    progArea.style.display='none'
    cancelBtn.style.display='none'
    document.getElementById('tr-stats').innerHTML='<b>Error:</b> '+startData.error
    document.getElementById('tr-area').style.display='block'
    return
  }
  let jobId = startData.job_id
  let cancelled = false
  cancelBtn.onclick = async function(){
    cancelled = true
    cancelBtn.disabled = true
    try{ await fetch('/api/trailer/cancel/'+jobId, {method:'POST'}) }catch(err){}
  }

  let d = await new Promise((resolve)=>{
    let poll = async ()=>{
      if(cancelled){ resolve({error:'Cancelled'}); return }
      let r
      try{ r = await fetch('/api/trailer/progress/'+jobId) }
      catch(err){ setTimeout(poll, 1200); return }
      let j = await r.json()
      if(j.error){ resolve({error: j.error}); return }
      let pct = j.percent||0
      progBar.style.width=pct+'%'; progPct.textContent=pct+'%'
      progStep.textContent=j.step||'Working...'
      if(j.done){ resolve(j.result || {error:'Job finished with no result.'}); return }
      setTimeout(poll, 800)
    }
    poll()
  })
  cancelBtn.disabled = false
  cancelBtn.style.display='none'

  progArea.style.display='none'
  if(d.error){document.getElementById('tr-stats').innerHTML='<b>Error:</b> '+d.error; document.getElementById('tr-area').style.display='block'; return}
  var srcLabel=function(s){return {woosh:'AI-generated (Woosh)',ai_generated:'AI-generated (ACE-Step)',uploaded:'uploaded',synth_fallback:'placeholder synth (Woosh/ACE-Step unavailable)',tts:'text-to-speech',none:'none'}[s]||s}
  var audioNote=''
  if(d.bgm_source && d.bgm_source!=='none') audioNote+=' | Music: '+srcLabel(d.bgm_source)+(d.sync_beats?' (beat-synced cuts)':'')
  if(d.sfx_source && d.sfx_source!=='none') audioNote+=' | SFX: '+srcLabel(d.sfx_source)
  if(d.vo_source && d.vo_source!=='none') audioNote+=' | VO: '+srcLabel(d.vo_source)
  if(d.whisper_enhance) audioNote+=' | Scene selection: dialogue-enhanced'
  var fallbackUsed=(d.bgm_source==='synth_fallback'||d.sfx_source==='synth_fallback')
  document.getElementById('tr-stats').innerHTML='Trailer: '+d.trailer_duration+'s (target '+d.trailer_length+'s) from '+d.selected_scenes+'/'+d.total_scenes+' scenes | Raw video: '+d.video_duration+'s'+audioNote+(fallbackUsed?'<br><small style="color:var(--amber)">Note: AI music/SFX service was unavailable, so a lower-fidelity generated placeholder was used instead.</small>':'')+(d.vo_error?'<br><small style="color:var(--amber)">Voiceover note: '+d.vo_error+'</small>':'')+(d.hook_line?'<br><small>Suggested hook line ('+d.hook_line.start+'s–'+d.hook_line.end+'s): &ldquo;'+d.hook_line.text+'&rdquo;</small>':'')
  var dlFilename = d.trailer_url.split('/').pop()
  var dlName = d.orig_name
  var dlUrl = function(fmt){ return '/download/'+dlFilename+'?name='+encodeURIComponent(dlName)+'&format='+fmt }
  document.getElementById('tr-video').innerHTML='<video controls style=max-width:100%;border-radius:8px><source src="'+d.trailer_url+'" type="video/mp4"></video>'+
    '<div style="display:flex;gap:10px;align-items:center;margin-top:8px;flex-wrap:wrap">'+
    '<select id=tr-export-format style="width:auto">'+
    '<option value=mp4_high selected>MP4 (H.264 High Profile)</option>'+
    '<option value=prores_hq_2997>Apple ProRes 422 HQ — 29.97fps</option>'+
    '<option value=prores_hq_2398>Apple ProRes 422 HQ — 23.976fps</option>'+
    '<option value=avci100i>AVC-Intra 100i (approximation)</option>'+
    '</select>'+
    '<a href="'+dlUrl('mp4_high')+'" id=tr-download-link class="btn" style="display:inline-block;text-decoration:none">Download</a>'+
    '</div>'
  document.getElementById('tr-export-format').addEventListener('change', function(){
    document.getElementById('tr-download-link').href = dlUrl(this.value)
  })
  let rows=''
  d.scenes.forEach(s=>{rows+='<tr><td>'+s.scene+'</td><td>'+s.start+'s</td><td>'+s.end+'s</td><td>'+s.quality+'</td><td>'+s.duration+'</td><td>'+s.description+'</td></tr>'})
  document.getElementById('tr-table').innerHTML='<tr><th>#</th><th>Start</th><th>End</th><th>Score</th><th>Used</th><th>Description</th></tr>'+rows
  document.getElementById('tr-area').style.display='block'
})
</script>


<script>
// Load AI Vision models into dropdowns
async function loadModels(){
  try{
    let r=await fetch('/api/vision/models')
    let d=await r.json()
    let opts=d.models.map(m=>'<option value="'+m+'">'+m+'</option>').join('')
    if(!opts) opts='<option value="qwen3-vl:8b" selected>qwen3-vl:8b</option><option value="qwen2.5vl:7b">qwen2.5vl:7b</option>'
    document.getElementById('vision-model').innerHTML=opts
    document.getElementById('trailer-model').innerHTML=opts
  }catch(e){
    document.getElementById('vision-model').innerHTML='<option value="qwen3-vl:8b" selected>qwen3-vl:8b</option><option value="qwen2.5vl:7b">qwen2.5vl:7b</option>'
    document.getElementById('trailer-model').innerHTML=document.getElementById('vision-model').innerHTML
  }
}
loadModels()

document.getElementById('vf').addEventListener('submit', async function(e){
  e.preventDefault()
  document.getElementById('vr-area').style.display='none'
  document.getElementById('vr-prompt').style.display='block'
  document.getElementById('vr-prompt').textContent='Analyzing...'
  let r=await fetch('/api/vision/analyze',{method:'POST',body:new FormData(this)})
  let d=await r.json()
  document.getElementById('vr-prompt').style.display='none'
  if(d.error){document.getElementById('vr-result').innerHTML='<b>Error:</b> '+d.error; document.getElementById('vr-area').style.display='block'; return}
  let h='<table><tr><th>#</th><th>Time</th><th>Scene</th><th>AI Response</th></tr>'
  d.results.forEach(r=>{
    let sc=r.scene?'S'+r.scene+' ('+r.scene_start+'-'+r.scene_end+', '+r.scene_duration+'s)':'-'
    h+='<tr><td>'+r.frame_idx+'</td><td>'+r.time_sec+'s</td><td>'+sc+'</td><td>'+r.ollama_response+'</td></tr>'
  })
  h+='</table>'
  document.getElementById('vr-result').innerHTML='<p>Analyzed '+d.frames_analyzed+' frames across '+d.total_scenes+' scenes.</p>'+h
  document.getElementById('vr-area').style.display='block'
})
</script>

<!-- API -->
<div id="p-api" class="panel">
<h2>API</h2>
<p><strong>Health check</strong>:</p>
<div class=card>
  <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:10px">
    <button type=button id="health-check-btn" class=btn onclick="runHealthCheck()">Check services</button>
    <span id="health-summary" style="font-size:12px;opacity:.8"></span>
  </div>
  <div id="health-results" class=info></div>
</div>
<pre>GET /api/health</pre>
<p style="font-size:12px;opacity:.8">Pings Ollama, Fish Audio S2, Cozy Voice 3, faster-whisper, ACE-Step, and Woosh and reports whether each is reachable, with response latency. Doesn't call any generation endpoints, so it's safe (and cheap) to run anytime.</p>
<p><code>POST multipart/form-data</code> with <code>file</code>:</p>
<div class=card>
<div class=info>
  <div class=info-item><strong>POST</strong> /api/opencv/info</div>
  <div class=info-item><strong>POST</strong> /api/opencv/analyze<br><small>+ num_frames</small></div>
  <div class=info-item><strong>POST</strong> /api/scenedetect/detect<br><small>+ threshold, min_scene_len</small></div>
</div>
</div>
<p><strong>Player</strong>:</p>
<pre>POST /api/playback/start  + file/url
GET  /api/playback/stream/raw|gray|edges|hsv|blur|face|motion
GET  /api/playback/pause
GET  /api/playback/resume
GET  /api/playback/seek?t=seconds
GET  /api/playback/stop</pre>
<p><strong>Narration (Fish Audio S2 or Cozy Voice 3)</strong>:</p>
<pre>POST /api/trailer/generate  + vo_mode=tts &amp; vo_text=&lt;script&gt; &amp; vo_engine=fish_audio|cozy_voice
GET  /api/voices?engine=fish_audio|cozy_voice     list that engine's voices + languages
POST /api/vo/preview  + text, rate, language, engine, voice or ref_upload</pre>
<p style="font-size:12px;opacity:.8">
<code>vo_engine</code> picks the narration engine (defaults to <code>fish_audio</code>).
There's no bundled default voice — every narration job needs either a voice picked from
<code>/api/voices</code> or an uploaded reference sample to clone zero-shot (sent as a
base64 <code>references</code>/<code>reference_audio</code> entry so speech is generated
in that voice, no pre-registration required). The Narration section fetches the current
engine's voice list live and reloads it if you switch engines; Fish Audio's list is only
populated with <code>FISH_AUDIO_API_KEY</code> set (cloud API) or if your self-hosted
server happens to expose its own model-listing endpoint, and Cozy Voice 3's list depends
on <code>COZY_VOICE_URL</code>/voices being reachable — if a list comes back empty,
upload a reference sample instead. You can also override the auto-detected language, and
generate a short preview via <code>/api/vo/preview</code> before running the full trailer
job. Configured via env vars:
</p>
<div class=card>
<div class=info>
  <div class=info-item><strong>FISH_AUDIO_URL</strong><br><small>self-hosted or api.fish.audio/v1/tts</small></div>
  <div class=info-item><strong>FISH_AUDIO_API_KEY</strong><br><small>blank for self-hosted</small></div>
  <div class=info-item><strong>FISH_AUDIO_MODEL</strong><br><small>s2.1-pro-free (cloud only)</small></div>
  <div class=info-item><strong>COZY_VOICE_URL</strong><br><small>local Cozy Voice 3 server, default localhost:8040</small></div>
</div>
</div>
</div>

<div id="p-docs" class="panel">
<h2>Docs</h2>

<h3 style="margin-top:0">Workflow</h3>
<p>What happens, in order, from upload to finished export:</p>
<div style="overflow-x:auto;margin:14px 0 18px;background:var(--sunken);border:1px solid var(--line);border-radius:var(--radius);padding:16px">
<svg viewBox="0 0 1008 308" style="min-width:820px;width:100%;height:auto;font-family:'JetBrains Mono',monospace">
<defs>
  <marker id="dg-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="var(--phosphor)"/>
  </marker>
  <marker id="dg-arrow-amber" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="var(--amber)"/>
  </marker>
</defs>
<rect x="210" y="4" width="388" height="44" rx="8" fill="none" stroke="var(--amber)" stroke-width="1.5" stroke-dasharray="5,4"/>
<text x="404.0" y="30.0" text-anchor="middle" font-size="12" fill="var(--amber)">Beat / Dialogue Prep (optional)</text>
<line x1="304.0" y1="48" x2="304.0" y2="92" stroke="var(--amber)" stroke-width="1.5" stroke-dasharray="4,4" marker-end="url(#dg-arrow-amber)"/>
<line x1="504.0" y1="48" x2="504.0" y2="92" stroke="var(--amber)" stroke-width="1.5" stroke-dasharray="4,4" marker-end="url(#dg-arrow-amber)"/>
<rect x="20" y="92" width="168" height="56" rx="8" fill="var(--elevated)" stroke="var(--line)" stroke-width="1.5"/>
<text x="104.0" y="116.0" text-anchor="middle" font-size="12.5" fill="var(--ink)">Scene</text>
<text x="104.0" y="133.0" text-anchor="middle" font-size="12.5" fill="var(--ink)">Detection</text>
<rect x="220" y="92" width="168" height="56" rx="8" fill="var(--elevated)" stroke="var(--line)" stroke-width="1.5"/>
<text x="304.0" y="125.0" text-anchor="middle" font-size="12.5" fill="var(--ink)">Scoring</text>
<rect x="420" y="92" width="168" height="56" rx="8" fill="var(--elevated)" stroke="var(--line)" stroke-width="1.5"/>
<text x="504.0" y="116.0" text-anchor="middle" font-size="12.5" fill="var(--ink)">Scene</text>
<text x="504.0" y="133.0" text-anchor="middle" font-size="12.5" fill="var(--ink)">Selection</text>
<rect x="620" y="92" width="168" height="56" rx="8" fill="var(--elevated)" stroke="var(--line)" stroke-width="1.5"/>
<text x="704.0" y="125.0" text-anchor="middle" font-size="12.5" fill="var(--ink)">Assembly</text>
<rect x="820" y="92" width="168" height="56" rx="8" fill="var(--elevated)" stroke="var(--line)" stroke-width="1.5"/>
<text x="904.0" y="116.0" text-anchor="middle" font-size="12.5" fill="var(--ink)">Audio</text>
<text x="904.0" y="133.0" text-anchor="middle" font-size="12.5" fill="var(--ink)">Normalize</text>
<rect x="820" y="232" width="168" height="56" rx="8" fill="var(--elevated)" stroke="var(--line)" stroke-width="1.5"/>
<text x="904.0" y="265.0" text-anchor="middle" font-size="12.5" fill="var(--ink)">SFX at Cuts</text>
<rect x="620" y="232" width="168" height="56" rx="8" fill="var(--elevated)" stroke="var(--line)" stroke-width="1.5"/>
<text x="704.0" y="256.0" text-anchor="middle" font-size="12.5" fill="var(--ink)">Background</text>
<text x="704.0" y="273.0" text-anchor="middle" font-size="12.5" fill="var(--ink)">Music</text>
<rect x="420" y="232" width="168" height="56" rx="8" fill="var(--elevated)" stroke="var(--amber)" stroke-width="1.5" stroke-dasharray="5,4"/>
<text x="504.0" y="256.0" text-anchor="middle" font-size="12.5" fill="var(--ink)">Narration</text>
<text x="504.0" y="273.0" text-anchor="middle" font-size="12.5" fill="var(--ink)">(optional)</text>
<rect x="220" y="232" width="168" height="56" rx="8" fill="var(--elevated)" stroke="var(--line)" stroke-width="1.5"/>
<text x="304.0" y="265.0" text-anchor="middle" font-size="12.5" fill="var(--ink)">Final Mix</text>
<rect x="20" y="232" width="168" height="56" rx="8" fill="var(--elevated)" stroke="var(--line)" stroke-width="1.5"/>
<text x="104.0" y="265.0" text-anchor="middle" font-size="12.5" fill="var(--ink)">Export</text>
<line x1="188" y1="120.0" x2="214" y2="120.0" stroke="var(--phosphor)" stroke-width="2" marker-end="url(#dg-arrow)"/>
<line x1="388" y1="120.0" x2="414" y2="120.0" stroke="var(--phosphor)" stroke-width="2" marker-end="url(#dg-arrow)"/>
<line x1="588" y1="120.0" x2="614" y2="120.0" stroke="var(--phosphor)" stroke-width="2" marker-end="url(#dg-arrow)"/>
<line x1="788" y1="120.0" x2="814" y2="120.0" stroke="var(--phosphor)" stroke-width="2" marker-end="url(#dg-arrow)"/>
<line x1="904.0" y1="148" x2="904.0" y2="226" stroke="var(--phosphor)" stroke-width="2" marker-end="url(#dg-arrow)"/>
<line x1="820" y1="260.0" x2="794" y2="260.0" stroke="var(--phosphor)" stroke-width="2" marker-end="url(#dg-arrow)"/>
<line x1="620" y1="260.0" x2="594" y2="260.0" stroke="var(--phosphor)" stroke-width="2" marker-end="url(#dg-arrow)"/>
<line x1="420" y1="260.0" x2="394" y2="260.0" stroke="var(--phosphor)" stroke-width="2" marker-end="url(#dg-arrow)"/>
<line x1="220" y1="260.0" x2="194" y2="260.0" stroke="var(--phosphor)" stroke-width="2" marker-end="url(#dg-arrow)"/>
</svg>
<p style="margin:8px 2px 0;font-size:11px;color:var(--ink-dim)">Amber dashed = optional, feeds into the main pipeline rather than sitting inline with it.</p>
</div>
<ol style="line-height:1.9">
  <li><strong>Scene detection</strong> — PySceneDetect (ContentDetector) finds every cut in the source video, using the same sensitivity/minimum-length settings as the "Preview scene cuts" tool so what you preview is what actually gets used.</li>
  <li><strong>Scoring</strong> — each scene is scored on sharpness/brightness/duration/face presence (OpenCV), boosted by AI Vision's 1-5 quality rating, and boosted further for scenes with quotable dialogue when VISION + STT is selected (faster-whisper transcription).</li>
  <li><strong>Beat/dialogue prep (optional)</strong> — if "sync cuts to beat" is on, a music track is prepared early and its beat grid drives cut timing; if dialogue transcription is on, word-level timestamps are generated for clean-cut alignment.</li>
  <li><strong>Scene selection</strong> — highest-scoring scenes are picked to fill the target duration, respecting min/max clip length and a minimum spacing between picks so the trailer isn't built from one over-represented stretch of the source, snapping cut points to the beat grid and/or word boundaries when enabled, and topping up across a few passes to land within ~0.5s of the target length.</li>
  <li><strong>Assembly</strong> — selected scenes plus title/end cards are concatenated with the genre's signature transition (see table below), or a custom transition driven by an uploaded matte video/image if selected; any clip that fails to extract is dropped and the reported scene count reflects what actually made it in.</li>
  <li><strong>Audio level normalization</strong> — SOT is loudness-normalized immediately after assembly so every later step works off a predictable baseline.</li>
  <li><strong>SFX at cuts</strong> — Woosh (Sony AI) is tried first, then ACE-Step, then a procedural synth fallback, using each genre's SFX prompt.</li>
  <li><strong>Background music</strong> — ACE-Step-generated or uploaded, normalized, and ducked under SOT.</li>
  <li><strong>Narration (optional)</strong> — Fish Audio S2 (self-hosted or cloud) or Cozy Voice 3 generates speech from a script using a voice picked from that engine's live voice list or an uploaded reference sample cloned zero-shot, or an uploaded VO track is used instead; either way it's loudness-normalized before mixing.</li>
  <li><strong>Final audio mix</strong> — if a voiceover is present, SOT ducks to near-silence under it while BGM only ducks a moderate amount; if "broadcast dual-mono" is on, the result is collapsed so every element is identically audible in both channels; a final loudness/true-peak pass targets your configured broadcast spec.</li>
  <li><strong>Export</strong> — download as MP4 (H.264 High Profile), Apple ProRes 422 HQ (29.97 or 23.976fps), or an AVC-Intra 100i approximation.</li>
</ol>
<p>Jobs run through a concurrency-limited queue (default 2 at a time, configurable via <code>/api/queue/limit</code>) so multiple simultaneous requests don't overload the local model servers (ACE-Step, Woosh, Ollama, Fish Audio, Cozy Voice 3).</p>

<h3>Genre presets</h3>
<p>Each genre has a signature transition + crossfade duration, whether SFX-at-cuts is on by default, and the theme used for AI-generated music/SFX prompts.</p>
<table>
<tr><th>Genre</th><th>Transition</th><th>Crossfade</th><th>SFX at cuts</th><th>Music theme</th><th>SFX theme</th></tr>
{% for g in genre_rows %}
<tr>
  <td>{{ g.genre }}</td>
  <td>{{ g.transition }}</td>
  <td>{{ g.xfade_dur }}s</td>
  <td>{{ 'yes' if g.sfx else 'no' }}</td>
  <td style="max-width:260px">{{ g.music_theme }}</td>
  <td style="max-width:220px">{{ g.sfx_theme }}</td>
</tr>
{% endfor %}
</table>
</div>

</div><!-- container -->

<script>
// service health check
function runHealthCheck(){
  var btn=document.getElementById('health-check-btn')
  var summary=document.getElementById('health-summary')
  var results=document.getElementById('health-results')
  btn.disabled=true; btn.textContent='Checking...'
  summary.textContent=''
  fetch('/api/health').then(r=>r.json()).then(d=>{
    var up=d.services.filter(s=>s.status==='up').length
    summary.textContent=up+'/'+d.services.length+' services up'
    summary.style.color=d.ok?'var(--phosphor)':'var(--amber)'
    results.innerHTML=d.services.map(function(s){
      var color=s.status==='up'?'var(--phosphor)':'var(--amber)'
      var detail=s.status==='up'?(s.latency_ms+'ms (HTTP '+s.http_status+')'):(s.error||'unreachable')
      return '<div class="info-item"><strong style="color:'+color+'">&#9679;</strong> '+s.name+'<br><small>'+detail+'</small></div>'
    }).join('')
    if(typeof applyServiceGating === 'function') applyServiceGating(d.services)
    var engineSel = document.querySelector('input[name=vo_engine]:checked')
    var currentEngine = engineSel ? engineSel.value : 'fish_audio'
    var engineSvc = (d.services || []).find(function(s){ return s.name === currentEngine })
    if(engineSvc && engineSvc.status === 'up' && typeof loadVoices === 'function') loadVoices()
  }).catch(function(e){
    summary.textContent='Health check failed: '+e
    summary.style.color='var(--amber)'
  }).finally(function(){
    btn.disabled=false; btn.textContent='Check services'
  })
}
document.querySelector('.tab[onclick*="p-api"]').addEventListener('click', function(){
  if(!document.getElementById('health-results').innerHTML) runHealthCheck()
})
// Run once automatically at load so the "Generate music" / "From genre" SFX /
// "Narration (AI voice)" options are already greyed out if their service is
// down, before the person even opens the API tab.
document.addEventListener('DOMContentLoaded', runHealthCheck)
</script>

<script>
// playback wire-up
['raw','gray','edges','hsv','blur','face','motion'].forEach(m=>{
  document.getElementById('pb-'+m).addEventListener('click',()=>{
    document.getElementById('pb-feed').src='/api/playback/stream/'+m
    ;['raw','gray','edges','hsv','blur','face','motion'].forEach(x=>{
      document.getElementById('pb-'+x).classList.toggle('active-filter', x===m)
    })
  })
})
document.getElementById('pb-playbtn').addEventListener('click',async function(){
  let b=this
  if(b.textContent=='Pause'){await fetch('/api/playback/pause');b.textContent='Play'}
  else{await fetch('/api/playback/resume');b.textContent='Pause'}
})
document.getElementById('pb-stopbtn').addEventListener('click',async function(){
  await fetch('/api/playback/stop')
  document.getElementById('pb-feed').src=''
  document.getElementById('pb-area').style.display='none'
  document.getElementById('pb-prompt').style.display='block'
})
document.getElementById('pb-progress').addEventListener('input',function(){
  fetch('/api/playback/seek?t='+this.value)
})

document.getElementById('pf').addEventListener('submit', async function(e){
  e.preventDefault()
  let r=await fetch('/api/playback/start',{method:'POST',body:new FormData(this)})
  let d=await r.json()
  if(d.error){alert(d.error);return}
  document.getElementById('pb-area').style.display='block'
  document.getElementById('pb-prompt').style.display='none'
  document.getElementById('pb-info').innerHTML=
    `<div class=info-item><strong>W:</strong> ${d.width}</div>
     <div class=info-item><strong>H:</strong> ${d.height}</div>
     <div class=info-item><strong>FPS:</strong> ${d.fps}</div>
     <div class=info-item><strong>Dur:</strong> ${d.duration_sec}s</div>`
  document.getElementById('pb-feed').src='/api/playback/stream/raw'
  document.getElementById('pb-progress').max=Math.floor(d.duration_sec)
  document.getElementById('pb-playbtn').textContent='Pause'
  ;['raw','gray','edges','hsv','blur','face','motion'].forEach(x=>{
    document.getElementById('pb-'+x).classList.toggle('active-filter', x==='raw')
  })
})
</script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(UI, genre_rows=GENRE_DOCS_ROWS)

@app.route('/upload', methods=['POST'])
def upload():
    path, err = load_video(request)
    if not path:
        return render_template_string(UI, r={'error': err}, genre_rows=GENRE_DOCS_ROWS)
    info = get_video_info(path)
    cap = cv2.VideoCapture(path)
    total = info['total_frames']
    step = max(total // 10, 1)
    frames = []
    for i in range(0, total, step):
        if len(frames) >= 10:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, f = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=20, qualityLevel=0.01, minDistance=10)
        frames.append({'idx': i, 'shape': list(f.shape),
                       'mean_bgr': [round(float(c), 1) for c in cv2.mean(f)[:3]],
                       'brightness': round(float(np.mean(gray)), 1),
                       'edge_pixels': int(np.sum(edges > 0)),
                       'corners': len(corners) if corners is not None else 0})
    cap.release()
    scene_list = detect_scenes(path, threshold=30.0)
    scenes = [{'scene': i+1, 'start': s.get_timecode(), 'end': e.get_timecode(),
               'start_sec': round(s.get_seconds(), 2), 'end_sec': round(e.get_seconds(), 2),
               'duration': round(e.get_seconds() - s.get_seconds(), 2)}
              for i, (s, e) in enumerate(scene_list)]
    return render_template_string(UI, r={'info': info, 'frames': frames, 'scenes': scenes}, genre_rows=GENRE_DOCS_ROWS)

@app.route('/uploads/<filename>')
def uploaded(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/download/<filename>')
def download_file(filename):
    orig = request.args.get('name', filename)
    fmt_key = request.args.get('format', 'mp4_high')
    base_name, _ = os.path.splitext(orig)

    if fmt_key not in EXPORT_FORMATS:
        return jsonify(error=f'Unknown export format: {fmt_key}'), 400

    src_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(src_path):
        return jsonify(error='File not found'), 404

    ext = EXPORT_FORMATS[fmt_key]['ext']
    cache_name = f'{os.path.splitext(filename)[0]}_{fmt_key}.{ext}'
    cache_path = os.path.join(app.config['UPLOAD_FOLDER'], cache_name)
    if not (os.path.exists(cache_path) and os.path.getsize(cache_path) > 0):
        cmd = build_export_cmd(src_path, cache_path, fmt_key)
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if not (os.path.exists(cache_path) and os.path.getsize(cache_path) > 0):
            return jsonify(error=f'Export to {fmt_key} failed: {r.stderr[-800:]}'), 500

    resp = send_from_directory(app.config['UPLOAD_FOLDER'], cache_name)
    resp.headers['Content-Disposition'] = f'attachment; filename="{base_name}.{ext}"'
    return resp

if __name__ == '__main__':
    print(' * Server starting...')
    print(' * HTTP:  http://0.0.0.0:5000/')
    print(' * Access from local machine: http://localhost:5000/')
    print(' * Access from other devices: http://YOUR_IP:5000/')
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
