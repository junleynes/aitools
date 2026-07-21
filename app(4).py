import os, cv2, numpy as np, tempfile, urllib.request, threading, time, pathlib, base64, json, requests, subprocess, shutil
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from flask import Flask, render_template_string, request, send_from_directory, jsonify, Response
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2GB
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'}
ACE_STEP_URL = 'http://localhost:8001'

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
    if 'video_url' in req.form and req.form['video_url'].strip():
        url = req.form['video_url'].strip()
        fn = secure_filename(url.split('/')[-1] or 'video.mp4')
        path = os.path.join(app.config['UPLOAD_FOLDER'], fn)
        urllib.request.urlretrieve(url, path)
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
    video = open_video(path)
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=th))
    sm.detect_scenes(video)
    scenes = [{'scene': i+1, 'start': s.get_timecode(), 'end': e.get_timecode(),
               'start_sec': round(s.get_seconds(), 2), 'end_sec': round(e.get_seconds(), 2),
               'duration': round(e.get_seconds() - s.get_seconds(), 2)}
              for i, (s, e) in enumerate(sm.get_scene_list())]
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

    video = open_video(path)
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=30.0))
    sm.detect_scenes(video)
    scene_list = sm.get_scene_list()
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

# Progress tracking: the browser generates a job_id and submits it with the
# generate request, then polls this dict via /api/trailer/progress/<job_id>
# while the (synchronous) generate request is in flight. Flask runs threaded,
# so the GET poll is served concurrently on its own thread.
TRAILER_PROGRESS = {}
_progress_lock = threading.Lock()

def _set_progress(job_id, percent, message, done=False, error=False):
    if not job_id:
        return
    with _progress_lock:
        TRAILER_PROGRESS[job_id] = {
            'percent': percent, 'message': message,
            'done': done, 'error': error,
        }

@app.route('/api/trailer/progress/<job_id>')
def api_trailer_progress(job_id):
    with _progress_lock:
        p = TRAILER_PROGRESS.get(job_id)
    if not p:
        return jsonify(percent=0, message='Waiting to start...', done=False, error=False)
    return jsonify(**p)

@app.route('/api/trailer/generate', methods=['POST'])
def api_trailer():
    job_id = request.form.get('job_id', '').strip()
    _set_progress(job_id, 2, 'Loading video...')
    path, orig_name = load_video(request)
    if not path:
        _set_progress(job_id, 100, orig_name, done=True, error=True)
        return jsonify(error=orig_name), 400

    mode = request.form.get('mode', 'auto')
    genre = request.form.get('genre', '').strip()
    scoring_mode = request.form.get('scoring_mode', 'generate')
    trailer_length = int(request.form.get('trailer_length', 15))
    if trailer_length not in (15, 30, 45, 60):
        trailer_length = 30
    transition = request.form.get('transition', 'fade')
    VALID_TRANSITIONS = {'fade','fadeblack','fadewhite','fadefast','fadegrays',
        'wipeleft','wiperight','wipeup','wipedown',
        'slideleft','slideright','slideup','slidedown',
        'smoothleft','smoothright','smoothup','smoothdown',
        'circlecrop','rectcrop','circleopen','circleclose',
        'distance','pixelize','diagtl','diagtr','diagbl','diagbr',
        'hlslice','hrslice','vuslice','vdslice',
        'radial','zoomin','dissolve','hblur','squeezev','squeezeh',
        'horzopen','horzclose','vertopen','vertclose'}
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
    target_loudness = float(request.form.get('target_loudness', -16))
    true_peak = float(request.form.get('true_peak', -1.5))
    beat_match = request.form.get('beat_match') == 'on'
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

    # J/L cut: bleed a bit of the incoming scene's audio early (J) or let the
    # outgoing scene's audio linger past the cut (L). 'none' keeps the plain
    # symmetric audio crossfade that was already used at every transition.
    jl_cut = request.form.get('jl_cut', 'none')
    if jl_cut not in ('none', 'j', 'l', 'both'):
        jl_cut = 'none'
    try:
        jl_offset = float(request.form.get('jl_offset', 0.35))
    except ValueError:
        jl_offset = 0.35
    jl_offset = max(0.1, min(1.5, jl_offset))

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

    prompt = request.form.get('prompt',
        'Describe this scene in 3-5 words, then rate it 1-5 for a movie trailer. Format: DESC: <words> | SCORE: <digit>')

    _set_progress(job_id, 5, 'Reading video info...')
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
        err = f'Video is only {video_duration:.1f}s long, but a {trailer_length}s trailer requires at least {min_required:.1f}s of raw video. Upload a longer video or select a shorter trailer length.'
        _set_progress(job_id, 100, err, done=True, error=True)
        return jsonify(error=err), 400

    # Measure card durations before selecting scenes
    card_files = []
    card_durations = []
    if end_card_path and os.path.exists(end_card_path):
        card_files.append(end_card_path)
    if schedule_card_path and os.path.exists(schedule_card_path):
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

    # Detect scenes via PySceneDetect
    _set_progress(job_id, 10, 'Detecting scene boundaries...')
    video = open_video(path)
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=30.0))
    sm.detect_scenes(video)
    scene_list = sm.get_scene_list()
    if not scene_list:
        _set_progress(job_id, 100, 'No scene changes detected.', done=True, error=True)
        return jsonify(error='No scene changes detected. Try a video with clear cuts.'), 400

    # Score scenes
    _set_progress(job_id, 18, f'Scoring {len(scene_list)} scenes (sharpness, brightness, faces)...')
    from statistics import median
    scenes_data = []
    cap = cv2.VideoCapture(path)
    for scene_i, (start, end) in enumerate(scene_list):
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
        if scene_i % 5 == 0 or scene_i == len(scene_list) - 1:
            pct = 18 + int(22 * (scene_i + 1) / len(scene_list))
            _set_progress(job_id, pct, f'Scoring scenes... ({scene_i + 1}/{len(scene_list)})')
    cap.release()
    if not scenes_data:
        _set_progress(job_id, 100, 'No frames could be read.', done=True, error=True)
        return jsonify(error='No frames could be read.'), 400

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
        s['quality_score'] = score

    if mode == 'ai':
        _set_progress(job_id, 42, f'Asking Ollama Vision to rate {len(scenes_data)} scenes...')
        for ai_i, s in enumerate(scenes_data):
            if ai_i % 3 == 0:
                _set_progress(job_id, 42 + int(13 * (ai_i + 1) / len(scenes_data)),
                              f'AI scoring scenes... ({ai_i + 1}/{len(scenes_data)})')
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
                s['ai_desc'] = desc_m.group(1).strip() if desc_m else ''
                s['total_score'] = s['quality_score'] + (int(score_m.group(1)) if score_m else 3)
            except:
                s['total_score'] = s['quality_score'] + 3
    else:
        for s in scenes_data:
            s['total_score'] = s['quality_score']

    # Pick top scenes by score to fill target, then sort by timecode
    # Iterative: xfade transitions shorten output, so compensate
    trailer_duration = base_target
    for pass_attempt in range(3):
        scenes_data.sort(key=lambda x: x['total_score'], reverse=True)
        selected = []
        total_sel = 0
        for s in scenes_data:
            if total_sel >= trailer_duration:
                break
            seg_dur = min(s['duration'], trailer_duration - total_sel)
            s['selected_dur'] = seg_dur
            selected.append(s)
            total_sel += seg_dur
        selected.sort(key=lambda x: x['start'])

        n_seg = len(selected) + len(card_files)
        xfade_loss = max(0, (n_seg - 1)) * xfade_dur
        expected_total = total_sel + total_card_dur - xfade_loss
        shortfall = trailer_length - expected_total
        if abs(shortfall) <= 0.5 or pass_attempt == 2:
            break
        trailer_duration = total_sel + shortfall * 1.15

    if not selected:
        _set_progress(job_id, 100, 'No scenes selected.', done=True, error=True)
        return jsonify(error='No scenes selected.'), 400

    # Extract selected segments + card videos as temp files
    _set_progress(job_id, 58, f'Extracting {len(selected)} selected segments...')
    seg_files = []
    base_ts = int(time.time())
    for seg_i, seg in enumerate(selected):
        _set_progress(job_id, 58 + int(12 * (seg_i + 1) / len(selected)),
                      f'Extracting segments... ({seg_i + 1}/{len(selected)})')
        out_seg = os.path.join(app.config['UPLOAD_FOLDER'], f'seg_{base_ts}_{seg_i}.mp4')
        r = subprocess.run([FFMPEG, '-y', '-ss', str(seg['start']), '-i', path,
                            '-t', str(seg['selected_dur']),
                            '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '28', '-pix_fmt', 'yuv420p',
                            '-c:a', 'aac', '-b:a', '128k', out_seg], capture_output=True, text=True)
        if os.path.exists(out_seg) and os.path.getsize(out_seg) > 0:
            seg_files.append(out_seg)
        elif r.returncode != 0:
            print(f'FFMPEG seg extraction error: {r.stderr[:500]}')

    all_inputs = seg_files + card_files
    n_total = len(all_inputs)

    out_path = os.path.join(app.config['UPLOAD_FOLDER'], f'trailer_{base_ts}.mp4')
    sfx_timestamps = []
    _set_progress(job_id, 72, 'Rendering trailer with transitions...')

    norm = (f'scale={src_w}:{src_h}:force_original_aspect_ratio=decrease,'
            f'pad={src_w}:{src_h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={src_fps}')

    if n_total == 1:
        r = subprocess.run([FFMPEG, '-y', '-i', all_inputs[0], '-vf', norm,
                            '-c:v', 'libx264', '-preset', 'fast', '-crf', '22',
                            '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '128k', out_path],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f'FFMPEG single concat error: {r.stderr[:500]}')
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
        prev_label = 'n0'
        for i in range(n_total - 1):
            offset = sum(durations[:i + 1]) - (i + 1) * xfade_dur
            sfx_timestamps.append(max(offset, 0) + xfade_dur * 0.5)
            out_label = f'v{i+1}'
            filter_parts.append(
                f'[{prev_label}][n{i+1}]xfade=transition={transition}:duration={xfade_dur}:offset={max(offset, 0)}[{out_label}]')
            prev_label = out_label

        # Audio acrossfade chain — matches xfade video timing exactly
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
        cmd.extend(['-filter_complex', ';'.join(filter_parts)])
        last_vlabel = f'[{prev_label}]'
        cmd.extend(['-map', last_vlabel, '-map', f'[{last_audio_label}]'])
        cmd.extend(['-c:v', 'libx264', '-preset', 'fast', '-crf', '22', '-pix_fmt', 'yuv420p',
                     '-c:a', 'aac', '-b:a', '128k', out_path])
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            print(f'FFMPEG xfade error: {r.stderr[:1000]}')

    # Cleanup segment files
    for f in seg_files:
        if os.path.exists(f):
            os.remove(f)

    filename = os.path.basename(out_path)
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        _set_progress(job_id, 100, 'Trailer generation failed (ffmpeg output empty).', done=True, error=True)
        return jsonify(error='Trailer generation failed (ffmpeg output empty).'), 500
    # Verify it's a valid video
    v = subprocess.run([FFPROBE, '-v', 'error', '-show_entries', 'format=duration',
                        '-of', 'default=noprint_wrappers=1:nokey=1', out_path],
                       capture_output=True, text=True)
    if v.returncode != 0 or float(v.stdout.strip() or 0) <= 0:
        _set_progress(job_id, 100, 'Trailer generation failed (invalid/corrupt output).', done=True, error=True)
        return jsonify(error='Trailer generation failed (invalid/corrupt output).'), 500
    _set_progress(job_id, 82, 'Base trailer rendered.')

    # Generate SFX and mix into trailer audio. Whatever the source (AI-generated,
    # uploaded, or synthesized), the *same* hit gets stamped at every cut via
    # stamp_hits() so it's never just a single one-shot lost at the front.
    sfx_ok = False
    sfx_source = 'none'  # 'ai_generated' | 'uploaded' | 'synth_fallback' | 'none'
    sfx_path = os.path.join(app.config['UPLOAD_FOLDER'], f'sfx_{base_ts}.wav')
    if sfx_mode != 'none' and sfx_timestamps:
        _set_progress(job_id, 85, 'Generating and stamping sound effects...')
        hit_wave = None
        if sfx_mode == 'upload' and sfx_upload_path:
            hit_wave = load_hit_waveform(sfx_upload_path)
            sfx_source = 'uploaded' if hit_wave is not None else 'none'
        elif sfx_mode == 'genre' and genre:
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

    # J/L cut: pull a short pre-roll (J) or post-roll (L) of real audio from the
    # ORIGINAL source around each scene-to-scene cut and layer it into the trailer
    # at the matching timeline position. This only applies to cuts between two
    # selected scenes (not into end/schedule cards, which have no source timing).
    if jl_cut != 'none' and len(selected) > 1:
        _set_progress(job_id, 88, 'Applying J/L audio cuts...')
        pad_jobs = []  # (delay_seconds, source_start, source_dur)
        for i in range(len(selected) - 1):
            if i >= len(sfx_timestamps):
                break
            cut_time = sfx_timestamps[i]  # trailer-timeline position of this cut
            if jl_cut in ('j', 'both'):
                nxt = selected[i + 1]
                dur = min(jl_offset, max(0.05, nxt['end'] - nxt['start']))
                pad_jobs.append((max(cut_time - dur, 0), nxt['start'], dur))
            if jl_cut in ('l', 'both'):
                cur = selected[i]
                dur = min(jl_offset, max(0.05, cur['end'] - cur['start']))
                pad_jobs.append((cut_time, max(cur['end'] - dur, cur['start']), dur))

        pad_files = []
        for j, (delay_s, src_start, dur) in enumerate(pad_jobs):
            pad_path = os.path.join(app.config['UPLOAD_FOLDER'], f'jlpad_{base_ts}_{j}.m4a')
            r = subprocess.run([FFMPEG, '-y', '-ss', str(src_start), '-i', path,
                                '-t', str(dur), '-vn', '-c:a', 'aac', '-b:a', '192k', pad_path],
                               capture_output=True, text=True, timeout=30)
            if os.path.exists(pad_path) and os.path.getsize(pad_path) > 0:
                pad_files.append((delay_s, pad_path))

        if pad_files:
            jl_mixed = os.path.join(app.config['UPLOAD_FOLDER'], f'jlmix_{base_ts}.mp4')
            cmd = [FFMPEG, '-y', '-i', out_path]
            for _, pf in pad_files:
                cmd.extend(['-i', pf])
            fc_parts = []
            mix_labels = ['0:a']
            for idx, (delay_s, _) in enumerate(pad_files):
                ms = max(0, int(delay_s * 1000))
                fc_parts.append(f'[{idx+1}:a]adelay={ms}|{ms}[pad{idx}]')
                mix_labels.append(f'pad{idx}')
            fc_parts.append('[' + ']['.join(mix_labels) + f']amix=inputs={len(mix_labels)}:'
                             'duration=first:dropout_transition=0:normalize=0[outa]')
            cmd.extend(['-filter_complex', ';'.join(fc_parts),
                        '-map', '0:v', '-map', '[outa]',
                        '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-shortest', jl_mixed])
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if os.path.exists(jl_mixed) and os.path.getsize(jl_mixed) > 0:
                os.replace(jl_mixed, out_path)
            else:
                print(f'J/L cut mix error: {r.stderr[:500]}')
        for _, pf in pad_files:
            if os.path.exists(pf):
                os.remove(pf)

    # Mix scoring audio if enabled — duck background music behind original audio (SOT)
    bgm_source = 'none'  # 'uploaded' | 'ai_generated' | 'synth_fallback' | 'none'
    if scoring_audio_path and total_sel > 0:
        _set_progress(job_id, 90, 'Preparing background music...')
        processed_audio = os.path.join(app.config['UPLOAD_FOLDER'], f'score_{base_ts}.m4a')
        fade_in = 2.0
        fade_out = 3.0
        scenes_dur = total_sel
        prepared_bgm = None

        if scoring_audio_path == 'GENERATE':
            gen_audio = os.path.join(app.config['UPLOAD_FOLDER'], f'gen_{base_ts}.m4a')
            acestep_ok = False
            try:
                prompt = GENRE_PROMPTS.get(genre, 'Cinematic background music, instrumental, no vocals')
                r = requests.post(f'{ACE_STEP_URL}/release_task', json={
                    'prompt': prompt,
                    'audio_duration': scenes_dur,
                    'thinking': False,
                    'inference_steps': 8,
                    'batch_size': 1,
                }, timeout=5)
                data = r.json()
                task_id = data.get('data', {}).get('task_id')
                if task_id:
                    for poll_i in range(60):
                        time.sleep(2)
                        _set_progress(job_id, 90 + min(4, poll_i // 15),
                                      f'Waiting for AI music generation... ({poll_i + 1}/60)')
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
                subprocess.run([FFMPEG, '-y',
                                '-f', 'lavfi', '-i',
                                f'aevalsrc=exprs=\'{lavfi_src}\':d={scenes_dur}:s=44100:c=stereo',
                                '-af',
                                # lowpass softens the raw additive-sine harmonics, aecho
                                # adds a touch of ambient space so the fallback reads as
                                # a pad rather than a bare synthesizer tone
                                f'lowpass=f=6000,tremolo=f=0.15:d=0.4,volume=1.5,'
                                f'aecho=0.8:0.7:60:0.25,'
                                f'afade=t=in:d={fade_in},'
                                f'afade=t=out:st={max(scenes_dur - fade_out, 0)}:d={min(fade_out, scenes_dur)}',
                                '-c:a', 'aac', '-b:a', '192k', gen_audio],
                               capture_output=True, text=True, timeout=30)
                bgm_source = 'synth_fallback'
            if os.path.exists(gen_audio) and os.path.getsize(gen_audio) > 0:
                prepared_bgm = gen_audio
            else:
                bgm_source = 'none'
        else:
            r = subprocess.run([FFMPEG, '-y', '-i', scoring_audio_path,
                                '-af', (f'atrim=duration={scenes_dur},'
                                        f'afade=t=in:d={fade_in},'
                                        f'afade=t=out:st={max(scenes_dur - fade_out, 0)}:d={min(fade_out, scenes_dur)}'),
                                '-c:a', 'aac', '-b:a', '192k', '-vn', processed_audio],
                               capture_output=True, text=True, timeout=60)
            if os.path.exists(processed_audio) and os.path.getsize(processed_audio) > 0:
                prepared_bgm = processed_audio
                bgm_source = 'uploaded'

        if prepared_bgm and beat_match:
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
            _set_progress(job_id, 96, 'Mixing and normalizing final audio...')
            mixed = os.path.join(app.config['UPLOAD_FOLDER'], f'mixed_{base_ts}.mp4')
            r = subprocess.run([FFMPEG, '-y', '-i', out_path, '-i', prepared_bgm,
                                '-filter_complex',
                                '[0:a]asplit[original][side];[1:a]volume=0.55[bgm];'
                                '[bgm][side]sidechaincompress=threshold=0.08:ratio=6:attack=50:release=400:makeup=1.3[bgm_ducked];'
                                f'[original][bgm_ducked]amix=inputs=2:duration=first:dropout_transition=2:normalize=0,'
                                f'loudnorm=I={target_loudness}:TP={true_peak}:LRA=7[outa]',
                                '-map', '0:v', '-map', '[outa]',
                                '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
                                '-shortest', mixed],
                               capture_output=True, text=True, timeout=120)
            if os.path.exists(mixed) and os.path.getsize(mixed) > 0:
                os.replace(mixed, out_path)
            if os.path.exists(prepared_bgm):
                os.remove(prepared_bgm)

    _set_progress(job_id, 100, 'Trailer ready!', done=True)
    return jsonify(status='ok', trailer_url=f'/uploads/{filename}',
                   orig_name=orig_name,
                   total_scenes=len(scene_list), selected_scenes=len(selected),
                   trailer_duration=round(total_sel, 1),
                   video_duration=round(video_duration, 1),
                   trailer_length=trailer_length,
                   bgm_source=bgm_source, sfx_source=sfx_source, jl_cut=jl_cut,
                    scenes=[{
                        'scene': i+1, 'start': round(s['start'], 1), 'end': round(s['end'], 1),
                        'quality': s['total_score'], 'duration': round(s['selected_dur'], 1),
                        'description': _scene_desc(s)
                    } for i, s in enumerate(selected)])

# ---- UI ----

UI = '''
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Video Toolkit - OpenCV, PySceneDetect, Ollama Vision & FFmpeg</title>
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
.or{text-align:center;color:var(--ink-dim);margin:12px 0;font-size:10px;font-family:'JetBrains Mono',monospace;text-transform:uppercase;letter-spacing:.1em;display:flex;align-items:center;gap:10px}
.or::before,.or::after{content:'';flex:1;height:1px;background:var(--line)}

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
.tr-progress-wrap{margin-top:14px;padding:16px;background:var(--elevated);border:1px solid var(--line);border-radius:8px}
.tr-progress-track{width:100%;height:8px;background:var(--sunken);border-radius:4px;overflow:hidden;margin-top:10px}
.tr-progress-fill{height:100%;background:linear-gradient(90deg,var(--phosphor-dim),var(--phosphor));border-radius:4px;transition:width .3s ease;width:0%}
.tr-progress-fill.tr-progress-error{background:var(--tally)}
.tr-progress-label{display:flex;justify-content:space-between;align-items:center;font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--ink-dim)}
.tr-progress-pct{color:var(--phosphor);font-weight:600}

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
  <h1>AI Video Toolkit<small>Computer vision &amp; video engineering console</small></h1>
  <div class="hdr-engines">opencv &middot; pyscenedetect &middot; ollama &middot; ffmpeg</div>
</div></div>
<div class="container">
<div class="tabs">
  <div class="tab active" onclick="switchTab('p-trailer',this)" role="button" tabindex="0"><span class=tab-icon>&#9636;</span>Generate Promo Plug<div class=tab-sub>ai+ffmpeg</div></div>
  <div class="tab" onclick="switchTab('p-tools',this)" role="button" tabindex="0"><span class=tab-icon>&#9881;</span>Tools<div class=tab-sub>upload+player+vision</div></div>
  <div class="tab" onclick="switchTab('p-api',this)" role="button" tabindex="0"><span class=tab-icon>{ }</span>API<div class=tab-sub>reference</div></div>
  <div class="tab" onclick="switchTab('p-docs',this)" role="button" tabindex="0"><span class=tab-icon>&#128214;</span>Workflow Docs<div class=tab-sub>how it works</div></div>
</div>

<script>function switchTab(id,btn){document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));document.querySelectorAll('.sub-panel').forEach(p=>p.classList.remove('active'));document.querySelectorAll('.sub-tab').forEach(t=>t.classList.remove('active'));btn.classList.add('active');document.getElementById(id).classList.add('active');if(id==='p-tools'){var fst=document.querySelector('#p-tools .sub-tab');if(fst)fst.click()}}function switchSubTab(id,btn){document.querySelectorAll('.sub-tab').forEach(t=>t.classList.remove('active'));document.querySelectorAll('.sub-panel').forEach(p=>p.classList.remove('active'));btn.classList.add('active');document.getElementById(id).classList.add('active')}</script>

<!-- Tools -->
<div id="p-tools" class="panel">
<div class="sub-tabs">
  <div class="sub-tab active" onclick="switchSubTab('p-upload',this)" role="button" tabindex="0">&#8682; Upload</div>
  <div class="sub-tab" onclick="switchSubTab('p-player',this)" role="button" tabindex="0">&#9654; Player</div>
  <div class="sub-tab" onclick="switchSubTab('p-vision',this)" role="button" tabindex="0">&#9673; Vision AI</div>
</div>

<!-- Upload -->
<div id="p-upload" class="sub-panel active">
<h2>Upload &amp; Analyze</h2>
<form method=POST action=/upload enctype=multipart/form-data onsubmit="return true">
  <input type=file name=file accept=video/*>
  <div class=or>or</div>
  <input type=url name=video_url placeholder="https://example.com/video.mp4">
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

<!-- Player -->
<div id="p-player" class="sub-panel">
<h2>Video Player</h2>
<form id=pf method=POST action=/api/playback/start enctype=multipart/form-data>
  <input type=file name=file accept=video/*>
  <div class=or>or</div>
  <input type=url name=video_url placeholder="https://example.com/video.mp4">
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

<!-- Vision -->
<div id="p-vision" class="sub-panel">
<h2>Vision AI with Ollama</h2>
<form id=vf method=POST action=/api/vision/analyze enctype=multipart/form-data>
  <input type=file name=file accept=video/*>
  <div class=or>or</div>
  <input type=url name=video_url placeholder="https://example.com/video.mp4">
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
<div id=vr-prompt class=no-data>Upload a video to analyze frames with Ollama vision model.</div>
</div>
</div>

<!-- Trailer -->
<div id="p-trailer" class="panel active">
<h2>Trailer Generator</h2>
<p>Uses PySceneDetect to find scene boundaries, OpenCV to score quality (sharpness, brightness), and optionally Ollama Vision for AI scoring.</p>
<form id=tf method=POST action=/api/trailer/generate enctype=multipart/form-data>
  <input type=file name=file accept=video/*>
  <div class=or>or</div>
  <input type=url name=video_url placeholder="https://example.com/video.mp4">
  <label>Mode:</label>
  <select name=mode>
    <option value=auto>Auto (OpenCV quality scoring)</option>
    <option value=ai>AI (OpenCV + Ollama Vision scoring)</option>
  </select>
  <label>Trailer length:</label>
  <select name=trailer_length>
    <option value=15 selected>15 sec (needs 22.5s raw video)</option>
    <option value=30>30 sec (needs 45s raw video)</option>
    <option value=45>45 sec (needs 67.5s raw video)</option>
    <option value=60>60 sec (needs 90s raw video)</option>
  </select>
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
  <div class="genre-manual" style="display:none">
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
  </select>
  <label>Transition duration (s):</label>
  <input type=number name=xfade_dur value=0.3 min=0.1 max=2 step=0.1 style="width:100px">
  <label>Audio normalisation:</label>
  <div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:8px 0">
    <span style="font-size:12px">Target&nbsp;LUFS:</span>
    <input type=number name=target_loudness value=-16 min=-30 max=-10 step=0.5 style="width:70px">
    <span style="font-size:12px">True&nbsp;peak&nbsp;(dB):</span>
    <input type=number name=true_peak value=-1.5 min=-6 max=0 step=0.5 style="width:70px">
    <label style="font-size:13px;text-transform:none;letter-spacing:0;display:flex;align-items:center;gap:4px;margin:0 0 0 12px;cursor:pointer">
      <input type=checkbox name=beat_match> Beat match (librosa)
    </label>
  </div>
  </div>
  <label>AI model (if AI mode):</label>
  <select name=model id=trailer-model><option value="">Loading...</option></select>
  <label>End card video (optional):</label>
  <input type=file name=end_card_video accept=video/*>
  <label>Schedule card video (optional):</label>
  <input type=file name=schedule_video accept=video/*>
  <label>Background music:</label>
  <div style="display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:8px 0">
    <label style="font-size:13px;text-transform:none;letter-spacing:0;display:flex;align-items:center;gap:4px;margin:0;cursor:pointer">
      <input type=radio name=scoring_mode value=generate checked> Generate music
    </label>
    <label style="font-size:13px;text-transform:none;letter-spacing:0;display:flex;align-items:center;gap:4px;margin:0;cursor:pointer">
      <input type=radio name=scoring_mode value=none> None
    </label>
    <label style="font-size:13px;text-transform:none;letter-spacing:0;display:flex;align-items:center;gap:4px;margin:0;cursor:pointer">
      <input type=radio name=scoring_mode value=upload> Upload
    </label>
  </div>
  <div id=scoring-upload-area style="display:none">
    <input type=file name=scoring_audio accept="audio/*,.mp3,.wav,.m4a,.flac,.ogg">
  </div>

  <label>Sound effects (stamped at every scene cut):</label>
  <div style="display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:8px 0">
    <label style="font-size:13px;text-transform:none;letter-spacing:0;display:flex;align-items:center;gap:4px;margin:0;cursor:pointer">
      <input type=radio name=sfx_mode value=genre checked> From genre
    </label>
    <label style="font-size:13px;text-transform:none;letter-spacing:0;display:flex;align-items:center;gap:4px;margin:0;cursor:pointer">
      <input type=radio name=sfx_mode value=upload> Upload one-shot
    </label>
    <label style="font-size:13px;text-transform:none;letter-spacing:0;display:flex;align-items:center;gap:4px;margin:0;cursor:pointer">
      <input type=radio name=sfx_mode value=none> None
    </label>
  </div>
  <div id=sfx-upload-area style="display:none">
    <input type=file name=sfx_upload accept="audio/*,.mp3,.wav,.m4a,.flac,.ogg">
    <p style="margin-top:4px">A single hit/whoosh/impact sound — it gets stamped at every cut, not looped as music.</p>
  </div>

  <label>J/L cut (bleed audio across the cut point):</label>
  <div style="display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:8px 0">
    <select name=jl_cut style="width:auto">
      <option value=none selected>None (plain crossfade)</option>
      <option value=j>J-cut (hear next scene early)</option>
      <option value=l>L-cut (linger on outgoing audio)</option>
      <option value=both>Both (bleed both directions)</option>
    </select>
    <span style="font-size:12px">Bleed (s):</span>
    <input type=number name=jl_offset value=0.35 min=0.1 max=1.5 step=0.05 style="width:80px">
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
  document.querySelector('select[name=genre]').addEventListener('change', function(){
    var custom = this.value === ''
    document.querySelectorAll('.genre-manual').forEach(function(el){ el.style.display = custom ? '' : 'none' })
  })
  </script>
  <button class=btn type=submit>Generate Trailer</button>
</form>
<div id=tr-progress-wrap class="tr-progress-wrap" style=display:none>
  <div class=tr-progress-label><span id=tr-progress-msg>Starting...</span><span class=tr-progress-pct id=tr-progress-pct>0%</span></div>
  <div class=tr-progress-track><div class=tr-progress-fill id=tr-progress-fill></div></div>
</div>
<div id=tr-area style=display:none>
  <div class=card id=tr-stats></div>
  <div style=margin:10px 0 id=tr-video></div>
  <table id=tr-table><tr><th>Scene</th><th>Start</th><th>End</th><th>Quality</th><th>Used (s)</th></tr></table>
</div>
<div id=tr-prompt class=no-data>Upload a video to generate a trailer from the best scenes.</div>
</div>

<script>
function trSetProgress(pct, msg, isError){
  var fill=document.getElementById('tr-progress-fill')
  var pctEl=document.getElementById('tr-progress-pct')
  var msgEl=document.getElementById('tr-progress-msg')
  fill.style.width=Math.max(0,Math.min(100,pct))+'%'
  fill.classList.toggle('tr-progress-error', !!isError)
  pctEl.textContent=Math.round(pct)+'%'
  msgEl.textContent=msg||''
}

document.getElementById('tf').addEventListener('submit', async function(e){
  e.preventDefault()
  document.getElementById('tr-area').style.display='none'
  document.getElementById('tr-prompt').style.display='none'
  document.getElementById('tr-progress-wrap').style.display='block'
  trSetProgress(0, 'Starting...', false)

  var jobId='trjob_'+Date.now()+'_'+Math.random().toString(36).slice(2)
  var fd=new FormData(this)
  fd.append('job_id', jobId)

  var polling=true
  var pollTimer=setInterval(async function(){
    if(!polling) return
    try{
      let pr=await fetch('/api/trailer/progress/'+jobId)
      let pd=await pr.json()
      trSetProgress(pd.percent||0, pd.message||'', !!pd.error)
    }catch(err){}
  }, 1000)

  let r=await fetch('/api/trailer/generate',{method:'POST',body:fd})
  let d=await r.json()
  polling=false
  clearInterval(pollTimer)
  document.getElementById('tr-progress-wrap').style.display='none'
  if(d.error){document.getElementById('tr-stats').innerHTML='<b>Error:</b> '+d.error; document.getElementById('tr-area').style.display='block'; return}
  trSetProgress(100, 'Trailer ready!', false)
  var srcLabel=function(s){return {ai_generated:'AI-generated',uploaded:'uploaded',synth_fallback:'placeholder synth (ACE-Step unavailable)',none:'none'}[s]||s}
  var audioNote=''
  if(d.bgm_source && d.bgm_source!=='none') audioNote+=' | Music: '+srcLabel(d.bgm_source)
  if(d.sfx_source && d.sfx_source!=='none') audioNote+=' | SFX: '+srcLabel(d.sfx_source)
  if(d.jl_cut && d.jl_cut!=='none') audioNote+=' | Cut: '+({j:'J-cut',l:'L-cut',both:'J+L cut'}[d.jl_cut]||d.jl_cut)
  var fallbackUsed=(d.bgm_source==='synth_fallback'||d.sfx_source==='synth_fallback')
  document.getElementById('tr-stats').innerHTML='Trailer: '+d.trailer_duration+'s (target '+d.trailer_length+'s) from '+d.selected_scenes+'/'+d.total_scenes+' scenes | Raw video: '+d.video_duration+'s'+audioNote+(fallbackUsed?'<br><small style="color:var(--amber)">Note: AI music/SFX service was unavailable, so a lower-fidelity generated placeholder was used instead.</small>':'')
  var dlUrl ='/download/'+d.trailer_url.split('/').pop()+'?name='+encodeURIComponent(d.orig_name)
  document.getElementById('tr-video').innerHTML='<video controls style=max-width:100%;border-radius:8px><source src="'+d.trailer_url+'" type="video/mp4"></video><br><a href="'+dlUrl+'" class="btn" style="display:inline-block;margin-top:8px;text-decoration:none">Download</a>'
  let rows=''
  d.scenes.forEach(s=>{rows+='<tr><td>'+s.scene+'</td><td>'+s.start+'s</td><td>'+s.end+'s</td><td>'+s.quality+'</td><td>'+s.duration+'</td><td>'+s.description+'</td></tr>'})
  document.getElementById('tr-table').innerHTML='<tr><th>#</th><th>Start</th><th>End</th><th>Score</th><th>Used</th><th>Description</th></tr>'+rows
  document.getElementById('tr-area').style.display='block'
})
</script>


<script>
// Load Ollama vision models into dropdowns
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
<p><code>POST multipart/form-data</code> with <code>file</code> or <code>video_url</code>:</p>
<div class=card>
<div class=info>
  <div class=info-item><strong>POST</strong> /api/opencv/info</div>
  <div class=info-item><strong>POST</strong> /api/opencv/analyze<br><small>+ num_frames</small></div>
  <div class=info-item><strong>POST</strong> /api/scenedetect/detect<br><small>+ threshold</small></div>
</div>
</div>
<p><strong>Player</strong>:</p>
<pre>POST /api/playback/start  + file/url
GET  /api/playback/stream/raw|gray|edges|hsv|blur|face|motion
GET  /api/playback/pause
GET  /api/playback/resume
GET  /api/playback/seek?t=seconds
GET  /api/playback/stop</pre>
<p><strong>Trailer generation</strong>:</p>
<pre>POST /api/trailer/generate       + file/url, job_id, mode, genre, trailer_length, ...
GET  /api/trailer/progress/&lt;job_id&gt;   poll while generate is in flight</pre>
</div>

<!-- Workflow Documentation -->
<div id="p-docs" class="panel">
<h2>Workflow Documentation</h2>
<p>How a video becomes a trailer, stage by stage. The percentages match what the progress bar on the Generate Promo Plug tab shows while a job runs.</p>

<div class=card>
<strong>1. Upload / fetch source video &mdash; 0&ndash;5%</strong>
<p>Accepts a direct file upload or a <code>video_url</code> to fetch. Validated against the allowed container list (mp4, avi, mov, mkv, wmv, flv, webm), then probed with OpenCV for FPS, frame count, resolution, and total duration.</p>
</div>

<div class=card>
<strong>2. Scene detection &mdash; 5&ndash;18%</strong>
<p>PySceneDetect's <code>ContentDetector</code> (threshold 30.0) walks the video and returns every hard cut as a (start, end) timecode pair. If the source is shorter than 1.5&times; the requested trailer length, or no cuts are found at all, generation stops here with an error.</p>
</div>

<div class=card>
<strong>3. Scene scoring &mdash; 18&ndash;55%</strong>
<p>Every detected scene is scored twice:</p>
<ul style="margin-left:18px;color:var(--ink-dim);font-size:13.5px;line-height:1.7">
<li><strong>Auto (OpenCV quality):</strong> the middle frame of each scene is graded on sharpness (Laplacian variance), brightness, and clip duration (1&ndash;8s scenes score best). Face presence, hue, saturation, and value are also recorded for the scene description shown in results.</li>
<li><strong>AI (Ollama Vision):</strong> only runs in AI mode, on top of the auto score. Each scene's middle frame is sent to the selected Ollama vision model with a prompt asking for a short description and a 1&ndash;5 trailer-worthiness rating, parsed back out with a regex and added to the quality score.</li>
</ul>
</div>

<div class=card>
<strong>4. Scene selection &mdash; ~55&ndash;58%</strong>
<p>Scenes are sorted by total score (highest first) and greedily added, trimming the last one if needed, until the target duration is filled. Selected scenes are then re-sorted back into chronological order. Because crossfade transitions eat into the runtime, this pass repeats up to 3 times, nudging the target duration to converge on the requested trailer length.</p>
</div>

<div class=card>
<strong>5. Segment extraction &mdash; 58&ndash;70%</strong>
<p>Each selected scene is cut out of the source video with ffmpeg (fast preset, CRF 28) as its own temp clip, alongside any uploaded end-card or schedule-card videos.</p>
</div>

<div class=card>
<strong>6. Transition rendering &mdash; 70&ndash;82%</strong>
<p>All clips are normalized to a common resolution/framerate, then stitched together with an ffmpeg <code>xfade</code> filter chain using the chosen (or genre-preset) transition and crossfade duration, with a matching <code>acrossfade</code> for audio. A single clip skips straight to a plain re-encode.</p>
</div>

<div class=card>
<strong>7. Sound effects &mdash; 82&ndash;88%</strong>
<p>If SFX are enabled, a one-shot "hit" sound (AI-generated via ACE-Step, uploaded by the user, or procedurally synthesized as a fallback) is stamped into the audio at every scene-to-scene cut point.</p>
</div>

<div class=card>
<strong>8. J/L cuts &mdash; 88&ndash;90%</strong>
<p>Optional: bleeds a short slice of the <em>original</em> scene audio early (J-cut) or lets it linger past the cut (L-cut), layered in at the matching timeline position for a more natural audio transition between scenes.</p>
</div>

<div class=card>
<strong>9. Background music &mdash; 90&ndash;98%</strong>
<p>If scoring audio is enabled, music is either taken from an upload or generated by ACE-Step from a genre-matched prompt (polled every ~2s, up to 60 attempts). If ACE-Step is unreachable or times out, a procedurally synthesized ambient pad takes its place instead. Optional beat-matching aligns it to the video, then it's sidechain-ducked under the original audio and loudness-normalized (target LUFS / true peak) into the final mix.</p>
</div>

<div class=card>
<strong>10. Done &mdash; 100%</strong>
<p>The finished trailer, per-scene breakdown, and a note on which audio sources were AI-generated vs. fallback are returned to the browser.</p>
</div>

<p style="margin-top:16px;color:var(--ink-dim);font-size:13px">Progress is tracked server-side per <code>job_id</code> and polled via <code>GET /api/trailer/progress/&lt;job_id&gt;</code> once a second while <code>/api/trailer/generate</code> is running.</p>
</div>

</div><!-- container -->

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
    return render_template_string(UI)

@app.route('/upload', methods=['POST'])
def upload():
    path, err = load_video(request)
    if not path:
        return render_template_string(UI, r={'error': err})
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
    video = open_video(path)
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=30.0))
    sm.detect_scenes(video)
    scenes = [{'scene': i+1, 'start': s.get_timecode(), 'end': e.get_timecode(),
               'start_sec': round(s.get_seconds(), 2), 'end_sec': round(e.get_seconds(), 2),
               'duration': round(e.get_seconds() - s.get_seconds(), 2)}
              for i, (s, e) in enumerate(sm.get_scene_list())]
    return render_template_string(UI, r={'info': info, 'frames': frames, 'scenes': scenes})

@app.route('/uploads/<filename>')
def uploaded(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/download/<filename>')
def download_file(filename):
    orig = request.args.get('name', filename)
    resp = send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    resp.headers['Content-Disposition'] = f'attachment; filename="{orig}"'
    return resp

if __name__ == '__main__':
    print(' * Server starting...')
    print(' * HTTP:  http://0.0.0.0:5000/')
    print(' * Access from local machine: http://localhost:5000/')
    print(' * Access from other devices: http://YOUR_IP:5000/')
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
