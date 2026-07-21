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
    'action': {'transition': 'fade', 'xfade_dur': 0.2, 'sfx': True},
    'drama': {'transition': 'fade', 'xfade_dur': 0.6, 'sfx': False},
    'horror': {'transition': 'wipeleft', 'xfade_dur': 0.3, 'sfx': True},
    'comedy': {'transition': 'slideleft', 'xfade_dur': 0.25, 'sfx': True},
    'documentary': {'transition': 'fade', 'xfade_dur': 0.5, 'sfx': False},
    'thriller': {'transition': 'fade', 'xfade_dur': 0.2, 'sfx': True},
    'scifi': {'transition': 'fadeblack', 'xfade_dur': 0.3, 'sfx': True},
    'fantasy': {'transition': 'fade', 'xfade_dur': 0.5, 'sfx': False},
    'romance': {'transition': 'fade', 'xfade_dur': 0.8, 'sfx': False},
    'adventure': {'transition': 'fade', 'xfade_dur': 0.2, 'sfx': True},
    'mystery': {'transition': 'fadeblack', 'xfade_dur': 0.4, 'sfx': False},
    'western': {'transition': 'slideleft', 'xfade_dur': 0.3, 'sfx': True},
    'sports': {'transition': 'fade', 'xfade_dur': 0.15, 'sfx': True},
    'noir': {'transition': 'fadeblack', 'xfade_dur': 0.6, 'sfx': False},
    'war': {'transition': 'fade', 'xfade_dur': 0.3, 'sfx': True},
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

def sfx_for_genre(genre, timestamps, output_path, sample_rate=22050, sfx_dur=0.4):
    try:
        total_dur = max(timestamps) + sfx_dur + 0.5 if timestamps else 1.0
        total_samples = int(total_dur * sample_rate)
        track = np.zeros(total_samples)
        n_sfx = int(sfx_dur * sample_rate)
        t = np.arange(n_sfx) / sample_rate
        if genre == 'action' or genre == 'war':
            sfx = np.random.randn(n_sfx) * np.exp(-t * 20) * 0.5
        elif genre == 'horror':
            sfx = np.sin(2*np.pi * (2000 + t * 6000) * t) * np.exp(-t * 8) * 0.3
        elif genre == 'comedy':
            sfx = np.sin(2*np.pi * (600 - t * 1200) * t) * np.exp(-t * 5) * 0.4
        elif genre in ('thriller', 'adventure', 'scifi'):
            sfx = np.sin(2*np.pi * (100 + t * 3000) * t) * np.exp(-t * 6) * 0.3
        elif genre == 'western':
            sfx = np.random.randn(n_sfx) * np.exp(-t * 50) * 0.5
        elif genre == 'sports':
            sq = np.sign(np.sin(2*np.pi * 440 * t)) * 0.3 + np.sign(np.sin(2*np.pi * 880 * t)) * 0.15
            sfx = sq * np.exp(-t * 3)
        elif genre == 'fantasy':
            sfx = (np.sin(2*np.pi * 528 * t) * 0.3 + np.sin(2*np.pi * 1056 * t) * 0.15 +
                   np.sin(2*np.pi * 1584 * t) * 0.08) * np.exp(-t * 4)
        else:
            return False
        peak = np.max(np.abs(sfx))
        if peak > 0:
            sfx = sfx / peak * 0.5
        for ts in timestamps:
            start = int(ts * sample_rate)
            end = min(start + len(sfx), total_samples)
            track[start:end] += sfx[:end-start]
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
    except Exception as e:
        print(f'SFX generation error: {e}')
        return False

def acestep_sfx(genre, output_path, duration=1.5):
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

_wc_lock = threading.Lock()
_wc_cap = None
_wc_running = False

@app.route('/api/webcam/start', methods=['POST'])
def wc_start():
    global _wc_cap, _wc_running
    cam_id = int(request.form.get('camera', 0))
    with _wc_lock:
        if _wc_cap:
            _wc_cap.release()
        _wc_cap = cv2.VideoCapture(cam_id)
        if not _wc_cap.isOpened():
            return jsonify(error=f'Could not open camera {cam_id}'), 400
        _wc_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        _wc_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        _wc_running = True
    return jsonify(status='ok')

@app.route('/api/webcam/stream/<mode>')
def wc_stream(mode):
    global _wc_cap, _wc_running
    if mode not in ('raw', 'gray', 'edges', 'hsv', 'blur'):
        mode = 'raw'
    def gen():
        prev = None
        while _wc_running:
            with _wc_lock:
                cap = _wc_cap
                if cap is None or not cap.isOpened():
                    break
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.02)
                    continue
            frame, prev = apply_filter(frame, mode, prev)
            _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n'
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/webcam/stop')
def wc_stop():
    global _wc_cap, _wc_running
    with _wc_lock:
        _wc_running = False
        if _wc_cap:
            _wc_cap.release()
            _wc_cap = None
    return '', 204

@app.route('/api/webcam/cameras')
def wc_cameras():
    cams = []
    for i in range(6):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cams.append({'id': i, 'name': f'Camera {i}'})
            cap.release()
    return jsonify(cameras=cams)

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

@app.route('/api/trailer/generate', methods=['POST'])
def api_trailer():
    path, orig_name = load_video(request)
    if not path:
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
        return jsonify(error=f'Video is only {video_duration:.1f}s long, but a {trailer_length}s trailer requires at least {min_required:.1f}s of raw video. Upload a longer video or select a shorter trailer length.'), 400

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
    video = open_video(path)
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=30.0))
    sm.detect_scenes(video)
    scene_list = sm.get_scene_list()
    if not scene_list:
        return jsonify(error='No scene changes detected. Try a video with clear cuts.'), 400

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
        for s in scenes_data:
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
        return jsonify(error='No scenes selected.'), 400

    # Extract selected segments + card videos as temp files
    seg_files = []
    base_ts = int(time.time())
    for seg_i, seg in enumerate(selected):
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
        return jsonify(error='Trailer generation failed (ffmpeg output empty).'), 500
    # Verify it's a valid video
    v = subprocess.run([FFPROBE, '-v', 'error', '-show_entries', 'format=duration',
                        '-of', 'default=noprint_wrappers=1:nokey=1', out_path],
                       capture_output=True, text=True)
    if v.returncode != 0 or float(v.stdout.strip() or 0) <= 0:
        return jsonify(error='Trailer generation failed (invalid/corrupt output).'), 500

    # Generate genre SFX and mix into trailer audio
    sfx_ok = False
    sfx_path = os.path.join(app.config['UPLOAD_FOLDER'], f'sfx_{base_ts}.wav')
    if genre and genre in GENRE_PRESETS and GENRE_PRESETS[genre].get('sfx') and sfx_timestamps:
        acestep_sfx_path = os.path.join(app.config['UPLOAD_FOLDER'], f'acestep_sfx_{base_ts}.wav')
        if acestep_sfx(genre, acestep_sfx_path, duration=2.0) and os.path.getsize(acestep_sfx_path) > 0:
            sfx_path = acestep_sfx_path
            sfx_ok = True
        elif sfx_for_genre(genre, sfx_timestamps, sfx_path):
            sfx_ok = True
        if sfx_ok:
            sfx_m4a = os.path.join(app.config['UPLOAD_FOLDER'], f'sfx_{base_ts}.m4a')
            subprocess.run([FFMPEG, '-y', '-i', sfx_path, '-c:a', 'aac', '-b:a', '192k', sfx_m4a],
                           capture_output=True, text=True, timeout=30)
            if os.path.exists(sfx_m4a) and os.path.getsize(sfx_m4a) > 0:
                # Mix SFX into the trailer audio
                with_sfx = os.path.join(app.config['UPLOAD_FOLDER'], f'with_sfx_{base_ts}.mp4')
                r = subprocess.run([FFMPEG, '-y', '-i', out_path, '-i', sfx_m4a,
                                    '-filter_complex', '[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=2[outa]',
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

    # Mix scoring audio if enabled — duck background music behind original audio (SOT)
    if scoring_audio_path and total_sel > 0:
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
                    for _ in range(60):
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
            if not acestep_ok:
                lavfi_src = GENRE_LAVFI.get(genre, GENRE_LAVFI['default'])
                subprocess.run([FFMPEG, '-y',
                                '-f', 'lavfi', '-i',
                                f'aevalsrc=exprs=\'{lavfi_src}\':d={scenes_dur}:s=44100:c=stereo',
                                '-af',
                                f'tremolo=f=0.15:d=0.4,volume=1.5,'
                                f'afade=t=in:d={fade_in},'
                                f'afade=t=out:st={max(scenes_dur - fade_out, 0)}:d={min(fade_out, scenes_dur)}',
                                '-c:a', 'aac', '-b:a', '192k', gen_audio],
                               capture_output=True, text=True, timeout=30)
            if os.path.exists(gen_audio) and os.path.getsize(gen_audio) > 0:
                prepared_bgm = gen_audio
        else:
            r = subprocess.run([FFMPEG, '-y', '-i', scoring_audio_path,
                                '-af', (f'atrim=duration={scenes_dur},'
                                        f'afade=t=in:d={fade_in},'
                                        f'afade=t=out:st={max(scenes_dur - fade_out, 0)}:d={min(fade_out, scenes_dur)}'),
                                '-c:a', 'aac', '-b:a', '192k', '-vn', processed_audio],
                               capture_output=True, text=True, timeout=60)
            if os.path.exists(processed_audio) and os.path.getsize(processed_audio) > 0:
                prepared_bgm = processed_audio

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
            mixed = os.path.join(app.config['UPLOAD_FOLDER'], f'mixed_{base_ts}.mp4')
            r = subprocess.run([FFMPEG, '-y', '-i', out_path, '-i', prepared_bgm,
                                '-filter_complex',
                                '[0:a]asplit[original][side];[1:a]volume=0.3[bgm];'
                                '[bgm][side]sidechaincompress=threshold=0.05:ratio=10:attack=50:release=500:makeup=1[bgm_ducked];'
                                f'[original][bgm_ducked]amix=inputs=2:duration=first:dropout_transition=2,'
                                f'loudnorm=I={target_loudness}:TP={true_peak}:LRA=7[outa]',
                                '-map', '0:v', '-map', '[outa]',
                                '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
                                '-shortest', mixed],
                               capture_output=True, text=True, timeout=120)
            if os.path.exists(mixed) and os.path.getsize(mixed) > 0:
                os.replace(mixed, out_path)
            if os.path.exists(prepared_bgm):
                os.remove(prepared_bgm)

    return jsonify(status='ok', trailer_url=f'/uploads/{filename}',
                   orig_name=orig_name,
                   total_scenes=len(scene_list), selected_scenes=len(selected),
                   trailer_duration=round(total_sel, 1),
                   video_duration=round(video_duration, 1),
                   trailer_length=trailer_length,
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
  <div class="tab" onclick="switchTab('p-tools',this)" role="button" tabindex="0"><span class=tab-icon>&#9881;</span>Tools<div class=tab-sub>upload+player+webcam+vision</div></div>
  <div class="tab" onclick="switchTab('p-api',this)" role="button" tabindex="0"><span class=tab-icon>{ }</span>API<div class=tab-sub>reference</div></div>
</div>

<script>function switchTab(id,btn){document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));document.querySelectorAll('.sub-panel').forEach(p=>p.classList.remove('active'));document.querySelectorAll('.sub-tab').forEach(t=>t.classList.remove('active'));btn.classList.add('active');document.getElementById(id).classList.add('active');if(id==='p-tools'){var fst=document.querySelector('#p-tools .sub-tab');if(fst)fst.click()}}function switchSubTab(id,btn){document.querySelectorAll('.sub-tab').forEach(t=>t.classList.remove('active'));document.querySelectorAll('.sub-panel').forEach(p=>p.classList.remove('active'));btn.classList.add('active');document.getElementById(id).classList.add('active')}</script>

<!-- Tools -->
<div id="p-tools" class="panel">
<div class="sub-tabs">
  <div class="sub-tab active" onclick="switchSubTab('p-upload',this)" role="button" tabindex="0">&#8682; Upload</div>
  <div class="sub-tab" onclick="switchSubTab('p-player',this)" role="button" tabindex="0">&#9654; Player</div>
  <div class="sub-tab" onclick="switchSubTab('p-webcam',this)" role="button" tabindex="0">&#9679; Webcam</div>
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

<!-- Webcam -->
<div id="p-webcam" class="sub-panel">
<h2>Live Webcam <small>Browser or server capture</small></h2>
<div id=wc-https-warn class=no-data style="display:none;color:var(--amber);border-color:var(--amber)">
  Browser camera requires HTTPS. Use <b>Server Capture</b> mode below instead, or open <a href="https://localhost:5001" style="color:var(--amber)">https://localhost:5001</a>.
</div>
<div class=filters id=wc-controls style=display:none>
  <button class="btn small" id="wc-start">Start Camera</button>
  <span class="rec-label" id="wc-rec"><span class=rec-dot></span>Live</span>
  <button class="btn small active-filter" id="wc-raw">Raw</button>
  <button class="btn small" id="wc-gray">Gray</button>
  <button class="btn small" id="wc-edges">Edges</button>
  <button class="btn small" id="wc-hsv">HSV</button>
  <button class="btn small" id="wc-blur">Blur</button>
  <button class="btn small danger" id="wc-stop">Stop</button>
</div>
<div class=stream-wrap>
  <i class="corner tl"></i><i class="corner tr"></i><i class="corner bl"></i><i class="corner br"></i>
  <video id=wc-video autoplay playsinline style="display:none"></video>
  <canvas id=wc-canvas></canvas>
  <img id=wc-server-feed style="display:none;max-width:100%;border-radius:4px">
</div>
<div class=card style="margin-top:12px">
  <p style="margin:0 0 8px;font-size:11px;color:var(--ink-dim)">Server Capture — works on HTTP, captures from server webcam</p>
  <div class=filters>
    <select id=wc-server-cam style="width:auto;padding:6px 10px;background:var(--elevated);border:1px solid var(--line);border-radius:7px;color:var(--ink);font-size:12px">
      <option value=0>Camera 0</option>
    </select>
    <button class="btn small" id="wc-server-start">Start Server Cam</button>
    <span class="rec-label" id="wc-server-rec"><span class=rec-dot></span>Live</span>
    <button class="btn small active-filter" id="wc-s-raw">Raw</button>
    <button class="btn small" id="wc-s-gray">Gray</button>
    <button class="btn small" id="wc-s-edges">Edges</button>
    <button class="btn small" id="wc-s-hsv">HSV</button>
    <button class="btn small" id="wc-s-blur">Blur</button>
    <button class="btn small danger" id="wc-server-stop">Stop</button>
  </div>
</div>
</div>
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
  <script>
  document.querySelectorAll('input[name=scoring_mode]').forEach(r=>{
    r.addEventListener('change',()=>{
      document.getElementById('scoring-upload-area').style.display=
        document.querySelector('input[name=scoring_mode]:checked').value==='upload'?'':'none'
    })
  })
  document.querySelector('select[name=genre]').addEventListener('change', function(){
    var custom = this.value === ''
    document.querySelectorAll('.genre-manual').forEach(function(el){ el.style.display = custom ? '' : 'none' })
  })
  </script>
  <button class=btn type=submit>Generate Trailer</button>
</form>
<div id=tr-area style=display:none>
  <div class=card id=tr-stats></div>
  <div style=margin:10px 0 id=tr-video></div>
  <table id=tr-table><tr><th>Scene</th><th>Start</th><th>End</th><th>Quality</th><th>Used (s)</th></tr></table>
</div>
<div id=tr-prompt class=no-data>Upload a video to generate a trailer from the best scenes.</div>
</div>

<script>
document.getElementById('tf').addEventListener('submit', async function(e){
  e.preventDefault()
  document.getElementById('tr-area').style.display='none'
  document.getElementById('tr-prompt').style.display='block'
  document.getElementById('tr-prompt').textContent='Generating trailer... (this may take a while)'
  let r=await fetch('/api/trailer/generate',{method:'POST',body:new FormData(this)})
  let d=await r.json()
  document.getElementById('tr-prompt').style.display='none'
  if(d.error){document.getElementById('tr-stats').innerHTML='<b>Error:</b> '+d.error; document.getElementById('tr-area').style.display='block'; return}
  document.getElementById('tr-stats').innerHTML='Trailer: '+d.trailer_duration+'s (target '+d.trailer_length+'s) from '+d.selected_scenes+'/'+d.total_scenes+' scenes | Raw video: '+d.video_duration+'s'
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
<p><strong>Webcam</strong>: Client-side via browser <code>getUserMedia</code> + Canvas filters (no server needed)</p>
<p><strong>Player</strong>:</p>
<pre>POST /api/playback/start  + file/url
GET  /api/playback/stream/raw|gray|edges|hsv|blur|face|motion
GET  /api/playback/pause
GET  /api/playback/resume
GET  /api/playback/seek?t=seconds
GET  /api/playback/stop</pre>
</div>

</div><!-- container -->

<script>// webcam
// Webcam (client-side via getUserMedia + Canvas)
(function(){
  let isSecure = location.protocol==='https:' || location.hostname==='localhost' || location.hostname==='127.0.0.1';
  if(isSecure) document.getElementById('wc-controls').style.display='';
  else document.getElementById('wc-https-warn').style.display='block';
})();
let wcStream=null, wcFilter='raw', wcAnimId=null
function setWCActiveBtn(f){
  ['raw','gray','edges','hsv','blur'].forEach(m=>{
    document.getElementById('wc-'+m).classList.toggle('active-filter', m===f)
  })
}
async function startCam(){
  try{
    if(!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia){
      // Try legacy prefixed APIs
      let gum = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia
      if(gum){
        gum({video:true}, function(stream){
          wcStream=stream
          let v=document.getElementById('wc-video')
          v.srcObject=stream
          v.play()
          document.getElementById('wc-rec').classList.add('live')
          drawWC()
        }, function(e){ alert('Camera error: '+e.message) })
        return
      }
      // Check if page is insecure (HTTP non-localhost)
      if(location.protocol!=='https:' && location.hostname!=='localhost' && location.hostname!=='127.0.0.1'){
        alert('Camera requires HTTPS when accessing from another device.\\nOpen https://'+location.hostname+':5001/ instead.')
      } else {
        alert('Camera not supported in this browser.')
      }
      return
    }
    wcStream=await navigator.mediaDevices.getUserMedia({video:true})
    let v=document.getElementById('wc-video')
    v.srcObject=wcStream
    await v.play()
    document.getElementById('wc-rec').classList.add('live')
    drawWC()
  }catch(e){alert('Camera error: '+e.message)}
}
function setWCFilter(f){ wcFilter=f; setWCActiveBtn(f) }
function drawWC(){
  let v=document.getElementById('wc-video')
  let c=document.getElementById('wc-canvas')
  if(!v.videoWidth) return
  c.width=v.videoWidth; c.height=v.videoHeight
  let ctx=c.getContext('2d')
  ctx.drawImage(v,0,0)
  let img=ctx.getImageData(0,0,c.width,c.height)
  let d=img.data
  if(wcFilter=='gray'){
    for(let i=0;i<d.length;i+=4){let g=(d[i]+d[i+1]+d[i+2])/3;d[i]=d[i+1]=d[i+2]=g}
  }else if(wcFilter=='edges'){
    // Simple grayscale then sobel-like: just luma
    for(let i=0;i<d.length;i+=4){let g=(d[i]+d[i+1]+d[i+2])/3;d[i]=d[i+1]=d[i+2]=g>128?255:0}
  }else if(wcFilter=='hsv'){
    for(let i=0;i<d.length;i+=4){
      let r=d[i]/255,g=d[i+1]/255,b=d[i+2]/255
      let mx=Math.max(r,g,b),mn=Math.min(r,g,b),h=0,s=0,v=mx
      let df=mx-mn
      if(df){s=df/mx;if(mx==r)h=((g-b)/df)%6;else if(mx==g)h=(b-r)/df+2;else h=(r-g)/df+4;h/=6}
      d[i]=h*255;d[i+1]=s*255;d[i+2]=v*255
    }
  }else if(wcFilter=='blur'){
    ctx.filter='blur(8px)'
    ctx.drawImage(v,0,0)
    ctx.filter='none'
    img=ctx.getImageData(0,0,c.width,c.height)
    d=img.data
  }
  ctx.putImageData(img,0,0)
  wcAnimId=requestAnimationFrame(drawWC)
}
function stopCam(){
  if(wcStream){wcStream.getTracks().forEach(t=>t.stop());wcStream=null}
  if(wcAnimId){cancelAnimationFrame(wcAnimId);wcAnimId=null}
  document.getElementById('wc-canvas').width=0;document.getElementById('wc-canvas').height=0
  document.getElementById('wc-rec').classList.remove('live')
}
document.getElementById('wc-start').addEventListener('click',startCam)
document.getElementById('wc-raw').addEventListener('click',()=>setWCFilter('raw'))
document.getElementById('wc-gray').addEventListener('click',()=>setWCFilter('gray'))
document.getElementById('wc-edges').addEventListener('click',()=>setWCFilter('edges'))
document.getElementById('wc-hsv').addEventListener('click',()=>setWCFilter('hsv'))
document.getElementById('wc-blur').addEventListener('click',()=>setWCFilter('blur'))
document.getElementById('wc-stop').addEventListener('click',stopCam)

// Server-side webcam
let wcServerMode='raw'
function setWCServerActiveBtn(f){
  ['raw','gray','edges','hsv','blur'].forEach(m=>{
    document.getElementById('wc-s-'+m).classList.toggle('active-filter', m===f)
  })
}
async function loadCameras(){
  try{
    let r=await fetch('/api/webcam/cameras')
    let d=await r.json()
    let sel=document.getElementById('wc-server-cam')
    sel.innerHTML=d.cameras.map(c=>'<option value='+c.id+'>'+c.name+'</option>').join('')||'<option value=0>Camera 0</option>'
  }catch(e){}
}
loadCameras()
document.getElementById('wc-server-start').addEventListener('click',async function(){
  let cam=document.getElementById('wc-server-cam').value
  await fetch('/api/webcam/start',{method:'POST',body:new URLSearchParams({camera:cam})})
  document.getElementById('wc-server-feed').style.display='block'
  document.getElementById('wc-server-feed').src='/api/webcam/stream/'+wcServerMode
  document.getElementById('wc-server-rec').classList.add('live')
})
document.getElementById('wc-server-stop').addEventListener('click',async function(){
  await fetch('/api/webcam/stop')
  document.getElementById('wc-server-feed').style.display='none'
  document.getElementById('wc-server-feed').src=''
  document.getElementById('wc-server-rec').classList.remove('live')
})
;['raw','gray','edges','hsv','blur'].forEach(m=>{
  document.getElementById('wc-s-'+m).addEventListener('click',()=>{
    wcServerMode=m
    setWCServerActiveBtn(m)
    let feed=document.getElementById('wc-server-feed')
    if(feed.style.display!=='none') feed.src='/api/webcam/stream/'+m
  })
})

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
    print(' * HTTPS: https://0.0.0.0:5001/ (for webcam from other devices)')
    print(' * Access from local machine: http://localhost:5000/')
    print(' * Access from other devices: https://YOUR_IP:5001/')
    # Run HTTP on 5000 and HTTPS on 5001
    import ssl, threading
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain('cert.pem', 'key.pem')
    threading.Thread(target=lambda: app.run(debug=False, host='0.0.0.0', port=5000, threaded=True, use_reloader=False), daemon=True).start()
    app.run(debug=False, host='0.0.0.0', port=5001, ssl_context=ctx, threaded=True, use_reloader=False)
