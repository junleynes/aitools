import os, cv2, numpy as np, tempfile, urllib.request, threading, time, pathlib, base64, requests, subprocess, shutil
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from flask import Flask, render_template_string, request, send_from_directory, jsonify, Response
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2GB
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'}

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

@app.route('/api/playback/start', methods=['POST'])
def pb_start():
    global _pb_cap
    path, err = load_video(request)
    if not path:
        return jsonify(error=err), 400
    with _pb_lock:
        if _pb_cap:
            _pb_cap.release()
        _pb_cap = cv2.VideoCapture(path)
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

    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total // num_frames, 1) if total > num_frames else 1
    fps = cap.get(cv2.CAP_PROP_FPS)

    results = []
    for i in range(0, total, step):
        if len(results) >= num_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        ts = round(i / fps, 2) if fps > 0 else 0
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64 = base64.b64encode(buf.tobytes()).decode()

        try:
            r = requests.post(f'{OLLAMA_URL}/api/generate', json={
                'model': model, 'prompt': prompt, 'stream': False,
                'images': [b64]
            }, timeout=60)
            data = r.json()
            resp = data.get('response', '')
        except Exception as e:
            resp = f'Error: {e}'

        results.append({'frame_idx': i, 'time_sec': ts, 'ollama_response': resp})

    cap.release()
    return jsonify(frames_analyzed=len(results), results=results)

@app.errorhandler(413)
def too_large(e):
    return jsonify(error='File too large (max 2GB).'), 413

@app.route('/api/vision/models')
def api_vision_models():
    try:
        r = requests.get(f'{OLLAMA_URL}/api/tags', timeout=10)
        models = [m['name'] for m in r.json().get('models', []) if 'vision' in str(m.get('capabilities', []))]
        return jsonify(models=models)
    except:
        return jsonify(models=[])

# ---- Trailer Generator (ffmpeg) ----

@app.route('/api/trailer/generate', methods=['POST'])
def api_trailer():
    path, err = load_video(request)
    if not path:
        return jsonify(error=err), 400

    mode = request.form.get('mode', 'auto')
    trailer_pct = max(5, min(50, int(request.form.get('trailer_pct', 20))))
    model = request.form.get('model', 'qwen2.5vl:7b')
    end_card_path = None
    schedule_card_path = None
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
        'Rate this scene 1-5 for use in a movie trailer. Consider action, emotion, and visual quality. Reply only with a number 1-5.')

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps if fps else 0
    cap.release()

    trailer_duration = max(5, min(video_duration * trailer_pct / 100, 120))
    if video_duration < trailer_duration * 1.5:
        return jsonify(error=f'Video is only {video_duration:.1f}s long, but trailer ({trailer_duration}s) needs the video to be at least {trailer_duration*1.5:.0f}s. Upload a longer video or reduce the trailer %.'), 400

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
        scenes_data.append({
            'start': start.get_seconds(), 'end': end.get_seconds(),
            'start_f': start.get_frames(), 'end_f': end.get_frames(),
            'duration': dur, 'laplacian': round(lap, 2), 'brightness': round(bri, 1),
            'frame': frame, 'frame_idx': mid_f,
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
                }, timeout=30)
                txt = r.json().get('response', '')
                import re
                m = re.search(r'[1-5]', txt)
                s['total_score'] = s['quality_score'] + (int(m.group()) if m else 3)
            except:
                s['total_score'] = s['quality_score'] + 3
    else:
        for s in scenes_data:
            s['total_score'] = s['quality_score']

    scenes_data.sort(key=lambda x: x['total_score'], reverse=True)

    selected = []
    total_sel = 0
    for s in scenes_data:
        if total_sel >= trailer_duration:
            break
        seg_dur = min(s['duration'], trailer_duration - total_sel)
        selected.append({**s, 'selected_dur': seg_dur})
        total_sel += seg_dur

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
                            '-an', out_seg], capture_output=True, text=True)
        if os.path.exists(out_seg) and os.path.getsize(out_seg) > 0:
            seg_files.append(out_seg)
        elif r.returncode != 0:
            print(f'FFMPEG seg extraction error: {r.stderr[:500]}')

    card_files = []
    if end_card_path and os.path.exists(end_card_path):
        card_files.append(end_card_path)
    if schedule_card_path and os.path.exists(schedule_card_path):
        card_files.append(schedule_card_path)

    all_inputs = seg_files + card_files
    n_total = len(all_inputs)

    out_path = os.path.join(app.config['UPLOAD_FOLDER'], f'trailer_{base_ts}.mp4')

    if n_total == 1:
        r = subprocess.run([FFMPEG, '-y', '-i', all_inputs[0],
                            '-c:v', 'libx264', '-preset', 'fast', '-crf', '22',
                            '-pix_fmt', 'yuv420p', out_path], capture_output=True, text=True)
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

        xfade_dur = 0.3
        filter_parts = []
        for i in range(n_total - 1):
            offset = sum(durations[:i + 1]) - (i + 1) * xfade_dur
            filter_parts.append(f'[{i}][{i+1}]xfade=transition=fade:duration={xfade_dur}:offset={offset}[v{i+1}]')

        cmd = [FFMPEG, '-y']
        for f in all_inputs:
            cmd.extend(['-i', f])
        cmd.extend(['-filter_complex', ';'.join(filter_parts)])
        last_label = f'[v{n_total-1}]'
        cmd.extend(['-map', last_label])
        cmd.extend(['-c:v', 'libx264', '-preset', 'fast', '-crf', '22', '-pix_fmt', 'yuv420p', '-an', out_path])
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

    return jsonify(status='ok', trailer_url=f'/uploads/{filename}',
                   total_scenes=len(scene_list), selected_scenes=len(selected),
                   trailer_duration=round(total_sel, 1),
                   video_duration=round(video_duration, 1),
                   trailer_pct=trailer_pct,
                   scenes=[{
                       'scene': i+1, 'start': round(s['start'], 1), 'end': round(s['end'], 1),
                       'quality': s['total_score'], 'duration': round(s['selected_dur'], 1)
                   } for i, s in enumerate(selected)])

# ---- UI ----

UI = '''
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>AI Video Toolkit - OpenCV, PySceneDetect, Ollama Vision & FFmpeg</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#f0f2f5;color:#333;padding:30px;font-family:Segoe UI,sans-serif}
.container{max-width:1100px;margin:0 auto}
h1 .small{font-size:14px;color:#666;font-weight:400}
.tabs{display:flex;gap:2px;margin:20px 0;flex-wrap:wrap}
.tab{padding:12px 20px;cursor:pointer;background:#ddd;border:none;font-size:15px;border-radius:6px 6px 0 0;white-space:nowrap;font-family:inherit;outline:none;user-select:none}
.tab:hover{background:#ccc}
.tab.active{background:#fff;font-weight:700}
.tab-sub{font-size:11px;color:#666;display:block;font-weight:400}
.panel{display:none;background:#fff;border-radius:0 6px 6px 6px;box-shadow:0 2px 8px rgba(0,0,0,.1);padding:20px;margin-bottom:20px}
.panel.active{display:block}
.btn{background:#007bff;color:#fff;border:none;padding:8px 16px;border-radius:4px;cursor:pointer;font-size:14px;display:inline-block;text-decoration:none}
.btn:hover{background:#0056b3}.btn.danger{background:#dc3545}
.btn.small{padding:5px 12px;font-size:13px}
label{display:block;margin:10px 0 4px;font-weight:700}
input[type=file]{display:block;margin:10px 0}
input[type=url],input[type=number]{width:100%;padding:8px;border:1px solid #ccc;border-radius:4px;margin:8px 0;font-size:15px}
.or{text-align:center;color:#999;margin:8px 0;font-size:13px}
.card{background:#f8f9fa;border-radius:6px;padding:15px;margin:10px 0}
table{width:100%;border-collapse:collapse;margin:12px 0;font-size:14px}
th,td{border:1px solid #ddd;padding:6px 10px;text-align:left}
th{background:#f0f0f0}
.filters{display:flex;gap:6px;margin:12px 0;flex-wrap:wrap}
.stream-wrap{text-align:center}
.stream-wrap img{max-width:100%;border:1px solid #ddd;border-radius:4px;max-height:550px}
.info{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px;margin:10px 0}
.info-item{background:#f0f0f0;padding:6px 10px;border-radius:4px;font-size:13px}
.info-item strong{display:inline-block;min-width:60px}
.no-data{text-align:center;padding:30px;color:#888;font-size:15px}
.progress{width:100%;margin:8px 0;cursor:pointer}
</style>
</head>
<body>
<div class="container">
<h1>AI Video Toolkit <small>OpenCV + PySceneDetect + Ollama Vision + FFmpeg</small></h1>
<div class="tabs">
  <div class="tab active" onclick="switchTab('p-upload',this)" role="button" tabindex="0">Upload<div class=tab-sub>OpenCV+PySceneDetect</div></div>
  <div class="tab" onclick="switchTab('p-player',this)" role="button" tabindex="0">Player<div class=tab-sub>OpenCV</div></div>
  <div class="tab" onclick="switchTab('p-webcam',this)" role="button" tabindex="0">Webcam<div class=tab-sub>Browser JS+Canvas</div></div>
  <div class="tab" onclick="switchTab('p-vision',this)" role="button" tabindex="0">Vision AI<div class=tab-sub>Ollama</div></div>
  <div class="tab" onclick="switchTab('p-trailer',this)" role="button" tabindex="0">Trailer<div class=tab-sub>All+FFmpeg</div></div>
  <div class="tab" onclick="switchTab('p-api',this)" role="button" tabindex="0">API</div>
</div>

<script>function switchTab(id,btn){document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));btn.classList.add('active');document.getElementById(id).classList.add('active')}</script>

<!-- Upload -->
<div id="p-upload" class="panel active">
<h2>Upload & Analyze</h2>
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
<div id="p-player" class="panel">
<h2>Video Player</h2>
<form id=pf enctype=multipart/form-data>
  <input type=file name=file accept=video/*>
  <div class=or>or</div>
  <input type=url name=video_url placeholder="https://example.com/video.mp4">
  <button class=btn type=submit>Load</button>
</form>
<div id=pb-area style=display:none>
  <div class=info id=pb-info></div>
  <div class=stream-wrap><img id=pb-feed></div>
  <input type=range class=progress id=pb-progress min=0 max=100 value=0>
  <div class=filters>
    <button class="btn small" id="pb-raw">Raw</button>
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
<div id="p-webcam" class="panel">
<h2>Live Webcam <small>Runs in your browser</small></h2>
<div class=filters>
  <button class="btn small" id="wc-start">Start Camera</button>
  <button class="btn small" id="wc-raw">Raw</button>
  <button class="btn small" id="wc-gray">Gray</button>
  <button class="btn small" id="wc-edges">Edges</button>
  <button class="btn small" id="wc-hsv">HSV</button>
  <button class="btn small" id="wc-blur">Blur</button>
  <button class="btn small danger" id="wc-stop">Stop</button>
</div>
<div class=stream-wrap>
  <video id=wc-video autoplay playsinline style="display:none"></video>
  <canvas id=wc-canvas></canvas>
</div>
</div>

<!-- Trailer -->
<div id="p-trailer" class="panel">
<h2>Trailer Generator</h2>
<p>Uses PySceneDetect to find scene boundaries, OpenCV to score quality (sharpness, brightness), and optionally Ollama Vision for AI scoring.</p>
<form id=tf enctype=multipart/form-data>
  <input type=file name=file accept=video/*>
  <div class=or>or</div>
  <input type=url name=video_url placeholder="https://example.com/video.mp4">
  <label>Mode:</label>
  <select name=mode>
    <option value=auto>Auto (OpenCV quality scoring)</option>
    <option value=ai>AI (OpenCV + Ollama Vision scoring)</option>
  </select>
  <label>Trailer length (% of video):</label>
  <input type=range name=trailer_pct id=trailer-pct value=20 min=5 max=50 oninput="document.getElementById('tr-pct-val').textContent=this.value+'%'">
  <span id=tr-pct-val>20%</span>
  <label>AI model (if AI mode):</label>
  <select name=model id=trailer-model><option value="">Loading...</option></select>
  <label>End card video (optional):</label>
  <input type=file name=end_card_video accept=video/*>
  <label>Schedule card video (optional):</label>
  <input type=file name=schedule_video accept=video/*>
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
  document.getElementById('tr-prompt').textContent='Generating trailer... (this may take a while)'
  let r=await fetch('/api/trailer/generate',{method:'POST',body:new FormData(this)})
  let d=await r.json()
  document.getElementById('tr-prompt').style.display='none'
  if(d.error){document.getElementById('tr-stats').innerHTML='<b>Error:</b> '+d.error; document.getElementById('tr-area').style.display='block'; return}
  document.getElementById('tr-stats').innerHTML='Trailer: '+d.trailer_duration+'s ('+d.trailer_pct+'% of '+d.video_duration+'s video) from '+d.selected_scenes+'/'+d.total_scenes+' scenes'
  document.getElementById('tr-video').innerHTML='<video controls style=max-width:100%><source src="'+d.trailer_url+'" type="video/mp4"></video>'
  let rows=''
  d.scenes.forEach(s=>{rows+='<tr><td>'+s.scene+'</td><td>'+s.start+'s</td><td>'+s.end+'s</td><td>'+s.quality+'</td><td>'+s.duration+'</td></tr>'})
  document.getElementById('tr-table').innerHTML='<tr><th>#</th><th>Start</th><th>End</th><th>Score</th><th>Used</th></tr>'+rows
  document.getElementById('tr-area').style.display='block'
})
</script>

<!-- Vision -->
<div id="p-vision" class="panel">
<h2>Vision AI with Ollama</h2>
<form id=vf enctype=multipart/form-data>
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

<script>
// Load Ollama vision models into dropdowns
async function loadModels(){
  try{
    let r=await fetch('/api/vision/models')
    let d=await r.json()
    let opts=d.models.map(m=>'<option value="'+m+'">'+m+'</option>').join('')
    if(!opts) opts='<option value="qwen2.5vl:7b">qwen2.5vl:7b</option><option value="qwen3-vl:8b">qwen3-vl:8b</option>'
    document.getElementById('vision-model').innerHTML=opts
    document.getElementById('trailer-model').innerHTML=opts
  }catch(e){
    document.getElementById('vision-model').innerHTML='<option value="qwen2.5vl:7b">qwen2.5vl:7b</option><option value="qwen3-vl:8b">qwen3-vl:8b</option>'
    document.getElementById('trailer-model').innerHTML=document.getElementById('vision-model').innerHTML
  }
}
loadModels()

document.getElementById('vf').addEventListener('submit', async function(e){
  e.preventDefault()
  document.getElementById('vr-area').style.display='none'
  document.getElementById('vr-prompt').textContent='Analyzing...'
  let r=await fetch('/api/vision/analyze',{method:'POST',body:new FormData(this)})
  let d=await r.json()
  document.getElementById('vr-prompt').style.display='none'
  if(d.error){document.getElementById('vr-result').innerHTML='<b>Error:</b> '+d.error; document.getElementById('vr-area').style.display='block'; return}
  let h='<table><tr><th>#</th><th>Time</th><th>AI Response</th></tr>'
  d.results.forEach(r=>{h+='<tr><td>'+r.frame_idx+'</td><td>'+r.time_sec+'s</td><td>'+r.ollama_response+'</td></tr>'})
  h+='</table>'
  document.getElementById('vr-result').innerHTML='<p>Analyzed '+d.frames_analyzed+' frames.</p>'+h
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
let wcStream=null, wcFilter='raw', wcAnimId=null
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
          drawWC()
        }, function(e){ alert('Camera error: '+e.message) })
        return
      }
      // Check if page is insecure (HTTP non-localhost)
      if(location.protocol!=='https:' && location.hostname!=='localhost' && location.hostname!=='127.0.0.1'){
        alert('Camera requires HTTPS when accessing from another device.\nOpen https://'+location.hostname+':5001/ instead.')
      } else {
        alert('Camera not supported in this browser.')
      }
      return
    }
    wcStream=await navigator.mediaDevices.getUserMedia({video:true})
    let v=document.getElementById('wc-video')
    v.srcObject=wcStream
    await v.play()
    drawWC()
  }catch(e){alert('Camera error: '+e.message)}
}
function setWCFilter(f){ wcFilter=f }
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
    wcFilter='raw'; // single frame blur, revert
  }
  ctx.putImageData(img,0,0)
  wcAnimId=requestAnimationFrame(drawWC)
}
function stopCam(){
  if(wcStream){wcStream.getTracks().forEach(t=>t.stop());wcStream=null}
  if(wcAnimId){cancelAnimationFrame(wcAnimId);wcAnimId=null}
  document.getElementById('wc-canvas').width=0;document.getElementById('wc-canvas').height=0
}
document.getElementById('wc-start').addEventListener('click',startCam)
document.getElementById('wc-raw').addEventListener('click',()=>setWCFilter('raw'))
document.getElementById('wc-gray').addEventListener('click',()=>setWCFilter('gray'))
document.getElementById('wc-edges').addEventListener('click',()=>setWCFilter('edges'))
document.getElementById('wc-hsv').addEventListener('click',()=>setWCFilter('hsv'))
document.getElementById('wc-blur').addEventListener('click',()=>setWCFilter('blur'))
document.getElementById('wc-stop').addEventListener('click',stopCam)

// playback wire-up
['raw','gray','edges','hsv','blur','face','motion'].forEach(m=>{
  document.getElementById('pb-'+m).addEventListener('click',()=>{
    document.getElementById('pb-feed').src='/api/playback/stream/'+m
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
    threading.Thread(target=lambda: app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False), daemon=True).start()
    app.run(debug=False, host='0.0.0.0', port=5001, ssl_context=ctx, use_reloader=False)
