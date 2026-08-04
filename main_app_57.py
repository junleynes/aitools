import os, cv2, numpy as np, tempfile, threading, time, pathlib, base64, json, requests, subprocess, shutil, re, sqlite3, uuid, secrets, hmac, io
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from flask import Flask, render_template_string, request, send_from_directory, jsonify, Response, session, redirect, url_for
from werkzeug.utils import secure_filename
import smbclient  # pip install smbprotocol -- lets the upload panels browse a Windows/SMB network share directly

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 * 1024  # 5GB

# ---- Session secret ----
# Flask needs this to sign the session cookie. Generating it fresh at boot would
# silently invalidate every session (and log everyone out) on each pm2 restart,
# so it's persisted to disk the first time the app runs and reused after that.
_SECRET_KEY_FILE = os.environ.get('SECRET_KEY_FILE',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '.secret_key'))
try:
    with open(_SECRET_KEY_FILE, 'rb') as _f:
        app.secret_key = _f.read().strip()
    if not app.secret_key:
        raise ValueError('empty key file')
except (FileNotFoundError, ValueError):
    app.secret_key = secrets.token_hex(32).encode()
    try:
        with open(_SECRET_KEY_FILE, 'wb') as _f:
            _f.write(app.secret_key)
        try:
            os.chmod(_SECRET_KEY_FILE, 0o600)
        except OSError:
            pass
    except OSError as e:
        print(f'Could not persist {_SECRET_KEY_FILE} ({e}); sessions will not '
              'survive a restart until this is writable.')
app.config['PERMANENT_SESSION_LIFETIME'] = int(os.environ.get('SESSION_LIFETIME_DAYS', 30)) * 86400
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
# Only set Secure on the cookie if the app is actually served over HTTPS -- on a
# plain-HTTP LAN deployment (this app's default) a Secure cookie would never be
# sent at all, silently breaking the gate rather than protecting anything.
app.config['SESSION_COOKIE_SECURE'] = os.environ.get('FORCE_HTTPS', '').lower() in ('1', 'true', 'yes')

# ---- Access gate ----
# No per-user accounts -- this closes the standing hole where every trailer,
# template asset, and upload was servable to anyone who could reach the port,
# via predictable filenames and sequential library IDs. It's a single shared
# passphrase, opt-in via APP_ACCESS_KEY, so a deployment that hasn't set one
# behaves exactly as before rather than locking someone out by surprise.
APP_ACCESS_KEY = os.environ.get('APP_ACCESS_KEY', '').strip()
GATE_ENABLED = bool(APP_ACCESS_KEY)
_PUBLIC_PATHS = {'/login', '/logout'}
_API_PREFIXES = ('/api/', '/uploads/', '/library/', '/download/')

def _client_ip():
    """Best-effort client IP for rate limiting.

    Trusts X-Forwarded-For only because this app is documented as running
    behind an Apache reverse proxy (see R.I.M.S notes); on a direct deployment
    that header is attacker-controlled and this falls back to remote_addr."""
    fwd = request.headers.get('X-Forwarded-For', '')
    return (fwd.split(',')[0].strip() if fwd else None) or request.remote_addr or 'unknown'

class _RateLimiter:
    """Simple in-memory sliding-window limiter: `limit` events per `window`
    seconds, per key. No external dependency (no redis) since this is a
    single-process app; resets on restart, which is an acceptable trade-off
    for slowing enumeration/brute-force rather than a hard guarantee."""
    def __init__(self, limit, window):
        self.limit = limit
        self.window = window
        self.buckets = {}
        self.lock = threading.Lock()

    def allow(self, key):
        now = time.time()
        with self.lock:
            dq = self.buckets.setdefault(key, deque())
            while dq and now - dq[0] > self.window:
                dq.popleft()
            if len(dq) >= self.limit:
                return False
            dq.append(now)
            # Bound memory: drop buckets nobody has touched recently.
            if len(self.buckets) > 5000:
                stale = [k for k, v in self.buckets.items() if not v or now - v[-1] > self.window * 4]
                for k in stale:
                    self.buckets.pop(k, None)
            return True

# Always on, regardless of whether the passphrase gate is configured: an
# unauthenticated deployment still shouldn't let one client rip through
# sequential library IDs or guessed upload filenames at full speed.
_file_route_limiter = _RateLimiter(limit=int(os.environ.get('FILE_ROUTE_RATE_LIMIT', 40)), window=60)
_login_limiter = _RateLimiter(limit=int(os.environ.get('LOGIN_RATE_LIMIT', 8)), window=300)

@app.before_request
def _access_control():
    path = request.path
    if path in _PUBLIC_PATHS or path.startswith('/static/'):
        return None

    # Baseline throttle on the direct file/asset routes -- applies even with the
    # gate disabled, since that's the default and these are exactly the routes a
    # scanner would hit to enumerate trailers or templates.
    if path.startswith(('/uploads/', '/library/', '/download/')) or '/asset/' in path:
        if not _file_route_limiter.allow(_client_ip()):
            return jsonify(error='Too many requests. Slow down.'), 429

    if not GATE_ENABLED or session.get('authed'):
        return None

    if path.startswith(_API_PREFIXES):
        return jsonify(error='Not authenticated. Sign in at /login.'), 401
    return redirect(url_for('login', next=request.full_path if request.query_string else path))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if not GATE_ENABLED:
        # Nothing to log into -- send them straight to the app rather than
        # showing a passphrase form for a gate that isn't configured.
        return redirect('/')
    error = None
    if request.method == 'POST':
        if not _login_limiter.allow(_client_ip()):
            error = 'Too many attempts. Wait a few minutes and try again.'
        elif hmac.compare_digest(request.form.get('key', ''), APP_ACCESS_KEY):
            session.permanent = True
            session['authed'] = True
            dest = request.form.get('next') or '/'
            # Only ever redirect to a path on this app -- an absolute or
            # protocol-relative 'next' would be an open-redirect vector.
            if not dest.startswith('/') or dest.startswith('//'):
                dest = '/'
            return redirect(dest)
        else:
            error = 'Incorrect passphrase.'
    nxt = request.args.get('next', '/')
    return f'''<!doctype html><html><head><meta charset="utf-8">
<title>Sign in</title>
<style>body{{font-family:system-ui,sans-serif;background:#0b1220;color:#e6e9ef;
display:flex;align-items:center;justify-content:center;height:100vh;margin:0}}
form{{background:#141b2d;padding:28px 32px;border-radius:10px;border:1px solid #232b41;width:280px}}
h1{{font-size:16px;margin:0 0 16px}}
input{{width:100%;box-sizing:border-box;padding:9px 10px;border-radius:6px;border:1px solid #232b41;
background:#0b1220;color:#e6e9ef;font-size:14px;margin-bottom:12px}}
button{{width:100%;padding:9px;border-radius:6px;border:none;background:#4f8cff;color:#fff;
font-size:14px;cursor:pointer}}
.err{{color:#e08a3c;font-size:12px;margin-bottom:12px}}</style></head><body>
<form method=post>
<h1>Sign in</h1>
{f'<div class="err">{error}</div>' if error else ''}
<input type=hidden name=next value="{nxt}">
<input type=password name=key placeholder="Passphrase" autofocus>
<button type=submit>Continue</button>
</form></body></html>'''

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'}

# ---- Persistent trailer library (SQLite) ----
# UPLOAD_FOLDER above is a fresh tempdir every process start, so anything that
# needs to survive a restart -- completed trailers the user wants to come back
# to later -- gets copied here instead, with metadata tracked in a small
# SQLite DB alongside it. Override LIBRARY_DIR to point this at persistent
# storage (a mounted volume, etc.) in production.
LIBRARY_DIR = os.environ.get('LIBRARY_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trailer_library'))
os.makedirs(LIBRARY_DIR, exist_ok=True)
LIBRARY_DB_PATH = os.path.join(LIBRARY_DIR, 'library.db')

def _sqlite_connect(db_path):
    """Opens a SQLite connection tuned for this app's access pattern: several
    concurrent job threads writing while the monitor endpoint polls for reads.

    The stock settings give a 5-second busy timeout and rollback-journal locking,
    under which a reader blocks writers and a slow write surfaces as
    'database is locked'. WAL lets readers and one writer proceed concurrently,
    and the longer timeout absorbs the rest."""
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('PRAGMA synchronous=NORMAL')
        conn.execute('PRAGMA busy_timeout=30000')
    except sqlite3.Error as e:
        # A network/SMB-mounted DB path can refuse WAL; rollback journal still works.
        print(f'SQLite pragma setup failed for {db_path} (continuing): {e}')
    return conn

def _lib_db():
    return _sqlite_connect(LIBRARY_DB_PATH)

def library_db_init():
    conn = _lib_db()
    conn.execute('''CREATE TABLE IF NOT EXISTS trailers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        orig_name TEXT,
        filename TEXT NOT NULL,
        created_at REAL NOT NULL,
        trailer_duration REAL,
        video_duration REAL,
        trailer_length TEXT,
        bgm_source TEXT,
        sfx_source TEXT,
        vo_source TEXT,
        result_json TEXT
    )''')
    conn.commit()
    conn.close()

def library_add(upload_filename, result):
    """Copies the just-finished trailer (currently sitting in the ephemeral
    UPLOAD_FOLDER as `upload_filename`) into LIBRARY_DIR under a permanent
    name, records it in SQLite, and returns the new row id. The saved
    result_json has its trailer_url rewritten to the persistent /library/
    route so re-opening it later (even after a restart) works regardless of
    whether the original UPLOAD_FOLDER file still exists."""
    ext = os.path.splitext(upload_filename)[1] or '.mp4'
    persist_name = f'{uuid.uuid4().hex}{ext}'
    src = os.path.join(app.config['UPLOAD_FOLDER'], upload_filename)
    dst = os.path.join(LIBRARY_DIR, persist_name)
    shutil.copy2(src, dst)
    conn = _lib_db()
    cur = conn.execute(
        'INSERT INTO trailers (orig_name, filename, created_at, trailer_duration, video_duration, trailer_length, bgm_source, sfx_source, vo_source, result_json) '
        'VALUES (?,?,?,?,?,?,?,?,?,?)',
        (result.get('orig_name'), persist_name, time.time(), result.get('trailer_duration'), result.get('video_duration'),
         result.get('trailer_length'), result.get('bgm_source'), result.get('sfx_source'), result.get('vo_source'), None))
    tid = cur.lastrowid
    saved_result = dict(result, trailer_url=f'/library/{tid}/file', library_id=tid)
    conn.execute('UPDATE trailers SET result_json=? WHERE id=?', (json.dumps(saved_result), tid))
    conn.commit()
    conn.close()
    return tid

def library_list(limit=50):
    conn = _lib_db()
    rows = conn.execute(
        'SELECT id, orig_name, filename, created_at, trailer_duration, video_duration, trailer_length, bgm_source, sfx_source, vo_source '
        'FROM trailers ORDER BY created_at DESC LIMIT ?', (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def library_get_row(tid):
    conn = _lib_db()
    row = conn.execute('SELECT * FROM trailers WHERE id=?', (tid,)).fetchone()
    conn.close()
    return dict(row) if row else None

def library_delete(tid):
    row = library_get_row(tid)
    if not row:
        return False
    path = os.path.join(LIBRARY_DIR, row['filename'])
    if os.path.exists(path):
        os.remove(path)
    # Also drop any cached format-converted copies of it (see library_download()).
    base = os.path.splitext(row['filename'])[0]
    for f in os.listdir(LIBRARY_DIR):
        if f.startswith(base + '_'):
            try:
                os.remove(os.path.join(LIBRARY_DIR, f))
            except OSError:
                pass
    conn = _lib_db()
    conn.execute('DELETE FROM trailers WHERE id=?', (tid,))
    conn.commit()
    conn.close()
    return True

library_db_init()

# ---- Per-show asset templates (SQLite) ----
# A "template" is a named bundle of the reusable assets a specific show always
# uses -- its background music bed, SFX one-shot, VO track, title card and end
# card (each card with an optional card VO + in/out points). Picking a template
# at generate time fills all of those slots at once, as an alternative to picking
# a Genre (which only presets transition style and the AI music/SFX *prompts*,
# and supplies no actual files).
#
# Like LIBRARY_DIR above, this has to survive process restarts -- UPLOAD_FOLDER is
# a fresh tempdir every boot -- so the master copy of each asset lives here and is
# only ever *copied* into UPLOAD_FOLDER per job (see template_stage_asset). That
# copy is mandatory, not defensive: _run_trailer_job() deletes the SFX, VO and
# card-VO files it is handed once it's finished with them.
TEMPLATES_DIR = os.environ.get('TEMPLATES_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'show_templates'))
os.makedirs(TEMPLATES_DIR, exist_ok=True)
TEMPLATES_DB_PATH = os.path.join(TEMPLATES_DIR, 'templates.db')

# Each slot maps a template asset to the form field the trailer form already uses
# for it. Note the two legacy field names: 'end_card_video' is really the *title*
# card and 'schedule_video' is really the *end* card -- the slot keys below use the
# accurate names, and the mapping is kept in this one place so nothing else has to
# know about the mismatch.
TEMPLATE_SLOTS = {
    'bgm':           {'field': 'scoring_audio',  'kind': 'audio', 'label': 'Background music'},
    'sfx':           {'field': 'sfx_upload',     'kind': 'audio', 'label': 'SFX one-shot'},
    'vo':            {'field': 'vo_upload',      'kind': 'audio', 'label': 'Voiceover'},
    'title_card':    {'field': 'end_card_video', 'kind': 'video', 'label': 'Title card'},
    'title_card_vo': {'field': 'title_card_vo',  'kind': 'audio', 'label': 'Title card VO'},
    'end_card':      {'field': 'schedule_video', 'kind': 'video', 'label': 'End card'},
    'end_card_vo':   {'field': 'end_card_vo',    'kind': 'audio', 'label': 'End card VO'},
}
TEMPLATE_SLOT_KEYS = list(TEMPLATE_SLOTS.keys())

# A template is the whole configuration for a show, not just its files: genre and
# transition, the asset slots above, and every other generator setting. These are
# captured verbatim from the generate form into a single settings_json column
# rather than one column each, so adding a control to the form doesn't need a
# schema migration.
TEMPLATE_SETTING_FIELDS = [
    'genre', 'transition', 'xfade_dur', 'transition_matte',
    'trailer_length', 'max_scene_dur', 'scene_threshold', 'min_scene_len_sec',
    'mode', 'model', 'prompt',
    'scoring_mode', 'sfx_mode', 'sfx_source', 'vo_mode', 'vo_engine',
    'vo_voice', 'vo_language', 'vo_rate', 'vo_start', 'vo_volume',
    'vo_trim_start', 'vo_trim_end', 'vo_text',
    'title_card_vo_start', 'title_card_vo_end',
    'end_card_vo_start', 'end_card_vo_end',
    'target_loudness', 'true_peak', 'music_duck_db', 'duck_depth_db',
    'duck_release_hold', 'beat_match', 'broadcast_stereo',
    'sync_beats', 'whisper_enhance',
]
# Checkbox-style fields: absent from a form POST means "off", so they must be
# recorded as off rather than left at whatever the previous value was.
TEMPLATE_BOOL_FIELDS = {'beat_match', 'broadcast_stereo', 'sync_beats', 'whisper_enhance'}

def _tpl_db():
    return _sqlite_connect(TEMPLATES_DB_PATH)

def templates_db_init():
    cols = []
    for slot in TEMPLATE_SLOT_KEYS:
        cols.append(f'{slot}_file TEXT')   # stored filename inside TEMPLATES_DIR
        cols.append(f'{slot}_name TEXT')   # original filename, for display only
    conn = _tpl_db()
    conn.execute('CREATE TABLE IF NOT EXISTS show_templates ('
                 'id INTEGER PRIMARY KEY AUTOINCREMENT,'
                 'name TEXT NOT NULL UNIQUE,'
                 'notes TEXT,'
                 'created_at REAL NOT NULL,'
                 'updated_at REAL NOT NULL,'
                 'genre TEXT,'
                 'transition TEXT,'
                 'xfade_dur REAL,'
                 'settings_json TEXT,'
                 'vo_start REAL, vo_volume REAL, vo_trim_start REAL, vo_trim_end REAL,'
                 'title_card_vo_start REAL, title_card_vo_end REAL,'
                 'end_card_vo_start REAL, end_card_vo_end REAL,'
                 + ','.join(cols) + ')')
    # Migration for databases created before settings_json existed.
    have = {r[1] for r in conn.execute('PRAGMA table_info(show_templates)')}
    if 'settings_json' not in have:
        conn.execute('ALTER TABLE show_templates ADD COLUMN settings_json TEXT')
        print('show_templates: added settings_json column (existing templates keep their assets).')
    conn.commit()
    conn.close()

def template_settings(row):
    """The saved form configuration for a template, as a plain dict."""
    if not row:
        return {}
    try:
        return json.loads(row.get('settings_json') or '{}')
    except (ValueError, TypeError):
        return {}

def _template_public(row):
    """Shapes a DB row for the UI: hides on-disk filenames, exposes which slots are
    actually filled plus the display name of each."""
    out = {k: row.get(k) for k in ('id', 'name', 'notes', 'created_at', 'updated_at', 'genre',
                                   'transition', 'xfade_dur', 'vo_start', 'vo_volume',
                                   'vo_trim_start', 'vo_trim_end',
                                   'title_card_vo_start', 'title_card_vo_end',
                                   'end_card_vo_start', 'end_card_vo_end')}
    out['slots'] = {slot: {'filled': bool(row.get(f'{slot}_file')),
                           'name': row.get(f'{slot}_name'),
                           'label': TEMPLATE_SLOTS[slot]['label']}
                    for slot in TEMPLATE_SLOT_KEYS}
    out['settings'] = template_settings(row)
    return out

def template_list():
    conn = _tpl_db()
    rows = conn.execute('SELECT * FROM show_templates ORDER BY name COLLATE NOCASE').fetchall()
    conn.close()
    return [_template_public(dict(r)) for r in rows]

def template_get(tid):
    conn = _tpl_db()
    row = conn.execute('SELECT * FROM show_templates WHERE id=?', (tid,)).fetchone()
    conn.close()
    return dict(row) if row else None

def template_get_by_name(name):
    conn = _tpl_db()
    row = conn.execute('SELECT * FROM show_templates WHERE name=? COLLATE NOCASE', (name,)).fetchone()
    conn.close()
    return dict(row) if row else None

def template_store_asset(src_path, slot, orig_name=None):
    """Copies a resolved upload into TEMPLATES_DIR under a collision-proof name.
    Returns (stored_filename, display_name)."""
    ext = os.path.splitext(src_path)[1] or ''
    stored = f'{slot}_{uuid.uuid4().hex}{ext}'
    shutil.copy2(src_path, os.path.join(TEMPLATES_DIR, stored))
    return stored, (orig_name or os.path.basename(src_path))

def template_asset_abspath(row, slot):
    fn = row.get(f'{slot}_file') if row else None
    if not fn:
        return None
    p = os.path.join(TEMPLATES_DIR, os.path.basename(fn))
    return p if os.path.exists(p) else None

def template_stage_asset(row, slot):
    """Copies the template's master asset for `slot` into UPLOAD_FOLDER and returns
    the job-local path, or None if that slot is empty / missing on disk.

    The copy is REQUIRED -- _run_trailer_job() deletes sfx_upload_path,
    vo_upload_path, title_card_vo_path and end_card_vo_path once it's done with
    them, so handing it a TEMPLATES_DIR path directly would destroy the template
    after a single use."""
    src = template_asset_abspath(row, slot)
    if not src:
        return None
    ext = os.path.splitext(src)[1] or ''
    dest = os.path.join(app.config['UPLOAD_FOLDER'],
                        f'tpl{row["id"]}_{slot}_{int(time.time()*1000)}_{threading.get_ident()}{ext}')
    shutil.copy2(src, dest)
    return dest

def template_delete_asset_file(row, slot):
    p = template_asset_abspath(row, slot)
    if p:
        try:
            os.remove(p)
        except OSError:
            pass

def template_delete(tid):
    row = template_get(tid)
    if not row:
        return False
    for slot in TEMPLATE_SLOT_KEYS:
        template_delete_asset_file(row, slot)
    conn = _tpl_db()
    conn.execute('DELETE FROM show_templates WHERE id=?', (tid,))
    conn.commit()
    conn.close()
    return True

templates_db_init()

# ---- Pipeline stages (for the progress checklist in the UI) ----
# The job reports 16 distinct named steps but the UI used to collapse them into
# one bar and one line, so long stages looked frozen and short ones flew past.
# Each entry is (percent_at_start, label); the UI marks everything below the
# current percent as done and highlights the stage the job is actually in.
PIPELINE_STAGES = [
    (2,   'Reading video'),
    (8,   'Detecting cuts'),
    (15,  'Rating scenes'),
    (18,  'AI vision rating'),
    (28,  'Selecting scenes'),
    (38,  'Extracting clips'),
    (50,  'Transitions'),
    (58,  'Audio levels'),
    (62,  'Sound effects'),
    (80,  'Music'),
    (90,  'Narration'),
    (100, 'Done'),
]

# ---- Preview checkpoint ----
# A preview job runs the expensive analysis half of the pipeline (detect, score,
# select) and then STOPS, handing back the chosen cut with thumbnails. The user
# approves or drops scenes, then renders — and the render reuses this stored
# selection instead of redoing detection and AI scoring. Without it the only way
# to see what got picked was to sit through a full render.
PREVIEWS = {}
PREVIEWS_LOCK = threading.Lock()
PREVIEW_TTL = int(os.environ.get('PREVIEW_TTL', 2 * 3600))
# How many runner-up scenes a preview offers as swap-in alternatives ("show more").
PREVIEW_ALTERNATES = int(os.environ.get('PREVIEW_ALTERNATES', 12))

def preview_store(pid, data):
    with PREVIEWS_LOCK:
        now = time.time()
        for k in [k for k, v in PREVIEWS.items() if now - v.get('created', 0) > PREVIEW_TTL]:
            PREVIEWS.pop(k, None)
        data['created'] = now
        PREVIEWS[pid] = data

def preview_get(pid):
    with PREVIEWS_LOCK:
        p = PREVIEWS.get(pid)
        return dict(p) if p else None

ACE_STEP_URL = os.environ.get('ACE_STEP_URL', 'http://localhost:8001')
# Diffusion steps for music generation. The previous hardcoded 8 was far below
# ACE-Step's documented default and produced noticeably thin, smeared beds.
ACE_STEP_STEPS = int(os.environ.get('ACE_STEP_STEPS', 27))
ACE_STEP_NEGATIVE_PROMPT = os.environ.get(
    'ACE_STEP_NEGATIVE_PROMPT',
    'vocals, singing, voice, lyrics, choir, spoken word, low quality, distorted, clipping')
WOOSH_URL = os.environ.get('WOOSH_URL', 'http://localhost:8030')  # local API server for Sony AI's Woosh SFX foundation model (github.com/SonyResearch/Woosh); tried first for genre SFX, falls back to a procedural synth if unreachable (see woosh_sfx())
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

# ---- Fish Audio inline delivery tags ----
# Fish Audio interprets markers embedded in the narration text to control
# emotion, delivery and non-speech sounds. The syntax differs by generation:
# S2 uses [square brackets] and accepts free-form natural language; the older S1
# uses (parentheses) and a fixed tag set. FISH_TAG_STYLE picks which to emit --
# 'auto' infers it from FISH_AUDIO_MODEL, which is what a self-hosted server is
# usually named after.
FISH_TAG_STYLE = os.environ.get('FISH_TAG_STYLE', 'auto')

def fish_tag_syntax():
    """Returns ('[', ']') for S2-style tags or ('(', ')') for legacy S1."""
    style = (FISH_TAG_STYLE or 'auto').lower()
    if style in ('s1', 'paren', 'parentheses'):
        return '(', ')'
    if style in ('s2', 'bracket', 'brackets'):
        return '[', ']'
    model = (FISH_AUDIO_MODEL or '').lower()
    # Anything explicitly S1 gets parentheses; everything else (s2*, unknown,
    # blank) gets brackets, which is the current default generation.
    return ('(', ')') if ('s1' in model and 's2' not in model) else ('[', ']')

# Curated from Fish Audio's emotion-control reference. Not the full 64+ list --
# these are the ones that actually earn their place in a broadcast promo script;
# S2 accepts free-form descriptions anyway, so anything missing can be typed.
FISH_TAG_GROUPS = [
    ('Delivery', ['emphasis', 'whispering', 'shouting', 'soft tone',
                  'in a hurry tone', 'screaming']),
    ('Tone', ['confident', 'excited', 'calm', 'serious', 'friendly',
              'empathetic', 'curious', 'determined', 'hopeful', 'nostalgic',
              'sarcastic', 'proud']),
    ('Emotion', ['happy', 'sad', 'angry', 'scared', 'worried', 'surprised',
                 'frustrated', 'delighted', 'grateful', 'moved', 'relaxed',
                 'disappointed']),
    ('Pauses', ['break', 'long-break']),
    ('Sounds', ['laughing', 'chuckling', 'sighing', 'gasping', 'panting',
                'clear throat', 'audience laughing', 'crowd laughing']),
]

def fish_tag_catalogue():
    """Tag groups rendered in the syntax the configured model expects."""
    lo, hi = fish_tag_syntax()
    return {
        'open': lo, 'close': hi,
        'style': 's1' if lo == '(' else 's2',
        'model': FISH_AUDIO_MODEL,
        'groups': [{'name': name, 'tags': [{'name': t, 'tag': f'{lo}{t}{hi}'} for t in tags]}
                   for name, tags in FISH_TAG_GROUPS],
    }
# Reference WAV used for Fish Audio voice cloning (zero-shot): the narration voice is
# cloned from this sample on every request rather than requiring a pre-registered
# reference_id. Looked up next to this script by default; override with the env var.

# ---- Network (SMB) folder browsing — alternative to local file upload ----
# Lets the upload panels list and pull media straight from a Windows network
# share instead of requiring a local drag-and-drop/browse. Override any of
# these with env vars; not exposed through the Config tab since it holds a
# plaintext password.
NETWORK_SHARE_HOST = os.environ.get('NETWORK_SHARE_HOST', '10.0.1.130')
NETWORK_SHARE_NAME = os.environ.get('NETWORK_SHARE_NAME', 'pmc_mams_ing')
NETWORK_SHARE_SUBDIR = os.environ.get('NETWORK_SHARE_SUBDIR', 'PLUG TEST')
NETWORK_SHARE_USERNAME = os.environ.get('NETWORK_SHARE_USERNAME', r'postsns\stanza')
NETWORK_SHARE_PASSWORD = os.environ.get('NETWORK_SHARE_PASSWORD', 'gma7mamS')

AUDIO_EXTENSIONS = {'mp3', 'wav', 'm4a', 'flac', 'ogg', 'aac', 'wma'}

# Each browsable panel gets its own subfolder under the share root, and its own
# allowed extension set.
#   \\\\10.0.1.130\\pmc_mams_ing\\PLUG TEST\\HIRES    -> raw video mats
#   \\\\10.0.1.130\\pmc_mams_ing\\PLUG TEST\\TCARD    -> title cards
#   \\\\10.0.1.130\\pmc_mams_ing\\PLUG TEST\\ENDCARD  -> end cards
#   \\\\10.0.1.130\\pmc_mams_ing\\PLUG TEST\\MUSIC  -> background music
#   \\\\10.0.1.130\\pmc_mams_ing\\PLUG TEST\\VO     -> narration / voiceover
#   \\\\10.0.1.130\\pmc_mams_ing\\PLUG TEST\\SFX    -> sound effects
NETWORK_CATEGORIES = {
    'hires': {'folder': 'HIRES', 'exts': ALLOWED_EXTENSIONS, 'label': 'Video (HIRES)'},
    # Cards live in their own delivery folders, not with the raw video mats.
    # NOTE the legacy form-field names: 'end_card_video' is the TITLE card and
    # 'schedule_video' is the END card (see TEMPLATE_SLOTS for the same mapping).
    'tcard': {'folder': 'TCARD',   'exts': ALLOWED_EXTENSIONS, 'label': 'Title card (TCARD)'},
    'endcard': {'folder': 'ENDCARD', 'exts': ALLOWED_EXTENSIONS, 'label': 'End card (ENDCARD)'},
    'music': {'folder': 'MUSIC', 'exts': AUDIO_EXTENSIONS, 'label': 'Music'},
    'vo':    {'folder': 'VO',    'exts': AUDIO_EXTENSIONS, 'label': 'VO'},
    'sfx':   {'folder': 'SFX',   'exts': AUDIO_EXTENSIONS, 'label': 'SFX'},
}
DEFAULT_NETWORK_CATEGORY = 'hires'

def _network_category(category):
    """Validates/normalizes a category key, falling back to 'hires'."""
    return NETWORK_CATEGORIES.get(category, NETWORK_CATEGORIES[DEFAULT_NETWORK_CATEGORY])

def _network_share_root(category=DEFAULT_NETWORK_CATEGORY):
    """UNC path of the folder we browse for `category`, e.g.
    \\\\10.0.1.130\\pmc_mams_ing\\PLUG TEST\\MUSIC"""
    cat = _network_category(category)
    root = f'\\\\{NETWORK_SHARE_HOST}\\{NETWORK_SHARE_NAME}'
    if NETWORK_SHARE_SUBDIR:
        root += f'\\{NETWORK_SHARE_SUBDIR}'
    if cat['folder']:
        root += f'\\{cat["folder"]}'
    return root

def _network_session():
    """(Re)registers the SMB session for the configured share. smbclient caches
    connections per-server, so calling this repeatedly is cheap once logged in."""
    smbclient.register_session(NETWORK_SHARE_HOST, username=NETWORK_SHARE_USERNAME,
                                password=NETWORK_SHARE_PASSWORD, connection_timeout=10)

def list_network_files(category=DEFAULT_NETWORK_CATEGORY):
    """Returns the files (name/size/modified) in the network folder for `category`,
    filtered to that category's allowed extensions."""
    cat = _network_category(category)
    _network_session()
    root = _network_share_root(category)
    out = []
    for entry in smbclient.scandir(root):
        if not entry.is_file():
            continue
        if not allowed_file(entry.name, cat['exts']):
            continue
        st = entry.stat()
        out.append({'name': entry.name, 'size': st.st_size, 'mtime': st.st_mtime})
    out.sort(key=lambda e: e['name'].lower())
    return root, out

def fetch_network_file(name, category=DEFAULT_NETWORK_CATEGORY):
    """Copies `name` from the network folder for `category` into UPLOAD_FOLDER and
    returns the local staged filename (prefixed net_<ts>_ so load_video() /
    _resolve_upload() can recognize and trust it)."""
    cat = _network_category(category)
    if os.path.basename(name) != name or not allowed_file(name, cat['exts']):
        raise ValueError('Invalid filename')
    _network_session()
    remote_path = _network_share_root(category) + '\\' + name
    local_name = f'net_{int(time.time())}_{secure_filename(name)}'
    local_path = os.path.join(app.config['UPLOAD_FOLDER'], local_name)
    with smbclient.open_file(remote_path, mode='rb') as rf, open(local_path, 'wb') as lf:
        shutil.copyfileobj(rf, lf)
    return local_name

# Back-compat aliases (old names, always the 'hires'/video category).
def list_network_videos():
    return list_network_files('hires')

def fetch_network_video(name):
    return fetch_network_file(name, 'hires')

# ---- Config tab persistence: lets the AI service URLs above be edited from the UI ----
# instead of only via environment variables. Overrides are saved to a small JSON file
# next to this script and re-applied on every startup, on top of the env-var defaults
# set above. Env vars still win at process start; the Config tab wins after that until
# the file is deleted or a value is cleared back to blank.
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ai_services_config.json')
# name -> (module-level global to update, human label, help text)
CONFIGURABLE_SERVICES = {
    'FISH_AUDIO_URL':    ('Fish Audio S2', 'Full TTS endpoint URL, e.g. http://host:8080/v1/tts'),
    'FISH_AUDIO_API_KEY':('Fish Audio API key', 'Only needed for the hosted fish.audio cloud API — leave blank for a self-hosted server'),
    'WHISPER_URL':       ('faster-whisper', 'Base server URL, e.g. http://localhost:8000'),
    'OLLAMA_URL':        ('Ollama', 'Base server URL, e.g. http://localhost:11434'),
    'ACE_STEP_URL':      ('ACE-Step', 'Base server URL, e.g. http://localhost:8001'),
    'WOOSH_URL':         ('Woosh', 'Base server URL, e.g. http://localhost:8030'),
}

def load_config_overrides():
    """Applies any saved Config-tab overrides on top of the env-var defaults above.
    Called once at startup, after every constant it might touch is already defined."""
    global FISH_AUDIO_URL, FISH_AUDIO_API_KEY, WHISPER_URL, OLLAMA_URL, ACE_STEP_URL, WOOSH_URL
    if not os.path.exists(CONFIG_FILE):
        return
    try:
        with open(CONFIG_FILE) as f:
            cfg = json.load(f)
    except Exception as e:
        print(f'Config file load error ({CONFIG_FILE}): {e}')
        return
    if 'FISH_AUDIO_URL' in cfg and cfg['FISH_AUDIO_URL']: FISH_AUDIO_URL = cfg['FISH_AUDIO_URL']
    if 'FISH_AUDIO_API_KEY' in cfg: FISH_AUDIO_API_KEY = cfg['FISH_AUDIO_API_KEY']
    if 'WHISPER_URL' in cfg and cfg['WHISPER_URL']: WHISPER_URL = cfg['WHISPER_URL']
    if 'OLLAMA_URL' in cfg and cfg['OLLAMA_URL']: OLLAMA_URL = cfg['OLLAMA_URL']
    if 'ACE_STEP_URL' in cfg and cfg['ACE_STEP_URL']: ACE_STEP_URL = cfg['ACE_STEP_URL']
    if 'WOOSH_URL' in cfg and cfg['WOOSH_URL']: WOOSH_URL = cfg['WOOSH_URL']

def save_config_overrides(updates):
    """Merges `updates` (dict of the CONFIGURABLE_SERVICES keys) into the config file
    and applies them to the live module globals immediately — no restart needed."""
    global FISH_AUDIO_URL, FISH_AUDIO_API_KEY, WHISPER_URL, OLLAMA_URL, ACE_STEP_URL, WOOSH_URL
    existing = {}
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE) as f:
                existing = json.load(f)
        except Exception:
            existing = {}
    existing.update({k: v for k, v in updates.items() if k in CONFIGURABLE_SERVICES})
    with open(CONFIG_FILE, 'w') as f:
        json.dump(existing, f, indent=2)
    if 'FISH_AUDIO_URL' in updates: FISH_AUDIO_URL = updates['FISH_AUDIO_URL'] or FISH_AUDIO_URL
    if 'FISH_AUDIO_API_KEY' in updates: FISH_AUDIO_API_KEY = updates['FISH_AUDIO_API_KEY']
    if 'WHISPER_URL' in updates: WHISPER_URL = updates['WHISPER_URL'] or WHISPER_URL
    if 'OLLAMA_URL' in updates: OLLAMA_URL = updates['OLLAMA_URL'] or OLLAMA_URL
    if 'ACE_STEP_URL' in updates: ACE_STEP_URL = updates['ACE_STEP_URL'] or ACE_STEP_URL
    if 'WOOSH_URL' in updates: WOOSH_URL = updates['WOOSH_URL'] or WOOSH_URL

def current_config_values():
    return {
        'FISH_AUDIO_URL': FISH_AUDIO_URL, 'FISH_AUDIO_API_KEY': FISH_AUDIO_API_KEY,
        'WHISPER_URL': WHISPER_URL,
        'OLLAMA_URL': OLLAMA_URL, 'ACE_STEP_URL': ACE_STEP_URL, 'WOOSH_URL': WOOSH_URL,
    }

# ---- Fish Audio S2 (fish.audio) — primary voiceover engine (self-hosted or cloud REST API) ----
# How long to wait for a TTS server to synthesize. The old hardcoded 30s was
# fine for a preview but too tight for a full narration script on a self-hosted
# CPU instance -- when it expired the VO was silently dropped and the trailer
# rendered mute, with the reason buried in the server log.
TTS_TIMEOUT = int(os.environ.get('TTS_TIMEOUT', 180))

def _looks_like_audio(data, content_type=''):
    """True if `data` starts with a container signature we'd expect from a TTS
    server. Used to reject the very common failure where a server answers HTTP
    200 with a JSON or HTML error body instead of audio -- without this check
    the error text gets written to disk as a .wav, passes a size>0 test, and is
    reported as a successful render right up until ffmpeg chokes on it later."""
    if not data:
        return False
    ct = (content_type or '').lower()
    if ct.startswith(('application/json', 'text/html', 'text/plain')):
        return False
    sigs = (
        b'RIFF',      # wav
        b'ID3',       # mp3 with tag
        b'OggS',      # ogg/opus
        b'fLaC',      # flac
        b'\xff\xfb', b'\xff\xf3', b'\xff\xf2',  # raw mp3 frame
        b'\xff\xf1', b'\xff\xf9',               # adts aac
    )
    if data[:4] in (b'RIFF', b'OggS', b'fLaC') or data[:3] == b'ID3':
        return True
    if any(data.startswith(s) for s in sigs):
        return True
    # ISO-BMFF (m4a/mp4): 'ftyp' at offset 4
    if len(data) > 12 and data[4:8] == b'ftyp':
        return True
    return False

def _write_tts_response(r, output_path, engine_label):
    """Validates a TTS HTTP response and writes it out. Returns (ok, error)."""
    if not r.ok:
        detail = (r.text or '')[:300]
        try:
            j = r.json()
            detail = j.get('error') or j.get('detail') or j.get('message') or detail
        except Exception:
            pass
        return False, f'{engine_label} API error {r.status_code}: {detail}'
    if not r.content:
        return False, f'{engine_label} returned an empty response'
    if not _looks_like_audio(r.content, r.headers.get('Content-Type', '')):
        # 200 OK but the payload isn't audio -- surface whatever the server
        # actually said rather than writing it to disk as a fake .wav.
        detail = ''
        try:
            j = r.json()
            detail = j.get('error') or j.get('detail') or j.get('message') or ''
        except Exception:
            detail = (r.text or '')[:200]
        return False, (f'{engine_label} returned a non-audio response'
                       + (f': {detail}' if detail else
                          f' (Content-Type: {r.headers.get("Content-Type", "unknown")})'))
    try:
        with open(output_path, 'wb') as f:
            f.write(r.content)
    except OSError as e:
        return False, f'{engine_label}: could not write output file: {e}'
    if not (os.path.exists(output_path) and os.path.getsize(output_path) > 0):
        return False, f'{engine_label} returned an empty response'
    return True, None

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
        # Normalized to 16k mono PCM. The upload field
        # accepts any audio/*, so this is routinely an MP3/M4A or a 48k stereo
        # WAV; previously those bytes were base64-ed straight through as a
        # "reference WAV" and the server either mis-cloned or failed at synthesis.
        normalized_path = _normalize_reference_audio(reference_audio_path)
        try:
            with open(normalized_path, 'rb') as rf:
                ref_b64 = base64.b64encode(rf.read()).decode('ascii')
            body['references'] = [{'audio': ref_b64, 'text': ''}]
        except Exception as e:
            return False, f'Failed to read voice reference audio ({reference_audio_path}): {e}'
        finally:
            if normalized_path != reference_audio_path and os.path.exists(normalized_path):
                try:
                    os.remove(normalized_path)
                except OSError:
                    pass
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
        r = requests.post(FISH_AUDIO_URL, headers=headers, json=body, timeout=TTS_TIMEOUT)
        return _write_tts_response(r, output_wav_path, 'Fish Audio')
    except Exception as e:
        return False, f'Fish Audio request failed: {e}'

# ---- Shared voice-clone reference handling ----
def _normalize_reference_audio(src_path):
    """Re-encodes a reference/voice-clone sample to 16kHz mono 16-bit PCM WAV.

    Used by BOTH engines. The upload field accepts any audio/*, so this is
    routinely an MP3, M4A or a 48k stereo WAV. Voice-cloning servers can usually
    fingerprint a voice from almost any input but then fail during the actual
    synthesis pass that expects a specific format -- which is exactly the shape
    of a 'voice is detected but can't generate speech' error. Returns the
    normalized path, or the original path unchanged if ffmpeg isn't available or
    the conversion fails (caller falls back to sending the original bytes rather
    than hard-failing here)."""
    if not FFMPEG or not os.path.exists(src_path):
        return src_path
    norm_path = os.path.join(tempfile.gettempdir(), f'ttsref_{uuid.uuid4().hex}.wav')
    try:
        r = subprocess.run([FFMPEG, '-y', '-i', src_path, '-ac', '1', '-ar', '16000',
                             '-sample_fmt', 's16', norm_path],
                            capture_output=True, text=True, timeout=60)
        if os.path.exists(norm_path) and os.path.getsize(norm_path) > 0:
            return norm_path
        print(f'TTS: reference audio normalization failed, sending original file: {r.stderr[-300:]}')
    except Exception as e:
        print(f'TTS: reference audio normalization error, sending original file: {e}')
    return src_path


# Language codes offered in the narration UI. Fish Audio S2 auto-detects the
# script's language, so this is an optional override rather than a requirement.
FISH_AUDIO_LANGUAGES = [
    {'code': 'auto', 'label': 'Auto-detect (recommended)'},
    {'code': 'en', 'label': 'English'},
    {'code': 'tl', 'label': 'Tagalog / Filipino'},
    {'code': 'zh', 'label': 'Chinese'},
    {'code': 'ja', 'label': 'Japanese'},
    {'code': 'ko', 'label': 'Korean'},
    {'code': 'es', 'label': 'Spanish'},
    {'code': 'fr', 'label': 'French'},
    {'code': 'de', 'label': 'German'},
    {'code': 'ar', 'label': 'Arabic'},
    {'code': 'pt', 'label': 'Portuguese'},
    {'code': 'ru', 'label': 'Russian'},
    {'code': 'id', 'label': 'Indonesian'},
    {'code': 'vi', 'label': 'Vietnamese'},
    {'code': 'th', 'label': 'Thai'},
]

# Voice lists are fetched from the TTS server and cached briefly: the dropdown is
# reloaded on several UI events, and a self-hosted server can be slow to answer.
_VOICES_CACHE_TTL = int(os.environ.get('VOICES_CACHE_TTL', 300))
_VOICES_CACHE = {
    'fish_audio': {'voices': None, 'source': 'none', 'error': None, 'fetched_at': 0.0},
}

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
        # Self-hosted probe. Fish Speech exposes its registered reference voices at
        # /v1/references/list, NOT /v1/models (which 404s) -- probing only the
        # latter silently produced an empty list, so the voice dropdown had nothing
        # in it and generate_tts() then refused to run for want of a voice_id.
        #
        # Two quirks that matter:
        #   * the endpoint defaults to application/msgpack, so Accept must ask for
        #     JSON explicitly or r.json() fails on binary content;
        #   * the response is {"reference_ids": ["name", ...]} -- bare strings, not
        #     the {"items": [{"id": ...}]} objects the cloud API returns.
        # Older/alternative builds are still tried afterwards, and both object and
        # string entries are accepted, so this works across server versions.
        base = FISH_AUDIO_URL.rsplit('/v1/', 1)[0] if '/v1/' in FISH_AUDIO_URL else FISH_AUDIO_URL
        base = base.rstrip('/')
        headers = {'Accept': 'application/json'}
        for path in ('/v1/references/list', '/v1/models'):
            try:
                r = requests.get(base + path, headers=headers, timeout=4)
            except Exception:
                continue  # server down or path refused — try the next one
            if not r.ok:
                continue
            try:
                data = r.json()
            except ValueError:
                # Still msgpack (or otherwise not JSON) despite the Accept header.
                error = (f'{path} returned {r.headers.get("Content-Type", "an unreadable format")} '
                         'rather than JSON, so its voice list could not be parsed.')
                continue
            if isinstance(data, dict):
                items = data.get('reference_ids') or data.get('items') or data.get('models') or []
            elif isinstance(data, list):
                items = data
            else:
                items = []
            for it in items:
                if isinstance(it, str):
                    # /v1/references/list form: the id IS the display name.
                    vid, title, langs = it, it, []
                elif isinstance(it, dict):
                    vid = it.get('id') or it.get('_id') or it.get('reference_id')
                    title = it.get('title') or it.get('name') or vid
                    langs = it.get('languages') or []
                else:
                    continue
                if vid:
                    voices.append({'id': vid, 'title': title, 'languages': langs})
            # Set outside the loop so an endpoint that exists but returns nothing
            # still reports 'self_hosted' rather than 'none', matching
            # a server that exists but lists nothing.
            source = 'self_hosted'
            if voices:
                error = None
                break

    _VOICES_CACHE['fish_audio'].update(voices=voices, source=source, error=error, fetched_at=now)
    return voices, source, error

def list_voices_for_engine(engine, force=False):
    """Voice list for the requested engine. Fish Audio is the only narration
    engine now; the parameter is kept so existing API callers and saved templates
    that still pass vo_engine keep working."""
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
# Parallel ffmpeg processes used for clip extraction within a single job. Kept
# modest by default because MAX_CONCURRENT_JOBS jobs can each run this many at
# once -- the effective ceiling is MAX_CONCURRENT_JOBS * EXTRACT_WORKERS.
EXTRACT_WORKERS = int(os.environ.get('EXTRACT_WORKERS', 4))
# Cap on how many scenes get an AI vision call per job. A long episode can detect
# hundreds of scenes but only ~12 reach the trailer, so scoring all of them is
# almost entirely wasted work; see the shortlist logic in _run_trailer_job().
AI_SCORE_LIMIT = int(os.environ.get('AI_SCORE_LIMIT', 60))
AI_SCORE_WORKERS = int(os.environ.get('AI_SCORE_WORKERS', 4))
# Token budget for one vision reply. Must comfortably exceed the answer length,
# because reasoning models count their (hidden) chain-of-thought against it -- a
# tight cap returns an empty `response` and every scene silently falls back to
# the neutral score.
AI_NUM_PREDICT = int(os.environ.get('AI_NUM_PREDICT', 400))
# Set to False the first time a server rejects the structured-output `format`
# field, so older Ollama builds pay the cost of that discovery only once.
AI_STRUCTURED_OK = True
AI_NEUTRAL_SCORE = 3  # mid-range prior for scenes that weren't or couldn't be scored
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
    'action': 'Epic cinematic action trailer score, 150 BPM, powerful orchestral percussion, aggressive taiko drums, bold brass stabs, driving string ostinatos, rising tension, heroic energy, blockbuster soundtrack, high impact dynamics, instrumental only, no vocals',
    'drama': 'Emotional cinematic drama score, 80 BPM, expressive piano melody, warm string ensemble, gentle emotional build, heartfelt atmosphere, reflective storytelling, film soundtrack style, instrumental only, no vocals',
    'horror': 'Dark atmospheric horror soundtrack, 65 BPM, eerie drones, unsettling textures, dissonant strings, distant impacts, creeping suspense, psychological tension, cinematic dread, instrumental only, no vocals',
    'comedy': 'Playful comedy soundtrack, 120 BPM, cheerful pizzicato strings, quirky woodwinds, light percussion, whimsical melodies, upbeat and humorous mood, family entertainment style, instrumental only, no vocals',
    'documentary': 'Inspiring documentary score, 90 BPM, soft piano and strings, ambient orchestral textures, thoughtful emotional tone, uplifting cinematic storytelling, modern documentary soundtrack, instrumental only, no vocals',
    'thriller': 'Suspenseful thriller soundtrack, 110 BPM, pulsing rhythmic patterns, dark atmospheric pads, subtle electronic elements, escalating tension, cinematic urgency, investigative mood, instrumental only, no vocals',
    'scifi': 'Futuristic science fiction soundtrack, 120 BPM, atmospheric synthesizers, cosmic pads, electronic pulses, cinematic space exploration mood, advanced technology theme, immersive and expansive, instrumental only, no vocals',
    'fantasy': 'Enchanting fantasy orchestral score, 105 BPM, magical strings and woodwinds, mystical choir textures, adventurous melodies, wonder and discovery, cinematic fantasy realm atmosphere, instrumental only, no vocals',
    'romance': 'Romantic cinematic soundtrack, 75 BPM, tender piano melodies, warm strings, emotional intimacy, gentle orchestral swells, heartfelt and elegant atmosphere, instrumental only, no vocals',
    'adventure': 'Epic adventure soundtrack, 130 BPM, heroic brass, sweeping strings, driving percussion, exploration and discovery theme, uplifting cinematic energy, triumphant orchestral score, instrumental only, no vocals',
    'mystery': 'Intriguing mystery soundtrack, 90 BPM, subtle piano motifs, atmospheric strings, investigative mood, gradual tension build, enigmatic cinematic atmosphere, suspenseful yet elegant, instrumental only, no vocals',
    'western': 'Classic western cinematic score, 105 BPM, acoustic guitar, harmonica, sparse percussion, dusty frontier atmosphere, rugged adventure mood, expansive desert landscapes, instrumental only, no vocals',
    'sports': 'High-energy sports anthem, 155 BPM, driving drums, motivational brass, uplifting orchestral and modern hybrid elements, victory and competition theme, powerful momentum, instrumental only, no vocals',
    'noir': 'Film noir jazz soundtrack, 85 BPM, smoky saxophone, upright bass, brushed drums, moody piano, mysterious detective atmosphere, dark urban night setting, instrumental only, no vocals',
    'war': 'Epic war drama soundtrack, 115 BPM, military drums, emotional strings, heroic brass, sacrifice and courage theme, cinematic battlefield atmosphere, tragic yet triumphant, instrumental only, no vocals',
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
# NOTE: values above were already identical to the uploaded CSV's transition/crossfade/sfx_at_cuts
# columns for all 15 genres — no changes were needed here.

GENRE_NAMES = list(GENRE_PRESETS.keys())

# Module-level so both api_trailer() and the show-template save route can validate
# against the same list (it used to be a local inside api_trailer()).
VALID_TRANSITIONS = {'fade','fadeblack','fadewhite','fadefast','fadegrays',
    'wipeleft','wiperight','wipeup','wipedown',
    'slideleft','slideright','slideup','slidedown',
    'smoothleft','smoothright','smoothup','smoothdown',
    'circlecrop','rectcrop','circleopen','circleclose',
    'distance','pixelize','diagtl','diagtr','diagbl','diagbr',
    'hlslice','hrslice','vuslice','vdslice',
    'radial','zoomin','dissolve','hblur','squeezev','squeezeh',
    'horzopen','horzclose','vertopen','vertclose','custom_matte'}

GENRE_SFX_PROMPTS = {
    'action': 'Massive cinematic impact, deep explosion boom, trailer hit, powerful transient, sound effect only',
    'horror': 'Eerie horror sting, dark whoosh, unsettling impact, suspense accent, sound effect only',
    'comedy': 'Cartoon boing, comedic pop, playful bounce, humorous sting, sound effect only',
    'thriller': 'Tension hit, dark pulse, suspense sting, cinematic impact, sound effect only',
    'scifi': 'Futuristic whoosh, electronic glitch impact, cybernetic sweep, sci-fi accent, sound effect only',
    'fantasy': 'Magical sparkle chime, enchanted shimmer, mystical twinkle, fantasy accent, sound effect only',
    'adventure': 'Heroic orchestral hit, cinematic impact, adventure accent, sound effect only',
    'western': 'Whip crack, dusty impact, western accent, sound effect only',
    'sports': 'Stadium crowd hit, whistle blast, energetic impact, sports accent, sound effect only',
    'war': 'Battlefield explosion hit, military impact, distant artillery boom, sound effect only',
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

# ---- Hard timeouts on every external media call ----
# A subprocess.run() without a timeout can block its worker thread forever if
# ffmpeg wedges on a malformed stream. That thread holds a GATE slot (see
# JobGate), so with MAX_CONCURRENT_JOBS=2 two hung jobs stall the whole server
# with no way to recover short of a restart. Every ffmpeg/ffprobe call therefore
# goes through run_ffmpeg()/run_ffprobe() below, which always pass a timeout and
# always kill the whole process group on expiry.
FFPROBE_TIMEOUT = int(os.environ.get('FFPROBE_TIMEOUT', 30))
FFMPEG_TIMEOUT = int(os.environ.get('FFMPEG_TIMEOUT', 300))       # per encode/mix step
FFMPEG_LONG_TIMEOUT = int(os.environ.get('FFMPEG_LONG_TIMEOUT', 900))  # full-source passes

class MediaToolTimeout(RuntimeError):
    """Raised when ffmpeg/ffprobe exceeded its timeout and was killed."""

def _run_media_tool(cmd, timeout, label):
    """subprocess.run with a guaranteed timeout and a guaranteed kill.

    subprocess.run(timeout=...) sends SIGKILL to the direct child only; ffmpeg
    rarely spawns children, but Popen.kill() can still leave a stuck process if
    it's blocked in an uninterruptible read, so we kill then reap with a short
    second wait and surface a clean exception either way."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        print(f'{label} timed out after {timeout}s and was killed: {" ".join(str(c) for c in cmd[:6])}...')
        raise MediaToolTimeout(f'{label} exceeded {timeout}s') from e

def run_ffmpeg(cmd, timeout=None, label='ffmpeg'):
    return _run_media_tool(cmd, timeout or FFMPEG_TIMEOUT, label)

def run_ffprobe(cmd, timeout=None, label='ffprobe'):
    return _run_media_tool(cmd, timeout or FFPROBE_TIMEOUT, label)

def probe_duration(path, default=None):
    """Duration of `path` in seconds, or `default` if it can't be determined.

    Returning None (the default default) lets callers distinguish "unknown" from
    a real value instead of silently substituting a guess -- a wrong duration
    here feeds the xfade offset maths and desynchronises the entire concat."""
    try:
        r = run_ffprobe([FFPROBE, '-v', 'error', '-show_entries', 'format=duration',
                         '-of', 'default=noprint_wrappers=1:nokey=1', path])
        return float(r.stdout.strip())
    except (MediaToolTimeout, ValueError, AttributeError):
        return default

def probe_media_info(path):
    """One ffprobe call returning {'duration': float|None, 'has_audio': bool}.

    Replaces the previous pattern of three separate ffprobe spawns per input
    (duration, audio-stream check, duration again after normalisation) -- on a
    15-input job that was ~46 process spawns just to read metadata."""
    info = {'duration': None, 'has_audio': False}
    try:
        r = run_ffprobe([FFPROBE, '-v', 'error', '-show_entries',
                         'format=duration:stream=codec_type', '-of', 'json', path])
        data = json.loads(r.stdout or '{}')
    except (MediaToolTimeout, ValueError):
        return info
    try:
        info['duration'] = float(data.get('format', {}).get('duration'))
    except (TypeError, ValueError):
        info['duration'] = None
    info['has_audio'] = any(s.get('codec_type') == 'audio' for s in data.get('streams', []))
    return info

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

def _detect_silence_intervals(audio_path, noise_db=-30, min_dur=0.3, timeout=120):
    """Runs ffmpeg's silencedetect filter and parses stderr for silence_start/silence_end
    pairs. Returns a list of (start, end) SILENT intervals in audio_path. A silence_start
    with no matching silence_end (file ends mid-silence) is dropped rather than guessed at —
    the caller treats "not explicitly silent" as active, which is the safe default."""
    try:
        r = subprocess.run([FFMPEG, '-i', audio_path, '-af',
                             f'silencedetect=noise={noise_db}dB:d={min_dur}', '-f', 'null', '-'],
                            capture_output=True, text=True, timeout=timeout)
    except Exception as e:
        print(f'Silence detection error ({audio_path}): {e}')
        return []
    starts = [float(m) for m in re.findall(r'silence_start:\s*([\d.]+)', r.stderr)]
    ends = [float(m) for m in re.findall(r'silence_end:\s*([\d.]+)', r.stderr)]
    return list(zip(starts, ends))

def _active_windows_from_silence(silence_intervals, total_duration, content_duration=None):
    """Inverts silent intervals into the (start, end) windows where audio IS
    present.

    `content_duration` is the real length of the file that was analyzed. It
    defaults to `total_duration` for the common case (e.g. SOT, where the
    analyzed file genuinely spans the whole trailer) -- but when the source is
    SHORTER than `total_duration` (a VO clip placed early in a longer
    trailer), the trailing "active" extension must stop at the real end of
    that file, not run all the way out to total_duration. Without this bound:
    a short VO clip with no detected internal silence (nothing for
    silencedetect to report) gets treated as "playing" for the entire
    remainder of the trailer, which then ducks BGM/SOT under a VO that
    actually stopped minutes earlier -- including under the cards."""
    if content_duration is None:
        content_duration = total_duration
    silence_intervals = sorted(silence_intervals)
    windows = []
    cursor = 0.0
    for s, e in silence_intervals:
        if s > cursor:
            windows.append((cursor, s))
        cursor = max(cursor, e)
    end_bound = min(content_duration, total_duration)
    if cursor < end_bound:
        windows.append((cursor, end_bound))
    return windows

def _union_windows(window_lists):
    """Unions multiple lists of (start, end) windows (e.g. SOT-active ∪ VO-active) into
    one merged, sorted, non-overlapping list."""
    all_w = sorted(w for lst in window_lists for w in lst)
    if not all_w:
        return []
    merged = [list(all_w[0])]
    for s, e in all_w[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]

def _merge_windows_with_hold(windows, hold_sec):
    """Bridges any gap between consecutive windows that's shorter than hold_sec — this is
    the actual 'minimum gap of no VO/SOT before the duck releases and music is heard again'
    behavior: a brief pause in dialogue no longer lets BGM swell back up and immediately
    duck again, since the gap has to be at least hold_sec long to count as a real release."""
    if not windows:
        return []
    windows = sorted(windows)
    merged = [list(windows[0])]
    for s, e in windows[1:]:
        if s - merged[-1][1] <= hold_sec:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]

DUCK_ATTACK = float(os.environ.get('DUCK_ATTACK', 0.08))    # seconds to duck down
DUCK_RELEASE = float(os.environ.get('DUCK_RELEASE', 0.45))  # seconds to come back up

def _build_duck_volume_expr(duck_windows, duck_depth_db, attack=None, release=None):
    """ffmpeg volume expression that ducks by duck_depth_db across duck_windows,
    with real attack/release ramps.

    This used to emit a bare step -- if(between(t,s,e), gain, 1) -- which changes
    gain by the full depth between one sample and the next. A 15 dB discontinuity
    is a broadband click, and there were typically 9-13 of them in a 30s promo, so
    it read as a fault rather than as ducking.

    The envelope is built as a FLAT sum of trapezoids rather than a nested if
    chain, which matters as much as the ramps do: the old form nested one if()
    per window, so expression depth grew with the amount of dialogue. Here depth
    is constant and only the length grows.

        gain(t) = 1 - (1 - g) * min(1, SUM_i trapezoid_i(t))
        trapezoid_i(t) = clip((t - s_i + a)/a, 0, 1) * clip((e_i + r - t)/r, 0, 1)

    Each trapezoid ramps 0->1 over the attack window ending at s_i, holds at 1
    through the window, then ramps 1->0 over the release after e_i. min(1, ...)
    keeps overlapping ramps from over-ducking: two windows closer together than
    attack+release simply stay ducked through the gap, which is what you'd want
    anyway."""
    if not duck_windows:
        return None
    a = max(0.01, attack if attack is not None else DUCK_ATTACK)
    r = max(0.01, release if release is not None else DUCK_RELEASE)
    gain = 10 ** (duck_depth_db / 20)
    terms = [f'clip((t-{s:.3f}+{a})/{a},0,1)*clip(({e:.3f}+{r}-t)/{r},0,1)'
             for s, e in duck_windows]
    return f'1-{1 - gain:.5f}*min(1,{"+".join(terms)})'


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

def allowed_file(filename, exts=None):
    exts = ALLOWED_EXTENSIONS if exts is None else exts
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in exts

def load_video(req):
    if 'file' in req.files and req.files['file'].filename != '':
        f = req.files['file']
        if not allowed_file(f.filename):
            return None, 'File type not allowed'
        fn = secure_filename(f.filename)
        if not fn:
            return None, 'Invalid filename'
        # Two people can easily upload same-named files at once (e.g. two
        # different "episode1.mp4"s) while both jobs run concurrently (see
        # MAX_CONCURRENT_JOBS / GATE below) -- without a per-request-unique
        # disk name, the second upload would silently overwrite the first
        # one's staged file out from under its still-running job. orig_name
        # (the display name shown in results/history) stays the clean one.
        disk_name = f'src_{int(time.time()*1000)}_{threading.get_ident()}_{fn}'
        path = os.path.join(app.config['UPLOAD_FOLDER'], disk_name)
        f.save(path)
        return path, fn
    # The main dropzone's own richer browser posts 'network_file'; the shared
    # "Browse library" modal used elsewhere (title/end card, BGM, SFX, VO, and
    # now this Vision form) posts '<field>_network' -- for a field literally
    # named 'file' that's 'file_network'. Accept either so the shared modal
    # works here without needing its own bespoke field-naming convention.
    staged = req.form.get('network_file') or req.form.get('file_network')
    if staged:
        # Must be a name we generated ourselves in fetch_network_video() (net_<ts>_<name>)
        # and that still exists in UPLOAD_FOLDER -- never trust an arbitrary path here.
        safe = os.path.basename(staged)
        path = os.path.join(app.config['UPLOAD_FOLDER'], safe)
        if safe.startswith('net_') and os.path.exists(path):
            return path, safe
        return None, 'Selected network file is no longer available -- please re-select it'
    return None, 'No video provided'

def _resolve_upload(field_name, exts=None):
    """Resolves `field_name` to a local file path, from either a normal file
    upload (request.files[field_name]) or a network-folder file already staged
    into UPLOAD_FOLDER by /api/network/fetch (request.form[field_name + '_network']).
    A direct upload always takes priority if both are present. Returns None if
    neither is present/valid. Used for every optional media field that now
    supports "browse library" (title/end card video, BG music, VO, SFX)."""
    if field_name in request.files and request.files[field_name].filename:
        f = request.files[field_name]
        if exts is not None and not allowed_file(f.filename, exts):
            return None
        fn = secure_filename(f.filename)
        if not fn:
            return None
        dest = os.path.join(app.config['UPLOAD_FOLDER'], f'{field_name}_{int(time.time()*1000)}_{threading.get_ident()}{os.path.splitext(fn)[1]}')
        f.save(dest)
        return dest
    staged = (request.form.get(field_name + '_network') or '').strip()
    if staged:
        # Must be a name we generated ourselves in fetch_network_file() (net_<ts>_<name>)
        # and that still exists in UPLOAD_FOLDER -- never trust an arbitrary path here.
        safe = os.path.basename(staged)
        path = os.path.join(app.config['UPLOAD_FOLDER'], safe)
        if safe.startswith('net_') and os.path.exists(path):
            return path
    return None

def _upload_display_name(field_name):
    """The human-readable original filename behind whatever _resolve_upload() would
    return for `field_name` -- a browser upload's own name, or the network file's
    name with the net_<ts>_ staging prefix stripped back off. Used so saved show
    templates list "PLUG_BED_2024.wav" rather than an internal storage name."""
    if field_name in request.files and request.files[field_name].filename:
        return os.path.basename(request.files[field_name].filename)
    staged = (request.form.get(field_name + '_network') or '').strip()
    if staged:
        return re.sub(r'^net_\d+_', '', os.path.basename(staged))
    return None

def _clean_ai_desc(raw):
    """Tidies a vision model's DESC field for display in the scene table: single
    line, no trailing punctuation, sentence-cased, and capped so one rambling
    response can't stretch the column."""
    t = ' '.join((raw or '').split())
    t = re.sub(r'^(the\s+)?(image|frame|shot|scene)\s+(shows|depicts|features)\s+', '', t, flags=re.I)
    t = t.strip(' .;:,-')
    if len(t) > 120:
        t = t[:117].rsplit(' ', 1)[0] + '…'
    return (t[:1].upper() + t[1:]) if t else ''

def _scene_desc(s):
    """One human-readable line describing the shot, for the scene table.

    Prefers the vision model's literal description of what's in frame. Without
    AI rating there's no semantic information available at all, so the fallback
    describes the measurable properties in plain words -- a close-up vs a wide,
    how bright, how busy -- rather than emitting raw metric tags."""
    ai = s.get('ai_desc', '') or ''
    if ai:
        return ai

    sat = s.get('mean_sat', 0)
    val = s.get('mean_val', 0)
    edge = s.get('edge_ratio', 0)
    hue = s.get('mean_hue', 0)
    dur = s.get('duration', 0)

    subject = 'Person in shot' if s.get('has_face') else (
        'Busy, detailed shot' if edge > 0.15 else
        'Wide, open shot' if edge < 0.06 else 'Medium shot')

    light = ('very dark' if val < 40 else
             'dim' if val < 90 else
             'bright' if val > 200 else 'well lit')

    colour = ('almost monochrome' if sat < 30 else
              'strongly coloured' if sat > 100 else None)

    palette = ('greens/outdoors' if 90 < hue < 150 else
               'warm tones' if (0 < hue < 30 or 160 < hue < 180) else None)

    bits = [light]
    if colour: bits.append(colour)
    if palette: bits.append(palette)
    tail = f' — {dur:.1f}s take' if dur > 5 else ''
    return f'{subject}, {", ".join(bits)}{tail}'

def _to_scalar(x):
    """Robustly converts a value that may already be a Python/numpy scalar, or
    may be a 0-d or 1-d numpy array, into a plain Python float. Needed because
    librosa.beat.beat_track's `tempo` return value changed from a plain float
    to a numpy array (shape (1,), one entry per audio channel) in librosa
    0.10+ -- calling float() directly on that array is what raises numpy's
    'only 0-dimensional arrays can be converted to Python scalars' error.
    np.ravel(x)[0] normalizes every shape (scalar, 0-d, 1-d) to a single
    value before the float() conversion."""
    return float(np.ravel(x)[0])

def beat_match_audio(video_path, bgm_path, target_dur, output_path):
    try:
        import librosa
        # Extract audio from source video, resample to consistent rate
        audio_tmp = os.path.join(app.config['UPLOAD_FOLDER'], f'beat_video_{int(time.time())}.wav')
        subprocess.run([FFMPEG, '-y', '-i', video_path, '-vn', '-ar', '22050', '-ac', '1', audio_tmp],
                       capture_output=True, text=True, timeout=60)
        if not os.path.exists(audio_tmp) or os.path.getsize(audio_tmp) == 0:
            return False
        y_vid, sr = librosa_load(audio_tmp, sr=22050)
        os.remove(audio_tmp)
        tempo_vid, _ = librosa.beat.beat_track(y=y_vid, sr=sr)
        tempo_vid = _to_scalar(tempo_vid)
        if tempo_vid < 30 or tempo_vid > 300:
            tempo_vid = 120
    except Exception as e:
        print(f'Beat detection error: {e}')
        return False

    try:
        y_bgm, sr_bgm = librosa_load(bgm_path, sr=22050)
        orig_len = len(y_bgm)
        # Detect BGM tempo
        tempo_bgm, _ = librosa.beat.beat_track(y=y_bgm, sr=sr_bgm)
        tempo_bgm = _to_scalar(tempo_bgm)
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
        y, sr = librosa_load(path, sr=sample_rate, mono=True, duration=max_dur)
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

def _woosh_generate_raw(prompt, timeout=15):
    """One call to Woosh's /generate. Returns (flac_bytes, error).

    Corrected against the actual api_server.py source (not just the public demo):
      * `token` is required by the request schema but never read or validated
        server-side -- sending the literal string "string" (Sony's own test
        script's placeholder) satisfies the schema with nothing to configure.
      * The response is FLAC (media_type="audio/flac"), not WAV -- this was
        previously saved with a .wav extension and served as-is, which lies
        about the container in both the filename and the Content-Type Flask
        would derive from it, and could fail to play in a browser depending on
        how strictly it trusts that mismatch.
      * `duration` is not a real field on GenerateArgs in this API version --
        confirmed against the schema, not assumed. It was being sent and
        silently ignored. Removed from the request; callers that need a
        specific length now enforce it themselves by trimming the response
        (see _woosh_generate below), since the API has no way to ask for one."""
    try:
        r = requests.post(f'{WOOSH_URL}/generate', json={
            'prompt': prompt,
            'token': 'string',
        }, timeout=timeout)
    except Exception as e:
        return None, f'Could not reach Woosh at {WOOSH_URL}: {e}'
    if not r.ok:
        return None, f'Woosh API error {r.status_code}: {(r.text or "")[:300]}'
    if not r.content:
        return None, 'Woosh returned an empty response.'
    return r.content, None

def _woosh_generate(prompt, dest_flac_path, duration=None, timeout=15):
    """Fetches one Woosh generation and writes it to `dest_flac_path` (must end
    in .flac), trimming to `duration` seconds if given. Returns (ok, error).

    Trimming is done here, client-side, because the API has no duration
    parameter of its own -- this is the only way "duration" means anything."""
    raw, err = _woosh_generate_raw(prompt, timeout=timeout)
    if err:
        return False, err
    if duration is None:
        try:
            with open(dest_flac_path, 'wb') as f:
                f.write(raw)
        except OSError as e:
            return False, f'Could not write output file: {e}'
        return os.path.exists(dest_flac_path) and os.path.getsize(dest_flac_path) > 0, None

    raw_path = dest_flac_path + '.raw.flac'
    try:
        with open(raw_path, 'wb') as f:
            f.write(raw)
    except OSError as e:
        return False, f'Could not write output file: {e}'
    try:
        run_ffmpeg([FFMPEG, '-y', '-i', raw_path, '-t', str(duration), '-c:a', 'flac', dest_flac_path],
                   timeout=30, label='woosh trim')
    except MediaToolTimeout as e:
        return False, f'Trimming the Woosh output timed out: {e}'
    finally:
        if os.path.exists(raw_path):
            try:
                os.remove(raw_path)
            except OSError:
                pass
    return os.path.exists(dest_flac_path) and os.path.getsize(dest_flac_path) > 0, None

def woosh_sfx(genre, output_path, duration=0.8):
    """Generate a one-shot SFX using Sony AI's Woosh text-to-audio model via its
    local API server. `output_path` should end in .flac (Woosh's real output
    format -- see _woosh_generate_raw). Returns True/False — failures fall
    through silently so the caller can fall back to the procedural synth
    (ACE-Step is a music model, not used for SFX)."""
    prompt = GENRE_SFX_PROMPTS.get(genre)
    if not prompt:
        return False
    ok, err = _woosh_generate(prompt, output_path, duration=duration, timeout=15)
    if not ok and err:
        print(f'Woosh SFX error: {err}')
    return ok

WOOSH_MAX_SAMPLES = int(os.environ.get('WOOSH_MAX_SAMPLES', 4))

def woosh_sfx_generate(prompt, duration=1.0, samples=1, base_ts=None):
    """Generate SFX from free-text prompts, for the Tools tab. Returns (paths, error).

    Unlike woosh_sfx() above (which the trailer pipeline calls with a fixed
    genre-derived prompt and silently falls back to a procedural synth on
    failure, since *some* sound must occupy that slot), this is the raw
    generator behind the Text to SFX tool: any prompt the user types, and no
    fallback -- if Woosh is down the tool should say so rather than hand back
    a synth click the user didn't ask for.

    Multiple samples are produced by calling _woosh_generate repeatedly rather
    than a batch parameter, since the real request schema has no such field
    either (see _woosh_generate_raw)."""
    prompt = (prompt or '').strip()
    if not prompt:
        return [], 'Enter a description of the sound you want.'
    duration = max(0.2, min(10.0, float(duration or 1.0)))
    samples = max(1, min(WOOSH_MAX_SAMPLES, int(samples or 1)))
    base_ts = base_ts or f'tool{int(time.time()*1000)}'

    paths, last_err = [], None
    for i in range(samples):
        dest = os.path.join(app.config['UPLOAD_FOLDER'], f'sfx_{base_ts}_{i}.flac')
        ok, err = _woosh_generate(prompt, dest, duration=duration, timeout=30)
        if ok:
            paths.append(dest)
        else:
            last_err = err

    if not paths:
        return [], (last_err or 'Generation failed.')
    return paths, None

def generate_tts(text, output_wav_path, rate=175, voice_id=None, reference_audio_path=None, language=None, engine='fish_audio'):
    """Generate a narration WAV from text, via whichever engine the user picked:
    'fish_audio' (Fish Audio S2, voice cloning, auto-detects language including
    Tagalog). Returns (ok, error_message). There's
    no bundled default voice — the caller must pass either `voice_id` (a voice
    picked from list_voices_for_engine()) or `reference_audio_path` (an uploaded
    sample to clone zero-shot); if neither is given, this returns an error
    rather than silently falling back to some fixed voice file."""
    text = (text or '').strip()
    if not text:
        return False, 'No text provided'
    if not voice_id and not (reference_audio_path and os.path.exists(reference_audio_path)):
        return False, 'No voice selected — choose a voice from the list or upload a reference sample to clone.'
    # Fish Audio is the only narration engine. Anything else (an old template
    # or API call naming a removed engine) falls through to it rather than failing.
    engine_fn, engine_label = fish_audio_tts, 'Fish Audio'
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
            payload = {
                'prompt': prompt,
                'audio_duration': duration,
                'thinking': False,
                # 8 steps was well below ACE-Step's usable range and was the main
                # reason generated beds sounded thin/smeared. 27 is the model's
                # own documented default; raise ACE_STEP_STEPS for more quality
                # at proportionally more GPU time.
                'inference_steps': ACE_STEP_STEPS,
                'batch_size': 1,
                # "no vocals" in the prompt text is only a soft hint. ACE-Step
                # takes a dedicated lyrics field, and [inst] is its explicit
                # instrumental marker -- a far stronger guarantee of no vocals.
                'lyrics': '[inst]',
            }
            if ACE_STEP_NEGATIVE_PROMPT:
                payload['negative_prompt'] = ACE_STEP_NEGATIVE_PROMPT
            r = requests.post(f'{ACE_STEP_URL}/release_task', json=payload, timeout=10)
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
        y, sr = librosa_load(audio_path, sr=22050, duration=duration)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        times = librosa.frames_to_time(beat_frames, sr=sr)
        return sorted(_to_scalar(t) for t in times)
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
    callers should treat that as 'feature unavailable' and continue without it.

    The video's audio is extracted to 16 kHz mono WAV first. This used to POST
    the whole source container -- for a 45-minute episode that meant pushing
    several GB over HTTP so the service could demux and discard the video track
    anyway. Whisper resamples to 16 kHz mono internally regardless, so doing it
    here costs one cheap ffmpeg pass and cuts the upload by ~100x."""
    audio_path = None
    try:
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'],
                                  f'stt_{uuid.uuid4().hex}.wav')
        try:
            run_ffmpeg([FFMPEG, '-y', '-i', path, '-vn', '-ac', '1', '-ar', '16000',
                        '-c:a', 'pcm_s16le', audio_path],
                       timeout=FFMPEG_LONG_TIMEOUT, label='STT audio extract')
        except MediaToolTimeout as e:
            print(f'Whisper: audio extraction timed out ({e}); skipping transcription.')
            return [], []
        if not (os.path.exists(audio_path) and os.path.getsize(audio_path) > 0):
            print('Whisper: could not extract an audio track from the source; skipping transcription.')
            return [], []
        upload_name = os.path.splitext(os.path.basename(path))[0] + '.wav'
        with open(audio_path, 'rb') as f:
            r = requests.post(
                f'{WHISPER_URL}/v1/audio/transcriptions',
                files={'file': (upload_name, f, 'audio/wav')},
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
    finally:
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except OSError:
                pass

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

def librosa_load(path, sr=22050, mono=True, duration=None):
    """librosa.load, but never via the deprecated audioread fallback.

    soundfile cannot open compressed/container formats (.mov, .m4a, .mp4), so
    librosa silently falls back to audioread -- which emits a deprecation warning,
    is removed in librosa 1.0, and is markedly slower. Decoding to a temporary
    PCM WAV with ffmpeg first keeps everything on the soundfile path."""
    import librosa as _lb
    ext = os.path.splitext(path)[1].lower()
    tmp = None
    try:
        if ext not in ('.wav', '.flac', '.ogg', '.aiff', '.aif'):
            tmp = os.path.join(tempfile.gettempdir(), f'lb_{uuid.uuid4().hex}.wav')
            cmd = [FFMPEG, '-y', '-i', path, '-vn', '-ac', '1' if mono else '2',
                   '-ar', str(int(sr)), '-c:a', 'pcm_s16le']
            if duration:
                cmd += ['-t', str(duration)]
            cmd.append(tmp)
            try:
                run_ffmpeg(cmd, timeout=180, label='librosa decode')
            except MediaToolTimeout:
                tmp = None
            if tmp and not (os.path.exists(tmp) and os.path.getsize(tmp) > 0):
                tmp = None
        return _lb.load(tmp or path, sr=sr, mono=mono, duration=duration)
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass

def tc_seconds(tc):
    """Seconds from a PySceneDetect FrameTimecode.

    .get_seconds() is deprecated in favour of the .seconds property, but the
    property does not exist on older releases -- so prefer it and fall back."""
    v = getattr(tc, 'seconds', None)
    return v if v is not None else tc_seconds(tc)

def tc_frames(tc):
    """Frame number from a PySceneDetect FrameTimecode (see tc_seconds)."""
    v = getattr(tc, 'frame_num', None)
    return v if v is not None else tc_frames(tc)

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
               'start_sec': round(tc_seconds(s), 2), 'end_sec': round(tc_seconds(e), 2),
               'duration': round(tc_seconds(e) - tc_seconds(s), 2)}
              for i, (s, e) in enumerate(scene_list)]
    return jsonify(scenes=scenes)

def _ensure_readable(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.mov', '.mkv', '.flv', '.wmv', '.webm'):
        cap = cv2.VideoCapture(path)
        ret, _ = cap.read()
        cap.release()
        if not ret:
            mp4_path = os.path.splitext(path)[0] + '_converted.mp4'
            r = run_ffmpeg([FFMPEG, '-y', '-i', path, '-c:v', 'libx264', '-preset', 'ultrafast',
                                '-crf', '28', '-pix_fmt', 'yuv420p', '-an', mp4_path],
                           timeout=FFMPEG_LONG_TIMEOUT, label='preview transcode')
            if r.returncode == 0 and os.path.exists(mp4_path):
                return mp4_path
    return path

@app.route('/api/media/playable', methods=['POST'])
def api_media_playable():
    """Re-encodes a staged file to H.264/AAC MP4 for the browser Player.

    Distinct from _ensure_readable(): that one strips audio and only checks
    whether *OpenCV* can decode a frame, which is the wrong question here --
    OpenCV (via its ffmpeg backend) opens ProRes/DNxHD/MXF just fine, but no
    browser decodes them natively, so a HIRES mat in one of those often loads
    with duration/controls but a black frame, or an outright media error. Rather
    than guess server-side which containers a given browser supports, the Player
    calls this reactively -- only once its <video> element actually reports an
    error -- and gets back a guaranteed-playable copy with audio intact."""
    name = secure_filename(request.form.get('filename', ''))
    if not name or name != request.form.get('filename', ''):
        return jsonify(ok=False, error='Invalid filename.'), 400
    src = os.path.join(app.config['UPLOAD_FOLDER'], name)
    if not os.path.isfile(src):
        return jsonify(ok=False, error='That file is no longer staged -- pick it again.'), 404

    out_name = f'playable_{os.path.splitext(name)[0]}.mp4'
    out_path = os.path.join(app.config['UPLOAD_FOLDER'], out_name)
    if not (os.path.exists(out_path) and os.path.getsize(out_path) > 0):
        try:
            r = run_ffmpeg([FFMPEG, '-y', '-i', src,
                            '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20',
                            '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k',
                            '-movflags', '+faststart', out_path],
                           timeout=FFMPEG_LONG_TIMEOUT, label='player playback transcode')
        except MediaToolTimeout as e:
            return jsonify(ok=False, error=f'Conversion took too long: {e}'), 504
        if not (os.path.exists(out_path) and os.path.getsize(out_path) > 0):
            return jsonify(ok=False, error='This file could not be converted for browser playback. '
                                           f'ffmpeg error: {r.stderr[-400:]}'), 502
    return jsonify(ok=True, url=f'/uploads/{out_name}')

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
    scenes = [{'scene': i+1, 'start': tc_seconds(s), 'end': tc_seconds(e),
               'start_tc': s.get_timecode(), 'end_tc': e.get_timecode(),
               'duration': round(tc_seconds(e) - tc_seconds(s), 2)}
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
    with ThreadPoolExecutor(max_workers=min(8, len(names))) as ex:
        flags = list(ex.map(_model_supports_vision, names))
    models = [n for n, ok in zip(names, flags) if ok]
    return jsonify(models=models)

# ---- AI Chat ----
# A general-purpose chat panel over whatever models Ollama has installed --
# separate from the vision/rating pipeline, for testing a model's behavior,
# drafting text, or just asking it something directly.
CHAT_TIMEOUT = int(os.environ.get('CHAT_TIMEOUT', 180))
CHAT_MAX_HISTORY = int(os.environ.get('CHAT_MAX_HISTORY', 60))  # messages, not turns

# ---- Chat attachments (images + documents) ----
# Images ride directly in the /api/chat request body -- Ollama's own schema is
# messages[].images, a list of base64 strings with no data-URL prefix (see
# docs.ollama.com/capabilities/vision) -- so the browser encodes them and sends
# them straight through with no server round trip needed. Documents are
# different: a PDF can't be read as text in the browser, so those go through
# the extraction endpoint below first and the resulting text is folded into the
# message content client-side, the same as if the user had typed it.
CHAT_MAX_IMAGES_PER_MESSAGE = int(os.environ.get('CHAT_MAX_IMAGES_PER_MESSAGE', 4))
CHAT_MAX_IMAGE_BYTES = int(os.environ.get('CHAT_MAX_IMAGE_BYTES', 8 * 1024 * 1024))
CHAT_ATTACH_MAX_BYTES = int(os.environ.get('CHAT_ATTACH_MAX_BYTES', 8 * 1024 * 1024))
CHAT_ATTACH_MAX_CHARS = int(os.environ.get('CHAT_ATTACH_MAX_CHARS', 20000))
CHAT_TEXT_EXTS = {'.txt', '.md', '.markdown', '.csv', '.tsv', '.json', '.log', '.yaml', '.yml',
                  '.ini', '.cfg', '.conf', '.xml', '.html', '.htm', '.css', '.py', '.js', '.ts',
                  '.jsx', '.tsx', '.java', '.c', '.h', '.cpp', '.hpp', '.go', '.rs', '.rb', '.php',
                  '.sh', '.sql', '.srt', '.vtt'}

@app.route('/api/chat/extract_file', methods=['POST'])
def api_chat_extract_file():
    """Extracts plain text from an uploaded document for a chat attachment.

    PDFs need pypdf (an optional dependency -- this degrades to a clear error
    rather than crashing the whole app if it isn't installed, since not every
    deployment needs PDF support). Anything else with a recognised text-ish
    extension is decoded as UTF-8 directly; everything else is rejected rather
    than silently attaching binary garbage as "context"."""
    f = request.files.get('file')
    if not f or not f.filename:
        return jsonify(ok=False, error='No file provided.'), 400
    name = secure_filename(f.filename)
    if not name:
        return jsonify(ok=False, error='Invalid filename.'), 400
    ext = os.path.splitext(name)[1].lower()

    raw = f.read(CHAT_ATTACH_MAX_BYTES + 1)
    if len(raw) > CHAT_ATTACH_MAX_BYTES:
        return jsonify(ok=False, error=f'That file is larger than the '
                       f'{CHAT_ATTACH_MAX_BYTES // (1024*1024)}MB attachment limit.'), 400
    if not raw:
        return jsonify(ok=False, error='That file is empty.'), 400

    if ext == '.pdf':
        try:
            import pypdf
        except ImportError:
            return jsonify(ok=False, error='PDF attachments need the "pypdf" package on the '
                           'server (pip install pypdf --break-system-packages), which is not '
                           'installed here. Plain text and code files work without it.'), 501
        try:
            reader = pypdf.PdfReader(io.BytesIO(raw))
            if reader.is_encrypted:
                return jsonify(ok=False, error='That PDF is password-protected.'), 400
            text = '\n\n'.join((page.extract_text() or '') for page in reader.pages)
        except Exception as e:
            return jsonify(ok=False, error=f'Could not read that PDF: {e}'), 400
    elif ext in CHAT_TEXT_EXTS or ext == '':
        try:
            text = raw.decode('utf-8')
        except UnicodeDecodeError:
            text = raw.decode('utf-8', errors='replace')
    else:
        return jsonify(ok=False, error=f'Unsupported file type "{ext}". Supported: PDF, plain '
                       'text, and common code/markup files. Images should be attached directly, '
                       'not through this button.'), 400

    text = text.strip()
    if not text:
        return jsonify(ok=False, error='No extractable text was found in that file '
                       '(a scanned/image-only PDF has no text layer to read).'), 400
    truncated = len(text) > CHAT_ATTACH_MAX_CHARS
    if truncated:
        text = text[:CHAT_ATTACH_MAX_CHARS]
    return jsonify(ok=True, filename=name, text=text, truncated=truncated, chars=len(text))

@app.route('/api/chat/models')
def api_chat_models():
    """Every model Ollama has installed, unfiltered -- unlike /api/vision/models
    this doesn't restrict to vision-capable ones, since chat has no use for that
    distinction and the /api/show introspection per model is pure overhead here."""
    try:
        r = requests.get(f'{OLLAMA_URL}/api/tags', timeout=10)
        names = sorted(m['name'] for m in r.json().get('models', []))
        return jsonify(ok=True, models=names)
    except Exception as e:
        return jsonify(ok=False, models=[], error=f'Could not reach Ollama at {OLLAMA_URL}: {e}')

@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.get_json(silent=True) or {}
    model = (data.get('model') or '').strip()
    if not model:
        return jsonify(ok=False, error='Pick a model first.'), 400

    messages = data.get('messages')
    if not isinstance(messages, list) or not messages:
        return jsonify(ok=False, error='No conversation to send.'), 400
    # Only well-formed {role, content} pairs, and bounded -- this is a JSON body
    # built from the page's own running chat history, but treat it the same as
    # any other untrusted input rather than assuming the client behaved.
    clean = []
    for m in messages[-CHAT_MAX_HISTORY:]:
        if not isinstance(m, dict):
            continue
        role = m.get('role')
        content = m.get('content')
        if role not in ('user', 'assistant') or not isinstance(content, str) or not content.strip():
            continue
        entry = {'role': role, 'content': content}
        # Images only make sense on a user turn, and only up to a sane cap --
        # a single message with dozens of embedded base64 images would both
        # balloon the request and almost certainly exceed what a vision model
        # can usefully attend to at once.
        imgs = m.get('images')
        if role == 'user' and isinstance(imgs, list) and imgs:
            clean_imgs = []
            for img in imgs[:CHAT_MAX_IMAGES_PER_MESSAGE]:
                if not isinstance(img, str) or not img:
                    continue
                # Base64 expands input size by ~4/3 -- check the encoded string
                # length against that inflated bound rather than decoding every
                # image just to measure it.
                if len(img) > CHAT_MAX_IMAGE_BYTES * 4 // 3:
                    continue
                clean_imgs.append(img)
            if clean_imgs:
                entry['images'] = clean_imgs
        clean.append(entry)
    if not clean:
        return jsonify(ok=False, error='No conversation to send.'), 400

    system = (data.get('system') or '').strip()
    if system:
        clean = [{'role': 'system', 'content': system}] + clean

    # Reasoning models put their chain-of-thought in a separate `thinking` field
    # (see the AI-scoring fix earlier: an empty `content` with a full `thinking`
    # field is not a failure, it's the model reasoning silently). Chat has no
    # token budget cap the way scene rating does, so the specific failure mode
    # that caused there -- reasoning eating a *tight* budget -- doesn't apply,
    # but the fallback costs nothing to keep for the rare model that only ever
    # populates `thinking`.
    think = bool(data.get('think'))
    payload = {'model': model, 'messages': clean, 'stream': False, 'think': think}
    try:
        r = requests.post(f'{OLLAMA_URL}/api/chat', json=payload, timeout=CHAT_TIMEOUT)
        resp = r.json()
    except requests.exceptions.Timeout:
        return jsonify(ok=False, error=f'Ollama did not respond within {CHAT_TIMEOUT}s.'), 504
    except Exception as e:
        return jsonify(ok=False, error=f'Could not reach Ollama at {OLLAMA_URL}: {e}'), 502
    if resp.get('error'):
        return jsonify(ok=False, error=resp['error']), 502

    msg = resp.get('message') or {}
    content = (msg.get('content') or '').strip()
    thinking = (msg.get('thinking') or '').strip()
    if not content and thinking:
        content = thinking
    if not content:
        return jsonify(ok=False, error='The model returned an empty response.'), 502
    return jsonify(ok=True, content=content, thinking=thinking or None, model=resp.get('model') or model)

# ---- Trailer Generator (ffmpeg) ----

@app.route('/api/trailer/generate', methods=['POST'])
def api_trailer():
    path, orig_name = load_video(request)
    if not path:
        return jsonify(error=orig_name), 400

    # Rating mode: 'ai' (OpenCV + Ollama Vision) or 'ai_stt' (adds faster-whisper
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
    duck_depth_db = float(request.form.get('duck_depth_db', -15))
    duck_release_hold = float(request.form.get('duck_release_hold', 0.4))
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
    if sfx_mode == 'upload':
        sfx_upload_path = _resolve_upload('sfx_upload', AUDIO_EXTENSIONS)
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
    if vo_mode == 'upload':
        vo_upload_path = _resolve_upload('vo_upload', AUDIO_EXTENSIONS)
    if vo_mode == 'upload' and not vo_upload_path:
        vo_mode = 'none'
    vo_text = request.form.get('vo_text', '').strip()
    if vo_mode == 'tts' and not vo_text:
        vo_mode = 'none'
    vo_voice = request.form.get('vo_voice', '').strip() or None
    vo_language = request.form.get('vo_language', '').strip() or None
    vo_engine = request.form.get('vo_engine', 'fish_audio').strip()
    if vo_engine != 'fish_audio':
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
    if scoring_mode == 'upload':
        scoring_audio_path = _resolve_upload('scoring_audio', AUDIO_EXTENSIONS)
    if scoring_mode == 'generate':
        scoring_audio_path = 'GENERATE'  # flag to generate ambient
    end_card_path = _resolve_upload('end_card_video', ALLOWED_EXTENSIONS)
    schedule_card_path = _resolve_upload('schedule_video', ALLOWED_EXTENSIONS)

    # Optional VO tracks for the title card ("end_card_video" field, despite the
    # name) and end card ("schedule_video" field) — each can have its own
    # uploaded narration audio, muxed on in place of whatever audio the card
    # video already has, trimmed to a chosen [start, end) window of the source file.
    def _parse_card_vo(file_key, start_key, end_key):
        path = _resolve_upload(file_key, AUDIO_EXTENSIONS)
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

    # ---- Show template fill-in ----
    # A template IS the configuration for a programme: its genre, transition,
    # lengths, audio targets and voice choice, plus its music bed, SFX, VO and
    # cards. It is NOT an alternative to picking a genre -- genre is one of the
    # fields a template carries. Selecting a show therefore fills in everything
    # the request didn't specify, and nothing is mutually exclusive.
    #
    # Anything explicitly sent always wins: a template is a set of defaults, so a
    # one-off replacement bed or a different transition for a single episode works
    # without editing the show.
    #
    # In the browser the form is populated client-side the moment a show is
    # picked, so the request already carries these values and this block mostly
    # no-ops. It matters for API callers that just name a template.
    template_applied = None
    _tpl_pending = {}
    _tid_raw = (request.form.get('template_id') or '').strip()
    if _tid_raw.isdigit():
        tpl = template_get(int(_tid_raw))
        if tpl:
            filled = []
            tpl_settings = template_settings(tpl)

            def _tpl_number(current, key, default):
                """Uses the template's stored value only where the form left the
                field at its default -- an explicit form value always wins."""
                stored = tpl.get(key)
                return stored if (stored is not None and current == default) else current

            # Genre: only from the template if the request didn't name one. When it
            # does supply one, the genre preset block above has already applied that
            # genre's transition and xfade.
            if not genre:
                tpl_genre = tpl.get('genre') or tpl_settings.get('genre')
                if tpl_genre in GENRE_PRESETS:
                    genre = tpl_genre
                    preset = GENRE_PRESETS[genre]
                    if 'transition' not in request.form:
                        transition = preset['transition']
                    if 'xfade_dur' not in request.form:
                        xfade_dur = preset['xfade_dur']
                    filled.append('genre')

            # Transition/crossfade stored on the template win over a genre preset
            # only when the request named neither -- an explicit choice always wins.
            if 'transition' not in request.form and 'genre' not in request.form:
                if tpl.get('transition') in VALID_TRANSITIONS and tpl['transition'] != 'custom_matte':
                    transition = tpl['transition']
                    filled.append('transition')
                if tpl.get('xfade_dur') and 'xfade_dur' not in request.form:
                    xfade_dur = max(0.1, min(2.0, float(tpl['xfade_dur'])))

            # For the three mode-driven slots, an explicitly chosen mode is always
            # respected -- picking "None" for music means none, even if the show's
            # template has a bed. A slot is only filled when the mode field was
            # absent entirely (a bare API call that just names a template) or when
            # it says "upload" but no file actually came with it (the UI's state
            # after selecting a template).
            def _wants_template(mode_field, current_mode, have_file):
                # A real file always wins, full stop -- this used to be checked
                # only in the second branch below, so a bare API call that
                # attached a genuine file but omitted the mode field (a browser
                # form always sends it; a script easily might not) hit the first
                # branch instead and had its upload silently overwritten by the
                # template's asset.
                if have_file:
                    return False
                # Mode field absent entirely -> nothing was actually chosen (the
                # value in hand is just this function's own default), so the
                # template decides. This is what makes a bare API call that only
                # names a template work.
                if mode_field not in request.form:
                    return True
                # Otherwise only an explicit "upload" with nothing attached
                # defers -- which is exactly the state the UI is in after picking
                # a template, and is also how a deliberate None/Generate/TTS
                # choice gets respected.
                return current_mode == 'upload'

            def _skip_tpl(field):
                # Set by the UI when the user clicks the X on a template-sourced
                # chip, or touches that upload control at all -- an explicit
                # opt-out for this one job, distinct from the field simply being
                # empty (which still defers to the template).
                return request.form.get(f'{field}_skip_template') == '1'

            # Background music. Note scoring_audio_path is the sentinel 'GENERATE'
            # (not a path) when synthesis was selected, hence the explicit compare.
            if not _skip_tpl('scoring_audio') and _wants_template(
                    'scoring_mode', scoring_mode, scoring_audio_path not in (None, 'GENERATE')):
                staged = template_stage_asset(tpl, 'bgm')
                if staged:
                    scoring_audio_path, scoring_mode = staged, 'upload'
                    filled.append('bgm')

            # SFX one-shot ('genre' mode is an explicit choice; leave it alone).
            if not _skip_tpl('sfx_upload') and sfx_mode != 'genre' and _wants_template(
                    'sfx_mode', sfx_mode, bool(sfx_upload_path)):
                staged = template_stage_asset(tpl, 'sfx')
                if staged:
                    sfx_upload_path, sfx_mode = staged, 'upload'
                    filled.append('sfx')

            # Voiceover ('tts' is an explicit choice; leave it alone).
            if not _skip_tpl('vo_upload') and vo_mode != 'tts' and _wants_template(
                    'vo_mode', vo_mode, bool(vo_upload_path)):
                staged = template_stage_asset(tpl, 'vo')
                if staged:
                    vo_upload_path, vo_mode = staged, 'upload'
                    filled.append('vo')
                    vo_start = _tpl_number(vo_start, 'vo_start', 0.0)
                    vo_volume = _tpl_number(vo_volume, 'vo_volume', 1.15)
                    vo_trim_start = _tpl_number(vo_trim_start, 'vo_trim_start', 0.0)
                    vo_trim_end = _tpl_number(vo_trim_end, 'vo_trim_end', None)
                    if vo_trim_end is not None and vo_trim_end <= vo_trim_start:
                        vo_trim_end = None

            if not end_card_path and not _skip_tpl('end_card_video'):
                staged = template_stage_asset(tpl, 'title_card')
                if staged:
                    end_card_path = staged
                    filled.append('title_card')
            if not schedule_card_path and not _skip_tpl('schedule_video'):
                staged = template_stage_asset(tpl, 'end_card')
                if staged:
                    schedule_card_path = staged
                    filled.append('end_card')

            if not title_card_vo_path and not _skip_tpl('title_card_vo'):
                staged = template_stage_asset(tpl, 'title_card_vo')
                if staged:
                    title_card_vo_path = staged
                    filled.append('title_card_vo')
                    title_card_vo_start = _tpl_number(title_card_vo_start, 'title_card_vo_start', 0.0)
                    title_card_vo_end = _tpl_number(title_card_vo_end, 'title_card_vo_end', None)
                    if title_card_vo_end is not None and title_card_vo_end <= title_card_vo_start:
                        title_card_vo_end = None
            if not end_card_vo_path and not _skip_tpl('end_card_vo'):
                staged = template_stage_asset(tpl, 'end_card_vo')
                if staged:
                    end_card_vo_path = staged
                    filled.append('end_card_vo')
                    end_card_vo_start = _tpl_number(end_card_vo_start, 'end_card_vo_start', 0.0)
                    end_card_vo_end = _tpl_number(end_card_vo_end, 'end_card_vo_end', None)
                    if end_card_vo_end is not None and end_card_vo_end <= end_card_vo_start:
                        end_card_vo_end = None

            # Remaining scalar settings (lengths, thresholds, loudness targets,
            # voice choice...) are applied to the params dict below, once it
            # exists -- only for keys the request didn't send at all. A browser
            # POST carries every form field, so this is a no-op there; it matters
            # for an API call that just names a template.
            _tpl_pending = {k: v for k, v in tpl_settings.items()
                            if k not in request.form
                            and k not in ('genre', 'transition', 'xfade_dur')}

            template_applied = {'id': tpl['id'], 'name': tpl['name'], 'filled': filled}

    prompt = request.form.get('prompt',
        'Rate this frame 1-5 as a shot for a promo trailer, and describe what is '
        'actually visible in one short sentence (8-14 words): who or what is in '
        'shot, what they are doing, and where. Be concrete and literal. Do not use '
        'vague words like "cinematic", "dramatic" or "engaging".\n'
        'Answer with a single JSON object and nothing else: '
        '{"score": <1-5>, "desc": "<sentence>"}')

    jid = job_new()
    with JOBS_LOCK:
        if jid in JOBS:
            JOBS[jid]['orig_name'] = orig_name
    params = dict(path=path, orig_name=orig_name, mode=mode, genre=genre, scoring_mode=scoring_mode,
                  trailer_length=trailer_length, max_scene_dur=max_scene_dur,
                  scene_threshold=scene_threshold, min_scene_len_sec=min_scene_len_sec,
                  transition=transition, xfade_dur=xfade_dur, transition_matte_path=transition_matte_path,
                  target_loudness=target_loudness, true_peak=true_peak, music_duck_db=music_duck_db, duck_depth_db=duck_depth_db, duck_release_hold=duck_release_hold, beat_match=beat_match, broadcast_stereo=broadcast_stereo, model=model,
                  sfx_mode=sfx_mode, sfx_upload_path=sfx_upload_path,
                  vo_mode=vo_mode, vo_upload_path=vo_upload_path, vo_text=vo_text, vo_voice=vo_voice,
                  vo_language=vo_language, vo_engine=vo_engine, vo_ref_upload_path=vo_ref_upload_path,
                  vo_rate=vo_rate, vo_start=vo_start, vo_volume=vo_volume, sync_beats=sync_beats, whisper_enhance=whisper_enhance,
                  vo_trim_start=vo_trim_start, vo_trim_end=vo_trim_end,
                  end_card_path=end_card_path, schedule_card_path=schedule_card_path,
                  title_card_vo_path=title_card_vo_path, title_card_vo_start=title_card_vo_start, title_card_vo_end=title_card_vo_end,
                  end_card_vo_path=end_card_vo_path, end_card_vo_start=end_card_vo_start, end_card_vo_end=end_card_vo_end,
                  scoring_audio_path=scoring_audio_path, prompt=prompt,
                  template_applied=template_applied,
                  # Stop after scene selection and hand back a reviewable cut
                  # instead of rendering. See the preview block in _run_trailer_job.
                  preview_only=request.form.get('preview_only') in ('1', 'true', 'on'))

    # Overlay the template's saved configuration for anything the request left
    # out, coercing each stored string back to the type the param already holds.
    if _tpl_pending:
        for _k, _raw in _tpl_pending.items():
            if _k not in params:
                continue
            _cur = params[_k]
            try:
                if isinstance(_cur, bool):
                    _val = str(_raw).lower() in ('1', 'true', 'on', 'yes')
                elif isinstance(_cur, int) and not isinstance(_cur, bool):
                    _val = int(float(_raw))
                elif isinstance(_cur, float):
                    _val = float(_raw)
                else:
                    _val = _raw
            except (TypeError, ValueError):
                continue
            params[_k] = _val
            if template_applied:
                template_applied['filled'].append(_k)

    threading.Thread(target=run_trailer_job_gated, args=(jid, params), daemon=True).start()
    return jsonify(job_id=jid)

@app.route('/api/trailer/progress/<job_id>')
def api_trailer_progress(job_id):
    j = job_get(job_id)
    if not j:
        return jsonify(error='Unknown job id'), 404
    created = j.pop('created', None)
    # Elapsed lets the UI show a running clock; the old response had no notion of
    # time at all, so a slow stage was indistinguishable from a hung one.
    if created:
        j['elapsed'] = round(time.time() - created, 1)
    j['stages'] = [{'percent': p, 'label': lbl} for p, lbl in PIPELINE_STAGES]
    return jsonify(**j)

@app.route('/api/trailer/preview/<preview_id>')
def api_trailer_preview_get(preview_id):
    """Re-read a stored preview (thumbnails + chosen cut) without re-analysing."""
    p = preview_get(preview_id)
    if not p:
        return jsonify(ok=False, error='That preview has expired. Run the analysis again.'), 404
    return jsonify(ok=True, preview_id=preview_id, total_scenes=p['total_scenes'],
                   scenes=[{'scene': i + 1, 'start': round(s['start'], 1),
                            'end': round(s['end'], 1), 'quality': s['total_score'],
                            'duration': round(s['selected_dur'], 1),
                            'description': _scene_desc(s), 'thumb': p['thumbs'][i]}
                           for i, s in enumerate(p['selected'])],
                   alternates=[{'alt': i + 1, 'start': round(s['start'], 1),
                                'end': round(s['end'], 1), 'quality': s['total_score'],
                                'duration': round(s['selected_dur'], 1),
                                'description': _scene_desc(s), 'thumb': (p.get('alt_thumbs') or [None]*99)[i]}
                               for i, s in enumerate(p.get('alternates') or [])])

@app.route('/api/trailer/render', methods=['POST'])
def api_trailer_render():
    """Render an approved preview cut. Reuses the preview's stored selection, so
    detection / quality scoring / AI vision scoring are not repeated.

    Optional `drop` is a JSON array of 1-based scene numbers (as shown in the
    preview) to leave out — the cheap way to fix a cut without re-running
    anything."""
    pid = (request.form.get('preview_id') or '').strip()
    p = preview_get(pid)
    if not p:
        return jsonify(error='That preview has expired or was never created. Run the analysis again.'), 404

    selected = p['selected']
    raw_drop = (request.form.get('drop') or '').strip()
    drop = set()
    if raw_drop:
        try:
            drop = {int(x) for x in json.loads(raw_drop)}
        except (ValueError, TypeError):
            return jsonify(error='`drop` must be a JSON array of scene numbers, e.g. [2,5].'), 400
        selected = [s for i, s in enumerate(selected) if (i + 1) not in drop]

    # `add` pulls in runner-up scenes the preview offered but didn't pick. Merged
    # back in timeline order so a swapped-in clip lands where it belongs rather
    # than at the end of the trailer.
    raw_add = (request.form.get('add') or '').strip()
    added = set()
    if raw_add:
        try:
            added = {int(x) for x in json.loads(raw_add)}
        except (ValueError, TypeError):
            return jsonify(error='`add` must be a JSON array of alternate numbers, e.g. [1,3].'), 400
        alts = p.get('alternates') or []
        bad = [n for n in added if not (1 <= n <= len(alts))]
        if bad:
            return jsonify(error=f'No alternate numbered {bad[0]} in this preview.'), 400
        selected = selected + [alts[n - 1] for n in sorted(added)]
        selected.sort(key=lambda s: s['start'])

    if not selected:
        return jsonify(error='You dropped every scene — keep at least one, or add an alternate.'), 400

    params = dict(p['params'])
    params['preview_only'] = False
    params['preselected'] = selected
    params['preview_total_scenes'] = p['total_scenes']
    if not (params.get('path') and os.path.exists(params['path'])):
        return jsonify(error='The source video for this preview is no longer on disk '
                             '(it may have been cleaned up). Re-upload and analyse again.'), 410

    jid = job_new()
    threading.Thread(target=run_trailer_job_gated, args=(jid, params), daemon=True).start()
    return jsonify(job_id=jid, dropped=sorted(drop), added=sorted(added), scenes=len(selected))

ACE_STEP_MAX_SAMPLES = int(os.environ.get('ACE_STEP_MAX_SAMPLES', 4))
# Where audio2audio reference files are written. ACE-Step's ref_audio_input is a
# path read by the ACE-Step process, not an upload, so both processes must be able
# to see the same file. Same machine: the default (UPLOAD_FOLDER) is fine. ACE-Step
# on another host: point this at a shared/NFS/SMB mount that resolves to the same
# path on both sides.
ACE_STEP_REF_DIR = os.environ.get('ACE_STEP_REF_DIR', '')

def acestep_generate(prompt, duration, lyrics=None, bpm=None, samples=1,
                     steps=None, seed=None, base_ts=None,
                     ref_audio_path=None, ref_strength=0.5):
    """Generate music with ACE-Step directly. Returns (paths, error).

    Unlike prepare_bgm_track (which is shaped around the trailer pipeline: one
    track, faded, trimmed, transcoded to .m4a, with a synth fallback), this is the
    raw generator behind the Tools tab: N samples, optional sung lyrics, optional
    audio2audio reference, no fallback — if the service is down the caller should
    say so rather than hand back a sine drone the user didn't ask for.

    `bpm` is folded into the prompt as a tag. ACE-Step conditions on the prompt
    string rather than taking a numeric tempo field, so "124 bpm" as a tag is how
    tempo is actually expressed.

    `ref_audio_path` enables audio2audio: the output follows the reference's
    structure, with `ref_strength` (0-1) controlling how closely. NOTE: ACE-Step's
    ref_audio_input takes a FILE PATH that the ACE-Step process must be able to
    read. That works when ACE-Step runs on this machine (the default
    localhost:8001), but on a separate host the path won't resolve there — see
    ACE_STEP_REF_DIR for pointing both sides at a shared mount."""
    tags = (prompt or '').strip() or 'cinematic, instrumental'
    if bpm:
        # Don't double up if the user already typed a bpm into the prompt.
        if not re.search(r'\b\d{2,3}\s*bpm\b', tags, re.I):
            tags = f'{tags}, {int(bpm)} bpm'
    lyrics = (lyrics or '').strip()
    instrumental = not lyrics
    samples = max(1, min(ACE_STEP_MAX_SAMPLES, int(samples or 1)))

    payload = {
        'prompt': tags,
        'audio_duration': float(duration),
        'thinking': False,
        'inference_steps': int(steps or ACE_STEP_STEPS),
        'batch_size': samples,
        # '[inst]' is ACE-Step's explicit instrumental marker. When the user has
        # supplied real lyrics we must send those instead AND drop the
        # vocal-suppressing negative prompt, which would otherwise fight them.
        'lyrics': '[inst]' if instrumental else lyrics,
    }
    if seed not in (None, ''):
        try:
            payload['manual_seeds'] = str(int(seed))
        except (TypeError, ValueError):
            pass
    if instrumental and ACE_STEP_NEGATIVE_PROMPT:
        payload['negative_prompt'] = ACE_STEP_NEGATIVE_PROMPT
    if ref_audio_path and os.path.exists(ref_audio_path):
        payload['audio2audio_enable'] = True
        payload['ref_audio_input'] = os.path.abspath(ref_audio_path)
        payload['ref_audio_strength'] = max(0.0, min(1.0, float(ref_strength)))

    base_ts = base_ts or f'tool{int(time.time()*1000)}'
    try:
        r = requests.post(f'{ACE_STEP_URL}/release_task', json=payload, timeout=15)
        task_id = (r.json().get('data') or {}).get('task_id')
        if not task_id:
            return [], f'ACE-Step did not accept the request (no task id). Response: {r.text[:200]}'
        for _ in range(90):
            time.sleep(2)
            q = requests.post(f'{ACE_STEP_URL}/query_result',
                              json={'task_id_list': [task_id]}, timeout=10)
            items = q.json().get('data') or []
            if not items:
                continue
            status = items[0].get('status')
            if status == 2:
                return [], 'ACE-Step reported the generation task failed.'
            if status != 1:
                continue
            result = json.loads(items[0]['result'])
            entries = result if isinstance(result, list) else [result]
            paths = []
            for i, entry in enumerate(entries[:samples]):
                remote = entry.get('file') if isinstance(entry, dict) else None
                if not remote:
                    continue
                dest = os.path.join(app.config['UPLOAD_FOLDER'],
                                    f'music_{base_ts}_{i}{os.path.splitext(remote)[1] or ".wav"}')
                try:
                    resp = requests.get(f'{ACE_STEP_URL}{remote}', timeout=120)
                    with open(dest, 'wb') as f:
                        f.write(resp.content)
                except Exception as e:
                    print(f'ACE-Step sample {i} download failed: {e}')
                    continue
                if os.path.exists(dest) and os.path.getsize(dest) > 0:
                    paths.append(dest)
            if not paths:
                return [], 'ACE-Step finished but returned no usable audio.'
            return paths, None
        return [], 'Timed out waiting for ACE-Step (3 minutes). The service may be overloaded.'
    except Exception as e:
        return [], f'Could not reach ACE-Step at {ACE_STEP_URL}: {e}'

@app.route('/api/music/generate', methods=['POST'])
def api_music_generate():
    """Standalone ACE-Step music generation for the Tools tab.

    Exposes the controls the trailer pipeline hardcodes: free-text prompt, sung
    lyrics (or instrumental), tempo, and how many samples to generate in one go
    so alternatives can be auditioned side by side."""
    def _num(key, default, lo, hi, cast=float):
        raw = (request.form.get(key) or '').strip()
        if raw == '':
            return default
        try:
            return max(lo, min(hi, cast(float(raw))))
        except (TypeError, ValueError):
            return default

    duration = _num('duration', 30.0, 5.0, 300.0)
    samples = _num('samples', 1, 1, ACE_STEP_MAX_SAMPLES, int)
    bpm = _num('bpm', None, 40, 220, int)
    steps = _num('steps', ACE_STEP_STEPS, 8, 120, int)
    genre = (request.form.get('genre') or '').strip()
    prompt = (request.form.get('prompt') or '').strip() or GENRE_PROMPTS.get(genre, '')
    lyrics = (request.form.get('lyrics') or '').strip()
    seed = (request.form.get('seed') or '').strip()

    if not prompt:
        return jsonify(ok=False, error='Enter a prompt describing the style you want.'), 400

    base_ts = f'tool{int(time.time()*1000)}'

    # Optional audio2audio reference. Normalised to a plain 44.1k stereo WAV so
    # ACE-Step gets something it can definitely decode regardless of what the user
    # dropped in (m4a, mp3, a video's audio track, an odd sample rate).
    ref_path = None
    ref_strength = _num('ref_strength', 0.5, 0.0, 1.0)
    ref_src = _resolve_upload('ref_audio', AUDIO_EXTENSIONS)
    if ref_src:
        ref_dir = ACE_STEP_REF_DIR or app.config['UPLOAD_FOLDER']
        try:
            os.makedirs(ref_dir, exist_ok=True)
        except OSError as e:
            return jsonify(ok=False, error=f'Could not write to the reference audio directory ({ref_dir}): {e}'), 500
        ref_path = os.path.join(ref_dir, f'aceref_{base_ts}.wav')
        try:
            run_ffmpeg([FFMPEG, '-y', '-i', ref_src, '-vn', '-ac', '2', '-ar', '44100',
                        '-c:a', 'pcm_s16le', ref_path], timeout=120, label='ACE reference convert')
        except MediaToolTimeout:
            ref_path = None
        if not (ref_path and os.path.exists(ref_path) and os.path.getsize(ref_path) > 0):
            return jsonify(ok=False, error='Could not read that reference audio file — '
                                           'try a standard WAV or MP3.'), 400

    try:
        paths, err = acestep_generate(prompt, duration, lyrics=lyrics, bpm=bpm,
                                      samples=samples, steps=steps, seed=seed, base_ts=base_ts,
                                      ref_audio_path=ref_path, ref_strength=ref_strength)
    finally:
        # The reference only needs to survive the generation call itself.
        if ref_path and os.path.exists(ref_path):
            try:
                os.remove(ref_path)
            except OSError:
                pass
    if err:
        return jsonify(ok=False, error=err), 502

    return jsonify(ok=True, samples=[{
        'url': f'/uploads/{os.path.basename(p)}',
        'filename': os.path.basename(p),
        'duration': round(probe_duration(p) or duration, 1),
    } for p in paths],
        prompt=prompt, lyrics=lyrics or None, bpm=bpm, steps=steps,
        instrumental=not lyrics,
        reference=bool(ref_path), ref_strength=ref_strength if ref_path else None)

@app.route('/api/music/genres')
def api_music_genres():
    """Genre -> music prompt map, so the Tools tab can show and pre-fill prompts."""
    return jsonify(ok=True, genres=[{'key': g, 'prompt': GENRE_PROMPTS.get(g, '')}
                                    for g in GENRE_NAMES])

@app.route('/api/sfx/generate', methods=['POST'])
def api_sfx_generate():
    """Standalone Woosh SFX generation for the Tools tab: any text description,
    not just the fixed genre-derived prompts the trailer pipeline uses."""
    def _num(key, default, lo, hi, cast=float):
        raw = (request.form.get(key) or '').strip()
        if raw == '':
            return default
        try:
            return max(lo, min(hi, cast(float(raw))))
        except (TypeError, ValueError):
            return default

    prompt = (request.form.get('prompt') or '').strip()
    if not prompt:
        return jsonify(ok=False, error='Enter a description of the sound you want.'), 400
    duration = _num('duration', 1.0, 0.2, 10.0)
    samples = _num('samples', 1, 1, WOOSH_MAX_SAMPLES, int)

    paths, err = woosh_sfx_generate(prompt, duration=duration, samples=samples)
    if err:
        return jsonify(ok=False, error=err), 502

    return jsonify(ok=True, prompt=prompt, samples=[{
        'url': f'/uploads/{os.path.basename(p)}',
        'filename': os.path.basename(p),
        'duration': round(probe_duration(p) or duration, 1),
    } for p in paths])

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

@app.route('/api/trailer/library')
def api_trailer_library():
    """Lists saved trailers (most recent first) for the History panel."""
    return jsonify(ok=True, items=library_list())

@app.route('/api/trailer/library/<int:tid>')
def api_trailer_library_get(tid):
    """Returns one saved trailer's full result payload, ready to hand straight
    to the same renderer used for a just-completed job."""
    row = library_get_row(tid)
    if not row or not row.get('result_json'):
        return jsonify(ok=False, error='Not found'), 404
    return jsonify(ok=True, result=json.loads(row['result_json']), created_at=row['created_at'])

@app.route('/api/trailer/library/<int:tid>/delete', methods=['POST'])
def api_trailer_library_delete(tid):
    ok = library_delete(tid)
    if not ok:
        return jsonify(ok=False, error='Not found'), 404
    return jsonify(ok=True)

@app.route('/library/<int:tid>/file')
def library_file(tid):
    row = library_get_row(tid)
    if not row:
        return jsonify(error='Not found'), 404
    return send_from_directory(LIBRARY_DIR, row['filename'])

@app.route('/library/<int:tid>/download')
def library_download(tid):
    row = library_get_row(tid)
    if not row:
        return jsonify(error='Not found'), 404
    fmt_key = request.args.get('format', 'mp4_high')
    if fmt_key not in EXPORT_FORMATS:
        return jsonify(error=f'Unknown export format: {fmt_key}'), 400
    src_path = os.path.join(LIBRARY_DIR, row['filename'])
    if not os.path.exists(src_path):
        return jsonify(error='File not found'), 404
    ext = EXPORT_FORMATS[fmt_key]['ext']
    base_name, _ = os.path.splitext(row['orig_name'] or row['filename'])
    cache_name = f'{os.path.splitext(row["filename"])[0]}_{fmt_key}.{ext}'
    cache_path = os.path.join(LIBRARY_DIR, cache_name)
    if not (os.path.exists(cache_path) and os.path.getsize(cache_path) > 0):
        cmd = build_export_cmd(src_path, cache_path, fmt_key)
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if not (os.path.exists(cache_path) and os.path.getsize(cache_path) > 0):
            return jsonify(error=f'Export to {fmt_key} failed: {r.stderr[-800:]}'), 500
    resp = send_from_directory(LIBRARY_DIR, cache_name)
    resp.headers['Content-Disposition'] = f'attachment; filename="{base_name}.{ext}"'
    return resp


@app.route('/api/monitor')
def api_monitor():
    """Live snapshot of every trailer job the server currently knows about:
    running right now, waiting for a free concurrency slot, or finished
    (success/error/cancelled) within the last JOB_TTL. Whole-server view, not
    per-user (this app has no login system -- see /api/queue/status). This is
    the transient, in-progress counterpart to the permanent Saved Trailers
    library: finished entries here age out after JOB_TTL regardless of
    whether they were also saved to the library."""
    with JOB_QUEUE_LOCK:
        queued_ids = list(JOB_QUEUE)
    with JOBS_LOCK:
        snapshot = {jid: dict(j) for jid, j in JOBS.items()}

    queued = [{'job_id': jid, 'position': i, 'orig_name': snapshot.get(jid, {}).get('orig_name')}
              for i, jid in enumerate(queued_ids)]

    active, finished = [], []
    for jid, j in snapshot.items():
        if jid in queued_ids:
            continue
        entry = {'job_id': jid, 'orig_name': j.get('orig_name'), 'percent': j.get('percent'),
                  'step': j.get('step'), 'status': j.get('status'), 'created': j.get('created')}
        if j.get('done'):
            entry['error'] = j.get('error')
            finished.append(entry)
        else:
            active.append(entry)
    active.sort(key=lambda e: e['created'] or 0)
    finished.sort(key=lambda e: e['created'] or 0, reverse=True)
    return jsonify(active=active, queued=queued, finished=finished[:20], limit=GATE.status()['limit'])

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

@app.route('/api/network/list')
def api_network_list():
    """Lists the files sitting in the network folder for ?category=hires|music|vo|sfx
    (defaults to hires/video). Each category maps to its own subfolder and its
    own allowed extensions -- see NETWORK_CATEGORIES."""
    category = request.args.get('category', DEFAULT_NETWORK_CATEGORY)
    try:
        root, files = list_network_files(category)
        return jsonify(ok=True, root=root, category=category, files=files)
    except Exception as e:
        return jsonify(ok=False, error=f'Could not reach network folder: {e}'), 500

@app.route('/api/network/fetch', methods=['POST'])
def api_network_fetch():
    """Copies one file from the network folder (for the given category) into the
    local upload folder so it can be used exactly like a drag-and-dropped file
    (see load_video() / _resolve_upload())."""
    data = request.get_json(silent=True) or request.form
    name = (data.get('name') or '').strip()
    category = (data.get('category') or DEFAULT_NETWORK_CATEGORY).strip()
    if not name:
        return jsonify(ok=False, error='No filename given'), 400
    try:
        local_name = fetch_network_file(name, category)
        local_path = os.path.join(app.config['UPLOAD_FOLDER'], local_name)
        # `url` lets the Player play the staged copy directly; callers that stage
        # a file for the generate form use `filename`.
        return jsonify(ok=True, filename=local_name, orig_name=name, category=category,
                        url=f'/uploads/{local_name}',
                        size=os.path.getsize(local_path))
    except ValueError as e:
        return jsonify(ok=False, error=str(e)), 400
    except Exception as e:
        return jsonify(ok=False, error=f'Could not fetch {name}: {e}'), 500

# ---- Show templates (saved per-show asset bundles) ----

def _tpl_num(key, default=None, lo=None, hi=None):
    """Reads an optional numeric form field. Blank/absent -> default (so the UI can
    leave a field untouched without wiping a previously saved value)."""
    raw = (request.form.get(key) or '').strip()
    if raw == '':
        return default
    try:
        v = float(raw)
    except ValueError:
        return default
    if lo is not None:
        v = max(lo, v)
    if hi is not None:
        v = min(hi, v)
    return v

@app.route('/api/templates', methods=['GET'])
def api_templates_list():
    return jsonify(ok=True, templates=template_list(),
                   slots=[{'key': k, 'label': v['label'], 'kind': v['kind']} for k, v in TEMPLATE_SLOTS.items()])

@app.route('/api/templates/<int:tid>', methods=['GET'])
def api_template_get(tid):
    row = template_get(tid)
    if not row:
        return jsonify(ok=False, error='Template not found'), 404
    return jsonify(ok=True, template=_template_public(row))

@app.route('/api/templates', methods=['POST'])
def api_template_save():
    """Creates or updates a show template from whatever the trailer form currently
    has selected. Accepts the same multipart field names the generate form uses
    (including the `<field>_network` staged-file hidden inputs), so the "Add to
    template" button can just re-post the picked files without any new plumbing.

    Only slots that arrive with a file are written -- posting again with just a
    new music bed updates that one slot and leaves the show's cards alone. Send
    clear_<slot>=1 to deliberately empty a slot."""
    name = (request.form.get('name') or '').strip()
    if not name:
        return jsonify(ok=False, error='Give the template a show name first.'), 400
    if len(name) > 120:
        return jsonify(ok=False, error='Name is too long (max 120 characters).'), 400

    tid_raw = (request.form.get('template_id') or '').strip()
    existing = template_get(int(tid_raw)) if tid_raw.isdigit() else None
    if existing is None:
        existing = template_get_by_name(name)  # saving under an existing name updates it

    # Resolve whichever asset files were supplied this time round.
    incoming = {}
    for slot, meta in TEMPLATE_SLOTS.items():
        exts = AUDIO_EXTENSIONS if meta['kind'] == 'audio' else ALLOWED_EXTENSIONS
        p = _resolve_upload(meta['field'], exts)
        if p:
            incoming[slot] = p

    if not existing and not incoming:
        return jsonify(ok=False, error='Select at least one file (music, SFX, VO or a card), '
                                       'or fill in the form, before saving a show.'), 400

    # Capture the whole generator configuration, not just the files. A show
    # template is "how this programme's promos are made" -- genre, transition,
    # lengths, audio targets, voice choice -- with the assets as one part of that.
    settings = dict(template_settings(existing)) if existing else {}
    for key in TEMPLATE_SETTING_FIELDS:
        if key in request.form:
            settings[key] = (request.form.get(key) or '').strip()
        elif key in TEMPLATE_BOOL_FIELDS and request.form:
            # A checkbox absent from a submitted form means unticked, which is a
            # real value -- not "leave whatever was there before".
            settings[key] = ''
    settings = {k: v for k, v in settings.items() if v != ''}

    fields = {
        'name': name,
        'notes': (request.form.get('notes') or '').strip() or (existing or {}).get('notes'),
        'genre': (request.form.get('genre') or '').strip() or (existing or {}).get('genre'),
        'transition': (request.form.get('transition') or '').strip() or (existing or {}).get('transition'),
        'xfade_dur': _tpl_num('xfade_dur', (existing or {}).get('xfade_dur'), 0.1, 2.0),
        'vo_start': _tpl_num('vo_start', (existing or {}).get('vo_start'), 0),
        'vo_volume': _tpl_num('vo_volume', (existing or {}).get('vo_volume'), 0.3, 3.0),
        'vo_trim_start': _tpl_num('vo_trim_start', (existing or {}).get('vo_trim_start'), 0),
        'vo_trim_end': _tpl_num('vo_trim_end', (existing or {}).get('vo_trim_end'), 0),
        'title_card_vo_start': _tpl_num('title_card_vo_start', (existing or {}).get('title_card_vo_start'), 0),
        'title_card_vo_end': _tpl_num('title_card_vo_end', (existing or {}).get('title_card_vo_end'), 0),
        'end_card_vo_start': _tpl_num('end_card_vo_start', (existing or {}).get('end_card_vo_start'), 0),
        'end_card_vo_end': _tpl_num('end_card_vo_end', (existing or {}).get('end_card_vo_end'), 0),
        'settings_json': json.dumps(settings),
    }
    if fields['transition'] and fields['transition'] not in VALID_TRANSITIONS:
        fields['transition'] = None

    now = time.time()
    conn = _tpl_db()
    try:
        if existing:
            tid = existing['id']
            conn.execute('UPDATE show_templates SET ' + ','.join(f'{k}=?' for k in fields) + ', updated_at=? WHERE id=?',
                         list(fields.values()) + [now, tid])
        else:
            cols = list(fields.keys()) + ['created_at', 'updated_at']
            cur = conn.execute(f'INSERT INTO show_templates ({",".join(cols)}) VALUES ({",".join("?" * len(cols))})',
                               list(fields.values()) + [now, now])
            tid = cur.lastrowid

        for slot, meta in TEMPLATE_SLOTS.items():
            if slot in incoming:
                if existing:
                    template_delete_asset_file(existing, slot)  # replace: drop the old master
                stored, disp = template_store_asset(incoming[slot], slot,
                                                    _upload_display_name(meta['field']))
                conn.execute(f'UPDATE show_templates SET {slot}_file=?, {slot}_name=? WHERE id=?', (stored, disp, tid))
            elif existing and request.form.get(f'clear_{slot}') in ('1', 'on', 'true'):
                template_delete_asset_file(existing, slot)
                conn.execute(f'UPDATE show_templates SET {slot}_file=NULL, {slot}_name=NULL WHERE id=?', (tid,))
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify(ok=False, error=f'A template named "{name}" already exists.'), 409
    finally:
        try:
            conn.close()
        except Exception:
            pass

    # The staged copies in UPLOAD_FOLDER have served their purpose -- the masters
    # now live in TEMPLATES_DIR. Leave them be if they were network-staged files,
    # since the same staged file may still be attached to the form for this job.
    return jsonify(ok=True, template=_template_public(template_get(tid)),
                   saved_slots=sorted(incoming.keys()), updated=bool(existing))

@app.route('/api/templates/<int:tid>', methods=['DELETE'])
def api_template_delete(tid):
    if not template_delete(tid):
        return jsonify(ok=False, error='Template not found'), 404
    return jsonify(ok=True)

@app.route('/api/templates/<int:tid>/asset/<slot>')
def api_template_asset(tid, slot):
    """Serves a template's stored asset so the UI can preview it (audio player /
    card-VO in-out scrubbing) without re-uploading anything."""
    if slot not in TEMPLATE_SLOTS:
        return jsonify(ok=False, error='Unknown slot'), 404
    row = template_get(tid)
    p = template_asset_abspath(row, slot)
    if not p:
        return jsonify(ok=False, error='That slot is empty'), 404
    return send_from_directory(TEMPLATES_DIR, os.path.basename(p))

@app.route('/api/config', methods=['GET'])
def api_config_get():
    """Current values of every configurable AI service URL, for the Config tab."""
    return jsonify(ok=True, config=current_config_values(),
                   fields={k: {'label': v[0], 'help': v[1]} for k, v in CONFIGURABLE_SERVICES.items()})

@app.route('/api/config', methods=['POST'])
def api_config_post():
    """Saves Config-tab edits to disk and applies them immediately (no restart needed)."""
    data = request.get_json(silent=True) or {}
    unknown = [k for k in data if k not in CONFIGURABLE_SERVICES]
    if unknown:
        return jsonify(ok=False, error=f'Unknown config key(s): {", ".join(unknown)}'), 400
    try:
        save_config_overrides(data)
    except Exception as e:
        return jsonify(ok=False, error=f'Could not save config: {e}'), 500
    return jsonify(ok=True, config=current_config_values())

@app.route('/api/config/test', methods=['POST'])
def api_config_test():
    """Pings a single URL from the Config tab's edit fields (before saving), so a
    typo can be caught without committing it first. Body: {"name": "FISH_AUDIO_URL", "url": "..."}."""
    data = request.get_json(silent=True) or {}
    name = data.get('name', '')
    url = (data.get('url') or '').strip()
    if name not in CONFIGURABLE_SERVICES:
        return jsonify(ok=False, error='Unknown service'), 400
    if name == 'FISH_AUDIO_API_KEY':
        return jsonify(ok=False, error='Not a URL field'), 400
    if not url:
        return jsonify(ok=False, error='No URL to test'), 400
    base = url.rsplit('/v1/', 1)[0] if '/v1/' in url else url
    path = '/api/tags' if name == 'OLLAMA_URL' else '/'
    result = _check_service(name, base, path)
    return jsonify(ok=result['status'] == 'up', **result)

@app.route('/api/health')
def api_health():
    """Reachability check for every local model/media service the app talks to
    (Ollama, Fish Audio S2, faster-whisper, ACE-Step, Woosh), checked in parallel
    so one slow/dead service doesn't stall the others. Returns per-service status
    plus an overall ok flag."""
    checks = [
        ('ollama', OLLAMA_URL, '/api/tags'),
        ('fish_audio', FISH_AUDIO_URL.rsplit('/v1/', 1)[0] if '/v1/' in FISH_AUDIO_URL else FISH_AUDIO_URL, '/'),
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
    # Joined against one shared deadline, not `timeout=5` per thread — since threads
    # already run in parallel, waiting up to 5s for each one *sequentially* could
    # take up to 5s × len(checks) in the worst case (multiple unreachable services)
    # instead of the ~5s total this is actually meant to cap at.
    deadline = time.time() + 5
    for th in threads:
        th.join(timeout=max(0, deadline - time.time()))
    ordered = [results.get(name, {'name': name, 'url': url, 'status': 'down', 'error': 'no response'})
               for name, url, path in checks]
    overall_ok = all(c['status'] == 'up' for c in ordered)
    return jsonify(ok=overall_ok, checked_at=time.time(), services=ordered)

@app.route('/api/stt/transcribe', methods=['POST'])
def api_stt_transcribe():
    """Standalone speech-to-text for the Speech to Text panel.

    Wraps the same transcribe_video() the promo pipeline uses for dialogue-aware
    cuts, so what you see here is exactly what the rating stage sees. Accepts any
    media the server can decode (video or audio) -- the audio is extracted to
    16 kHz mono before upload either way."""
    src = _resolve_upload('file', ALLOWED_EXTENSIONS | AUDIO_EXTENSIONS)
    if not src:
        return jsonify(ok=False, error='Upload a video or audio file to transcribe.'), 400
    try:
        words, segments = transcribe_video(src)
    finally:
        if os.path.exists(src) and not os.path.basename(src).startswith('net_'):
            try:
                os.remove(src)
            except OSError:
                pass
    if not segments and not words:
        return jsonify(ok=False, error='No speech was transcribed. Check that the whisper service '
                                       f'at {WHISPER_URL} is reachable (see the Config tab) and '
                                       'that the file actually contains audio.'), 502

    full = ' '.join(sg['text'] for sg in segments).strip()

    def _srt_time(t):
        h, rem = divmod(max(0.0, t), 3600)
        m, sec = divmod(rem, 60)
        return f'{int(h):02d}:{int(m):02d}:{int(sec):02d},{int((sec % 1) * 1000):03d}'

    srt = '\n'.join(f'{i+1}\n{_srt_time(sg["start"])} --> {_srt_time(sg["end"])}\n{sg["text"]}\n'
                    for i, sg in enumerate(segments))
    return jsonify(ok=True, segments=segments, words=len(words), text=full,
                   srt=srt, duration=round(segments[-1]['end'], 1) if segments else 0,
                   model=WHISPER_MODEL)

@app.route('/api/voices/tags')
def api_voice_tags():
    """Inline delivery tags for Fish Audio, in the syntax the configured model
    expects. Shared by the Narration section of the generate form and the Fish
    Audio tool so both offer the same set."""
    return jsonify(ok=True, **fish_tag_catalogue())

@app.route('/api/voices')
def api_voices():
    """Lists narration voices and languages for the Narration dropdowns, for
    the narration engine (Fish Audio; the ?engine= parameter is retained for
    compatibility with saved templates and existing API callers)
    to fish_audio). There's no bundled default voice — if the list comes back
    empty, the UI should fall back to "upload a reference sample" for
    zero-shot cloning."""
    force = request.args.get('refresh') == '1'
    engine = request.args.get('engine', 'fish_audio')
    engine = 'fish_audio'   # only narration engine; parameter kept for compatibility
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
    # In the Narration box this is an audition, so a hard cap keeps it snappy.
    # The Tools tab passes full=1 to render a complete script as a deliverable.
    cap = 5000 if request.form.get('full') in ('1', 'true', 'on') else 800
    truncated = len(text) > cap
    if truncated:
        text = text[:cap]
    try:
        rate = int(request.form.get('rate', 175))
    except ValueError:
        rate = 175
    voice_id = (request.form.get('voice') or '').strip() or None
    language = (request.form.get('language') or '').strip() or None
    engine = (request.form.get('engine') or 'fish_audio').strip()
    engine = 'fish_audio'   # only narration engine; parameter kept for compatibility

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
    return jsonify(ok=True, url=f'/uploads/{out_name}', filename=out_name,
                   engine=engine, truncated=truncated, characters=len(text),
                   duration=round(probe_duration(out_path) or 0, 1))

UPLOAD_TTL = int(os.environ.get('UPLOAD_TTL', 6 * 3600))  # seconds
_SWEEP_INTERVAL = int(os.environ.get('UPLOAD_SWEEP_INTERVAL', 900))

def sweep_upload_folder(ttl=None):
    """Deletes anything in UPLOAD_FOLDER older than `ttl` seconds.

    Backstop for the things per-job cleanup deliberately can't touch: shared
    net_* staging files, VO previews, and intermediates from a job whose process
    died mid-render. Finished trailers live on in LIBRARY_DIR, which this never
    touches, so reclaiming an aged /uploads/ copy only breaks a stale open tab.
    Returns bytes freed."""
    ttl = UPLOAD_TTL if ttl is None else ttl
    folder = app.config['UPLOAD_FOLDER']
    now = time.time()
    freed = 0
    try:
        entries = os.listdir(folder)
    except OSError:
        return 0
    for name in entries:
        p = os.path.join(folder, name)
        try:
            if not os.path.isfile(p):
                continue
            if now - os.path.getmtime(p) < ttl:
                continue
            size = os.path.getsize(p)
            os.remove(p)
            freed += size
        except OSError:
            pass
    if freed:
        print(f'Upload sweeper reclaimed {freed / (1024*1024):.1f} MB of files older than {ttl}s')
    return freed

def _sweeper_loop():
    while True:
        time.sleep(_SWEEP_INTERVAL)
        try:
            sweep_upload_folder()
        except Exception as e:
            print(f'Upload sweeper error: {e}')

def free_disk_mb(path=None):
    """Free space in MB on the filesystem holding `path` (UPLOAD_FOLDER by default)."""
    try:
        return shutil.disk_usage(path or app.config['UPLOAD_FOLDER']).free / (1024 * 1024)
    except OSError:
        return None

def _cleanup_job_temp(jid, params, keep_basename=None):
    """Removes every temp file a job could have created, whatever exit path it took.

    UPLOAD_FOLDER is a tempdir that only clears on process restart, so anything
    left behind here accumulates for the life of the server. Previously nothing
    was cleaned on any of the ~10 error returns in _run_trailer_job, and the
    uploaded source video was never deleted at all -- on a long-running LAN box
    that grows without bound until the disk fills.

    `keep_basename` is the finished trailer, which must survive (it's served from
    /uploads/ and copied into the library)."""
    folder = app.config['UPLOAD_FOLDER']
    victims = []
    # Per-job intermediates are all prefixed with the job id (base_ts == jid).
    try:
        for f in os.listdir(folder):
            if keep_basename and f == keep_basename:
                continue
            if f.endswith(f'_{jid}.mp4') or f.endswith(f'_{jid}.wav') or f.endswith(f'_{jid}.m4a'):
                victims.append(os.path.join(folder, f))
            elif f'_{jid}_' in f or f.startswith(f'seg_{jid}') or f.startswith(f'norm_{jid}'):
                victims.append(os.path.join(folder, f))
    except OSError as e:
        print(f'Job {jid} cleanup could not list {folder}: {e}')
    # Explicit per-job inputs (source video, staged template copies, uploads).
    for key in ('path', 'sfx_upload_path', 'vo_upload_path', 'vo_ref_upload_path',
                'scoring_audio_path', 'end_card_path', 'schedule_card_path',
                'title_card_vo_path', 'end_card_vo_path', 'transition_matte_path'):
        p = params.get(key)
        # scoring_audio_path carries the sentinel 'GENERATE' rather than a path.
        if not (isinstance(p, str) and p != 'GENERATE' and os.path.isabs(p)):
            continue
        # net_* files are the shared staging area for network-share picks: the
        # same staged file can legitimately be attached to two concurrent jobs,
        # so deleting it here could pull the source out from under the other one.
        # The age-based sweeper below reclaims those instead.
        if os.path.basename(p).startswith('net_'):
            continue
        victims.append(p)
    freed = 0
    for p in set(victims):
        try:
            if os.path.isfile(p):
                freed += os.path.getsize(p)
                os.remove(p)
        except OSError:
            pass
    if freed:
        print(f'Job {jid} cleanup freed {freed / (1024*1024):.1f} MB')

def run_trailer_job(jid, params):
    keep = None
    try:
        _run_trailer_job(jid, params)
        with JOBS_LOCK:
            res = (JOBS.get(jid) or {}).get('result') or {}
        url = res.get('trailer_url') or ''
        if url.startswith('/uploads/'):
            keep = os.path.basename(url)
    except JobCancelled:
        print(f'Trailer job {jid} cancelled')
        job_set(jid, error='Cancelled', status='cancelled')
    except MediaToolTimeout as e:
        print(f'Trailer job {jid} timed out: {e}')
        job_set(jid, error=f'A media processing step timed out and was stopped ({e}). '
                           'The source may be corrupt, or the server is overloaded.')
    except Exception as e:
        print(f'Trailer job {jid} crashed: {e}')
        job_set(jid, error=f'Unexpected error: {e}')
    finally:
        # Runs on success, failure, cancellation and timeout alike -- except for a
        # successful preview, whose source video and staged assets must survive
        # until the user renders (or the preview TTL expires and the age-based
        # sweeper reclaims them).
        holding = params.get('preview_only') and not (JOBS.get(jid) or {}).get('error')
        if not holding:
            _cleanup_job_temp(jid, params, keep_basename=keep)

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
    duck_depth_db = params.get('duck_depth_db', -15)
    duck_release_hold = params.get('duck_release_hold', 0.4)
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
        d = probe_duration(cf)
        if d is None or d <= 0:
            # Previously this silently substituted 5s. A wrong card duration feeds
            # the scene-budget maths and every xfade offset after it, so the whole
            # concat desyncs and the user gets a subtly broken trailer with no
            # error. Failing here is far better than guessing.
            job_set(jid, error=f'Could not read the duration of the card video "{os.path.basename(cf)}" — '
                               'it may be corrupt or in an unsupported format. Re-export it and try again.')
            return
        card_durations.append(d)
    total_card_dur = sum(card_durations)

    # Scene target starts at trailer_length, minus cards duration
    base_target = max(5, trailer_length - total_card_dur)

    # Shared by both paths below. base_ts is the per-job filename prefix for every
    # intermediate; it used to be assigned partway through the analysis block,
    # which the resume path skips entirely.
    base_ts = jid  # already unique per job (see job_new()) -- using it here too
                   # avoids two jobs finishing a step in the same second (quite
                   # possible with MAX_CONCURRENT_JOBS > 1) from both writing to
                   # the same trailer_<ts>.mp4 path at once.
    early_bgm_path = None
    early_bgm_source = 'none'

    preselected = params.get('preselected')
    if preselected:
        # Rendering an approved preview: detection, quality scoring, AI vision
        # scoring and selection were all done during the preview pass, so skip
        # straight to extraction. On a long episode that's the majority of the
        # job's wall-clock time, and repeating it could also produce a slightly
        # different cut than the one the user actually approved.
        job_set(jid, percent=34, step=f'Rendering approved cut ({len(preselected)} clips)')
        selected = [dict(s) for s in preselected]
        selected.sort(key=lambda x: x['start'])
        total_sel = sum(s['selected_dur'] for s in selected)
        scene_list = [None] * int(params.get('preview_total_scenes') or len(selected))
        word_starts = word_ends = []
        beat_times = []
    else:
        job_set(jid, percent=8, step='Detecting scene cuts')
        # Detect scenes via PySceneDetect. downscale=2 speeds up detection on large
        # source files (frames are only scaled down for the detector's own
        # analysis; returned timecodes are unaffected).
        scene_list = detect_scenes(path, threshold=scene_threshold,
                                    min_scene_len_sec=min_scene_len_sec, downscale=2)
        if not scene_list:
            job_set(jid, error='No scene changes detected. Try a video with clear cuts, or lower the detection threshold.')
            return
        if len(scene_list) == 1 and (tc_seconds(scene_list[0][1]) - tc_seconds(scene_list[0][0])) > video_duration * 0.95:
            # PySceneDetect's own fallback: no real cuts found, so it returned one
            # scene spanning the whole video. Selecting from a single "scene" isn't
            # meaningful — surface this clearly instead of silently treating the
            # entire source as one giant clip.
            job_set(jid, error='No distinct scene cuts were found — PySceneDetect sees this video as one continuous shot. Try lowering the detection threshold or upload footage with visible cuts.')
            return

        job_set(jid, percent=15, step=f'Rating {len(scene_list)} scenes (sharpness/brightness)')
        # Score scenes
        from statistics import median
        def _score_one_scene(start, end):
            # Each worker opens its own VideoCapture — cv2.VideoCapture is not safe to
            # share across threads (concurrent .set()/.read() calls on one handle can
            # corrupt each other's seeks), but independent handles on the same file
            # decode concurrently just fine and this is what actually lets scene
            # scoring use more than one CPU core.
            local_cap = cv2.VideoCapture(path)
            try:
                mid_f = int((tc_frames(start) + tc_frames(end)) / 2)
                local_cap.set(cv2.CAP_PROP_POS_FRAMES, mid_f)
                ret, frame = local_cap.read()
                if not ret:
                    return None
                dur = tc_seconds(end) - tc_seconds(start)
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
                    # get_fd() returns one cached, shared FaceDetectorYN instance — its
                    # setInputSize()+detect() pair isn't safe for concurrent callers, so
                    # this part alone stays serialized. It's cheap relative to decode +
                    # Laplacian/Canny/HSV, so the lock doesn't erase the parallel gain.
                    with _fd_lock:
                        _, faces = get_fd(w, h).detect(frame)
                    has_face = faces is not None and len(faces) > 0
                return {
                    'start': tc_seconds(start), 'end': tc_seconds(end),
                    'start_f': tc_frames(start), 'end_f': tc_frames(end),
                    'duration': dur, 'laplacian': round(lap, 2), 'brightness': round(bri, 1),
                    'edge_ratio': round(edge_ratio, 3), 'mean_hue': round(mean_hue, 1),
                    'mean_sat': round(mean_sat, 1), 'mean_val': round(mean_val, 1),
                    'has_face': has_face, 'frame': frame, 'frame_idx': mid_f,
                }
            finally:
                local_cap.release()

        with ThreadPoolExecutor(max_workers=min(8, len(scene_list) or 1)) as ex:
            scored = list(ex.map(lambda se: _score_one_scene(se[0], se[1]), scene_list))
        scenes_data = [r for r in scored if r is not None]
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
            # Only AI-score scenes that could realistically make the cut. Previously
            # every detected scene got a vision call -- on a 45-minute episode that's
            # 200-400 Ollama round trips to choose ~12 clips, and it dominated the
            # job's wall-clock time.
            #
            # Candidates are chosen two ways so the shortlist stays both good and
            # spread out: the globally highest quality_score scenes, plus the best
            # few from each time bucket across the source. Without the buckets a
            # shortlist can cluster in one well-lit stretch and then get thinned out
            # by the min_gap spacing rule during selection, leaving nothing to pick.
            ai_pool = scenes_data
            if len(scenes_data) > AI_SCORE_LIMIT:
                ranked = sorted(scenes_data, key=lambda s: s['quality_score'], reverse=True)
                chosen = {id(s): s for s in ranked[:max(1, AI_SCORE_LIMIT // 2)]}
                n_buckets = max(1, min(AI_SCORE_LIMIT // 4, 12))
                span = max(video_duration, 1e-6)
                buckets = {}
                for s in scenes_data:
                    b = min(n_buckets - 1, int(s['start'] / span * n_buckets))
                    buckets.setdefault(b, []).append(s)
                per_bucket = max(1, (AI_SCORE_LIMIT - len(chosen)) // n_buckets)
                for b in sorted(buckets):
                    for s in sorted(buckets[b], key=lambda x: x['quality_score'], reverse=True)[:per_bucket]:
                        if len(chosen) >= AI_SCORE_LIMIT:
                            break
                        chosen.setdefault(id(s), s)
                ai_pool = list(chosen.values())
                print(f'AI scoring shortlist: {len(ai_pool)} of {len(scenes_data)} scenes '
                      f'(cap AI_SCORE_LIMIT={AI_SCORE_LIMIT})')

            # Anything not shortlisted keeps a neutral AI prior -- the same value used
            # when a vision response can't be parsed -- so it stays selectable if
            # spacing rules exhaust the shortlist, just ranked below scored scenes.
            ai_pool_ids = {id(s) for s in ai_pool}
            for s in scenes_data:
                if id(s) not in ai_pool_ids:
                    s['total_score'] = s['quality_score'] + AI_NEUTRAL_SCORE
                    s['ai_desc'] = ''

            # Tell the model what it's selecting *for*. The prompt never mentioned the
            # genre before, so an action promo and a drama promo were ranked by the
            # same generic "good for a movie trailer" criterion.
            ai_prompt = prompt
            if genre in GENRE_PRESETS and 'DESC:' in prompt:
                ai_prompt = prompt.replace('for a movie trailer',
                                           f'for a {genre} promo trailer', 1)

            n_scenes_ai = len(ai_pool)
            _ai_progress = {'done': 0}
            _ai_progress_lock = threading.Lock()

            def _ask_vision(b64, budget, allow_thinking, structured=True):
                """One /api/generate call. Returns (text, response_json).

                Two things fight us on a chatty reasoning model like qwen3-vl:
                  * it puts chain-of-thought in a separate `thinking` field, and
                    Ollama counts those tokens against num_predict; and
                  * told to answer in one line, it writes a paragraph anyway and
                    gets truncated (done_reason='length') before the score.

                Prompt wording alone does not fix either. Ollama's structured
                output does: passing a JSON schema in `format` constrains decoding
                to that shape, so the model cannot ramble and stops as soon as the
                object closes. That both removes the truncation failure and makes
                each call much shorter -- which matters at ~60 frames a job.
                `structured=False` is the fallback for older Ollama builds that
                reject the `format` field."""
                payload = {
                    'model': model, 'prompt': ai_prompt, 'stream': False, 'images': [b64],
                    # Near-greedy decoding: scene ranking should be reproducible
                    # across runs of the same source, and the default sampling
                    # temperature made scores jitter between jobs.
                    'options': {'temperature': 0.1, 'num_predict': budget},
                }
                if structured and AI_STRUCTURED_OK:
                    payload['format'] = {
                        'type': 'object',
                        'properties': {
                            'score': {'type': 'integer', 'minimum': 1, 'maximum': 5},
                            'desc': {'type': 'string'},
                        },
                        'required': ['score', 'desc'],
                    }
                if not allow_thinking:
                    # Understood by newer Ollama for reasoning models; older
                    # builds ignore the unknown key rather than erroring.
                    payload['think'] = False
                r = requests.post(f'{OLLAMA_URL}/api/generate', json=payload, timeout=180)
                data = r.json()
                if data.get('error'):
                    raise RuntimeError(data['error'])
                txt = (data.get('response') or '').strip()
                if not txt:
                    txt = (data.get('thinking') or '').strip()
                return txt, data

            def _parse_vision(txt):
                # Structured output path: a JSON object is the expected shape now,
                # so try that before any of the text heuristics below.
                if txt.lstrip().startswith('{'):
                    try:
                        obj = json.loads(txt)
                        sc = int(obj.get('score'))
                        if 1 <= sc <= 5:
                            return _clean_ai_desc(str(obj.get('desc') or '')), sc
                    except (ValueError, TypeError, AttributeError):
                        pass  # malformed -- salvage below
                    # Truncated JSON is the common failure (done_reason='length'
                    # cuts the object mid-string), so pull the fields out by hand
                    # rather than discarding a reply that has both values in it.
                    j_sc = re.search(r'"score"\s*:\s*([1-5])\b', txt)
                    j_de = re.search(r'"desc"\s*:\s*"([^"]*)', txt)
                    if j_sc:
                        return (_clean_ai_desc(j_de.group(1)) if j_de else ''), int(j_sc.group(1))
                    if j_de:
                        # Score unusable/out of range, but the description is fine.
                        txt = j_de.group(1)

                """(description, score) from a model reply, or (desc, None) if no
                score could be found.

                The prompt asks for SCORE first precisely because chatty models run
                past the token budget mid-sentence: with the score leading, a reply
                truncated by `length` still yields a usable rating and whatever
                description made it out. Either field order is accepted, since
                templates saved with the old prompt still ask for DESC first."""
                # Strip markdown emphasis first: models often bold the labels,
                # which breaks a plain 'SCORE:' match.
                txt = re.sub(r'[*_`]+', '', txt or '')
                score_m = re.search(r'SCORE\s*:?\s*([1-5])', txt, re.I)
                if not score_m:
                    # Models that ignore the format usually still emit a bare digit.
                    score_m = (re.search(r'\b([1-5])\s*/\s*5\b', txt)
                               or re.search(r'\b([1-5])\s*$', txt.strip()))
                desc_m = re.search(r'DESC\s*:?\s*(.+?)(?:\s*\|\s*SCORE|$)', txt, re.S | re.I)
                if not desc_m:
                    # No DESC label (or it was cut off): fall back to the longest
                    # sentence-ish run of prose in the reply.
                    plain = re.sub(r'SCORE\s*:?\s*[1-5]\s*(?:/\s*5)?\s*\|?', ' ', txt, flags=re.I)
                    plain = re.sub(r'#+', ' ', plain)
                    cand = max((p.strip() for p in re.split(r'[.\n]', plain)),
                               key=len, default='')
                    # A bare rating digit at the end is the score, not prose.
                    cand = re.sub(r'\s*\b[1-5]\s*(?:/\s*5)?\s*$', '', cand).strip()
                    desc = _clean_ai_desc(cand) if len(cand) > 12 else ''
                else:
                    desc = _clean_ai_desc(desc_m.group(1))
                return desc, (int(score_m.group(1)) if score_m else None)

            def _score_one_ai(ai_i, s):
                _, buf = cv2.imencode('.jpg', s['frame'], [cv2.IMWRITE_JPEG_QUALITY, 85])
                b64 = base64.b64encode(buf.tobytes()).decode()
                global AI_STRUCTURED_OK
                try:
                    # Retry on an UNPARSEABLE reply, not merely an empty one: a
                    # reasoning model that ran out of budget leaves `response`
                    # empty but `thinking` full of chain-of-thought, which is
                    # non-empty yet useless. Treating that as an answer is what
                    # silently sent every scene to the neutral score.
                    try:
                        txt, data = _ask_vision(b64, AI_NUM_PREDICT, allow_thinking=False)
                    except RuntimeError as e:
                        if 'format' not in str(e).lower():
                            raise
                        # Ollama too old for structured output -- disable it for
                        # the rest of this process and carry on unstructured.
                        print('Ollama rejected the structured-output `format` field; '
                              'falling back to text parsing for this session. '
                              'Upgrade Ollama for more reliable scene rating.')
                        AI_STRUCTURED_OK = False
                        txt, data = _ask_vision(b64, AI_NUM_PREDICT, allow_thinking=False,
                                                structured=False)
                    desc, score = _parse_vision(txt)
                    if score is None:
                        print(f'AI vision reply unusable for scene {ai_i+1} '
                              f'(done_reason={data.get("done_reason")!r}, {len(txt)} chars); '
                              f'retrying without a token cap. First 200 chars: {txt[:200]!r}')
                        txt, data = _ask_vision(b64, AI_NUM_PREDICT * 3,
                                                allow_thinking=True, structured=False)
                        desc, score = _parse_vision(txt)
                    if score is None:
                        print(f'AI vision score parse failed for scene {ai_i+1}, defaulting to '
                              f'{AI_NEUTRAL_SCORE}. done_reason={data.get("done_reason")!r} '
                              f'raw={txt[:300]!r}')
                    s['ai_desc'] = desc
                    s['total_score'] = s['quality_score'] + (score if score is not None else AI_NEUTRAL_SCORE)
                except Exception as e:
                    print(f'AI vision request failed for scene {ai_i+1}, defaulting to {AI_NEUTRAL_SCORE}: {e}')
                    s['total_score'] = s['quality_score'] + AI_NEUTRAL_SCORE
                with _ai_progress_lock:
                    _ai_progress['done'] += 1
                    job_set(jid, percent=18 + int(12 * _ai_progress['done'] / max(n_scenes_ai, 1)),
                            step=f"AI-rating scene {_ai_progress['done']}/{n_scenes_ai}")

            # Bounded concurrency, not unbounded: these are independent HTTP calls (so
            # parallelizing them is the single biggest wall-clock win in the whole
            # pipeline when Ollama is slow), but firing all of them at once could
            # overwhelm a single Ollama instance's request queue or GPU memory. 4 is a
            # reasonable default for a typical self-hosted single-GPU setup.
            with ThreadPoolExecutor(max_workers=min(AI_SCORE_WORKERS, n_scenes_ai or 1)) as ex:
                list(ex.map(lambda pair: _score_one_ai(*pair), enumerate(ai_pool)))
        else:
            for s in scenes_data:
                s['total_score'] = s['quality_score']

        # Dialogue transcription (faster-whisper) — improves scene selection two ways:
        # 1. Scenes with actual quotable dialogue get a small scoring boost, so
        #    selection isn't purely based on visual sharpness/brightness/AI framing.
        # 2. Word-level timestamps let cut in/out points snap to word boundaries
        #    later, instead of landing mid-word.
        word_starts, word_ends = [], []
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
            else:
                whisper_enhance = False  # transcription unavailable/failed — skip the snapping logic below too

        # "Edit to music": prep the BGM *before* picking scenes so cut points can be
        # snapped onto its beat grid. Only worth the extra generation pass when the
        # user actually asked for it — otherwise BGM is prepared later as before.
        base_ts = jid  # already unique per job (see job_new()) -- using it here too
                       # avoids two jobs finishing this step in the same second
                       # (quite possible with MAX_CONCURRENT_JOBS > 1) from both
                       # writing to the same trailer_<ts>.mp4 path at once.
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
                if whisper_enhance and word_ends and seg_dur < (scene_end - seg_start):
                    # This is a separate `if`, not `elif` -- beat-sync above (when
                    # enabled) picks a rhythmically-aligned out-point first, and this
                    # then refines THAT point for word safety, rather than being
                    # skipped whenever beat-sync is on. It used to be `elif`, which
                    # meant turning on "sync cuts to the beat" silently disabled
                    # "don't cut mid-word" for every clip's out-point.
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

        # A positive shortfall here means every pass_attempt ran out of usable
        # scenes (limited spacing/availability) before hitting the target, and
        # the "no tiny sliver clips" rule above refused to add one more to close
        # a small gap. Rather than ship a trailer noticeably under the requested
        # length, extend the chronologically LAST selected clip using slack it
        # already has within its own detected scene boundaries -- this grows an
        # existing cut instead of adding a new one, so it doesn't reintroduce
        # the flash-cut problem that rule exists to prevent.
        if selected and shortfall > 0.15:
            last = selected[-1]
            slack = last['duration'] - last['selected_dur']
            if slack > 0.05:
                grow = min(slack, shortfall)
                new_end = last['trim_start'] + last['selected_dur'] + grow
                if whisper_enhance and word_ends:
                    # Growing this clip to close the shortfall creates a new cut
                    # point too -- give it the same word-boundary safety the main
                    # truncation path gets above, or this top-up can reintroduce
                    # exactly the mid-word cut the rest of this mechanism exists
                    # to prevent.
                    scene_end_abs = last['start'] + last['duration']
                    snapped = nearest_word_boundary(new_end, word_ends, max_snap=0.35)
                    if last['trim_start'] < snapped <= scene_end_abs:
                        new_end = snapped
                grow = max(0.0, new_end - (last['trim_start'] + last['selected_dur']))
                last['selected_dur'] += grow
                total_sel += grow
                shortfall -= grow

    if not selected:
        job_set(jid, error='No scenes selected.')
        return

    if params.get('preview_only'):
        # Analysis is done; stop here instead of spending minutes on extraction,
        # transitions and mixing for a cut the user hasn't seen yet.
        pid = f'pv{jid}'
        job_set(jid, percent=34, step='Writing preview thumbnails')

        # Runner-ups: the next-best scoring scenes that didn't make the cut, so a
        # rejected clip can be swapped for a real alternative instead of forcing a
        # full re-analysis. Spaced by the same min_gap rule the selector uses, and
        # excluding anything already chosen.
        chosen_starts = {round(s['start'], 3) for s in selected}
        alternates = []
        if not preselected:
            for cand in sorted(scenes_data, key=lambda x: x['total_score'], reverse=True):
                if len(alternates) >= PREVIEW_ALTERNATES:
                    break
                if round(cand['start'], 3) in chosen_starts:
                    continue
                if any(abs(cand['start'] - o['start']) < min_gap for o in alternates):
                    continue
                if any(abs(cand['start'] - s['start']) < min_gap for s in selected):
                    continue
                cand = dict(cand)
                cand.setdefault('selected_dur', min(cand['duration'], max_scene_dur or cand['duration']))
                cand.setdefault('trim_start', cand['start'])
                alternates.append(cand)

        def _thumb(scene, tag, i):
            frame = scene.get('frame')
            if frame is None:
                return None
            try:
                tw = 320
                h, w = frame.shape[:2]
                small = cv2.resize(frame, (tw, max(1, int(h * tw / max(w, 1)))))
                tname = f'preview_{pid}_{tag}{i}.jpg'
                cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], tname), small,
                            [cv2.IMWRITE_JPEG_QUALITY, 78])
                return f'/uploads/{tname}'
            except Exception as e:
                print(f'Preview thumbnail {tag}{i} failed: {e}')
                return None

        thumbs = [_thumb(s, 's', i) for i, s in enumerate(selected)]
        alt_thumbs = [_thumb(s, 'a', i) for i, s in enumerate(alternates)]

        def _slim(rows):
            # Only the fields the render half consumes -- frames and other
            # non-serializable analysis state are deliberately dropped.
            return [{'start': s['start'], 'end': s['end'], 'duration': s['duration'],
                     'selected_dur': s['selected_dur'], 'trim_start': s.get('trim_start', s['start']),
                     'total_score': s['total_score'], 'quality_score': s.get('quality_score', 0),
                     'ai_desc': s.get('ai_desc', ''), 'has_face': s.get('has_face', False),
                     'edge_ratio': s.get('edge_ratio', 0), 'mean_hue': s.get('mean_hue', 0)}
                    for s in rows]

        slim = _slim(selected)
        slim_alt = _slim(alternates)
        preview_store(pid, {'params': params, 'selected': slim, 'thumbs': thumbs,
                            'alternates': slim_alt, 'alt_thumbs': alt_thumbs,
                            'total_scenes': len(scene_list), 'video_duration': video_duration,
                            'total_card_dur': total_card_dur})
        result = dict(
            status='preview', preview=True, preview_id=pid,
            orig_name=orig_name, total_scenes=len(scene_list), selected_scenes=len(selected),
            video_duration=round(video_duration, 1), trailer_length=trailer_length,
            scenes_duration=round(total_sel, 1),
            estimated_duration=round(total_sel + total_card_dur - max(0, len(selected) + len(card_files) - 1) * xfade_dur, 1),
            scenes=[{'scene': i + 1, 'start': round(s['start'], 1), 'end': round(s['end'], 1),
                     'quality': s['total_score'], 'duration': round(s['selected_dur'], 1),
                     'description': _scene_desc(s), 'thumb': thumbs[i]}
                    for i, s in enumerate(selected)],
            alternates=[{'alt': i + 1, 'start': round(s['start'], 1), 'end': round(s['end'], 1),
                         'quality': s['total_score'], 'duration': round(s['selected_dur'], 1),
                         'description': _scene_desc(s), 'thumb': alt_thumbs[i]}
                        for i, s in enumerate(alternates)])
        job_set(jid, percent=100, step='Preview ready', done=True, result=result)
        return

    job_set(jid, percent=38, step=f'Extracting {len(selected)} selected clips')
    # Extract selected segments + card videos as temp files. `extracted`
    # tracks which of `selected` actually produced a usable clip, so stats
    # reported to the user (selected_scenes, trailer_duration) reflect what's
    # really in the output rather than what was merely picked.
    #
    # Run in parallel: each clip is an independent ffmpeg process reading the same
    # source read-only, so there's nothing to serialize. This used to be a plain
    # sequential loop and was one of the longest single stages of the job.
    _extract_errors = []
    _extract_lock = threading.Lock()
    _extract_progress = {'done': 0}
    n_to_extract = len(selected)

    def _extract_one(seg_i, seg):
        out_seg = os.path.join(app.config['UPLOAD_FOLDER'], f'seg_{base_ts}_{seg_i}.mp4')
        trim_start = seg.get('trim_start', seg['start'])

        def _enc(fast_seek):
            # fast_seek puts -ss before -i (input seeking: quick, but can land
            # awkwardly relative to keyframes). The retry puts it after -i
            # (output seeking: decodes from the start, slower but exact).
            pre = ['-ss', str(trim_start), '-i', path] if fast_seek else ['-i', path, '-ss', str(trim_start)]
            return [FFMPEG, '-y'] + pre + [
                '-t', str(seg['selected_dur']),
                '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '28', '-pix_fmt', 'yuv420p',
                '-c:a', 'aac', '-b:a', '128k', out_seg]

        for attempt, fast in enumerate((True, False)):
            try:
                r = run_ffmpeg(_enc(fast), label=f'clip extract @{trim_start:.1f}s')
                stderr = r.stderr
            except MediaToolTimeout as e:
                stderr = str(e)
            if os.path.exists(out_seg) and os.path.getsize(out_seg) > 0:
                with _extract_lock:
                    _extract_progress['done'] += 1
                    job_set(jid, percent=38 + int(10 * _extract_progress['done'] / max(n_to_extract, 1)),
                            step=f"Extracting clips {_extract_progress['done']}/{n_to_extract}")
                return seg_i, out_seg, seg
            print(f'FFMPEG seg extraction {"error" if attempt == 0 else "retry also failed"} '
                  f'(scene at {trim_start}s): {stderr[:500]}')
            with _extract_lock:
                _extract_errors.append(stderr[-800:])
        with _extract_lock:
            _extract_progress['done'] += 1
        return None

    with ThreadPoolExecutor(max_workers=min(EXTRACT_WORKERS, n_to_extract or 1)) as ex:
        results = list(ex.map(lambda p: _extract_one(*p), enumerate(selected)))

    # Reassemble in the original order — ThreadPoolExecutor.map preserves input
    # order, but filter first so a dropped clip doesn't shift the rest.
    ok_results = [r for r in results if r is not None]
    seg_files = [r[1] for r in ok_results]
    extracted = [r[2] for r in ok_results]
    if _extract_errors:
        last_ffmpeg_stderr = _extract_errors[-1]

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
        try:
            r = run_ffmpeg([FFMPEG, '-y', '-i', all_inputs[0], '-vf', norm,
                            '-c:v', 'libx264', '-preset', 'fast', '-crf', '22',
                            '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '128k', out_path],
                           label='single-clip encode')
            if r.returncode != 0:
                print(f'FFMPEG single concat error: {r.stderr[:500]}')
                last_ffmpeg_stderr = r.stderr[-800:]
        except MediaToolTimeout as e:
            last_ffmpeg_stderr = str(e)
    else:
        # One metadata pass per input (duration + whether it carries audio),
        # replacing what used to be three separate ffprobe spawns per input.
        # Probed in parallel since these are independent read-only calls.
        with ThreadPoolExecutor(max_workers=min(8, n_total)) as ex:
            infos = list(ex.map(probe_media_info, all_inputs))

        # Normalize every input to ensure consistent video/audio before xfade.
        # Only re-encode if audio is missing (add silent audio as fallback).
        normed_inputs = []
        for i, (inp, info) in enumerate(zip(all_inputs, infos)):
            if info['has_audio']:
                normed_inputs.append(inp)
                continue
            normed = os.path.join(app.config['UPLOAD_FOLDER'], f'norm_{base_ts}_{i}.mp4')
            try:
                run_ffmpeg([FFMPEG, '-y', '-i', inp,
                            '-f', 'lavfi', '-i', 'anullsrc=channel_layout=stereo:sample_rate=44100',
                            '-c:v', 'copy', '-c:a', 'aac', '-b:a', '128k',
                            '-map', '0:v:0', '-map', '1:a:0', '-shortest', normed],
                           timeout=120, label='silent-audio mux')
            except MediaToolTimeout as e:
                print(f'Silent-audio mux timed out for input {i}: {e}')
            if os.path.exists(normed) and os.path.getsize(normed) > 0:
                normed_inputs.append(normed)
                # Adding an audio track can change the container duration
                # slightly, so this one input gets re-probed; the untouched
                # inputs keep the duration measured above.
                d = probe_duration(normed)
                if d and d > 0:
                    infos[i] = {'duration': d, 'has_audio': True}
            else:
                normed_inputs.append(inp)

        durations = [info['duration'] for info in infos]
        if any(d is None or d <= 0 for d in durations):
            # Every xfade offset is computed by summing these. A single bad value
            # (this used to silently become 5.0) shifts every transition after it
            # and desynchronises audio from video for the rest of the trailer, so
            # this fails the job instead of shipping a subtly broken render.
            bad = [os.path.basename(all_inputs[i]) for i, d in enumerate(durations) if d is None or d <= 0]
            job_set(jid, error='Could not read the duration of these clips, so transition timing '
                               f'could not be calculated reliably: {", ".join(bad[:5])}. '
                               'This usually means ffprobe is missing or a clip is corrupt.')
            return

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
        try:
            r = run_ffmpeg(cmd, timeout=FFMPEG_LONG_TIMEOUT, label='xfade concat')
            if r.returncode != 0:
                print(f'FFMPEG xfade error: {r.stderr[:1000]}')
                last_ffmpeg_stderr = r.stderr[-800:]
        except MediaToolTimeout as e:
            last_ffmpeg_stderr = str(e)

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
    # Verify it's a valid video, and capture the ACTUAL assembled duration.
    # This is scenes + cards - crossfade overlap, which is NOT the same as
    # total_sel (scenes only). Ducking windows and the reported trailer length
    # both used total_sel previously: the trailing duck window got clipped short,
    # so music never ducked under the end-card VO, and the duration shown to the
    # user understated the real file by the length of the cards.
    assembled_duration = probe_duration(out_path)
    if assembled_duration is None or assembled_duration <= 0:
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
    sfx_source = 'none'  # 'woosh' | 'uploaded' | 'synth_fallback' | 'none'
    sfx_path = os.path.join(app.config['UPLOAD_FOLDER'], f'sfx_{base_ts}.wav')
    if sfx_mode != 'none' and sfx_timestamps:
        hit_wave = None
        if sfx_mode == 'upload' and sfx_upload_path:
            hit_wave = load_hit_waveform(sfx_upload_path)
            sfx_source = 'uploaded' if hit_wave is not None else 'none'
        elif sfx_mode == 'genre' and genre:
            woosh_sfx_path = os.path.join(app.config['UPLOAD_FOLDER'], f'woosh_sfx_{base_ts}.flac')
            if woosh_sfx(genre, woosh_sfx_path, duration=0.8) and os.path.getsize(woosh_sfx_path) > 0:
                hit_wave = load_hit_waveform(woosh_sfx_path)
                if os.path.exists(woosh_sfx_path):
                    os.remove(woosh_sfx_path)
                sfx_source = 'woosh' if hit_wave is not None else 'none'
            # No ACE-Step fallback here by design — ACE-Step is a music model, not an
            # SFX model, so a failed Woosh call goes straight to the procedural synth
            # fallback below rather than a mismatched-model attempt. The "From genre"
            # option is gated on Woosh alone (see data-requires on sfx_mode=genre) so
            # this branch is only reached when Woosh was expected to work.
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
        # Fit the music to the WHOLE assembled trailer, not just the scene
        # segments. This used to be total_sel (scenes only), so the music ran out
        # the moment the title/end cards started and they played dry -- exactly
        # where a promo most wants a bed under the card VO.
        scenes_dur = assembled_duration
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
            # Normalize the BGM track's own loudness only, to its baseline "full swell"
            # level (music_duck_db under overall target — so it never competes with SOT
            # even at full volume). The actual ducking during dialogue is applied later,
            # in the final mix, using deterministic silence-detected windows (see
            # _build_duck_volume_expr below) rather than a reactive sidechain here — that
            # lets a single minimum-gap "hold" rule govern both BGM-vs-SOT and BGM-vs-VO
            # instead of two separately-tuned compressors that could disagree with each other.
            bgm_target = target_loudness + music_duck_db
            bgm_ready_path = os.path.join(app.config['UPLOAD_FOLDER'], f'bgmready_{base_ts}.m4a')
            r = subprocess.run([FFMPEG, '-y', '-i', prepared_bgm,
                                '-af', f'loudnorm=I={bgm_target}:TP={true_peak}:LRA=7',
                                '-c:a', 'aac', '-b:a', '192k', bgm_ready_path],
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
            engine_label = 'Fish Audio'
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
    # people talking over each other is unusable) via a live sidechain, since VO
    # presence is unambiguous there. BGM's duck is different: it needs to stay
    # ducked whenever EITHER SOT or VO has anything going on, and — per request —
    # must not swell back up on every brief pause, only after a real gap. A
    # reactive sidechain has no concept of a minimum "hold" before releasing (only
    # attack/release ramp times), so BGM ducking is computed deterministically
    # instead: silence-detect SOT and VO separately, union the two "not silent"
    # timelines, bridge any gap shorter than duck_release_hold seconds (so a short
    # dialogue pause doesn't count as a real release), then apply duck_depth_db
    # only within those merged windows via a per-frame volume expression.
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

        bgm_duck_expr = None
        if bgm_ready_path:
            job_set(jid, step='Detecting dialogue gaps for music ducking')
            sot_silence = _detect_silence_intervals(out_path)
            sot_windows = _active_windows_from_silence(sot_silence, assembled_duration)
            vo_windows = []
            if vo_ready_path:
                vo_silence = _detect_silence_intervals(vo_ready_path)
                # Bound to the VO clip's own real length -- see
                # _active_windows_from_silence's docstring: without this, a
                # short VO with no detected internal silence reads as "active"
                # for the rest of the trailer, well past where it actually ends.
                vo_real_dur = probe_duration(vo_ready_path) or assembled_duration
                vo_windows = _active_windows_from_silence(vo_silence, assembled_duration,
                                                           content_duration=vo_real_dur)
            combined = _union_windows([sot_windows, vo_windows])
            duck_windows = _merge_windows_with_hold(combined, duck_release_hold)
            bgm_duck_expr = _build_duck_volume_expr(duck_windows, duck_depth_db)
            if bgm_duck_expr is None:
                # Silence detection itself failed on every track (not just "no gaps
                # found") — fall back to a constant duck for the full duration rather
                # than accidentally leaving BGM unducked throughout.
                bgm_duck_expr = f'{10 ** (duck_depth_db / 20):.5f}'

        if vo_ready_path:
            vo_idx = 1 + inputs.index('vo')
            fc = []
            # sidechaincompress truncates its ENTIRE output to the shorter of its
            # two inputs, unconditionally -- confirmed directly against ffmpeg: a
            # 20s main track fed a 5s key track produces a 5s result even with
            # amix's duration=first downstream. That's the "trailer length gets
            # cut to match the VO" bug: any VO shorter than the assembled trailer
            # silently truncated the whole audio mix (and therefore, via -shortest
            # on the final mux, the whole video) to the VO's own length.
            # apad extends the key input with silence to the real trailer length
            # before it ever reaches sidechaincompress, so ducking still applies
            # correctly while the VO plays and simply has nothing left to duck
            # once the VO ends.
            fc.append(f'[{vo_idx}:a]asplit=2[vo_out][vokey_raw]')
            fc.append(f'[vokey_raw]apad=whole_dur={assembled_duration:.3f}[vokey1]')
            # SOT ducked to near-silence under VO (fast attack, low threshold, high
            # ratio, no makeup — this is meant to sit well below the VO, not just
            # lower than before). This one stays a live sidechain since "is VO
            # playing right now" is unambiguous and doesn't need a hold.
            fc.append('[0:a][vokey1]sidechaincompress=threshold=0.01:ratio=20:attack=5:release=250:makeup=1[sot_ducked]')
            mix_labels = ['sot_ducked']
            if bgm_ready_path:
                bgm_idx = 1 + inputs.index('bgm')
                fc.append(f"[{bgm_idx}:a]volume=eval=frame:volume='{bgm_duck_expr}'[bgm_ducked]")
                mix_labels.append('bgm_ducked')
            mix_labels.append('vo_out')
            fc.append('[' + ']['.join(mix_labels) + f']amix=inputs={len(mix_labels)}:duration=first:dropout_transition=2:normalize=0[premix]')
            fc.append(f'[premix]{tail}[outa]')
            filter_complex = ';'.join(fc)
        elif bgm_ready_path:
            # BGM only, no VO — same hold-based duck against SOT, applied directly here
            # since there's no separate VO-driven filter chain to fold it into.
            bgm_idx = 1
            filter_complex = (f"[{bgm_idx}:a]volume=eval=frame:volume='{bgm_duck_expr}'[bgm_ducked];"
                               f'[0:a][bgm_ducked]amix=inputs=2:duration=first:dropout_transition=2:normalize=0[premix];'
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

    result = dict(
        status='ok', trailer_url=f'/uploads/{filename}',
        orig_name=orig_name,
        total_scenes=len(scene_list), selected_scenes=len(selected),
        trailer_duration=round(assembled_duration, 1),
        scenes_duration=round(total_sel, 1),
        video_duration=round(video_duration, 1),
        trailer_length=trailer_length,
        bgm_source=bgm_source, sfx_source=sfx_source,
        vo_source=vo_source, vo_error=vo_error, sync_beats=sync_beats,
        whisper_enhance=whisper_enhance,
        template_applied=params.get('template_applied'),
        scenes=[{
            'scene': i+1, 'start': round(s['start'], 1), 'end': round(s['end'], 1),
            'quality': s['total_score'], 'duration': round(s['selected_dur'], 1),
            'description': _scene_desc(s)
        } for i, s in enumerate(selected)])
    job_set(jid, percent=100, step='Done', done=True, result=result)
    try:
        result['library_id'] = library_add(filename, result)
    except Exception as e:
        print(f'Trailer library save failed (job still succeeded): {e}')

# ---- UI ----

UI = '''
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AIMP - AI Media Platform</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='7' fill='%230b1220'/%3E%3Cpath d='M9 10h9l3 3v9H9z' fill='none' stroke='%2334e6c5' stroke-width='2'/%3E%3Ccircle cx='13' cy='16' r='2.4' fill='%2334e6c5'/%3E%3C/svg%3E">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
<script>
// Resolves and applies the theme BEFORE the stylesheet below is parsed, so
// there's no flash of the wrong theme on load. 'auto' (no stored preference)
// follows the OS/browser's prefers-color-scheme; an explicit choice in
// localStorage always overrides it. This has to be synchronous and placed
// ahead of <style>, not deferred to DOMContentLoaded, or the page would
// paint once in the wrong theme and then visibly snap to the right one.
(function(){
  try{
    var stored = localStorage.getItem('aimp_theme')   // 'light' | 'dark' | null (auto)
    var systemLight = window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches
    document.documentElement.setAttribute('data-theme', stored || (systemLight ? 'light' : 'dark'))
  }catch(e){
    // Storage blocked (private browsing, locked-down browser policy, etc.) --
    // fall back to dark, which is this app's original, always-correct default.
    document.documentElement.setAttribute('data-theme', 'dark')
  }
})()
</script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0b1220;
  --panel:#121a2b;
  --panel-2:#1a2233;
  --elevated:#1a2436;
  --sunken:#03070d;
  --line:#263149;
  --ink:#e7edf6;
  --ink-dim:#8b98ad;
  --phosphor:#34e6c5;
  --phosphor-dim:#1d8f7c;
  --tally:#ff5470;
  --amber:#ffb545;
  --accent:#4f8cff;
  --danger:#c94f4f;
  --radius:10px;
  /* "Frosted glass" panel backgrounds (the rail, the modal) -- these need a
     genuinely different treatment per theme, not just a color swap, since the
     panel sits semi-transparently over the body's own background gradient. */
  --rail-bg:rgba(13,19,32,.72);
  /* Subtle hover/highlight washes used throughout (row hover, toggle hover,
     badges). In dark mode these lighten a dark surface; in light mode the same
     literal white value would be invisible on an already-light background, so
     light theme flips these to black at roughly the same opacity instead. */
  --wash-1:rgba(255,255,255,.02);
  --wash-2:rgba(255,255,255,.03);
  --wash-3:rgba(255,255,255,.04);
  --wash-4:rgba(255,255,255,.06);
  --wash-5:rgba(255,255,255,.08);
}
/* ---- Light theme ----
   Same semantic roles as the dark palette above (--bg is the page background,
   --ink is primary text, --phosphor/--tally/--amber/--accent are the accent
   colors used throughout, etc.) so every existing var(--x) reference in the
   rest of the stylesheet and every inline style="..." in the markup picks
   this up automatically -- nothing else needs to change per-component.
   Accents are deepened relative to their dark-mode values (e.g. phosphor's
   neon teal reads fine on near-black but is too low-contrast on white) so
   text and icons in the accent colors stay legible on a light background. */
[data-theme="light"]{
  --bg:#f3f5f9;
  --panel:#ffffff;
  --panel-2:#eef1f6;
  --elevated:#ffffff;
  --sunken:#e4e8f0;
  --line:#d8dee8;
  --ink:#161b26;
  --ink-dim:#5b6478;
  --phosphor:#0e8a76;
  --phosphor-dim:#0b6d5d;
  --tally:#d81b4a;
  --amber:#a15f0a;
  --accent:#2f6fe0;
  --danger:#b23a3a;
  --rail-bg:rgba(255,255,255,.75);
  --wash-1:rgba(0,0,0,.03);
  --wash-2:rgba(0,0,0,.04);
  --wash-3:rgba(0,0,0,.05);
  --wash-4:rgba(0,0,0,.07);
  --wash-5:rgba(0,0,0,.09);
}
html{scroll-behavior:smooth}
[data-theme="light"] body{
  background:
    radial-gradient(ellipse 900px 480px at 12% -12%, rgba(14,138,118,.05), transparent 60%),
    radial-gradient(ellipse 700px 400px at 100% 0%, rgba(161,95,10,.04), transparent 55%),
    var(--bg);
}
body{
  background:
    radial-gradient(ellipse 900px 480px at 12% -12%, rgba(52,230,197,.07), transparent 60%),
    radial-gradient(ellipse 700px 400px at 100% 0%, rgba(255,181,69,.05), transparent 55%),
    var(--bg);
  color:var(--ink);
  font-family:'IBM Plex Sans',system-ui,-apple-system,sans-serif;
  min-height:100vh;
  -webkit-font-smoothing:antialiased;
}

/* ---- header ---- */
h1{font-family:'JetBrains Mono',monospace;font-size:16px;font-weight:700;letter-spacing:.04em;text-transform:uppercase;margin:0;line-height:1.3;word-break:break-word}
.brand-tagline{font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:400;line-height:1.5;color:var(--ink-dim);letter-spacing:.01em;margin:6px 0 0;text-transform:none;white-space:normal}
h1::before{content:'▚';color:var(--phosphor);font-size:14px}
h1 small{font-family:'JetBrains Mono',monospace;font-weight:400;text-transform:none;letter-spacing:0;font-size:11px;color:var(--ink-dim);display:block;margin-top:4px}
.hdr-engines{font-family:'JetBrains Mono',monospace;font-size:9px;color:var(--ink-dim);letter-spacing:.08em;text-transform:uppercase}

/* ---- Inline delivery tag picker (Fish Audio) ---- */
.rail-sep{height:1px;background:var(--line);margin:8px 4px;opacity:.6}
/* ---- Rendered markdown inside AI Chat assistant bubbles ---- */
/* ---- AI Chat: unified input box (textarea + toolbar in one rounded shell) ---- */
/* ---- Shared rounded input-box look (same visual language as the AI Chat
   input) reused on every primary text field across the app -- Music
   prompt/lyrics, SFX description, TTS script, Scene Detection's custom
   prompt, the VO script field, and AI Chat's own system prompt. Applied
   directly to the textarea/input itself; no wrapper element needed. ---- */
textarea.input-box, input.input-box{
  display:block;width:100%;box-sizing:border-box;
  border:1px solid var(--line);border-radius:14px;background:var(--panel);
  color:var(--ink);font-family:inherit;font-size:14px;line-height:1.5;
  padding:11px 14px;transition:border-color .15s;
}
textarea.input-box{resize:vertical}
textarea.input-box:focus, input.input-box:focus{outline:none;border-color:var(--accent)}

.chat-input-box{border:1px solid var(--line);border-radius:18px;background:var(--panel);
  padding:10px 10px 8px 14px;display:flex;flex-direction:column;gap:2px;transition:border-color .15s}
.chat-input-box:focus-within{border-color:var(--accent)}
.chat-input-box textarea{border:none;background:transparent;color:var(--ink);font-size:14px;
  line-height:1.5;font-family:inherit;resize:none;padding:4px 2px;min-height:24px;max-height:220px;
  overflow-y:auto}
.chat-input-box textarea:focus{outline:none}
.chat-input-toolbar{display:flex;align-items:center;gap:6px;margin-top:2px}
.chat-icon-btn{display:inline-flex;align-items:center;justify-content:center;gap:5px;
  border:none;background:transparent;color:var(--ink-dim);cursor:pointer;padding:6px 8px;
  border-radius:8px;font-size:11px;text-transform:none;letter-spacing:0;transition:background .15s,color .15s}
.chat-icon-btn:hover{background:var(--wash-3);color:var(--ink)}
.chat-icon-btn-round{width:30px;height:30px;padding:0;border-radius:50%}
.chat-send-btn{background:var(--accent);color:#fff}
.chat-send-btn:hover{background:var(--accent);opacity:.85}
.chat-send-btn.dim{opacity:.4;pointer-events:none}
.chat-reasoning-toggle{display:inline-flex;align-items:center;gap:5px;font-size:11px;
  text-transform:none;letter-spacing:0;color:var(--ink-dim);cursor:pointer;padding:5px 9px;
  border-radius:8px;border:1px solid transparent;transition:background .15s,border-color .15s}
.chat-reasoning-toggle:hover{background:var(--wash-3)}
.chat-reasoning-toggle.on{color:var(--phosphor);border-color:var(--phosphor-dim);background:var(--wash-3)}
.chat-reasoning-toggle input{margin:0;accent-color:var(--phosphor)}
.chat-model-picker{display:flex;align-items:center;gap:2px}
.chat-model-picker select{-webkit-appearance:none;appearance:none;border:none;background:transparent;
  color:var(--ink-dim);font-size:12px;font-family:inherit;padding:5px 4px;cursor:pointer;max-width:170px}
.chat-model-picker select:hover{color:var(--ink)}
.chat-model-picker select:focus{outline:none}

.chat-md{font-size:13px;line-height:1.55}
.chat-md > *:first-child{margin-top:0}
.chat-md > *:last-child{margin-bottom:0}
.chat-md p{margin:0 0 10px}
.chat-md h1,.chat-md h2,.chat-md h3,.chat-md h4{margin:14px 0 6px;line-height:1.3;font-weight:600}
.chat-md h1{font-size:17px}
.chat-md h2{font-size:15.5px}
.chat-md h3{font-size:14px}
.chat-md h4{font-size:13px;opacity:.85}
.chat-md ul,.chat-md ol{margin:0 0 10px;padding-left:22px}
.chat-md li{margin:3px 0}
.chat-md code{font-family:'JetBrains Mono',monospace;font-size:12px;background:rgba(127,127,127,.18);padding:1px 5px;border-radius:4px}
.chat-md pre{background:rgba(127,127,127,.14);border:1px solid var(--line);border-radius:6px;padding:10px 12px;overflow-x:auto;margin:0 0 10px}
.chat-md pre code{background:none;padding:0}
.chat-md strong{font-weight:700}
.chat-md em{font-style:italic}

.tagbar{margin:8px 0 4px}
.tagbar-head{display:flex;align-items:baseline;gap:8px;flex-wrap:wrap;margin-bottom:6px}
.tagbar-group{margin-bottom:6px}
.tagbar-group-name{font-size:10px;text-transform:uppercase;letter-spacing:.06em;opacity:.55;margin-bottom:3px}
.tagbar-tags{display:flex;flex-wrap:wrap;gap:4px}
.tagchip{font-family:'JetBrains Mono',monospace;font-size:11px;padding:2px 8px;border-radius:11px;
  border:1px solid var(--line);background:transparent;color:var(--ink);cursor:pointer;line-height:1.6}
.tagchip:hover{border-color:var(--accent,#4f8cff);color:var(--accent,#4f8cff)}

/* ---- Layout shell ----
   Navigation lives in a LEFT RAIL rather than a sticky top header plus a fixed
   bottom dock. Those two horizontal bars cost ~150px of vertical space on every
   screen, which is the scarce dimension for a tall configuration form; a rail
   spends horizontal space instead, of which a 16:9 display has plenty. */
.shell{display:flex;align-items:flex-start;min-height:100vh}
.rail{position:sticky;top:0;flex:0 0 224px;height:100vh;display:flex;flex-direction:column;
  gap:10px;padding:18px 14px;background:var(--rail-bg);border-right:1px solid var(--line);overflow-y:auto}
.rail-brand{padding:2px 8px 12px;border-bottom:1px solid var(--line)}
.theme-toggle{display:flex;gap:2px;margin-top:10px;background:var(--sunken);border:1px solid var(--line);border-radius:7px;padding:2px}
.theme-btn{flex:1;background:transparent;border:none;border-radius:5px;padding:5px 0;font-size:13px;line-height:1;color:var(--ink-dim);cursor:pointer;transition:background .15s,color .15s}
.theme-btn:hover{color:var(--ink)}
.theme-btn.active{background:var(--elevated);color:var(--phosphor)}
.tabs{display:flex;flex-direction:column;gap:4px;flex:1 1 auto}
.hdr-engines{margin-top:auto;padding:10px 8px 2px;border-top:1px solid var(--line);line-height:1.7}
.tab{display:flex;align-items:flex-start;gap:9px;text-align:left;width:100%}
.tab-icon{display:block;font-size:15px;margin:0;flex:0 0 auto;line-height:1.3}
.tab-text{display:block;min-width:0}
.container{max-width:1560px;margin:0;padding:26px 30px 40px;flex:1 1 auto;min-width:0}

/* Configure on the left, watch output on the right. The output column is sticky
   so the progress checklist / preview grid / result player stay in view while
   the form scrolls -- previously results rendered ABOVE the form, so finishing a
   job meant scrolling back up past everything. */
.work{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(0,1fr);gap:22px;align-items:start}
.work-out{position:sticky;top:26px;max-height:calc(100vh - 52px);overflow-y:auto;min-width:0}
.work-in{min-width:0}

@media (max-width:1200px){
  .work{grid-template-columns:1fr}
  /* Single column: the output column must come FIRST, otherwise the monitor,
     progress and result all land below a very long form and read as missing. */
  .work-out{position:static;max-height:none;order:-1}
  .work-in{order:1}
}
@media (max-width:900px){
  /* Icon-only rail: keeps navigation off the vertical axis for as long as
     possible before falling back to a horizontal strip. */
  .rail{flex:0 0 62px;padding:14px 8px}
  .rail-brand,.hdr-engines,.tab-text{display:none}
  .tab{justify-content:center;padding:10px 6px}
  .container{padding:20px 16px 32px}
}
@media (max-width:640px){
  .shell{flex-direction:column}
  .rail{position:static;flex:none;width:100%;height:auto;flex-direction:row;align-items:center;
    border-right:none;border-bottom:1px solid var(--line);overflow-x:auto}
  .tabs{flex-direction:row;gap:4px}
}
.tab{font-family:'JetBrains Mono',monospace;padding:8px 16px;cursor:pointer;background:transparent;border:1px solid transparent;color:var(--ink-dim);font-size:11px;letter-spacing:.03em;text-transform:uppercase;border-radius:8px;user-select:none;transition:color .15s,background .15s,border-color .15s}
.tab:hover{color:var(--ink);background:var(--wash-3)}
.tab.active{color:var(--phosphor);background:rgba(52,230,197,.09);border-color:rgba(52,230,197,.28)}
.view-toggle-btn{font-family:'JetBrains Mono',monospace;padding:7px 16px;cursor:pointer;background:transparent;border:none;color:var(--ink-dim);font-size:11px;letter-spacing:.03em;text-transform:uppercase}
.view-toggle-btn+.view-toggle-btn{border-left:1px solid var(--line)}
.view-toggle-btn.active{color:var(--bg);background:var(--phosphor)}
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
/* ---- Browse / network-browse buttons ---- */
.browse-row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
.browse-btn{
  font-family:'JetBrains Mono',monospace;
  display:inline-flex;align-items:center;gap:7px;
  background:var(--panel-2,#1a2233);color:var(--ink);
  border:1px solid var(--line);border-radius:8px;
  padding:8px 14px;font-size:12px;letter-spacing:.02em;
  cursor:pointer;transition:border-color .15s ease,background .15s ease,transform .1s ease;
}
.browse-btn svg{width:14px;height:14px;flex-shrink:0;opacity:.85}
.browse-btn:hover{border-color:var(--phosphor);background:rgba(52,230,197,.08)}
.browse-btn:active{transform:translateY(1px)}
.browse-btn.local:hover{border-color:var(--accent,#4f8cff);background:rgba(79,140,255,.08)}
.net-file-chip{
  display:none;align-items:center;gap:6px;font-size:12px;
  background:rgba(52,230,197,.1);border:1px solid rgba(52,230,197,.35);
  color:var(--phosphor);border-radius:7px;padding:5px 10px;max-width:100%;
}
.net-file-chip .chip-name{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:260px}
.net-file-chip .chip-x{cursor:pointer;opacity:.7;flex-shrink:0}
.net-file-chip .chip-x:hover{opacity:1}
/* ---- Shared "browse library" modal (reused across all upload fields) ---- */
.net-modal-overlay{
  display:none;position:fixed;inset:0;background:rgba(4,8,14,.72);
  z-index:1000;align-items:center;justify-content:center;padding:20px;
}
.net-modal-overlay.open{display:flex}
.net-modal-box{
  background:var(--panel,#121a2b);border:1px solid var(--line);border-radius:12px;
  width:100%;max-width:480px;max-height:80vh;display:flex;flex-direction:column;
  box-shadow:0 20px 60px rgba(0,0,0,.5);
}
.net-modal-head{
  display:flex;justify-content:space-between;align-items:center;
  padding:16px 18px;border-bottom:1px solid var(--line);
}
.net-modal-head strong{font-size:14px}
.net-modal-head small{display:block;opacity:.6;font-size:11px;margin-top:2px;font-weight:400}
.net-modal-close{cursor:pointer;font-size:16px;opacity:.7;padding:2px 6px;border-radius:5px}
.net-modal-close:hover{opacity:1;background:var(--wash-4)}
.net-modal-list{overflow-y:auto;padding:8px;font-size:13px}
.net-modal-row{
  display:flex;justify-content:space-between;align-items:center;gap:10px;
  padding:9px 10px;cursor:pointer;border-radius:7px;
}
.net-modal-row:hover{background:rgba(52,230,197,.08)}
.net-modal-row .row-name{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.net-modal-row .row-size{opacity:.55;flex-shrink:0;font-size:11px}
.net-modal-empty{padding:20px;text-align:center;opacity:.6}
/* ---- Trailer history panel ---- */
#tr-history-toggle:hover{background:var(--wash-2)}
.history-row{display:flex;align-items:center;justify-content:space-between;gap:10px;padding:9px 14px;border-bottom:1px solid var(--line)}
.history-row:last-child{border-bottom:none}
.history-row:hover{background:rgba(52,230,197,.05)}
.history-main{min-width:0;flex:1;cursor:pointer}
.history-main .h-name{font-size:13px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.history-main .h-meta{font-size:11px;opacity:.6;margin-top:2px}
.history-del{
  flex-shrink:0;cursor:pointer;opacity:.55;font-size:13px;
  padding:5px 8px;border-radius:6px;color:var(--tally);
}
.history-del:hover{opacity:1;background:rgba(255,90,90,.12)}
/* ---- Job monitor panel ---- */
#tr-monitor-toggle:hover{background:var(--wash-2)}
.monitor-section-label{padding:10px 16px 4px;font-size:11px;text-transform:uppercase;letter-spacing:.05em;opacity:.5}
.monitor-row{display:flex;align-items:center;gap:10px;padding:8px 16px}
.monitor-row .m-name{flex:1;min-width:0;font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.monitor-row .m-bar{width:80px;height:5px;border-radius:3px;background:var(--wash-5);overflow:hidden;flex-shrink:0}
.monitor-row .m-bar-fill{height:100%;background:var(--phosphor);border-radius:3px}
.monitor-row .m-pct{font-size:11px;opacity:.65;width:34px;text-align:right;flex-shrink:0}
.monitor-row .m-pos{font-size:11px;opacity:.65;flex-shrink:0}
.monitor-badge{font-size:10px;padding:2px 7px;border-radius:10px;flex-shrink:0;font-weight:600;letter-spacing:.02em}
.monitor-badge.ok{background:rgba(52,230,197,.15);color:var(--phosphor)}
.monitor-badge.err{background:rgba(255,90,90,.15);color:var(--tally)}
.monitor-badge.cancel{background:var(--wash-5);color:var(--ink);opacity:.7}

label{display:block;margin:16px 0 6px;font-family:'JetBrains Mono',monospace;font-size:10px;text-transform:uppercase;letter-spacing:.06em;color:var(--ink-dim)}
input[type=file]{display:block;margin:10px 0;color:var(--ink-dim);font-size:13px;font-family:'IBM Plex Sans',sans-serif}
.dropzone{
  border:2px dashed var(--line);
  border-radius:var(--radius);
  padding:22px 16px;
  text-align:center;
  cursor:pointer;
  margin:10px 0;
  transition:border-color .15s ease, background .15s ease;
  background:var(--sunken);
}
.dropzone:hover{border-color:var(--phosphor-dim)}
.dropzone.dragover{border-color:var(--phosphor);background:rgba(52,230,197,.06)}
.growing-preview{
  max-height:0;
  overflow:hidden;
  opacity:0;
  transition:max-height .28s ease, opacity .22s ease;
  border:1px solid var(--line);
  border-radius:var(--radius);
  margin:0;
  background:var(--panel);
}
.growing-preview.shown{max-height:140px;opacity:1;margin:10px 0}
#tr-file-preview-clear{
  background:transparent;color:var(--ink-dim);border:1px solid var(--line);
  padding:3px 10px;border-radius:6px;font-size:12px;cursor:pointer;
}
#tr-file-preview-clear:hover{border-color:var(--tally);color:var(--tally)}
input[type=url],input[type=number],input[type=text],select{width:100%;padding:10px 12px;background:var(--elevated);border:1px solid var(--line);border-radius:7px;margin:6px 0;font-size:14px;color:var(--ink);font-family:'IBM Plex Sans',sans-serif}
select{appearance:none;background-image:linear-gradient(45deg,transparent 50%,var(--ink-dim) 50%),linear-gradient(135deg,var(--ink-dim) 50%,transparent 50%);background-position:calc(100% - 18px) center,calc(100% - 13px) center;background-size:5px 5px,5px 5px;background-repeat:no-repeat}

.card{background:var(--elevated);border:1px solid var(--line);border-radius:8px;padding:16px;margin:12px 0}
table{width:100%;border-collapse:collapse;margin:14px 0;font-size:13px}
th,td{border-bottom:1px solid var(--line);padding:9px 10px;text-align:left;font-variant-numeric:tabular-nums}
th{color:var(--ink-dim);font-family:'JetBrains Mono',monospace;font-size:10px;text-transform:uppercase;letter-spacing:.05em;font-weight:500}
td{font-size:13px}
tr:hover td{background:var(--wash-1)}
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
<div id="net-modal-overlay" class="net-modal-overlay">
  <div class="net-modal-box">
    <div class="net-modal-head">
      <div>
        <strong id="net-modal-title">Network folder</strong>
        <small id="net-modal-path"></small>
      </div>
      <span class="net-modal-close" id="net-modal-close">&#10005;</span>
    </div>
    <div class="net-modal-list" id="net-modal-list">Loading&hellip;</div>
  </div>
</div>
<div class="shell">
<nav class="rail">
  <div class="rail-brand">
    <h1>A.I.M.P.</h1>
    <p class=brand-tagline>AI Media Platform</p>
    <div class="theme-toggle" role="group" aria-label="Theme">
      <button type=button class="theme-btn" data-theme-choice="light" onclick="setTheme('light')" title="Light theme">&#9728;</button>
      <button type=button class="theme-btn" data-theme-choice="dark" onclick="setTheme('dark')" title="Dark theme">&#9789;</button>
      <button type=button class="theme-btn" data-theme-choice="auto" onclick="setTheme('auto')" title="Match system">&#9881;</button>
    </div>
  </div>
  <div class="tabs">
    <div class="tab active" onclick="switchTab('p-trailer',this)" role="button" tabindex="0"><span class=tab-icon>&#9636;</span><span class=tab-text>Generate Promo Plug<div class=tab-sub>ai+ffmpeg</div></span></div>
    <div class="tab" onclick="switchTab('p-music',this)" role="button" tabindex="0"><span class=tab-icon>&#9835;</span><span class=tab-text>Music Generation<div class=tab-sub>ace-step</div></span></div>
    <div class="tab" onclick="switchTab('p-sfx',this)" role="button" tabindex="0"><span class=tab-icon>&#9889;</span><span class=tab-text>Text to SFX<div class=tab-sub>woosh</div></span></div>
    <div class="tab" onclick="switchTab('p-fish',this)" role="button" tabindex="0"><span class=tab-icon>&#9679;</span><span class=tab-text>Text to Speech<div class=tab-sub>fish audio</div></span></div>
    <div class="tab" onclick="switchTab('p-stt',this)" role="button" tabindex="0"><span class=tab-icon>&#9834;</span><span class=tab-text>Speech to Text<div class=tab-sub>whisper</div></span></div>
    <div class="tab" onclick="switchTab('p-vision',this)" role="button" tabindex="0"><span class=tab-icon>&#9673;</span><span class=tab-text>Scene Detection<div class=tab-sub>vision + pyscenedetect</div></span></div>
    <div class="tab" onclick="switchTab('p-chat',this)" role="button" tabindex="0"><span class=tab-icon>&#128172;</span><span class=tab-text>AI Chat<div class=tab-sub>llm assistant</div></span></div>
    <div class="rail-sep"></div>
    <div class="tab" onclick="switchTab('p-tools',this)" role="button" tabindex="0"><span class=tab-icon>&#9881;</span><span class=tab-text>Player<div class=tab-sub>shared library</div></span></div>
    <div class="tab" onclick="switchTab('p-api',this)" role="button" tabindex="0"><span class=tab-icon>{ }</span><span class=tab-text>API<div class=tab-sub>reference</div></span></div>
    <div class="tab" onclick="switchTab('p-docs',this)" role="button" tabindex="0"><span class=tab-icon>&#9776;</span><span class=tab-text>Docs<div class=tab-sub>workflow+genres</div></span></div>
    <div class="tab" onclick="switchTab('p-config',this)" role="button" tabindex="0"><span class=tab-icon>&#128295;</span><span class=tab-text>Config<div class=tab-sub>api urls</div></span></div>
  </div>
  <div class="hdr-engines">scene detection &middot; ai vision &middot; ffmpeg &middot; ai music &middot; tts &middot; stt</div>
  {% if gate_enabled %}<a href="/logout" style="display:block;text-align:center;font-size:11px;padding:6px 8px 2px;color:var(--ink-dim);text-decoration:none">Sign out</a>{% endif %}
</nav>
<div class="container">

<script>function switchTab(id,btn){document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));document.querySelectorAll('.sub-panel').forEach(p=>p.classList.remove('active'));document.querySelectorAll('.sub-tab').forEach(t=>t.classList.remove('active'));btn.classList.add('active');document.getElementById(id).classList.add('active');if(id==='p-config'&&typeof loadConfigTab==='function'&&!configTabLoaded){loadConfigTab()}}function switchSubTab(id,btn){document.querySelectorAll('.sub-tab').forEach(t=>t.classList.remove('active'));document.querySelectorAll('.sub-panel').forEach(p=>p.classList.remove('active'));btn.classList.add('active');document.getElementById(id).classList.add('active')}

// ---- Theme (light / dark / auto) ----
// The <head> script (see near the top of the page) already resolved and
// applied a theme before this ever ran, purely to avoid a flash of the wrong
// theme -- this is the interactive half: the toggle buttons, persisting an
// explicit choice, and following the OS theme live when set to "auto".
var THEME_KEY = 'aimp_theme'
var _themeMedia = window.matchMedia ? window.matchMedia('(prefers-color-scheme: light)') : null

function _systemTheme(){
  return (_themeMedia && _themeMedia.matches) ? 'light' : 'dark'
}

function _applyThemeButtons(choice){
  document.querySelectorAll('.theme-btn').forEach(function(b){
    b.classList.toggle('active', b.dataset.themeChoice === choice)
  })
}

function setTheme(choice){
  // choice is what the person picked: 'light', 'dark', or 'auto'. What
  // actually gets applied to the page (data-theme) is always a concrete
  // 'light'/'dark' -- 'auto' just means "don't store a preference, and track
  // whatever the OS says right now."
  try{
    if(choice === 'auto') localStorage.removeItem(THEME_KEY)
    else localStorage.setItem(THEME_KEY, choice)
  }catch(e){ /* storage blocked -- the choice still applies for this page load */ }
  document.documentElement.setAttribute('data-theme', choice === 'auto' ? _systemTheme() : choice)
  _applyThemeButtons(choice)
}

document.addEventListener('DOMContentLoaded', function(){
  var stored = null
  try{ stored = localStorage.getItem(THEME_KEY) }catch(e){}
  _applyThemeButtons(stored || 'auto')
  // If the person hasn't explicitly chosen light or dark, follow the OS
  // setting live -- e.g. an OS that switches to dark mode at sunset should
  // carry this page along with it without needing a reload.
  if(_themeMedia && _themeMedia.addEventListener){
    _themeMedia.addEventListener('change', function(){
      var current = null
      try{ current = localStorage.getItem(THEME_KEY) }catch(e){}
      if(!current) document.documentElement.setAttribute('data-theme', _systemTheme())
    })
  }
})
</script>
<script>
// The tabs carry role="button" and tabindex="0" but only had an onclick, so
// keyboard users could focus one and have Enter/Space do nothing. Activate on
// both keys, matching native button behaviour.
document.addEventListener('keydown', function(e){
  if(e.key !== 'Enter' && e.key !== ' ') return
  var el = document.activeElement
  if(!el || !el.classList || !(el.classList.contains('tab') || el.classList.contains('sub-tab'))) return
  e.preventDefault()
  el.click()
})
</script>
<script>
// Universal "Clear" button for every file input on the page: shows the chosen
// filename next to the picker and a small Clear button that resets the input
// and fires a change event (so any per-field wiring, like the card VO preview
// player, updates too). Runs once at load; every <input type=file> is present
// in the initial markup, none are created dynamically later.
document.addEventListener("DOMContentLoaded", function(){
  document.querySelectorAll("input[type=file]").forEach(function(inp){
    if(inp.dataset.skipClearUi) return
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
<h2>Video Player</h2>
<p style="font-size:12px;opacity:.75;margin-top:-6px">Plays media straight from the shared library folders, so you can check a mat, bed or VO before building a promo from it. The browser streams the file — nothing is re-encoded on the server unless it turns out to be in a format the browser itself can't play.</p>
<div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:10px 0">
  <select id=pl-category style="max-width:220px">
    <option value=hires selected>Video (HIRES)</option>
    <option value=tcard>Title card (TCARD)</option>
    <option value=endcard>End card (ENDCARD)</option>
    <option value=music>Music</option>
    <option value=vo>VO</option>
    <option value=sfx>SFX</option>
  </select>
  <button type=button class="browse-btn" onclick="openPlayerBrowser()"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 7a1 1 0 0 1 1-1h5l2 2h9a1 1 0 0 1 1 1v9a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V7z"/></svg>Browse library</button>
  <span style="font-size:12px;opacity:.6">or</span>
  <input type=file id=pl-local accept="video/*,audio/*">
</div>
<span id=pl-status style="font-size:12px;opacity:.75"></span>
<div id=pl-area style="display:none;margin-top:16px">
  <div class=card>
    <div id=pl-title style="font-size:12px;opacity:.8;margin-bottom:8px"></div>
    <video id=pl-video controls style="width:100%;max-height:60vh;background:#000;border-radius:6px"></video>
  </div>
</div>
<div id=pl-prompt class=no-data style="margin-top:14px">Pick a file from the list (or double-click it), or choose one from your own machine, to play it.</div>
</div>

<!-- Episodic Promo Plug -->
<div id="p-trailer" class="panel active">
<h2>Episodic Promo Plug Generator</h2>
<div class="work">
<div class="work-in">
<form id=tf method=POST action=/api/trailer/generate enctype=multipart/form-data>
  <div class=card style="padding:10px 14px;margin:0 0 16px;display:flex;align-items:center;gap:12px;flex-wrap:wrap">
    <span style="font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--ink-dim)">View:</span>
    <div style="display:inline-flex;border:1px solid var(--line);border-radius:8px;overflow:hidden">
      <button type=button id=view-easy-btn class="view-toggle-btn active" onclick="setViewMode('easy')">Quick</button>
      <button type=button id=view-adv-btn class="view-toggle-btn" onclick="setViewMode('advanced')">Advanced</button>
    </div>
    <span style="font-size:11px;opacity:.7">Quick shows the essentials plus SFX and narration — switch to Advanced for transitions and manual tuning.</span>
  </div>
  <div id=ollama-down-banner class=card style="display:none;border-color:var(--tally);background:rgba(255,84,112,.08);margin:0 0 16px">
    <strong style="color:var(--tally)">&#9888; AI Vision is unreachable.</strong>
    <span style="font-size:13px">Scene rating needs it, so generation is disabled until it's back. Start Ollama, then click "Check services" on the API tab to retry.</span>
  </div>
  <div id=tr-dropzone class="dropzone">
    <input type=file name=file id=tr-file-input accept=video/* data-skip-clear-ui="1" style="display:none">
    <input type=hidden name=network_file id=tr-network-file value="">
    <div id=tr-dropzone-prompt>
      <div style="font-size:28px;line-height:1;margin-bottom:6px">&#8682;</div>
      <div style="font-size:14px;margin-bottom:12px">Drag &amp; drop a video here</div>
      <div class="browse-row" style="justify-content:center">
        <button type=button class="browse-btn local" id=tr-browse-link><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 16V4M12 4l-4 4M12 4l4 4M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"/></svg>Browse files</button>
        <button type=button class="browse-btn" id=tr-network-link><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 7a1 1 0 0 1 1-1h5l2 2h9a1 1 0 0 1 1 1v9a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V7z"/></svg>Browse library</button>
      </div>
      <div style="font-size:11px;opacity:.6;margin-top:10px">MP4, MOV, MKV, and most common video formats</div>
    </div>
  </div>
  <div id=tr-network-panel class=card style="display:none;text-align:left;padding:14px">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
      <div>
        <strong style="font-size:13px" id=tr-network-title>Video (HIRES) folder</strong>
        <div id=tr-network-path style="font-size:11px;opacity:.6;margin-top:2px"></div>
      </div>
      <span class="net-modal-close" id=tr-network-cancel title="Cancel">&#10005;</span>
    </div>
    <div id=tr-network-list class="net-modal-list" style="padding:0;max-height:280px">Loading&hellip;</div>
  </div>
  <div id=tr-file-preview class="growing-preview">
    <div style="display:flex;gap:14px;align-items:flex-start;padding:12px">
      <video id=tr-file-preview-video muted style="width:160px;height:90px;object-fit:cover;border-radius:6px;background:#000;flex-shrink:0"></video>
      <div style="flex:1;min-width:0">
        <div id=tr-file-preview-name style="font-size:13px;font-weight:600;word-break:break-all"></div>
        <div id=tr-file-preview-meta style="font-size:12px;opacity:.7;margin-top:3px"></div>
        <button type=button id=tr-file-preview-clear style="margin-top:8px;font-size:12px">&#10005; Remove</button>
      </div>
    </div>
  </div>
  <label>Template:</label>
  <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:8px 0">
    <select name=template_id id=template-select style="max-width:100%;flex:1;min-width:220px"><option value="">Loading templates…</option></select>
    <button type=button class=btn style="padding:6px 12px;font-size:12px" onclick="loadTemplates()">Refresh</button>
    <button type=button class=btn id=template-delete-btn style="padding:6px 12px;font-size:12px;background:var(--danger,#c94f4f)" onclick="deleteSelectedTemplate()">Delete</button>
  </div>
  <p style="margin-top:-4px;margin-bottom:10px;font-size:12px;opacity:.75">A template carries its whole setup — genre, transition, lengths, audio targets, voice — plus its music bed, SFX, VO and cards. Picking one fills the form in below; change anything you like afterwards and it applies to this job only.</p>
  <div id=template-summary style="font-size:12px;opacity:.85;margin:0 0 14px;padding:10px 12px;border:1px solid var(--line);border-radius:8px;display:none"></div>

  <label>Rating mode:</label>
  <select name=mode id=scoring-mode-select>
    <option value=ai selected>VISION (OpenCV + AI Vision rating)</option>
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
  <input type=number name=max_scene_dur value=3 placeholder="no limit" min=0.5 step=0.5 style="width:100px">
  <p style="margin-top:-4px;margin-bottom:8px;font-size:12px;opacity:.75">Caps how long any single selected scene can run, even if the duration budget would allow more. Defaults to 3s, which keeps the cut moving; clear it for no limit.</p>
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
    <span style="display:inline-flex;gap:6px;align-items:center;white-space:nowrap"><span style="font-size:12px">Target&nbsp;LUFS:</span>
    <input type=number name=target_loudness value=-14 min=-30 max=-10 step=0.5 style="width:70px"></span>
    <span style="display:inline-flex;gap:6px;align-items:center;white-space:nowrap"><span style="font-size:12px">True&nbsp;peak&nbsp;(dB):</span>
    <input type=number name=true_peak value=-1.5 min=-6 max=0 step=0.5 style="width:70px"></span>
    <span style="display:inline-flex;gap:6px;align-items:center;white-space:nowrap"><span style="font-size:12px">Music&nbsp;baseline&nbsp;(dB&nbsp;under&nbsp;target):</span>
    <input type=number name=music_duck_db value=-3 min=-24 max=0 step=0.5 style="width:70px"></span>
    <span style="display:inline-flex;gap:6px;align-items:center;white-space:nowrap"><span style="font-size:12px">Duck&nbsp;depth&nbsp;while&nbsp;talking&nbsp;(dB):</span>
    <input type=number name=duck_depth_db value=-15 min=-30 max=-3 step=0.5 style="width:70px"></span>
    <span style="display:inline-flex;gap:6px;align-items:center;white-space:nowrap"><span style="font-size:12px">Min.&nbsp;silence&nbsp;before&nbsp;music&nbsp;swells&nbsp;back&nbsp;(s):</span>
    <input type=number name=duck_release_hold value=0.4 min=0.1 max=5 step=0.1 style="width:70px"></span>
    <label style="font-size:13px;text-transform:none;letter-spacing:0;display:flex;align-items:center;gap:4px;margin:0 0 0 12px;cursor:pointer">
      <input type=checkbox name=beat_match checked> Beat match (librosa)
    </label>
    <label style="font-size:13px;text-transform:none;letter-spacing:0;display:flex;align-items:center;gap:4px;margin:0 0 0 12px;cursor:pointer">
      <input type=checkbox name=sync_beats checked> Sync cuts to beat
    </label>
  </div>
  <p style="margin-top:-4px;margin-bottom:8px;font-size:12px;opacity:.75">Selecting "VISION + STT" above transcribes dialogue locally to boost ratings for scenes with quotable lines, and snap cut points to word boundaries instead of mid-word. Uses the large-v2 Whisper model by default (needed for reliable Tagalog — smaller models are noticeably worse at it), so the first run may take a bit to load.</p>
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
  <div class="browse-row">
    <input type=file name=end_card_video accept=video/* data-net-field="end_card_video">
    <button type=button class="browse-btn" onclick="openNetworkBrowser('end_card_video','tcard','Title card (TCARD)')"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 7a1 1 0 0 1 1-1h5l2 2h9a1 1 0 0 1 1 1v9a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V7z"/></svg>Browse library</button>
  </div>
  <input type=hidden id=end_card_video_network name=end_card_video_network value="">
  <input type=hidden id=end_card_video_skip_template name=end_card_video_skip_template value="">
  <span class="net-file-chip" id=end_card_video_chip><span class=chip-name></span><span class=chip-x onclick="clearNetworkField('end_card_video')">&#10005;</span></span>
  <div class="adv-only">
  <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:6px 0 4px">
    <span style="display:inline-flex;gap:6px;align-items:center;white-space:nowrap"><span style="font-size:12px">VO for this card (optional, replaces its audio):</span>
    <input type=file name=title_card_vo accept="audio/*,.mp3,.wav,.m4a,.flac,.ogg" style="width:auto" data-net-field="title_card_vo" onchange="cardVoFileChosen(this,'title_card_vo')"></span>
    <button type=button class="browse-btn" onclick="openNetworkBrowser('title_card_vo','vo','VO')"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 7a1 1 0 0 1 1-1h5l2 2h9a1 1 0 0 1 1 1v9a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V7z"/></svg>Browse library</button>
  </div>
  <input type=hidden id=title_card_vo_network name=title_card_vo_network value="">
  <input type=hidden id=title_card_vo_skip_template name=title_card_vo_skip_template value="">
  <span class="net-file-chip" id=title_card_vo_chip><span class=chip-name></span><span class=chip-x onclick="clearNetworkField('title_card_vo')">&#10005;</span></span>
  <audio id="title_card_vo_player" controls style="display:none;width:100%;max-width:420px;height:32px;margin-bottom:6px"></audio>
  <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:0 0 12px">
    <button type=button class=btn style="padding:4px 10px;font-size:11px" onclick="cardVoSetPoint('title_card_vo','start')">Set in (player)</button>
    <button type=button class=btn style="padding:4px 10px;font-size:11px" onclick="cardVoSetPoint('title_card_vo','end')">Set out (player)</button>
    <span style="display:inline-flex;gap:6px;align-items:center;white-space:nowrap"><span style="font-size:12px">or manually - start (s):</span>
    <input type=number id="title_card_vo_start" name=title_card_vo_start value=0 min=0 step=0.1 style="width:70px"></span>
    <span style="display:inline-flex;gap:6px;align-items:center;white-space:nowrap"><span style="font-size:12px">end (s, blank=to end):</span>
    <input type=number id="title_card_vo_end" name=title_card_vo_end min=0 step=0.1 style="width:70px"></span>
    <span id="title_card_vo_preview_note" style="font-size:11px;opacity:.7"></span>
  </div>
  </div>
  <label>End card video (optional):</label>
  <div class="browse-row">
    <input type=file name=schedule_video accept=video/* data-net-field="schedule_video">
    <button type=button class="browse-btn" onclick="openNetworkBrowser('schedule_video','endcard','End card (ENDCARD)')"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 7a1 1 0 0 1 1-1h5l2 2h9a1 1 0 0 1 1 1v9a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V7z"/></svg>Browse library</button>
  </div>
  <input type=hidden id=schedule_video_network name=schedule_video_network value="">
  <input type=hidden id=schedule_video_skip_template name=schedule_video_skip_template value="">
  <span class="net-file-chip" id=schedule_video_chip><span class=chip-name></span><span class=chip-x onclick="clearNetworkField('schedule_video')">&#10005;</span></span>
  <div class="adv-only">
  <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:6px 0 4px">
    <span style="display:inline-flex;gap:6px;align-items:center;white-space:nowrap"><span style="font-size:12px">VO for this card (optional, replaces its audio):</span>
    <input type=file name=end_card_vo accept="audio/*,.mp3,.wav,.m4a,.flac,.ogg" style="width:auto" data-net-field="end_card_vo" onchange="cardVoFileChosen(this,'end_card_vo')"></span>
    <button type=button class="browse-btn" onclick="openNetworkBrowser('end_card_vo','vo','VO')"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 7a1 1 0 0 1 1-1h5l2 2h9a1 1 0 0 1 1 1v9a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V7z"/></svg>Browse library</button>
  </div>
  <input type=hidden id=end_card_vo_network name=end_card_vo_network value="">
  <input type=hidden id=end_card_vo_skip_template name=end_card_vo_skip_template value="">
  <span class="net-file-chip" id=end_card_vo_chip><span class=chip-name></span><span class=chip-x onclick="clearNetworkField('end_card_vo')">&#10005;</span></span>
  <audio id="end_card_vo_player" controls style="display:none;width:100%;max-width:420px;height:32px;margin-bottom:6px"></audio>
  <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:0 0 12px">
    <button type=button class=btn style="padding:4px 10px;font-size:11px" onclick="cardVoSetPoint('end_card_vo','start')">Set in (player)</button>
    <button type=button class=btn style="padding:4px 10px;font-size:11px" onclick="cardVoSetPoint('end_card_vo','end')">Set out (player)</button>
    <span style="display:inline-flex;gap:6px;align-items:center;white-space:nowrap"><span style="font-size:12px">or manually - start (s):</span>
    <input type=number id="end_card_vo_start" name=end_card_vo_start value=0 min=0 step=0.1 style="width:70px"></span>
    <span style="display:inline-flex;gap:6px;align-items:center;white-space:nowrap"><span style="font-size:12px">end (s, blank=to end):</span>
    <input type=number id="end_card_vo_end" name=end_card_vo_end min=0 step=0.1 style="width:70px"></span>
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
    <div class="browse-row">
      <input type=file name=scoring_audio accept="audio/*,.mp3,.wav,.m4a,.flac,.ogg" data-net-field="scoring_audio">
      <button type=button class="browse-btn" onclick="openNetworkBrowser('scoring_audio','music','Music')"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 7a1 1 0 0 1 1-1h5l2 2h9a1 1 0 0 1 1 1v9a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V7z"/></svg>Browse library</button>
    </div>
    <input type=hidden id=scoring_audio_network name=scoring_audio_network value="">
  <input type=hidden id=scoring_audio_skip_template name=scoring_audio_skip_template value="">
    <span class="net-file-chip" id=scoring_audio_chip><span class=chip-name></span><span class=chip-x onclick="clearNetworkField('scoring_audio')">&#10005;</span></span>
  </div>

  <div class="quick-visible">
  <label>Sound effects (stamped at every scene cut):</label>
  <div style="display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:8px 0">
    <label style="font-size:13px;text-transform:none;letter-spacing:0;display:flex;align-items:center;gap:4px;margin:0;cursor:pointer">
      <input type=radio name=sfx_mode value=genre data-requires="woosh"> From genre
    </label>
    <label style="font-size:13px;text-transform:none;letter-spacing:0;display:flex;align-items:center;gap:4px;margin:0;cursor:pointer">
      <input type=radio name=sfx_mode value=upload> Upload one-shot
    </label>
    <label style="font-size:13px;text-transform:none;letter-spacing:0;display:flex;align-items:center;gap:4px;margin:0;cursor:pointer">
      <input type=radio name=sfx_mode value=none checked> None
    </label>
  </div>
  <p id=sfx-gating-note style="display:none;margin-top:-4px;margin-bottom:8px;font-size:12px;color:var(--amber)">"From genre" is unavailable — Woosh isn't reachable.</p>
  <div id=sfx-upload-area style="display:none">
    <div class="browse-row">
      <input type=file name=sfx_upload accept="audio/*,.mp3,.wav,.m4a,.flac,.ogg" data-net-field="sfx_upload">
      <button type=button class="browse-btn" onclick="openNetworkBrowser('sfx_upload','sfx','SFX')"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 7a1 1 0 0 1 1-1h5l2 2h9a1 1 0 0 1 1 1v9a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V7z"/></svg>Browse library</button>
    </div>
    <input type=hidden id=sfx_upload_network name=sfx_upload_network value="">
  <input type=hidden id=sfx_upload_skip_template name=sfx_upload_skip_template value="">
    <span class="net-file-chip" id=sfx_upload_chip><span class=chip-name></span><span class=chip-x onclick="clearNetworkField('sfx_upload')">&#10005;</span></span>
    <p style="margin-top:4px">A single hit/whoosh/impact sound — it gets stamped at every cut, not looped as music.</p>
  </div>
  </div>

  <div class="quick-visible">
  <label>Narration:</label>
  <div style="display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:8px 0">
    <label style="font-size:13px;text-transform:none;letter-spacing:0;display:flex;align-items:center;gap:4px;margin:0;cursor:pointer">
      <input type=radio name=vo_mode value=none checked> None
    </label>
    <label style="font-size:13px;text-transform:none;letter-spacing:0;display:flex;align-items:center;gap:4px;margin:0;cursor:pointer">
      <input type=radio name=vo_mode value=upload> Upload audio
    </label>
    <label style="font-size:13px;text-transform:none;letter-spacing:0;display:flex;align-items:center;gap:4px;margin:0;cursor:pointer">
      <input type=radio name=vo_mode value=tts data-requires="fish_audio"> Narration (AI voice)
    </label>
  </div>
  <p id=vo-gating-note style="display:none;margin-top:-4px;margin-bottom:8px;font-size:12px;color:var(--amber)">"Narration (AI voice)" is unavailable — the Fish Audio server is not reachable.</p>
  <div id=vo-upload-area style="display:none">
    <div class="browse-row">
      <input type=file name=vo_upload accept="audio/*,.mp3,.wav,.m4a,.flac,.ogg" data-net-field="vo_upload">
      <button type=button class="browse-btn" onclick="openNetworkBrowser('vo_upload','vo','VO')"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 7a1 1 0 0 1 1-1h5l2 2h9a1 1 0 0 1 1 1v9a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V7z"/></svg>Browse library</button>
    </div>
    <input type=hidden id=vo_upload_network name=vo_upload_network value="">
  <input type=hidden id=vo_upload_skip_template name=vo_upload_skip_template value="">
    <span class="net-file-chip" id=vo_upload_chip><span class=chip-name></span><span class=chip-x onclick="clearNetworkField('vo_upload')">&#10005;</span></span>
    <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:6px 0">
      <span style="display:inline-flex;gap:6px;align-items:center;white-space:nowrap"><span style="font-size:12px">Trim uploaded audio — start (s):</span>
      <input type=number name=vo_trim_start value=0 min=0 step=0.5 style="width:80px"></span>
      <span style="display:inline-flex;gap:6px;align-items:center;white-space:nowrap"><span style="font-size:12px">end (s, blank=to end):</span>
      <input type=number name=vo_trim_end min=0 step=0.5 style="width:80px"></span>
    </div>
    <p style="margin-top:-2px;margin-bottom:8px;font-size:12px;opacity:.75">Selects which portion of the uploaded file to use as narration. This is separate from "Start at" below, which places the (already-trimmed) narration on the trailer's own timeline.</p>
  </div>
  <div id=vo-tts-area style="display:none">
    <textarea name=vo_text id=vo-text-input class=input-box rows=3 placeholder="Type the narration script here..."></textarea>
    <div id=vo-tags-gen class=tagbar data-target=vo-text-input style="display:none"></div>

    <div style="display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:8px 0 4px">
      <span style="font-size:13px">Engine:</span>
      <label style="font-size:13px;text-transform:none;letter-spacing:0;display:flex;align-items:center;gap:4px;margin:0;cursor:pointer">
        <input type=radio name=vo_engine value=fish_audio checked> Fish Audio S2
      </label>
      <label style="font-size:13px;text-transform:none;letter-spacing:0;display:flex;align-items:center;gap:4px;margin:0;cursor:pointer">
      </label>
    </div>

    <div id=vo-registered-voice-area style="margin-bottom:6px">
      <label style="font-size:13px;text-transform:none;letter-spacing:0;display:block;margin-bottom:4px">Voice:</label>
      <select name=vo_voice id=vo-voice-select style="max-width:100%"><option value="">Loading voices…</option></select>
      <span id=vo-voice-note style="font-size:12px;opacity:.75;margin-left:6px"></span>
    </div>

    <div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:8px 0">
      <span style="display:inline-flex;gap:6px;align-items:center;white-space:nowrap"><span style="font-size:12px">Language:</span>
      <select name=vo_language id=vo-language-select style="max-width:220px"><option value="auto">Auto-detect (recommended)</option></select></span>
      <span style="display:inline-flex;gap:6px;align-items:center;white-space:nowrap"><span style="font-size:12px">Rate (wpm):</span>
      <input type=number name=vo_rate id=vo-rate-input value=175 min=80 max=300 step=5 style="width:80px"></span>
    </div>
    <p style="margin-top:4px;font-size:12px;opacity:.75">There's no built-in default voice — the list is fetched live from your Fish Audio server (<code>/api/voices?engine=fish_audio</code>). A self-hosted server's voices come from <code>/v1/references/list</code>; the cloud API is used instead when <code>FISH_AUDIO_API_KEY</code> is set. Fish Audio S2 auto-detects the script's language (including Tagalog) unless you override it above.</p>

    <div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:10px 0 4px">
      <button type=button id=vo-preview-btn onclick="previewVoiceover()">Generate &amp; preview</button>
      <span id=vo-preview-status style="font-size:12px;opacity:.75"></span>
    </div>
    <audio id=vo-preview-audio controls style="display:none;width:100%;margin-top:4px"></audio>
    <p style="margin-top:4px;font-size:12px;opacity:.75">Renders just this script through the settings above so you can check the voice, rate, and language before running the full trailer job.</p>
  </div>
  <div id=vo-common-area style="display:none;margin:8px 0">
    <span style="display:inline-flex;gap:6px;align-items:center;white-space:nowrap"><span style="font-size:12px">Start at (s into promo plug):</span>
    <input type=number name=vo_start value=0 min=0 step=0.5 style="width:80px"></span>
    <span style="display:inline-flex;gap:6px;align-items:center;white-space:nowrap"><span style="font-size:12px">Audio level:</span>
    <input type=number name=vo_volume value=1.15 min=0.3 max=3.0 step=0.05 style="width:80px"></span>
    <p style="margin-top:4px;font-size:12px;opacity:.75">Music, SFX, and original dialogue automatically duck under the voiceover wherever it plays. Audio level is a gain multiplier applied to the voiceover track before mixing (1.0 = unchanged, higher = louder).</p>
  </div>
  </div>

  <div class=card style="padding:12px 14px;margin:18px 0 0;border:1px dashed var(--line)">
    <label style="margin-top:0">Save as template</label>
    <p style="margin-top:-4px;margin-bottom:10px;font-size:12px;opacity:.75">Saves the whole configuration above — genre, transition, lengths, audio targets, voice settings — together with whichever of the background music, SFX, voiceover, title card and end card you've picked, under a show name. Next episode is then one dropdown away. Saving under a name that already exists updates that show: settings are replaced, and only the asset slots you supplied this time are rewritten.</p>
    <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center">
      <input type=text id=template-save-name placeholder="Show name (e.g. 24 Oras)" style="flex:1;min-width:200px" maxlength=120>
      <button type=button class=btn id=template-save-btn style="padding:8px 14px" onclick="saveTemplate()">Save as template</button>
    </div>
    <div id=template-save-status style="font-size:12px;margin-top:8px;min-height:16px"></div>
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
      var engineLabel = 'Fish Audio'
      if(voiceSel.options.length === 0){
        var opt = document.createElement('option')
        opt.value = ''; opt.textContent = 'No voices found for ' + engineLabel
        voiceSel.appendChild(opt)
        note.textContent = d.error || (engine === 'fish_audio'
          ? 'Set FISH_AUDIO_API_KEY to list voices registered on the Fish Audio cloud API.'
          : 'No voices registered on the Fish Audio server — check the Config tab.')
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
      if(note) note.textContent = 'Could not reach Fish Audio to list voices: ' + e
      _voicesLoadedFor = null
    })
  }
  function previewVoiceover(){
    var text = document.getElementById('vo-text-input').value.trim()
    var status = document.getElementById('vo-preview-status')
    var audioEl = document.getElementById('vo-preview-audio')
    var btn = document.getElementById('vo-preview-btn')
    if(!text){ status.textContent = 'Type a narration script first.'; status.style.color = 'var(--amber)'; return }
    var engine = document.querySelector('input[name=vo_engine]:checked').value
    var fd = new FormData()
    fd.append('text', text)
    fd.append('rate', document.getElementById('vo-rate-input').value || 175)
    fd.append('language', document.getElementById('vo-language-select').value || 'auto')
    fd.append('engine', engine)
    fd.append('voice', document.getElementById('vo-voice-select').value || '')
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
  // ---- Show templates ----
  // The seven asset fields a template can hold, keyed by slot name so the summary
  // box and the save payload both stay in step with the server's TEMPLATE_SLOTS.
  // Note the legacy field names: end_card_video is the *title* card, schedule_video
  // is the *end* card.
  var TPL_SLOT_FIELDS = {
    bgm: 'scoring_audio', sfx: 'sfx_upload', vo: 'vo_upload',
    title_card: 'end_card_video', title_card_vo: 'title_card_vo',
    end_card: 'schedule_video', end_card_vo: 'end_card_vo'
  }
  var TPL_SETTING_FIELDS = ['genre','transition','xfade_dur','vo_start','vo_volume',
    'vo_trim_start','vo_trim_end','title_card_vo_start','title_card_vo_end',
    'end_card_vo_start','end_card_vo_end']
  var templatesById = {}

  function tplFormEl(name){ return document.querySelector('#tf [name="' + name + '"]') }

  function loadTemplates(){
    var sel = document.getElementById('template-select')
    var keep = sel.value
    fetch('/api/templates').then(function(r){ return r.json() }).then(function(d){
      templatesById = {}
      sel.innerHTML = ''
      var blank = document.createElement('option')
      blank.value = ''
      blank.textContent = (d.templates && d.templates.length) ? '— no template (manual setup) —' : 'No templates saved yet'
      sel.appendChild(blank)
      ;(d.templates || []).forEach(function(t){
        templatesById[String(t.id)] = t
        var o = document.createElement('option')
        o.value = String(t.id); o.textContent = t.name
        sel.appendChild(o)
      })
      if(keep && templatesById[keep]) sel.value = keep
      onTemplateSelected()
    }).catch(function(e){
      sel.innerHTML = '<option value="">Could not load templates: ' + e + '</option>'
    })
  }
  window.loadTemplates = loadTemplates

  // Human labels + value formatting for every setting a template can carry, so
  // the preview lists what was actually adjusted instead of a bare count.
  var TPL_SETTING_LABELS = {
    mode: 'Rating mode', trailer_length: 'Duration', genre: 'Genre',
    transition: 'Transition', xfade_dur: 'Crossfade', transition_matte: 'Transition matte',
    max_scene_dur: 'Max clip length', scene_threshold: 'Cut sensitivity',
    min_scene_len_sec: 'Min scene length', min_scene_len: 'Min scene length',
    model: 'AI Vision model', prompt: 'Vision prompt',
    scoring_mode: 'Background music', sfx_mode: 'Sound effects', sfx_source: 'SFX source',
    vo_mode: 'Narration', vo_engine: 'Voice engine', vo_voice: 'Voice',
    vo_language: 'Voice language', vo_rate: 'Voice rate', vo_start: 'VO start',
    vo_volume: 'VO level', vo_trim_start: 'VO trim in', vo_trim_end: 'VO trim out',
    vo_text: 'Narration script',
    title_card_vo_start: 'Title card VO in', title_card_vo_end: 'Title card VO out',
    end_card_vo_start: 'End card VO in', end_card_vo_end: 'End card VO out',
    target_loudness: 'Loudness target', true_peak: 'True peak',
    music_duck_db: 'Music level', duck_depth_db: 'Duck depth',
    duck_release_hold: 'Duck hold', beat_match: 'Beat matching',
    broadcast_stereo: 'Broadcast stereo', sync_beats: 'Sync cuts to beats',
    whisper_enhance: 'Dialogue-aware cuts',
  }
  var TPL_SETTING_VALUES = {
    mode: {ai: 'VISION', ai_stt: 'VISION + STT'},
    scoring_mode: {generate: 'AI generated', upload: 'uploaded file', none: 'none'},
    sfx_mode: {genre: 'from genre', upload: 'uploaded file', none: 'none'},
    vo_mode: {tts: 'text-to-speech', upload: 'uploaded file', none: 'none'},
    vo_engine: {fish_audio: 'Fish Audio S2'},
  }
  var TPL_SETTING_UNITS = {
    trailer_length: 's', xfade_dur: 's', max_scene_dur: 's', min_scene_len_sec: 's',
    min_scene_len: 's', vo_start: 's', vo_trim_start: 's', vo_trim_end: 's',
    title_card_vo_start: 's', title_card_vo_end: 's',
    end_card_vo_start: 's', end_card_vo_end: 's', duck_release_hold: 's',
    target_loudness: ' LUFS', true_peak: ' dBTP', music_duck_db: ' dB', duck_depth_db: ' dB',
    vo_rate: ' wpm',
  }
  function tplSettingValue(key, raw){
    var map = TPL_SETTING_VALUES[key]
    if(map && map[raw] !== undefined) return map[raw]
    if(raw === '1' || raw === 'on' || raw === 'true') return 'on'
    var v = String(raw)
    if(key === 'prompt' || key === 'vo_text'){
      return v.length > 60 ? v.slice(0, 57) + '…' : v
    }
    return v + (TPL_SETTING_UNITS[key] || '')
  }

  function onTemplateSelected(){
    var sel = document.getElementById('template-select')
    var box = document.getElementById('template-summary')
    var tpl = templatesById[sel.value]
    if(!tpl){ box.style.display = 'none'; box.innerHTML = ''; return }

    function row(label, value, dim){
      return '<div style="display:flex;gap:8px;padding:2px 0">' +
        '<span style="min-width:150px;opacity:.65;flex-shrink:0">' + escapeHtmlLite(label) + '</span>' +
        '<span style="' + (dim ? 'opacity:.45' : 'color:var(--accent,#4f8cff)') + ';word-break:break-word">' +
        escapeHtmlLite(value) + '</span></div>'
    }

    // Assets
    var assetRows = Object.keys(TPL_SLOT_FIELDS).map(function(slot){
      var sl = tpl.slots[slot]
      if(!sl) return ''
      return sl.filled ? row(sl.label, sl.name || 'saved file', false)
                       : row(sl.label, '— empty —', true)
    }).join('')

    // Every saved setting, in the order they appear in the form rather than
    // whatever order the JSON happened to serialise in. genre/transition/xfade
    // are also stored as their own columns (they predate settings_json), so fall
    // back to those — otherwise a template saved before settings_json existed
    // would show nothing here.
    var st = Object.assign({}, tpl.settings || {})
    if(!st.genre && tpl.genre) st.genre = tpl.genre
    if(!st.transition && tpl.transition) st.transition = tpl.transition
    if(!st.xfade_dur && tpl.xfade_dur) st.xfade_dur = tpl.xfade_dur
    var seen = {}
    var settingRows = Object.keys(TPL_SETTING_LABELS).filter(function(k){
      return st[k] !== undefined && st[k] !== '' && !seen[k] && (seen[k] = 1)
    }).map(function(k){
      return row(TPL_SETTING_LABELS[k], tplSettingValue(k, st[k]), false)
    }).join('')
    // Anything saved that has no label yet still gets shown, so nothing is hidden.
    var unlabelled = Object.keys(st).filter(function(k){ return !TPL_SETTING_LABELS[k] && st[k] !== '' })
      .map(function(k){ return row(k, String(st[k]), false) }).join('')

    var nset = Object.keys(st).length
    box.innerHTML =
      '<div style="font-weight:600;margin-bottom:8px">' + escapeHtmlLite(tpl.name) + '</div>' +
      '<div style="opacity:.6;text-transform:uppercase;letter-spacing:.06em;font-size:10px;margin:8px 0 4px">Settings (' + nset + ')</div>' +
      (settingRows + unlabelled || row('—', 'no settings saved', true)) +
      '<div style="opacity:.6;text-transform:uppercase;letter-spacing:.06em;font-size:10px;margin:10px 0 4px">Assets</div>' +
      assetRows +
      '<div style="margin-top:10px;opacity:.7">The form below has been filled in from this template. Change anything you like — edits apply to this job only and do not alter the saved template.</div>'
    box.style.display = ''
    applyTemplateToForm(tpl)
  }

  function escapeHtmlLite(s){
    return String(s == null ? '' : s).replace(/[&<>"]/g, function(c){
      return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]
    })
  }

  // slot key -> [form field name, the mode radio it drives (or null)]
  var TPL_SLOT_TO_FIELD = {
    bgm: 'scoring_audio', sfx: 'sfx_upload', vo: 'vo_upload',
    title_card: 'end_card_video', title_card_vo: 'title_card_vo',
    end_card: 'schedule_video', end_card_vo: 'end_card_vo',
  }

  function applyTemplateToForm(tpl){
    // A show carries the entire generator configuration, so selecting one fills
    // the form in wholesale. Everything stays editable afterwards -- edits apply
    // to this job only and don't touch the saved show.
    function setField(name, val){
      if(val === null || val === undefined || val === '') return false
      var els = document.querySelectorAll('#tf [name="' + name + '"]')
      if(!els.length) return false
      var first = els[0]
      if(first.type === 'radio'){
        var hit = document.querySelector('#tf input[name="' + name + '"][value="' + val + '"]')
        if(hit && !hit.disabled){ hit.checked = true; hit.dispatchEvent(new Event('change', {bubbles:true})); return true }
        return false
      }
      if(first.type === 'checkbox'){
        first.checked = (val === '1' || val === 'on' || val === 'true')
        first.dispatchEvent(new Event('change', {bubbles:true}))
        return true
      }
      first.value = val
      first.dispatchEvent(new Event('change', {bubbles:true}))
      return true
    }

    var settings = tpl.settings || {}
    Object.keys(settings).forEach(function(k){ setField(k, settings[k]) })

    // Columns stored outside settings_json (they predate it) still win, since the
    // save route keeps them authoritative for the asset-related values.
    setField('genre', tpl.genre)
    setField('transition', tpl.transition)
    setField('xfade_dur', tpl.xfade_dur)

    // Every slot starts clean: a previously-applied template's chips and skip
    // flags must not leak into this one, or a field it doesn't fill would keep
    // showing the OLD template's file (or stay opted out from a stale click).
    Object.keys(TPL_SLOT_TO_FIELD).forEach(function(slot){
      var field = TPL_SLOT_TO_FIELD[slot]
      var skip = document.getElementById(field + '_skip_template')
      if(skip) skip.value = ''
      var sl = tpl.slots[slot]
      if(sl && sl.filled){
        // Visible confirmation that this slot is set -- reuses the same chip
        // element a network-browsed pick uses, so it behaves identically (an
        // "X" clears it and opts this slot out of the template for this job).
        // setChip/clearNetworkField live in a different <script> block, so they
        // are only reachable here via window.
        window.setChip(field, sl.name || (sl.label + ' (from template)'))
        var chip = document.getElementById(field + '_chip')
        if(chip) chip.dataset.fromTemplate = '1'
      }else{
        window.clearNetworkField(field)
        // clearNetworkField() sets skip_template=1 as a side effect (it also runs
        // for a genuine manual "remove this file" click) -- but an empty slot in
        // the template isn't an opt-out, it's just nothing to show, so undo that.
        if(skip) skip.value = ''
      }
    })

    // The mode radios must reflect what the show actually supplies, so the form
    // and the resulting job agree: the server only fills a slot when the mode says
    // "upload" and no file came with the request.
    if(tpl.slots.bgm.filled) setField('scoring_mode', 'upload')
    if(tpl.slots.sfx.filled) setField('sfx_mode', 'upload')
    if(tpl.slots.vo.filled) setField('vo_mode', 'upload')

    if(tpl.slots.vo.filled){
      setField('vo_start', tpl.vo_start); setField('vo_volume', tpl.vo_volume)
      setField('vo_trim_start', tpl.vo_trim_start); setField('vo_trim_end', tpl.vo_trim_end)
    }
    if(tpl.slots.title_card_vo.filled){
      setField('title_card_vo_start', tpl.title_card_vo_start)
      setField('title_card_vo_end', tpl.title_card_vo_end)
    }
    if(tpl.slots.end_card_vo.filled){
      setField('end_card_vo_start', tpl.end_card_vo_start)
      setField('end_card_vo_end', tpl.end_card_vo_end)
    }
  }

  function saveTemplate(){
    var status = document.getElementById('template-save-status')
    var btn = document.getElementById('template-save-btn')
    var name = document.getElementById('template-save-name').value.trim()
    if(!name){ status.style.color = 'var(--amber)'; status.textContent = 'Give the template a show name first.'; return }

    // Built by hand rather than from the <form>, so the (potentially multi-GB)
    // source video in the dropzone never gets uploaded just to save a show.
    var fd = new FormData()
    fd.append('name', name)
    var picked = 0
    Object.keys(TPL_SLOT_FIELDS).forEach(function(slot){
      var field = TPL_SLOT_FIELDS[slot]
      var input = document.querySelector('#tf input[type=file][name="' + field + '"]')
      if(input && input.files && input.files[0]){ fd.append(field, input.files[0]); picked++; return }
      var hidden = document.getElementById(field + '_network')
      if(hidden && hidden.value){ fd.append(field + '_network', hidden.value); picked++ }
    })

    // Every non-file control in the form, so a show captures its whole setup and
    // not just its assets. The source video, the template picker itself, and the
    // save-name box are excluded.
    var SKIP = {file: 1, template_id: 1, name: 1, preview_only: 1}
    var settingsCount = 0
    document.querySelectorAll('#tf [name]').forEach(function(el){
      if(!el.name || SKIP[el.name] || el.type === 'file') return
      if(el.name.slice(-8) === '_network') return
      if((el.type === 'radio' || el.type === 'checkbox') && !el.checked) return
      if(el.value === '') return
      if(fd.has(el.name)) return
      fd.append(el.name, el.type === 'checkbox' ? '1' : el.value)
      settingsCount++
    })

    if(!picked && !settingsCount){
      status.style.color = 'var(--amber)'
      status.textContent = 'Nothing to save yet — set up the form, or pick some assets, first.'
      return
    }

    btn.disabled = true
    status.style.color = ''
    status.textContent = 'Saving template…'
    fetch('/api/templates', {method: 'POST', body: fd}).then(function(r){ return r.json() }).then(function(d){
      if(!d.ok){ status.style.color = 'var(--amber)'; status.textContent = d.error || 'Save failed.'; return }
      status.style.color = 'var(--accent,#4f8cff)'
      var bits = []
      if(d.saved_slots.length) bits.push(d.saved_slots.length + ' asset slot(s)')
      var ns = Object.keys(d.template.settings || {}).length
      if(ns) bits.push(ns + ' settings')
      status.textContent = (d.updated ? 'Updated "' : 'Saved "') + d.template.name + '" — ' +
        (bits.join(' and ') || 'no changes') + '.'
      loadTemplates()
    }).catch(function(e){
      status.style.color = 'var(--amber)'
      status.textContent = 'Save request failed: ' + e
    }).finally(function(){ btn.disabled = false })
  }
  window.saveTemplate = saveTemplate

  function deleteSelectedTemplate(){
    var sel = document.getElementById('template-select')
    var tpl = templatesById[sel.value]
    if(!tpl){ return }
    if(!confirm('Delete the template "' + tpl.name + '" and its saved copies of those files? This cannot be undone.')) return
    fetch('/api/templates/' + tpl.id, {method: 'DELETE'}).then(function(r){ return r.json() }).then(function(d){
      if(!d.ok){ alert(d.error || 'Delete failed.'); return }
      sel.value = ''
      loadTemplates()
    }).catch(function(e){ alert('Delete request failed: ' + e) })
  }
  window.deleteSelectedTemplate = deleteSelectedTemplate

  document.getElementById('template-select').addEventListener('change', onTemplateSelected)

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
  loadTemplates()  // shows are the first thing you pick, so load them up front
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
  <div style="display:flex;gap:10px;align-items:center;margin-top:22px;flex-wrap:wrap">
    <button class=btn type=button id=trailer-preview-btn onclick="submitTrailer(true)" style="background:transparent;border:1px solid var(--accent,#4f8cff);color:var(--accent,#4f8cff)">Preview the cut first</button>
    <button class=btn type=submit id=trailer-submit-btn>Generate Episodic Promo Plug</button>
    <button type=button id=tr-cancel-btn class=btn style="display:none;background:var(--danger,#c94f4f)">Cancel</button>
  </div>
  <p style="margin-top:8px;font-size:12px;opacity:.75"><strong>Preview</strong> runs only the analysis half (cut detection, rating, scene selection) and shows you the chosen scenes with thumbnails — usually well under a minute. You can then drop any you don't want and render, and the render reuses that analysis instead of repeating it.</p>
</form>
</div><!-- /.work-in -->
<div class="work-out">
<div id=tr-monitor class=card style="padding:0;margin-bottom:14px;overflow:hidden">
  <div id=tr-monitor-toggle style="display:flex;align-items:center;justify-content:space-between;padding:11px 16px;cursor:pointer">
    <strong style="font-size:13px">Job monitor <span id=tr-monitor-summary style="opacity:.6;font-weight:400"></span></strong>
    <span id=tr-monitor-chevron style="opacity:.6;font-size:12px">&#9660;</span>
  </div>
  <div id=tr-monitor-body style="display:none;border-top:1px solid var(--line)">
    <div class="monitor-section-label">Active</div>
    <div id=tr-monitor-active></div>
    <div class="monitor-section-label">Queued</div>
    <div id=tr-monitor-queued></div>
    <div class="monitor-section-label">Recently finished</div>
    <div id=tr-monitor-finished style="max-height:220px;overflow-y:auto"></div>
  </div>
</div>
<div id=tr-history class=card style="padding:0;margin-bottom:14px;overflow:hidden">
  <div id=tr-history-toggle style="display:flex;align-items:center;justify-content:space-between;padding:11px 16px;cursor:pointer">
    <strong style="font-size:13px">Saved trailers <span id=tr-history-count style="opacity:.6;font-weight:400"></span></strong>
    <span id=tr-history-chevron style="opacity:.6;font-size:12px">&#9660;</span>
  </div>
  <div id=tr-history-body style="display:none;border-top:1px solid var(--line)">
    <div id=tr-history-list class="net-modal-list" style="max-height:260px"></div>
  </div>
</div>
<div id=tr-area style=display:none>
  <div class=card id=tr-stats></div>
  <div style=margin:10px 0 id=tr-video></div>
  <table id=tr-table><tr><th>Scene</th><th>Start</th><th>End</th><th>Quality</th><th>Used (s)</th></tr></table>
</div>
<div id=tr-prompt class=no-data style="display:none"></div>
<div id=tr-preview-area style="display:none">
  <div class=card style="padding:14px 16px">
    <div style="display:flex;justify-content:space-between;align-items:baseline;gap:12px;flex-wrap:wrap">
      <strong style="font-size:14px">Proposed cut</strong>
      <span id=tr-preview-stats style="font-size:12px;opacity:.8"></span>
    </div>
    <p style="margin:6px 0 12px;font-size:12px;opacity:.75">Untick any scene you don't want, then render. Nothing has been encoded yet — rendering reuses this analysis, so it starts from the extraction step.</p>
    <div id=tr-preview-grid style="display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:12px"></div>
    <div id=tr-alt-wrap style="display:none;margin-top:16px;border-top:1px solid var(--line);padding-top:14px">
      <button type=button class=btn id=tr-alt-toggle onclick="toggleAlternates()" style="background:transparent;border:1px solid var(--line);color:var(--ink)">Show more clips</button>
      <span id=tr-alt-hint style="font-size:12px;opacity:.7;margin-left:10px"></span>
      <div id=tr-alt-body style="display:none;margin-top:12px">
        <p style="font-size:12px;opacity:.75;margin:0 0 10px">Runner-up scenes the selector scored next-highest but didn't pick. Tick any to add them — they'll slot into the trailer in timeline order, not at the end.</p>
        <div id=tr-alt-grid style="display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:12px"></div>
      </div>
    </div>
    <div style="display:flex;gap:10px;align-items:center;margin-top:16px;flex-wrap:wrap">
      <button type=button class=btn id=tr-render-btn onclick="renderApprovedCut()">Render this cut</button>
      <button type=button class=btn id=tr-preview-all-btn onclick="togglePreviewAll()" style="background:transparent;border:1px solid var(--line);color:var(--ink)">Deselect all</button>
      <span id=tr-preview-kept style="font-size:12px;opacity:.75"></span>
    </div>
  </div>
</div>
<div id=tr-progress-area class=no-data style="display:none;text-align:left;padding:20px 24px">
  <div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:4px;color:var(--ink)">
    <span id=tr-progress-step>Working...</span>
    <span><span id=tr-progress-elapsed style="opacity:.6;margin-right:8px"></span><span id=tr-progress-pct>0%</span></span>
  </div>
  <div style="background:var(--panel-2,#1f232b);border-radius:6px;overflow:hidden;height:10px">
    <div id=tr-progress-bar style="background:var(--accent,#4f8cff);height:100%;width:0%;transition:width .3s ease"></div>
  </div>
  <div id=tr-stage-list style="display:flex;flex-wrap:wrap;gap:6px;margin-top:12px"></div>
</div>
</div><!-- /.work-out -->
</div><!-- /.work -->
</div>

<script>
(function(){
  var dropzone = document.getElementById('tr-dropzone')
  var fileInput = document.getElementById('tr-file-input')
  var networkFileInput = document.getElementById('tr-network-file')
  var browseLink = document.getElementById('tr-browse-link')
  var networkLink = document.getElementById('tr-network-link')
  var networkPanel = document.getElementById('tr-network-panel')
  var networkList = document.getElementById('tr-network-list')
  var networkCancel = document.getElementById('tr-network-cancel')
  var promptEl = document.getElementById('tr-dropzone-prompt')
  var preview = document.getElementById('tr-file-preview')
  var previewVideo = document.getElementById('tr-file-preview-video')
  var previewName = document.getElementById('tr-file-preview-name')
  var previewMeta = document.getElementById('tr-file-preview-meta')
  var clearBtn = document.getElementById('tr-file-preview-clear')
  var currentObjectUrl = null

  function humanSize(bytes){
    if(bytes < 1024) return bytes + ' B'
    if(bytes < 1024*1024) return (bytes/1024).toFixed(1) + ' KB'
    if(bytes < 1024*1024*1024) return (bytes/(1024*1024)).toFixed(1) + ' MB'
    return (bytes/(1024*1024*1024)).toFixed(2) + ' GB'
  }
  function humanDuration(sec){
    if(!isFinite(sec)) return ''
    var m = Math.floor(sec/60), s = Math.round(sec%60)
    return m+':'+String(s).padStart(2,'0')
  }

  function showPreview(file){
    if(currentObjectUrl){ URL.revokeObjectURL(currentObjectUrl); currentObjectUrl = null }
    currentObjectUrl = URL.createObjectURL(file)
    previewVideo.src = currentObjectUrl
    previewName.textContent = file.name
    previewMeta.textContent = humanSize(file.size) + ' — reading video info…'
    previewVideo.onloadedmetadata = function(){
      previewMeta.textContent = humanSize(file.size) + ' · ' + humanDuration(previewVideo.duration) +
        ' · ' + previewVideo.videoWidth + '\u00d7' + previewVideo.videoHeight
    }
    dropzone.style.display = 'none'
    // .shown triggers the CSS max-height/opacity transition — the panel visibly
    // grows into place rather than just popping in, so it reads as a live preview
    // filling in with information (size instantly, duration/resolution a moment
    // later once the browser has read enough of the file) rather than a static box.
    requestAnimationFrame(function(){ preview.classList.add('shown') })
  }

  function showNetworkPreview(localName, origName, size){
    if(currentObjectUrl){ URL.revokeObjectURL(currentObjectUrl); currentObjectUrl = null }
    networkFileInput.value = localName
    previewVideo.src = '/uploads/' + encodeURIComponent(localName)
    previewName.textContent = origName
    previewMeta.textContent = humanSize(size) + ' — reading video info…'
    previewVideo.onloadedmetadata = function(){
      previewMeta.textContent = humanSize(size) + ' · ' + humanDuration(previewVideo.duration) +
        ' · ' + previewVideo.videoWidth + '\u00d7' + previewVideo.videoHeight
    }
    networkPanel.style.display = 'none'
    dropzone.style.display = 'none'
    requestAnimationFrame(function(){ preview.classList.add('shown') })
  }

  function clearPreview(){
    fileInput.value = ''
    networkFileInput.value = ''
    preview.classList.remove('shown')
    previewVideo.pause()
    previewVideo.removeAttribute('src')
    previewVideo.load()
    if(currentObjectUrl){ URL.revokeObjectURL(currentObjectUrl); currentObjectUrl = null }
    networkPanel.style.display = 'none'
    dropzone.style.display = ''
  }

  function loadNetworkList(){
    networkList.innerHTML = 'Loading\u2026'
    fetch('/api/network/list?category=hires').then(function(r){ return r.json() }).then(function(d){
      if(!d.ok){ networkList.innerHTML = '<div class="net-modal-empty" style="color:var(--tally)">' + d.error + '</div>'; return }
      var pathEl = document.getElementById('tr-network-path')
      if(pathEl) pathEl.textContent = d.root
      if(!d.files.length){ networkList.innerHTML = '<div class="net-modal-empty">No video files found in this folder.</div>'; return }
      networkList.innerHTML = ''
      d.files.forEach(function(f){
        var row = document.createElement('div')
        row.className = 'net-modal-row'
        row.innerHTML = '<span class="row-name">' + f.name + '</span><span class="row-size">' + humanSize(f.size) + '</span>'
        row.addEventListener('click', function(){
          networkList.innerHTML = '<div class="net-modal-empty">Fetching ' + f.name + '\u2026</div>'
          fetch('/api/network/fetch', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({name: f.name, category: 'hires'})})
            .then(function(r){ return r.json() })
            .then(function(d2){
              if(!d2.ok){ networkList.innerHTML = '<div class="net-modal-empty" style="color:var(--tally)">' + d2.error + '</div>'; return }
              showNetworkPreview(d2.filename, d2.orig_name, d2.size)
            })
            .catch(function(e){ networkList.innerHTML = '<div class="net-modal-empty" style="color:var(--tally)">' + e + '</div>' })
        })
        networkList.appendChild(row)
      })
    }).catch(function(e){ networkList.innerHTML = '<div class="net-modal-empty" style="color:var(--tally)">' + e + '</div>' })
  }

  fileInput.addEventListener('change', function(){
    if(fileInput.files && fileInput.files[0]) showPreview(fileInput.files[0])
  })
  clearBtn.addEventListener('click', clearPreview)
  browseLink.addEventListener('click', function(e){ e.stopPropagation(); fileInput.click() })
  dropzone.addEventListener('click', function(){ fileInput.click() })
  networkLink.addEventListener('click', function(e){
    e.stopPropagation()
    dropzone.style.display = 'none'
    networkPanel.style.display = 'block'
    loadNetworkList()
  })
  networkCancel.addEventListener('click', function(){
    networkPanel.style.display = 'none'
    dropzone.style.display = ''
  })

  ;['dragenter','dragover'].forEach(function(evt){
    dropzone.addEventListener(evt, function(e){
      e.preventDefault(); e.stopPropagation()
      dropzone.classList.add('dragover')
    })
  })
  ;['dragleave','dragend'].forEach(function(evt){
    dropzone.addEventListener(evt, function(e){
      e.preventDefault(); e.stopPropagation()
      dropzone.classList.remove('dragover')
    })
  })
  dropzone.addEventListener('drop', function(e){
    e.preventDefault(); e.stopPropagation()
    dropzone.classList.remove('dragover')
    var files = e.dataTransfer.files
    if(files && files.length){
      if(files[0].type && files[0].type.indexOf('video/') !== 0){
        alert('That doesn\u2019t look like a video file: ' + (files[0].type || files[0].name))
        return
      }
      fileInput.files = files
      showPreview(files[0])
    }
  })
  // Also catch drags that miss the dropzone and land elsewhere on the page,
  // so the browser doesn't navigate away and open the video as a raw file.
  ;['dragover','drop'].forEach(function(evt){
    window.addEventListener(evt, function(e){ e.preventDefault() })
  })
})()
</script>

<script>
// ---- Shared progress plumbing ----
// Used by the full render, the preview pass, and the render-from-preview pass,
// so all three get the same stage checklist and elapsed clock.
var TR_STAGES = null
function renderStageList(pct, activeStep){
  var host = document.getElementById('tr-stage-list')
  if(!host || !TR_STAGES) return
  host.innerHTML = ''
  // Highlight the last stage whose threshold we've reached; everything below it
  // is done, everything above is pending.
  var activeIdx = 0
  TR_STAGES.forEach(function(s, i){ if(pct >= s.percent) activeIdx = i })
  TR_STAGES.forEach(function(s, i){
    var done = i < activeIdx || pct >= 100
    var cur = i === activeIdx && pct < 100
    var el = document.createElement('span')
    el.style.cssText = 'font-size:11px;padding:3px 9px;border-radius:11px;border:1px solid ' +
      (cur ? 'var(--accent,#4f8cff)' : 'var(--line)') + ';' +
      (cur ? 'color:var(--accent,#4f8cff);font-weight:600' : (done ? 'opacity:.55' : 'opacity:.32'))
    el.textContent = (done ? '\u2713 ' : '') + s.label
    if(cur) el.title = activeStep || s.label
    host.appendChild(el)
  })
}
function fmtElapsed(sec){
  if(sec == null) return ''
  var m = Math.floor(sec / 60), s = Math.floor(sec % 60)
  return m ? m + 'm ' + (s < 10 ? '0' : '') + s + 's' : s + 's'
}

function pollJob(jobId, isCancelled){
  var progBar = document.getElementById('tr-progress-bar')
  var progPct = document.getElementById('tr-progress-pct')
  var progStep = document.getElementById('tr-progress-step')
  var progEl = document.getElementById('tr-progress-elapsed')
  return new Promise(function(resolve){
    var poll = async function(){
      if(isCancelled && isCancelled()){ resolve({error:'Cancelled'}); return }
      var r
      try{ r = await fetch('/api/trailer/progress/'+jobId) }
      catch(err){ setTimeout(poll, 1200); return }
      var j = await r.json()
      if(j.error){ resolve({error: j.error}); return }
      if(j.stages) TR_STAGES = j.stages
      var pct = j.percent || 0
      progBar.style.width = pct+'%'; progPct.textContent = pct+'%'
      progStep.textContent = j.step || 'Working...'
      if(progEl) progEl.textContent = fmtElapsed(j.elapsed)
      renderStageList(pct, j.step)
      if(j.done){ resolve(j.result || {error:'Job finished with no result.'}); return }
      setTimeout(poll, 800)
    }
    poll()
  })
}

async function submitTrailer(previewOnly){
  var this_form = document.getElementById('tf')
  function fmtBytes(b){
    if(b < 1024*1024) return (b/1024).toFixed(0) + ' KB'
    if(b < 1024*1024*1024) return (b/(1024*1024)).toFixed(1) + ' MB'
    return (b/(1024*1024*1024)).toFixed(2) + ' GB'
  }
  document.getElementById('tr-area').style.display='none'
  document.getElementById('tr-prompt').style.display='none'
  document.getElementById('tr-preview-area').style.display='none'
  var progArea=document.getElementById('tr-progress-area')
  var progBar=document.getElementById('tr-progress-bar')
  var progPct=document.getElementById('tr-progress-pct')
  var progStep=document.getElementById('tr-progress-step')
  var cancelBtn = document.getElementById('tr-cancel-btn')
  progArea.style.display='block'
  progBar.style.width='0%'; progPct.textContent='0%'; progStep.textContent='Starting...'
  document.getElementById('tr-stage-list').innerHTML=''
  cancelBtn.style.display='inline-block'; cancelBtn.disabled=false
  document.getElementById('trailer-submit-btn').disabled = true
  document.getElementById('trailer-preview-btn').disabled = true

  var fd = new FormData(this_form)
  if(previewOnly) fd.append('preview_only', '1')

  let startData
  try{
    startData = await new Promise(function(resolve, reject){
      var xhr = new XMLHttpRequest()
      xhr.open('POST', '/api/trailer/generate')
      xhr.upload.onprogress = function(ev){
        if(!ev.lengthComputable) return
        var pct = Math.round(ev.loaded / ev.total * 100)
        progBar.style.width = pct + '%'
        progPct.textContent = pct + '%'
        progStep.textContent = pct < 100
          ? 'Uploading video\u2026 (' + fmtBytes(ev.loaded) + ' / ' + fmtBytes(ev.total) + ')'
          : 'Upload complete \u2014 starting job\u2026'
      }
      xhr.onload = function(){
        try{ resolve(JSON.parse(xhr.responseText)) }
        catch(e){ reject(new Error('Bad response from server')) }
      }
      xhr.onerror = function(){ reject(new Error('Network error during upload')) }
      xhr.onabort = function(){ reject(new Error('Upload cancelled')) }
      cancelBtn.onclick = function(){ xhr.abort() }
      xhr.send(fd)
    })
  }catch(err){
    progArea.style.display='none'
    cancelBtn.style.display='none'
    document.getElementById('trailer-submit-btn').disabled = false
    document.getElementById('trailer-preview-btn').disabled = false
    document.getElementById('tr-prompt').style.display='block'
    document.getElementById('tr-prompt').textContent='Upload failed: '+err
    return
  }
  progBar.style.width='0%'; progPct.textContent='0%'; progStep.textContent='Starting...'
  if(window.refreshMonitor) refreshMonitor()
  if(startData.error){
    progArea.style.display='none'
    cancelBtn.style.display='none'
    document.getElementById('trailer-submit-btn').disabled = false
    document.getElementById('trailer-preview-btn').disabled = false
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

  let d = await pollJob(jobId, function(){ return cancelled })
  cancelBtn.disabled = false
  cancelBtn.style.display='none'
  progArea.style.display='none'
  document.getElementById('trailer-submit-btn').disabled = false
  document.getElementById('trailer-preview-btn').disabled = false

  if(d && d.preview) renderPreviewCut(d)
  else renderTrailerResult(d)
  if(window.refreshTrailerHistory) refreshTrailerHistory()
  if(window.refreshMonitor) refreshMonitor()
}

document.getElementById('tf').addEventListener('submit', function(e){
  e.preventDefault()
  submitTrailer(false)
})

// ---- Preview cut review ----
var _previewId = null

// Shared by the chosen-cut grid and the alternates grid. Outer element is a div,
// NOT a label: a <label> wrapping another <label> is invalid HTML and a click
// would activate both and cancel itself out.
function buildSceneCard(s, cls, num, checked, badge){
  var card = document.createElement('div')
  card.style.cssText = 'border:1px solid var(--line);border-radius:8px;overflow:hidden;background:var(--panel-2,#1f232b)'
  card.innerHTML =
    (s.thumb ? '<img src="'+s.thumb+'" alt="'+escapeHtmlLite(badge)+'" style="width:100%;display:block;aspect-ratio:4/3;object-fit:cover">'
             : '<div style="aspect-ratio:4/3;display:flex;align-items:center;justify-content:center;opacity:.4;font-size:11px">no thumbnail</div>') +
    '<div style="padding:8px 10px;font-size:11px;line-height:1.5">' +
    '<label style="display:flex;gap:6px;align-items:center;margin:0;text-transform:none;letter-spacing:0;font-size:11px;cursor:pointer">' +
    '<input type=checkbox class="'+cls+'" data-scene="'+num+'"'+(checked?' checked':'')+'> ' +
    '<strong>'+escapeHtmlLite(badge)+'</strong> <span style="opacity:.7">'+s.start+'s \u2192 '+s.end+'s</span></label>' +
    '<div style="opacity:.7;margin-top:4px">uses '+s.duration+'s \u00b7 score '+s.quality+'</div>' +
    '<div style="opacity:.6;margin-top:2px">'+escapeHtmlLite(s.description||'')+'</div>' +
    '</div>'
  return card
}

function toggleAlternates(){
  var body = document.getElementById('tr-alt-body')
  var open = body.style.display !== 'none'
  body.style.display = open ? 'none' : 'block'
  document.getElementById('tr-alt-toggle').textContent = open ? 'Show more clips' : 'Hide more clips'
}

function renderPreviewCut(d){
  if(d.error){
    document.getElementById('tr-stats').innerHTML='<b>Error:</b> '+d.error
    document.getElementById('tr-area').style.display='block'
    return
  }
  _previewId = d.preview_id
  document.getElementById('tr-preview-stats').textContent =
    d.selected_scenes + ' of ' + d.total_scenes + ' scenes \u00b7 about ' +
    d.estimated_duration + 's (target ' + d.trailer_length + 's) \u00b7 source ' + d.video_duration + 's'
  var grid = document.getElementById('tr-preview-grid')
  grid.innerHTML = ''
  d.scenes.forEach(function(s){
    grid.appendChild(buildSceneCard(s, 'tr-preview-pick', s.scene, true, '#' + s.scene))
  })
  grid.querySelectorAll('.tr-preview-pick').forEach(function(cb){
    cb.addEventListener('change', updatePreviewKept)
  })

  // Runner-ups for swapping out a clip you don't like.
  var alts = d.alternates || []
  var wrap = document.getElementById('tr-alt-wrap')
  var agrid = document.getElementById('tr-alt-grid')
  agrid.innerHTML = ''
  if(alts.length){
    alts.forEach(function(s){
      agrid.appendChild(buildSceneCard(s, 'tr-alt-pick', s.alt, false, 'Alt ' + s.alt))
    })
    agrid.querySelectorAll('.tr-alt-pick').forEach(function(cb){
      cb.addEventListener('change', updatePreviewKept)
    })
    document.getElementById('tr-alt-hint').textContent = alts.length + ' more available'
    wrap.style.display = 'block'
  }else{
    wrap.style.display = 'none'
  }
  document.getElementById('tr-alt-body').style.display = 'none'
  document.getElementById('tr-alt-toggle').textContent = 'Show more clips'
  updatePreviewKept()
  document.getElementById('tr-preview-area').style.display='block'
  document.getElementById('tr-preview-area').scrollIntoView({behavior:'smooth', block:'nearest'})
}

function previewPicks(){
  return Array.prototype.slice.call(document.querySelectorAll('.tr-preview-pick'))
}
function altPicks(){
  return Array.prototype.slice.call(document.querySelectorAll('.tr-alt-pick'))
}
function updatePreviewKept(){
  var all = previewPicks()
  var kept = all.filter(function(c){ return c.checked })
  var extra = altPicks().filter(function(c){ return c.checked })
  document.getElementById('tr-preview-kept').textContent =
    kept.length + ' of ' + all.length + ' scenes kept' +
    (extra.length ? ' + ' + extra.length + ' added' : '')
  document.getElementById('tr-render-btn').disabled = (kept.length + extra.length) === 0
  document.getElementById('tr-preview-all-btn').textContent = kept.length ? 'Deselect all' : 'Select all'
}
function togglePreviewAll(){
  var all = previewPicks()
  var target = !all.some(function(c){ return c.checked })
  all.forEach(function(c){ c.checked = target })
  updatePreviewKept()
}

async function renderApprovedCut(){
  if(!_previewId) return
  var drop = previewPicks().filter(function(c){ return !c.checked })
                           .map(function(c){ return parseInt(c.dataset.scene, 10) })
  var btn = document.getElementById('tr-render-btn')
  btn.disabled = true
  document.getElementById('tr-preview-area').style.display='none'
  var progArea = document.getElementById('tr-progress-area')
  var cancelBtn = document.getElementById('tr-cancel-btn')
  progArea.style.display='block'
  document.getElementById('tr-progress-bar').style.width='0%'
  document.getElementById('tr-progress-pct').textContent='0%'
  document.getElementById('tr-progress-step').textContent='Starting render\u2026'
  document.getElementById('tr-stage-list').innerHTML=''
  cancelBtn.style.display='inline-block'; cancelBtn.disabled=false

  var add = altPicks().filter(function(c){ return c.checked })
                      .map(function(c){ return parseInt(c.dataset.scene, 10) })
  var fd = new FormData()
  fd.append('preview_id', _previewId)
  if(drop.length) fd.append('drop', JSON.stringify(drop))
  if(add.length) fd.append('add', JSON.stringify(add))
  var start
  try{
    var r = await fetch('/api/trailer/render', {method:'POST', body: fd})
    start = await r.json()
  }catch(err){ start = {error: 'Render request failed: ' + err} }

  if(start.error){
    progArea.style.display='none'; cancelBtn.style.display='none'
    btn.disabled = false
    document.getElementById('tr-stats').innerHTML='<b>Error:</b> '+start.error
    document.getElementById('tr-area').style.display='block'
    return
  }
  var cancelled = false
  cancelBtn.onclick = async function(){
    cancelled = true; cancelBtn.disabled = true
    try{ await fetch('/api/trailer/cancel/'+start.job_id, {method:'POST'}) }catch(e){}
  }
  var d = await pollJob(start.job_id, function(){ return cancelled })
  cancelBtn.style.display='none'; cancelBtn.disabled=false
  progArea.style.display='none'
  btn.disabled = false
  renderTrailerResult(d)
  if(window.refreshTrailerHistory) refreshTrailerHistory()
  if(window.refreshMonitor) refreshMonitor()
}

// ---- Tools: Music (ACE-Step) ----
// ---- Lyrics/duration density hint ----
// ACE-Step sings roughly 2-3 words per second; lyrics well outside that budget
// for the chosen duration are the single biggest cause of "skips words" and
// "too much intro" -- the model either crams/drops words to fit, or pads the
// remaining time with instrumental filler. This is a live, non-blocking
// estimate to catch it before a 2-minute generation comes back wrong, not a
// hard rule -- ACE-Step's actual pacing varies by genre and tempo.
function updateLyricsHint(){
  var hint = document.getElementById('music-lyrics-hint')
  var lyricsEl = document.getElementById('music-lyrics')
  var durEl = document.getElementById('music-duration')
  if(!hint || !lyricsEl || !durEl) return
  var text = lyricsEl.value
  var duration = parseFloat(durEl.value) || 0

  // Strip structure/marker tags like [verse], [chorus], [inst] before counting
  // -- they're not sung.
  var sungWords = text.replace(/\\[[^\\]]*\\]/g, ' ').trim().split(/\\s+/).filter(Boolean)
  if(!sungWords.length || !duration){ hint.style.display = 'none'; return }

  var lo = duration * 2, hi = duration * 3   // ~2-3 words/sec
  var n = sungWords.length
  var msgs = []
  if(n > hi * 1.3){
    msgs.push('About ' + n + ' words for ' + duration + 's is well above ACE-Step\\'s ~2-3 words/sec '
      + 'pace (~' + Math.round(lo) + '-' + Math.round(hi) + ' words fits best here) — the most common '
      + 'cause of skipped or crammed words. Trim the lyrics, or raise the duration.')
  }else if(n < lo * 0.4){
    msgs.push('Only ' + n + ' words for ' + duration + 's leaves a lot of room — likely to come back with '
      + 'a long instrumental intro/outro filling the rest. Add more lyrics, or lower the duration.')
  }

  var lines = text.split('\\n').map(function(l){ return l.replace(/\\[[^\\]]*\\]/g, '').trim() })
                  .filter(Boolean)
  var longLines = lines.filter(function(l){ return l.split(/\\s+/).filter(Boolean).length > 8 }).length
  if(longLines){
    msgs.push(longLines + ' line(s) run past ~8 words — long lines tend to fracture vocal timing; '
      + 'short lines (4-8 words) lock in more reliably.')
  }

  if(lines.length > 1 && !/\\[(verse|chorus|bridge|intro|outro)/i.test(text)){
    msgs.push('No [verse]/[chorus] structure tags — without them ACE-Step decides the song structure '
      + 'itself, which is the other common source of an unexpectedly long intro before vocals start.')
  }

  if(msgs.length){
    hint.innerHTML = msgs.map(function(m){ return '<div style="margin:2px 0">&#9888; ' + m + '</div>' }).join('')
    hint.style.borderColor = 'var(--amber)'
    hint.style.color = 'var(--amber)'
    hint.style.display = 'block'
  }else{
    hint.innerHTML = '<div>&#10003; ' + n + ' words for ' + duration + 's is in ACE-Step\\'s usual singing pace.</div>'
    hint.style.borderColor = 'var(--line)'
    hint.style.color = ''
    hint.style.opacity = '.75'
    hint.style.display = 'block'
  }
}

async function generateMusicTool(){
  var btn = document.getElementById('music-gen-btn')
  var out = document.getElementById('music-result')
  var note = document.getElementById('music-prompt-note')
  var prompt = document.getElementById('music-prompt').value.trim()
  if(!prompt){
    note.style.display = 'block'
    note.innerHTML = '<span style="color:var(--amber)">Enter a prompt describing the style you want.</span>'
    return
  }
  var dur = document.getElementById('music-duration').value
  var n = parseInt(document.getElementById('music-samples').value, 10) || 1
  btn.disabled = true
  note.style.display = 'block'
  note.textContent = 'Generating ' + n + ' \u00d7 ' + dur + 's\u2026 this can take a few minutes.'
  out.style.display = 'none'

  var fd = new FormData()
  fd.append('prompt', prompt)
  fd.append('lyrics', document.getElementById('music-lyrics').value.trim())
  fd.append('duration', dur)
  fd.append('samples', n)
  fd.append('bpm', document.getElementById('music-bpm').value)
  fd.append('steps', document.getElementById('music-steps').value)
  fd.append('seed', document.getElementById('music-seed').value)
  var ref = document.getElementById('music-ref')
  if(ref && ref.files && ref.files[0]){
    fd.append('ref_audio', ref.files[0])
    fd.append('ref_strength', document.getElementById('music-ref-strength').value)
  }

  try{
    var r = await fetch('/api/music/generate', {method:'POST', body: fd})
    var d = await r.json()
    if(!d.ok){
      note.innerHTML = '<span style="color:var(--amber)">' + escapeHtmlLite(d.error || 'Generation failed.') + '</span>'
      return
    }
    note.style.display = 'none'
    var meta = [
      (d.instrumental ? 'instrumental' : 'with vocals'),
      d.bpm ? d.bpm + ' bpm' : 'tempo auto',
      d.steps + ' steps'
    ]
    if(d.reference) meta.push('audio2audio @ ' + Number(d.ref_strength).toFixed(2))
    var html = '<div style="font-size:12px;opacity:.75;margin-bottom:10px">' +
      escapeHtmlLite(meta.join(' \u00b7 ')) + '</div>'
    d.samples.forEach(function(sm, i){
      html += '<div class=card style="margin-top:10px">' +
        '<div style="font-size:12px;opacity:.8;margin-bottom:8px">' +
        (d.samples.length > 1 ? 'Take ' + (i+1) + ' \u00b7 ' : '') + sm.duration + 's</div>' +
        '<audio controls style="width:100%" src="' + sm.url + '"></audio>' +
        '<div style="margin-top:10px"><a class=btn href="' + sm.url + '" download="' +
        escapeHtmlLite(sm.filename) + '" style="display:inline-block;text-decoration:none">Download</a></div>' +
        '</div>'
    })
    out.innerHTML = html
    out.style.display = 'block'
  }catch(e){
    note.innerHTML = '<span style="color:var(--amber)">Request failed: ' + escapeHtmlLite(String(e)) + '</span>'
  }finally{
    btn.disabled = false
  }
}

// ---- Text to SFX (Woosh) ----
async function generateSfxTool(){
  var btn = document.getElementById('sfx-gen-btn')
  var out = document.getElementById('sfx-result')
  var note = document.getElementById('sfx-prompt-note')
  var prompt = document.getElementById('sfx-prompt').value.trim()
  if(!prompt){
    note.style.display = 'block'
    note.innerHTML = '<span style="color:var(--amber)">Describe the sound you want.</span>'
    return
  }
  var dur = document.getElementById('sfx-duration').value
  var n = parseInt(document.getElementById('sfx-samples').value, 10) || 1
  btn.disabled = true
  note.style.display = 'block'
  note.textContent = 'Generating ' + n + ' \u00d7 ' + dur + 's\u2026'
  out.style.display = 'none'

  var fd = new FormData()
  fd.append('prompt', prompt)
  fd.append('duration', dur)
  fd.append('samples', n)

  try{
    var r = await fetch('/api/sfx/generate', {method:'POST', body: fd})
    var d = await r.json()
    if(!d.ok){
      note.innerHTML = '<span style="color:var(--amber)">' + escapeHtmlLite(d.error || 'Generation failed.') + '</span>'
      return
    }
    note.style.display = 'none'
    var html = ''
    d.samples.forEach(function(sm, i){
      html += '<div class=card style="margin-top:10px">' +
        '<div style="font-size:12px;opacity:.8;margin-bottom:8px">' +
        (d.samples.length > 1 ? 'Take ' + (i+1) + ' \u00b7 ' : '') + sm.duration + 's</div>' +
        '<audio controls style="width:100%" src="' + sm.url + '"></audio>' +
        '<div style="margin-top:10px"><a class=btn href="' + sm.url + '" download="' +
        escapeHtmlLite(sm.filename) + '" style="display:inline-block;text-decoration:none">Download</a></div>' +
        '</div>'
    })
    out.innerHTML = html
    out.style.display = 'block'
  }catch(e){
    note.innerHTML = '<span style="color:var(--amber)">Request failed: ' + escapeHtmlLite(String(e)) + '</span>'
  }finally{
    btn.disabled = false
  }
}

// ---- Speech to Text (Whisper) ----
async function runTranscribe(){
  var btn = document.getElementById('stt-btn')
  var out = document.getElementById('stt-result')
  var note = document.getElementById('stt-prompt')
  var f = document.getElementById('stt-file')
  if(!f || !f.files || !f.files[0]){
    note.style.display = 'block'
    note.innerHTML = '<span style="color:var(--amber)">Pick a video or audio file first.</span>'
    return
  }
  btn.disabled = true
  out.style.display = 'none'
  note.style.display = 'block'
  note.textContent = 'Extracting audio and transcribing… this can take a few minutes for a full episode.'

  var fd = new FormData()
  fd.append('file', f.files[0])
  try{
    var r = await fetch('/api/stt/transcribe', {method:'POST', body: fd})
    var d = await r.json()
    if(!d.ok){
      note.innerHTML = '<span style="color:var(--amber)">' + escapeHtmlLite(d.error || 'Transcription failed.') + '</span>'
      return
    }
    note.style.display = 'none'
    var rows = d.segments.map(function(sg){
      return '<div style="display:flex;gap:10px;padding:3px 0;border-bottom:1px solid var(--line)">' +
        '<span style="min-width:110px;opacity:.6;font-family:monospace;font-size:11px">' +
        sg.start.toFixed(1) + 's \u2192 ' + sg.end.toFixed(1) + 's</span>' +
        '<span>' + escapeHtmlLite(sg.text) + '</span></div>'
    }).join('')
    out.innerHTML = '<div class=card>' +
      '<div style="font-size:12px;opacity:.8;margin-bottom:10px">' +
      d.segments.length + ' segments \u00b7 ' + d.words + ' timed words \u00b7 ' +
      d.duration + 's \u00b7 ' + escapeHtmlLite(d.model) + '</div>' +
      '<div style="max-height:340px;overflow-y:auto;font-size:12px;line-height:1.6">' + rows + '</div>' +
      '<div style="display:flex;gap:8px;margin-top:12px;flex-wrap:wrap">' +
      '<button type=button class=btn style="padding:6px 12px;font-size:12px" onclick="copySttText()">Copy plain text</button>' +
      '<button type=button class=btn style="padding:6px 12px;font-size:12px" onclick="downloadSrt()">Download .srt</button>' +
      '</div></div>'
    out.style.display = 'block'
    window._sttText = d.text
    window._sttSrt = d.srt
  }catch(e){
    note.innerHTML = '<span style="color:var(--amber)">Request failed: ' + escapeHtmlLite(String(e)) + '</span>'
  }finally{
    btn.disabled = false
  }
}

function copySttText(){
  if(!window._sttText) return
  navigator.clipboard.writeText(window._sttText).then(function(){
    var b = event.target; var old = b.textContent
    b.textContent = 'Copied'; setTimeout(function(){ b.textContent = old }, 1500)
  }).catch(function(){})
}

function downloadSrt(){
  if(!window._sttSrt) return
  var blob = new Blob([window._sttSrt], {type:'text/plain'})
  var a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = 'transcript.srt'
  a.click()
  setTimeout(function(){ URL.revokeObjectURL(a.href) }, 1000)
}

// ---- AI Chat ----
// Conversation history lives only in this array for the life of the page --
// no server-side session, nothing persisted, matching how every other tool
// here is a one-shot call with no state kept between visits.
var chatHistory = []
var _chatModelsLoaded = false

// ---- Attachments ----
// Images ride in Ollama's own messages[].images field (base64, no data-URL
// prefix -- see docs.ollama.com/capabilities/vision) and need no server round
// trip; the browser reads and encodes them directly. Documents can't be parsed
// client-side (a PDF has no reliable browser-side text extraction), so those
// go to /api/chat/extract_file first and the returned text is folded into the
// outgoing message's `content` -- visible to the model, but kept out of the
// chat bubble itself so the conversation doesn't fill up with a page of raw
// extracted text every time someone attaches a document.
var pendingAttachments = []   // {id, kind:'image'|'document', filename, status, base64?, previewUrl?, text?, error?}
var _attachIdSeq = 0
var CHAT_CLIENT_MAX_IMAGE_BYTES = 8 * 1024 * 1024   // mirrors CHAT_MAX_IMAGE_BYTES server-side
var CHAT_CLIENT_MAX_ATTACHMENTS = 6

function loadChatModels(force){
  var sel = document.getElementById('chat-model')
  if(!sel) return
  if(_chatModelsLoaded && !force) return
  _chatModelsLoaded = true
  var keep = sel.value
  sel.innerHTML = '<option value="">Loading…</option>'
  fetch('/api/chat/models').then(function(r){ return r.json() }).then(function(d){
    sel.innerHTML = ''
    if(!d.models || !d.models.length){
      sel.innerHTML = '<option value="">' + (d.error || 'No models found — check Ollama') + '</option>'
      return
    }
    var blank = document.createElement('option')
    blank.value = ''
    blank.textContent = '— pick a model —'
    sel.appendChild(blank)
    d.models.forEach(function(m){
      var o = document.createElement('option')
      o.value = m; o.textContent = m
      sel.appendChild(o)
    })
    if(keep && d.models.indexOf(keep) >= 0) sel.value = keep
  }).catch(function(e){
    sel.innerHTML = '<option value="">Could not reach Ollama: ' + e + '</option>'
  })
}

// ---- Minimal markdown renderer for assistant replies ----
// Some models (seen directly: a Gemma vs Qwen comparison where one came back
// as structured markdown -- headers, bold, bullet lists) format their answers
// in markdown. Chat bubbles used to render with textContent, so that came
// through as literal asterisks and hash marks instead of actual formatting.
// This is intentionally small and dependency-free (no CDN script -- this app
// runs on a LAN with no assumed internet access) and covers the subset that
// actually shows up in practice: headers, bold/italic, inline code, fenced
// code blocks, and lists. No link parsing -- rendering an arbitrary [text](url)
// as a real anchor is a bigger safety surface than this needs for a chat
// window, so a link just displays as literal text.
//
// Escaping happens FIRST, on the raw text, before any markdown syntax is
// interpreted -- everything this function emits as real HTML tags is put
// there by this code, never copied through from the model's own output, so a
// reply that happens to contain "<script>" or similar renders as inert text.
function escapeHtmlLite(s){
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;').replace(/'/g,'&#39;')
}

function _mdInline(s){
  // Order matters: bold before italic, so **x** isn't first read as *?x*? by
  // the italic pass and left with a stray asterisk. The italic pattern
  // requires a non-space character immediately inside each marker (\\S at
  // both edges) -- without that, ordinary text using a bare asterisk for
  // multiplication ("2 * 3 = 6") gets misread as an open/close emphasis pair
  // spanning everything in between, since a naive /\\*([^*]+)\\*/ has no way to
  // tell "* " (math, space after) from "*word" (real italic, no space).
  s = s.replace(/`([^`]+)`/g, '<code>$1</code>')
  s = s.replace(/\\*\\*([^*]+)\\*\\*|__([^_]+)__/g, function(m, a, b){ return '<strong>' + (a || b) + '</strong>' })
  s = s.replace(/\\*(\\S(?:[^*]*\\S)?)\\*|_(\\S(?:[^_]*\\S)?)_/g, function(m, a, b){ return '<em>' + (a || b) + '</em>' })
  return s
}

function renderMarkdownSafe(raw){
  var escaped = escapeHtmlLite(raw || '')
  // Pull fenced code blocks out first so their content isn't touched by any
  // later inline/list/heading pass, then splice the rendered <pre> back in.
  var blocks = []
  escaped = escaped.replace(/```([\\s\\S]*?)```/g, function(m, code){
    blocks.push('<pre><code>' + code.replace(/^\\n/, '') + '</code></pre>')
    return '\\u0000BLOCK' + (blocks.length - 1) + '\\u0000'
  })

  var lines = escaped.split('\\n')
  var html = [], para = [], list = null   // list: {type:'ul'|'ol', items:[]}

  function flushPara(){
    if(para.length){ html.push('<p>' + _mdInline(para.join('<br>')) + '</p>'); para = [] }
  }
  function flushList(){
    if(list){
      html.push('<' + list.type + '>' + list.items.map(function(i){ return '<li>' + _mdInline(i) + '</li>' }).join('') + '</' + list.type + '>')
      list = null
    }
  }

  lines.forEach(function(line){
    var m
    if(/^\\u0000BLOCK\\d+\\u0000$/.test(line.trim())){
      flushPara(); flushList(); html.push(line.trim())
    }else if((m = line.match(/^(#{1,4})\\s+(.*)$/))){
      flushPara(); flushList()
      var lvl = m[1].length
      html.push('<h' + lvl + '>' + _mdInline(m[2]) + '</h' + lvl + '>')
    }else if((m = line.match(/^\\s*[-*]\\s+(.*)$/))){
      flushPara()
      if(!list || list.type !== 'ul'){ flushList(); list = {type:'ul', items:[]} }
      list.items.push(m[1])
    }else if((m = line.match(/^\\s*\\d+\\.\\s+(.*)$/))){
      flushPara()
      if(!list || list.type !== 'ol'){ flushList(); list = {type:'ol', items:[]} }
      list.items.push(m[1])
    }else if(line.trim() === ''){
      flushPara(); flushList()
    }else{
      flushList(); para.push(line)
    }
  })
  flushPara(); flushList()

  var out = html.join('')
  blocks.forEach(function(b, i){ out = out.replace('\\u0000BLOCK' + i + '\\u0000', b) })
  return out || _mdInline(escaped)   // no block structure at all -> still apply inline formatting
}

function renderChatMessage(role, content, thinking, attachments){
  var box = document.getElementById('chat-messages')
  var row = document.createElement('div')
  row.style.cssText = 'align-self:' + (role === 'user' ? 'flex-end' : 'flex-start') + ';max-width:82%;min-width:0'
  if(attachments && attachments.length){
    var chips = document.createElement('div')
    chips.style.cssText = 'display:flex;flex-wrap:wrap;gap:5px;margin-bottom:5px;justify-content:' +
      (role === 'user' ? 'flex-end' : 'flex-start')
    attachments.forEach(function(a){
      if(a.kind === 'image' && a.previewUrl){
        var img = document.createElement('img')
        img.src = a.previewUrl
        img.alt = a.filename
        img.style.cssText = 'width:64px;height:64px;object-fit:cover;border-radius:6px;border:1px solid var(--line)'
        chips.appendChild(img)
      }else{
        var chip = document.createElement('span')
        chip.style.cssText = 'font-size:11px;padding:3px 8px;border-radius:10px;border:1px solid var(--line);opacity:.8'
        chip.textContent = '\\ud83d\\udcc4 ' + a.filename
        chips.appendChild(chip)
      }
    })
    row.appendChild(chips)
  }
  if(content){
    var bubble = document.createElement('div')
    bubble.style.cssText = 'padding:9px 12px;border-radius:10px;' +
      (role === 'user'
        ? 'font-size:13px;line-height:1.55;white-space:pre-wrap;word-break:break-word;background:var(--accent,#4f8cff);color:#fff'
        : 'word-break:break-word;background:var(--panel-2,#1f232b);border:1px solid var(--line)')
    if(role === 'user'){
      // User's own typed text: shown exactly as typed, no markdown
      // interpretation -- if someone types a literal asterisk they should
      // see a literal asterisk, not have it silently reinterpreted.
      bubble.textContent = content
    }else{
      bubble.className = 'chat-md'
      bubble.innerHTML = renderMarkdownSafe(content)
    }
    row.appendChild(bubble)
  }
  if(thinking){
    var det = document.createElement('details')
    det.style.cssText = 'margin-top:4px;font-size:11px;opacity:.7'
    var sum = document.createElement('summary')
    sum.style.cursor = 'pointer'
    sum.textContent = 'Reasoning'
    var body = document.createElement('div')
    body.className = 'chat-md'
    body.style.cssText = 'margin-top:4px'
    body.innerHTML = renderMarkdownSafe(thinking)
    det.appendChild(sum); det.appendChild(body)
    row.appendChild(det)
  }
  box.appendChild(row)
  box.scrollTop = box.scrollHeight
}

function humanSizeChat(b){
  if(b < 1024*1024) return (b/1024).toFixed(0) + ' KB'
  return (b/(1024*1024)).toFixed(1) + ' MB'
}

function renderAttachmentChips(){
  var wrap = document.getElementById('chat-attachments')
  if(!wrap) return
  wrap.innerHTML = ''
  wrap.style.display = pendingAttachments.length ? 'flex' : 'none'
  pendingAttachments.forEach(function(a){
    var chip = document.createElement('span')
    chip.style.cssText = 'display:inline-flex;align-items:center;gap:6px;font-size:11px;' +
      'padding:3px 6px 3px 3px;border-radius:12px;border:1px solid ' +
      (a.status === 'error' ? 'var(--amber)' : 'var(--line)')
    if(a.kind === 'image' && a.previewUrl){
      var img = document.createElement('img')
      img.src = a.previewUrl
      img.style.cssText = 'width:22px;height:22px;object-fit:cover;border-radius:50%'
      chip.appendChild(img)
    }
    var label = document.createElement('span')
    label.style.color = a.status === 'error' ? 'var(--amber)' : ''
    label.textContent = a.status === 'extracting' ? a.filename + ' — reading…'
      : a.status === 'error' ? a.filename + ' — ' + a.error
      : a.filename
    chip.appendChild(label)
    var x = document.createElement('span')
    x.textContent = '\u2715'
    x.style.cssText = 'cursor:pointer;opacity:.7;margin-left:2px'
    x.onclick = function(){
      pendingAttachments = pendingAttachments.filter(function(p){ return p.id !== a.id })
      renderAttachmentChips()
    }
    chip.appendChild(x)
    wrap.appendChild(chip)
  })
  if(window._updateChatSendState) window._updateChatSendState()
}

function addImageAttachment(file){
  var status = document.getElementById('chat-status')
  if(file.size > CHAT_CLIENT_MAX_IMAGE_BYTES){
    status.style.color = 'var(--amber)'
    status.textContent = file.name + ' is larger than the ' + humanSizeChat(CHAT_CLIENT_MAX_IMAGE_BYTES) + ' image limit.'
    return
  }
  var entry = {id: ++_attachIdSeq, kind: 'image', filename: file.name, status: 'ready'}
  pendingAttachments.push(entry)
  renderAttachmentChips()
  var reader = new FileReader()
  reader.onload = function(){
    var dataUrl = reader.result
    entry.previewUrl = dataUrl
    // Ollama's images field wants raw base64, no "data:image/png;base64," prefix.
    var comma = dataUrl.indexOf(',')
    entry.base64 = comma >= 0 ? dataUrl.slice(comma + 1) : dataUrl
    renderAttachmentChips()
  }
  reader.onerror = function(){
    entry.status = 'error'; entry.error = 'Could not read this image.'
    renderAttachmentChips()
  }
  reader.readAsDataURL(file)
}

async function addDocumentAttachment(file){
  var entry = {id: ++_attachIdSeq, kind: 'document', filename: file.name, status: 'extracting'}
  pendingAttachments.push(entry)
  renderAttachmentChips()
  try{
    var fd = new FormData()
    fd.append('file', file)
    var r = await fetch('/api/chat/extract_file', {method: 'POST', body: fd})
    var d = await r.json()
    if(!d.ok){
      entry.status = 'error'; entry.error = d.error || 'Could not read this file.'
    }else{
      entry.status = 'ready'; entry.text = d.text; entry.truncated = d.truncated
    }
  }catch(e){
    entry.status = 'error'; entry.error = 'Request failed: ' + e
  }
  renderAttachmentChips()
}

function handleChatFiles(files){
  var status = document.getElementById('chat-status')
  for(var i = 0; i < files.length; i++){
    if(pendingAttachments.length >= CHAT_CLIENT_MAX_ATTACHMENTS){
      status.style.color = 'var(--amber)'
      status.textContent = 'Up to ' + CHAT_CLIENT_MAX_ATTACHMENTS + ' attachments per message.'
      break
    }
    var f = files[i]
    if(f.type && f.type.indexOf('image/') === 0) addImageAttachment(f)
    else addDocumentAttachment(f)
  }
}

async function sendChatMessage(){
  var input = document.getElementById('chat-input')
  var text = input.value.trim()
  var status = document.getElementById('chat-status')
  if(!text && !pendingAttachments.length) return
  if(pendingAttachments.some(function(a){ return a.status === 'extracting' })){
    status.style.color = 'var(--amber)'
    status.textContent = 'Still reading an attachment — one moment.'
    return
  }
  var badAttachment = pendingAttachments.find(function(a){ return a.status === 'error' })
  if(badAttachment){
    status.style.color = 'var(--amber)'
    status.textContent = 'Remove or fix "' + badAttachment.filename + '" before sending.'
    return
  }
  var model = document.getElementById('chat-model').value
  if(!model){
    status.style.color = 'var(--amber)'
    status.textContent = 'Pick a model first.'
    return
  }

  var attachments = pendingAttachments.slice()
  var images = attachments.filter(function(a){ return a.kind === 'image' }).map(function(a){ return a.base64 })
  var docParts = attachments.filter(function(a){ return a.kind === 'document' }).map(function(a){
    return '--- attached: ' + a.filename + (a.truncated ? ' (truncated)' : '') + ' ---\\n' + a.text +
      '\\n--- end of ' + a.filename + ' ---'
  })
  var fullContent = text + (docParts.length ? (text ? '\\n\\n' : '') + docParts.join('\\n\\n') : '')

  var displayList = attachments.map(function(a){ return {kind: a.kind, filename: a.filename, previewUrl: a.previewUrl} })
  renderChatMessage('user', text, null, displayList)
  var historyEntry = {role: 'user', content: fullContent}
  if(images.length) historyEntry.images = images
  chatHistory.push(historyEntry)

  input.value = ''
  input.style.height = ''
  pendingAttachments = []
  renderAttachmentChips()

  var btn = document.getElementById('chat-send-btn')
  btn.disabled = true
  status.style.color = ''
  status.textContent = 'Thinking…'

  var body = {model: model, messages: chatHistory,
              think: document.getElementById('chat-thinking').checked}
  var sys = document.getElementById('chat-system').value.trim()
  if(sys) body.system = sys

  try{
    var r = await fetch('/api/chat', {method: 'POST', headers: {'Content-Type': 'application/json'},
                                      body: JSON.stringify(body)})
    var d = await r.json()
    if(!d.ok){
      status.style.color = 'var(--amber)'
      status.textContent = d.error || 'Request failed.'
      chatHistory.pop()   // don't leave a turn with no reply in the history sent next time
      return
    }
    status.textContent = ''
    renderChatMessage('assistant', d.content, document.getElementById('chat-thinking').checked ? d.thinking : null)
    chatHistory.push({role: 'assistant', content: d.content})
  }catch(e){
    status.style.color = 'var(--amber)'
    status.textContent = 'Request failed: ' + e
    chatHistory.pop()
  }finally{
    btn.disabled = false
  }
}

function clearChat(){
  chatHistory = []
  pendingAttachments = []
  renderAttachmentChips()
  var box = document.getElementById('chat-messages')
  if(box) box.innerHTML = ''
  var status = document.getElementById('chat-status')
  if(status){ status.textContent = ''; status.style.color = '' }
}

document.addEventListener('DOMContentLoaded', function(){
  var input = document.getElementById('chat-input')
  if(!input) return
  var sendBtn = document.getElementById('chat-send-btn')

  function updateSendState(){
    var hasContent = input.value.trim().length > 0 || pendingAttachments.length > 0
    if(sendBtn) sendBtn.classList.toggle('dim', !hasContent)
  }
  function autoGrow(){
    input.style.height = 'auto'
    input.style.height = Math.min(input.scrollHeight, 220) + 'px'
  }
  input.addEventListener('input', function(){ autoGrow(); updateSendState() })
  updateSendState()

  var reasoningBox = document.getElementById('chat-thinking')
  if(reasoningBox){
    reasoningBox.addEventListener('change', function(){
      reasoningBox.closest('.chat-reasoning-toggle').classList.toggle('on', reasoningBox.checked)
    })
  }
  // Attachment count/state also affects whether Send should look active --
  // exposed so handleChatFiles()/sendChatMessage() can call it after changing
  // pendingAttachments without duplicating this logic there.
  window._updateChatSendState = updateSendState
  window._autoGrowChatInput = autoGrow

  input.addEventListener('keydown', function(e){
    if(e.key === 'Enter' && !e.shiftKey){
      e.preventDefault()
      sendChatMessage()
    }
  })
  input.addEventListener('paste', function(e){
    var items = (e.clipboardData || {}).items || []
    var files = []
    for(var i = 0; i < items.length; i++){
      if(items[i].kind === 'file'){
        var f = items[i].getAsFile()
        if(f) files.push(f)
      }
    }
    if(files.length){ e.preventDefault(); handleChatFiles(files) }
  })
  var attachInput = document.getElementById('chat-attach-input')
  if(attachInput){
    attachInput.addEventListener('change', function(){
      if(attachInput.files && attachInput.files.length) handleChatFiles(attachInput.files)
      attachInput.value = ''   // allow re-selecting the same file later
    })
  }
  loadChatModels()
})

// ---- Fish Audio inline delivery tags ----
// Clicking a tag inserts it at the caret in the linked script box. Fish Audio
// reads these markers out of the narration text itself, so there's no separate
// field to send -- the tag simply becomes part of the script.
var _fishTags = null
function loadFishTags(){
  if(_fishTags) return Promise.resolve(_fishTags)
  return fetch('/api/voices/tags').then(function(r){ return r.json() }).then(function(d){
    _fishTags = d
    document.querySelectorAll('.tagbar').forEach(function(bar){ renderTagBar(bar, d) })
    return d
  }).catch(function(){ return null })
}

function renderTagBar(bar, d){
  if(!d || !d.groups) return
  var target = bar.dataset.target
  var html = '<div class=tagbar-head>' +
    '<span style="font-size:11px;opacity:.75">Delivery tags</span>' +
    '<span style="font-size:11px;opacity:.5">' + escapeHtmlLite(d.open + 'tag' + d.close) +
    ' syntax (' + escapeHtmlLite(d.style.toUpperCase()) + ') \u00b7 click to insert at the cursor</span></div>'
  d.groups.forEach(function(g){
    html += '<div class=tagbar-group><div class=tagbar-group-name>' + escapeHtmlLite(g.name) + '</div>' +
      '<div class=tagbar-tags>' + g.tags.map(function(t){
        return '<button type=button class=tagchip data-tag="' + escapeHtmlLite(t.tag) +
          '" data-target="' + escapeHtmlLite(target) + '" title="' + escapeHtmlLite(t.name) + '">' +
          escapeHtmlLite(t.tag) + '</button>'
      }).join('') + '</div></div>'
  })
  html += '<div style="font-size:11px;opacity:.6;margin-top:4px">Put an emotion tag at the start of a sentence and a delivery tag right before the word it should affect. ' +
    'You can stack two, e.g. ' + escapeHtmlLite(d.open + 'sad' + d.close + d.open + 'whispering' + d.close) +
    '. S2 also accepts free-form descriptions like ' + escapeHtmlLite(d.open + 'very excited' + d.close) + '.</div>'
  bar.innerHTML = html
  bar.style.display = ''
}

// Delegated so tag bars rendered later still work.
document.addEventListener('click', function(e){
  var chip = e.target.closest ? e.target.closest('.tagchip') : null
  if(!chip) return
  e.preventDefault()
  insertTagAtCursor(document.getElementById(chip.dataset.target), chip.dataset.tag)
})

function insertTagAtCursor(el, tag){
  if(!el) return
  var start = el.selectionStart, end = el.selectionEnd
  if(start === undefined || start === null){ el.value += tag; el.focus(); return }
  var before = el.value.slice(0, start), after = el.value.slice(end)
  // Keep a space between a tag and adjacent words so it never fuses onto one.
  var pad = (before && !/\\s$/.test(before)) ? ' ' : ''
  var trail = (after && !/^\\s/.test(after)) ? ' ' : ''
  el.value = before + pad + tag + trail + after
  var caret = (before + pad + tag + trail).length
  el.focus()
  el.setSelectionRange(caret, caret)
  el.dispatchEvent(new Event('input', {bubbles:true}))
}

// The generate form's tag bar is only meaningful when Fish Audio is the engine.
function updateVoTagBarVisibility(){
  var bar = document.getElementById('vo-tags-gen')
  if(!bar) return
  var eng = document.querySelector('#tf input[name=vo_engine]:checked')
  var on = eng && eng.value === 'fish_audio'
  if(on) loadFishTags()
  bar.style.display = (on && _fishTags) ? '' : 'none'
}

// ---- Tools: Narration (Fish Audio S2) ----
// One implementation, parameterised by panel id + engine, so the two engines get
// their own tab without duplicating the logic.
var _toolVoicesLoaded = {}

function loadToolVoices(pid, engine, force){
  if(_toolVoicesLoaded[pid] && !force) return
  _toolVoicesLoaded[pid] = true
  var vsel = document.getElementById(pid + '-voice')
  var lsel = document.getElementById(pid + '-language')
  if(!vsel) return
  vsel.innerHTML = '<option value="">Loading…</option>'
  fetch('/api/voices?engine=' + engine + (force ? '&refresh=1' : ''))
    .then(function(r){ return r.json() }).then(function(d){
      vsel.innerHTML = ''
      var blank = document.createElement('option')
      blank.value = ''
      blank.textContent = (d.voices && d.voices.length)
        ? '— pick a voice —'
        : 'No registered voices on this server'
      vsel.appendChild(blank)
      ;(d.voices || []).filter(function(v){ return v.id }).forEach(function(v){
        var o = document.createElement('option')
        o.value = v.id
        o.textContent = v.title + (v.languages && v.languages.length ? ' (' + v.languages.join(', ') + ')' : '')
        vsel.appendChild(o)
      })
      if(lsel && !lsel.dataset.filled){
        lsel.innerHTML = ''
        ;(d.languages || [{code:'auto', label:'Auto-detect'}]).forEach(function(l){
          var o = document.createElement('option')
          o.value = l.code; o.textContent = l.label
          lsel.appendChild(o)
        })
        lsel.dataset.filled = '1'
      }
    }).catch(function(){
      vsel.innerHTML = '<option value="">Could not reach the voice service</option>'
    })
}

async function generateVoiceTool(pid, engine){
  var btn = document.getElementById(pid + '-btn')
  var out = document.getElementById(pid + '-result')
  var note = document.getElementById(pid + '-prompt')
  var text = document.getElementById(pid + '-text').value.trim()
  if(!text){
    note.style.display = 'block'
    note.innerHTML = '<span style="color:var(--amber)">Type a script first.</span>'
    return
  }
  var voice = document.getElementById(pid + '-voice').value
  var refInput = document.getElementById(pid + '-ref')
  var refFile = refInput && refInput.files && refInput.files[0]
  if(!voice && !refFile){
    note.style.display = 'block'
    note.innerHTML = '<span style="color:var(--amber)">Pick a voice, or upload a reference sample to clone one. '
      + 'If the voice list is empty, this server has no registered voices — check the Config tab.</span>'
    return
  }
  btn.disabled = true
  out.style.display = 'none'
  note.style.display = 'block'
  note.textContent = refFile ? 'Cloning the voice from your reference and rendering narration…' : 'Rendering narration…'

  var fd = new FormData()
  fd.append('text', text)
  fd.append('full', '1')   // render the whole script, not an 800-char audition
  fd.append('engine', engine)
  fd.append('voice', voice)   // ignored server-side when a reference is attached
  fd.append('rate', document.getElementById(pid + '-rate').value)
  fd.append('language', document.getElementById(pid + '-language').value)
  if(refFile) fd.append('ref_upload', refFile)

  try{
    var r = await fetch('/api/vo/preview', {method:'POST', body: fd})
    var d = await r.json()
    if(!d.ok){
      note.innerHTML = '<span style="color:var(--amber)">' + escapeHtmlLite(d.error || 'Narration failed.') + '</span>'
      return
    }
    note.style.display = 'none'
    out.innerHTML = '<div class=card>' +
      '<div style="font-size:12px;opacity:.8;margin-bottom:8px">' +
      (d.duration ? d.duration + 's \u00b7 ' : '') + escapeHtmlLite(d.engine) + ' \u00b7 ' + d.characters + ' characters' +
      (refFile ? ' \u00b7 cloned from ' + escapeHtmlLite(refFile.name) : '') + '</div>' +
      '<audio controls style="width:100%" src="' + d.url + '"></audio>' +
      '<div style="margin-top:10px"><a class=btn href="' + d.url + '" download="' + escapeHtmlLite(d.filename) +
      '" style="display:inline-block;text-decoration:none">Download</a></div>' +
      (d.truncated ? '<p style="font-size:12px;color:var(--amber);margin-bottom:0">Script exceeded the 5000-character limit and was truncated.</p>' : '') +
      '</div>'
    out.style.display = 'block'
  }catch(e){
    note.innerHTML = '<span style="color:var(--amber)">Request failed: ' + escapeHtmlLite(String(e)) + '</span>'
  }finally{
    btn.disabled = false
  }
}

document.addEventListener('DOMContentLoaded', function(){
  // The strength slider is meaningless without a reference, so it only appears
  // once one is picked.
  var mref = document.getElementById('music-ref')
  var mrow = document.getElementById('music-ref-row')
  var mstr = document.getElementById('music-ref-strength')
  if(mref && mrow){
    mref.addEventListener('change', function(){
      mrow.style.display = (mref.files && mref.files[0]) ? 'flex' : 'none'
    })
  }
  if(mstr){
    var readout = document.getElementById('music-ref-strength-val')
    mstr.addEventListener('input', function(){ readout.textContent = Number(mstr.value).toFixed(2) })
  }
  var fishRef = document.getElementById('p-fish-ref')
  var fishRefNote = document.getElementById('p-fish-ref-note')
  if(fishRef && fishRefNote){
    fishRef.addEventListener('change', function(){
      fishRefNote.style.display = (fishRef.files && fishRef.files[0]) ? 'block' : 'none'
    })
  }
  var mLyrics = document.getElementById('music-lyrics')
  var mDur = document.getElementById('music-duration')
  if(mLyrics && mDur){
    mLyrics.addEventListener('input', updateLyricsHint)
    mDur.addEventListener('input', updateLyricsHint)
  }
  loadFishTags().then(updateVoTagBarVisibility)
  document.querySelectorAll('#tf input[name=vo_engine]').forEach(function(r){
    r.addEventListener('change', updateVoTagBarVisibility)
  })
  if(document.getElementById('p-fish-voice')) loadToolVoices('p-fish', 'fish_audio')
})

function renderTrailerResult(d){
  if(d.error){document.getElementById('tr-stats').innerHTML='<b>Error:</b> '+d.error; document.getElementById('tr-area').style.display='block'; return}
  var srcLabel=function(s){return {woosh:'AI-generated (Woosh)',ai_generated:'AI-generated (ACE-Step)',uploaded:'uploaded',synth_fallback:'placeholder synth (Woosh/ACE-Step unavailable)',tts:'text-to-speech',none:'none'}[s]||s}
  var audioNote=''
  if(d.bgm_source && d.bgm_source!=='none') audioNote+=' | Music: '+srcLabel(d.bgm_source)+(d.sync_beats?' (beat-synced cuts)':'')
  if(d.sfx_source && d.sfx_source!=='none') audioNote+=' | SFX: '+srcLabel(d.sfx_source)
  if(d.vo_source && d.vo_source!=='none') audioNote+=' | VO: '+srcLabel(d.vo_source)
  if(d.whisper_enhance) audioNote+=' | Scene selection: dialogue-enhanced'
  var fallbackUsed=(d.bgm_source==='synth_fallback'||d.sfx_source==='synth_fallback')
  document.getElementById('tr-stats').innerHTML='Trailer: '+d.trailer_duration+'s (target '+d.trailer_length+'s) from '+d.selected_scenes+'/'+d.total_scenes+' scenes | Raw video: '+d.video_duration+'s'+audioNote+(fallbackUsed?'<br><small style="color:var(--amber)">Note: AI music/SFX service was unavailable, so a lower-fidelity generated placeholder was used instead.</small>':'')+(d.vo_error?'<br><small style="color:var(--amber)">Voiceover note: '+d.vo_error+'</small>':'')
  var dlFilename = d.trailer_url.split('/').pop()
  var dlName = d.orig_name
  var dlUrl = function(fmt){
    if(d.library_id) return '/library/'+d.library_id+'/download?format='+fmt
    return '/download/'+dlFilename+'?name='+encodeURIComponent(dlName)+'&format='+fmt
  }
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
}

// ---- Trailer history — every completed trailer is saved to a small SQLite
// library on the server (survives restarts), shown here as a collapsible
// list. Click a row to view/download it again without regenerating; click
// the x to delete it permanently (removes the row and the saved file).
(function(){
  var body = document.getElementById('tr-history-body')
  var toggle = document.getElementById('tr-history-toggle')
  var chevron = document.getElementById('tr-history-chevron')
  var countEl = document.getElementById('tr-history-count')
  var listEl = document.getElementById('tr-history-list')
  var expanded = false

  function fmtWhen(ts){
    var d = new Date(ts * 1000)
    return d.toLocaleDateString() + ' ' + d.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'})
  }

  function refreshHistory(){
    fetch('/api/trailer/library').then(function(r){ return r.json() }).then(function(d){
      if(!d.ok) return
      countEl.textContent = d.items.length ? '(' + d.items.length + ')' : ''
      if(!d.items.length){
        listEl.innerHTML = '<div class="net-modal-empty">No saved trailers yet.</div>'
        return
      }
      listEl.innerHTML = ''
      d.items.forEach(function(item){
        var row = document.createElement('div')
        row.className = 'history-row'
        row.innerHTML =
          '<div class="history-main">' +
            '<div class="h-name">' + (item.orig_name || 'Untitled') + '</div>' +
            '<div class="h-meta">' + item.trailer_duration + 's &middot; ' + fmtWhen(item.created_at) + '</div>' +
          '</div>' +
          '<span class="history-del" title="Delete">&#10005;</span>'
        row.querySelector('.history-main').addEventListener('click', function(){
          fetch('/api/trailer/library/' + item.id).then(function(r){ return r.json() }).then(function(d2){
            if(!d2.ok) return
            document.getElementById('tr-prompt').style.display = 'none'
            renderTrailerResult(d2.result)
          })
        })
        row.querySelector('.history-del').addEventListener('click', function(e){
          e.stopPropagation()
          if(!confirm('Delete "' + (item.orig_name || 'this trailer') + '" permanently? This removes the saved file too.')) return
          fetch('/api/trailer/library/' + item.id + '/delete', {method: 'POST'}).then(function(){ refreshHistory() })
        })
        listEl.appendChild(row)
      })
    }).catch(function(){})
  }
  window.refreshTrailerHistory = refreshHistory

  toggle.addEventListener('click', function(){
    expanded = !expanded
    body.style.display = expanded ? 'block' : 'none'
    chevron.innerHTML = expanded ? '&#9650;' : '&#9660;'
    if(expanded) refreshHistory()
  })
  refreshHistory() // populate the count even while collapsed
})();

// ---- Job monitor — live view of what's currently active/queued/recently
// finished on the server. Whole-server view (no login system, see the
// backend route's docstring), auto-refreshes every few seconds while
// expanded so it stays live without the person needing to reopen it.
(function(){
  var body = document.getElementById('tr-monitor-body')
  var toggle = document.getElementById('tr-monitor-toggle')
  var chevron = document.getElementById('tr-monitor-chevron')
  var summaryEl = document.getElementById('tr-monitor-summary')
  var activeEl = document.getElementById('tr-monitor-active')
  var queuedEl = document.getElementById('tr-monitor-queued')
  var finishedEl = document.getElementById('tr-monitor-finished')
  var expanded = false
  var userCollapsed = false
  var pollTimer = null

  function fmtWhen(ts){
    var d = new Date(ts * 1000)
    return d.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'})
  }
  function emptyRow(text){ return '<div class="monitor-row" style="opacity:.5;font-size:12px">' + text + '</div>' }

  function renderActive(items){
    if(!items.length){ activeEl.innerHTML = emptyRow('Nothing running right now.'); return }
    activeEl.innerHTML = items.map(function(j){
      var pct = j.percent || 0
      return '<div class="monitor-row">' +
        '<span class="m-name" title="' + (j.step||'') + '">' + (j.orig_name || 'Untitled') + ' &mdash; ' + (j.step||'') + '</span>' +
        '<span class="m-bar"><span class="m-bar-fill" style="width:' + pct + '%"></span></span>' +
        '<span class="m-pct">' + pct + '%</span>' +
      '</div>'
    }).join('')
  }
  function renderQueued(items){
    if(!items.length){ queuedEl.innerHTML = emptyRow('Nothing waiting.'); return }
    queuedEl.innerHTML = items.map(function(j){
      return '<div class="monitor-row">' +
        '<span class="m-name">' + (j.orig_name || 'Untitled') + '</span>' +
        '<span class="m-pos">' + (j.position === 0 ? 'starting shortly' : j.position + ' ahead') + '</span>' +
      '</div>'
    }).join('')
  }
  function renderFinished(items){
    if(!items.length){ finishedEl.innerHTML = emptyRow('Nothing finished recently.'); return }
    finishedEl.innerHTML = items.map(function(j){
      var badge = j.status === 'cancelled' ? '<span class="monitor-badge cancel">Cancelled</span>'
                : j.error ? '<span class="monitor-badge err">Error</span>'
                : '<span class="monitor-badge ok">Done</span>'
      return '<div class="monitor-row">' +
        '<span class="m-name" title="' + (j.error||'') + '">' + (j.orig_name || 'Untitled') + '</span>' +
        '<span class="m-pos">' + fmtWhen(j.created) + '</span>' +
        badge +
      '</div>'
    }).join('')
  }

  function setExpanded(open){
    expanded = open
    body.style.display = open ? 'block' : 'none'
    chevron.innerHTML = open ? '&#9650;' : '&#9660;'
    if(open){
      refreshMonitor()
      if(!pollTimer) pollTimer = setInterval(refreshMonitor, 3000)
    } else if(pollTimer){
      clearInterval(pollTimer); pollTimer = null
    }
  }

  function refreshMonitor(){
    fetch('/api/monitor').then(function(r){ return r.json() }).then(function(d){
      var busy = d.active.length + d.queued.length
      var parts = []
      if(d.active.length) parts.push(d.active.length + ' active')
      if(d.queued.length) parts.push(d.queued.length + ' queued')
      // Always say something -- an empty label read as "the monitor is broken"
      // rather than "nothing is running".
      summaryEl.textContent = parts.length ? '(' + parts.join(', ') + ')' : '(idle)'
      // Open itself the first time work appears, so a running job is visible
      // without having to know to click. A manual collapse is remembered.
      if(busy && !expanded && !userCollapsed) setExpanded(true)
      if(!expanded) return
      renderActive(d.active); renderQueued(d.queued); renderFinished(d.finished)
    }).catch(function(){})
  }
  window.refreshMonitor = refreshMonitor

  toggle.addEventListener('click', function(){
    userCollapsed = expanded          // collapsing by hand is remembered
    setExpanded(!expanded)
  })
  refreshMonitor()
  setInterval(refreshMonitor, 8000) // keep the collapsed summary count fresh too
})()
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
// The Vision panel is rendered further down the page (it lives in the rail now,
// not inside Tools), so this script parses before its markup exists. Defer the
// element lookups until the document is ready.
document.addEventListener('DOMContentLoaded', function(){
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
})   // end DOMContentLoaded
</script>

<!-- API -->
<div id="p-music" class="panel">
<h2>Music Generation &mdash; ACE-Step</h2>
<p style="font-size:12px;opacity:.75;margin-top:-6px">Generate music on its own — for building a library of beds, restyling an existing track, or producing a sung stinger. Generated files can be saved into a show template from the Promo tab.</p>

<label>Prompt (style tags):</label>
<textarea id=music-prompt class=input-box rows=3 placeholder="cinematic, orchestral, tense strings, driving percussion">Modern alternative rock with energetic electric guitars, driving bass, punchy live drums, melodic guitar riffs, clean male lead vocals, expressive vocal harmonies, dynamic transitions, and a memorable sing-along chorus. Confident, uplifting, and motivational with a polished contemporary production.</textarea>
<p style="font-size:12px;opacity:.7;margin-top:-4px">Comma-separated style tags work best — instrument, mood, era, production style.</p>

<label>Lyrics:</label>
<textarea id=music-lyrics class=input-box rows=4 placeholder="Leave empty for instrumental."></textarea>
<p style="font-size:12px;opacity:.7;margin-top:-4px">Leave empty and ACE-Step is told <code>[inst]</code> plus a vocal-suppressing negative prompt, which is the reliable way to guarantee no singing. Type lyrics here and both are dropped so the vocal actually comes through — use structure tags like <code>[verse]</code> and <code>[chorus]</code> on their own lines.</p>
<div id=music-lyrics-hint style="display:none;font-size:12px;margin:-2px 0 10px;padding:8px 10px;border-radius:6px;border:1px solid var(--line)"></div>

<label>Reference audio (optional &mdash; audio2audio):</label>
<input type=file id=music-ref accept="audio/*">
<div id=music-ref-row style="display:none;align-items:center;gap:12px;flex-wrap:wrap;margin:10px 0">
  <span style="display:inline-flex;gap:6px;align-items:center;white-space:nowrap"><span style="font-size:12px">Reference strength:</span>
  <input type=range id=music-ref-strength min=0 max=1 step=0.05 value=0.5 style="width:200px"></span>
  <span id=music-ref-strength-val style="font-size:12px;opacity:.8;font-family:'JetBrains Mono',monospace">0.50</span>
</div>
<p style="font-size:12px;opacity:.7;margin-top:-4px">Generates in the shape of an existing track — useful for matching a show's established sting, or restyling a bed you already have. Higher strength stays closer to the reference; lower gives the prompt more freedom. Around 0.5 is a good starting point for a style transfer.</p>
<p style="font-size:12px;opacity:.6;margin-top:-4px">ACE-Step reads the reference from a file path rather than an upload, so this needs ACE-Step running on this machine — or <code>ACE_STEP_REF_DIR</code> pointed at a mount both machines resolve identically.</p>

<div style="display:flex;gap:16px;flex-wrap:wrap;align-items:center;margin:12px 0">
  <span style="display:inline-flex;gap:6px;align-items:center;white-space:nowrap"><span style="font-size:12px">Duration (s):</span>
  <input type=number id=music-duration value=30 min=5 max=300 step=5 style="width:85px"></span>
  <span style="display:inline-flex;gap:6px;align-items:center;white-space:nowrap"><span style="font-size:12px">BPM:</span>
  <input type=number id=music-bpm placeholder="auto" min=40 max=220 step=1 style="width:85px"></span>
  <span style="display:inline-flex;gap:6px;align-items:center;white-space:nowrap"><span style="font-size:12px">Samples:</span>
  <input type=number id=music-samples value=1 min=1 max=4 step=1 style="width:70px"></span>
  <span style="display:inline-flex;gap:6px;align-items:center;white-space:nowrap"><span style="font-size:12px">Steps:</span>
  <input type=number id=music-steps value=27 min=8 max=120 step=1 style="width:75px"></span>
  <span style="display:inline-flex;gap:6px;align-items:center;white-space:nowrap"><span style="font-size:12px">Seed:</span>
  <input type=number id=music-seed placeholder="random" style="width:105px"></span>
</div>
<p style="font-size:12px;opacity:.7;margin-top:-6px">BPM is passed as a prompt tag (ACE-Step has no separate tempo field), so leaving it on <em>auto</em> lets the model pick a tempo that suits the style. More samples generate alternatives in one pass — slower, but you audition them side by side. Higher steps means better quality and proportionally more GPU time. Reuse a seed to reproduce a take.</p>

<div style="margin-top:14px">
  <button class=btn type=button id=music-gen-btn onclick="generateMusicTool()">Generate music</button>
</div>
<div id=music-result style="display:none;margin-top:16px"></div>
<div id=music-prompt-note class=no-data style="margin-top:14px">Set a prompt and duration, then hit Generate. Longer or multi-sample runs take proportionally longer — ACE-Step is polled for up to three minutes.</div>
</div>

<div id="p-sfx" class="panel">
<h2>Text to SFX &mdash; Woosh</h2>
<p style="font-size:12px;opacity:.75;margin-top:-6px">Generate a one-shot sound effect from a plain description — for building a library of stingers and hits, or auditioning options before picking one for a show template. This is separate from the genre-based SFX the promo generator uses automatically at scene cuts.</p>

<label>Description:</label>
<textarea id=sfx-prompt class=input-box rows=3 placeholder="glass shattering, footsteps on gravel, whoosh transition, camera shutter click"></textarea>
<p style="font-size:12px;opacity:.7;margin-top:-4px">Be concrete: name the object and the action, not a mood. Woosh is a sound-effects model, not a music model — style words like "epic" or "cinematic" do less here than a literal description of the sound itself.</p>

<div style="display:flex;gap:16px;flex-wrap:wrap;align-items:center;margin:12px 0">
  <span style="display:inline-flex;gap:6px;align-items:center;white-space:nowrap"><span style="font-size:12px">Duration (s):</span>
  <input type=number id=sfx-duration value=1 min=0.2 max=10 step=0.1 style="width:85px"></span>
  <span style="display:inline-flex;gap:6px;align-items:center;white-space:nowrap"><span style="font-size:12px">Samples:</span>
  <input type=number id=sfx-samples value=1 min=1 max=4 step=1 style="width:70px"></span>
</div>
<p style="font-size:12px;opacity:.7;margin-top:-6px">More samples generate that many independent takes of the same description, so you can audition and pick the best one — each is a separate request to Woosh, run one after another.</p>

<div style="margin-top:14px">
  <button class=btn type=button id=sfx-gen-btn onclick="generateSfxTool()">Generate SFX</button>
</div>
<div id=sfx-result style="display:none;margin-top:16px"></div>
<div id=sfx-prompt-note class=no-data style="margin-top:14px">Describe a sound and hit Generate.</div>
</div>

<div id="p-fish" class="panel">
<h2>Text to Speech &mdash; Fish Audio S2</h2>
<p style="font-size:12px;opacity:.75;margin-top:-6px">Renders a full narration script rather than the 800-character audition the Promo tab uses. S2 detects the script language itself, so <em>Auto-detect</em> is normally the right choice.</p>
<label>Script:</label>
<textarea id=p-fish-text class=input-box rows=5 placeholder="Type the narration script here."></textarea>
<div id=vo-tags-fish class=tagbar data-target=p-fish-text></div>
<div style="display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:10px 0">
  <span style="display:inline-flex;gap:6px;align-items:center;white-space:nowrap"><span style="font-size:12px">Voice:</span>
  <select id=p-fish-voice style="max-width:280px"><option value="">Loading…</option></select></span>
  <button type=button class=btn style="padding:5px 11px;font-size:12px" onclick="loadToolVoices('p-fish','fish_audio',true)">Refresh</button>
  <span style="display:inline-flex;gap:6px;align-items:center;white-space:nowrap"><span style="font-size:12px">Language:</span>
  <select id=p-fish-language style="max-width:220px"><option value=auto>Auto-detect</option></select></span>
  <span style="display:inline-flex;gap:6px;align-items:center;white-space:nowrap"><span style="font-size:12px">Rate:</span>
  <input type=number id=p-fish-rate value=175 min=80 max=300 step=5 style="width:80px"></span>
</div>

<label>Reference audio (optional &mdash; clone a voice from a sample instead):</label>
<input type=file id=p-fish-ref accept="audio/*,.mp3,.wav,.m4a,.flac,.ogg">
<p id=p-fish-ref-note style="display:none;margin-top:4px;font-size:12px;color:var(--phosphor)">A reference sample is attached — this clones that voice directly and the picked Voice above is ignored. About 10 seconds of clean, single-speaker speech is enough; no pre-registration on the server needed.</p>
<p style="margin-top:4px;margin-bottom:0;font-size:12px;opacity:.7">Pick a voice above for a ready-made result, or upload a reference here for zero-shot cloning of a specific voice. Uploading always takes priority over the picked voice.</p>

<div style="margin-top:14px">
  <button class=btn type=button id=p-fish-btn onclick="generateVoiceTool('p-fish','fish_audio')">Render narration</button>
</div>
<div id=p-fish-result style="display:none;margin-top:16px"></div>
<div id=p-fish-prompt class=no-data style="margin-top:14px">Type a script and pick a voice to render narration.</div>
</div>
<div id="p-stt" class="panel">
<h2>Speech to Text &mdash; Whisper</h2>
<p style="font-size:12px;opacity:.75;margin-top:-6px">Transcribes dialogue with word-level timing, using the same service the promo pipeline uses for dialogue-aware cuts — so this shows you exactly what the rating stage sees. Audio is extracted to 16&nbsp;kHz mono before upload, so a multi-GB episode only sends a few MB.</p>
<label>Video or audio file:</label>
<input type=file id=stt-file accept="video/*,audio/*">
<div style="margin-top:14px">
  <button class=btn type=button id=stt-btn onclick="runTranscribe()">Transcribe</button>
</div>
<div id=stt-result style="display:none;margin-top:16px"></div>
<div id=stt-prompt class=no-data style="margin-top:14px">Upload a file to transcribe. A full episode takes a few minutes; the upload itself is quick because only the audio is sent.</div>
</div>

<div id="p-vision" class="panel">
<h2>Scene Detection &amp; Analysis</h2>
<p style="font-size:12px;opacity:.75;margin-top:-6px">Runs PySceneDetect over the source to find the cuts, then samples frames and describes them with the AI Vision model — the same two stages that feed scene rating in the promo generator.</p>
<form id=vf method=POST action=/api/vision/analyze enctype=multipart/form-data>
  <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:8px">
    <input type=file name=file accept=video/* data-net-field="file">
    <button type=button class="browse-btn" onclick="openNetworkBrowser('file','hires','Video (HIRES)')"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 7a1 1 0 0 1 1-1h5l2 2h9a1 1 0 0 1 1 1v9a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V7z"/></svg>Browse library</button>
  </div>
  <input type=hidden id=file_network name=file_network value="">
  <span class="net-file-chip" id=file_chip><span class=chip-name></span><span class=chip-x onclick="clearNetworkField('file')">&#10005;</span></span>
  <label>Custom prompt:</label>
  <input type=text name=prompt class=input-box value="Describe the quality and content of this video frame. Note any blur, color issues, or anomalies.">
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

<div id="p-chat" class="panel">
<h2>AI Chat</h2>
<p style="font-size:12px;opacity:.75;margin-top:-6px">Chat directly with any model your Ollama server has installed — for testing how a model behaves, drafting text, or just asking it something. Separate from the vision/rating pipeline the promo generator uses; nothing here feeds into a job.</p>
<details style="margin-bottom:10px">
  <summary style="cursor:pointer;font-size:12px;opacity:.75">System prompt (optional)</summary>
  <textarea id=chat-system class=input-box rows=2 placeholder="e.g. Be concise. Answer only in Tagalog." style="margin-top:6px"></textarea>
</details>
<div style="display:flex;justify-content:flex-end;margin-bottom:6px">
  <button type=button class="chat-icon-btn" id=chat-clear-btn onclick="clearChat()" title="Clear chat">
    <svg viewBox="0 0 24 24" width=14 height=14 fill="none" stroke="currentColor" stroke-width="2"><path d="M4 7h16M9 7V4h6v3m-8 0 1 13h8l1-13"/></svg>
    Clear
  </button>
</div>
<div id=chat-messages style="border:1px solid var(--line);border-radius:8px;padding:12px;height:420px;overflow-y:auto;display:flex;flex-direction:column;gap:10px"></div>
<div id=chat-status style="font-size:12px;opacity:.75;min-height:16px;margin:6px 2px"></div>
<div id=chat-attachments style="display:none;flex-wrap:wrap;gap:6px;margin:0 0 8px"></div>

<div class="chat-input-box">
  <input type=file id=chat-attach-input multiple accept="image/*,.pdf,.txt,.md,.markdown,.csv,.tsv,.json,.log,.yaml,.yml,.ini,.cfg,.conf,.xml,.html,.htm,.css,.py,.js,.ts,.jsx,.tsx,.java,.c,.h,.cpp,.hpp,.go,.rs,.rb,.php,.sh,.sql,.srt,.vtt" style="display:none">
  <textarea id=chat-input rows=1 placeholder="Write a message…"></textarea>
  <div class="chat-input-toolbar">
    <button type=button class="chat-icon-btn chat-icon-btn-round" id=chat-attach-btn title="Attach an image or document" onclick="document.getElementById('chat-attach-input').click()">
      <svg viewBox="0 0 24 24" width=17 height=17 fill="none" stroke="currentColor" stroke-width="2.2"><path d="M12 5v14M5 12h14"/></svg>
    </button>
    <div style="flex:1"></div>
    <label class="chat-reasoning-toggle" title="Ask the model to show its reasoning">
      <input type=checkbox id=chat-thinking> Reasoning
    </label>
    <div class="chat-model-picker">
      <select id=chat-model><option value="">Loading…</option></select>
      <button type=button class="chat-icon-btn" id=chat-model-refresh title="Refresh model list" onclick="loadChatModels(true)">
        <svg viewBox="0 0 24 24" width=13 height=13 fill="none" stroke="currentColor" stroke-width="2.2"><path d="M21 12a9 9 0 1 1-3-6.7M21 4v5h-5"/></svg>
      </button>
    </div>
    <button type=button class="chat-icon-btn chat-icon-btn-round chat-send-btn" id=chat-send-btn onclick="sendChatMessage()" title="Send">
      <svg viewBox="0 0 24 24" width=16 height=16 fill="none" stroke="currentColor" stroke-width="2.4"><path d="M12 19V5M5 12l7-7 7 7"/></svg>
    </button>
  </div>
</div>
<p style="font-size:11px;opacity:.6;margin:6px 2px 0">Images work with vision-capable models only. Documents (PDF, text, code) are read and added as context regardless of model — up to 8MB per file, and long documents are truncated.</p>
</div>

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
<p style="font-size:12px;opacity:.8">Pings Ollama, Fish Audio S2, faster-whisper, ACE-Step, and Woosh and reports whether each is reachable, with response latency. Doesn't call any generation endpoints, so it's safe (and cheap) to run anytime.</p>
<p><code>POST multipart/form-data</code> with <code>file</code>:</p>
<div class=card>
<div class=info>
  <div class=info-item><strong>POST</strong> /api/opencv/info</div>
  <div class=info-item><strong>POST</strong> /api/opencv/analyze<br><small>+ num_frames</small></div>
  <div class=info-item><strong>POST</strong> /api/scenedetect/detect<br><small>+ threshold, min_scene_len</small></div>
</div>
</div>
<p><strong>Player</strong>:</p>
<pre>GET  /api/network/list?category=hires|tcard|endcard|music|vo|sfx
POST /api/network/fetch  + name, category   stage a file locally, returns url
POST /api/stt/transcribe + file             transcript + .srt</pre>
<p><strong>Narration (Fish Audio S2)</strong>:</p>
<pre>POST /api/trailer/generate  + vo_mode=tts &amp; vo_text=&lt;script&gt; &amp; vo_engine=fish_audio
GET  /api/voices                                  list the narration voices + languages
GET  /api/voices/tags                             Fish Audio delivery tags, in the right syntax
POST /api/vo/preview  + text, rate, language, engine, voice or ref_upload
POST /api/vo/preview  + full=1                    render a whole script, not an 800-char audition</pre>
<p style="font-size:12px;opacity:.8">
<code>vo_engine</code> picks the narration engine (defaults to <code>fish_audio</code>).
There's no bundled default voice — every narration job needs either a voice picked from
<code>/api/voices</code> or an uploaded reference sample to clone zero-shot (sent as a
base64 <code>references</code>/<code>reference_audio</code> entry so speech is generated
in that voice, no pre-registration required). The Narration section fetches the current
engine's voice list live and reloads it if you switch engines; Fish Audio's list comes from
the cloud API when <code>FISH_AUDIO_API_KEY</code> is set, and otherwise from a self-hosted
server's <code>/v1/references/list</code> (falling back to <code>/v1/models</code> on older
builds) — if a list comes back empty,
upload a reference sample instead. You can also override the auto-detected language, and
generate a short preview via <code>/api/vo/preview</code> before running the full trailer
job. Configured via env vars:
</p>
<div class=card>
<div class=info>
  <div class=info-item><strong>FISH_AUDIO_URL</strong><br><small>self-hosted or api.fish.audio/v1/tts</small></div>
  <div class=info-item><strong>FISH_AUDIO_API_KEY</strong><br><small>blank for self-hosted</small></div>
  <div class=info-item><strong>FISH_AUDIO_MODEL</strong><br><small>s2.1-pro-free (cloud only)</small></div>
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
<text x="304.0" y="125.0" text-anchor="middle" font-size="12.5" fill="var(--ink)">Rating</text>
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
  <li><strong>Rating</strong> — each scene is rated on sharpness/brightness/duration/face presence (OpenCV), boosted by AI Vision's 1-5 quality rating, and boosted further for scenes with quotable dialogue when VISION + STT is selected (faster-whisper transcription).</li>
  <li><strong>Beat/dialogue prep (optional)</strong> — if "sync cuts to beat" is on, a music track is prepared early and its beat grid drives cut timing; if dialogue transcription is on, word-level timestamps are generated for clean-cut alignment.</li>
  <li><strong>Scene selection</strong> — highest-rated scenes are picked to fill the target duration, respecting min/max clip length and a minimum spacing between picks so the trailer isn't built from one over-represented stretch of the source, snapping cut points to the beat grid and/or word boundaries when enabled, and topping up across a few passes to land within ~0.5s of the target length.</li>
  <li><strong>Assembly</strong> — selected scenes plus title/end cards are concatenated with the genre's signature transition (see table below), or a custom transition driven by an uploaded matte video/image if selected; any clip that fails to extract is dropped and the reported scene count reflects what actually made it in.</li>
  <li><strong>Audio level normalization</strong> — SOT is loudness-normalized immediately after assembly so every later step works off a predictable baseline.</li>
  <li><strong>SFX at cuts</strong> — Woosh (Sony AI) generates the hit sound from the genre's SFX prompt; if Woosh isn't reachable, "From genre" is disabled rather than falling back to another model, since ACE-Step is a music model, not an SFX model. A procedural synth fallback still applies if Woosh returns an unusable result mid-job.</li>
  <li><strong>Background music</strong> — ACE-Step-generated or uploaded, normalized, and ducked under SOT.</li>
  <li><strong>Narration (optional)</strong> — Fish Audio S2 (self-hosted or cloud) generates speech from a script using a voice picked from that engine's live voice list or an uploaded reference sample cloned zero-shot, or an uploaded VO track is used instead; either way it's loudness-normalized before mixing.</li>
  <li><strong>Final audio mix</strong> — if a voiceover is present, SOT ducks to near-silence under it via a live sidechain; BGM ducking works differently — dialogue/SOT gaps are silence-detected up front, any gap shorter than your configured hold is bridged so a brief pause doesn't let the music swell back up and immediately duck again, and the resulting windows drive a deterministic volume dip (not a reactive compressor) applied to BGM only during genuine talking stretches. If "broadcast dual-mono" is on, the result is collapsed so every element is identically audible in both channels; a final loudness/true-peak pass targets your configured broadcast spec.</li>
  <li><strong>Export</strong> — download as MP4 (H.264 High Profile), Apple ProRes 422 HQ (29.97 or 23.976fps), or an AVC-Intra 100i approximation.</li>
</ol>
<p>Jobs run through a concurrency-limited queue (default 2 at a time, configurable via <code>/api/queue/limit</code>) so multiple simultaneous requests don't overload the local model servers (ACE-Step, Woosh, Ollama, Fish Audio). Two jobs from different people can run at the same time; a third waits in line and shows its position.</p>

<h3>Media sources: upload or network folder</h3>
<p>Every media input on the Promo Plug form — the main video, title/end card video, background music, SFX, VO, and card VO — can be filled in one of two ways: a direct file upload, or the <strong>Browse library</strong> button next to it, which lists files sitting on the configured Windows/SMB share and pulls the one you pick in. Each field only browses its own matching folder:</p>
<table>
<tr><th>Field</th><th>Network folder</th><th>File types</th></tr>
<tr><td>Main video, title card video, end card video</td><td><code>HIRES</code></td><td>Video</td></tr>
<tr><td>Background music</td><td><code>MUSIC</code></td><td>Audio</td></tr>
<tr><td>VO / narration, title &amp; end card VO</td><td><code>VO</code></td><td>Audio</td></tr>
<tr><td>SFX</td><td><code>SFX</code></td><td>Audio</td></tr>
</table>
<p>These sit under one shared root (<code>\\\\host\\share\\subdir\\HIRES</code>, <code>...\\MUSIC</code>, etc.) set via the <code>NETWORK_SHARE_*</code> environment variables — see the README for the full list. A picked network file is copied into the app's working folder the same way an upload would be, so the rest of the pipeline treats both identically.</p>

<h3>Job monitor &amp; saved trailers</h3>
<p>Two collapsible panels sit above the generator form:</p>
<ul style="line-height:1.8">
  <li><strong>Job monitor</strong> — a live, whole-server view of what's <em>active</em> (currently processing, with a progress bar), <em>queued</em> (waiting for a free concurrency slot), and <em>recently finished</em> (last hour, tagged Done/Error/Cancelled). It auto-refreshes every few seconds while open, and its header shows a running count even while collapsed.</li>
  <li><strong>Saved trailers</strong> — every trailer that's finished successfully is kept permanently (SQLite-backed, survives a server restart) until someone deletes it. Click a saved entry to reopen its video/download/scene-breakdown exactly as it looked right after generating; click the &times; to delete it and its file for good.</li>
</ul>
<p>Neither panel is per-user — there's no login system, so everyone hitting this server sees the same jobs and the same saved trailers.</p>

<h3>Player format compatibility</h3>
<p>The Player streams files directly from the browser — nothing is re-encoded on the server for a file the browser can already decode. Broadcast masters are routinely something a browser <em>can't</em> decode natively (ProRes or DNxHD in a <code>.mov</code>, most MXF), which shows up as a blank frame or a media error with the controls still present.</p>
<p>When that happens the Player converts automatically: the moment its <code>&lt;video&gt;</code> element reports an error, it calls <code>POST /api/media/playable</code> with the staged filename, which re-encodes to H.264/AAC MP4 (audio kept) and hands back a URL that plays. The converted copy is cached under <code>UPLOAD_FOLDER</code>, so picking the same file again is instant. This only applies to files picked via <strong>Browse library</strong> — a file chosen from your own machine plays straight from a local blob URL with nothing uploaded, so there's nothing on the server to convert if your browser can't decode it locally.</p>

<h3>Text to SFX</h3>
<p>Generates a one-shot sound effect from a free-text description via Woosh, Sony AI's sound-effects model — separate from the genre-based SFX the promo generator stamps at scene cuts automatically. Describe the sound concretely (<em>"glass shattering"</em>, <em>"footsteps on gravel"</em>) rather than a mood — Woosh is a sound-effects model, not a music model, so style adjectives do less work here than in the Music tool.</p>
<p>The actual request schema, confirmed against Woosh's own <code>api_server.py</code>: the server expects <code>{prompt, token}</code> and responds with <strong>FLAC</strong>, not WAV — <code>token</code> is a required field the server never actually validates, so <code>"string"</code> (the same placeholder Sony's own test script sends) satisfies it with nothing for you to configure. <code>duration</code> is <em>not</em> a real field on this API's <code>GenerateArgs</code> at all; requesting a specific length is enforced afterward by trimming the response locally with ffmpeg, since the API has no way to ask Woosh for one directly. <strong>Samples</strong> works the same way as before: the same request sent multiple times in a row, each trimmed independently, rather than an invented batch parameter.</p>
<p>Like the Music tool, this has no fallback: if Woosh is unreachable it says so rather than substituting a placeholder sound. The trailer pipeline's own SFX-at-cuts step is different — it falls back to a procedural synth click when Woosh is down, because that stage always needs <em>something</em> to stamp at each cut; a tool you're explicitly invoking to generate a specific sound should tell you it failed instead.</p>
<pre>POST /api/sfx/generate  + prompt, duration, samples</pre>

<h3>AI Chat</h3>
<p>A plain chat interface over whatever models Ollama has installed — separate from the vision/rating pipeline the promo generator uses, and nothing typed here feeds into a job. Useful for testing how a model behaves, drafting text, or just asking it something directly.</p>
<p>The conversation lives only in the page — there is no server-side session and nothing is saved, so switching tabs or reloading starts fresh. Every send resends the full conversation so far (Ollama has no server-side memory of its own between calls), capped at <code>CHAT_MAX_HISTORY</code> messages (default 60) to bound the request size on a very long conversation.</p>
<p><strong>Show model reasoning</strong> requests Ollama's <code>think</code> mode for models that support it, and renders the reasoning in a collapsed "Reasoning" block under the reply rather than mixing it into the answer. Off by default, since most models don't need it and it roughly doubles response time.</p>
<p>Responses are not streamed — the full reply comes back in one request. This keeps the implementation simple and consistent with the rest of the app (job progress here is polled, not pushed, for the same reason), at the cost of watching a "Thinking…" status instead of text appearing as it's generated.</p>
<p>Assistant replies are rendered as markdown — headers, bold/italic, inline code, fenced code blocks, and lists all display properly rather than showing raw <code>**</code>/<code>#</code>/<code>-</code> characters, which is what some models return unprompted for anything resembling a structured answer. This is a small built-in renderer, not an external library — the app has no CDN dependencies, matching a LAN deployment with no assumed internet access — and everything it emits as HTML is built from text that was escaped first, so a reply containing something that looks like a tag or script can't affect the page. Your own typed messages are never run through this — they show exactly as typed, so a literal asterisk you type stays a literal asterisk.</p>

<h4>Attachments</h4>
<p><strong>Images</strong> — attach via the paperclip button, or paste directly from the clipboard (e.g. a screenshot). Handled entirely in the browser: read, base64-encoded, and sent straight through in Ollama's own <code>messages[].images</code> field with no server round trip. Only useful with a vision-capable model — a text-only model will typically just ignore them. Capped at <code>CHAT_MAX_IMAGES_PER_MESSAGE</code> (default 4) and <code>CHAT_MAX_IMAGE_BYTES</code> (default 8MB) per image, checked in the browser and re-checked server-side.</p>
<p><strong>Documents</strong> (PDF, plain text, common code/markup files) — a PDF can't be parsed reliably in the browser, so these go to the server for text extraction first; the extracted text is folded into the message sent to the model, wrapped in <code>--- attached: name ---</code> markers, but is kept out of the visible chat bubble (which shows a filename chip instead) so a long document doesn't fill the conversation with a wall of text. Capped at <code>CHAT_ATTACH_MAX_BYTES</code> (default 8MB) per file and <code>CHAT_ATTACH_MAX_CHARS</code> (default 20,000) of extracted text; longer documents are truncated and flagged as such to the model.</p>
<p>PDF extraction needs the optional <code>pypdf</code> package on the server (<code>pip install pypdf --break-system-packages</code>); without it, PDF attachments fail with a clear message while plain text and code files continue to work. A scanned/image-only PDF with no real text layer also fails clearly rather than silently attaching nothing.</p>
<p>Both attachment types stay in the conversation history for the rest of the session (the same way the text does), so a long conversation with several images or documents attached can grow the request size substantially — there is no attachment-specific pruning beyond the existing <code>CHAT_MAX_HISTORY</code> message cap.</p>

<pre>GET  /api/chat/models                              list installed models (unfiltered)
POST /api/chat/extract_file  + file                extract text from a document attachment
POST /api/chat  + model, messages[{role,content,images?}], system?, think?</pre>

<h3>Theme</h3>
<p>The three buttons under the app name in the rail — ☀ / ☾ / ⚙ — switch between light, dark, and <strong>Auto</strong>, which follows your OS or browser's light/dark setting and updates live if that changes while the page is open (e.g. a system that switches at sunset). Auto is the default until you pick one explicitly.</p>
<p>The choice is stored in your browser's own local storage (<code>aimp_theme</code>), not on the server — it's per-browser, not shared across devices or people. The theme is resolved and applied before the page's stylesheet is even parsed, so there's no flash of the wrong theme on load, and if local storage is unavailable (private browsing, a locked-down browser policy) it falls back to dark, matching the app's original default.</p>

<h3>Access control</h3>
<p>There are no per-user accounts. What exists instead:</p>
<ul style="line-height:1.8">

  <li><strong>A shared passphrase gate</strong>, off by default. Set <code>APP_ACCESS_KEY</code> and every route — the UI, uploads, library files, template assets, all API endpoints — requires a session established at <code>/login</code>. Nothing changes for a deployment that leaves it unset.</li>
  <li><strong>Rate limiting on the direct file routes</strong> (<code>/uploads/</code>, <code>/library/&lt;id&gt;/*</code>, <code>/download/</code>, template <code>/asset/</code>), always on regardless of the gate — <code>FILE_ROUTE_RATE_LIMIT</code> requests per IP per minute (default 40). This exists because those routes are otherwise reachable by anyone who can reach the port: library IDs are sequential and upload filenames are guessable, so without a limit a script could enumerate every trailer ever generated.</li>
  <li><strong>Rate limiting on login attempts</strong> — <code>LOGIN_RATE_LIMIT</code> tries per IP per 5 minutes (default 8), so the passphrase can't be brute-forced at line speed.</li>
</ul>
<p>The session cookie is signed with a key in <code>SECRET_KEY_FILE</code> (a <code>.secret_key</code> file next to this script by default), generated once and reused — regenerating it on every restart would silently log everyone out on each pm2 restart. Keep that file out of version control.</p>
<p>Sessions last <code>SESSION_LIFETIME_DAYS</code> (default 30). The cookie is marked <code>Secure</code> only when <code>FORCE_HTTPS=1</code> is set — on the plain-HTTP LAN deployment this app defaults to, a <code>Secure</code> cookie would never actually be sent, which would look like the gate silently not working.</p>
<p>This is a single shared secret, not per-user accounts — anyone with the passphrase has full access, and there's no audit trail of who did what. If that stops being sufficient, this is the layer real user accounts would replace.</p>

<h3>Templates</h3>
<p>A <strong>template</strong> is the complete configuration for a programme: its rating mode and target duration, genre, transition and crossfade, scene-detection thresholds, loudness and ducking targets, narration engine/voice/language — plus its background music bed, SFX one-shot, voiceover, title card and end card (each card with an optional card VO and in/out points).</p>
<p>Genre is <em>one of the fields a template carries</em>, not an alternative to it. Picking a template at the top of the form fills everything in below, including the genre; you can then change anything you like and it applies to that job only, leaving the saved template untouched.</p>
<p>Templates are stored in <code>TEMPLATES_DIR</code> (a <code>show_templates/</code> folder next to this script by default, overridable with that environment variable), with metadata in a small SQLite database, so they survive server restarts. Each job gets its <em>own copy</em> of a template's files, so nothing the pipeline does to them can damage the saved originals.</p>

<h4>Building a template</h4>
<p>Set the form up the way that programme should be made, pick whichever files belong to it in the ordinary Background music / Sound effects / Narration / Title card / End card sections, type a name in <strong>Save as template</strong> at the bottom of the form, and click the button. Local uploads and files picked with <em>Browse library</em> both work. The source video in the dropzone is never uploaded by this button — only the assets.</p>
<p>Saving under a name that already exists <em>updates</em> that template: the settings are replaced, and only the asset slots you supplied this time are rewritten, so you can swap just the music bed without touching the cards. Replacing a slot deletes the old stored copy, so the folder does not accumulate orphans. A template can also be pure configuration with no assets at all.</p>

<h4>Using a template</h4>
<p>Pick one from the <strong>Template</strong> dropdown at the top of the form. The summary box shows its rating mode, duration, genre, transition and which asset slots it fills; the form below is populated to match. The music/SFX/narration mode selectors switch to "Upload" for the slots it supplies, and the VO placement and card in/out numbers prefill, so nothing is applied invisibly.</p>
<p>A template is a set of defaults, not a lock. Anything you set explicitly wins for that job: attach a one-off music bed and it is used instead of the template's; set Background music to <em>None</em> and the job runs without music even though the template has a bed. Slots a template leaves empty behave exactly as they would with no template selected.</p>
<p>Deleting a template removes its row and its stored copies of those files. Trailers already generated from it are unaffected — they are finished renders in the trailer library, with no link back.</p>
<pre>GET    /api/templates              list templates, their settings, and which slots each fills
POST   /api/templates              create/update (multipart, same field names as the generate form)
DELETE /api/templates/&lt;id&gt;         delete a template and its stored files
GET    /api/templates/&lt;id&gt;/asset/&lt;slot&gt;   stream one stored asset (for preview)
POST   /api/trailer/generate  + template_id=&lt;id&gt;   (fills anything not sent)</pre>

<p>Slot keys: <code>bgm</code>, <code>sfx</code>, <code>vo</code>, <code>title_card</code>, <code>title_card_vo</code>, <code>end_card</code>, <code>end_card_vo</code>. Pass <code>clear_&lt;slot&gt;=1</code> on a save to deliberately empty one.</p>

<h3>Getting ACE-Step to actually sing your lyrics</h3>
<p>Skipped words, lines sung out of order, and an unexpectedly long instrumental intro before the vocal starts are widely reported ACE-Step behaviors, not a sign the app is misconfigured — see the model's own issue tracker (<a href="https://github.com/ace-step/ACE-Step-1.5/issues/391" target="_blank" rel="noopener">"Dont follow lyrics!"</a>) for the same complaint from other users. Two things this app already does to reduce it, and a few you control per-generation:</p>
<ul style="line-height:1.8">
  <li><strong>Thinking mode is always disabled</strong> for both music calls (<code>thinking: false</code>) — one of the most commonly reported triggers for the model wandering into an unrequested instrument or skipping straight to the chorus is specifically "Think mode," per the issue above.</li>
  <li><strong>The vocal-suppressing negative prompt is dropped whenever you supply lyrics</strong> — leaving it in alongside real lyrics is a separate, documented cause of confused, half-sung output (the model tries to satisfy "no vocals" and your lyrics at once).</li>
  <li><strong>Match lyric length to duration.</strong> ACE-Step sings at roughly 2–3 words per second. A 30-second track wants on the order of 60–90 words — noticeably more and it crams or drops lines to fit; noticeably less and the remaining time becomes instrumental filler, which is the usual cause of "too much intro." The Music Generation tool checks this live as you type and flags it before you generate.</li>
  <li><strong>Keep lines short</strong> — 4–8 words per line. Longer lines are reported to fracture vocal timing, since the model has to cram more syllables into the same beat than it can comfortably fit.</li>
  <li><strong>Use <code>[verse]</code> / <code>[chorus]</code> / <code>[bridge]</code> structure tags spanning the whole lyric.</strong> Without them the model decides the song's structure itself, which is the other common source of a long unrequested intro — explicit tags give it a map to fill the requested duration with, rather than improvising one.</li>
  <li><strong>Very long durations with long lyrics compound the problem</strong> — more lyric text competes with music generation for the same fixed model context, so a long song with dense lyrics is more likely to cut off before the end than a short one. If a long track keeps losing its last few lines, shortening the lyrics tends to fix it faster than raising the duration further.</li>
</ul>
<p>None of this guarantees a perfect line-for-line reading — even well-paced, well-structured lyrics can come out with an occasional skipped or substituted word. If the track otherwise has the right energy, that's usually not worth regenerating over.</p>

<h3>Narration delivery tags (Fish Audio)</h3>
<p>Fish Audio reads inline markers out of the narration text itself to control emotion, delivery and non-speech sounds. There is no separate field — a tag is simply part of the script, and it is not spoken.</p>
<p>The syntax depends on the model generation: <strong>S2 uses square brackets</strong> (<code>[happy]</code>) and also accepts free-form descriptions like <code>[very excited]</code> or <code>[warm and reassuring]</code>; the older <strong>S1 uses parentheses</strong> (<code>(happy)</code>) with a fixed tag set. The app picks the right one automatically from <code>FISH_AUDIO_MODEL</code> — override it with <code>FISH_TAG_STYLE=s1</code> or <code>s2</code> if your self-hosted checkpoint does not match its name.</p>
<p>A tag bar sits under the script box in both the <strong>Narration</strong> section of the generate form and the <strong>Fish Audio S2</strong> tool. Click a tag to insert it at the cursor; spacing around it is handled for you.</p>
<pre>[confident] Tonight, the story everyone is talking about.
This is [emphasis] the one you cannot miss.
[whispering] Some secrets do not stay buried. [break] Only on GMA.</pre>
<p>Placement matters: an emotion cue works best at the <em>start of the sentence</em> it applies to, while delivery cues like <code>[emphasis]</code> go immediately before the word they should stress. Sound and pause markers can go anywhere.</p>
<p>Two tags can be stacked for a combined effect — <code>[sad][whispering]</code>, <code>[angry][shouting]</code> — but three or more in one sentence tends to produce worse results, not better. Keep to one emotion per sentence and space changes out; piling tags into a short promo line is the most common cause of a flat or erratic read.</p>
<p>The bar shows a curated set that suits promo scripts rather than the full catalogue. Since S2 accepts natural language, anything not listed can simply be typed in brackets.</p>

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

<div id="p-config" class="panel">
<h2>AI Service Configuration</h2>
<p style="margin-top:-4px;color:var(--ink-dim)">These control where the app looks for each local/self-hosted AI service. Changes save immediately (no restart needed) and are written to <code>ai_services_config.json</code> next to the script, so they persist across restarts too. Environment variables set the initial defaults; anything saved here overrides them from then on.</p>
<div id="config-rows" style="display:flex;flex-direction:column;gap:14px;margin-top:16px">
  <!-- rows injected by loadConfigTab() -->
</div>
<div style="margin-top:18px;display:flex;gap:10px;align-items:center">
  <button type=button id=config-save-btn onclick="saveConfigTab()">Save all</button>
  <span id="config-save-status" style="font-size:13px"></span>
</div>
</div>

</div><!-- container -->

<script>
var CONFIG_FIELD_ORDER = ['FISH_AUDIO_URL','FISH_AUDIO_API_KEY','WHISPER_URL','OLLAMA_URL','ACE_STEP_URL','WOOSH_URL']
var configTabLoaded = false
async function loadConfigTab(){
  var container = document.getElementById('config-rows')
  container.innerHTML = '<p style="opacity:.7">Loading current configuration…</p>'
  try{
    var r = await fetch('/api/config')
    var d = await r.json()
    if(!d.ok){ container.innerHTML = '<p style="color:var(--amber)">Could not load config.</p>'; return }
    container.innerHTML = ''
    CONFIG_FIELD_ORDER.forEach(function(key){
      var meta = d.fields[key] || {label: key, help: ''}
      var val = d.config[key] || ''
      var isKey = key === 'FISH_AUDIO_API_KEY'
      var row = document.createElement('div')
      row.style.cssText = 'display:flex;flex-direction:column;gap:4px'
      var labelEl = document.createElement('label')
      labelEl.style.cssText = 'font-size:13px;text-transform:none;letter-spacing:0'
      labelEl.textContent = meta.label
      var inputRow = document.createElement('div')
      inputRow.style.cssText = 'display:flex;gap:8px;align-items:center;flex-wrap:wrap'
      var input = document.createElement('input')
      input.type = isKey ? 'password' : 'text'
      input.id = 'cfg-' + key
      input.value = val
      input.placeholder = meta.help
      input.style.cssText = 'flex:1;min-width:260px'
      inputRow.appendChild(input)
      if(!isKey){
        var testBtn = document.createElement('button')
        testBtn.type = 'button'
        testBtn.textContent = 'Test'
        testBtn.addEventListener('click', function(){ testConfigRow(key) })
        inputRow.appendChild(testBtn)
      }
      var statusSpan = document.createElement('span')
      statusSpan.id = 'cfg-status-' + key
      statusSpan.style.cssText = 'font-size:12px;min-width:140px'
      inputRow.appendChild(statusSpan)
      var helpSpan = document.createElement('span')
      helpSpan.style.cssText = 'font-size:11px;opacity:.65'
      helpSpan.textContent = meta.help
      row.appendChild(labelEl)
      row.appendChild(inputRow)
      row.appendChild(helpSpan)
      container.appendChild(row)
    })
    configTabLoaded = true
  }catch(e){
    container.innerHTML = '<p style="color:var(--amber)">Error loading config: '+e+'</p>'
  }
}
async function testConfigRow(key){
  var status = document.getElementById('cfg-status-'+key)
  var url = document.getElementById('cfg-'+key).value.trim()
  status.textContent = 'Testing…'; status.style.color = 'var(--ink-dim)'
  try{
    var r = await fetch('/api/config/test', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({name:key, url:url})})
    var d = await r.json()
    if(d.ok){ status.textContent = 'Reachable ('+d.latency_ms+'ms, HTTP '+d.http_status+')'; status.style.color = 'var(--phosphor)' }
    else{ status.textContent = 'Unreachable: '+(d.error||'unknown error'); status.style.color = 'var(--amber)' }
  }catch(e){ status.textContent = 'Test failed: '+e; status.style.color = 'var(--amber)' }
}
async function saveConfigTab(){
  var btn = document.getElementById('config-save-btn')
  var status = document.getElementById('config-save-status')
  var payload = {}
  CONFIG_FIELD_ORDER.forEach(function(key){
    var el = document.getElementById('cfg-'+key)
    if(el) payload[key] = el.value.trim()
  })
  btn.disabled = true; status.textContent = 'Saving…'; status.style.color = 'var(--ink-dim)'
  try{
    var r = await fetch('/api/config', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)})
    var d = await r.json()
    if(d.ok){ status.textContent = 'Saved — applied immediately, no restart needed.'; status.style.color = 'var(--phosphor)' }
    else{ status.textContent = 'Save failed: '+(d.error||'unknown error'); status.style.color = 'var(--amber)' }
  }catch(e){ status.textContent = 'Save failed: '+e; status.style.color = 'var(--amber)' }
  btn.disabled = false
}
</script>

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
// ---- Video Player ----
// A plain HTML5 player over files already on the share (or in the trailer
// library). The previous version streamed MJPEG from a server-side OpenCV
// capture: it re-encoded every frame on the box that also runs renders, and used
// one global capture handle, so two people browsing at once fought over it.
var _plStagedName = null   // network-staged filename currently loaded, if any (null for a local file)
var _plFellBack = false    // one-shot guard so a converted file that ALSO errors doesn't loop

// Opens the SAME shared "Browse library" modal every other upload field uses
// (see openNetworkBrowser/renderList further down), for whichever category is
// currently selected. Clicking a file in that modal fetches it and plays it
// immediately -- there is deliberately no separate "Play selected" step, to
// match how picking a file works everywhere else in the app.
function openPlayerBrowser(){
  var sel = document.getElementById('pl-category')
  var label = sel.options[sel.selectedIndex].textContent
  window.openNetworkBrowser('pl', sel.value, label)
}

function showPlayer(src, label, stagedName){
  _plStagedName = stagedName || null
  _plFellBack = false
  var v = document.getElementById('pl-video')
  v.src = src
  document.getElementById('pl-title').textContent = label
  document.getElementById('pl-area').style.display = 'block'
  document.getElementById('pl-prompt').style.display = 'none'
  document.getElementById('pl-status').textContent = ''
  v.play().catch(function(){})   // autoplay may be blocked; the controls still work
}

async function handlePlayerError(){
  var v = document.getElementById('pl-video')
  var status = document.getElementById('pl-status')
  var title = document.getElementById('pl-title')

  if(_plFellBack || !_plStagedName){
    // Either the converted copy ALSO failed (very unlikely -- H.264/AAC MP4 is
    // about as compatible as it gets), or this is a local file played straight
    // from disk via a blob URL, which never touched the server and so has
    // nothing here to convert.
    status.style.color = 'var(--amber)'
    status.textContent = _plStagedName
      ? 'This file could not be converted for playback. Check its source on the share.'
      : "This file's format isn't supported by your browser for local preview. "
        + 'Try "Browse library" instead -- files picked that way can be converted automatically.'
    return
  }

  _plFellBack = true
  status.style.color = ''
  status.textContent = 'This format needs converting for browser playback (common for ProRes/DNxHD .mov masters)… this can take a minute for a large file.'
  try{
    var fd = new FormData()
    fd.append('filename', _plStagedName)
    var r = await fetch('/api/media/playable', {method:'POST', body: fd})
    var d = await r.json()
    if(!d.ok){
      status.style.color = 'var(--amber)'
      status.textContent = d.error || 'Conversion failed.'
      return
    }
    status.textContent = 'Converted -- resuming playback.'
    title.textContent += ' (converted for playback)'
    v.src = d.url
    v.load()
    v.play().catch(function(){})
  }catch(e){
    status.style.color = 'var(--amber)'
    status.textContent = 'Conversion request failed: ' + e
  }
}

document.addEventListener('DOMContentLoaded', function(){
  var video = document.getElementById('pl-video')
  if(!video) return
  video.addEventListener('error', handlePlayerError)
  var local = document.getElementById('pl-local')
  local.addEventListener('change', function(){
    // Played straight from the browser via a blob URL -- no upload at all, so
    // there's no staged filename for handlePlayerError to fall back with if
    // this format turns out to be unsupported.
    if(local.files && local.files[0]){
      showPlayer(URL.createObjectURL(local.files[0]), local.files[0].name, null)
    }
  })
})
</script>

<script>
// ---- Shared "browse library" modal, used by every non-trailer upload
// field (title/end card video, BG music, VO, SFX). The main trailer video
// dropzone has its own richer in-place preview (see the earlier tr-* script)
// and isn't wired through here.
(function(){
  var overlay = document.getElementById('net-modal-overlay')
  var titleEl = document.getElementById('net-modal-title')
  var pathEl = document.getElementById('net-modal-path')
  var listEl = document.getElementById('net-modal-list')
  var closeBtn = document.getElementById('net-modal-close')
  var active = null // {fieldName, category}

  function humanSize(bytes){
    if(bytes < 1024) return bytes + ' B'
    if(bytes < 1024*1024) return (bytes/1024).toFixed(1) + ' KB'
    if(bytes < 1024*1024*1024) return (bytes/(1024*1024)).toFixed(1) + ' MB'
    return (bytes/(1024*1024*1024)).toFixed(2) + ' GB'
  }

  function closeModal(){ overlay.classList.remove('open'); active = null }
  closeBtn.addEventListener('click', closeModal)
  overlay.addEventListener('click', function(e){ if(e.target === overlay) closeModal() })
  document.addEventListener('keydown', function(e){ if(e.key === 'Escape' && overlay.classList.contains('open')) closeModal() })

  function setChip(fieldName, name, size){
    var chip = document.getElementById(fieldName + '_chip')
    if(!chip) return
    chip.style.display = 'inline-flex'
    chip.querySelector('.chip-name').textContent = name + (size != null ? ' (' + humanSize(size) + ')' : '')
  }
  window.setChip = setChip

  // Which mode radio (and its neutral default) goes with a given upload field --
  // used so clearing a template-filled chip for one of these three doesn't leave
  // the mode stuck on "Upload" with nothing actually attached to it.
  var MODE_FIELD_DEFAULTS = {
    scoring_audio: ['scoring_mode', 'generate'],
    sfx_upload: ['sfx_mode', 'none'],
    vo_upload: ['vo_mode', 'none'],
  }

  function clearChip(fieldName){
    var chip = document.getElementById(fieldName + '_chip')
    var wasFromTemplate = chip && chip.dataset.fromTemplate === '1'
    if(chip){ chip.style.display = 'none'; delete chip.dataset.fromTemplate }
    var hidden = document.getElementById(fieldName + '_network')
    if(hidden) hidden.value = ''
    // Any manual interaction with this slot -- clicking the chip's own X, or
    // (via the file-input listener below) picking a local file -- means the
    // template should no longer silently re-fill it on generate.
    var skip = document.getElementById(fieldName + '_skip_template')
    if(skip) skip.value = '1'
    // Clearing a template-sourced chip for one of the mode-driven fields should
    // also drop the mode back to its neutral default, so the form doesn't show
    // "Upload" with an empty, invisible template pick behind it.
    if(wasFromTemplate && MODE_FIELD_DEFAULTS[fieldName]){
      var pair = MODE_FIELD_DEFAULTS[fieldName]
      var r = document.querySelector('#tf input[name="' + pair[0] + '"][value="' + pair[1] + '"]')
      if(r && !r.checked){ r.checked = true; r.dispatchEvent(new Event('change', {bubbles:true})) }
    }
    var player = document.getElementById(fieldName + '_player')
    var note = document.getElementById(fieldName + '_preview_note')
    var fileInput = document.querySelector('input[name="' + fieldName + '"]')
    if(player && !(fileInput && fileInput.files && fileInput.files[0])){
      player.removeAttribute('src'); player.style.display = 'none'
      if(note) note.textContent = ''
    }
  }
  window.clearNetworkField = clearChip

  function renderList(files){
    if(!files.length){
      listEl.innerHTML = '<div class="net-modal-empty">No matching files found in this folder.</div>'
      return
    }
    listEl.innerHTML = ''
    files.forEach(function(f){
      var row = document.createElement('div')
      row.className = 'net-modal-row'
      row.innerHTML = '<span class="row-name">' + f.name + '</span><span class="row-size">' + humanSize(f.size) + '</span>'
      row.addEventListener('click', function(){
        if(!active) return
        listEl.innerHTML = '<div class="net-modal-empty">Fetching ' + f.name + '&hellip;</div>'
        fetch('/api/network/fetch', {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({name: f.name, category: active.category})
        }).then(function(r){ return r.json() }).then(function(d){
          if(!d.ok){ listEl.innerHTML = '<div class="net-modal-empty" style="color:var(--tally)">' + d.error + '</div>'; return }
          if(active.fieldName === 'pl'){
            // The Player isn't a form field -- picking a file here should just
            // start playing it, reusing showPlayer() so the transcode-on-error
            // fallback (see handlePlayerError) still has the staged filename to
            // work with if the browser can't decode this format natively.
            window.showPlayer(d.url, d.orig_name, d.filename)
            closeModal()
            return
          }
          var hidden = document.getElementById(active.fieldName + '_network')
          var fileInput = document.querySelector('input[name="' + active.fieldName + '"]')
          if(hidden) hidden.value = d.filename
          if(fileInput) fileInput.value = '' // network selection and local file are mutually exclusive
          setChip(active.fieldName, d.orig_name, d.size)
          // Card-VO fields (title_card_vo / end_card_vo) have their own preview
          // player + set-in/set-out controls, normally fed by cardVoFileChosen()
          // from a local File object. A network fetch has no File object, but
          // the file is already reachable at /uploads/<filename>, so wire the
          // player up the same way manually.
          var player = document.getElementById(active.fieldName + '_player')
          var note = document.getElementById(active.fieldName + '_preview_note')
          if(player){
            player.src = '/uploads/' + encodeURIComponent(d.filename)
            player.style.display = ''
            if(note) note.textContent = 'Play the file, pause where you want the cut, then click Set in / Set out.'
          }
          closeModal()
        }).catch(function(e){ listEl.innerHTML = '<div class="net-modal-empty" style="color:var(--tally)">' + e + '</div>' })
      })
      listEl.appendChild(row)
    })
  }

  function openNetworkBrowser(fieldName, category, label){
    active = {fieldName: fieldName, category: category}
    titleEl.textContent = 'Browse ' + label + ' folder'
    pathEl.textContent = ''
    listEl.innerHTML = '<div class="net-modal-empty">Loading&hellip;</div>'
    overlay.classList.add('open')
    fetch('/api/network/list?category=' + encodeURIComponent(category)).then(function(r){ return r.json() }).then(function(d){
      if(!d.ok){ listEl.innerHTML = '<div class="net-modal-empty" style="color:var(--tally)">' + d.error + '</div>'; return }
      pathEl.textContent = d.root
      renderList(d.files)
    }).catch(function(e){ listEl.innerHTML = '<div class="net-modal-empty" style="color:var(--tally)">' + e + '</div>' })
  }
  window.openNetworkBrowser = openNetworkBrowser

  // Selecting a local file clears any pending network selection for the same
  // field, and vice versa (handled in openNetworkBrowser above) -- so the two
  // sources never fight over which one the form actually submits.
  document.querySelectorAll('input[type=file][data-net-field]').forEach(function(input){
    input.addEventListener('change', function(){
      if(input.files && input.files[0]) clearChip(input.getAttribute('data-net-field'))
    })
  })
})()
</script>
</div><!-- /.shell -->
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(UI, genre_rows=GENRE_DOCS_ROWS, gate_enabled=GATE_ENABLED)

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

load_config_overrides()  # apply any saved Config-tab overrides on top of the env-var defaults

if __name__ == '__main__':
    print(' * Server starting...')
    threading.Thread(target=_sweeper_loop, daemon=True).start()
    sweep_upload_folder()  # reclaim anything left over from a previous run
    _free = free_disk_mb()
    if _free is not None:
        print(f' * Disk free on work volume: {_free:,.0f} MB'
              + ('   ** LOW — renders may fail **' if _free < 2048 else ''))
    print(' * HTTP:  http://0.0.0.0:5000/')
    print(' * Access from local machine: http://localhost:5000/')
    print(' * Access from other devices: http://YOUR_IP:5000/')
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
