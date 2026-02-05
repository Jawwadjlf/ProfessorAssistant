"""
Professor Assistant (Windows) — Production-ready single-file

Features:
- Wake-word via openWakeWord (.onnx) — supports multiple wakewords for multi-user profiles
- VAD auto-stop recording (faster than fixed-length recording)
- STT: OpenAI Whisper (default) + optional local faster-whisper
- Brain: OpenAI ChatGPT (default) + optional Gemini (tray switch)
- Real Urdu output (Urdu script) + Urdu TTS voice (Edge-TTS ur-PK)
- Smart-home commands: Home Assistant + optional MQTT
- Always-on local conversation memory (per profile) with auto-summary
- Tray icon: enable/disable listening, switch brain, open setup, startup install/remove, exit
- Hotkey push-to-talk backup (Ctrl+Alt+Space by default)
- Startup installer (creates a .cmd in Windows Startup folder)

Folder layout (recommended):
  Project/
    assistant.py
    models/
      professor.onnx
      jawad.onnx      (optional)
      salma.onnx      (optional)

Build EXE (GitHub Actions windows-latest):
  pyinstaller --noconsole --onefile --name ProfessorAssistant assistant.py

IMPORTANT:
- Do NOT hardcode keys. On first run, the app shows a Setup window and saves config to:
  %APPDATA%\\ProfessorAssistant\\config.json
"""

import os
import sys
import io
import re
import time
import json
import queue
import asyncio
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import numpy as np
import sounddevice as sd
import soundfile as sf
import webrtcvad
import requests

from openwakeword import Model as WakeModel

import edge_tts
import pystray
from pystray import MenuItem as item
from PIL import Image, ImageDraw

# Optional deps (graceful if missing)
try:
    import keyboard  # hotkey
except Exception:
    keyboard = None

try:
    import paho.mqtt.client as mqtt
except Exception:
    mqtt = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    import google.generativeai as genai
except Exception:
    genai = None

try:
    from faster_whisper import WhisperModel  # optional local STT
except Exception:
    WhisperModel = None


# =========================
# PATHS
# =========================

def is_frozen() -> bool:
    return getattr(sys, "frozen", False)

BASE_DIR = Path(sys.executable).resolve().parent if is_frozen() else Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

APPDATA = Path(os.getenv("APPDATA", str(BASE_DIR)))
CFG_DIR = APPDATA / "ProfessorAssistant"
CFG_DIR.mkdir(parents=True, exist_ok=True)
CFG_PATH = CFG_DIR / "config.json"

MEM_DIR = CFG_DIR / "memory"
MEM_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = CFG_DIR / "assistant.log"


# =========================
# LOGGING
# =========================

def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    try:
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


# =========================
# DEFAULT CONFIG
# =========================

DEFAULT_CFG = {
    # LLM provider: "openai" (default) or "gemini"
    "llm_provider": "openai",

    # Keys (filled via setup wizard)
    "openai_api_key": "",
    "openai_chat_model": "gpt-4o-mini",
    "gemini_api_key": "",
    "gemini_model": "gemini-1.5-flash",

    # Audio/Wake/VAD
    "wakeword_threshold": 0.55,
    "mic_sample_rate": 16000,
    "mic_blocksize": 512,
    "vad_mode": 2,                 # 0..3
    "max_record_seconds": 10,
    "end_silence_ms": 900,
    "min_speech_ms": 350,
    "followup_window_seconds": 6,

    # Wakeword label -> profile name
    # (label MUST match your ONNX model label; common case is filename)
    "wakeword_map": {
        "professor": "default"
        # "jawad": "jawad",
        # "salma": "salma"
    },

    # TTS voices (Edge TTS)
    "tts_en_voice": "en-US-JennyNeural",
    "tts_ur_voice": "ur-PK-UzmaNeural",

    # Smart home: Home Assistant
    "ha_url": "http://homeassistant.local:8123",
    "ha_token": "",

    # Optional MQTT
    "mqtt_host": "",
    "mqtt_port": 1883,
    "mqtt_user": "",
    "mqtt_pass": "",

    # Devices mapping: friendly name -> settings
    # Each device can have HA entity and/or MQTT topic.
    "devices": {
        # "lights": {"ha_entity": "light.living_room"},
        # "lab lights": {"ha_entity": "light.lab_3"},
        # "projector": {"mqtt_topic": "lab/projector/power", "mqtt_on_payload": "ON", "mqtt_off_payload": "OFF"}
    },

    # Hotkey (push to talk)
    "push_to_talk_hotkey": "ctrl+alt+space",

    # STT
    # mode: "openai" (default) or "local"
    "stt_mode": "openai",
    "local_stt_model": "small",     # faster-whisper model name
}


def load_cfg() -> dict:
    if CFG_PATH.exists():
        try:
            user_cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))
        except Exception:
            user_cfg = {}
    else:
        user_cfg = {}
    merged = dict(DEFAULT_CFG)
    merged.update({k: v for k, v in user_cfg.items() if v is not None})
    # deep-merge dict fields
    for k in ("wakeword_map", "devices"):
        merged[k] = dict(DEFAULT_CFG.get(k, {}))
        merged[k].update(user_cfg.get(k, {}) or {})
    return merged


def save_cfg(cfg: dict):
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    CFG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


# =========================
# SETUP WIZARD (Tkinter)
# =========================

def setup_wizard():
    import tkinter as tk
    from tkinter import messagebox

    cfg = load_cfg()

    root = tk.Tk()
    root.title("Professor Assistant — Setup")
    root.geometry("680x720")

    def label(text):
        tk.Label(root, text=text, font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=14, pady=(12, 4))

    def entry(key, show=None):
        e = tk.Entry(root, width=90, show=show)
        e.insert(0, cfg.get(key, ""))
        e.pack(padx=14)
        return e

    label("OpenAI API Key (required for ChatGPT + Whisper STT):")
    e_openai = entry("openai_api_key", show="*")

    label("OpenAI Chat model (default gpt-4o-mini):")
    e_chat_model = entry("openai_chat_model")

    label("Gemini API Key (optional, only if you want Gemini switch):")
    e_gemini = entry("gemini_api_key", show="*")

    label("Home Assistant URL:")
    e_ha_url = entry("ha_url")

    label("Home Assistant Long-Lived Access Token:")
    e_ha_tok = entry("ha_token", show="*")

    label("Optional MQTT Host (leave empty if not using MQTT):")
    e_mqtt_host = entry("mqtt_host")

    label("MQTT Port (default 1883):")
    e_mqtt_port = entry("mqtt_port")

    label("MQTT User:")
    e_mqtt_user = entry("mqtt_user")

    label("MQTT Pass:")
    e_mqtt_pass = entry("mqtt_pass", show="*")

    label("Devices mapping (one per line):  name=HA_ENTITY   OR   name=MQTT_TOPIC")
    tk.Label(root, text="Examples:\n  lights=light.living_room\n  lab lights=light.lab_3\n  projector=lab/projector/power",
             justify="left").pack(anchor="w", padx=14, pady=(0, 6))

    txt_devices = tk.Text(root, height=10, width=90)
    txt_devices.pack(padx=14)

    # Pre-fill device lines
    device_lines = []
    for name, d in (cfg.get("devices") or {}).items():
        if isinstance(d, dict):
            if d.get("ha_entity"):
                device_lines.append(f"{name}={d['ha_entity']}")
            elif d.get("mqtt_topic"):
                device_lines.append(f"{name}={d['mqtt_topic']}")
        elif isinstance(d, str):
            device_lines.append(f"{name}={d}")
    txt_devices.insert("1.0", "\n".join(device_lines))

    label("Wakewords (one per line): wakeword_label=profile_name")
    tk.Label(root, text="Models must exist at: models/<wakeword_label>.onnx\nExample:\n  professor=default\n  jawad=jawad",
             justify="left").pack(anchor="w", padx=14, pady=(0, 6))

    txt_wake = tk.Text(root, height=6, width=90)
    txt_wake.pack(padx=14)

    wake_lines = [f"{k}={v}" for k, v in (cfg.get("wakeword_map") or {}).items()]
    txt_wake.insert("1.0", "\n".join(wake_lines))

    label("Push-to-talk Hotkey (optional; requires 'keyboard' package):")
    e_hotkey = entry("push_to_talk_hotkey")

    label("STT mode: openai (default) OR local (requires faster-whisper):")
    e_stt_mode = entry("stt_mode")

    label("Local STT model name (if using local STT):")
    e_local_stt_model = entry("local_stt_model")

    def on_save():
        cfg["openai_api_key"] = e_openai.get().strip()
        cfg["openai_chat_model"] = e_chat_model.get().strip() or "gpt-4o-mini"
        cfg["gemini_api_key"] = e_gemini.get().strip()
        cfg["ha_url"] = e_ha_url.get().strip()
        cfg["ha_token"] = e_ha_tok.get().strip()

        cfg["mqtt_host"] = e_mqtt_host.get().strip()
        try:
            cfg["mqtt_port"] = int(e_mqtt_port.get().strip() or "1883")
        except Exception:
            cfg["mqtt_port"] = 1883
        cfg["mqtt_user"] = e_mqtt_user.get().strip()
        cfg["mqtt_pass"] = e_mqtt_pass.get().strip()

        cfg["push_to_talk_hotkey"] = e_hotkey.get().strip() or "ctrl+alt+space"
        cfg["stt_mode"] = (e_stt_mode.get().strip() or "openai").lower()
        cfg["local_stt_model"] = e_local_stt_model.get().strip() or "small"

        # Devices
        devices = {}
        for line in txt_devices.get("1.0", "end").splitlines():
            line = line.strip()
            if not line or "=" not in line:
                continue
            name, val = line.split("=", 1)
            name = name.strip().lower()
            val = val.strip()
            if val.startswith(("light.", "switch.", "fan.", "scene.", "script.", "media_player.", "climate.")):
                devices[name] = {"ha_entity": val}
            else:
                devices[name] = {"mqtt_topic": val, "mqtt_on_payload": "ON", "mqtt_off_payload": "OFF"}
        cfg["devices"] = devices

        # Wake map
        wake_map = {}
        for line in txt_wake.get("1.0", "end").splitlines():
            line = line.strip()
            if not line or "=" not in line:
                continue
            w, p = line.split("=", 1)
            w = w.strip().lower()
            p = p.strip() or w
            wake_map[w] = p
        if not wake_map:
            wake_map = {"professor": "default"}
        cfg["wakeword_map"] = wake_map

        # Validate required
        if not cfg["openai_api_key"]:
            messagebox.showerror("Missing", "OpenAI API Key is required (for ChatGPT + Whisper STT).")
            return

        save_cfg(cfg)
        messagebox.showinfo("Saved", f"Saved config to:\n{CFG_PATH}\n\nRestart the app.")
        root.destroy()

    tk.Button(root, text="Save", command=on_save, height=2, width=16).pack(pady=16)
    root.mainloop()


# =========================
# STATE
# =========================

@dataclass
class AppState:
    listening_enabled: bool = True
    stop_app: bool = False
    active_profile: str = "default"
    in_followup: bool = False
    followup_deadline: float = 0.0
    llm_provider: str = "openai"  # openai default


STATE = AppState()

wake_event = threading.Event()
wake_label_lock = threading.Lock()
wake_label: Optional[str] = None

recording_active = threading.Event()
wake_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=200)
record_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=400)


# =========================
# TEXT / LANGUAGE UTILS
# =========================

URDU_RANGE = re.compile(r"[\u0600-\u06FF]")

def is_urdu(s: str) -> bool:
    return bool(URDU_RANGE.search(s or ""))


# =========================
# STARTUP (Windows)
# =========================

def startup_folder() -> Path:
    return Path(os.getenv("APPDATA", "")) / r"Microsoft\Windows\Start Menu\Programs\Startup"

def startup_cmd_path() -> Path:
    return startup_folder() / "ProfessorAssistantStartup.cmd"

def install_startup():
    sfolder = startup_folder()
    sfolder.mkdir(parents=True, exist_ok=True)
    target = startup_cmd_path()

    if is_frozen():
        exe = Path(sys.executable).resolve()
        content = f'@echo off\nstart "" "{exe}"\n'
    else:
        py = Path(sys.executable).resolve()
        venv_act = BASE_DIR / "venv" / "Scripts" / "activate.bat"
        if venv_act.exists():
            content = f'@echo off\ncd /d "{BASE_DIR}"\ncall "{venv_act}"\npython "{BASE_DIR / "assistant.py"}"\n'
        else:
            content = f'@echo off\ncd /d "{BASE_DIR}"\n"{py}" "{BASE_DIR / "assistant.py"}"\n'

    target.write_text(content, encoding="utf-8")

def remove_startup():
    try:
        startup_cmd_path().unlink(missing_ok=True)
    except Exception:
        pass

def startup_installed() -> bool:
    return startup_cmd_path().exists()


# =========================
# LOAD CONFIG + INIT CLIENTS
# =========================

CFG = load_cfg()

if not CFG.get("openai_api_key"):
    setup_wizard()
    CFG = load_cfg()

STATE.llm_provider = (CFG.get("llm_provider") or "openai").lower().strip() or "openai"

MIC_SAMPLE_RATE = int(CFG.get("mic_sample_rate", 16000))
MIC_BLOCKSIZE = int(CFG.get("mic_blocksize", 512))
WAKEWORD_THRESHOLD = float(CFG.get("wakeword_threshold", 0.55))

VAD_MODE = int(CFG.get("vad_mode", 2))
MAX_RECORD_SECONDS = float(CFG.get("max_record_seconds", 10))
END_SILENCE_MS = int(CFG.get("end_silence_ms", 900))
MIN_SPEECH_MS = int(CFG.get("min_speech_ms", 350))
FOLLOWUP_WINDOW_SECONDS = int(CFG.get("followup_window_seconds", 6))

TTS_EN_VOICE = CFG.get("tts_en_voice", "en-US-JennyNeural")
TTS_UR_VOICE = CFG.get("tts_ur_voice", "ur-PK-UzmaNeural")
TTS_OUTPUT_FORMAT = "riff-24khz-16bit-mono-pcm"

WAKEWORD_MAP: Dict[str, str] = (CFG.get("wakeword_map") or {"professor": "default"})
DEVICES: Dict[str, dict] = (CFG.get("devices") or {})

HA_URL = (CFG.get("ha_url") or "").strip()
HA_TOKEN = (CFG.get("ha_token") or "").strip()

MQTT_HOST = (CFG.get("mqtt_host") or "").strip()
MQTT_PORT = int(CFG.get("mqtt_port") or 1883)
MQTT_USER = (CFG.get("mqtt_user") or "").strip()
MQTT_PASS = (CFG.get("mqtt_pass") or "").strip()

PUSH_TO_TALK_HOTKEY = (CFG.get("push_to_talk_hotkey") or "ctrl+alt+space").strip()

STT_MODE = (CFG.get("stt_mode") or "openai").lower().strip()
LOCAL_STT_MODEL_NAME = (CFG.get("local_stt_model") or "small").strip()

OPENAI_CHAT_MODEL = (CFG.get("openai_chat_model") or "gpt-4o-mini").strip()
OPENAI_API_KEY = (CFG.get("openai_api_key") or "").strip()

GEMINI_API_KEY = (CFG.get("gemini_api_key") or "").strip()
GEMINI_MODEL_NAME = (CFG.get("gemini_model") or "gemini-1.5-flash").strip()

openai_client = None
if OpenAI and OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

gemini_model = None
if genai and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    except Exception:
        gemini_model = None

vad = webrtcvad.Vad(VAD_MODE)

local_whisper = None
if STT_MODE == "local" and WhisperModel is not None:
    try:
        local_whisper = WhisperModel(LOCAL_STT_MODEL_NAME, device="cpu", compute_type="int8")
    except Exception:
        local_whisper = None
        STT_MODE = "openai"


mqtt_client = None
if mqtt and MQTT_HOST:
    try:
        mqtt_client = mqtt.Client()
        if MQTT_USER:
            mqtt_client.username_pw_set(MQTT_USER, MQTT_PASS)
        mqtt_client.connect(MQTT_HOST, MQTT_PORT, 60)
        mqtt_client.loop_start()
    except Exception:
        mqtt_client = None


# =========================
# WAKEWORD MODEL DISCOVERY
# =========================

def discover_wake_models() -> List[str]:
    MODELS_DIR.mkdir(exist_ok=True)
    paths = []
    for label in WAKEWORD_MAP.keys():
        p = MODELS_DIR / f"{label}.onnx"
        if p.exists():
            paths.append(str(p))
    if not paths:
        # load all onnx
        paths = [str(p) for p in MODELS_DIR.glob("*.onnx")]
    if not paths:
        raise RuntimeError(f"No wakeword models found. Put .onnx files in: {MODELS_DIR}")
    return paths

wake_model = WakeModel(wakeword_models=discover_wake_models())


# =========================
# AUDIO PLAYBACK + TTS
# =========================

def play_beep():
    duration = 0.12
    t = np.linspace(0, duration, int(MIC_SAMPLE_RATE * duration), False)
    tone = 0.25 * np.sin(2 * np.pi * 950 * t)
    sd.play(tone.astype(np.float32), MIC_SAMPLE_RATE)
    sd.wait()

def play_wav_bytes(wav_bytes: bytes):
    bio = io.BytesIO(wav_bytes)
    data, sr = sf.read(bio, dtype="float32")
    sd.play(data, sr)
    sd.wait()

def choose_tts_voice(text: str) -> str:
    return TTS_UR_VOICE if is_urdu(text) else TTS_EN_VOICE

async def tts_wav_bytes(text: str) -> bytes:
    voice = choose_tts_voice(text)
    tmp = CFG_DIR / f"_tts_{int(time.time()*1000)}.wav"
    try:
        communicate = edge_tts.Communicate(text=text, voice=voice, output_format=TTS_OUTPUT_FORMAT)
        await communicate.save(str(tmp))
        return tmp.read_bytes()
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


# =========================
# MEMORY (per profile)
# =========================

def mem_paths(profile: str) -> Tuple[Path, Path]:
    return MEM_DIR / f"{profile}.jsonl", MEM_DIR / f"{profile}.summary.txt"

def load_summary(profile: str) -> str:
    _, sp = mem_paths(profile)
    if sp.exists():
        return sp.read_text(encoding="utf-8", errors="ignore").strip()
    return ""

def load_recent_turns(profile: str, n: int = 8) -> List[dict]:
    jp, _ = mem_paths(profile)
    if not jp.exists():
        return []
    lines = jp.read_text(encoding="utf-8", errors="ignore").splitlines()
    out = []
    for line in lines[-n:]:
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out

def append_turn(profile: str, user: str, assistant: str):
    jp, _ = mem_paths(profile)
    rec = {"ts": int(time.time()), "user": user, "assistant": assistant}
    with jp.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def memory_context(profile: str) -> str:
    summary = load_summary(profile)
    recent = load_recent_turns(profile, n=8)
    parts = []
    if summary:
        parts.append("Memory summary:\n" + summary)
    if recent:
        parts.append("Recent turns:\n" + "\n".join(
            [f"User: {x.get('user','')}\nAssistant: {x.get('assistant','')}" for x in recent]
        ))
    return "\n\n".join(parts).strip()

def maybe_summarize(profile: str):
    jp, sp = mem_paths(profile)
    if not jp.exists():
        return
    lines = jp.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(lines) < 30:
        return

    old = lines[:-10]
    keep = lines[-10:]

    old_text = []
    for line in old:
        try:
            j = json.loads(line)
            old_text.append(f"User: {j.get('user','')}\nAssistant: {j.get('assistant','')}")
        except Exception:
            continue

    prompt = (
        "Summarize this conversation history into stable long-term memory notes "
        "for a voice assistant. Keep it short, factual, and useful. Keep Urdu in Urdu script.\n\n"
        + "\n\n".join(old_text)
    )
    summary = run_llm(profile, prompt, for_summary=True)
    if summary:
        sp.write_text(summary.strip(), encoding="utf-8")
        jp.write_text("\n".join(keep) + "\n", encoding="utf-8")


# =========================
# SMART HOME: HA + MQTT
# =========================

def ha_call(domain: str, service: str, data: dict) -> bool:
    if not HA_URL or not HA_TOKEN:
        return False
    url = f"{HA_URL.rstrip('/')}/api/services/{domain}/{service}"
    headers = {"Authorization": f"Bearer {HA_TOKEN}", "Content-Type": "application/json"}
    try:
        r = requests.post(url, headers=headers, json=data, timeout=4)
        return 200 <= r.status_code < 300
    except Exception:
        return False

def mqtt_publish(topic: str, payload: str) -> bool:
    if not mqtt_client:
        return False
    try:
        mqtt_client.publish(topic, payload, qos=0, retain=False)
        return True
    except Exception:
        return False

def resolve_device(name: str) -> dict:
    return DEVICES.get(name.lower().strip(), {})

def control_device(device_name: str, action: str) -> bool:
    d = resolve_device(device_name)
    ok = False

    if isinstance(d, str):
        # backward compatibility: if value is string treat as HA entity
        d = {"ha_entity": d}

    ha_entity = (d.get("ha_entity") or "").strip()
    if ha_entity:
        domain = ha_entity.split(".", 1)[0]
        service = "turn_on" if action == "on" else "turn_off"
        ok = ha_call(domain, service, {"entity_id": ha_entity}) or ok

    topic = (d.get("mqtt_topic") or "").strip()
    if topic:
        on_payload = d.get("mqtt_on_payload", "ON")
        off_payload = d.get("mqtt_off_payload", "OFF")
        payload = on_payload if action == "on" else off_payload
        ok = mqtt_publish(topic, payload) or ok

    return ok


# =========================
# COMMAND PARSING
# =========================

def parse_local_command(text: str) -> Optional[Tuple[str, str, str]]:
    """
    Returns:
      ("stop", "", "")                         for stop/cancel
      ("device", device_name, "on"/"off")      for device control
      None                                     for normal LLM
    """
    t = (text or "").strip().lower()
    if not t:
        return None

    # stop commands (English + common Urdu transcriptions)
    if t in {"stop", "cancel", "quiet", "silence", "bas", "chup", "band", "ruk"}:
        return ("stop", "", "")

    # "lights on/off"
    m = re.match(r"^(?:lights?|light)\s+(on|off)$", t)
    if m:
        return ("device", "lights", m.group(1))

    # "turn on/off X"
    m = re.match(r"^turn\s+(on|off)\s+(?:the\s+)?(.+)$", t)
    if m:
        return ("device", m.group(2).strip(), m.group(1))

    # Simple Urdu keywords sometimes appear as English in STT; keep above.

    return None


# =========================
# LLM (OpenAI default + optional Gemini)
# =========================

def instructions_text() -> str:
    return (
        "You are a fast, helpful voice assistant for a university teacher in Pakistan.\n"
        "The user may speak Urdu, English, or a mix.\n"
        "- If user is mostly Urdu, reply in Urdu script (NOT Roman Urdu).\n"
        "- If user is mostly English, reply in clear simple English.\n"
        "- Keep answers short: 2–4 spoken sentences.\n"
        "- If asked to control devices, confirm briefly and do not add long explanations.\n"
    )

def run_openai(profile: str, user_text: str) -> str:
    if not openai_client:
        return "OpenAI client not available. Please install the openai package."
    mem = memory_context(profile)
    instr = instructions_text()

    input_text = (mem + "\n\n" if mem else "") + "User: " + user_text

    try:
        resp = openai_client.responses.create(
            model=OPENAI_CHAT_MODEL,
            instructions=instr,
            input=input_text,
            store=False
        )
        return (resp.output_text or "").strip()
    except Exception as e:
        log(f"OpenAI error: {e}")
        return "I had a network error. Please try again."

def run_gemini(profile: str, user_text: str) -> str:
    if not gemini_model:
        return "Gemini is not configured."
    mem = memory_context(profile)
    instr = instructions_text()
    prompt = (mem + "\n\n" if mem else "") + instr + "\nUser: " + user_text
    try:
        r = gemini_model.generate_content(prompt)
        return (getattr(r, "text", "") or "").strip()
    except Exception as e:
        log(f"Gemini error: {e}")
        return "I had a network error. Please try again."

def run_llm(profile: str, user_text: str, for_summary: bool = False) -> str:
    provider = (STATE.llm_provider or "openai").lower()
    if provider == "gemini":
        return run_gemini(profile, user_text)
    return run_openai(profile, user_text)


# =========================
# STT (OpenAI Whisper default; optional faster-whisper)
# =========================

def stt_openai(audio_float32: np.ndarray) -> str:
    if not openai_client:
        return ""
    buf = io.BytesIO()
    sf.write(buf, audio_float32, MIC_SAMPLE_RATE, format="WAV")
    buf.seek(0)
    try:
        buf.name = "audio.wav"  # some backends like filename
    except Exception:
        pass
    try:
        tr = openai_client.audio.transcriptions.create(model="whisper-1", file=buf)
        return (tr.text or "").strip()
    except Exception as e:
        log(f"Whisper STT error: {e}")
        return ""

def stt_local(audio_float32: np.ndarray) -> str:
    if not local_whisper:
        return ""
    try:
        segments, info = local_whisper.transcribe(
            audio_float32,
            language=None,
            vad_filter=True
        )
        return " ".join([s.text.strip() for s in segments]).strip()
    except Exception as e:
        log(f"Local STT error: {e}")
        return ""

def speech_to_text(audio_float32: np.ndarray) -> str:
    # “Streaming STT” (true partial streaming) is complex; this gives low latency via VAD + optional local STT.
    if STT_MODE == "local" and local_whisper:
        return stt_local(audio_float32)
    return stt_openai(audio_float32)


# =========================
# MIC CALLBACK
# =========================

def mic_callback(indata, frames, time_info, status):
    if STATE.stop_app or not STATE.listening_enabled:
        return
    block = indata[:, 0].copy()  # float32 mono

    # wake queue always
    try:
        wake_q.put_nowait(block)
    except queue.Full:
        try:
            wake_q.get_nowait()
        except queue.Empty:
            pass

    # record queue only while recording
    if recording_active.is_set():
        try:
            record_q.put_nowait(block)
        except queue.Full:
            try:
                record_q.get_nowait()
            except queue.Empty:
                pass


# =========================
# WAKE DETECTOR LOOP (multi-user + followup)
# =========================

def wake_detector_loop():
    global wake_label
    frame_len = 320  # 20ms @ 16k

    while not STATE.stop_app:
        try:
            block = wake_q.get(timeout=0.2)
        except queue.Empty:
            continue

        now = time.time()

        # Follow-up: speech triggers without wakeword
        if STATE.in_followup and now <= STATE.followup_deadline:
            i16 = (block * 32768).astype(np.int16)
            speech = False
            for i in range(0, len(i16) - frame_len + 1, frame_len):
                if vad.is_speech(i16[i:i+frame_len].tobytes(), MIC_SAMPLE_RATE):
                    speech = True
                    break
            if speech:
                with wake_label_lock:
                    wake_label = "followup"
                wake_event.set()
                STATE.in_followup = False
                continue
        else:
            STATE.in_followup = False

        # Wakeword detection
        i16 = (block * 32768).astype(np.int16)
        scores = wake_model.predict(i16)
        if not scores:
            continue

        best_label, best_score = None, 0.0
        for k, v in scores.items():
            try:
                sv = float(v)
            except Exception:
                continue
            if sv > best_score:
                best_score, best_label = sv, k

        if best_label and best_score >= WAKEWORD_THRESHOLD:
            with wake_label_lock:
                wake_label = str(best_label).lower()
            wake_event.set()


# =========================
# VAD RECORDING
# =========================

def clear_record_queue():
    while True:
        try:
            record_q.get_nowait()
        except queue.Empty:
            break

def record_until_silence() -> np.ndarray:
    clear_record_queue()
    recording_active.set()

    start = time.time()
    chunks: List[np.ndarray] = []
    speech_ms = 0
    silence_ms = 0

    frame_len = 320  # 20ms
    pending = np.zeros(0, dtype=np.int16)

    try:
        while True:
            if STATE.stop_app or not STATE.listening_enabled:
                break
            if time.time() - start > MAX_RECORD_SECONDS:
                break

            try:
                block = record_q.get(timeout=1.0)
            except queue.Empty:
                continue

            chunks.append(block)

            i16 = (block * 32768).astype(np.int16)
            pending = np.concatenate([pending, i16])

            while len(pending) >= frame_len:
                frame = pending[:frame_len]
                pending = pending[frame_len:]

                if vad.is_speech(frame.tobytes(), MIC_SAMPLE_RATE):
                    speech_ms += 20
                    silence_ms = 0
                else:
                    silence_ms += 20

            if speech_ms >= MIN_SPEECH_MS and silence_ms >= END_SILENCE_MS:
                break

    finally:
        recording_active.clear()

    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32)


# =========================
# PIPELINE
# =========================

def handle_text(profile: str, text: str) -> str:
    cmd = parse_local_command(text)
    if cmd:
        kind, dev, act = cmd
        if kind == "stop":
            return ""

        if kind == "device":
            ok = control_device(dev, act)
            if ok:
                return "Done." if not is_urdu(text) else "ٹھیک ہے، ہو گیا۔"
            else:
                return "I couldn't control that device. Check Home Assistant/MQTT settings." if not is_urdu(text) else "میں وہ ڈیوائس کنٹرول نہیں کر سکا۔ سیٹنگز چیک کریں۔"

    # LLM answer
    return run_llm(profile, text)

def assistant_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    while not STATE.stop_app:
        if not STATE.listening_enabled:
            time.sleep(0.2)
            continue

        if not wake_event.wait(timeout=0.2):
            continue
        wake_event.clear()

        if STATE.stop_app or not STATE.listening_enabled:
            continue

        # Resolve profile from wake label
        with wake_label_lock:
            lbl = wake_label

        if lbl and lbl != "followup":
            STATE.active_profile = WAKEWORD_MAP.get(lbl, lbl) or "default"

        profile = STATE.active_profile

        play_beep()
        audio = record_until_silence()
        if audio.size < 1200:
            continue

        text = speech_to_text(audio)
        if not text:
            continue

        # Remove wakeword token if it appears in transcript
        for w in list(WAKEWORD_MAP.keys()):
            text = re.sub(rf"\b{re.escape(w)}\b[:,]?\s*", "", text, flags=re.IGNORECASE).strip()

        log(f"[{profile}] User: {text}")

        try:
            answer = handle_text(profile, text).strip()

            if answer:
                append_turn(profile, text, answer)
                maybe_summarize(profile)

                wav = loop.run_until_complete(tts_wav_bytes(answer))
                play_wav_bytes(wav)

            # Nest-like follow-up
            STATE.in_followup = True
            STATE.followup_deadline = time.time() + FOLLOWUP_WINDOW_SECONDS

        except Exception as e:
            log(f"Pipeline error: {e}")

    try:
        loop.stop()
        loop.close()
    except Exception:
        pass


# =========================
# HOTKEY (PUSH-TO-TALK)
# =========================

def hotkey_loop():
    if not keyboard:
        log("Hotkey disabled (keyboard package not installed).")
        return

    def trigger():
        with wake_label_lock:
            global wake_label
            wake_label = "followup"
        wake_event.set()

    try:
        keyboard.add_hotkey(PUSH_TO_TALK_HOTKEY, trigger)
        log(f"Push-to-talk hotkey: {PUSH_TO_TALK_HOTKEY}")
        while not STATE.stop_app:
            time.sleep(0.25)
    except Exception as e:
        log(f"Hotkey error: {e}")


# =========================
# TRAY ICON
# =========================

def tray_image():
    w, h = 64, 64
    color = "green" if STATE.listening_enabled else "red"
    img = Image.new("RGB", (w, h), color)
    dc = ImageDraw.Draw(img)
    dc.ellipse((16, 16, 48, 48), fill="white")
    return img

def toggle_listening(icon, _item):
    STATE.listening_enabled = not STATE.listening_enabled
    icon.icon = tray_image()
    icon.visible = True
    log(f"Listening: {STATE.listening_enabled}")

def switch_brain(icon, _item):
    # OpenAI default; allow switching to Gemini only if configured
    if STATE.llm_provider == "openai":
        if not gemini_model:
            log("Gemini not configured; staying on OpenAI.")
            return
        STATE.llm_provider = "gemini"
    else:
        STATE.llm_provider = "openai"
    log(f"LLM provider: {STATE.llm_provider}")

def open_setup(icon, _item):
    setup_wizard()
    # Reload config after wizard
    global CFG, WAKEWORD_MAP, DEVICES, HA_URL, HA_TOKEN, MQTT_HOST, MQTT_PORT, MQTT_USER, MQTT_PASS
    global MIC_SAMPLE_RATE, MIC_BLOCKSIZE, WAKEWORD_THRESHOLD, VAD_MODE, MAX_RECORD_SECONDS, END_SILENCE_MS
    global MIN_SPEECH_MS, FOLLOWUP_WINDOW_SECONDS, TTS_EN_VOICE, TTS_UR_VOICE, OPENAI_CHAT_MODEL
    global OPENAI_API_KEY, openai_client, GEMINI_API_KEY, GEMINI_MODEL_NAME, gemini_model
    global STT_MODE, LOCAL_STT_MODEL_NAME, local_whisper, vad, PUSH_TO_TALK_HOTKEY

    CFG = load_cfg()

    # Refresh runtime variables (some cannot fully re-init without restart; good enough for keys/devices)
    WAKEWORD_MAP = CFG.get("wakeword_map") or {"professor": "default"}
    DEVICES = CFG.get("devices") or {}
    HA_URL = (CFG.get("ha_url") or "").strip()
    HA_TOKEN = (CFG.get("ha_token") or "").strip()

    MQTT_HOST = (CFG.get("mqtt_host") or "").strip()
    MQTT_PORT = int(CFG.get("mqtt_port") or 1883)
    MQTT_USER = (CFG.get("mqtt_user") or "").strip()
    MQTT_PASS = (CFG.get("mqtt_pass") or "").strip()

    OPENAI_API_KEY = (CFG.get("openai_api_key") or "").strip()
    OPENAI_CHAT_MODEL = (CFG.get("openai_chat_model") or "gpt-4o-mini").strip()

    if OpenAI and OPENAI_API_KEY:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)

    GEMINI_API_KEY = (CFG.get("gemini_api_key") or "").strip()
    GEMINI_MODEL_NAME = (CFG.get("gemini_model") or "gemini-1.5-flash").strip()
    if genai and GEMINI_API_KEY:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        except Exception:
            gemini_model = None

    PUSH_TO_TALK_HOTKEY = (CFG.get("push_to_talk_hotkey") or "ctrl+alt+space").strip()

    STT_MODE = (CFG.get("stt_mode") or "openai").lower().strip()
    LOCAL_STT_MODEL_NAME = (CFG.get("local_stt_model") or "small").strip()
    if STT_MODE == "local" and WhisperModel is not None:
        try:
            local_whisper = WhisperModel(LOCAL_STT_MODEL_NAME, device="cpu", compute_type="int8")
        except Exception:
            local_whisper = None
            STT_MODE = "openai"

    log("Setup updated. (Some changes may need app restart for wake models.)")

def install_startup_menu(icon, _item):
    install_startup()
    log("Installed run-at-startup.")

def remove_startup_menu(icon, _item):
    remove_startup()
    log("Removed run-at-startup.")

def on_exit(icon, _item):
    STATE.stop_app = True
    try:
        if mqtt_client:
            mqtt_client.loop_stop()
    except Exception:
        pass
    icon.stop()

def setup_tray():
    menu = (
        item(lambda: "Disable listening" if STATE.listening_enabled else "Enable listening", toggle_listening),
        item(lambda: f"Switch brain (now: {STATE.llm_provider})", switch_brain),
        item("Open setup (keys/devices)", open_setup),
        item(lambda: "Install run-at-startup" if not startup_installed() else "Remove run-at-startup",
             install_startup_menu if not startup_installed() else remove_startup_menu),
        item("Exit", on_exit),
    )
    icon = pystray.Icon("ProfessorAssistant", tray_image(), "Professor Assistant", menu)
    icon.run()


# =========================
# MAIN
# =========================

def main():
    log("Starting Professor Assistant...")
    log(f"Base dir: {BASE_DIR}")
    log(f"Models dir: {MODELS_DIR}")
    log(f"Config: {CFG_PATH}")
    log(f"STT mode: {STT_MODE}")
    log(f"LLM provider default: {STATE.llm_provider}")
    log(f"Wakewords: {list(WAKEWORD_MAP.keys())}")

    if not MODELS_DIR.exists() or not any(MODELS_DIR.glob("*.onnx")):
        log(f"ERROR: No .onnx models found in {MODELS_DIR}. Put models there (e.g., models/professor.onnx).")
        return

    # Mic stream
    stream = sd.InputStream(
        channels=1,
        samplerate=MIC_SAMPLE_RATE,
        blocksize=MIC_BLOCKSIZE,
        dtype="float32",
        callback=mic_callback
    )
    stream.start()

    # Threads
    threading.Thread(target=wake_detector_loop, daemon=True).start()
    threading.Thread(target=assistant_loop, daemon=True).start()

    if keyboard:
        threading.Thread(target=hotkey_loop, daemon=True).start()

    # Tray in main thread
    setup_tray()

    # Shutdown
    STATE.stop_app = True
    try:
        stream.stop()
        stream.close()
    except Exception:
        pass

    log("Assistant exited.")

if __name__ == "__main__":
    main()
