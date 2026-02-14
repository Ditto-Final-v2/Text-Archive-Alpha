import json
import math
import calendar
import csv
import os
import re
import sqlite3
import threading
import hashlib
import shutil
import random
import time
import copy
import subprocess
import sys
import logging
from logging.handlers import RotatingFileHandler
import xml.etree.ElementTree as ET
from datetime import datetime
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import List
from urllib.parse import urlencode
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from PIL import Image, ImageOps

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIF_PREVIEW_AVAILABLE = True
except Exception:
    HEIF_PREVIEW_AVAILABLE = False

DB_PATH = "texts.db"
FAISS_INDEX_PATH = "texts.faiss"
SETUP_STATE_PATH = Path("setup_state.json")
UPLOADS_DIR = Path("uploads")
LOGS_DIR = Path("logs")
APP_LOG_PATH = LOGS_DIR / "app.log"
CSV_BODY_COLUMN_CANDIDATES = {
    "message", "content", "body", "text", "message_text", "sms", "imessage",
}
EMBED_MODEL_PREFERRED = os.environ.get("TEXTAPP_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_MODEL_FALLBACKS = [
    "sentence-transformers/all-MiniLM-L6-v2",
]
RERANKER_MODEL_NAME = os.environ.get("TEXTAPP_RERANKER_MODEL", "BAAI/bge-reranker-base")
ATTACHMENTS_DIR = Path("attachments")
ATTACHMENTS_PREVIEW_DIR = ATTACHMENTS_DIR / ".thumbs"
SIGNAL_COLUMNS = {
    "love": "love",
    "flirting": "flirting",
    "compliments": "compliments",
    "laughing": "laughing",
    "supportive": "supportive",
    "difficult": "difficult",
    "repair": "repair",
    "gratitude": "gratitude",
    "missing": "missing",
}
SIGNAL_THRESHOLDS = {
    "love": 0.67,
    "flirting": 0.70,
    "compliments": 0.486,
    "laughing": 0.48,
    "supportive": 0.30,
    "difficult": 0.208,
    "repair": 0.41,
    "gratitude": 0.36,
    "missing": 0.38,
}
DAY_HIGHLIGHT_THRESHOLD_SCALE = 0.85
DAY_HIGHLIGHT_MIN_WORDS = 8

ICON_OPTIONS = [
    {"key": "double_pink_hearts", "label": "Double pink hearts", "entity": "&#128149;"},
    {"key": "pink_heart", "label": "Pink heart", "entity": "&#129655;"},
    {"key": "purple_heart", "label": "Purple heart", "entity": "&#128156;"},
    {"key": "red_heart", "label": "Red heart", "entity": "&#10084;&#65039;"},
    {"key": "orange_heart", "label": "Orange heart", "entity": "&#129505;"},
    {"key": "yellow_heart", "label": "Yellow heart", "entity": "&#128155;"},
    {"key": "green_heart", "label": "Green heart", "entity": "&#128154;"},
    {"key": "blue_heart", "label": "Blue heart", "entity": "&#128153;"},
    {"key": "black_heart", "label": "Black heart", "entity": "&#128420;"},
    {"key": "white_heart", "label": "White heart", "entity": "&#129293;"},
    {"key": "sparkling_heart", "label": "Sparkling heart", "entity": "&#128150;"},
    {"key": "heart_with_ribbon", "label": "Heart with ribbon", "entity": "&#128157;"},
    {"key": "whale", "label": "Whale", "entity": "&#128011;"},
    {"key": "stegosaurus", "label": "Stegosaurus", "entity": "&#129429;"},
    {"key": "pleading_face", "label": "Pleading face", "entity": "&#129402;"},
    {"key": "dancing_girl", "label": "Dancing girl", "entity": "&#128131;"},
    {"key": "whale_with_water", "label": "Whale with water", "entity": "&#128051;"},
    {"key": "confetti_popper", "label": "Confetti popper", "entity": "&#127881;"},
    {"key": "ice_cream", "label": "Ice cream", "entity": "&#127848;"},
    {"key": "egg", "label": "Egg", "entity": "&#129370;"},
    {"key": "sparkles", "label": "Sparkles", "entity": "&#10024;"},
    {"key": "sunflower", "label": "Sunflower", "entity": "&#127803;"},
    {"key": "cherry_blossom", "label": "Cherry blossom", "entity": "&#127800;"},
    {"key": "rainbow", "label": "Rainbow", "entity": "&#127752;"},
    {"key": "star", "label": "Star", "entity": "&#11088;"},
    {"key": "fire", "label": "Fire", "entity": "&#128293;"},
    {"key": "smiling_face_hearts", "label": "Smiling face with hearts", "entity": "&#129392;"},
    {"key": "face_holding_back_tears", "label": "Face holding back tears", "entity": "&#129401;"},
    {"key": "camera", "label": "Camera", "entity": "&#128247;"},
    {"key": "heart_hands", "label": "Heart hands", "entity": "&#129782;"},
    {"key": "kiss_mark", "label": "Kiss mark", "entity": "&#128139;"},
    {"key": "ring", "label": "Ring", "entity": "&#128141;"},
    {"key": "rose", "label": "Rose", "entity": "&#127801;"},
    {"key": "tulip", "label": "Tulip", "entity": "&#127799;"},
    {"key": "hibiscus", "label": "Hibiscus", "entity": "&#127802;"},
    {"key": "dizzy", "label": "Dizzy", "entity": "&#128171;"},
    {"key": "two_stars", "label": "Glow stars", "entity": "&#10024;"},
    {"key": "milky_way", "label": "Milky way", "entity": "&#127756;"},
    {"key": "crescent_moon", "label": "Crescent moon", "entity": "&#127769;"},
    {"key": "sun", "label": "Sun", "entity": "&#9728;&#65039;"},
    {"key": "rain_cloud", "label": "Rain cloud", "entity": "&#127783;&#65039;"},
    {"key": "revolving_hearts", "label": "Revolving hearts", "entity": "&#128158;"},
    {"key": "butterfly", "label": "Butterfly", "entity": "&#129419;"},
    {"key": "paw_prints", "label": "Paw prints", "entity": "&#128062;"},
    {"key": "cat", "label": "Cat", "entity": "&#128049;"},
    {"key": "dog", "label": "Dog", "entity": "&#128054;"},
    {"key": "otter", "label": "Otter", "entity": "&#129446;"},
    {"key": "dolphin", "label": "Dolphin", "entity": "&#128044;"},
    {"key": "penguin", "label": "Penguin", "entity": "&#128039;"},
    {"key": "unicorn", "label": "Unicorn", "entity": "&#129412;"},
    {"key": "t_rex", "label": "T-Rex", "entity": "&#129430;"},
    {"key": "cherries", "label": "Cherries", "entity": "&#127826;"},
    {"key": "strawberry", "label": "Strawberry", "entity": "&#127827;"},
    {"key": "watermelon", "label": "Watermelon", "entity": "&#127817;"},
    {"key": "bubble_tea", "label": "Bubble tea", "entity": "&#129379;"},
    {"key": "coffee", "label": "Coffee", "entity": "&#9749;&#65039;"},
    {"key": "croissant", "label": "Croissant", "entity": "&#129360;"},
    {"key": "cupcake", "label": "Cupcake", "entity": "&#129473;"},
    {"key": "gift", "label": "Gift", "entity": "&#127873;"},
    {"key": "balloon", "label": "Balloon", "entity": "&#127880;"},
    {"key": "music_note", "label": "Music note", "entity": "&#127925;"},
    {"key": "headphones", "label": "Headphones", "entity": "&#127911;"},
    {"key": "microphone", "label": "Microphone", "entity": "&#127908;"},
    {"key": "movie_camera", "label": "Movie camera", "entity": "&#127909;"},
    {"key": "gamepad", "label": "Gamepad", "entity": "&#127918;"},
    {"key": "soccer_ball", "label": "Soccer ball", "entity": "&#9917;"},
    {"key": "basketball", "label": "Basketball", "entity": "&#127936;"},
    {"key": "mountain", "label": "Mountain", "entity": "&#9968;&#65039;"},
    {"key": "beach", "label": "Beach", "entity": "&#127958;&#65039;"},
    {"key": "camping", "label": "Camping", "entity": "&#127957;&#65039;"},
    {"key": "airplane", "label": "Airplane", "entity": "&#9992;&#65039;"},
    {"key": "rocket", "label": "Rocket", "entity": "&#128640;"},
    {"key": "gem", "label": "Gem", "entity": "&#128142;"},
    {"key": "lock", "label": "Lock", "entity": "&#128274;"},
    {"key": "key", "label": "Key", "entity": "&#128273;"},
    {"key": "book", "label": "Book", "entity": "&#128218;"},
    {"key": "scroll", "label": "Scroll", "entity": "&#128220;"},
    {"key": "hourglass", "label": "Hourglass", "entity": "&#9203;"},
    {"key": "wave", "label": "Wave", "entity": "&#128075;"},
    {"key": "clinking_glasses", "label": "Clinking glasses", "entity": "&#129346;"},
    {"key": "champagne", "label": "Champagne", "entity": "&#127870;"},
    {"key": "fireworks", "label": "Fireworks", "entity": "&#127878;"},
    {"key": "moon_face", "label": "Smirking moon", "entity": "&#127773;"},
    {"key": "wink", "label": "Wink", "entity": "&#128521;"},
    {"key": "smile", "label": "Smile", "entity": "&#128522;"},
    {"key": "happy_tears", "label": "Face with tears of joy", "entity": "&#128514;"},
    {"key": "smirk", "label": "Smirk", "entity": "&#128527;"},
    {"key": "pleased", "label": "Relieved face", "entity": "&#128524;"},
    {"key": "infinity", "label": "Infinity", "entity": "&#8734;"},
    {"key": "love_letter", "label": "Love letter", "entity": "&#128140;"},
]
ICON_MAP = {x["key"]: x for x in ICON_OPTIONS}
DEFAULT_ICON_KEY = "pink_heart"

# Simple app password (set env var on your machine for better practice)
APP_PASSWORD = os.environ.get("TEXTAPP_PASSWORD", "changeme")

# How many results and how much context
TOP_K = 25
CONTEXT_BEFORE = 30
CONTEXT_AFTER = 30
SEARCH_CANDIDATE_K = 650
RERANK_TOP_K = 320
CONTEXT_RERANK_TOP_K = 220
RESULT_CONTEXT_BEFORE = 2
RESULT_CONTEXT_AFTER = 2
MMR_LAMBDA = 0.76
MMR_POOL_K = 120
QUERY_EXPANDER_MODEL = os.environ.get("TEXTAPP_QUERY_EXPANDER_MODEL", "").strip()
FOCUS_ENFORCE_TOP_N = 8
FOCUS_HIGH_CONFIDENCE_THRESHOLD = 0.54
EXPERIMENT_K_LEX = int(os.environ.get("TEXTAPP_EXPERIMENT_K_LEX", "80"))
EXPERIMENT_K_SEM = int(os.environ.get("TEXTAPP_EXPERIMENT_K_SEM", "560"))
EXPERIMENT_MERGE_N = int(os.environ.get("TEXTAPP_EXPERIMENT_MERGE_N", "180"))
EXPERIMENT_TOP_K = int(os.environ.get("TEXTAPP_EXPERIMENT_TOP_K", "25"))
EXPERIMENT_RERANK_BATCH_SIZE = int(os.environ.get("TEXTAPP_EXPERIMENT_RERANK_BATCH_SIZE", "40"))
EXPERIMENT_MERGE_METHOD = os.environ.get("TEXTAPP_EXPERIMENT_MERGE_METHOD", "weighted").strip().lower()
EXPERIMENT_RRF_K = int(os.environ.get("TEXTAPP_EXPERIMENT_RRF_K", "60"))
EXPERIMENT_WEIGHT_LEX = float(os.environ.get("TEXTAPP_EXPERIMENT_WEIGHT_LEX", "0.16"))
EXPERIMENT_WEIGHT_SEM = float(os.environ.get("TEXTAPP_EXPERIMENT_WEIGHT_SEM", "0.84"))
EXPERIMENT_CONTEXT_RERANK_WEIGHT = float(os.environ.get("TEXTAPP_EXPERIMENT_CONTEXT_RERANK_WEIGHT", "0.64"))
EXPERIMENT_MESSAGE_RERANK_WEIGHT = float(os.environ.get("TEXTAPP_EXPERIMENT_MESSAGE_RERANK_WEIGHT", "0.14"))
EXPERIMENT_SEMANTIC_WEIGHT = float(os.environ.get("TEXTAPP_EXPERIMENT_SEMANTIC_WEIGHT", "0.12"))
EXPERIMENT_CONTEXT_WORD_WEIGHT = float(os.environ.get("TEXTAPP_EXPERIMENT_CONTEXT_WORD_WEIGHT", "0.22"))
EXPERIMENT_CENTER_LENGTH_WEIGHT = float(os.environ.get("TEXTAPP_EXPERIMENT_CENTER_LENGTH_WEIGHT", "0.08"))
TRENDS_MAX_TERMS = int(os.environ.get("TEXTAPP_TRENDS_MAX_TERMS", "5"))
TRENDS_BIN_DAYS_DEFAULT = int(os.environ.get("TEXTAPP_TRENDS_BIN_DAYS_DEFAULT", "30"))
TRENDS_EXACT_K = int(os.environ.get("TEXTAPP_TRENDS_EXACT_K", "1400"))
TRENDS_SEM_K = int(os.environ.get("TEXTAPP_TRENDS_SEM_K", "2200"))
TRENDS_HIGHLIGHT_LIMIT = int(os.environ.get("TEXTAPP_TRENDS_HIGHLIGHT_LIMIT", "25"))
TRENDS_MAX_COUNTED_MATCHES = int(os.environ.get("TEXTAPP_TRENDS_MAX_COUNTED_MATCHES", "5000"))
TRENDS_MIN_COUNTED_MATCHES = int(os.environ.get("TEXTAPP_TRENDS_MIN_COUNTED_MATCHES", "40"))
TRENDS_STRONG_MATCH_PERCENTILE = float(os.environ.get("TEXTAPP_TRENDS_STRONG_MATCH_PERCENTILE", "0.88"))
SEARCH_RESULTS_CACHE_TTL_SECONDS = int(os.environ.get("TEXTAPP_SEARCH_CACHE_TTL", "1800"))
SEARCH_RESULTS_CACHE_MAX_ITEMS = int(os.environ.get("TEXTAPP_SEARCH_CACHE_MAX", "80"))
SETUP_BUILD_TIMEOUT_SECONDS = int(os.environ.get("TEXTAPP_SETUP_BUILD_TIMEOUT", "5400"))

EXPERIMENT_LEXICAL_EXPANSIONS = {
    "love": ["adore", "cherish", "miss"],
    "loved": ["adored", "cherished", "missed"],
    "laugh": ["lol", "haha", "lmao"],
    "laughing": ["lol", "haha", "lmao"],
    "flirty": ["tease", "sexy", "desire"],
    "flirting": ["tease", "sexy", "desire"],
    "reassured": ["comfort", "support", "calm"],
    "reassure": ["comfort", "support", "calm"],
    "moving": ["move", "apartment", "house"],
    "roommates": ["roommate", "roomies", "housemates"],
}

QUERY_EXPANSION_RULES = {
    "pets": ["dog", "dogs", "puppy", "puppies", "cat", "cats", "kitty", "kittens", "pet", "vet", "walk"],
    "sexting": [
        "sexy", "hot", "horny", "nude", "nudes", "naked", "turned on", "turn me on",
        "fuck", "fucking", "wet", "hard", "thirsty", "desire"
    ],
    "flirting": ["sexy", "hot", "pretty", "beautiful", "gorgeous", "desire", "tease"],
    "support": ["comfort", "care", "here for you", "proud of you", "you got this", "reassure"],
    "fight": ["argue", "argument", "upset", "angry", "sad", "sorry", "apology", "repair"],
}

QUERY_SPELL_NORMALIZATIONS = {
    "messges": "messages",
    "messeges": "messages",
    "mesages": "messages",
    "roomates": "roommates",
    "roomate": "roommate",
    "roomies": "roommates",
    "definately": "definitely",
    "seperate": "separate",
    "wierd": "weird",
    "recieve": "receive",
    "occured": "occurred",
    "neccessary": "necessary",
}
GENERIC_REWRITE_PREFIXES = [
    "messages about",
    "conversation about",
    "talking about",
    "discussion about",
]
SEARCH_STOPWORDS = {
    "a", "an", "and", "or", "the", "to", "for", "with", "from", "of", "in", "on", "at",
    "is", "are", "was", "were", "be", "being", "been", "i", "we", "you", "he", "she",
    "they", "it", "this", "that", "these", "those", "our", "my", "your", "about",
}
PERSON_TOKENS = {"person_a", "person_b"}
PARTICIPANT_A_LABEL = os.environ.get("TEXTAPP_PARTICIPANT_A_LABEL", "Person A").strip() or "Person A"
PARTICIPANT_B_LABEL = os.environ.get("TEXTAPP_PARTICIPANT_B_LABEL", "Person B").strip() or "Person B"
_DIRECTION_TOKEN_IGNORE = {
    "sms", "mms", "imessage", "message", "messages", "text", "texts", "sent", "received",
    "incoming", "outgoing", "to", "from", "and", "the", "with", "chat", "phone",
}
PARTICIPANT_DIRECTION_HINTS = {
    "person_a": set(),
    "person_b": set(),
}
GENERIC_QUERY_FILLER = {
    "message", "messages", "text", "texts", "talk", "talking", "conversation", "discuss",
    "discussion", "show", "find", "search", "looking", "look", "anything", "stuff",
    "things", "related", "topic", "time", "times", "texted", "texting", "talked",
    "about", "regarding", "around", "when", "where", "did", "were", "was", "us",
}

TOPIC_PREFIX_PATTERNS = [
    r"^\s*(?:messages?|texts?)\s+(?:about|regarding|on)\s+",
    r"^\s*(?:times?|moments?)\s+we\s+(?:texted|talked|spoke|chatted)\s+(?:about|regarding|on)\s+",
    r"^\s*(?:when|where)\s+did\s+we\s+(?:text|talk|speak|chat)\s+(?:about|regarding|on)\s+",
    r"^\s*(?:show|find|search(?:\s+for)?|looking\s+for)\s+(?:messages?|texts?)\s+(?:about|regarding|on)\s+",
]

SIGNAL_QUERY_BOOSTS = {
    "sexting": [("flirting", 0.45), ("compliments", 0.12)],
    "flirty": [("flirting", 0.40)],
    "hot": [("flirting", 0.35), ("compliments", 0.10)],
    "beautiful": [("compliments", 0.35), ("love", 0.12)],
    "pretty": [("compliments", 0.32)],
    "pets": [],
    "upset": [("difficult", 0.42), ("repair", 0.22)],
    "frustrated": [("difficult", 0.40), ("repair", 0.20)],
    "conflict": [("difficult", 0.44), ("repair", 0.24)],
    "argue": [("difficult", 0.42), ("repair", 0.30)],
    "argument": [("difficult", 0.42), ("repair", 0.30)],
    "fight": [("difficult", 0.44), ("repair", 0.28)],
}

LAUGH_TOKEN_PATTERNS = [
    re.compile(r"\blol+\b", re.IGNORECASE),
    re.compile(r"\blmao+\b", re.IGNORECASE),
    re.compile(r"\blmfao+\b", re.IGNORECASE),
    re.compile(r"\brofl+\b", re.IGNORECASE),
    re.compile(r"(?:ha){2,}", re.IGNORECASE),
    re.compile(r"(?:he){2,}", re.IGNORECASE),
    re.compile(r"(?:h+a+h+a+){1,}", re.IGNORECASE),
]
LAUGH_PHRASE_PATTERNS = [
    re.compile(r"\bthat(?:'| i)?s so funny\b", re.IGNORECASE),
    re.compile(r"\bthat(?:'| i)?s funny\b", re.IGNORECASE),
    re.compile(r"\byou(?:'| a)?re funny\b", re.IGNORECASE),
    re.compile(r"\bi'?m dead\b", re.IGNORECASE),
]
LAUGH_EMOJIS = ("😂", "🤣", "😆", "😹")

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load once at startup
faiss_index = None
model = None
embed_model_name = ""
reranker = None
query_expander = None
query_expander_error = None
query_expander_lock = threading.Lock()
search_results_cache_lock = threading.Lock()
search_results_cache: "OrderedDict[tuple[str, str], tuple[float, str, list[dict]]]" = OrderedDict()

app_logger = logging.getLogger("textarchive")
if not app_logger.handlers:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    _log_handler = RotatingFileHandler(str(APP_LOG_PATH), maxBytes=2_000_000, backupCount=2, encoding="utf-8")
    _log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    app_logger.addHandler(_log_handler)
app_logger.setLevel(logging.INFO)
app_logger.propagate = False


def load_compatible_embedder(index):
    candidates = [EMBED_MODEL_PREFERRED] + [m for m in EMBED_MODEL_FALLBACKS if m != EMBED_MODEL_PREFERRED]
    for name in candidates:
        try:
            m = SentenceTransformer(name)
            if index is None:
                return m, name
            dim = int(m.get_sentence_embedding_dimension())
            if dim == int(index.d):
                return m, name
        except Exception:
            continue
    if index is None:
        raise RuntimeError("Could not load any embedding model. Set TEXTAPP_EMBED_MODEL or install a supported model.")
    raise RuntimeError(
        f"No embedding model matches FAISS dim={index.d}. Rebuild index with build_db.py or set TEXTAPP_EMBED_MODEL."
    )


def setup_is_complete() -> bool:
    return SETUP_STATE_PATH.exists()


def load_setup_state() -> dict:
    if not SETUP_STATE_PATH.exists():
        return {}
    try:
        return json.loads(SETUP_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_setup_state(data: dict) -> None:
    SETUP_STATE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def log_event(level: str, message: str) -> None:
    lvl = (level or "info").strip().lower()
    if lvl == "error":
        app_logger.error(message)
    elif lvl == "warning":
        app_logger.warning(message)
    else:
        app_logger.info(message)


def get_participant_labels() -> tuple[str, str]:
    state = load_setup_state()
    label_a = (str(state.get("participant_a_label") or "").strip() or PARTICIPANT_A_LABEL)
    label_b = (str(state.get("participant_b_label") or "").strip() or PARTICIPANT_B_LABEL)
    return label_a, label_b


def apply_participant_labels() -> None:
    label_a, label_b = get_participant_labels()
    templates.env.globals["participant_a_label"] = label_a
    templates.env.globals["participant_b_label"] = label_b


def _password_digest(password: str, salt_hex: str) -> str:
    salt = bytes.fromhex(salt_hex)
    dk = hashlib.pbkdf2_hmac("sha256", (password or "").encode("utf-8"), salt, 200_000)
    return dk.hex()


def set_configured_password(password: str) -> None:
    salt_hex = os.urandom(16).hex()
    state = load_setup_state()
    state["password_salt"] = salt_hex
    state["password_hash"] = _password_digest(password, salt_hex)
    state["configured_at"] = datetime.now().isoformat(timespec="seconds")
    save_setup_state(state)


def verify_configured_password(password: str) -> bool:
    state = load_setup_state()
    salt_hex = str(state.get("password_salt") or "").strip()
    digest = str(state.get("password_hash") or "").strip()
    if not salt_hex or not digest:
        return (password or "").strip().lower() == (APP_PASSWORD or "").strip().lower()
    return _password_digest((password or "").strip(), salt_hex) == digest


def initialize_search_resources() -> None:
    global faiss_index, model, embed_model_name, reranker
    idx = None
    if Path(FAISS_INDEX_PATH).exists():
        try:
            idx = faiss.read_index(FAISS_INDEX_PATH)
        except Exception:
            idx = None
    faiss_index = idx
    try:
        model, embed_model_name = load_compatible_embedder(faiss_index)
    except Exception:
        model = None
        embed_model_name = ""
    if reranker is None:
        try:
            reranker = CrossEncoder(RERANKER_MODEL_NAME, max_length=256)
        except Exception:
            reranker = None


initialize_search_resources()
apply_participant_labels()

def db():
    return sqlite3.connect(DB_PATH)


def _log_path_for_ui() -> str:
    try:
        return str(APP_LOG_PATH.resolve())
    except Exception:
        return str(APP_LOG_PATH)


def _tail_app_log(max_lines: int = 40) -> str:
    try:
        if not APP_LOG_PATH.exists():
            return ""
        lines = APP_LOG_PATH.read_text(encoding="utf-8", errors="ignore").splitlines()
        if not lines:
            return ""
        return "\n".join(lines[-max(1, int(max_lines)):])
    except Exception:
        return ""


def _render_setup_result(
    request: Request,
    success: bool,
    message: str,
    details: str = "",
    error_blob: str = "",
):
    return templates.TemplateResponse("setup_result.html", {
        "request": request,
        "success": bool(success),
        "message": message,
        "details": details,
        "error_blob": error_blob,
        "log_path": _log_path_for_ui(),
    })


def validate_uploaded_source(path: Path, suffix: str) -> tuple[bool, str]:
    if not path.exists() or path.stat().st_size <= 0:
        return False, "Uploaded file is empty."

    if suffix == ".xml":
        try:
            root = ET.parse(str(path)).getroot()
            root_tag = (root.tag or "").lower()
            child_tags = [(c.tag or "").lower() for c in list(root)[:8]]
            if "smses" not in root_tag and not any(("sms" in t or "mms" in t) for t in child_tags):
                return False, "XML format not recognized. Please upload an SMS Backup-style XML export."
            return True, ""
        except ET.ParseError:
            return False, "Could not parse XML. The file appears invalid or corrupted."
        except Exception:
            return False, "Could not read XML file."

    if suffix == ".csv":
        try:
            sample = path.read_text(encoding="utf-8-sig", errors="ignore")
            try:
                dialect = csv.Sniffer().sniff(sample[:4096], delimiters=",;\t|")
                delimiter = dialect.delimiter
            except Exception:
                delimiter = ","

            detected_headers = set()
            with path.open("r", encoding="utf-8-sig", newline="") as fh:
                reader = csv.reader(fh, delimiter=delimiter)
                for i, row in enumerate(reader):
                    if i >= 8:
                        break
                    normalized = {str(h or "").strip().lower() for h in row}
                    detected_headers |= normalized
                    if normalized & CSV_BODY_COLUMN_CANDIDATES:
                        detected_headers = normalized
                        break

            if detected_headers.isdisjoint(CSV_BODY_COLUMN_CANDIDATES):
                names = ", ".join(sorted(CSV_BODY_COLUMN_CANDIDATES))
                return False, f"CSV must include a message body column. Accepted names: {names}."
            return True, ""
        except StopIteration:
            return False, "CSV is empty."
        except Exception:
            return False, "Could not read CSV file."

    return False, "Unsupported file type. Please upload XML or CSV."


def _clear_runtime_caches() -> None:
    try:
        _cached_query_embedding_blob.cache_clear()
    except Exception:
        pass
    with search_results_cache_lock:
        search_results_cache.clear()


def _safe_unlink(path: Path) -> tuple[int, int]:
    # returns (removed_count, error_count)
    try:
        if path.exists() and path.is_file():
            path.unlink()
            return 1, 0
    except Exception:
        return 0, 1
    return 0, 0


def _safe_rmtree(path: Path) -> tuple[int, int]:
    # returns (removed_count, error_count)
    try:
        if path.exists() and path.is_dir():
            shutil.rmtree(path, ignore_errors=False)
            return 1, 0
    except Exception:
        return 0, 1
    return 0, 0


def reset_archive_files() -> dict:
    removed_files = 0
    removed_dirs = 0
    errors = 0

    # Core archive artifacts.
    for p in (Path(DB_PATH), Path(FAISS_INDEX_PATH), SETUP_STATE_PATH, Path(".server.pid")):
        rf, er = _safe_unlink(p)
        removed_files += rf
        errors += er

    # Remove entire uploads and attachments trees (including nested content + thumbs).
    rd, er = _safe_rmtree(UPLOADS_DIR)
    removed_dirs += rd
    errors += er
    rd, er = _safe_rmtree(ATTACHMENTS_DIR)
    removed_dirs += rd
    errors += er

    # Clear logs that may contain sensitive details.
    if LOGS_DIR.exists():
        for p in LOGS_DIR.iterdir():
            if not p.is_file():
                continue
            if p.resolve() == APP_LOG_PATH.resolve():
                # app logger may hold this file open; truncate in place.
                try:
                    p.write_text("", encoding="utf-8")
                except Exception:
                    errors += 1
                continue
            rf, er = _safe_unlink(p)
            removed_files += rf
            errors += er

    # Remove local python bytecode caches.
    for pyc_dir in Path(".").glob("**/__pycache__"):
        rd, er = _safe_rmtree(pyc_dir)
        removed_dirs += rd
        errors += er

    # Recreate required runtime folders.
    for d in (UPLOADS_DIR, ATTACHMENTS_DIR, ATTACHMENTS_PREVIEW_DIR, LOGS_DIR):
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception:
            errors += 1

    _clear_runtime_caches()
    initialize_search_resources()

    return {
        "removed_files": int(removed_files),
        "removed_dirs": int(removed_dirs),
        "errors": int(errors),
    }


def detect_cuda_available() -> bool:
    try:
        probe = subprocess.run(
            [sys.executable, "-c", "import torch; print(int(torch.cuda.is_available()))"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        return probe.returncode == 0 and str(probe.stdout or "").strip().endswith("1")
    except Exception:
        return False


def _extract_direction_tokens(direction: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for tok in re.findall(r"[a-z]{3,}", (direction or "").lower()):
        if tok in _DIRECTION_TOKEN_IGNORE:
            continue
        if tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out


def load_participant_direction_hints() -> None:
    conn = db()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT lower(direction) AS d, COUNT(*) AS c
            FROM messages
            WHERE trim(ifnull(direction,'')) != ''
            GROUP BY d
            ORDER BY c DESC;
            """
        )
    except sqlite3.OperationalError:
        conn.close()
        return
    token_counts: dict[str, int] = {}
    for d, c in cur.fetchall():
        for tok in _extract_direction_tokens(d or ""):
            token_counts[tok] = token_counts.get(tok, 0) + int(c or 0)
    conn.close()

    ranked = [k for k, _ in sorted(token_counts.items(), key=lambda kv: kv[1], reverse=True)]
    hints_a = set(filter(None, [x.strip().lower() for x in os.environ.get("TEXTAPP_PERSON_A_HINTS", "").split(",")]))
    hints_b = set(filter(None, [x.strip().lower() for x in os.environ.get("TEXTAPP_PERSON_B_HINTS", "").split(",")]))
    if ranked:
        if len(hints_a) == 0:
            hints_a.add(ranked[0])
        if len(ranked) > 1 and len(hints_b) == 0:
            hints_b.add(ranked[1])

    PARTICIPANT_DIRECTION_HINTS["person_a"] = hints_a
    PARTICIPANT_DIRECTION_HINTS["person_b"] = hints_b


def normalize_signal_senders() -> None:
    conn = db()
    cur = conn.cursor()
    try:
        cur.execute("SELECT sender, COUNT(*) AS c FROM message_signals GROUP BY sender ORDER BY c DESC;")
    except sqlite3.OperationalError:
        conn.close()
        return
    rows = cur.fetchall()
    known = {"person_a", "person_b", "unknown"}
    extras = [str(r[0] or "").strip().lower() for r in rows if str(r[0] or "").strip().lower() not in known]
    if extras:
        mapping = {}
        mapping[extras[0]] = "person_a"
        if len(extras) > 1:
            mapping[extras[1]] = "person_b"
        for raw_sender, normalized in mapping.items():
            cur.execute(
                "UPDATE message_signals SET sender = ? WHERE lower(sender) = ?;",
                (normalized, raw_sender),
            )
        conn.commit()
    conn.close()

def ensure_bookmarks_schema():
    conn = db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS bookmark_categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            icon_key TEXT NOT NULL DEFAULT 'pink_heart',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS bookmarks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id INTEGER NOT NULL,
            category_id INTEGER NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(message_id, category_id)
        );
    """)
    cur.execute("PRAGMA table_info(bookmark_categories);")
    cols = {r[1] for r in cur.fetchall()}
    if "icon_key" not in cols:
        cur.execute("ALTER TABLE bookmark_categories ADD COLUMN icon_key TEXT NOT NULL DEFAULT 'pink_heart';")
    conn.commit()
    conn.close()

def ensure_attachments_schema():
    conn = db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS attachments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id INTEGER NOT NULL,
            kind TEXT,
            mime_type TEXT,
            filename TEXT,
            stored_path TEXT NOT NULL,
            width INTEGER,
            height INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_attachments_message_id ON attachments(message_id);")
    conn.commit()
    conn.close()

def ensure_signals_schema():
    conn = db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS message_signals (
            message_id INTEGER PRIMARY KEY,
            ts_unix INTEGER NOT NULL,
            sender TEXT NOT NULL,
            word_count INTEGER NOT NULL,
            love REAL,
            flirting REAL,
            compliments REAL,
            laughing REAL,
            supportive REAL,
            difficult REAL,
            repair REAL,
            gratitude REAL,
            missing REAL,
            top_labels_json TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_message_signals_ts_unix ON message_signals(ts_unix);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_message_signals_sender_ts ON message_signals(sender, ts_unix);")
    conn.commit()
    conn.close()

@app.on_event("startup")
def startup():
    ensure_bookmarks_schema()
    ensure_attachments_schema()
    ensure_signals_schema()
    load_participant_direction_hints()
    normalize_signal_senders()
    ATTACHMENTS_DIR.mkdir(parents=True, exist_ok=True)
    ATTACHMENTS_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

def is_bookmarked(conn, message_id: int) -> bool:
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM bookmarks WHERE message_id = ? LIMIT 1;", (message_id,))
    return cur.fetchone() is not None


def fetch_categories(conn):
    cur = conn.cursor()
    cur.execute("SELECT id, name, icon_key FROM bookmark_categories ORDER BY name;")
    rows = cur.fetchall()
    out = []
    for row in rows:
        icon_key = (row[2] or DEFAULT_ICON_KEY)
        if icon_key not in ICON_MAP:
            icon_key = DEFAULT_ICON_KEY
        out.append({
            "id": row[0],
            "name": row[1],
            "icon_key": icon_key,
            "icon_entity": ICON_MAP[icon_key]["entity"],
        })
    return out

def fetch_bookmarked_ids(conn, message_ids: List[int]) -> set[int]:
    if not message_ids:
        return set()
    placeholders = ",".join(["?"] * len(message_ids))
    cur = conn.cursor()
    cur.execute(
        f"SELECT DISTINCT message_id FROM bookmarks WHERE message_id IN ({placeholders});",
        tuple(message_ids),
    )
    return {int(r[0]) for r in cur.fetchall()}

def fetch_attachments_for_message_ids(conn, message_ids: List[int]) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {mid: [] for mid in message_ids}
    if not message_ids:
        return out

    placeholders = ",".join(["?"] * len(message_ids))
    cur = conn.cursor()
    cur.execute(
        f"""
        SELECT id, message_id, kind, mime_type, filename
        FROM attachments
        WHERE message_id IN ({placeholders})
        ORDER BY id ASC;
        """,
        tuple(message_ids),
    )
    for aid, mid, kind, mime_type, filename in cur.fetchall():
        mt = (mime_type or "").lower()
        fn = (filename or "").strip()
        if fn.lower() in {"null", "none", "(null)", "undefined"}:
            fn = ""
        out.setdefault(int(mid), []).append({
            "id": int(aid),
            "kind": kind or "",
            "mime_type": mime_type or "",
            "filename": fn,
            "is_image": mt.startswith("image/"),
            "url": f"/attachments/{aid}",
            "preview_url": f"/attachments/{aid}/preview",
        })
    return out

def _is_heic_like(mime_type: str, filename: str, stored_path: str) -> bool:
    mt = (mime_type or "").lower()
    fn = (filename or "").lower()
    sp = (stored_path or "").lower()
    return (
        "heic" in mt
        or "heif" in mt
        or fn.endswith(".heic")
        or fn.endswith(".heif")
        or sp.endswith(".heic")
        or sp.endswith(".heif")
    )

def _preview_cache_path(attachment_id: int, source_path: Path) -> Path:
    st = source_path.stat()
    stamp = f"{attachment_id}:{int(st.st_mtime)}:{st.st_size}"
    digest = hashlib.sha1(stamp.encode("utf-8")).hexdigest()[:16]
    return ATTACHMENTS_PREVIEW_DIR / f"{attachment_id}-{digest}.jpg"

def _render_preview_jpeg(source_path: Path, preview_path: Path, max_size: int = 1200):
    ATTACHMENTS_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as im:
        im = ImageOps.exif_transpose(im)
        im.thumbnail((max_size, max_size))
        if im.mode != "RGB":
            im = im.convert("RGB")
        im.save(preview_path, format="JPEG", quality=86, optimize=True)

def get_available_years(conn) -> List[int]:
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT strftime('%Y', ts_unix, 'unixepoch', 'localtime') AS y
        FROM messages
        WHERE strftime('%Y', ts_unix, 'unixepoch', 'localtime') IS NOT NULL
          AND strftime('%Y', ts_unix, 'unixepoch', 'localtime') != ''
        ORDER BY y;
    """)
    years = []
    for row in cur.fetchall():
        try:
            years.append(int(row[0]))
        except (TypeError, ValueError):
            continue
    return years

def get_rhythm_day_stats(conn, year: int) -> List[dict]:
    cur = conn.cursor()
    cur.execute("""
        SELECT
            date(ts_unix, 'unixepoch', 'localtime') AS d,
            COUNT(*) AS msg_count,
            SUM(CASE
                WHEN LENGTH(TRIM(body)) = 0 THEN 0
                ELSE LENGTH(TRIM(body)) - LENGTH(REPLACE(TRIM(body), ' ', '')) + 1
            END) AS total_words,
            AVG(LENGTH(body)) AS avg_len,
            SUM(CASE
                WHEN strftime('%H', ts_unix, 'unixepoch', 'localtime') IN ('23','00','01','02','03','04','05') THEN 1
                ELSE 0
            END) AS late_night_count
        FROM messages
        WHERE strftime('%Y', ts_unix, 'unixepoch', 'localtime') = ?
        GROUP BY d
        ORDER BY d;
    """, (str(year),))
    base_rows = cur.fetchall()
    cur.execute(
        """
        SELECT
            date(ts_unix, 'unixepoch', 'localtime') AS d,
            direction,
            COUNT(*) AS c
        FROM messages
        WHERE strftime('%Y', ts_unix, 'unixepoch', 'localtime') = ?
        GROUP BY d, direction;
        """,
        (str(year),),
    )
    sender_counts: dict[str, dict[str, int]] = {}
    for d, direction, count in cur.fetchall():
        bucket = detect_sender(direction or "")
        if bucket not in ("person_a", "person_b"):
            continue
        sender_counts.setdefault(d, {"person_a": 0, "person_b": 0})
        sender_counts[d][bucket] += int(count or 0)

    out = []
    for row in base_rows:
        msg_count = int(row[1] or 0)
        late_night_count = int(row[4] or 0)
        day_counts = sender_counts.get(row[0], {"person_a": 0, "person_b": 0})
        out.append({
            "date": row[0],
            "msg_count": msg_count,
            "person_a_count": int(day_counts.get("person_a", 0)),
            "person_b_count": int(day_counts.get("person_b", 0)),
            "total_words": int(row[2] or 0),
            "avg_len": float(row[3] or 0.0),
            "late_night_count": late_night_count,
            "late_night_pct": (late_night_count / msg_count * 100.0) if msg_count else 0.0,
        })
    return out

def get_rhythm_hour_grid(conn, year: int) -> List[List[int]]:
    cur = conn.cursor()
    grid = [[0 for _ in range(24)] for _ in range(7)]  # 0=Sun ... 6=Sat
    cur.execute("""
        SELECT
            strftime('%w', ts_unix, 'unixepoch', 'localtime') AS dow,
            strftime('%H', ts_unix, 'unixepoch', 'localtime') AS hour,
            COUNT(*) AS c
        FROM messages
        WHERE strftime('%Y', ts_unix, 'unixepoch', 'localtime') = ?
        GROUP BY dow, hour;
    """, (str(year),))
    for dow, hour, count in cur.fetchall():
        di = int(dow)
        hi = int(hour)
        grid[di][hi] = int(count or 0)
    return grid


def get_top_rhythm_month_preview(conn) -> dict | None:
    cur = conn.cursor()
    cur.execute("""
        SELECT
            strftime('%Y-%m', ts_unix, 'unixepoch', 'localtime') AS ym,
            COUNT(*) AS c
        FROM messages
        GROUP BY ym
        ORDER BY c DESC, ym DESC
        LIMIT 1;
    """)
    row = cur.fetchone()
    if not row or not row[0]:
        return None

    ym = str(row[0])
    total_messages = int(row[1] or 0)
    year = int(ym[:4])
    month = int(ym[5:7])

    cur.execute("""
        SELECT
            CAST(strftime('%d', ts_unix, 'unixepoch', 'localtime') AS INTEGER) AS day_num,
            COUNT(*) AS c
        FROM messages
        WHERE strftime('%Y-%m', ts_unix, 'unixepoch', 'localtime') = ?
        GROUP BY day_num;
    """, (ym,))
    by_day = {int(r[0]): int(r[1] or 0) for r in cur.fetchall()}
    days_in_month = calendar.monthrange(year, month)[1]
    first_dow = datetime(year, month, 1).weekday()  # Mon=0..Sun=6
    first_dow_sun0 = (first_dow + 1) % 7
    max_count = max(by_day.values()) if by_day else 0

    total_cells = math.ceil((first_dow_sun0 + days_in_month) / 7) * 7
    cells = []
    for i in range(total_cells):
        day_num = i - first_dow_sun0 + 1
        if day_num < 1 or day_num > days_in_month:
            cells.append({
                "in_month": False,
                "day": 0,
                "count": 0,
                "level": 0,
                "date": "",
            })
            continue

        count = int(by_day.get(day_num, 0))
        if max_count <= 0 or count <= 0:
            level = 0
        else:
            ratio = count / max_count
            if ratio <= 0.20:
                level = 1
            elif ratio <= 0.40:
                level = 2
            elif ratio <= 0.60:
                level = 3
            elif ratio <= 0.80:
                level = 4
            else:
                level = 5

        cells.append({
            "in_month": True,
            "day": day_num,
            "count": count,
            "level": level,
            "date": f"{year:04d}-{month:02d}-{day_num:02d}",
        })

    month_label = datetime(year, month, 1).strftime("%b %Y")
    return {
        "year": year,
        "month": month,
        "month_label": month_label,
        "total_messages": total_messages,
        "weekday_labels": ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"],
        "cells": cells,
    }


def build_signals_love_preview(conn, bins_to_show: int | None = None) -> dict | None:
    threshold = float(SIGNAL_THRESHOLDS.get("love", 0.67))
    data = get_signal_bins(conn, signal="love", bin_days=30, threshold=threshold)
    bins = data.get("bins", [])
    if not bins:
        return None

    if bins_to_show is None:
        use_bins = bins
    else:
        use_bins = bins[-max(6, int(bins_to_show)):]
    values = [float(b.get("combined_rate", 0.0)) for b in use_bins]
    v_max = max(values) if values else 0.0
    width, height, pad = 360, 110, 10
    inner_w = max(1, width - pad * 2)
    inner_h = max(1, height - pad * 2)

    points = []
    for i, v in enumerate(values):
        x = pad if len(values) <= 1 else (pad + (i / (len(values) - 1)) * inner_w)
        y = pad + inner_h * (1.0 - (v / v_max if v_max > 0 else 0.0))
        points.append(f"{x:.1f},{y:.1f}")

    start_label = datetime.fromtimestamp(int(use_bins[0]["start_ts"])).strftime("%b %Y")
    end_label = datetime.fromtimestamp(int(use_bins[-1]["end_ts"]) - 1).strftime("%b %Y")
    min_ts = data.get("min_ts")
    max_ts = data.get("max_ts")
    start_pretty = format_pretty_date(int(min_ts)) if min_ts is not None else start_label
    end_pretty = format_pretty_date(int(max_ts)) if max_ts is not None else end_label
    return {
        "signal": "love",
        "bin_days": 30,
        "series": "combined",
        "metric": "rate",
        "svg_width": width,
        "svg_height": height,
        "polyline_points": " ".join(points),
        "values": values,
        "max_rate": v_max,
        "start_label": start_label,
        "end_label": end_label,
        "start_pretty": start_pretty,
        "end_pretty": end_pretty,
    }


def build_trends_preview(
    conn: sqlite3.Connection,
    terms: List[str] | None = None,
    bin_days: int = 30,
    bins_to_show: int | None = 18,
) -> dict | None:
    preview_terms = [normalize_query_text(t) for t in (terms or ["love", "laugh"])]
    preview_terms = [t for t in preview_terms if t]
    if not preview_terms:
        return None

    min_ts, max_ts = get_message_time_bounds(conn)
    if min_ts is None or max_ts is None:
        return None

    all_series: list[dict] = []
    bins_ref: list[dict] | None = None
    global_max = 0
    first_term_scores: dict[int, float] | None = None

    palette = ["#a58fdc", "#74c4bd", "#d7a06a", "#9b8ad2", "#8ea2df"]
    for idx, term in enumerate(preview_terms):
        scores = compute_trend_match_scores(conn, term)
        if idx == 0:
            first_term_scores = scores
        _, bins, _ = build_term_bins(conn, term, scores, min_ts, max_ts, bin_days)
        use_bins = bins[-max(6, int(bins_to_show)):] if bins_to_show is not None else bins
        counts = [int(b.get("count", 0)) for b in use_bins]
        if counts:
            global_max = max(global_max, max(counts))
        if bins_ref is None:
            bins_ref = use_bins
        all_series.append({
            "term": term,
            "counts": counts,
            "color": palette[idx % len(palette)],
        })

    if not bins_ref:
        return None

    width, height, pad = 360, 110, 10
    inner_w = max(1, width - pad * 2)
    inner_h = max(1, height - pad * 2)
    denom = max(1, global_max)

    lines = []
    for series in all_series:
        vals = series["counts"]
        points = []
        for i, v in enumerate(vals):
            x = pad if len(vals) <= 1 else (pad + (i / (len(vals) - 1)) * inner_w)
            y = pad + inner_h * (1.0 - (float(v) / float(denom)))
            points.append(f"{x:.1f},{y:.1f}")
        lines.append({
            "term": series["term"],
            "color": series["color"],
            "points": " ".join(points),
        })

    top_message = None
    if first_term_scores:
        ranked = sorted(first_term_scores.items(), key=lambda x: x[1], reverse=True)[:1]
        if ranked:
            top_id = int(ranked[0][0])
            msgs = fetch_messages_by_ids(conn, [top_id])
            if msgs:
                top_message = msgs[0]

    start_pretty = format_pretty_date(int(bins_ref[0]["start_ts"])) if bins_ref else ""
    end_pretty = format_pretty_date(int((bins_ref[-1]["end_ts"] - 1))) if bins_ref else ""
    return {
        "terms": preview_terms,
        "bin_days": int(bin_days),
        "svg_width": width,
        "svg_height": height,
        "lines": lines,
        "start_pretty": start_pretty,
        "end_pretty": end_pretty,
        "top_message": top_message,
    }


def get_top_signal_match_preview(conn: sqlite3.Connection, signal: str = "love", series: str = "combined") -> dict | None:
    if signal not in SIGNAL_COLUMNS:
        return None
    signal_col = SIGNAL_COLUMNS[signal]
    where_sql = ""
    params: list = []
    if series == "person_a":
        where_sql = "WHERE ms.sender = ?"
        params.append("person_a")
    elif series == "person_b":
        where_sql = "WHERE ms.sender = ?"
        params.append("person_b")

    cur = conn.cursor()
    cur.execute(f"""
        SELECT
            m.id,
            m.ts,
            m.ts_unix,
            m.direction,
            m.body,
            ms.sender,
            ms.{signal_col} AS signal_score
        FROM messages m
        JOIN message_signals ms ON ms.message_id = m.id
        {where_sql}
        ORDER BY ms.{signal_col} DESC, m.ts_unix DESC, m.id DESC
        LIMIT 1;
    """, tuple(params))
    row = cur.fetchone()
    if not row:
        return None

    out = {
        "id": int(row[0]),
        "ts": row[1],
        "ts_unix": int(row[2] or 0),
        "direction": row[3] or "",
        "body": row[4] or "",
        "sender": row[5] or detect_sender(row[3] or ""),
        "signal_score": float(row[6] or 0.0),
    }
    add_display_timestamp_fields([out])
    return out


def get_day_highlight_bundle(conn, date_str: str) -> dict:
    cur = conn.cursor()
    cur.execute("""
        SELECT
            COUNT(*) AS msg_count,
            SUM(CASE
                WHEN LENGTH(TRIM(body)) = 0 THEN 0
                ELSE LENGTH(TRIM(body)) - LENGTH(REPLACE(TRIM(body), ' ', '')) + 1
            END) AS total_words,
            AVG(LENGTH(body)) AS avg_len,
            SUM(CASE
                WHEN strftime('%H', ts_unix, 'unixepoch', 'localtime') IN ('23','00','01','02','03','04','05') THEN 1
                ELSE 0
            END) AS late_night_count
        FROM messages
        WHERE date(ts_unix, 'unixepoch', 'localtime') = ?;
    """, (date_str,))
    row = cur.fetchone()

    cur.execute("""
        SELECT
            m.id,
            m.ts,
            m.direction,
            m.body,
            m.ts_unix,
            ms.word_count,
            ms.love,
            ms.flirting,
            ms.compliments,
            ms.laughing,
            ms.supportive,
            ms.difficult,
            ms.repair,
            ms.gratitude,
            ms.missing
        FROM messages m
        LEFT JOIN message_signals ms ON ms.message_id = m.id
        WHERE date(m.ts_unix, 'unixepoch', 'localtime') = ?
        ORDER BY m.ts_unix ASC, m.id ASC;
    """, (date_str,))
    rows = cur.fetchall()

    signal_keys = ["love", "flirting", "compliments", "laughing", "supportive", "difficult", "repair", "gratitude", "missing"]
    day_thresholds = {k: (SIGNAL_THRESHOLDS[k] * DAY_HIGHLIGHT_THRESHOLD_SCALE) for k in signal_keys}
    scored_messages = []
    person_a_count = 0
    person_b_count = 0
    for r in rows:
        direction = r[2] or ""
        body = r[3] or ""
        wc = int(r[5] or 0)
        if wc <= 0:
            wc = len(body.split()) if body.strip() else 0

        signal_scores = {
            "love": float(r[6] or 0.0),
            "flirting": float(r[7] or 0.0),
            "compliments": float(r[8] or 0.0),
            "laughing": float(r[9] or 0.0),
            "supportive": float(r[10] or 0.0),
            "difficult": float(r[11] or 0.0),
            "repair": float(r[12] or 0.0),
            "gratitude": float(r[13] or 0.0),
            "missing": float(r[14] or 0.0),
        }
        hit_signals = [k for k in signal_keys if signal_scores[k] >= day_thresholds[k]]
        sender = detect_sender(direction)
        if sender == "person_a":
            person_a_count += 1
        elif sender == "person_b":
            person_b_count += 1

        scored_messages.append({
            "id": r[0],
            "ts": r[1],
            "direction": direction,
            "body": body,
            "ts_unix": int(r[4] or 0),
            "word_count": wc,
            "sender": sender,
            "hit_signals": hit_signals,
            "hit_count": len(hit_signals),
            "qualifies": (wc >= DAY_HIGHLIGHT_MIN_WORDS and len(hit_signals) > 0),
        })

    highlighted = [m for m in scored_messages if m["qualifies"]]
    highlighted.sort(key=lambda m: (-m["hit_count"], -m["word_count"], m["ts_unix"], m["id"]))

    if not highlighted:
        highlighted = sorted(scored_messages, key=lambda m: (-m["word_count"], m["ts_unix"], m["id"]))[:80]

    add_display_timestamp_fields(highlighted)
    msg_count = int((row[0] if row and row[0] is not None else 0))
    late_night_count = int((row[3] if row and row[3] is not None else 0))
    summary = {
        "msg_count": msg_count,
        "person_a_count": person_a_count,
        "person_b_count": person_b_count,
        "total_words": int((row[1] if row and row[1] is not None else 0)),
        "avg_len": float((row[2] if row and row[2] is not None else 0.0)),
        "late_night_count": late_night_count,
        "late_night_pct": (late_night_count / msg_count * 100.0) if msg_count else 0.0,
    }
    first_message_id = scored_messages[0]["id"] if scored_messages else None
    return {
        "summary": summary,
        "messages": highlighted,
        "displayed_count": len(highlighted),
        "first_message_id": first_message_id,
    }


def get_on_this_day_preview(conn: sqlite3.Connection, years_ago: int = 1) -> dict | None:
    anchor = datetime.now()
    target_year = anchor.year - max(1, int(years_ago))
    month = anchor.month
    day = anchor.day

    # Handle leap-day edge case by falling back to Feb 28 on non-leap years.
    try:
        target_dt = datetime(target_year, month, day)
    except ValueError:
        if month == 2 and day == 29:
            target_dt = datetime(target_year, 2, 28)
        else:
            return None

    date_str = target_dt.strftime("%Y-%m-%d")
    bundle = get_day_highlight_bundle(conn, date_str)
    if bundle["summary"]["msg_count"] <= 0 or not bundle["messages"]:
        return None

    center_id = int(bundle["messages"][0]["id"])
    msgs = get_result_context_batch(conn, center_id, before=2, after=2)
    if not msgs:
        return None

    back_to = f"/on-this-day?date={anchor.strftime('%Y-%m-%d')}"
    return {
        "date_str": date_str,
        "pretty_date": format_pretty_date(int(target_dt.timestamp())),
        "subtitle_date": f"{target_dt.strftime('%b')} {target_dt.day} {target_dt.year}",
        "center_id": center_id,
        "center_href": build_chat_href(center_id, back_to, "On This Day"),
        "messages": msgs,
    }

def get_signal_time_bounds(conn) -> tuple[int | None, int | None]:
    cur = conn.cursor()
    cur.execute("SELECT MIN(ts_unix), MAX(ts_unix) FROM message_signals;")
    row = cur.fetchone()
    if not row or row[0] is None or row[1] is None:
        return None, None
    return int(row[0]), int(row[1])

def get_signal_bins(conn, signal: str, bin_days: int, threshold: float) -> dict:
    col = SIGNAL_COLUMNS[signal]
    bin_seconds = int(bin_days) * 86400
    min_ts, max_ts = get_signal_time_bounds(conn)
    if min_ts is None or max_ts is None:
        return {
            "min_ts": None,
            "max_ts": None,
            "bin_seconds": bin_seconds,
            "bins": [],
        }

    cur = conn.cursor()
    cur.execute(f"""
        SELECT
            CAST((ts_unix - ?) / ? AS INTEGER) AS bin_idx,
            COUNT(*) AS total_msgs,
            AVG(word_count) AS avg_word_count,
            SUM(CASE WHEN {col} >= ? THEN 1 ELSE 0 END) AS combined_hits,
            SUM(CASE WHEN sender = 'person_a' THEN 1 ELSE 0 END) AS person_a_total,
            SUM(CASE WHEN sender = 'person_b' THEN 1 ELSE 0 END) AS person_b_total,
            SUM(CASE WHEN sender = 'person_a' AND {col} >= ? THEN 1 ELSE 0 END) AS person_a_hits,
            SUM(CASE WHEN sender = 'person_b' AND {col} >= ? THEN 1 ELSE 0 END) AS person_b_hits
        FROM message_signals
        GROUP BY bin_idx
        ORDER BY bin_idx ASC;
    """, (min_ts, bin_seconds, threshold, threshold, threshold))
    rows = cur.fetchall()
    by_idx = {int(r[0]): r for r in rows}

    min_idx = 0
    max_idx = (max_ts - min_ts) // bin_seconds
    bins = []
    for idx in range(min_idx, max_idx + 1):
        start_ts = min_ts + idx * bin_seconds
        end_ts = start_ts + bin_seconds
        r = by_idx.get(idx)
        if r:
            total_msgs = int(r[1] or 0)
            avg_word_count = float(r[2] or 0.0)
            combined_hits = int(r[3] or 0)
            person_a_total = int(r[4] or 0)
            person_b_total = int(r[5] or 0)
            person_a_hits = int(r[6] or 0)
            person_b_hits = int(r[7] or 0)
        else:
            total_msgs = 0
            avg_word_count = 0.0
            combined_hits = 0
            person_a_total = 0
            person_b_total = 0
            person_a_hits = 0
            person_b_hits = 0

        bins.append({
            "start_ts": start_ts,
            "end_ts": end_ts,
            "label": datetime.fromtimestamp(start_ts).strftime("%Y-%m-%d"),
            "total_msgs": total_msgs,
            "avg_word_count": avg_word_count,
            "combined_value": combined_hits,
            "person_a_value": person_a_hits,
            "person_b_value": person_b_hits,
            "combined_rate": (combined_hits / total_msgs) if total_msgs else 0.0,
            "person_a_rate": (person_a_hits / person_a_total) if person_a_total else 0.0,
            "person_b_rate": (person_b_hits / person_b_total) if person_b_total else 0.0,
            "person_a_total_msgs": person_a_total,
            "person_b_total_msgs": person_b_total,
        })

    return {
        "min_ts": min_ts,
        "max_ts": max_ts,
        "bin_seconds": bin_seconds,
        "bins": bins,
    }

def ordinal_suffix(day: int) -> str:
    if 11 <= day % 100 <= 13:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")

def format_pretty_date(ts: int) -> str:
    dt = datetime.fromtimestamp(ts)
    return f"{dt.strftime('%b')} {dt.day}{ordinal_suffix(dt.day)} {dt.year}"

def detect_sender(direction: str) -> str:
    d = (direction or "").lower()
    # Explicit direct markers first.
    if "person_a" in d or "curtis" in d:
        return "person_a"
    if "person_b" in d or "ollie" in d:
        return "person_b"

    # Common export semantics.
    if ("sent" in d or "outgoing" in d) and ("received" not in d and "incoming" not in d):
        return "person_a"
    if ("received" in d or "incoming" in d) and ("sent" not in d and "outgoing" not in d):
        return "person_b"
    for tok in PARTICIPANT_DIRECTION_HINTS["person_a"]:
        if re.search(rf"\b{re.escape(tok)}\b", d):
            return "person_a"
    for tok in PARTICIPANT_DIRECTION_HINTS["person_b"]:
        if re.search(rf"\b{re.escape(tok)}\b", d):
            return "person_b"
    return "unknown"

def parse_message_datetime(ts: str | None, ts_unix: int | None = None) -> datetime | None:
    if ts_unix is not None:
        try:
            return datetime.fromtimestamp(int(ts_unix))
        except (TypeError, ValueError, OSError):
            pass

    t = (ts or "").strip()
    if not t:
        return None

    try:
        return datetime.fromisoformat(t.replace("Z", "+00:00"))
    except ValueError:
        pass

    if len(t) >= 19:
        head = t[:19]
        for fmt in ("%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
            try:
                return datetime.strptime(head, fmt)
            except ValueError:
                continue

    m = re.search(r"(\d{4})-(\d{2})-(\d{2}).*?(\d{1,2}):(\d{2})", t)
    if m:
        try:
            y, mo, d, hh, mm = [int(x) for x in m.groups()]
            return datetime(y, mo, d, hh, mm)
        except ValueError:
            return None
    return None

def format_message_display(ts: str | None, ts_unix: int | None = None) -> tuple[str, str]:
    dt = parse_message_datetime(ts, ts_unix)
    if not dt:
        return "", ""
    date_txt = f"{dt.strftime('%b')} {dt.day}{ordinal_suffix(dt.day)} {dt.year}"
    time_txt = dt.strftime("%H:%M").lstrip("0")
    if time_txt.startswith(":"):
        time_txt = "0" + time_txt
    return date_txt, time_txt


def is_laugh_message(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    for pat in LAUGH_TOKEN_PATTERNS:
        if pat.search(t):
            return True
    for pat in LAUGH_PHRASE_PATTERNS:
        if pat.search(t):
            return True
    for em in LAUGH_EMOJIS:
        if em in t:
            return True
    return False


def laugh_intensity_score(text: str) -> float:
    t = (text or "").strip()
    if not t:
        return 0.0

    score = 0.0
    low = t.lower()

    # Core laughter token weights.
    score += len(re.findall(r"\blol+\b", low)) * 1.2
    score += len(re.findall(r"\blmao+\b", low)) * 1.7
    score += len(re.findall(r"\blmfao+\b", low)) * 1.9
    score += len(re.findall(r"\brofl+\b", low)) * 1.9

    # "haha" style repeated laughter: longer chain => higher intensity.
    for m in re.finditer(r"(ha){2,}", low):
        reps = max(2, len(m.group(0)) // 2)
        score += min(0.5 * reps, 6.0)
    for m in re.finditer(r"(he){2,}", low):
        reps = max(2, len(m.group(0)) // 2)
        score += min(0.35 * reps, 3.0)

    # Phrase-based laughter.
    if re.search(r"\bthat(?:'| i)?s so funny\b", low):
        score += 1.8
    elif re.search(r"\bthat(?:'| i)?s funny\b", low):
        score += 1.2
    if re.search(r"\bi'?m dead\b", low):
        score += 1.4

    # Punctuation and emphasis.
    exclam = t.count("!")
    score += min(exclam * 0.22, 2.0)
    multi_punct = len(re.findall(r"[!?]{2,}", t))
    score += min(0.4 * multi_punct, 1.2)

    # Caps ratio indicates stronger emotional intensity.
    letters = [c for c in t if c.isalpha()]
    if letters:
        upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
        if upper_ratio >= 0.7:
            score += 1.3
        elif upper_ratio >= 0.45:
            score += 0.8

    # Laugh emojis.
    score += sum(t.count(em) for em in LAUGH_EMOJIS) * 0.9

    return round(score, 3)

def add_display_timestamp_fields(messages: List[dict]) -> List[dict]:
    label_a, label_b = get_participant_labels()
    for m in messages:
        if not m.get("sender"):
            m["sender"] = detect_sender(m.get("direction") or "")
        if m["sender"] == "person_a":
            m["sender_label"] = label_a
        elif m["sender"] == "person_b":
            m["sender_label"] = label_b
        else:
            m["sender_label"] = (m.get("direction") or "Unknown")
        d, t = format_message_display(m.get("ts"), m.get("ts_unix"))
        m["display_date"] = d
        m["display_time"] = t
        dt = parse_message_datetime(m.get("ts"), m.get("ts_unix"))
        m["display_date_key"] = dt.strftime("%Y-%m-%d") if dt else ""
    return messages

def sanitize_internal_return_to(path: str | None, fallback: str = "/") -> str:
    p = (path or "").strip()
    if not p.startswith("/") or p.startswith("//"):
        return fallback
    if p.startswith("/chat/"):
        return fallback
    return p

def build_chat_href(message_id: int, return_to: str, return_label: str) -> str:
    q = urlencode({"return_to": return_to, "return_label": return_label})
    return f"/chat/{message_id}?{q}"

def require_auth(request: Request) -> bool:
    # super simple cookie auth
    return request.cookies.get("tauth") == "1"

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    authed = require_auth(request)
    if not setup_is_complete():
        return RedirectResponse(url="/setup", status_code=303)
    if not authed:
        return templates.TemplateResponse("login_only.html", {
            "request": request,
        })

    feature_previews = {}
    conn = db()
    try:
        feature_previews = build_feature_previews(conn)
    finally:
        conn.close()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "authed": authed,
        "icon_options": ICON_OPTIONS,
        "default_icon_key": DEFAULT_ICON_KEY,
        "feature_previews": feature_previews,
    })

@app.get("/setup", response_class=HTMLResponse)
def setup_page(request: Request):
    if setup_is_complete():
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse("setup.html", {"request": request})


@app.post("/setup", response_class=HTMLResponse)
async def setup_complete(
    request: Request,
    data_file: UploadFile = File(...),
    your_name: str = Form(...),
    partner_name: str = Form(...),
    password: str = Form(...),
    password_confirm: str = Form(...),
):
    if setup_is_complete():
        return RedirectResponse(url="/", status_code=303)

    pwd = (password or "").strip()
    pwd2 = (password_confirm or "").strip()
    your_name_txt = (your_name or "").strip()
    partner_name_txt = (partner_name or "").strip()
    if len(pwd) < 4:
        return _render_setup_result(request, False, "Setup failed.", "Password must be at least 4 characters.")
    if not your_name_txt or not partner_name_txt:
        return _render_setup_result(request, False, "Setup failed.", "Please enter both names.")
    if pwd != pwd2:
        return _render_setup_result(request, False, "Setup failed.", "Passwords do not match.")

    raw_name = (data_file.filename or "").strip()
    suffix = Path(raw_name).suffix.lower()
    if suffix not in {".xml", ".csv"}:
        return _render_setup_result(request, False, "Setup failed.", "Please upload an XML or CSV file.")

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    xml_path = UPLOADS_DIR / "messages.xml"
    csv_path = UPLOADS_DIR / "long_messages.csv"
    target_path = xml_path if suffix == ".xml" else csv_path
    other_path = csv_path if suffix == ".xml" else xml_path

    try:
        content = await data_file.read()
        if not content:
            return _render_setup_result(request, False, "Setup failed.", "Uploaded file is empty.")
        target_path.write_bytes(content)
        if other_path.exists():
            other_path.unlink()
    except Exception:
        log_event("error", "setup: failed to save uploaded file")
        return _render_setup_result(request, False, "Setup failed.", "Could not save uploaded file.")

    is_valid, validation_msg = validate_uploaded_source(target_path, suffix)
    if not is_valid:
        log_event("warning", f"setup: validation failed for {raw_name}: {validation_msg}")
        return _render_setup_result(request, False, "Setup failed.", validation_msg)

    cmd = [
        sys.executable,
        "build_db.py",
        "--xml", str(xml_path),
        "--csv", str(csv_path),
        "--db", DB_PATH,
        "--faiss", FAISS_INDEX_PATH,
        "--attachments-dir", str(ATTACHMENTS_DIR),
    ]
    use_cuda = detect_cuda_available()
    device_arg = "cuda" if use_cuda else "cpu"
    cmd.extend(["--device", device_arg])
    try:
        log_event("info", f"setup: build started ({raw_name}) device={device_arg}")
        proc = subprocess.run(
            cmd,
            cwd=str(Path(__file__).resolve().parent),
            check=False,
            timeout=SETUP_BUILD_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        log_event("error", f"setup: build timed out ({raw_name}) timeout={SETUP_BUILD_TIMEOUT_SECONDS}s")
        return _render_setup_result(
            request,
            False,
            "Build timed out.",
            f"The message-database build exceeded {SETUP_BUILD_TIMEOUT_SECONDS//60} minutes. You can retry setup from the button below. "
            "If this keeps happening, try a smaller export.",
            error_blob=_tail_app_log(),
        )
    except Exception:
        log_event("error", "setup: build process failed to start")
        return _render_setup_result(request, False, "Setup failed.", "Failed to start the data build process.")

    if proc.returncode != 0:
        log_event("error", f"setup: build failed ({raw_name}) rc={proc.returncode}")
        return _render_setup_result(
            request,
            False,
            "Build failed.",
            f"Database/embedding build exited with code {proc.returncode}. Retry setup, and if it fails again use the copied error details below.",
            error_blob=_tail_app_log(),
        )

    # Defensive cleanup: remove any stale signal rows before scoring this archive.
    try:
        conn_cleanup = db()
        cur_cleanup = conn_cleanup.cursor()
        ensure_signals_schema()
        cur_cleanup.execute("DELETE FROM message_signals;")
        conn_cleanup.commit()
        conn_cleanup.close()
    except Exception as exc:
        log_event("warning", f"setup: could not pre-clear signal rows: {str(exc)[:300]}")

    signals_cmd = [sys.executable, "build_signals.py", "--device", device_arg]
    try:
        log_event("info", f"setup: signal build started ({raw_name}) device={device_arg}")
        signals_proc = subprocess.run(
            signals_cmd,
            cwd=str(Path(__file__).resolve().parent),
            check=False,
            timeout=SETUP_BUILD_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        log_event("error", f"setup: signal build timed out ({raw_name}) timeout={SETUP_BUILD_TIMEOUT_SECONDS}s")
        return _render_setup_result(
            request,
            False,
            "Signal build timed out.",
            f"Signal analysis exceeded {SETUP_BUILD_TIMEOUT_SECONDS//60} minutes. You can retry setup from the button below.",
            error_blob=_tail_app_log(),
        )
    except Exception:
        log_event("error", "setup: signal build process failed to start")
        return _render_setup_result(request, False, "Setup failed.", "Failed to start signal analysis build.")

    if signals_proc.returncode != 0:
        log_event("error", f"setup: signal build failed ({raw_name}) rc={signals_proc.returncode}")
        return _render_setup_result(
            request,
            False,
            "Signal build failed.",
            f"Signal scoring exited with code {signals_proc.returncode}. Retry setup, and if it fails again use the copied error details below.",
            error_blob=_tail_app_log(),
        )

    try:
        set_configured_password(pwd)
        state = load_setup_state()
        state["source_file_name"] = raw_name
        # UI naming convention (flipped): person_a = you, person_b = partner.
        state["participant_a_label"] = your_name_txt
        state["participant_b_label"] = partner_name_txt
        save_setup_state(state)
        apply_participant_labels()
        ensure_signals_schema()
        initialize_search_resources()
        load_participant_direction_hints()
        normalize_signal_senders()
        log_event("info", f"setup: build completed ({raw_name})")
    except Exception as exc:
        log_event("error", f"setup: post-build finalize failed ({raw_name}) detail={str(exc)[:500]}")
        return _render_setup_result(
            request,
            False,
            "Setup failed after build.",
            f"Build succeeded, but final setup steps failed: {str(exc)[:800]}",
            error_blob=_tail_app_log(),
        )

    resp = _render_setup_result(
        request,
        True,
        "Archive built successfully.",
        "You can continue to the homepage now. If needed, you can also reset and run setup again.",
    )
    resp.set_cookie("tauth", "1", httponly=True, samesite="lax")
    return resp


@app.post("/setup/reset")
def setup_reset(request: Request, confirm: str = Form("")):
    if (confirm or "").strip().upper() != "RESET":
        return HTMLResponse("Reset confirmation missing", status_code=400)
    summary = reset_archive_files()
    apply_participant_labels()
    log_event(
        "warning",
        "setup: archive reset requested "
        f"(files={summary.get('removed_files', 0)}, dirs={summary.get('removed_dirs', 0)}, errors={summary.get('errors', 0)})"
    )
    resp = RedirectResponse(url="/setup", status_code=303)
    resp.delete_cookie("tauth")
    return resp


@app.get("/features", response_class=HTMLResponse)
def features_page(request: Request):
    if not require_auth(request):
        return HTMLResponse("Not authorized", status_code=401)

    conn = db()
    try:
        feature_previews = build_feature_previews(conn)
    finally:
        conn.close()

    feature_cards = [
        {
            "title": "Rhythm of Us",
            "subtitle": "Calendar + daily rhythm over time.",
            "href": "/rhythm",
            "kind": "rhythm",
        },
        {
            "title": "Cherished Memories",
            "subtitle": "Open and manage your folders of bookmarked texts.",
            "href": "/#bookmark-folders",
            "kind": "bookmarks",
        },
        {
            "title": "Times We Laughed Together",
            "subtitle": "Strong laugh-response moments with context windows.",
            "href": "/laughs",
            "kind": "laugh",
        },
        {
            "title": "On This Day",
            "subtitle": "See what happened on this date in prior years.",
            "href": "/on-this-day",
            "kind": "on_this_day",
        },
        {
            "title": "Signals Timeline",
            "subtitle": "Love/laughing/supportive signals over seasons.",
            "href": "/rhythm/signals",
            "kind": "signals",
        },
        {
            "title": "Explore our Trends",
            "subtitle": "Compare terms and see how often they appeared.",
            "href": "/trends",
            "kind": "trends",
        },
    ]

    return templates.TemplateResponse("features.html", {
        "request": request,
        "feature_previews": feature_previews,
        "feature_cards": feature_cards,
        "icon_options": ICON_OPTIONS,
        "default_icon_key": DEFAULT_ICON_KEY,
    })

@app.get("/rhythm", response_class=HTMLResponse)
def rhythm(request: Request, year: int | None = None):
    if not require_auth(request):
        return HTMLResponse("Not authorized", status_code=401)

    conn = db()
    years = get_available_years(conn)
    conn.close()
    default_year = (year if year in years else (years[-1] if years else None))

    return templates.TemplateResponse("rhythm.html", {
        "request": request,
        "years": years,
        "default_year": default_year,
    })


@app.get("/laughs", response_class=HTMLResponse)
def laughs_page(request: Request, limit: int = 25):
    if not require_auth(request):
        return HTMLResponse("Not authorized", status_code=401)

    safe_limit = max(1, min(int(limit), 100))
    conn = db()
    picked = get_laugh_ranked(conn, limit=safe_limit)
    result_batches = []
    return_to = "/laughs"
    for trigger_id, total_score, meta in picked:
        msgs = get_result_context_batch(conn, int(trigger_id), before=2, after=2)
        if not msgs:
            continue
        result_batches.append({
            "center_id": int(trigger_id),
            "center_href": build_chat_href(int(trigger_id), return_to, "Times We Laughed"),
            "messages": msgs,
            "laugh_score": round(float(total_score), 2),
            "laugh_count": int(meta["laugh_count"]),
            "laugh_sender": meta["best_laugh_sender"],
            "laugh_text": meta["best_laugh_text"],
        })
    conn.close()

    return templates.TemplateResponse("laughs.html", {
        "request": request,
        "result_batches": result_batches,
        "limit": safe_limit,
    })


@app.get("/longest-messages", response_class=HTMLResponse)
def longest_messages_page(request: Request, limit: int = 100):
    if not require_auth(request):
        return HTMLResponse("Not authorized", status_code=401)

    safe_limit = max(1, min(int(limit), 200))
    conn = db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            id,
            CASE
                WHEN LENGTH(TRIM(body)) = 0 THEN 0
                ELSE LENGTH(TRIM(body)) - LENGTH(REPLACE(TRIM(body), ' ', '')) + 1
            END AS wc,
            LENGTH(body) AS body_len
        FROM messages
        ORDER BY wc DESC, body_len DESC, id DESC
        LIMIT ?;
        """,
        (safe_limit,),
    )
    rows = cur.fetchall()

    result_batches = []
    longest_id = int(rows[0][0]) if rows else None
    return_to = "/longest-messages"
    for r in rows:
        mid = int(r[0])
        wc = int(r[1] or 0)
        msgs = get_result_context_batch(conn, mid, before=2, after=2)
        if not msgs:
            continue
        result_batches.append({
            "center_id": mid,
            "center_href": build_chat_href(mid, return_to, "Longest Messages"),
            "messages": msgs,
            "word_count": wc,
            "is_longest": (mid == longest_id),
        })
    conn.close()

    return templates.TemplateResponse("longest_messages.html", {
        "request": request,
        "result_batches": result_batches,
        "limit": safe_limit,
        "longest_id": longest_id,
    })


def get_laugh_ranked(conn: sqlite3.Connection, limit: int = 25) -> List[tuple[int, float, dict]]:
    cur = conn.cursor()
    cur.execute("""
        SELECT id, ts, ts_unix, direction, body
        FROM messages
        ORDER BY ts_unix ASC, id ASC;
    """)
    rows = cur.fetchall()

    # Candidate laugh responses linked to the preceding message.
    candidates = []
    prev = None
    for r in rows:
        msg = {
            "id": int(r[0]),
            "ts": r[1],
            "ts_unix": int(r[2] or 0),
            "direction": r[3] or "",
            "body": r[4] or "",
        }
        if prev and is_laugh_message(msg["body"]):
            score = laugh_intensity_score(msg["body"])
            if score > 0:
                prev_sender = detect_sender(prev["direction"])
                cur_sender = detect_sender(msg["direction"])
                if prev_sender == cur_sender and prev_sender != "unknown":
                    score *= 0.72  # likely self-laugh continuation, weaker "response" signal
                candidates.append({
                    "trigger_id": prev["id"],
                    "laugh_id": msg["id"],
                    "laugh_ts_unix": msg["ts_unix"],
                    "laugh_sender": cur_sender,
                    "laugh_text": msg["body"],
                    "score": float(score),
                })
        prev = msg

    # Aggregate multiple laugh responses tied to the same trigger.
    by_trigger = {}
    for c in candidates:
        t_id = int(c["trigger_id"])
        row = by_trigger.get(t_id)
        if row is None:
            by_trigger[t_id] = {
                "trigger_id": t_id,
                "best_score": float(c["score"]),
                "laugh_count": 1,
                "best_laugh_text": c["laugh_text"],
                "best_laugh_sender": c["laugh_sender"],
                "best_laugh_id": int(c["laugh_id"]),
                "latest_laugh_ts_unix": int(c["laugh_ts_unix"]),
            }
        else:
            row["laugh_count"] += 1
            if float(c["score"]) > float(row["best_score"]):
                row["best_score"] = float(c["score"])
                row["best_laugh_text"] = c["laugh_text"]
                row["best_laugh_sender"] = c["laugh_sender"]
                row["best_laugh_id"] = int(c["laugh_id"])
            row["latest_laugh_ts_unix"] = max(int(row["latest_laugh_ts_unix"]), int(c["laugh_ts_unix"]))

    ranked = []
    for row in by_trigger.values():
        bonus = min((row["laugh_count"] - 1) * 0.35, 2.0)
        total = float(row["best_score"]) + bonus
        ranked.append((row["trigger_id"], total, row))
    ranked.sort(key=lambda x: (-x[1], -int(x[2]["latest_laugh_ts_unix"]), -int(x[0])))
    safe_limit = max(1, min(int(limit), 100))
    return ranked[:safe_limit]


def get_laugh_preview_batch(conn: sqlite3.Connection) -> dict | None:
    ranked = get_laugh_ranked(conn, limit=10)
    if not ranked:
        return None
    trigger_id, total_score, meta = random.choice(ranked)
    msgs = get_result_context_batch(conn, int(trigger_id), before=2, after=2)
    if not msgs:
        return None
    return {
        "center_id": int(trigger_id),
        "center_href": build_chat_href(int(trigger_id), "/laughs", "Times We Laughed"),
        "messages": msgs,
        "laugh_score": round(float(total_score), 2),
        "laugh_count": int(meta["laugh_count"]),
    }


def build_feature_previews(conn: sqlite3.Connection) -> dict:
    return {
        "rhythm_month": get_top_rhythm_month_preview(conn),
        "signals_love": build_signals_love_preview(conn, bins_to_show=None),
        "signals_love_top": get_top_signal_match_preview(conn, signal="love", series="combined"),
        "trends": build_trends_preview(conn, terms=["i love you"], bin_days=30, bins_to_show=None),
        "laugh_batch": get_laugh_preview_batch(conn),
        "on_this_day_1y": get_on_this_day_preview(conn, years_ago=1),
    }

@app.get("/rhythm/data")
def rhythm_data(request: Request, year: int | None = None):
    if not require_auth(request):
        return JSONResponse({"error": "Not authorized"}, status_code=401)

    conn = db()
    years = get_available_years(conn)
    if not years:
        conn.close()
        return JSONResponse({
            "year": None,
            "years": [],
            "days": [],
            "hour_grid": [[0 for _ in range(24)] for _ in range(7)],
        })

    selected_year = year if year in years else years[-1]
    days = get_rhythm_day_stats(conn, selected_year)
    hour_grid = get_rhythm_hour_grid(conn, selected_year)
    conn.close()

    return JSONResponse({
        "year": selected_year,
        "years": years,
        "days": days,
        "hour_grid": hour_grid,
    })

@app.get("/rhythm/day/{date_str}", response_class=HTMLResponse)
def rhythm_day(request: Request, date_str: str, year: int | None = None):
    if not require_auth(request):
        return HTMLResponse("Not authorized", status_code=401)

    # strict date format guard
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return HTMLResponse("Invalid date format", status_code=400)

    conn = db()
    bundle = get_day_highlight_bundle(conn, date_str)
    conn.close()
    messages = bundle["messages"]
    summary = bundle["summary"]
    first_message_id = bundle["first_message_id"]
    day_query = urlencode({"year": year}) if year is not None else ""
    return_to = f"/rhythm/day/{date_str}" + (f"?{day_query}" if day_query else "")
    for m in messages:
        m["chat_href"] = build_chat_href(int(m["id"]), return_to, "Rhythm of Us")
    first_message_href = build_chat_href(first_message_id, return_to, "Rhythm of Us") if first_message_id else None
    back_href = f"/rhythm?year={year}" if year is not None else "/rhythm"

    return templates.TemplateResponse("rhythm_day.html", {
        "request": request,
        "date": date_str,
        "pretty_date": format_pretty_date(int(datetime.strptime(date_str, "%Y-%m-%d").timestamp())),
        "summary": summary,
        "messages": messages,
        "displayed_count": bundle["displayed_count"],
        "first_message_id": first_message_id,
        "first_message_href": first_message_href,
        "back_href": back_href,
    })


@app.get("/on-this-day", response_class=HTMLResponse)
def on_this_day(request: Request, date: str | None = None):
    if not require_auth(request):
        return HTMLResponse("Not authorized", status_code=401)

    if date:
        try:
            anchor = datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            return HTMLResponse("Invalid date format", status_code=400)
    else:
        anchor = datetime.now()
    month = anchor.month
    day = anchor.day
    mmdd = f"{month:02d}-{day:02d}"

    conn = db()
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT strftime('%Y', ts_unix, 'unixepoch', 'localtime') AS y
        FROM messages
        WHERE strftime('%m-%d', ts_unix, 'unixepoch', 'localtime') = ?
        ORDER BY y DESC;
    """, (mmdd,))
    years = [int(r[0]) for r in cur.fetchall() if r and r[0]]

    sections = []
    back_to = f"/on-this-day?date={anchor.strftime('%Y-%m-%d')}"
    for year in years:
        date_str = f"{year:04d}-{month:02d}-{day:02d}"
        bundle = get_day_highlight_bundle(conn, date_str)
        if bundle["summary"]["msg_count"] <= 0:
            continue
        years_ago = anchor.year - year
        section_label = f"{years_ago} year ago today" if years_ago == 1 else f"{years_ago} years ago today"
        top_messages = bundle["messages"][:4]
        result_batches = []
        for center in top_messages:
            mid = int(center["id"])
            msgs = get_result_context_batch(conn, mid, before=2, after=2)
            if not msgs:
                continue
            result_batches.append({
                "center_id": mid,
                "center_href": build_chat_href(mid, back_to, "On This Day"),
                "messages": msgs,
            })
        first_message_href = build_chat_href(bundle["first_message_id"], back_to, "On This Day") if bundle["first_message_id"] else None
        sections.append({
            "year": year,
            "date_str": date_str,
            "section_label": section_label,
            "pretty_date": format_pretty_date(int(datetime.strptime(date_str, "%Y-%m-%d").timestamp())),
            "first_message_href": first_message_href,
            "result_batches": result_batches,
        })
    conn.close()

    return templates.TemplateResponse("on_this_day.html", {
        "request": request,
        "anchor_date": anchor.strftime("%Y-%m-%d"),
        "anchor_pretty_date": format_pretty_date(int(anchor.replace(hour=0, minute=0, second=0, microsecond=0).timestamp())),
        "month_day_label": f"{anchor.strftime('%b')} {anchor.day}{ordinal_suffix(anchor.day)}",
        "sections": sections,
    })

@app.get("/rhythm/signals", response_class=HTMLResponse)
def rhythm_signals(
    request: Request,
    signal: str = "love",
    bin_days: int = 7,
    metric: str = "rate",
    series: str = "combined",
    overlay: int = 0,
):
    if not require_auth(request):
        return HTMLResponse("Not authorized", status_code=401)

    if signal not in SIGNAL_COLUMNS:
        signal = "love"
    if bin_days not in (1, 3, 7, 30):
        bin_days = 7
    if metric not in ("count", "rate"):
        metric = "rate"
    if series not in ("combined", "person_a", "person_b", "both"):
        series = "combined"
    overlay = 1 if overlay else 0

    return templates.TemplateResponse("rhythm_signals.html", {
        "request": request,
        "signals": list(SIGNAL_COLUMNS.keys()),
        "thresholds": SIGNAL_THRESHOLDS,
        "initial_signal": signal,
        "initial_bin_days": bin_days,
        "initial_metric": metric,
        "initial_series": series,
        "initial_overlay": overlay,
    })


@app.get("/trends", response_class=HTMLResponse)
def trends_page(
    request: Request,
    terms: str = "love, laugh",
    bin_days: int = TRENDS_BIN_DAYS_DEFAULT,
):
    if not require_auth(request):
        return HTMLResponse("Not authorized", status_code=401)
    back_href = sanitize_internal_return_to(request.query_params.get("return_to"), "/")
    # Avoid back-looping between chat <-> trends.
    if back_href.startswith("/trends") or back_href.startswith("/chat/"):
        back_href = "/"
    safe_bin_days = bin_days if bin_days in (1, 3, 7, 14, 30, 60, 90) else TRENDS_BIN_DAYS_DEFAULT
    return templates.TemplateResponse("trends.html", {
        "request": request,
        "default_bin_days": TRENDS_BIN_DAYS_DEFAULT,
        "default_highlight_limit": TRENDS_HIGHLIGHT_LIMIT,
        "back_href": back_href,
        "initial_terms": terms,
        "initial_bin_days": safe_bin_days,
    })


@app.get("/trends/data")
def trends_data(request: Request, terms: str = "", bin_days: int = TRENDS_BIN_DAYS_DEFAULT):
    if not require_auth(request):
        return JSONResponse({"error": "Not authorized"}, status_code=401)

    parsed_terms = parse_trend_terms(terms, max_terms=TRENDS_MAX_TERMS)
    if not parsed_terms:
        return JSONResponse({"error": "Please enter at least one term"}, status_code=400)

    safe_bin_days = bin_days if bin_days in (1, 3, 7, 14, 30, 60, 90) else TRENDS_BIN_DAYS_DEFAULT
    conn = db()
    min_ts, max_ts = get_message_time_bounds(conn)
    if min_ts is None or max_ts is None:
        conn.close()
        return JSONResponse({"terms": parsed_terms, "bins": [], "series": {}})

    out_series = {}
    out_top_message_ids = {}
    bins_ref = None
    top_ids_per_term = {}
    for term in parsed_terms:
        scores = compute_trend_match_scores(conn, term)
        top_ids, b, bin_top_ids = build_term_bins(conn, term, scores, min_ts, max_ts, safe_bin_days)
        out_series[term] = [x["count"] for x in b]
        out_top_message_ids[term] = bin_top_ids
        top_ids_per_term[term] = top_ids
        if bins_ref is None:
            bins_ref = b

    conn.close()
    return JSONResponse({
        "terms": parsed_terms,
        "bin_days": safe_bin_days,
        "min_ts": min_ts,
        "max_ts": max_ts,
        "bins": bins_ref or [],
        "series": out_series,
        "top_message_ids": out_top_message_ids,
        "top_ids_per_term": top_ids_per_term,
    })


@app.get("/trends/highlights", response_class=HTMLResponse)
def trends_highlights(request: Request, term: str = "", limit: int = TRENDS_HIGHLIGHT_LIMIT):
    if not require_auth(request):
        return HTMLResponse("Not authorized", status_code=401)

    q = normalize_query_text(term)
    if not q:
        return templates.TemplateResponse("trends_highlights.html", {
            "request": request,
            "messages": [],
            "term": term,
        })

    safe_limit = max(1, min(int(limit), 100))
    conn = db()
    scores = compute_trend_match_scores(conn, q)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:safe_limit]
    ids = [int(mid) for mid, _ in ranked]

    msgs = fetch_messages_by_ids(conn, ids)
    by_id = {m["id"]: m for m in msgs}
    vals = [float(s) for _, s in ranked]
    lo = min(vals) if vals else 0.0
    hi = max(vals) if vals else 1.0
    span = (hi - lo) if hi > lo else 1.0

    out = []
    for mid, score in ranked:
        m = by_id.get(int(mid))
        if not m:
            continue
        pct = int(round(((float(score) - lo) / span) * 100))
        m["match_pct"] = pct
        m["chat_href"] = build_chat_href(int(mid), "/trends", "Explore our Trends")
        out.append(m)
    conn.close()

    return templates.TemplateResponse("trends_highlights.html", {
        "request": request,
        "messages": out,
        "term": q,
    })


@app.get("/trends/window", response_class=HTMLResponse)
def trends_window(
    request: Request,
    term: str = "",
    start_ts: int = 0,
    end_ts: int = 0,
    matched_only: int = 1,
    sort: str = "score_desc",
    terms: str = "",
    bin_days: int = TRENDS_BIN_DAYS_DEFAULT,
):
    if not require_auth(request):
        return HTMLResponse("Not authorized", status_code=401)
    q = normalize_query_text(term)
    if not q:
        return HTMLResponse("Invalid term", status_code=400)
    if end_ts <= start_ts:
        return HTMLResponse("Invalid window", status_code=400)
    if sort not in ("time", "score_desc"):
        sort = "score_desc"
    safe_bin_days = bin_days if bin_days in (1, 3, 7, 14, 30, 60, 90) else TRENDS_BIN_DAYS_DEFAULT

    conn = db()
    scores = compute_trend_match_scores(conn, q)
    max_score = max(scores.values()) if scores else 1.0
    if max_score <= 0:
        max_score = 1.0

    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, ts, ts_unix, direction, body
        FROM messages
        WHERE ts_unix >= ? AND ts_unix < ?
        ORDER BY ts_unix ASC, id ASC;
        """,
        (int(start_ts), int(end_ts)),
    )
    rows = cur.fetchall()
    conn.close()

    messages = []
    matched_count = 0
    for r in rows:
        mid = int(r[0])
        sc = float(scores.get(mid, 0.0))
        matched = sc > 0.0
        if matched:
            matched_count += 1
        if matched_only and not matched:
            continue
        messages.append({
            "id": mid,
            "ts": r[1],
            "ts_unix": int(r[2] or 0),
            "direction": r[3] or "",
            "body": r[4] or "",
            "sender": detect_sender(r[3] or ""),
            "match_score": sc,
            "match_pct": int(round((sc / max_score) * 100)) if matched else 0,
            "matched": matched,
        })

    if sort == "score_desc":
        messages.sort(key=lambda m: (-m["match_score"], m["ts_unix"], m["id"]))
    else:
        messages.sort(key=lambda m: (m["ts_unix"], m["id"]))

    add_display_timestamp_fields(messages)
    back_query = urlencode({"terms": terms, "bin_days": safe_bin_days})
    window_query = urlencode({
        "term": q,
        "start_ts": int(start_ts),
        "end_ts": int(end_ts),
        "matched_only": int(matched_only),
        "sort": sort,
        "terms": terms,
        "bin_days": safe_bin_days,
    })
    window_return_to = f"/trends/window?{window_query}"
    for m in messages:
        m["chat_href"] = build_chat_href(int(m["id"]), window_return_to, "Explore our Trends")

    period_days = max(1, int((int(end_ts) - int(start_ts)) // 86400))
    date_end = max(int(start_ts), int(end_ts) - 1)
    return templates.TemplateResponse("trends_window.html", {
        "request": request,
        "term": q,
        "start_ts": int(start_ts),
        "end_ts": int(end_ts),
        "matched_only": int(matched_only),
        "sort": sort,
        "bin_days": safe_bin_days,
        "terms": terms,
        "messages": messages,
        "summary": {
            "total_msgs": len(rows),
            "matched_msgs": matched_count,
        },
        "date_range_label": f"{format_pretty_date(int(start_ts))} - {format_pretty_date(date_end)}",
        "period_label": f"{period_days}-day period",
        "back_query": back_query,
    })

@app.get("/rhythm/signals/data")
def rhythm_signals_data(
    request: Request,
    signal: str = "love",
    bin_days: int = 3,
    metric: str = "count",
    series: str = "combined",
    overlay: int = 1,
    threshold_mode: str = "default",
):
    if not require_auth(request):
        return JSONResponse({"error": "Not authorized"}, status_code=401)

    if signal not in SIGNAL_COLUMNS:
        return JSONResponse({"error": "Invalid signal"}, status_code=400)
    if bin_days not in (1, 3, 7, 30):
        return JSONResponse({"error": "Invalid bin_days"}, status_code=400)
    if metric not in ("count", "rate"):
        return JSONResponse({"error": "Invalid metric"}, status_code=400)
    if series not in ("combined", "person_a", "person_b", "both"):
        return JSONResponse({"error": "Invalid series"}, status_code=400)

    threshold = SIGNAL_THRESHOLDS[signal]
    conn = db()
    data = get_signal_bins(conn, signal=signal, bin_days=bin_days, threshold=threshold)
    conn.close()

    return JSONResponse({
        "signal": signal,
        "bin_days": bin_days,
        "metric": metric,
        "series": series,
        "overlay": 1 if overlay else 0,
        "threshold_mode": threshold_mode,
        "threshold": threshold,
        "min_ts": data["min_ts"],
        "max_ts": data["max_ts"],
        "bins": data["bins"],
    })

@app.get("/rhythm/signals/highlights", response_class=HTMLResponse)
def rhythm_signals_highlights(
    request: Request,
    signal: str = "love",
    series: str = "combined",
    bin_days: int = 7,
    metric: str = "rate",
    overlay: int = 0,
    limit: int = 50,
):
    if not require_auth(request):
        return HTMLResponse("Not authorized", status_code=401)
    if signal not in SIGNAL_COLUMNS:
        return HTMLResponse("Invalid signal", status_code=400)
    if series not in ("combined", "person_a", "person_b", "both"):
        return HTMLResponse("Invalid series", status_code=400)
    if bin_days not in (1, 3, 7, 30):
        bin_days = 7
    if metric not in ("count", "rate"):
        metric = "rate"
    overlay = 1 if overlay else 0

    safe_limit = min(max(int(limit), 1), 100)
    signal_col = SIGNAL_COLUMNS[signal]
    where_parts = []
    params: list = []
    if series == "person_a":
        where_parts.append("ms.sender = ?")
        params.append("person_a")
    elif series == "person_b":
        where_parts.append("ms.sender = ?")
        params.append("person_b")

    where_sql = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""

    conn = db()
    cur = conn.cursor()
    cur.execute(f"""
        SELECT
            m.id,
            m.ts,
            m.ts_unix,
            m.direction,
            m.body,
            ms.sender,
            ms.{signal_col} AS signal_score
        FROM messages m
        JOIN message_signals ms ON ms.message_id = m.id
        {where_sql}
        ORDER BY ms.{signal_col} DESC, m.ts_unix DESC, m.id DESC
        LIMIT ?;
    """, tuple(params + [safe_limit]))
    rows = cur.fetchall()
    conn.close()

    messages = []
    for r in rows:
        sender = (r[5] or "unknown")
        messages.append({
            "id": int(r[0]),
            "ts": r[1],
            "ts_unix": int(r[2] or 0),
            "direction": r[3] or "",
            "body": r[4] or "",
            "sender": sender if sender in ("person_a", "person_b") else "unknown",
            "signal_score": float(r[6] or 0.0),
        })
    add_display_timestamp_fields(messages)
    return_to = f"/rhythm/signals?{urlencode({'signal': signal, 'bin_days': bin_days, 'metric': metric, 'series': series, 'overlay': overlay})}"
    for m in messages:
        m["chat_href"] = build_chat_href(int(m["id"]), return_to, "Signals Timeline")

    return templates.TemplateResponse("rhythm_signals_highlights.html", {
        "request": request,
        "signal": signal,
        "series": series,
        "messages": messages,
    })

@app.get("/rhythm/signals/window", response_class=HTMLResponse)
def rhythm_signals_window(
    request: Request,
    signal: str = "love",
    start_ts: int = 0,
    end_ts: int = 0,
    threshold: float | None = None,
    counted_only: int = 1,
    sort: str = "score_desc",
    bin_days: int = 7,
    metric: str = "rate",
    series: str = "combined",
    overlay: int = 0,
):
    if not require_auth(request):
        return HTMLResponse("Not authorized", status_code=401)
    if signal not in SIGNAL_COLUMNS:
        return HTMLResponse("Invalid signal", status_code=400)
    if end_ts <= start_ts:
        return HTMLResponse("Invalid window", status_code=400)
    if sort not in ("time", "score_desc"):
        return HTMLResponse("Invalid sort", status_code=400)
    if bin_days not in (1, 3, 7, 30):
        bin_days = 7
    if metric not in ("count", "rate"):
        metric = "rate"
    if series not in ("combined", "person_a", "person_b", "both"):
        series = "combined"
    overlay = 1 if overlay else 0

    signal_col = SIGNAL_COLUMNS[signal]
    active_threshold = threshold if threshold is not None else SIGNAL_THRESHOLDS[signal]

    conn = db()
    cur = conn.cursor()

    where_extra = ""
    params = [start_ts, end_ts]
    if counted_only:
        where_extra = f" AND ms.{signal_col} >= ? "
        params.append(active_threshold)

    if sort == "score_desc":
        order_by = f" ORDER BY ms.{signal_col} DESC, m.ts_unix ASC, m.id ASC "
    else:
        order_by = " ORDER BY m.ts_unix ASC, m.id ASC "

    cur.execute(f"""
        SELECT
            m.id,
            m.ts,
            m.ts_unix,
            m.direction,
            m.body,
            ms.{signal_col} AS signal_score,
            ms.top_labels_json,
            ms.sender
        FROM messages m
        JOIN message_signals ms ON ms.message_id = m.id
        WHERE m.ts_unix >= ? AND m.ts_unix < ?
        {where_extra}
        {order_by};
    """, tuple(params))
    rows = cur.fetchall()

    messages = []
    for r in rows:
        labels = []
        if r[6]:
            try:
                labels = json.loads(r[6])
            except json.JSONDecodeError:
                labels = []
        messages.append({
            "id": r[0],
            "ts": r[1],
            "ts_unix": int(r[2] or 0),
            "direction": r[3],
            "body": r[4],
            "signal_score": float(r[5] or 0.0),
            "top_labels": labels[:3],
            "sender": r[7],
            "counted": float(r[5] or 0.0) >= active_threshold,
        })
    add_display_timestamp_fields(messages)

    cur.execute(f"""
        SELECT
            COUNT(*) AS total_msgs,
            SUM(CASE WHEN ms.{signal_col} >= ? THEN 1 ELSE 0 END) AS counted_msgs
        FROM messages m
        JOIN message_signals ms ON ms.message_id = m.id
        WHERE m.ts_unix >= ? AND m.ts_unix < ?;
    """, (active_threshold, start_ts, end_ts))
    summary_row = cur.fetchone()
    conn.close()

    period_days = max(1, int((end_ts - start_ts) // 86400))
    period_label = f"{period_days}-day period"
    date_range_label = f"{format_pretty_date(start_ts)} - {format_pretty_date(end_ts)}"
    back_query = urlencode({
        "signal": signal,
        "bin_days": bin_days,
        "metric": metric,
        "series": series,
        "overlay": overlay,
    })
    window_query = urlencode({
        "signal": signal,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "threshold": active_threshold,
        "counted_only": int(counted_only),
        "sort": sort,
        "bin_days": bin_days,
        "metric": metric,
        "series": series,
        "overlay": overlay,
    })
    window_return_to = f"/rhythm/signals/window?{window_query}"
    for m in messages:
        m["chat_href"] = build_chat_href(int(m["id"]), window_return_to, "Signals Timeline")

    return templates.TemplateResponse("rhythm_signals_window.html", {
        "request": request,
        "signal": signal,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "start_label": datetime.fromtimestamp(start_ts).strftime("%Y-%m-%d %H:%M"),
        "end_label": datetime.fromtimestamp(end_ts).strftime("%Y-%m-%d %H:%M"),
        "date_range_label": date_range_label,
        "period_label": period_label,
        "threshold": active_threshold,
        "counted_only": int(counted_only),
        "sort": sort,
        "bin_days": bin_days,
        "metric": metric,
        "series": series,
        "overlay": overlay,
        "back_query": back_query,
        "summary": {
            "total_msgs": int(summary_row[0] or 0) if summary_row else 0,
            "counted_msgs": int(summary_row[1] or 0) if summary_row else 0,
        },
        "messages": messages,
    })

@app.post("/login")
def login(password: str = Form(...)):
    if not setup_is_complete():
        return RedirectResponse(url="/setup", status_code=303)
    if not verify_configured_password(password):
        log_event("warning", "auth: failed login attempt")
        return RedirectResponse(url="/?bad=1", status_code=303)
    log_event("info", "auth: login success")
    resp = RedirectResponse(url="/", status_code=303)
    resp.set_cookie("tauth", "1", httponly=True, samesite="lax")
    return resp

@app.post("/logout")
def logout():
    log_event("info", "auth: logout")
    resp = RedirectResponse(url="/", status_code=303)
    resp.delete_cookie("tauth")
    return resp


@app.post("/stop-program", response_class=HTMLResponse)
def stop_program(request: Request):
    if not require_auth(request) and setup_is_complete():
        return HTMLResponse("Not authorized", status_code=401)
    log_event("warning", "system: stop program requested from browser")

    def _shutdown():
        time.sleep(0.6)
        os._exit(0)

    threading.Thread(target=_shutdown, daemon=True).start()
    return HTMLResponse(
        """
        <!doctype html>
        <html>
        <head>
          <meta charset="utf-8" />
          <meta name="viewport" content="width=device-width, initial-scale=1" />
          <title>Stopping Program</title>
          <link rel="stylesheet" href="/static/style.css">
        </head>
        <body class="home-page">
          <div class="wrap">
            <div class="card setup-card">
              <h2>Program Stopping</h2>
              <p class="muted">The local server is shutting down now. You can close this tab.</p>
            </div>
          </div>
        </body>
        </html>
        """
    )

def exact_search(conn: sqlite3.Connection, query: str, limit: int) -> List[int]:
    cur = conn.cursor()
    ids: List[int] = []
    seen = set()
    q = (query or "").strip()
    if not q:
        return []

    # Try phrase-first for precision.
    try:
        safe_q = q.replace('"', '""')
        cur.execute(
            "SELECT rowid FROM messages_fts WHERE body MATCH ? LIMIT ?;",
            (f'"{safe_q}"', limit),
        )
        for row in cur.fetchall():
            mid = int(row[0])
            if mid not in seen:
                seen.add(mid)
                ids.append(mid)
    except sqlite3.OperationalError:
        pass

    # If phrase is sparse, backfill with token OR query for broader recall.
    tokens = [t for t in re.findall(r"[a-z0-9']+", q.lower()) if len(t) >= 2]
    if tokens and len(ids) < limit:
        or_query = " OR ".join([f'"{t}"' for t in tokens[:10]])
        try:
            cur.execute(
                "SELECT rowid FROM messages_fts WHERE body MATCH ? LIMIT ?;",
                (or_query, max(limit * 2, limit)),
            )
            for row in cur.fetchall():
                mid = int(row[0])
                if mid not in seen:
                    seen.add(mid)
                    ids.append(mid)
                    if len(ids) >= limit:
                        break
        except sqlite3.OperationalError:
            pass

    return ids[:limit]


def normalize_query_text(query: str) -> str:
    q = (query or "").strip()
    if not q:
        return ""
    ql = q.lower().replace("’", "'")
    ql = re.sub(r"\bperson\s+a\b", "person_a", ql)
    ql = re.sub(r"\bperson\s+b\b", "person_b", ql)
    ql = re.sub(r"\b(person_a|person_b)'s\b", r"\1", ql)
    ql = re.sub(r"\b(person_a|person_b)'\b", r"\1", ql)
    for bad, good in QUERY_SPELL_NORMALIZATIONS.items():
        ql = re.sub(rf"\b{re.escape(bad)}\b", good, ql)
    return " ".join(ql.split())


def normalize_experiment_query(query: str) -> str:
    q = normalize_query_text(query)
    if not q:
        return ""
    # Keep this conservative to avoid over-normalizing meaning queries.
    q = re.sub(r"\s+", " ", q).strip()
    return q


def experiment_prefers_earliest(query: str) -> bool:
    q = (query or "").lower()
    signals = [
        "first time",
        "earliest",
        "when did we first",
        "first time we",
        "first time i",
        "first time you",
    ]
    return any(s in q for s in signals)


def _month_name_to_num(month_name: str) -> int | None:
    months = {
        "jan": 1, "january": 1,
        "feb": 2, "february": 2,
        "mar": 3, "march": 3,
        "apr": 4, "april": 4,
        "may": 5,
        "jun": 6, "june": 6,
        "jul": 7, "july": 7,
        "aug": 8, "august": 8,
        "sep": 9, "sept": 9, "september": 9,
        "oct": 10, "october": 10,
        "nov": 11, "november": 11,
        "dec": 12, "december": 12,
    }
    return months.get((month_name or "").lower())


def parse_experiment_date_range(query: str) -> tuple[int | None, int | None]:
    q = (query or "").lower()
    # Month + year (e.g., "feb 2026")
    m = re.search(
        r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
        r"sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(20\d{2})\b",
        q,
    )
    if m:
        month_num = _month_name_to_num(m.group(1))
        year = int(m.group(2))
        if month_num:
            start = datetime(year, month_num, 1)
            if month_num == 12:
                end = datetime(year + 1, 1, 1)
            else:
                end = datetime(year, month_num + 1, 1)
            return int(start.timestamp()), int(end.timestamp())

    # Year only
    m = re.search(r"\b(20\d{2})\b", q)
    if m:
        year = int(m.group(1))
        start = datetime(year, 1, 1)
        end = datetime(year + 1, 1, 1)
        return int(start.timestamp()), int(end.timestamp())
    return None, None


def build_experiment_lexical_query(query: str, topic_terms: List[str] | None = None) -> str:
    tokens = list(topic_terms or [])
    if not tokens:
        tokens = [t for t in extract_query_terms(query) if t not in SEARCH_STOPWORDS and t not in GENERIC_QUERY_FILLER]
    expanded = []
    seen = set()
    for t in tokens[:10]:
        if t not in seen:
            seen.add(t)
            expanded.append(t)
        for syn in EXPERIMENT_LEXICAL_EXPANSIONS.get(t, []):
            if syn not in seen:
                seen.add(syn)
                expanded.append(syn)
    if not expanded:
        return query
    return " OR ".join([f'"{x}"' for x in expanded[:18]])


def extract_query_terms(query: str) -> List[str]:
    raw = [t.replace("'", "") for t in re.findall(r"[a-z0-9']+", (query or "").lower())]
    tokens = [t for t in raw if len(t) >= 3]
    ordered = []
    seen = set()
    for t in tokens:
        if t not in seen:
            seen.add(t)
            ordered.append(t)
    return ordered[:12]


def singularize_token(token: str) -> str:
    t = token.strip().lower()
    if t in PERSON_TOKENS:
        return t
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 4 and t.endswith("is"):
        return t
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return t


def extract_focus_terms(query: str) -> List[str]:
    q = normalize_query_text(query)
    if not q:
        return []
    tokens = extract_query_terms(q)
    cleaned: List[str] = []
    seen = set()
    for t in tokens:
        if t in SEARCH_STOPWORDS or t in GENERIC_QUERY_FILLER:
            continue
        tt = singularize_token(t)
        if len(tt) < 3 or tt in SEARCH_STOPWORDS or tt in GENERIC_QUERY_FILLER or tt in PERSON_TOKENS:
            continue
        if tt not in seen:
            seen.add(tt)
            cleaned.append(tt)
    return cleaned[:8]


def strip_topic_prefixes(query: str) -> str:
    q = normalize_query_text(query)
    if not q:
        return ""
    for pat in TOPIC_PREFIX_PATTERNS:
        nq = re.sub(pat, "", q, flags=re.IGNORECASE)
        if nq != q:
            q = nq.strip()
            break
    return q


def extract_topic_terms(query: str) -> List[str]:
    # Prefer terms after common lead-ins like "messages about ...".
    base = strip_topic_prefixes(query)
    terms = extract_query_terms(base)
    cleaned: List[str] = []
    seen = set()
    for t in terms:
        tt = singularize_token(t)
        if (
            len(tt) < 3
            or tt in SEARCH_STOPWORDS
            or tt in GENERIC_QUERY_FILLER
            or tt in PERSON_TOKENS
        ):
            continue
        if tt not in seen:
            seen.add(tt)
            cleaned.append(tt)
    # Fallback to existing focus terms if aggressive cleaning removes everything.
    if cleaned:
        return cleaned[:8]
    return extract_focus_terms(query)


def parse_query_person_intent(query: str) -> dict:
    q = normalize_query_text(query)
    out = {
        "person": None,     # "person_a" | "person_b" | None
        "mode": None,       # "sender" | "about" | None
    }
    if not q:
        return out

    # Sender intent: "from person_a", "by person_b", "sent by person_a"
    m_sender = re.search(r"\b(?:from|by|sent by)\s+(person_a|person_b)\b", q)
    if m_sender:
        out["person"] = m_sender.group(1)
        out["mode"] = "sender"
        return out

    # About/possessive intent: "person_b's ...", "about person_a", "person_a plans"
    m_possessive = re.search(r"\b(person_a|person_b)'?s\b", q)
    if m_possessive:
        out["person"] = m_possessive.group(1)
        out["mode"] = "about"
        return out

    m_about = re.search(r"\babout\s+(person_a|person_b)\b", q)
    if m_about:
        out["person"] = m_about.group(1)
        out["mode"] = "about"
        return out

    m_named = re.search(r"\b(person_a|person_b)\b", q)
    if m_named:
        out["person"] = m_named.group(1)
        out["mode"] = "about"
        return out

    return out


def fetch_sender_map(conn: sqlite3.Connection, message_ids: List[int]) -> dict[int, str]:
    if not message_ids:
        return {}
    cur = conn.cursor()
    placeholders = ",".join(["?"] * len(message_ids))
    cur.execute(
        f"SELECT id, direction FROM messages WHERE id IN ({placeholders});",
        message_ids,
    )
    return {int(r[0]): detect_sender(r[1] or "") for r in cur.fetchall()}


def compute_focus_term_boost(text: str, focus_terms: List[str]) -> float:
    if not text or not focus_terms:
        return 0.0
    low = text.lower()
    hit_count = 0
    for t in focus_terms:
        if re.search(rf"\b{re.escape(t)}\w*\b", low):
            hit_count += 1
    if hit_count == 0:
        return -0.08
    return min(0.12 * hit_count, 0.48)


def has_focus_term_match(text: str, focus_terms: List[str]) -> bool:
    if not text or not focus_terms:
        return False
    low = text.lower()
    for t in focus_terms:
        if re.search(rf"\b{re.escape(t)}\w*\b", low):
            return True
    return False


def count_focus_term_matches(text: str, focus_terms: List[str]) -> int:
    if not text or not focus_terms:
        return 0
    low = text.lower()
    hits = 0
    for t in focus_terms:
        if re.search(rf"\b{re.escape(t)}\w*\b", low):
            hits += 1
    return hits


def is_strict_literal_term(term: str) -> bool:
    q = normalize_query_text(term)
    tokens = [t for t in extract_query_terms(q) if t not in SEARCH_STOPWORDS and t not in GENERIC_QUERY_FILLER]
    if len(tokens) != 1:
        return False
    t = tokens[0]
    if t in PERSON_TOKENS:
        return True
    return re.fullmatch(r"[a-z][a-z'-]{2,29}", t) is not None


def count_literal_term_matches(text: str, terms: List[str]) -> int:
    if not text or not terms:
        return 0
    low = text.lower()
    hits = 0
    for t in terms:
        # Exact token/possessive hit: gavin, gavin's
        if re.search(rf"\b{re.escape(t)}(?:'s)?\b", low):
            hits += 1
    return hits


def enforce_focus_terms_in_top_results(
    ids: List[int],
    window_texts: dict[int, str],
    focus_terms: List[str],
    semantic_scores: dict[int, float],
    top_n: int = FOCUS_ENFORCE_TOP_N,
    high_conf_threshold: float = FOCUS_HIGH_CONFIDENCE_THRESHOLD,
) -> List[int]:
    if not ids or not focus_terms:
        return ids
    n = min(top_n, len(ids))
    top = ids[:n]
    rest = ids[n:]

    weak_positions = []
    for i, mid in enumerate(top):
        has_focus = has_focus_term_match(window_texts.get(mid, ""), focus_terms)
        is_high_conf = float(semantic_scores.get(mid, 0.0)) >= high_conf_threshold
        if (not has_focus) and (not is_high_conf):
            weak_positions.append(i)

    if not weak_positions:
        return ids

    replacement_idxs = []
    for i, mid in enumerate(rest):
        if has_focus_term_match(window_texts.get(mid, ""), focus_terms):
            replacement_idxs.append(i)
            if len(replacement_idxs) >= len(weak_positions):
                break

    if not replacement_idxs:
        return ids

    displaced = []
    for pos, rep_idx in zip(weak_positions, replacement_idxs):
        displaced.append(top[pos])
        top[pos] = rest[rep_idx]

    used = set(replacement_idxs)
    new_rest = []
    for i, mid in enumerate(rest):
        if i not in used:
            new_rest.append(mid)
    new_rest = displaced + new_rest
    return top + new_rest


def get_query_expander():
    global query_expander, query_expander_error
    if not QUERY_EXPANDER_MODEL:
        return None
    if query_expander is not None:
        return query_expander
    if query_expander_error is not None:
        return None

    with query_expander_lock:
        if query_expander is not None:
            return query_expander
        if query_expander_error is not None:
            return None
        try:
            from transformers import pipeline

            device = 0 if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else -1
            query_expander = pipeline(
                "text2text-generation",
                model=QUERY_EXPANDER_MODEL,
                device=device,
            )
        except Exception as exc:
            query_expander_error = str(exc)
            query_expander = None
    return query_expander


def llm_expand_query_variants(query: str, limit: int = 4) -> List[str]:
    expander = get_query_expander()
    if expander is None:
        return []

    prompt = (
        "Generate concise alternate search phrasings for this topic query. "
        "Return one rewrite per line, no numbering, no quotes, max 8 words each.\n"
        f"Query: {query}"
    )
    try:
        out = expander(prompt, max_new_tokens=96, do_sample=False)
        text = (out[0].get("generated_text", "") if out else "").strip()
    except Exception:
        return []

    variants = []
    seen = set()
    for line in text.splitlines():
        v = normalize_query_text(line)
        if not v or v == query or v in seen:
            continue
        seen.add(v)
        variants.append(v)
        if len(variants) >= limit:
            break
    return variants


def generate_general_query_rewrites(query: str) -> List[str]:
    q = normalize_query_text(query)
    if not q:
        return []
    terms = [t for t in extract_query_terms(q) if t not in SEARCH_STOPWORDS]
    phrase = " ".join(terms[:6]) if terms else q

    rewrites = []
    for p in GENERIC_REWRITE_PREFIXES:
        rewrites.append(f"{p} {phrase}")
    if len(terms) >= 2:
        rewrites.append(f"{terms[0]} and {terms[1]} conversation")
        rewrites.append(f"{terms[0]} {terms[1]} related messages")
    rewrites.append(f"messages where we discuss {phrase}")

    return rewrites[:7]


def expand_query_variants(query: str) -> List[str]:
    q = normalize_query_text(query)
    if not q:
        return []

    variants = [q]
    ql = q.lower()
    variants.extend(generate_general_query_rewrites(q))
    variants.extend(llm_expand_query_variants(q, limit=4))

    for trigger, expansions in QUERY_EXPANSION_RULES.items():
        if trigger in ql:
            exp = " ".join(expansions[:8]).strip()
            if exp:
                variants.append(f"{q} {exp}")
                variants.append(exp)

    # Lightweight fallback expansion for very short queries.
    if len(q.split()) <= 2:
        compact = [t for t in re.findall(r"[a-z0-9']+", ql) if len(t) >= 3]
        if compact:
            variants.append(" ".join(compact))

    deduped = []
    seen = set()
    for v in variants:
        vv = " ".join(v.split()).strip()
        if vv and vv not in seen:
            seen.add(vv)
            deduped.append(vv)
        if len(deduped) >= 14:
            break
    return deduped


def map_positions_to_message_ids(conn: sqlite3.Connection, positions: List[int]) -> List[int]:
    valid_positions = [int(p) for p in positions if int(p) >= 0]
    if not valid_positions:
        return []

    cur = conn.cursor()
    unique_positions = sorted(set(valid_positions))
    placeholders = ",".join(["?"] * len(unique_positions))
    cur.execute(
        f"SELECT pos, message_id FROM faiss_map WHERE pos IN ({placeholders});",
        unique_positions,
    )
    mapping = {int(row[0]): int(row[1]) for row in cur.fetchall()}

    out = []
    seen = set()
    for pos in valid_positions:
        mid = mapping.get(pos)
        if mid is not None and mid not in seen:
            seen.add(mid)
            out.append(mid)
    return out


def semantic_search_scored(conn: sqlite3.Connection, query: str, k: int) -> dict[int, float]:
    q = (query or "").strip()
    if not q:
        return {}

    if model is None or faiss_index is None:
        return {}
    total = int(getattr(faiss_index, "ntotal", 0))
    if total <= 0:
        return {}

    k_eff = min(max(int(k), 1), total)
    qemb = model.encode([q], normalize_embeddings=True).astype("float32")
    scores, idxs = faiss_index.search(qemb, k_eff)

    positions = [int(x) for x in idxs[0].tolist() if int(x) >= 0]
    if not positions:
        return {}
    msg_ids = map_positions_to_message_ids(conn, positions)

    score_by_pos = {}
    for i, pos in enumerate(idxs[0].tolist()):
        pos_i = int(pos)
        if pos_i >= 0:
            score_by_pos[pos_i] = float(scores[0][i])

    scored = {}
    for i, mid in enumerate(msg_ids):
        if i < len(positions):
            pos = positions[i]
            sim = score_by_pos.get(pos, 0.0)
            prev = scored.get(mid)
            if prev is None or sim > prev:
                scored[mid] = sim
    return scored


@lru_cache(maxsize=256)
def _cached_query_embedding_blob(query: str) -> bytes:
    if model is None:
        return b""
    vec = model.encode([query], normalize_embeddings=True).astype("float32")[0]
    return vec.tobytes()


def semantic_search_scored_cached(conn: sqlite3.Connection, query: str, k: int) -> dict[int, float]:
    q = (query or "").strip()
    if not q:
        return {}
    if model is None or faiss_index is None:
        return {}
    total = int(getattr(faiss_index, "ntotal", 0))
    if total <= 0:
        return {}

    k_eff = min(max(int(k), 1), total)
    emb_blob = _cached_query_embedding_blob(q)
    if not emb_blob:
        return {}
    qvec = np.frombuffer(emb_blob, dtype="float32").copy().reshape(1, -1)
    scores, idxs = faiss_index.search(qvec, k_eff)

    positions = [int(x) for x in idxs[0].tolist() if int(x) >= 0]
    if not positions:
        return {}
    msg_ids = map_positions_to_message_ids(conn, positions)

    score_by_pos = {}
    for i, pos in enumerate(idxs[0].tolist()):
        pos_i = int(pos)
        if pos_i >= 0:
            score_by_pos[pos_i] = float(scores[0][i])

    scored = {}
    for i, mid in enumerate(msg_ids):
        if i < len(positions):
            pos = positions[i]
            sim = score_by_pos.get(pos, 0.0)
            prev = scored.get(mid)
            if prev is None or sim > prev:
                scored[mid] = sim
    return scored


def fts_search_candidates_scored(
    conn: sqlite3.Connection,
    query: str,
    limit: int,
    start_ts: int | None = None,
    end_ts: int | None = None,
) -> List[tuple[int, float]]:
    q = (query or "").strip()
    if not q:
        return []
    cur = conn.cursor()
    params: list = []
    where = ["messages_fts MATCH ?"]
    params.append(q)
    if start_ts is not None:
        where.append("m.ts_unix >= ?")
        params.append(int(start_ts))
    if end_ts is not None:
        where.append("m.ts_unix < ?")
        params.append(int(end_ts))
    params.append(int(limit))
    cur.execute(
        f"""
        SELECT m.id, bm25(messages_fts) AS rank_score
        FROM messages_fts
        JOIN messages m ON m.id = messages_fts.rowid
        WHERE {' AND '.join(where)}
        ORDER BY rank_score ASC
        LIMIT ?;
        """,
        tuple(params),
    )
    out = []
    for row in cur.fetchall():
        mid = int(row[0])
        rank_score = float(row[1] if row[1] is not None else 0.0)
        out.append((mid, rank_score))
    return out


def normalize_scores(score_map: dict[int, float]) -> dict[int, float]:
    if not score_map:
        return {}
    vals = list(score_map.values())
    lo = min(vals)
    hi = max(vals)
    span = hi - lo
    if span <= 1e-9:
        return {k: 1.0 for k in score_map}
    return {k: (v - lo) / span for k, v in score_map.items()}


def count_words(text: str) -> int:
    if not text:
        return 0
    return len(re.findall(r"[a-z0-9']+", text.lower()))


def filter_trend_scores_to_strong_matches(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}

    # Keep only strongest matches so trend lines represent topic-specific signal,
    # not all semantic candidates.
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    values = [float(v) for _, v in ordered]
    if not values:
        return {}

    pct = min(max(TRENDS_STRONG_MATCH_PERCENTILE, 0.0), 0.98)
    cut_idx = int(len(values) * pct)
    if cut_idx >= len(values):
        cut_idx = len(values) - 1
    cutoff = values[max(0, cut_idx)]

    filtered = {int(mid): float(sc) for mid, sc in ordered if float(sc) >= cutoff}
    if len(filtered) > TRENDS_MAX_COUNTED_MATCHES:
        keep = ordered[:TRENDS_MAX_COUNTED_MATCHES]
        filtered = {int(mid): float(sc) for mid, sc in keep}
    elif len(filtered) < TRENDS_MIN_COUNTED_MATCHES:
        keep_n = min(len(ordered), TRENDS_MIN_COUNTED_MATCHES)
        keep = ordered[:keep_n]
        filtered = {int(mid): float(sc) for mid, sc in keep}

    return filtered


def fetch_message_word_counts(conn: sqlite3.Connection, message_ids: List[int]) -> dict[int, int]:
    if not message_ids:
        return {}
    placeholders = ",".join(["?"] * len(message_ids))
    cur = conn.cursor()
    cur.execute(
        f"SELECT id, body FROM messages WHERE id IN ({placeholders});",
        tuple(message_ids),
    )
    return {int(r[0]): count_words(r[1] or "") for r in cur.fetchall()}


def center_message_length_prior(word_count: int) -> float:
    wc = int(word_count or 0)
    if wc <= 3:
        return -0.18
    if wc <= 8:
        return -0.08
    if wc <= 20:
        return 0.04
    if wc <= 80:
        return 0.12
    return 0.08


def parse_trend_terms(raw_terms: str, max_terms: int = TRENDS_MAX_TERMS) -> List[str]:
    if not raw_terms:
        return []
    out = []
    seen = set()
    for part in raw_terms.split(","):
        term = normalize_query_text(part)
        if not term:
            continue
        if term in seen:
            continue
        seen.add(term)
        out.append(term)
        if len(out) >= max_terms:
            break
    return out


def get_message_time_bounds(conn: sqlite3.Connection) -> tuple[int | None, int | None]:
    cur = conn.cursor()
    cur.execute("SELECT MIN(ts_unix), MAX(ts_unix) FROM messages;")
    row = cur.fetchone()
    if not row or row[0] is None or row[1] is None:
        return None, None
    return int(row[0]), int(row[1])


def trend_query_variants(term: str) -> List[str]:
    base = normalize_query_text(term)
    if not base:
        return []
    if is_strict_literal_term(base):
        return [base]
    variants = [base]
    for v in expand_query_variants(base):
        if v not in variants:
            variants.append(v)
        if len(variants) >= 8:
            break
    return variants


def compute_trend_match_scores(conn: sqlite3.Connection, term: str) -> dict[int, float]:
    variants = trend_query_variants(term)
    if not variants:
        return {}
    strict_literal = is_strict_literal_term(term)

    exact_ids: List[int] = []
    seen = set()
    per_variant_exact = max(120, TRENDS_EXACT_K // max(1, len(variants)))
    for v in variants:
        ids = exact_search(conn, v, per_variant_exact)
        for mid in ids:
            if mid not in seen:
                seen.add(mid)
                exact_ids.append(mid)
                if len(exact_ids) >= TRENDS_EXACT_K:
                    break
        if len(exact_ids) >= TRENDS_EXACT_K:
            break

    semantic_scores: dict[int, float] = {}
    per_variant_sem = max(180, TRENDS_SEM_K // max(1, len(variants)))
    if strict_literal:
        per_variant_sem = max(40, min(220, per_variant_sem // 6))
    for v in variants:
        vs = semantic_search_scored_cached(conn, v, per_variant_sem)
        for mid, s in vs.items():
            prev = semantic_scores.get(mid)
            if prev is None or s > prev:
                semantic_scores[mid] = float(s)

    combined = {}
    if exact_ids:
        n = max(len(exact_ids), 1)
        for idx, mid in enumerate(exact_ids):
            rank_score = 1.0 - (idx / n)
            combined[mid] = combined.get(mid, 0.0) + (0.62 + 0.38 * rank_score)
    if semantic_scores:
        sem_norm = normalize_scores(semantic_scores)
        for mid, s in sem_norm.items():
            combined[mid] = combined.get(mid, 0.0) + (s * 1.1)

    # Topic presence boost to suppress filler matches.
    topic_terms = extract_topic_terms(term)
    topic_hit_ids: set[int] = set()
    if topic_terms and combined:
        ids = list(combined.keys())
        placeholders = ",".join(["?"] * len(ids))
        cur = conn.cursor()
        cur.execute(f"SELECT id, body FROM messages WHERE id IN ({placeholders});", tuple(ids))
        for mid, body in cur.fetchall():
            if strict_literal:
                hits = count_literal_term_matches(body or "", topic_terms)
            else:
                hits = count_focus_term_matches(body or "", topic_terms)
            if hits > 0:
                topic_hit_ids.add(int(mid))
                combined[int(mid)] = combined.get(int(mid), 0.0) + min(0.25 * hits, 0.75)
            else:
                combined[int(mid)] = combined.get(int(mid), 0.0) - 0.22

    # Gate counting to topic-relevant messages:
    # keep lexical topic hits, plus only very high-confidence semantic-only matches.
    if combined and topic_terms:
        if strict_literal:
            # For single-name/literal terms, only keep true literal hits.
            combined = {int(mid): float(sc) for mid, sc in combined.items() if int(mid) in topic_hit_ids}
        else:
            ordered_scores = sorted(float(v) for v in combined.values())
            if ordered_scores:
                gate_idx = int(len(ordered_scores) * 0.94)
                if gate_idx >= len(ordered_scores):
                    gate_idx = len(ordered_scores) - 1
                high_conf_gate = ordered_scores[max(0, gate_idx)]
                gated = {}
                for mid, sc in combined.items():
                    if int(mid) in topic_hit_ids or float(sc) >= high_conf_gate:
                        gated[int(mid)] = float(sc)
                combined = gated
    return filter_trend_scores_to_strong_matches(combined)


def build_term_bins(
    conn: sqlite3.Connection,
    term: str,
    match_scores: dict[int, float],
    min_ts: int,
    max_ts: int,
    bin_days: int,
) -> tuple[List[int], List[dict], List[int | None]]:
    if min_ts is None or max_ts is None:
        return [], [], []
    bin_seconds = max(1, int(bin_days)) * 86400
    max_idx = (max_ts - min_ts) // bin_seconds
    series = [0 for _ in range(max_idx + 1)]
    bin_top_message_ids: List[int | None] = [None for _ in range(max_idx + 1)]
    bin_top_scores = [-1e18 for _ in range(max_idx + 1)]

    # Keep top matches for highlights with % confidence.
    ranked = sorted(match_scores.items(), key=lambda x: x[1], reverse=True)
    top_ids = [int(mid) for mid, _ in ranked[:TRENDS_HIGHLIGHT_LIMIT]]

    if not match_scores:
        bins = []
        for idx in range(max_idx + 1):
            start_ts = min_ts + idx * bin_seconds
            end_ts = start_ts + bin_seconds
            bins.append({
                "count": 0,
                "start_ts": start_ts,
                "end_ts": end_ts,
                "label": datetime.fromtimestamp(start_ts).strftime("%Y-%m-%d"),
            })
        return top_ids, bins, bin_top_message_ids

    ids = list(match_scores.keys())
    placeholders = ",".join(["?"] * len(ids))
    cur = conn.cursor()
    cur.execute(f"SELECT id, ts_unix FROM messages WHERE id IN ({placeholders});", tuple(ids))
    for mid, ts_unix in cur.fetchall():
        idx = (int(ts_unix) - min_ts) // bin_seconds
        if 0 <= idx <= max_idx:
            series[idx] += 1
            score = float(match_scores.get(int(mid), 0.0))
            if score > bin_top_scores[idx]:
                bin_top_scores[idx] = score
                bin_top_message_ids[idx] = int(mid)

    bins = []
    for idx in range(max_idx + 1):
        start_ts = min_ts + idx * bin_seconds
        end_ts = start_ts + bin_seconds
        bins.append({
            "start_ts": start_ts,
            "end_ts": end_ts,
            "label": datetime.fromtimestamp(start_ts).strftime("%Y-%m-%d"),
        })
    return top_ids, [{"count": series[i], **bins[i]} for i in range(len(bins))], bin_top_message_ids


def merge_candidates_experiment(
    lexical_ranked: List[tuple[int, float]],
    semantic_scores: dict[int, float],
    merge_n: int,
) -> List[int]:
    lex_ids = [mid for mid, _ in lexical_ranked]
    sem_ids = [mid for mid, _ in sorted(semantic_scores.items(), key=lambda x: x[1], reverse=True)]

    if EXPERIMENT_MERGE_METHOD == "weighted":
        lex_score_raw = {}
        for rank, (mid, bm25_score) in enumerate(lexical_ranked, start=1):
            # lower bm25 is better; invert rank and bm25 together for stability
            lex_score_raw[mid] = (1.0 / rank) + (1.0 / (1.0 + max(bm25_score, 0.0)))
        lex_norm = normalize_scores(lex_score_raw)
        sem_norm = normalize_scores(semantic_scores)
        merged = {}
        for mid in set(lex_ids + sem_ids):
            merged[mid] = (
                EXPERIMENT_WEIGHT_LEX * lex_norm.get(mid, 0.0)
                + EXPERIMENT_WEIGHT_SEM * sem_norm.get(mid, 0.0)
            )
        ordered = [mid for mid, _ in sorted(merged.items(), key=lambda x: x[1], reverse=True)]
        return ordered[:merge_n]

    # default RRF
    rrf = {}
    k_const = max(1, EXPERIMENT_RRF_K)
    for rank, mid in enumerate(lex_ids, start=1):
        rrf[mid] = rrf.get(mid, 0.0) + 1.0 / (k_const + rank)
    for rank, mid in enumerate(sem_ids, start=1):
        rrf[mid] = rrf.get(mid, 0.0) + 1.0 / (k_const + rank)
    ordered = [mid for mid, _ in sorted(rrf.items(), key=lambda x: x[1], reverse=True)]
    return ordered[:merge_n]


def infer_signal_boosts_for_query(query: str) -> List[tuple[str, float]]:
    ql = (query or "").lower()
    boosts: List[tuple[str, float]] = []
    for trigger, specs in SIGNAL_QUERY_BOOSTS.items():
        if trigger in ql:
            boosts.extend(specs)

    merged = {}
    for signal_name, weight in boosts:
        if signal_name in SIGNAL_COLUMNS:
            merged[signal_name] = max(float(weight), merged.get(signal_name, 0.0))
    return sorted(merged.items(), key=lambda x: x[1], reverse=True)


def get_search_weights(mode: str) -> dict[str, float]:
    if mode == "exact":
        return {"exact": 1.65, "semantic": 0.0, "rerank": 0.35, "baseline": 0.05}
    if mode == "experiment":
        return {"exact": 1.65, "semantic": 0.0, "rerank": 0.35, "baseline": 0.05}
    if mode == "semantic":
        return {"exact": 0.0, "semantic": 1.25, "rerank": 2.15, "baseline": 0.03}
    if mode == "hybrid":
        return {"exact": 0.60, "semantic": 1.05, "rerank": 1.95, "baseline": 0.04}
    # semantic_first default
    return {"exact": 0.22, "semantic": 1.30, "rerank": 2.35, "baseline": 0.03}


def fetch_signal_scores(conn: sqlite3.Connection, message_ids: List[int], signals: List[str]) -> dict[int, dict[str, float]]:
    if not message_ids or not signals:
        return {}
    valid = [s for s in signals if s in SIGNAL_COLUMNS]
    if not valid:
        return {}

    cur = conn.cursor()
    col_sql = ", ".join(valid)
    placeholders = ",".join(["?"] * len(message_ids))
    cur.execute(
        f"SELECT message_id, {col_sql} FROM message_signals WHERE message_id IN ({placeholders});",
        message_ids,
    )
    out = {}
    for row in cur.fetchall():
        mid = int(row[0])
        out[mid] = {}
        for i, sig in enumerate(valid):
            out[mid][sig] = float(row[i + 1] or 0.0)
    return out


def fetch_window_text_for_message(conn: sqlite3.Connection, center_id: int, before: int = RESULT_CONTEXT_BEFORE, after: int = RESULT_CONTEXT_AFTER) -> str:
    cur = conn.cursor()
    cur.execute("SELECT ts_unix, id FROM messages WHERE id = ?;", (center_id,))
    row = cur.fetchone()
    if not row:
        return ""
    center_ts = int(row[0])

    cur.execute(
        """
        SELECT body
        FROM messages
        WHERE (ts_unix < ?)
           OR (ts_unix = ? AND id <= ?)
        ORDER BY ts_unix DESC, id DESC
        LIMIT ?;
        """,
        (center_ts, center_ts, center_id, before + 1),
    )
    older = [r[0] or "" for r in cur.fetchall()][::-1]

    cur.execute(
        """
        SELECT body
        FROM messages
        WHERE (ts_unix > ?)
           OR (ts_unix = ? AND id > ?)
        ORDER BY ts_unix ASC, id ASC
        LIMIT ?;
        """,
        (center_ts, center_ts, center_id, after),
    )
    newer = [r[0] or "" for r in cur.fetchall()]

    return " \n ".join([x.strip() for x in (older + newer) if x and x.strip()])


def build_window_text_map(conn: sqlite3.Connection, center_ids: List[int], before: int = RESULT_CONTEXT_BEFORE, after: int = RESULT_CONTEXT_AFTER) -> dict[int, str]:
    out: dict[int, str] = {}
    for mid in center_ids:
        out[mid] = fetch_window_text_for_message(conn, mid, before=before, after=after)
    return out


def rerank_message_ids(
    conn: sqlite3.Connection,
    query: str,
    candidate_ids: List[int],
    top_k: int,
    batch_size: int = 32,
) -> dict[int, float]:
    if not candidate_ids:
        return {}

    cur = conn.cursor()
    placeholders = ",".join(["?"] * len(candidate_ids))
    cur.execute(f"SELECT id, body FROM messages WHERE id IN ({placeholders});", candidate_ids)
    text_by_id = {int(row[0]): (row[1] or "") for row in cur.fetchall()}

    ranked_ids = [mid for mid in candidate_ids if mid in text_by_id][:top_k]
    if not ranked_ids:
        return {}

    if reranker is None:
        n = max(len(ranked_ids), 1)
        return {mid: 1.0 - (i / n) for i, mid in enumerate(ranked_ids)}

    pairs = [[query, text_by_id[mid]] for mid in ranked_ids]
    try:
        raw_scores = reranker.predict(pairs, batch_size=max(1, int(batch_size)))
        out = {}
        for i, mid in enumerate(ranked_ids):
            out[mid] = float(raw_scores[i])
        return out
    except Exception:
        n = max(len(ranked_ids), 1)
        return {mid: 1.0 - (i / n) for i, mid in enumerate(ranked_ids)}


def rerank_context_windows(
    query: str,
    ordered_ids: List[int],
    window_text_map: dict[int, str],
    top_k: int,
    batch_size: int = 32,
) -> dict[int, float]:
    ranked_ids = [mid for mid in ordered_ids if (window_text_map.get(mid) or "").strip()][:top_k]
    if not ranked_ids:
        return {}

    if reranker is None:
        n = max(len(ranked_ids), 1)
        return {mid: 1.0 - (i / n) for i, mid in enumerate(ranked_ids)}

    pairs = [[query, window_text_map[mid]] for mid in ranked_ids]
    try:
        raw_scores = reranker.predict(pairs, batch_size=max(1, int(batch_size)))
        return {mid: float(raw_scores[i]) for i, mid in enumerate(ranked_ids)}
    except Exception:
        n = max(len(ranked_ids), 1)
        return {mid: 1.0 - (i / n) for i, mid in enumerate(ranked_ids)}


def score_window_semantic_similarity(query: str, ordered_ids: List[int], window_text_map: dict[int, str]) -> dict[int, float]:
    if model is None:
        return {}
    ranked_ids = [mid for mid in ordered_ids if (window_text_map.get(mid) or "").strip()]
    if not ranked_ids:
        return {}
    try:
        query_vec = model.encode([query], normalize_embeddings=True).astype("float32")
        text_vecs = model.encode(
            [window_text_map[mid] for mid in ranked_ids],
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=False,
        ).astype("float32")
        sims = np.dot(text_vecs, query_vec[0])
        return {mid: float(sims[i]) for i, mid in enumerate(ranked_ids)}
    except Exception:
        return {}


def select_diverse_results_by_mmr(
    query: str,
    ordered_ids: List[int],
    window_text_map: dict[int, str],
    base_score_map: dict[int, float],
    top_k: int,
    lambda_mult: float = MMR_LAMBDA,
) -> List[int]:
    if model is None:
        return ordered_ids[:top_k]
    candidates = [mid for mid in ordered_ids if (window_text_map.get(mid) or "").strip()]
    if not candidates:
        return ordered_ids[:top_k]

    pool = candidates[:max(top_k, MMR_POOL_K)]
    texts = [window_text_map[mid] for mid in pool]
    try:
        query_vec = model.encode([query], normalize_embeddings=True).astype("float32")[0]
        emb = model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=False,
        ).astype("float32")
        rel = np.dot(emb, query_vec)
    except Exception:
        return pool[:top_k]

    raw_scores = np.array([float(base_score_map.get(mid, 0.0)) for mid in pool], dtype="float32")
    if raw_scores.size > 0:
        lo = float(np.min(raw_scores))
        hi = float(np.max(raw_scores))
        denom = (hi - lo) if hi > lo else 1.0
        norm_scores = (raw_scores - lo) / denom
    else:
        norm_scores = raw_scores

    selected_idxs: List[int] = []
    candidate_idxs = list(range(len(pool)))
    while candidate_idxs and len(selected_idxs) < top_k:
        best_idx = None
        best_mmr = -1e9
        for idx in candidate_idxs:
            if not selected_idxs:
                div_penalty = 0.0
            else:
                sim_to_selected = np.dot(emb[idx], emb[selected_idxs].T)
                div_penalty = float(np.max(sim_to_selected))
            relevance = float(0.55 * rel[idx] + 0.45 * norm_scores[idx])
            mmr_score = lambda_mult * relevance - (1.0 - lambda_mult) * div_penalty
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = idx
        if best_idx is None:
            break
        selected_idxs.append(best_idx)
        candidate_idxs.remove(best_idx)

    return [pool[i] for i in selected_idxs][:top_k]


def fetch_messages_by_ids(conn: sqlite3.Connection, ids: List[int]) -> List[dict]:
    if not ids:
        return []
    # preserve order
    cur = conn.cursor()
    out = []
    for mid in ids:
        cur.execute("SELECT id, ts, ts_unix, direction, body FROM messages WHERE id = ?;", (mid,))
        row = cur.fetchone()
        if row:
            out.append({"id": row[0], "ts": row[1], "ts_unix": row[2], "direction": row[3], "body": row[4]})
    by_id = fetch_attachments_for_message_ids(conn, [m["id"] for m in out])
    for msg in out:
        attachments = by_id.get(msg["id"], [])
        msg["attachments"] = attachments
        msg["attachment_count"] = len(attachments)
    return add_display_timestamp_fields(out)

def get_context_window(conn: sqlite3.Connection, center_id: int, before: int, after: int):
    cur = conn.cursor()
    cur.execute("SELECT ts_unix, id FROM messages WHERE id = ?;", (center_id,))
    row = cur.fetchone()
    if not row:
        return [], None, None, (None, None)
    center_ts = int(row[0])

    # Grab older (including the center message)
    cur.execute("""
        SELECT id, ts, ts_unix, direction, body
        FROM messages
        WHERE (ts_unix < ?)
           OR (ts_unix = ? AND id <= ?)
        ORDER BY ts_unix DESC, id DESC
        LIMIT ?;
    """, (center_ts, center_ts, center_id, before + 1))
    older = cur.fetchall()[::-1]  # chronological

    # Grab newer (strictly after center)
    cur.execute("""
        SELECT id, ts, ts_unix, direction, body
        FROM messages
        WHERE (ts_unix > ?)
           OR (ts_unix = ? AND id > ?)
        ORDER BY ts_unix ASC, id ASC
        LIMIT ?;
    """, (center_ts, center_ts, center_id, after))
    newer = cur.fetchall()

    rows = older + newer
    msgs = [{"id": r[0], "ts": r[1], "ts_unix": r[2], "direction": r[3], "body": r[4]} for r in rows]
    by_id = fetch_attachments_for_message_ids(conn, [m["id"] for m in msgs])
    for msg in msgs:
        attachments = by_id.get(msg["id"], [])
        msg["attachments"] = attachments
        msg["attachment_count"] = len(attachments)
    add_display_timestamp_fields(msgs)

    oldest = msgs[0]["ts_unix"] if msgs else None
    newest = msgs[-1]["ts_unix"] if msgs else None

    # For scrolling helpers, return first and last message ids too
    first_id = msgs[0]["id"] if msgs else None
    last_id = msgs[-1]["id"] if msgs else None

    return msgs, oldest, newest, (first_id, last_id)

def get_result_context_batch(conn: sqlite3.Connection, center_id: int, before: int = 2, after: int = 2) -> list[dict]:
    cur = conn.cursor()
    cur.execute("SELECT ts_unix, id FROM messages WHERE id = ?;", (center_id,))
    row = cur.fetchone()
    if not row:
        return []
    center_ts = int(row[0])

    cur.execute("""
        SELECT id, ts, ts_unix, direction, body
        FROM messages
        WHERE (ts_unix < ?)
           OR (ts_unix = ? AND id <= ?)
        ORDER BY ts_unix DESC, id DESC
        LIMIT ?;
    """, (center_ts, center_ts, center_id, before + 1))
    older = cur.fetchall()[::-1]

    cur.execute("""
        SELECT id, ts, ts_unix, direction, body
        FROM messages
        WHERE (ts_unix > ?)
           OR (ts_unix = ? AND id > ?)
        ORDER BY ts_unix ASC, id ASC
        LIMIT ?;
    """, (center_ts, center_ts, center_id, after))
    newer = cur.fetchall()

    rows = older + newer
    msgs = []
    for r in rows:
        msgs.append({
            "id": r[0],
            "ts": r[1],
            "ts_unix": r[2],
            "direction": r[3],
            "body": r[4],
            "is_center": (r[0] == center_id),
        })

    by_id = fetch_attachments_for_message_ids(conn, [m["id"] for m in msgs])
    for msg in msgs:
        attachments = by_id.get(msg["id"], [])
        msg["attachments"] = attachments
        msg["attachment_count"] = len(attachments)

    return add_display_timestamp_fields(msgs)

def build_search_result_batches(
    conn: sqlite3.Connection,
    center_ids: List[int],
    reason_map: dict[int, list[str]] | None = None,
    return_to: str = "/",
    return_label: str = "Search",
) -> List[dict]:
    out = []
    for mid in center_ids:
        msgs = get_result_context_batch(conn, mid, before=RESULT_CONTEXT_BEFORE, after=RESULT_CONTEXT_AFTER)
        if not msgs:
            continue
        out.append({
            "center_id": mid,
            "center_href": build_chat_href(mid, return_to, return_label),
            "messages": msgs,
            "match_reason": (reason_map or {}).get(mid, []),
        })
    return out


def run_experiment_search(conn: sqlite3.Connection, raw_query: str) -> tuple[list[int], dict[int, list[str]]]:
    q = normalize_experiment_query(raw_query)
    if not q:
        return [], {}

    start_ts, end_ts = parse_experiment_date_range(q)
    prefer_earliest = experiment_prefers_earliest(q)
    topic_terms = extract_topic_terms(q)
    focus_terms = topic_terms or extract_focus_terms(q)
    query_terms = extract_query_terms(q)

    # Stage A: candidate generation (lexical + semantic).
    lex_query = build_experiment_lexical_query(q, topic_terms=topic_terms)
    lexical_ranked = fts_search_candidates_scored(
        conn,
        lex_query,
        limit=max(50, EXPERIMENT_K_LEX),
        start_ts=start_ts,
        end_ts=end_ts,
    )
    semantic_scores = semantic_search_scored_cached(conn, q, k=max(50, EXPERIMENT_K_SEM))
    merged_candidates = merge_candidates_experiment(
        lexical_ranked=lexical_ranked,
        semantic_scores=semantic_scores,
        merge_n=max(EXPERIMENT_TOP_K, EXPERIMENT_MERGE_N),
    )
    if start_ts is not None or end_ts is not None:
        cur = conn.cursor()
        placeholders = ",".join(["?"] * len(merged_candidates)) if merged_candidates else "NULL"
        where_parts = [f"id IN ({placeholders})"] if merged_candidates else ["1=0"]
        params: list = merged_candidates[:]
        if start_ts is not None:
            where_parts.append("ts_unix >= ?")
            params.append(start_ts)
        if end_ts is not None:
            where_parts.append("ts_unix < ?")
            params.append(end_ts)
        cur.execute(
            f"SELECT id FROM messages WHERE {' AND '.join(where_parts)};",
            tuple(params),
        )
        filtered = {int(r[0]) for r in cur.fetchall()}
        merged_candidates = [mid for mid in merged_candidates if mid in filtered]

    if not merged_candidates:
        return [], {}

    # Stage B: reranking (context-window first, center-message second).
    rerank_pool_n = min(len(merged_candidates), max(EXPERIMENT_TOP_K, EXPERIMENT_MERGE_N))
    merged_candidates = merged_candidates[:rerank_pool_n]
    merged_windows = build_window_text_map(
        conn,
        merged_candidates,
        before=RESULT_CONTEXT_BEFORE,
        after=RESULT_CONTEXT_AFTER,
    )
    context_rerank_scores = rerank_context_windows(
        q,
        merged_candidates,
        merged_windows,
        top_k=rerank_pool_n,
        batch_size=EXPERIMENT_RERANK_BATCH_SIZE,
    )
    message_rerank_scores = rerank_message_ids(
        conn,
        q,
        merged_candidates,
        top_k=rerank_pool_n,
        batch_size=EXPERIMENT_RERANK_BATCH_SIZE,
    )

    if context_rerank_scores:
        ranked_ids = [mid for mid, _ in sorted(context_rerank_scores.items(), key=lambda x: x[1], reverse=True)]
    elif message_rerank_scores:
        ranked_ids = [mid for mid, _ in sorted(message_rerank_scores.items(), key=lambda x: x[1], reverse=True)]
    else:
        ranked_ids = merged_candidates[:]

    # Keep context relevance high for snippets.
    ranked_windows = build_window_text_map(
        conn,
        ranked_ids[:max(MMR_POOL_K, EXPERIMENT_TOP_K * 4)],
        before=RESULT_CONTEXT_BEFORE,
        after=RESULT_CONTEXT_AFTER,
    )
    combined_for_mmr = {}
    norm_sem = normalize_scores(semantic_scores)
    norm_context_rerank = normalize_scores(context_rerank_scores) if context_rerank_scores else {}
    norm_message_rerank = normalize_scores(message_rerank_scores) if message_rerank_scores else {}
    center_word_counts = fetch_message_word_counts(conn, ranked_ids)
    context_word_counts = {mid: count_words(ranked_windows.get(mid, "")) for mid in ranked_ids}
    context_word_scores = normalize_scores(context_word_counts)
    for i, mid in enumerate(ranked_ids):
        recency_rank = 1.0 - (i / max(len(ranked_ids), 1))
        score = (
            (norm_context_rerank.get(mid, 0.0) * EXPERIMENT_CONTEXT_RERANK_WEIGHT)
            + (norm_message_rerank.get(mid, 0.0) * EXPERIMENT_MESSAGE_RERANK_WEIGHT)
            + (norm_sem.get(mid, 0.0) * EXPERIMENT_SEMANTIC_WEIGHT)
            + (context_word_scores.get(mid, 0.0) * EXPERIMENT_CONTEXT_WORD_WEIGHT)
            + (center_message_length_prior(center_word_counts.get(mid, 0)) * EXPERIMENT_CENTER_LENGTH_WEIGHT)
            + (recency_rank * 0.02)
        )
        if topic_terms:
            topic_hits = count_focus_term_matches(ranked_windows.get(mid, ""), topic_terms)
            if topic_hits > 0:
                score += min(0.08 * topic_hits, 0.24)
            else:
                # Strongly suppress generic high-frequency chatter for "about X" style queries.
                score -= 0.40
                if norm_sem.get(mid, 0.0) < 0.60:
                    score -= 0.20
        combined_for_mmr[mid] = score

    selected = select_diverse_results_by_mmr(
        q,
        ranked_ids,
        ranked_windows,
        combined_for_mmr,
        top_k=EXPERIMENT_TOP_K,
    )

    # Keep this conservative in experiment mode so literal term enforcement
    # does not overpower semantic intent.
    if selected and topic_terms and len(topic_terms) >= 1 and len(q.split()) >= 3:
        focus_windows = build_window_text_map(conn, selected, before=RESULT_CONTEXT_BEFORE, after=RESULT_CONTEXT_AFTER)
        focus_semantic = score_window_semantic_similarity(q, selected, focus_windows)
        selected = enforce_focus_terms_in_top_results(
            selected,
            focus_windows,
            topic_terms,
            focus_semantic,
            top_n=min(FOCUS_ENFORCE_TOP_N, EXPERIMENT_TOP_K),
            high_conf_threshold=0.82,
        )

    # Intent helper: for "first time"/"earliest", prefer earliest among high-confidence hits.
    if prefer_earliest and selected:
        cur = conn.cursor()
        placeholders = ",".join(["?"] * len(selected))
        cur.execute(
            f"SELECT id, ts_unix FROM messages WHERE id IN ({placeholders});",
            tuple(selected),
        )
        ts_map = {int(r[0]): int(r[1] or 0) for r in cur.fetchall()}
        selected = sorted(selected, key=lambda mid: (ts_map.get(mid, math.inf), selected.index(mid)))

    # Final quality pass: prioritize topic-present windows first for "about X" style queries.
    if selected and topic_terms:
        final_windows_for_topic = build_window_text_map(
            conn,
            selected,
            before=RESULT_CONTEXT_BEFORE,
            after=RESULT_CONTEXT_AFTER,
        )
        selected = sorted(
            selected,
            key=lambda mid: (
                0 if has_focus_term_match(final_windows_for_topic.get(mid, ""), topic_terms) else 1,
                selected.index(mid),
            ),
        )

    reason_map: dict[int, list[str]] = {}
    final_windows = build_window_text_map(conn, selected, before=RESULT_CONTEXT_BEFORE, after=RESULT_CONTEXT_AFTER)
    for mid in selected:
        text = (final_windows.get(mid) or "").lower()
        hits = []
        for t in (topic_terms if topic_terms else (focus_terms if focus_terms else query_terms)):
            if has_focus_term_match(text, [t]) and t not in hits:
                hits.append(t)
            if len(hits) >= 4:
                break
        reason_map[mid] = hits
    return selected[:EXPERIMENT_TOP_K], reason_map


def _search_cache_key(raw_q: str, mode: str) -> tuple[str, str]:
    explicit_mid = parse_explicit_message_id_query(raw_q or "")
    if explicit_mid is not None:
        return (f"id:{explicit_mid}", (mode or "semantic_first").strip().lower())
    return (normalize_query_text(raw_q or ""), (mode or "semantic_first").strip().lower())


def parse_explicit_message_id_query(raw_query: str) -> int | None:
    q = (raw_query or "").strip()
    if not q:
        return None

    # Normalize common mobile/full-width hash variants and quoted input.
    q = re.sub(r"[#＃﹟]", "#", q)
    if len(q) >= 2 and q[0] == q[-1] and q[0] in {"'", '"'}:
        q = q[1:-1].strip()

    # Primary: explicit hash form only (strict to avoid changing normal query behavior).
    m = re.fullmatch(r"#\s*(\d+)", q)
    if m:
        try:
            mid = int(m.group(1))
            return mid if mid > 0 else None
        except (TypeError, ValueError):
            return None

    # Alternate explicit ID forms.
    m = re.fullmatch(r"(?:id|msg|message)\s*[:#]?\s*(\d+)", q, flags=re.IGNORECASE)
    if m:
        try:
            mid = int(m.group(1))
            return mid if mid > 0 else None
        except (TypeError, ValueError):
            return None

    # Fallback: numeric-only query (e.g., "21768").
    if re.fullmatch(r"\d{1,9}", q):
        try:
            mid = int(q)
            return mid if mid > 0 else None
        except (TypeError, ValueError):
            return None
    return None


def get_cached_search_results(raw_q: str, mode: str) -> tuple[str, list[dict]] | None:
    if SEARCH_RESULTS_CACHE_TTL_SECONDS <= 0 or SEARCH_RESULTS_CACHE_MAX_ITEMS <= 0:
        return None
    key = _search_cache_key(raw_q, mode)
    now = time.time()
    with search_results_cache_lock:
        row = search_results_cache.get(key)
        if not row:
            return None
        ts, normalized_mode, payload = row
        if (now - float(ts)) > SEARCH_RESULTS_CACHE_TTL_SECONDS:
            search_results_cache.pop(key, None)
            return None
        search_results_cache.move_to_end(key)
        return normalized_mode, copy.deepcopy(payload)


def set_cached_search_results(raw_q: str, mode: str, normalized_mode: str, result_batches: list[dict]) -> None:
    if SEARCH_RESULTS_CACHE_TTL_SECONDS <= 0 or SEARCH_RESULTS_CACHE_MAX_ITEMS <= 0:
        return
    key = _search_cache_key(raw_q, mode)
    with search_results_cache_lock:
        search_results_cache[key] = (time.time(), normalized_mode, copy.deepcopy(result_batches))
        search_results_cache.move_to_end(key)
        while len(search_results_cache) > SEARCH_RESULTS_CACHE_MAX_ITEMS:
            search_results_cache.popitem(last=False)


def build_search_results(raw_q: str, mode: str) -> tuple[str, list[dict]]:
    raw_q = (raw_q or "").strip()
    if mode not in ("exact", "experiment", "semantic", "hybrid", "semantic_first"):
        mode = "semantic_first"
    return_to = f"/search?{urlencode({'q': raw_q, 'mode': mode})}"
    q = normalize_query_text(raw_q)
    if not q:
        return mode, []

    explicit_mid = parse_explicit_message_id_query(raw_q)
    if explicit_mid is not None:
        conn = db()
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM messages WHERE id = ? LIMIT 1;", (explicit_mid,))
        exists = cur.fetchone() is not None
        if exists:
            result_batches = build_search_result_batches(
                conn,
                [explicit_mid],
                reason_map={explicit_mid: [f"#{explicit_mid}"]},
                return_to=return_to,
                return_label="Search Results",
            )
            conn.close()
            set_cached_search_results(raw_q, mode, mode, result_batches)
            return mode, result_batches
        conn.close()

    cached = get_cached_search_results(raw_q, mode)
    if cached is not None:
        return cached

    conn = db()
    if mode == "experiment":
        ids, reason_map = run_experiment_search(conn, raw_q)
        result_batches = build_search_result_batches(
            conn,
            ids,
            reason_map=reason_map,
            return_to=return_to,
            return_label="Search Results",
        )
        conn.close()
        set_cached_search_results(raw_q, mode, mode, result_batches)
        return mode, result_batches

    variants = expand_query_variants(q)
    if not variants:
        variants = [q]
    query_terms = extract_query_terms(q)
    focus_terms = extract_focus_terms(q)
    person_intent = parse_query_person_intent(q)

    exact_ids: List[int] = []
    semantic_scores: dict[int, float] = {}
    candidate_ids: List[int] = []
    seen = set()
    weights = get_search_weights(mode)

    use_exact = mode in ("exact", "experiment", "hybrid", "semantic_first")
    use_semantic = mode in ("semantic", "hybrid", "semantic_first")

    if use_exact:
        exact_limit = SEARCH_CANDIDATE_K if mode in ("exact", "experiment", "hybrid") else max(80, SEARCH_CANDIDATE_K // 3)
        for variant in variants:
            hits = exact_search(conn, variant, limit=max(40, exact_limit // max(1, len(variants))))
            for mid in hits:
                if mid not in seen:
                    seen.add(mid)
                    exact_ids.append(mid)
                    candidate_ids.append(mid)
                    if len(exact_ids) >= exact_limit:
                        break
            if len(exact_ids) >= exact_limit:
                break

    if use_semantic:
        sem_k = SEARCH_CANDIDATE_K
        per_variant_k = max(120, sem_k // max(1, min(len(variants), 5)))
        for variant in variants:
            variant_scores = semantic_search_scored(conn, variant, k=per_variant_k)
            for mid, score in variant_scores.items():
                prev = semantic_scores.get(mid)
                if prev is None or score > prev:
                    semantic_scores[mid] = float(score)
        sem_ordered = [mid for mid, _ in sorted(semantic_scores.items(), key=lambda x: x[1], reverse=True)]
        for mid in sem_ordered:
            if mid not in seen:
                seen.add(mid)
                candidate_ids.append(mid)

    # Score fusion with context-window-first ranking.
    combined_scores: dict[int, float] = {}
    reason_map: dict[int, list[str]] = {}
    if candidate_ids:
        n = max(len(candidate_ids), 1)
        for idx, mid in enumerate(candidate_ids):
            combined_scores[mid] = combined_scores.get(mid, 0.0) + (1.0 - (idx / n)) * weights["baseline"]

        if exact_ids and weights["exact"] > 0.0:
            n_exact = max(len(exact_ids), 1)
            for idx, mid in enumerate(exact_ids):
                lexical_rank = max(0.0, 1.0 - (idx / n_exact))
                combined_scores[mid] = combined_scores.get(mid, 0.0) + weights["exact"] * (0.65 + lexical_rank * 0.35)

        if weights["semantic"] > 0.0:
            for mid, sim in semantic_scores.items():
                combined_scores[mid] = combined_scores.get(mid, 0.0) + max(float(sim), 0.0) * weights["semantic"]

        rerank_candidates = [mid for mid, _ in sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)]
        rerank_raw = rerank_message_ids(
            conn,
            q,
            rerank_candidates[:RERANK_TOP_K],
            top_k=min(len(rerank_candidates), RERANK_TOP_K),
        )
        if rerank_raw:
            vals = list(rerank_raw.values())
            lo = min(vals)
            hi = max(vals)
            span = (hi - lo) if hi > lo else 1.0
            for mid, raw_score in rerank_raw.items():
                normalized = (raw_score - lo) / span
                combined_scores[mid] = combined_scores.get(mid, 0.0) + normalized * weights["rerank"]

        # Context-aware reranking and semantic scoring on windows around each candidate.
        context_candidates = rerank_candidates[:CONTEXT_RERANK_TOP_K]
        window_text_map = build_window_text_map(
            conn,
            context_candidates,
            before=RESULT_CONTEXT_BEFORE,
            after=RESULT_CONTEXT_AFTER,
        )
        context_rerank_raw = rerank_context_windows(
            q,
            context_candidates,
            window_text_map,
            top_k=min(len(context_candidates), CONTEXT_RERANK_TOP_K),
        )
        if context_rerank_raw:
            vals = list(context_rerank_raw.values())
            lo = min(vals)
            hi = max(vals)
            span = (hi - lo) if hi > lo else 1.0
            for mid, raw_score in context_rerank_raw.items():
                normalized = (raw_score - lo) / span
                combined_scores[mid] = combined_scores.get(mid, 0.0) + normalized * (weights["rerank"] * 1.05)

        window_semantic = score_window_semantic_similarity(q, context_candidates, window_text_map)
        if window_semantic:
            vals = list(window_semantic.values())
            lo = min(vals)
            hi = max(vals)
            span = (hi - lo) if hi > lo else 1.0
            for mid, raw_score in window_semantic.items():
                normalized = (raw_score - lo) / span
                combined_scores[mid] = combined_scores.get(mid, 0.0) + normalized * 1.65

        # Subject-focus boost: prioritize windows containing core topic terms.
        if focus_terms:
            for mid in context_candidates:
                focus_boost = compute_focus_term_boost(window_text_map.get(mid, ""), focus_terms)
                combined_scores[mid] = combined_scores.get(mid, 0.0) + focus_boost

        # Lightweight person-aware interpretation for queries like:
        # "from person_a", "by person_b", "person_b's plans", "about person_a".
        if person_intent.get("person") in PERSON_TOKENS and person_intent.get("mode") in {"sender", "about"}:
            sender_map = fetch_sender_map(conn, context_candidates)
            person = person_intent["person"]
            mode = person_intent["mode"]
            opposite = "person_b" if person == "person_a" else "person_a"
            for mid in context_candidates:
                sender = sender_map.get(mid, "unknown")
                window_text = window_text_map.get(mid, "")
                if mode == "sender":
                    if sender == person:
                        combined_scores[mid] = combined_scores.get(mid, 0.0) + 0.42
                    else:
                        combined_scores[mid] = combined_scores.get(mid, 0.0) - 0.28
                else:
                    # About/possessive: prefer windows that mention the person name;
                    # slight boost when sender is the opposite person (often speaking about them).
                    if has_focus_term_match(window_text, [person]):
                        combined_scores[mid] = combined_scores.get(mid, 0.0) + 0.24
                    if sender == opposite:
                        combined_scores[mid] = combined_scores.get(mid, 0.0) + 0.08

        boosts = infer_signal_boosts_for_query(q)
        if boosts:
            score_map = fetch_signal_scores(conn, candidate_ids, [name for name, _ in boosts])
            for mid in candidate_ids:
                row = score_map.get(mid, {})
                for signal_name, weight in boosts:
                    combined_scores[mid] = combined_scores.get(mid, 0.0) + float(row.get(signal_name, 0.0)) * float(weight)

    if not combined_scores and q:
        fallback_ids = exact_search(conn, q, limit=TOP_K)
        ids = fallback_ids[:TOP_K]
    else:
        ranked_ids = [mid for mid, _ in sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)]
        final_windows = build_window_text_map(
            conn,
            ranked_ids[:max(MMR_POOL_K, TOP_K * 4)],
            before=RESULT_CONTEXT_BEFORE,
            after=RESULT_CONTEXT_AFTER,
        )
        ids = select_diverse_results_by_mmr(
            q,
            ranked_ids,
            final_windows,
            combined_scores,
            top_k=TOP_K,
        )

    if ids and focus_terms:
        focus_windows = build_window_text_map(conn, ids, before=RESULT_CONTEXT_BEFORE, after=RESULT_CONTEXT_AFTER)
        focus_semantic = score_window_semantic_similarity(q, ids, focus_windows)
        ids = enforce_focus_terms_in_top_results(
            ids,
            focus_windows,
            focus_terms,
            focus_semantic,
            top_n=min(FOCUS_ENFORCE_TOP_N, TOP_K),
            high_conf_threshold=FOCUS_HIGH_CONFIDENCE_THRESHOLD,
        )

    # Build simple "why matched" tags from the chosen context window.
    if ids:
        final_windows = build_window_text_map(conn, ids, before=RESULT_CONTEXT_BEFORE, after=RESULT_CONTEXT_AFTER)
        for mid in ids:
            text = (final_windows.get(mid) or "").lower()
            hits = []
            for t in (focus_terms if focus_terms else query_terms):
                if has_focus_term_match(text, [t]) and t not in hits:
                    hits.append(t)
                if len(hits) >= 4:
                    break
            reason_map[mid] = hits

    result_batches = build_search_result_batches(
        conn,
        ids,
        reason_map=reason_map,
        return_to=return_to,
        return_label="Search Results",
    )
    conn.close()
    set_cached_search_results(raw_q, mode, mode, result_batches)
    return mode, result_batches


@app.get("/search", response_class=HTMLResponse)
def search_page(request: Request, q: str = "", mode: str = "semantic_first"):
    if not require_auth(request):
        return HTMLResponse("Not authorized", status_code=401)

    normalized_mode, result_batches = build_search_results(q, mode)
    return templates.TemplateResponse("search_page.html", {
        "request": request,
        "q": (q or "").strip(),
        "mode": normalized_mode,
        "result_batches": result_batches,
    })


@app.post("/search", response_class=HTMLResponse)
def search_post(request: Request, q: str = Form(""), mode: str = Form("semantic_first")):
    if not require_auth(request):
        return HTMLResponse("Not authorized", status_code=401)
    params = urlencode({"q": (q or "").strip(), "mode": mode or "semantic_first"})
    return RedirectResponse(url=f"/search?{params}", status_code=303)

@app.get("/bookmarks/heart/{message_id}", response_class=HTMLResponse)
def heart_partial(request: Request, message_id: int):
    if not require_auth(request):
        return HTMLResponse("Not", status_code=401)

    conn = db()
    marked = is_bookmarked(conn, message_id)
    conn.close()

    return templates.TemplateResponse("heart.html", {
        "request": request,
        "message_id": message_id,
        "marked": marked
    })

@app.get("/bookmarks/picker", response_class=HTMLResponse)
def bookmark_picker(request: Request, message_id: int):
    if not require_auth(request):
        return HTMLResponse("Not authorized", status_code=401)

    conn = db()
    categories = fetch_categories(conn)
    conn.close()

    return templates.TemplateResponse("bookmark_picker.html", {
        "request": request,
        "message_id": message_id,
        "categories": categories
    })

@app.get("/bookmarks/picker/new-name", response_class=HTMLResponse)
def bookmark_picker_new_name(request: Request, message_id: int, name: str = "", error: str = ""):
    if not require_auth(request):
        return HTMLResponse("Not authorized", status_code=401)
    return templates.TemplateResponse("bookmark_new_name.html", {
        "request": request,
        "message_id": message_id,
        "name": (name or "").strip(),
        "error": (error or "").strip(),
    })

@app.post("/bookmarks/picker/new-name", response_class=HTMLResponse)
def bookmark_picker_new_name_post(request: Request, message_id: int = Form(...), name: str = Form("")):
    if not require_auth(request):
        return HTMLResponse("Not authorized", status_code=401)

    clean_name = (name or "").strip()
    if not clean_name:
        return templates.TemplateResponse("bookmark_new_name.html", {
            "request": request,
            "message_id": message_id,
            "name": "",
            "error": "Please enter a folder name.",
        })

    return templates.TemplateResponse("bookmark_new_icon.html", {
        "request": request,
        "message_id": message_id,
        "name": clean_name,
        "icon_options": ICON_OPTIONS,
        "default_icon_key": DEFAULT_ICON_KEY,
    })

@app.post("/bookmarks/add-new", response_class=HTMLResponse)
def add_bookmark_new_folder(
    request: Request,
    message_id: int = Form(...),
    name: str = Form(""),
    icon_key: str = Form(DEFAULT_ICON_KEY),
):
    if not require_auth(request):
        return HTMLResponse("Not authorized", status_code=401)

    clean_name = (name or "").strip()
    if not clean_name:
        return HTMLResponse("Folder name required", status_code=400)

    safe_icon_key = icon_key if icon_key in ICON_MAP else DEFAULT_ICON_KEY

    conn = db()
    cur = conn.cursor()

    try:
        cur.execute(
            "INSERT INTO bookmark_categories(name, icon_key) VALUES(?, ?);",
            (clean_name, safe_icon_key),
        )
        conn.commit()
        category_id = int(cur.lastrowid or 0)
    except sqlite3.IntegrityError:
        cur.execute("SELECT id FROM bookmark_categories WHERE name = ?;", (clean_name,))
        row = cur.fetchone()
        category_id = int(row[0]) if row else 0

    if category_id:
        cur.execute(
            "INSERT OR IGNORE INTO bookmarks(message_id, category_id) VALUES(?, ?);",
            (message_id, category_id),
        )
        conn.commit()

    marked = is_bookmarked(conn, message_id)
    conn.close()

    return templates.TemplateResponse("heart.html", {
        "request": request,
        "message_id": message_id,
        "marked": marked
    })

@app.post("/bookmarks/add", response_class=HTMLResponse)
def add_bookmark(request: Request, message_id: int = Form(...), category_id: int = Form(...)):
    if not require_auth(request):
        return HTMLResponse("Not authorized", status_code=401)

    conn = db()
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO bookmarks(message_id, category_id) VALUES(?, ?);", (message_id, category_id))
    conn.commit()

    marked = is_bookmarked(conn, message_id)
    conn.close()

    return templates.TemplateResponse("heart.html", {
        "request": request,
        "message_id": message_id,
        "marked": marked
    })

@app.post("/bookmarks/remove", response_class=HTMLResponse)
def remove_bookmark(request: Request, message_id: int = Form(...)):
    if not require_auth(request):
        return HTMLResponse("Not authorized", status_code=401)

    conn = db()
    cur = conn.cursor()
    cur.execute("DELETE FROM bookmarks WHERE message_id = ?;", (message_id,))
    conn.commit()
    conn.close()

    return templates.TemplateResponse("heart.html", {
        "request": request,
        "message_id": message_id,
        "marked": False,
    })


@app.post("/bookmarks/categories/{category_id}/delete", response_class=HTMLResponse)
def delete_category(request: Request, category_id: int):
    if not require_auth(request):
        return HTMLResponse("Not authorized", status_code=401)

    conn = db()
    cur = conn.cursor()

    # delete bookmarks in that category first
    cur.execute("DELETE FROM bookmarks WHERE category_id = ?;", (category_id,))
    cur.execute("DELETE FROM bookmark_categories WHERE id = ?;", (category_id,))
    conn.commit()

    categories = fetch_categories(conn)
    conn.close()

    return templates.TemplateResponse("bookmark_folders.html", {"request": request, "categories": categories})

@app.get("/chat/{message_id}", response_class=HTMLResponse)
def chat(request: Request, message_id: int):
    if not require_auth(request):
        return HTMLResponse("Not authorized", status_code=401)

    conn = db()

    msgs, oldest_ts, newest_ts, (first_id, last_id) = get_context_window(conn, message_id, CONTEXT_BEFORE, CONTEXT_AFTER)
    bookmarked_ids = fetch_bookmarked_ids(conn, [m["id"] for m in msgs])
    conn.close()
    back_href = sanitize_internal_return_to(request.query_params.get("return_to"), "/")
    back_label = (request.query_params.get("return_label") or "Search").strip()
    if not back_label:
        back_label = "Search"
    back_label = back_label[:60]

    return templates.TemplateResponse("chat.html", {
        "request": request,
        "center_id": message_id,
        "messages": msgs,
        "oldest_ts": oldest_ts,
        "newest_ts": newest_ts,
        "first_id": first_id,
        "last_id": last_id,
        "bookmarked_ids": bookmarked_ids,
        "back_href": back_href,
        "back_label": back_label,
    })



@app.get("/bookmarks/folders", response_class=HTMLResponse)
def bookmark_folders(request: Request):
    if not require_auth(request):
        return HTMLResponse("Not authorized", status_code=401)

    conn = db()
    categories = fetch_categories(conn)
    conn.close()
    return templates.TemplateResponse("bookmark_folders.html", {
        "request": request,
        "categories": categories
    })


@app.get("/chat/{message_id}/older", response_class=HTMLResponse)
def chat_older(request: Request, message_id: int, before_ts: int, before_id: int):
    if not require_auth(request):
        return HTMLResponse("Not authorized", status_code=401)

    conn = db()
    cur = conn.cursor()

    cur.execute("""
        SELECT id, ts, ts_unix, direction, body
        FROM messages
        WHERE (ts_unix < ?)
           OR (ts_unix = ? AND id < ?)
        ORDER BY ts_unix DESC, id DESC
        LIMIT ?;
    """, (before_ts, before_ts, before_id, 50))
    rows = cur.fetchall()[::-1]  # chronological
    msgs = [{"id": r[0], "ts": r[1], "ts_unix": r[2], "direction": r[3], "body": r[4]} for r in rows]
    by_id = fetch_attachments_for_message_ids(conn, [m["id"] for m in msgs])
    for msg in msgs:
        attachments = by_id.get(msg["id"], [])
        msg["attachments"] = attachments
        msg["attachment_count"] = len(attachments)
    add_display_timestamp_fields(msgs)
    bookmarked_ids = fetch_bookmarked_ids(conn, [m["id"] for m in msgs])
    conn.close()

    return templates.TemplateResponse("chat_chunk.html", {
        "request": request,
        "messages": msgs,
        "bookmarked_ids": bookmarked_ids,
    })

@app.get("/chat/{message_id}/newer", response_class=HTMLResponse)
def chat_newer(request: Request, message_id: int, after_ts: int, after_id: int):
    if not require_auth(request):
        return HTMLResponse("Not authorized", status_code=401)

    conn = db()
    cur = conn.cursor()

    cur.execute("""
        SELECT id, ts, ts_unix, direction, body
        FROM messages
        WHERE (ts_unix > ?)
           OR (ts_unix = ? AND id > ?)
        ORDER BY ts_unix ASC, id ASC
        LIMIT ?;
    """, (after_ts, after_ts, after_id, 50))
    rows = cur.fetchall()
    msgs = [{"id": r[0], "ts": r[1], "ts_unix": r[2], "direction": r[3], "body": r[4]} for r in rows]
    by_id = fetch_attachments_for_message_ids(conn, [m["id"] for m in msgs])
    for msg in msgs:
        attachments = by_id.get(msg["id"], [])
        msg["attachments"] = attachments
        msg["attachment_count"] = len(attachments)
    add_display_timestamp_fields(msgs)
    bookmarked_ids = fetch_bookmarked_ids(conn, [m["id"] for m in msgs])
    conn.close()

    return templates.TemplateResponse("chat_chunk.html", {
        "request": request,
        "messages": msgs,
        "bookmarked_ids": bookmarked_ids,
    })

@app.get("/api/bookmarks/categories")
def get_categories(request: Request):
    if not require_auth(request):
        return []

    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT id, name FROM bookmark_categories ORDER BY name;")
    rows = cur.fetchall()
    conn.close()

    return [{"id": r[0], "name": r[1]} for r in rows]

@app.post("/bookmarks/categories", response_class=HTMLResponse)
def create_category(request: Request, name: str = Form(...), icon_key: str = Form(DEFAULT_ICON_KEY)):
    if not require_auth(request):
        return HTMLResponse("Not authorized", status_code=401)

    conn = db()
    cur = conn.cursor()

    icon_key = icon_key if icon_key in ICON_MAP else DEFAULT_ICON_KEY
    n = (name or "").strip()
    if n:
        try:
            cur.execute("INSERT INTO bookmark_categories(name, icon_key) VALUES(?, ?);", (n, icon_key))
            conn.commit()
        except sqlite3.IntegrityError:
            pass
    categories = fetch_categories(conn)
    conn.close()
    return templates.TemplateResponse("bookmark_folders.html", {
        "request": request,
        "categories": categories
    })


@app.get("/bookmarks/{category_id}", response_class=HTMLResponse)
def view_category(request: Request, category_id: int):
    if not require_auth(request):
        return HTMLResponse("Not authorized", status_code=401)

    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT name, icon_key FROM bookmark_categories WHERE id = ?;", (category_id,))
    cat = cur.fetchone()
    if not cat:
        conn.close()
        return HTMLResponse("Category not found", status_code=404)

    cur.execute("""
        SELECT m.id
        FROM bookmarks b
        JOIN messages m ON m.id = b.message_id
        WHERE b.category_id = ?
        ORDER BY m.ts_unix ASC, m.id ASC;
    """, (category_id,))
    center_ids = [int(r[0]) for r in cur.fetchall()]

    result_batches = []
    return_to = f"/bookmarks/{category_id}"
    for mid in center_ids:
        msgs = get_result_context_batch(conn, mid, before=2, after=2)
        if not msgs:
            continue
        result_batches.append({
            "center_id": mid,
            "center_href": build_chat_href(mid, return_to, "Bookmarks"),
            "messages": msgs,
        })
    conn.close()

    return templates.TemplateResponse("bookmarks.html", {
        "request": request,
        "category_id": category_id,
        "category_name": cat[0] if cat else "Bookmarks",
        "category_icon_entity": ICON_MAP.get(cat[1] if cat else DEFAULT_ICON_KEY, ICON_MAP[DEFAULT_ICON_KEY])["entity"],
        "result_batches": result_batches,
    })

@app.get("/attachments/{attachment_id}")
def attachment_file(request: Request, attachment_id: int):
    if not require_auth(request):
        return HTMLResponse("Not authorized", status_code=401)

    conn = db()
    cur = conn.cursor()
    cur.execute(
        "SELECT stored_path, mime_type, filename FROM attachments WHERE id = ?;",
        (attachment_id,),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return HTMLResponse("Attachment not found", status_code=404)

    stored_path, mime_type, filename = row
    base = ATTACHMENTS_DIR.resolve()
    path = (ATTACHMENTS_DIR / stored_path).resolve()
    if base not in path.parents and path != base:
        return HTMLResponse("Bad attachment path", status_code=400)
    if not path.exists():
        return HTMLResponse("Attachment file missing", status_code=404)

    download_name = (filename or "").strip()
    if download_name.lower() in {"null", "none", "(null)", "undefined", ""}:
        ext = path.suffix or ""
        download_name = f"attachment-{attachment_id}{ext}"

    return FileResponse(
        path=str(path),
        media_type=(mime_type or None),
        filename=download_name,
    )

@app.get("/attachments/{attachment_id}/preview")
def attachment_preview(request: Request, attachment_id: int):
    if not require_auth(request):
        return HTMLResponse("Not authorized", status_code=401)

    conn = db()
    cur = conn.cursor()
    cur.execute(
        "SELECT stored_path, mime_type, filename FROM attachments WHERE id = ?;",
        (attachment_id,),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return HTMLResponse("Attachment not found", status_code=404)

    stored_path, mime_type, filename = row
    base = ATTACHMENTS_DIR.resolve()
    path = (ATTACHMENTS_DIR / stored_path).resolve()
    if base not in path.parents and path != base:
        return HTMLResponse("Bad attachment path", status_code=400)
    if not path.exists():
        return HTMLResponse("Attachment file missing", status_code=404)

    mt = (mime_type or "").lower()
    if not mt.startswith("image/"):
        return HTMLResponse("Attachment is not an image", status_code=415)

    if _is_heic_like(mime_type or "", filename or "", stored_path or ""):
        if not HEIF_PREVIEW_AVAILABLE:
            return HTMLResponse("HEIC preview support is not installed", status_code=501)
        try:
            preview_path = _preview_cache_path(attachment_id, path)
            if not preview_path.exists():
                _render_preview_jpeg(path, preview_path)
            return FileResponse(path=str(preview_path), media_type="image/jpeg")
        except Exception:
            return HTMLResponse("Could not render HEIC preview", status_code=500)

    return FileResponse(path=str(path), media_type=(mime_type or None))




