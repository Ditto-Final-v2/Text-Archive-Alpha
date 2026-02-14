import argparse
import json
import re
import sqlite3
from typing import Dict, List

import torch
from tqdm import tqdm
from transformers import pipeline

DB_PATH = "texts.db"
MODEL_NAME = "SamLowe/roberta-base-go_emotions"

SIGNAL_MAP = {
    "love": {"love": 0.70, "caring": 0.20, "gratitude": 0.10},
    "flirting": {"desire": 0.85, "love": 0.10, "excitement": 0.05},
    "compliments": {"admiration": 0.50, "approval": 0.30, "pride": 0.20},
    "laughing": {"amusement": 0.70, "joy": 0.30},
    "supportive": {"caring": 0.50, "gratitude": 0.30, "relief": 0.20},
    "difficult": {
        "sadness": 0.20,
        "anger": 0.20,
        "fear": 0.15,
        "nervousness": 0.15,
        "disappointment": 0.15,
        "grief": 0.15,
    },
    "repair": {"remorse": 0.50, "embarrassment": 0.30, "sadness": 0.10, "disappointment": 0.10},
    "gratitude": {"gratitude": 1.00},
    "missing": {"sadness": 0.40, "love": 0.35, "desire": 0.25},
}

SEXUAL_TERMS = [
    "sexy", "sexier", "hottest", "hot", "horny", "turned on", "turn me on", "nude", "nudes", "naked",
    "boobs", "tits", "ass", "booty", "cock", "dick", "pussy", "cum", "wet", "hard", "orgasm",
    "ride me", "sit on", "inside you", "make me", "fuck me", "f\\*ck me", "fucking",
]

ATTRACTION_TERMS = [
    "beautiful", "gorgeous", "pretty", "stunning", "handsome", "sexy", "hot", "cute", "fine",
]

FLIRT_STRONG_RE = re.compile("|".join(re.escape(x) for x in SEXUAL_TERMS), flags=re.IGNORECASE)
FLIRT_ATTR_RE = re.compile("|".join(re.escape(x) for x in ATTRACTION_TERMS), flags=re.IGNORECASE)
SENDER_TOKEN_IGNORE = {
    "sms", "mms", "imessage", "message", "messages", "text", "texts", "sent", "received",
    "incoming", "outgoing", "to", "from", "and", "the", "with", "chat", "phone",
}


def ensure_signals_schema(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute(
        """
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
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_message_signals_ts_unix ON message_signals(ts_unix);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_message_signals_sender_ts ON message_signals(sender, ts_unix);")
    conn.commit()


def infer_sender_hints(conn: sqlite3.Connection) -> tuple[set[str], set[str]]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT lower(direction) AS d, COUNT(*) AS c
        FROM messages
        WHERE trim(ifnull(direction,'')) != ''
        GROUP BY d
        ORDER BY c DESC;
        """
    )
    counts: dict[str, int] = {}
    for d, c in cur.fetchall():
        for tok in re.findall(r"[a-z]{3,}", d or ""):
            if tok in SENDER_TOKEN_IGNORE:
                continue
            counts[tok] = counts.get(tok, 0) + int(c or 0)
    ranked = [k for k, _ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)]
    a = set([ranked[0]]) if ranked else set()
    b = set([ranked[1]]) if len(ranked) > 1 else set()
    return a, b


def detect_sender(direction: str, person_a_hints: set[str], person_b_hints: set[str]) -> str:
    d = (direction or "").lower()
    if "person_a" in d or "curtis" in d:
        return "person_a"
    if "person_b" in d or "ollie" in d:
        return "person_b"
    if ("sent" in d or "outgoing" in d) and ("received" not in d and "incoming" not in d):
        return "person_a"
    if ("received" in d or "incoming" in d) and ("sent" not in d and "outgoing" not in d):
        return "person_b"
    for tok in person_a_hints:
        if re.search(rf"\b{re.escape(tok)}\b", d):
            return "person_a"
    for tok in person_b_hints:
        if re.search(rf"\b{re.escape(tok)}\b", d):
            return "person_b"
    return "unknown"


def word_count(text: str) -> int:
    t = (text or "").strip()
    if not t:
        return 0
    return len(t.split())


def to_label_scores(raw_scores: List[Dict]) -> Dict[str, float]:
    if isinstance(raw_scores, dict):
        raw_scores = [raw_scores]
    out = {}
    for row in raw_scores:
        label = str(row.get("label", "")).lower()
        score = float(row.get("score", 0.0))
        out[label] = score
    return out


def map_signals(label_scores: Dict[str, float]) -> Dict[str, float]:
    sig = {}
    for signal, weights in SIGNAL_MAP.items():
        value = 0.0
        for label, w in weights.items():
            value += w * float(label_scores.get(label, 0.0))
        sig[signal] = max(0.0, min(1.0, value))
    return sig

def flirting_text_boost(text: str, label_scores: Dict[str, float]) -> float:
    t = (text or "").strip()
    if not t:
        return 0.0
    base_desire = float(label_scores.get("desire", 0.0))
    strong_hits = len(FLIRT_STRONG_RE.findall(t))
    attraction_hits = len(FLIRT_ATTR_RE.findall(t))

    # Classifier-first: keyword boost only nudges and is capped.
    boost = 0.0
    if strong_hits > 0:
        boost += min(0.30, 0.15 + 0.08 * (strong_hits - 1))
    if attraction_hits > 0:
        boost += min(0.18, 0.08 + 0.04 * (attraction_hits - 1))
    boost += min(0.15, base_desire * 0.20)
    return min(0.45, boost)


def main():
    parser = argparse.ArgumentParser(description="Build message_signals from messages using local go_emotions classifier")
    parser.add_argument("--db", default=DB_PATH, help="Path to SQLite DB")
    parser.add_argument("--model", default=MODEL_NAME, help="HF model name")
    parser.add_argument("--limit", type=int, default=0, help="Optional max messages to process")
    parser.add_argument("--resume", action="store_true", help="Only process messages missing from message_signals")
    parser.add_argument("--batch-size", type=int, default=192, help="Inference batch size")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Inference device: auto (default), cpu, or cuda",
    )
    parser.add_argument("--max-length", type=int, default=256, help="Tokenizer max_length")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    ensure_signals_schema(conn)
    cur = conn.cursor()

    where_clause = ""
    if args.resume:
        where_clause = "WHERE m.id NOT IN (SELECT message_id FROM message_signals)"

    limit_clause = ""
    params: tuple = ()
    if args.limit and args.limit > 0:
        limit_clause = " LIMIT ? "
        params = (args.limit,)

    cur.execute(f"""
        SELECT m.id, m.ts_unix, m.direction, m.body
        FROM messages m
        {where_clause}
        ORDER BY m.id ASC
        {limit_clause};
    """, params)
    rows = cur.fetchall()

    if not rows:
        print("No messages to process.")
        conn.close()
        return
    person_a_hints, person_b_hints = infer_sender_hints(conn)

    cuda_available = torch.cuda.is_available()
    if args.device == "cuda":
        if not cuda_available:
            raise RuntimeError("Requested --device cuda but torch.cuda.is_available() is False.")
        device = 0
    elif args.device == "cpu":
        device = -1
    else:
        device = 0 if cuda_available else -1

    if device == 0:
        device_name = torch.cuda.get_device_name(0)
        print(f"Using CUDA device 0: {device_name}")
    else:
        print("Using CPU device")

    model_kwargs = {}
    if device == 0:
        model_kwargs["dtype"] = torch.float16

    try:
        clf = pipeline(
            "text-classification",
            model=args.model,
            tokenizer=args.model,
            top_k=None,
            function_to_apply="sigmoid",
            truncation=True,
            max_length=args.max_length,
            device=device,
            model_kwargs=model_kwargs,
        )
    except TypeError:
        clf = pipeline(
            "text-classification",
            model=args.model,
            tokenizer=args.model,
            return_all_scores=True,
            function_to_apply="sigmoid",
            truncation=True,
            max_length=args.max_length,
            device=device,
            model_kwargs=model_kwargs,
        )

    insert_sql = """
        INSERT INTO message_signals(
            message_id, ts_unix, sender, word_count,
            love, flirting, compliments, laughing, supportive, difficult, repair, gratitude, missing,
            top_labels_json
        )
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(message_id) DO UPDATE SET
            ts_unix=excluded.ts_unix,
            sender=excluded.sender,
            word_count=excluded.word_count,
            love=excluded.love,
            flirting=excluded.flirting,
            compliments=excluded.compliments,
            laughing=excluded.laughing,
            supportive=excluded.supportive,
            difficult=excluded.difficult,
            repair=excluded.repair,
            gratitude=excluded.gratitude,
            missing=excluded.missing,
            top_labels_json=excluded.top_labels_json;
    """

    out_rows = []
    texts = [str(r[3] or "") for r in rows]

    for i in tqdm(range(0, len(rows), args.batch_size), desc="Signals"):
        batch = rows[i : i + args.batch_size]
        batch_texts = texts[i : i + args.batch_size]
        scores_batch = clf(batch_texts)

        for rec, raw_scores in zip(batch, scores_batch):
            mid = int(rec[0])
            ts_unix = int(rec[1] or 0)
            sender = detect_sender(str(rec[2] or ""), person_a_hints, person_b_hints)
            body = str(rec[3] or "")
            wc = word_count(body)

            label_scores = to_label_scores(raw_scores)
            if isinstance(raw_scores, dict):
                score_rows = [raw_scores]
            else:
                score_rows = raw_scores
            top5 = sorted(score_rows, key=lambda x: float(x.get("score", 0.0)), reverse=True)[:5]
            top_labels_json = json.dumps(
                [{"label": str(x.get("label", "")).lower(), "score": float(x.get("score", 0.0))} for x in top5],
                ensure_ascii=True,
            )
            sig = map_signals(label_scores)
            sig["flirting"] = max(sig["flirting"], min(1.0, sig["flirting"] + flirting_text_boost(body, label_scores)))

            out_rows.append(
                (
                    mid,
                    ts_unix,
                    sender,
                    wc,
                    sig["love"],
                    sig["flirting"],
                    sig["compliments"],
                    sig["laughing"],
                    sig["supportive"],
                    sig["difficult"],
                    sig["repair"],
                    sig["gratitude"],
                    sig["missing"],
                    top_labels_json,
                )
            )

        if len(out_rows) >= 1000:
            cur.executemany(insert_sql, out_rows)
            conn.commit()
            out_rows.clear()

    if out_rows:
        cur.executemany(insert_sql, out_rows)
        conn.commit()

    conn.close()
    print(f"Processed {len(rows)} messages.")


if __name__ == "__main__":
    main()


