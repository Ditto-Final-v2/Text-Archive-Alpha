import argparse
import base64
import csv
import mimetypes
import os
import re
import sqlite3
import xml.etree.ElementTree as ET
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch

INPUT_CSV = "long_messages.csv"
INPUT_XML = "messages.xml"
DB_PATH = "texts.db"
FAISS_INDEX_PATH = "texts.faiss"
ATTACHMENTS_DIR = "attachments"
MODEL_NAME = os.environ.get("TEXTAPP_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

DEFAULT_OWNER_NAME = os.environ.get("TEXTAPP_OWNER_NAME", "person_a")
DEFAULT_PARTNER_NAME = os.environ.get("TEXTAPP_PARTNER_NAME", "person_b")
CSV_BODY_COLUMN_CANDIDATES = [
    "message", "content", "body", "text", "message_text", "sms", "imessage",
]
CSV_DIRECTION_COLUMN_CANDIDATES = [
    "direction", "sender", "from", "author", "who",
]
CSV_DATE_COLUMN_CANDIDATES = [
    "date", "timestamp", "time", "sent_at", "created_at", "datetime",
]


def _detect_csv_header_row(csv_path: Path, max_scan_rows: int = 8) -> tuple[int, str]:
    sample = csv_path.read_text(encoding="utf-8-sig", errors="ignore")
    try:
        dialect = csv.Sniffer().sniff(sample[:4096], delimiters=",;\t|")
        delimiter = dialect.delimiter
    except Exception:
        delimiter = ","

    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh, delimiter=delimiter)
        for i, row in enumerate(reader):
            if i >= max_scan_rows:
                break
            normalized = {str(cell or "").strip().lower() for cell in row}
            if normalized & set(CSV_BODY_COLUMN_CANDIDATES):
                return i, delimiter
    return 0, delimiter


def ensure_schema(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            ts_unix INTEGER NOT NULL,
            direction TEXT NOT NULL,
            body TEXT NOT NULL
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_messages_ts_unix ON messages(ts_unix);")

    cur.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
            body,
            content='messages',
            content_rowid='id'
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS attachments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id INTEGER NOT NULL,
            kind TEXT,
            mime_type TEXT,
            filename TEXT,
            stored_path TEXT NOT NULL,
            width INTEGER,
            height INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(message_id) REFERENCES messages(id)
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_attachments_message_id ON attachments(message_id);")

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS bookmark_categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            icon_key TEXT NOT NULL DEFAULT 'pink_heart',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS bookmarks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id INTEGER NOT NULL,
            category_id INTEGER NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(message_id, category_id),
            FOREIGN KEY(message_id) REFERENCES messages(id),
            FOREIGN KEY(category_id) REFERENCES bookmark_categories(id)
        )
        """
    )
    conn.commit()


def rebuild_fts(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute("INSERT INTO messages_fts(messages_fts) VALUES('rebuild');")
    conn.commit()


def clear_existing_data(conn: sqlite3.Connection):
    cur = conn.cursor()
    # Bookmarks reference message IDs; they become invalid after a rebuild.
    cur.execute("DELETE FROM bookmarks;")
    cur.execute("DELETE FROM attachments;")
    cur.execute("DELETE FROM messages;")
    cur.execute("DELETE FROM messages_fts;")
    # If signal scores exist from a previous archive, clear them too.
    try:
        cur.execute("DELETE FROM message_signals;")
    except sqlite3.OperationalError:
        pass
    cur.execute("DROP TABLE IF EXISTS faiss_map;")
    cur.execute("DELETE FROM sqlite_sequence WHERE name IN ('messages', 'attachments');")
    conn.commit()


def clean_attachments_dir(attachments_dir: Path):
    attachments_dir.mkdir(parents=True, exist_ok=True)
    for p in attachments_dir.iterdir():
        if p.is_file():
            p.unlink()


def local_name(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def ts_from_raw(raw_value: str) -> tuple[str, int]:
    try:
        value = int(raw_value or "0")
    except ValueError:
        value = 0

    # SMS Backup & Restore commonly stores ms for sms and sec for mms.
    ts_unix = value // 1000 if value > 10**12 else value
    if ts_unix < 0:
        ts_unix = 0

    if ts_unix == 0:
        return "", 0

    ts = pd.to_datetime(ts_unix, unit="s", errors="coerce")
    if pd.isna(ts):
        return "", 0
    return ts.strftime("%Y-%m-%d %H:%M:%S"), int(ts_unix)


def safe_filename(name: str) -> str:
    n = (name or "").strip()
    if not n or n.lower() in {"null", "none", "(null)", "undefined"}:
        return "attachment"
    n = re.sub(r"[^A-Za-z0-9._-]+", "_", n)
    return n.strip("._") or "attachment"


def first_valid_name(*values: str) -> str:
    for value in values:
        v = (value or "").strip()
        if not v:
            continue
        if v.lower() in {"null", "none", "(null)", "undefined"}:
            continue
        return v
    return "attachment"


def infer_sms_direction(msg_type: str, owner_name: str, partner_name: str) -> str:
    # 2 = sent, 1 = inbox/received
    return owner_name if str(msg_type) == "2" else partner_name


def infer_mms_direction(msg_box: str, owner_name: str, partner_name: str) -> str:
    # 2 = sent, 1 = inbox/received
    return owner_name if str(msg_box) == "2" else partner_name


def parse_messages_from_csv(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV file: {csv_path}")

    header_row, delimiter = _detect_csv_header_row(csv_path)
    df = pd.read_csv(csv_path, header=header_row, sep=delimiter)
    norm_to_actual = {str(c).strip().lower(): c for c in df.columns}

    def _pick(candidates: list[str]) -> str | None:
        for c in candidates:
            actual = norm_to_actual.get(c)
            if actual is not None:
                return actual
        return None

    body_col = _pick(CSV_BODY_COLUMN_CANDIDATES)
    direction_col = _pick(CSV_DIRECTION_COLUMN_CANDIDATES)
    date_col = _pick(CSV_DATE_COLUMN_CANDIDATES)

    if body_col is None:
        opts = ", ".join(CSV_BODY_COLUMN_CANDIDATES)
        raise ValueError(f"CSV must include a message body column. Accepted names: {opts}.")

    if date_col is not None:
        dates = pd.to_datetime(df[date_col], errors="coerce")
    else:
        dates = pd.to_datetime(pd.Series([""] * len(df)), errors="coerce")
    ts = dates.dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")
    # Use timestamp() per row to avoid unit mismatches across pandas datetime resolutions
    # (e.g. datetime64[us] vs datetime64[ns]) that can shift dates toward 1970.
    ts_unix = dates.apply(lambda d: int(d.timestamp()) if pd.notna(d) else 0).astype(int)

    body = df[body_col].fillna("").astype(str).str.strip()
    if direction_col is not None:
        direction = df[direction_col].fillna("").astype(str)
    else:
        direction = pd.Series([""] * len(df))

    rows = []
    for i in range(len(df)):
        msg_body = body.iloc[i]
        if not msg_body:
            continue
        rows.append(
            {
                "ts": str(ts.iloc[i]),
                "ts_unix": int(ts_unix.iloc[i]),
                "direction": str(direction.iloc[i]),
                "body": msg_body,
                "attachments": [],
            }
        )
    return rows


def iter_messages_from_xml(xml_path: Path, owner_name: str, partner_name: str):
    if not xml_path.exists():
        raise FileNotFoundError(f"Missing XML file: {xml_path}")

    for _, elem in ET.iterparse(str(xml_path), events=("end",)):
        tag = local_name(elem.tag)

        if tag == "sms":
            ts, ts_unix = ts_from_raw(elem.attrib.get("date", "0"))
            body = (elem.attrib.get("body") or "").strip()
            direction = infer_sms_direction(elem.attrib.get("type", ""), owner_name, partner_name)
            if body:
                yield {
                    "ts": ts,
                    "ts_unix": ts_unix,
                    "direction": direction,
                    "body": body,
                    "attachments": [],
                }
            elem.clear()
            continue

        if tag == "mms":
            ts, ts_unix = ts_from_raw(elem.attrib.get("date", "0"))
            direction = infer_mms_direction(elem.attrib.get("msg_box", ""), owner_name, partner_name)

            text_parts: list[str] = []
            attachments: list[dict] = []

            for part in elem.iter():
                if local_name(part.tag) != "part":
                    continue

                mime_type = (part.attrib.get("ct") or "").strip()
                mime_lower = mime_type.lower()

                if mime_lower.startswith("text/"):
                    txt = (part.attrib.get("text") or part.attrib.get("body") or "").strip()
                    if txt:
                        text_parts.append(txt)
                    continue

                if not mime_lower:
                    continue

                raw_data = part.attrib.get("data")
                if not raw_data:
                    continue

                try:
                    payload = base64.b64decode(raw_data)
                except Exception:
                    continue

                if not payload:
                    continue

                original_name = first_valid_name(
                    part.attrib.get("cl"),
                    part.attrib.get("name"),
                    part.attrib.get("fn"),
                )
                attachments.append(
                    {
                        "kind": "image" if mime_lower.startswith("image/") else "file",
                        "mime_type": mime_type,
                        "filename": original_name,
                        "bytes": payload,
                    }
                )

            body = " ".join([p for p in text_parts if p]).strip()
            if not body and attachments:
                body = "[Photo]" if all((a["mime_type"] or "").lower().startswith("image/") for a in attachments) else "[Attachment]"

            if body or attachments:
                yield {
                    "ts": ts,
                    "ts_unix": ts_unix,
                    "direction": direction,
                    "body": body,
                    "attachments": attachments,
                }

            elem.clear()


def insert_messages_and_attachments(conn: sqlite3.Connection, rows, attachments_dir: Path) -> int:
    cur = conn.cursor()
    inserted = 0

    for msg in rows:
        cur.execute(
            "INSERT INTO messages(ts, ts_unix, direction, body) VALUES(?,?,?,?)",
            (
                msg.get("ts", ""),
                int(msg.get("ts_unix", 0)),
                msg.get("direction", ""),
                msg.get("body", ""),
            ),
        )
        message_id = int(cur.lastrowid)
        inserted += 1

        for idx, att in enumerate(msg.get("attachments", []), start=1):
            mime_type = (att.get("mime_type") or "application/octet-stream").strip()
            base_name = safe_filename(att.get("filename") or "attachment")
            ext = Path(base_name).suffix
            if not ext:
                ext = mimetypes.guess_extension(mime_type) or ""
            if not ext:
                ext = ".bin"

            stored_name = f"{message_id}_{idx}_{safe_filename(Path(base_name).stem)}{ext}"
            stored_path = attachments_dir / stored_name
            with open(stored_path, "wb") as f:
                f.write(att.get("bytes") or b"")

            cur.execute(
                """
                INSERT INTO attachments(message_id, kind, mime_type, filename, stored_path)
                VALUES(?,?,?,?,?)
                """,
                (
                    message_id,
                    att.get("kind") or "file",
                    mime_type,
                    att.get("filename") or stored_name,
                    stored_name,
                ),
            )

    conn.commit()
    return inserted


def build_faiss(conn: sqlite3.Connection, model_name: str, faiss_path: Path, device: str = "auto"):
    cur = conn.cursor()
    cur.execute("SELECT id, body FROM messages ORDER BY id;")
    items = cur.fetchall()
    ids = [x[0] for x in items]
    texts = [x[1] for x in items]

    if not texts:
        raise ValueError("No messages found after ingest.")

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda but torch.cuda.is_available() is False.")
    model_device = "cuda" if (device == "cuda" or (device == "auto" and torch.cuda.is_available())) else "cpu"

    print(f"Loaded {len(texts)} messages into DB.")
    print("Loading embedding model:", model_name)
    print("Embedding device:", model_device)
    model = SentenceTransformer(model_name, device=model_device)

    embeddings = []
    batch_size = 256
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i : i + batch_size]
        emb_batch = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        embeddings.append(emb_batch)

    emb = np.vstack(embeddings).astype("float32")
    dim = emb.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(emb)
    faiss.write_index(index, str(faiss_path))

    cur.execute("CREATE TABLE faiss_map (pos INTEGER PRIMARY KEY, message_id INTEGER NOT NULL);")
    cur.executemany("INSERT INTO faiss_map(pos, message_id) VALUES(?,?)", list(enumerate(ids)))
    conn.commit()


def main():
    parser = argparse.ArgumentParser(description="Build Text Archive DB and FAISS index")
    parser.add_argument("--xml", default=INPUT_XML, help="Path to SMS XML export")
    parser.add_argument("--csv", default=INPUT_CSV, help="Path to CSV export")
    parser.add_argument("--db", default=DB_PATH, help="SQLite DB path")
    parser.add_argument("--faiss", default=FAISS_INDEX_PATH, help="FAISS index path")
    parser.add_argument("--attachments-dir", default=ATTACHMENTS_DIR, help="Attachment output directory")
    parser.add_argument("--owner-name", default=DEFAULT_OWNER_NAME, help="Owner/sender label for sent messages")
    parser.add_argument("--partner-name", default=DEFAULT_PARTNER_NAME, help="Partner label for received messages")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Embedding device: auto (default), cpu, or cuda",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    attachments_dir = Path(args.attachments_dir)

    xml_path = Path(args.xml)
    csv_path = Path(args.csv)

    if xml_path.exists():
        print(f"Ingesting from XML: {xml_path}")
        rows = iter_messages_from_xml(xml_path, args.owner_name, args.partner_name)
    elif csv_path.exists():
        print(f"XML not found. Falling back to CSV: {csv_path}")
        rows = parse_messages_from_csv(csv_path)
    else:
        raise FileNotFoundError(f"Neither XML ({xml_path}) nor CSV ({csv_path}) was found.")

    conn = sqlite3.connect(str(db_path))
    ensure_schema(conn)
    clear_existing_data(conn)
    clean_attachments_dir(attachments_dir)
    inserted = insert_messages_and_attachments(conn, rows, attachments_dir)
    if inserted == 0:
        conn.close()
        raise ValueError("No messages parsed from input source.")
    rebuild_fts(conn)
    build_faiss(conn, MODEL_NAME, Path(args.faiss), device=args.device)
    conn.close()

    print(f"Saved DB: {db_path}")
    print(f"Saved FAISS index: {args.faiss}")
    print(f"Saved attachments under: {attachments_dir.resolve()}")
    print("Done.")


if __name__ == "__main__":
    main()


