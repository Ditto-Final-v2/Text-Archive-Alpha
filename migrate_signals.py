import sqlite3

DB_PATH = "texts.db"


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


def main():
    conn = sqlite3.connect(DB_PATH)
    ensure_signals_schema(conn)
    conn.close()
    print("Signals schema migration complete.")


if __name__ == "__main__":
    main()
