import sqlite3

DB_PATH = "texts.db"

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS bookmark_categories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS bookmarks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id INTEGER NOT NULL,
    category_id INTEGER NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(message_id, category_id),
    FOREIGN KEY(message_id) REFERENCES messages(id),
    FOREIGN KEY(category_id) REFERENCES bookmark_categories(id)
)
""")

conn.commit()
conn.close()

print("✅ Migration complete: bookmark tables created (if they didn't already exist).")
