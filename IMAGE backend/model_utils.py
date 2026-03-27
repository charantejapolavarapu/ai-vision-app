import sqlite3

conn = sqlite3.connect("app.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    label TEXT,
    confidence REAL
)
""")

def insert_prediction(label, confidence):
    cursor.execute("INSERT INTO history (label, confidence) VALUES (?, ?))
    conn.commit()

def get_history():
    cursor.execute("SELECT label, confidence FROM history")
    return cursor.fetchall()
