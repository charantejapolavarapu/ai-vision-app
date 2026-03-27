import sqlite3

conn = sqlite3.connect("app.db", check_same_thread=False)
cursor = conn.cursor()

# Prediction history
cursor.execute("""
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    label TEXT,
    confidence REAL
)
""")

# Users
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    password TEXT
)
""")

def insert_prediction(label, confidence):
    cursor.execute("INSERT INTO history (label, confidence) VALUES (?, ?)", (label, confidence))
    conn.commit()

def get_history():
    cursor.execute("SELECT label, confidence FROM history")
    return cursor.fetchall()

def create_user(username, password):
    cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
    conn.commit()

def get_user(username, password):
    cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    return cursor.fetchone()