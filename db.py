import sqlite3
from pathlib import Path

DB_PATH = Path("recoverai.db")

def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    conn = get_conn()
    cur = conn.cursor()

    # One table for ALL roles (admin/clinician/patient)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        role TEXT NOT NULL CHECK(role IN ('admin','clinician','patient')),
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE,
        password TEXT NOT NULL,
        first_name TEXT,
        last_name TEXT,
        birthdate TEXT,
        phone TEXT,
        gender TEXT,
        created_at TEXT DEFAULT (datetime('now'))
    )
    """)

    conn.commit()
    conn.close()

def create_user(role, username, email, password, first_name=None, last_name=None,
                birthdate=None, phone=None, gender=None):
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
        INSERT INTO users (role, username, email, password, first_name, last_name, birthdate, phone, gender)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (role, username, email, password, first_name, last_name, birthdate, phone, gender))
        conn.commit()
        return True, "Account created ✅"
    except sqlite3.IntegrityError as e:
        msg = str(e).lower()
        if "users.username" in msg:
            return False, "Username already exists ❌"
        if "users.email" in msg:
            return False, "Email already exists ❌"
        return False, "Could not create account ❌"
    finally:
        conn.close()

def get_user_for_login(role, identifier):
    """
    identifier can be username OR email
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
    SELECT id, role, username, email, password, first_name, last_name
    FROM users
    WHERE role = ?
      AND (username = ? OR email = ?)
    """, (role, identifier, identifier))
    row = cur.fetchone()
    conn.close()
    return row
