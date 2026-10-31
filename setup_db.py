"""Create the claims database (data/claims.db). Run once: python setup_db.py"""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "claims.db"

CLAIMS = [
    ("CLM-10023", "Dental", "approved", "2026-09-14"),
    ("CLM-10456", "Medical", "pending", "2026-10-02"),
    ("CLM-10789", "Vision", "denied", "2026-08-21"),
]

conn = sqlite3.connect(DB_PATH)
conn.execute("DROP TABLE IF EXISTS claims")
conn.execute(
    """
    CREATE TABLE claims (
        claim_id   TEXT PRIMARY KEY,
        claim_type TEXT,
        status     TEXT,
        filed_date TEXT
    )
    """
)
conn.executemany("INSERT INTO claims VALUES (?, ?, ?, ?)", CLAIMS)
conn.commit()
conn.close()

print(f"Created {DB_PATH} with {len(CLAIMS)} claims")
