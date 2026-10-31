"""HR tools: plain Python functions the agent can use.

Version 1. This version has some weaknesses on purpose.
Do not fix them yet. The evaluation in Lessons 4 to 6 will find them.
"""
import json
import sqlite3
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "data" / "claims.db"
POLICIES_PATH = BASE_DIR / "data" / "policies.json"

# Load the policies once, when this file is imported.
with open(POLICIES_PATH, encoding="utf-8") as f:
    POLICIES = json.load(f)

# Build the search index once.
texts = [p["title"] + ". " + p["text"] for p in POLICIES]
vectorizer = TfidfVectorizer(stop_words="english")
policy_vectors = vectorizer.fit_transform(texts)


def search_policies(query: str, top_k: int = 3) -> list[dict]:
    """Return the top_k HR policies that best match the query."""
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, policy_vectors)[0]
    best = scores.argsort()[::-1][:top_k]

    results = []
    for i in best:
        results.append({
            "id": POLICIES[i]["id"],
            "title": POLICIES[i]["title"],
            "text": POLICIES[i]["text"],
            "score": round(float(scores[i]), 3),
        })
    return results


def get_claim_status(claim_id: str) -> dict:
    """Look up one claim in the claims database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT claim_id, claim_type, status, filed_date FROM claims WHERE claim_id = ?",
        (claim_id,),
    ).fetchone()
    conn.close()
    return dict(row)
