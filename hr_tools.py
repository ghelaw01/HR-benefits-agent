"""HR tools: plain Python functions the agent can use."""
import json
import re
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


# FIX 1a: accept "CLM-10023", "clm-10789", "claim 10456", "CLM 10023", "10456"
CLAIM_ID_PATTERN = re.compile(r"(?:clm|claim)?[\s#-]*(\d{5})", re.IGNORECASE)


def normalize_claim_id(raw: str) -> str | None:
    """Turn any common claim ID format into 'CLM-12345'. Returns None if no ID."""
    match = CLAIM_ID_PATTERN.search(raw)
    return f"CLM-{match.group(1)}" if match else None


def get_claim_status(claim_id: str) -> dict:
    """Look up one claim. Returns an 'error' key instead of crashing."""
    normalized = normalize_claim_id(claim_id)
    if normalized is None:
        return {"error": f"'{claim_id}' is not a valid claim ID. Use the format CLM-12345."}

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT claim_id, claim_type, status, filed_date FROM claims WHERE claim_id = ?",
        (normalized,),
    ).fetchone()
    conn.close()

    # FIX 1b: never crash and never guess. Say clearly that it was not found.
    if row is None:
        return {"error": f"Claim {normalized} was not found."}
    return dict(row)
