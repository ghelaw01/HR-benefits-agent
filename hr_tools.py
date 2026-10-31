"""HR tools: plain Python functions the agent can use."""
import json
import re
import sqlite3
from pathlib import Path

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "data" / "claims.db"
POLICIES_PATH = BASE_DIR / "data" / "policies.json"

MIN_SCORE = 0.25  # FIX 4: ignore matches weaker than this

# FIX 3: words employees use that the policies don't
SYNONYMS = {
    "vacation": "paid time off pto",
    "maternity": "parental",
    "paternity": "parental",
    "wfh": "remote work",
}

with open(POLICIES_PATH, encoding="utf-8") as f:
    POLICIES = json.load(f)

# FIX 2: reduce words to their stem, so "claims" matches "claim"
# and "traveling" matches "travel".
stemmer = PorterStemmer()
basic_analyzer = TfidfVectorizer(stop_words="english").build_analyzer()


def stemmed_words(text):
    return [stemmer.stem(word) for word in basic_analyzer(text)]


texts = [p["title"] + ". " + p["text"] for p in POLICIES]
vectorizer = TfidfVectorizer(analyzer=stemmed_words)
policy_vectors = vectorizer.fit_transform(texts)


def expand_query(query: str) -> str:
    """Add policy words for any employee words in the query."""
    words = re.findall(r"[a-z]+", query.lower())
    extra = [SYNONYMS[w] for w in words if w in SYNONYMS]
    return " ".join([query] + extra)


def search_policies(query: str, top_k: int = 3) -> list[dict]:
    """Return up to top_k HR policies that match the query well enough."""
    query_vector = vectorizer.transform([expand_query(query)])
    scores = cosine_similarity(query_vector, policy_vectors)[0]
    best = scores.argsort()[::-1][:top_k]

    results = []
    for i in best:
        if scores[i] < MIN_SCORE:  # FIX 4: skip weak matches
            continue
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
