"""Component evaluation: test each tool on its own. No API key needed.

Run from the project folder:  python eval/eval_components.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))  # so we can import hr_tools
import hr_tools

cases = json.loads((Path(__file__).parent / "eval_set.json").read_text())

# ---------------------------------------------------------------
# 1. Retrieval: does search_policies rank the right policy first?
# ---------------------------------------------------------------
rows = []
for case in cases:
    if case["category"] != "policy":
        continue
    results = hr_tools.search_policies(case["question"], top_k=3)
    ids = [r["id"] for r in results]
    expected = case["expected_policy"]
    rank = ids.index(expected) + 1 if expected in ids else None
    rows.append({
        "id": case["id"],
        "question": case["question"],
        "expected": expected,
        "got_top3": ids,
        "hit@1": rank == 1,
        "hit@3": rank is not None,
        "reciprocal_rank": 1 / rank if rank else 0.0,
    })

retrieval = pd.DataFrame(rows)
print("=== Retrieval (search_policies) ===")
print(f"Recall@1: {retrieval['hit@1'].mean():.0%}")
print(f"Recall@3: {retrieval['hit@3'].mean():.0%}")
print(f"MRR:      {retrieval['reciprocal_rank'].mean():.2f}")
misses = retrieval[~retrieval["hit@1"]]
print("\nNot ranked first:")
print(misses[["id", "question", "expected", "got_top3"]].to_string(index=False))

# ---------------------------------------------------------------
# 2. Claim tool: does get_claim_status handle real-world inputs?
# ---------------------------------------------------------------
rows = []
for case in cases:
    if case["category"] != "claim":
        continue
    try:
        result = hr_tools.get_claim_status(case["claim_input"])
        if case["expected_status"] == "not found":
            correct = "error" in result
        else:
            correct = result.get("status") == case["expected_status"]
        outcome = json.dumps(result)
    except Exception as error:
        correct = False
        outcome = f"CRASH: {type(error).__name__}: {error}"
    rows.append({"id": case["id"], "input": case["claim_input"],
                 "expected": case["expected_status"], "correct": correct,
                 "outcome": outcome[:70]})

claims = pd.DataFrame(rows)
print("\n=== Claim tool (get_claim_status) ===")
print(f"Accuracy: {claims['correct'].mean():.0%}")
print(claims.to_string(index=False))

# ---------------------------------------------------------------
# 3. Out of scope: are top search scores lower than for real questions?
# ---------------------------------------------------------------
rows = []
for case in cases:
    if case["category"] in ("policy", "out_of_scope"):
        top = hr_tools.search_policies(case["question"], top_k=1)
        top_score = top[0]["score"] if top else 0.0
        rows.append({"category": case["category"], "question": case["question"],
                     "top_score": top_score})

scope = pd.DataFrame(rows)
print("\n=== Search scores: real questions vs out of scope ===")
print(scope.groupby("category")["top_score"].agg(["min", "mean", "max"]).round(3))
matched = scope[(scope["category"] == "out_of_scope") & (scope["top_score"] > 0)]
print(f"\nOut-of-scope questions that matched a policy (score > 0): {len(matched)} of "
      f"{(scope['category'] == 'out_of_scope').sum()}")
