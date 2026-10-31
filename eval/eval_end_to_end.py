"""End-to-end evaluation: run the full agent on every case, then find root causes.

Needs your OpenAI API key in .env.
Run from the project folder:  python eval/eval_end_to_end.py
"""
import asyncio
import json
import sys
from pathlib import Path

import pandas as pd
from mcp import Client

sys.path.append(str(Path(__file__).parent.parent))  # so we can import agent
from agent import SERVER, run_agent, to_openai_tools

EVAL_DIR = Path(__file__).parent
cases = json.loads((EVAL_DIR / "eval_set.json").read_text())

ABSTAIN_PHRASES = ["don't know", "do not know", "not sure", "can't help",
                   "cannot help", "not able", "unable", "contact hr"]


def check_case(case, result):
    """Grade one case. Returns the checks and the root cause of any failure."""
    answer = (result["answer"] or "").lower()
    tools_used = [step["tool"] for step in result["trace"]]
    any_error = any(step["is_error"] for step in result["trace"])

    # Check 1: did the agent pick the right tool?
    expected_tool = case.get("expected_tool")
    tool_ok = expected_tool is None or expected_tool in tools_used

    # Check 2: did the search return the right policy?
    retrieval_ok = True
    if case.get("expected_policy"):
        search_outputs = [s["output"] for s in result["trace"] if s["tool"] == "search_policies"]
        retrieval_ok = any(f'"{case["expected_policy"]}"' in out for out in search_outputs)

    # Check 3: is the final answer right?
    if case["category"] == "out_of_scope":
        answer_ok = any(phrase in answer for phrase in ABSTAIN_PHRASES)
    else:
        answer_ok = all(word.lower() in answer for word in case["must_contain"])

    # Root cause: the FIRST step that went wrong
    if answer_ok:
        root_cause = ""
    elif not tool_ok:
        root_cause = "tool selection"
    elif any_error:
        root_cause = "tool error"
    elif not retrieval_ok:
        root_cause = "retrieval"
    else:
        root_cause = "answer generation"

    return {
        "tool_ok": tool_ok,
        "tool_error": any_error,
        "retrieval_ok": retrieval_ok,
        "passed": answer_ok,
        "root_cause": root_cause,
    }


async def main():
    rows = []
    async with Client(SERVER) as mcp_client:
        tools = to_openai_tools((await mcp_client.list_tools()).tools)
        for case in cases:
            result = await run_agent(case["question"], mcp_client, tools)
            checks = check_case(case, result)
            rows.append({
                "id": case["id"],
                "category": case["category"],
                "question": case["question"],
                "answer": result["answer"],
                "tools": [s["tool"] for s in result["trace"]],
                **checks,
            })
            print(f"{case['id']:>2}  {'PASS' if checks['passed'] else 'FAIL'}  {case['question']}")

    df = pd.DataFrame(rows)
    df.to_csv(EVAL_DIR / "results.csv", index=False)

    print(f"\nEnd-to-end accuracy: {df['passed'].mean():.0%} ({df['passed'].sum()}/{len(df)})")
    print("\nAccuracy by category:")
    print(df.groupby("category")["passed"].mean().map("{:.0%}".format))
    print("\nFailures by root cause:")
    print(df.loc[~df["passed"], "root_cause"].value_counts().to_string())
    print("\nFailed cases:")
    failed = df.loc[~df["passed"], ["id", "question", "root_cause", "answer"]].copy()
    failed["answer"] = failed["answer"].str[:60]
    print(failed.to_string(index=False))
    print("\nFull results saved to eval/results.csv")


if __name__ == "__main__":
    asyncio.run(main())
