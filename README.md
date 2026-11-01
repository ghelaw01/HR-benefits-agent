# HR Benefits Agent

![tests](https://github.com/ghelaw01/hr-agent/actions/workflows/tests.yml/badge.svg)

An AI agent that answers employee HR questions and checks benefits-claim status.
It uses **OpenAI tool calling** with tools served by an **MCP server**, and it was
improved through **evaluation-driven development**: component evaluation, end-to-end
evaluation, root-cause error analysis, and regression tests.

## What it does

- **Policy questions:** "How many sick days do I get?" → searches HR policies
- **Claim status:** "What's going on with claim 10456?" → looks up a claims database
- **Out of scope:** "Who won the Super Bowl?" → says it doesn't know, instead of guessing

## Architecture

```
Employee ──► agent.py ──► OpenAI model (chooses tools)
               │
               └──► mcp_server.py (MCP, stdio) ──► hr_tools.py
                                                   ├── search_policies  (TF-IDF over policies.json)
                                                   └── get_claim_status (SQLite: claims.db)
```

- **Tools are plain, tested Python.** The model only chooses tools and arguments.
- **MCP** makes the same tools usable from any MCP client (for example, Claude Desktop).
- **Every tool call is traced** (tool, arguments, output, errors) for debugging and evaluation.

## Evaluation results

A 21-case evaluation set covers policy questions (with synonyms and word-form
variations), claim lookups (with messy real-world ID formats and an unknown ID),
and out-of-scope questions.

**Component evaluation** (tools tested alone, no LLM):

| Metric | v1 | v2 |
| --- | --- | --- |
| Claim tool accuracy | 33% | 100% |
| Search Recall@1 | 83% | 100% |
| Search MRR | 0.92 | 1.00 |
| Out-of-scope questions matching a policy | 1 of 3 | 0 of 3 |
| Regression tests passing | 3 of 10 | 12 of 12 |

**What error analysis found, and how it was fixed** (one commit each, see history):

| Finding | Root cause | Fix |
| --- | --- | --- |
| Claim tool crashed on 4 of 6 inputs | No ID normalization; `None` row not handled | Regex normalization to `CLM-12345`; clear not-found errors |
| Search missed "dental claim", "daily meal limit" | Word forms: claim/claims, meal/meals, traveling/travel | Porter stemming |
| Stemming broke "vacation" and "maternity" | Regression: shared words gained weight | Query expansion with HR synonyms + regression tests |
| Unrelated questions still returned policies | No minimum relevance score | Drop matches below a score threshold |
| Agent could answer without evidence | Prompt gave no fallback rule | Answer only from tools; fixed "I don't know" fallback |

**End-to-end evaluation** runs the full agent on all 21 cases, grades tool choice,
retrieval, and the final answer, and assigns each failure a root cause:
`python eval/eval_end_to_end.py` (requires an OpenAI API key).

Results with `gpt-4o-mini`, each case run 3 times (63 runs per version), October 2026:

| Category | v1 | v2 |
| --- | --- | --- |
| Policy questions | 36 of 36 | 36 of 36 |
| Claim lookups | 12 of 18 | 18 of 18 |
| Out of scope (after grader fix) | 8 of 9 | 9 of 9 |
| **All runs** | **56 of 63 (89%)** | **63 of 63 (100%)** |

What end-to-end error analysis added beyond the component tests:

- **The model hid tool bugs, inconsistently.** After a tool error, the model sometimes
  retried with a corrected ID (`CLM-10789`), so v1 claim cases passed 1, 2, or 3 times
  out of 3. Flaky passes like these are why the tools are also tested alone.
- **Query rewriting masked the search bug.** The model rewrote "dental claim" questions
  into better search queries, so those cases passed end to end in v1, even though the
  search tool itself ranked the wrong policy first.
- **One real hallucination.** In one v1 run, the agent answered "Who won the Super Bowl
  last year?" from its own outside knowledge, with outdated information. The v2 prompt
  rules removed this: all out-of-scope runs return the fallback answer.
- **The grader was wrong 4 times.** Valid refusals like "I'm only able to assist with
  HR-related questions" were marked as failures. The original grader scored v1 at
  52 of 63 (83%); after adding these refusal phrases, 56 of 63 (89%). A keyword grader
  is brittle, so the next step is an LLM-as-judge validated against human grades.

## Run it

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then add your OpenAI API key
python setup_db.py

python agent.py                    # chat with the agent
python -m pytest -v                # regression tests (no API key)
python eval/eval_components.py     # component evaluation (no API key)
python eval/eval_end_to_end.py     # end-to-end evaluation (uses the API)
```

## Project structure

| File | Purpose |
| --- | --- |
| `hr_tools.py` | The two tools as plain Python functions |
| `mcp_server.py` | MCP server exposing the tools |
| `agent.py` | Tool-calling agent: MCP client + OpenAI, with tracing |
| `setup_db.py`, `data/` | Claims database and HR policies |
| `eval/eval_set.json` | 21 test cases with gold labels for each step |
| `eval/eval_components.py` | Tests each tool alone |
| `eval/eval_end_to_end.py` | Tests the full agent and finds root causes |
| `tests/test_tools.py` | pytest regression tests, run in CI |

## Next steps

- Replace TF-IDF with embeddings and hybrid search; compare Recall@1
- Add an LLM-as-judge for open answers, validated against human grades
- Run each end-to-end case several times to measure pass rates and flaky cases
- Serve the agent behind a FastAPI endpoint

## Tech

Python · OpenAI API · MCP Python SDK · SQLite · scikit-learn · NLTK · pandas · pytest · GitHub Actions
