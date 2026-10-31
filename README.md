# HR Benefits Agent: build it, then evaluate it

An AI agent that answers HR policy questions and checks claim status.
It uses OpenAI tool calling and an MCP server, then gets a full evaluation.

## Setup (once)
1. python -m venv .venv
2. Activate it. Windows: .venv\Scripts\activate   Mac/Linux: source .venv/bin/activate
3. pip install -r requirements.txt
4. Copy .env.example to .env and add your OpenAI API key
5. python setup_db.py

## Run
- Chat with the agent:        python agent.py
- Component evaluation:       python eval/eval_components.py   (no API key needed)
- End-to-end evaluation:      python eval/eval_end_to_end.py   (uses your API key)
- Regression tests:           python -m pytest -v              (no API key needed)

## Version 2
The files in solutions/ contain the fixes found by evaluation.
Copy them over the files with the same names to upgrade.
