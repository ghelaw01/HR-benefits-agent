"""The HR agent: an LLM that uses the MCP tools to answer questions.

Run:  python agent.py
"""
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from mcp import Client, StdioServerParameters
from openai import OpenAI

load_dotenv()  # reads OPENAI_API_KEY (and OPENAI_MODEL) from the .env file

llm = OpenAI()
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_STEPS = 5

SERVER = StdioServerParameters(
    command=sys.executable,
    args=[str(Path(__file__).parent / "mcp_server.py")],
)

SYSTEM_PROMPT = """You are an HR benefits assistant for employees.
Use the tools to answer questions. Keep answers short and clear.
Rules:
- Answer only from the tool results. Never use outside knowledge.
- If the tools return no results, an error, or nothing relevant, say:
  "I don't know. Please contact HR."
- Never guess a claim status. If a claim is not found, say so."""


def to_openai_tools(mcp_tools):
    """Convert MCP tool descriptions into the format OpenAI expects."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            },
        }
        for tool in mcp_tools
    ]


async def run_agent(question, mcp_client, tools):
    """Answer one question. Returns the answer and a trace of every tool call."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    trace = []

    for step in range(MAX_STEPS):
        response = llm.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
        )
        message = response.choices[0].message

        # No tool calls means the model has written its final answer.
        if not message.tool_calls:
            return {"answer": message.content, "trace": trace}

        messages.append(message)

        for call in message.tool_calls:
            name = call.function.name
            args = json.loads(call.function.arguments)

            result = await mcp_client.call_tool(name, args)
            output = "\n".join(c.text for c in result.content if c.type == "text")

            trace.append({
                "tool": name,
                "args": args,
                "output": output,
                "is_error": result.is_error,
            })
            messages.append({"role": "tool", "tool_call_id": call.id, "content": output})

    return {"answer": "Sorry, I could not finish this request.", "trace": trace}


async def main():
    async with Client(SERVER) as mcp_client:
        tools = to_openai_tools((await mcp_client.list_tools()).tools)
        print("Connected. Tools:", [t["function"]["name"] for t in tools])

        while True:
            question = input("\nYou: ").strip()
            if question.lower() in {"", "quit", "exit"}:
                break
            result = await run_agent(question, mcp_client, tools)
            for step in result["trace"]:
                print(f"  [tool] {step['tool']}({step['args']})")
            print("Agent:", result["answer"])


if __name__ == "__main__":
    asyncio.run(main())
