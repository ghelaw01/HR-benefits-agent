"""MCP server that shares the HR tools with any MCP-ready app.

Run it alone to test:   mcp dev mcp_server.py
The agent starts it automatically, so you don't need to run it yourself.
"""
from mcp.server.mcpserver import MCPServer

import hr_tools

mcp = MCPServer("hr-benefits")


@mcp.tool()
def search_policies(query: str) -> dict:
    """Search the company's HR policies.

    Use this for any question about HR rules: paid time off, sick leave,
    parental leave, remote work, health insurance, dental claims,
    travel reimbursement, or claim appeals.
    Returns the best matching policies, each with a relevance score.
    """
    return {"results": hr_tools.search_policies(query)}


@mcp.tool()
def get_claim_status(claim_id: str) -> dict:
    """Get the status of one benefits claim.

    Use this only when the user gives a claim ID, like 'CLM-10023'.
    Returns the claim type, status, and filed date.
    """
    return hr_tools.get_claim_status(claim_id)


if __name__ == "__main__":
    mcp.run()
