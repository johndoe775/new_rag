from fastapi import FastAPI, Request
from pydantic import BaseModel

from research.tools.pandas_tool import pandas_tool
from research.tools.rag_tool import rag_tool
from research.tools.sql_tool import pandasql_tool
from research.tools.state import GraphState
from research.tools.helpers import LLM

app = FastAPI()

class ToolRequest(BaseModel):
    user_input: str
    state: dict


# Dummy LLM-based tool selector (replace with your LLM logic)

def llm_select_tool(user_input: str) -> str:
    """
    Use LLM from helpers.py to select the tool based on user input.
    """
    llm = LLM().llm
    prompt = (
        "You are an intelligent tool selector. Based on the user's request, choose the best tool to use. "
        "Available tools: pandas_tool (for visualization), rag_tool (for PDF/document QA), pandasql_tool (for SQL analysis). "
        "Output ONLY the tool name: pandas_tool, rag_tool, or pandasql_tool.\n\n"
        f"User request: {user_input}"
    )
    response = llm.invoke(prompt)
    tool_name = response.content.strip().split()[0]  # get first word, tool name
    if tool_name not in ["pandas_tool", "rag_tool", "pandasql_tool"]:
        tool_name = "rag_tool"  # fallback
    return tool_name

@app.post("/run_tool")
async def run_tool(request: ToolRequest):
    state = GraphState(**request.state)
    tool_name = llm_select_tool(request.user_input)
    if tool_name == "pandas_tool":
        result = pandas_tool(state)
    elif tool_name == "rag_tool":
        result = rag_tool(state)
    elif tool_name == "pandasql_tool":
        result = pandasql_tool(state)
    else:
        return {"error": "Unknown tool selected"}
    return {"tool": tool_name, "state": result}

@app.get("/")
async def root():
    return {"message": "MCP Tool Server is running"}
