# app.py
from typing import List, Annotated, TypedDict
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool

# Your local modules
from research.helpers import LLM
from research.tools import pandas_tool, sql_tool, rag_tool


# -------------------------
# Graph state
# -------------------------
class GraphState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    inputs: str


# -------------------------
# Wrap existing functions as LangChain Tools
# (names can be anything, but they must match what the LLM may call)
# -------------------------
@tool
def pandas_exec(query: str) -> str:
    """Run a pandas task on in-memory data as described by `query`."""

    return str(pandas_tool.pandas_tool(query))


@tool
def sql_exec(query: str) -> str:
    """Run a SQL/pandasql task described by `query`."""
    return str(sql_tool.pandasql_tool(query))


@tool
def rag_exec(query: str) -> str:
    """Run a retrieval-augmented lookup to answer `query`."""
    return str(rag_tool.rag_tool(query))


TOOLS = [pandas_exec, sql_exec, rag_exec]


# -------------------------
# Build the graph
# -------------------------
def create_graph():
    builder = StateGraph(GraphState)

    # 1) Agent node: call model *bound to tools* so it can emit tool_calls
    agent_llm = LLM().llm.bind_tools(TOOLS)

    def agent_node(state: GraphState) -> GraphState:
        msgs = state.get("messages", [])
        # Ensure the latest user input is present as a HumanMessage
        if not msgs or not isinstance(msgs[-1], HumanMessage):
            msgs = msgs + [HumanMessage(content=state["inputs"])]

        ai = agent_llm.invoke(msgs)  # may include tool_calls
        return {"messages": [ai]}

    # 2) Tools node: executes tool_calls found in the last AI message
    tools_node = ToolNode(TOOLS)

    # 3) Register nodes
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tools_node)

    # 4) Wire edges
    builder.add_edge(START, "agent")

    # If the LLM requested a tool, go to ToolNode; else, END.
    # path_map keeps the diagram clean and avoids stray auto-edges in some versions.
    builder.add_conditional_edges("agent", tools_condition, path_map=["tools", END])

    # After running a tool, go back to the agent to continue or finish
    builder.add_edge("tools", "agent")

    return builder.compile()


# -------------------------
# CLI entrypoint
# -------------------------
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Process user input for data analysis."
    )
    parser.add_argument(
        "input_text", type=str, help="User request (e.g., 'summarize sales by region')."
    )
    args = parser.parse_args()

    app = create_graph()

    # Seed initial state. We add the user message up front; the agent node will use it.
    initial_state = {
        "inputs": args.input_text,
        "messages": [HumanMessage(content=args.input_text)],
    }

    # Run the graph
    result = app.invoke(initial_state)

    # Save the diagram (optional)
    app.get_graph().draw_mermaid_png(output_file_path="graph.png")
    print("Saved graph to graph.png")

    # Print final assistant response, if present
    final_msgs = result.get("messages", [])
    if final_msgs:
        print("\nAssistant:\n", final_msgs[-1].content)
    else:
        print("\nNo messages returned.")


if __name__ == "__main__":
    main()
