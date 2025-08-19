# app.py
from typing import List, Annotated, Literal, TypedDict
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate

# Your local modules (unchanged)
from research.helpers import LLM
from research.tools import pandas_tool, sql_tool, rag_tool


# If you already have a GraphState type in research.tools.state and want to reuse it,
# you can import it instead of using the TypedDict below.
@tool

# -------------------------
# Structured output schema
# -------------------------
class Choice(BaseModel):
    choice: Literal["pandas", "sql", "rag"] = Field(
        description="Return exactly one: 'pandas', 'sql', or 'rag'."
    )


# -------------------------
# Graph state definition
# -------------------------
class GraphState(TypedDict, total=False):
    # Keep messages in case you plan to extend the graph with conversational context.
    messages: Annotated[List[BaseMessage], add_messages]
    inputs: str
    choice: Choice  # set by the inputs_node after calling LLM


# -------------------------
# Graph builder
# -------------------------
def create_graph():
    graph = StateGraph(GraphState)

    # 1) Input node: decide which tool to use by calling the LLM with a structured schema
    def inputs_node(state: GraphState) -> GraphState:
        state["inputs"] = input("Enter your query: ")
        user_input = state["inputs"]

        prompt = PromptTemplate.from_template(
            """You are a router that decides which tool to use for a user request.

User input:
{user_input}

Return one of:
- "pandas" for DataFrame wrangling/visualizations
- "sql" for queries that should be answered with SQL (incl. pandasql)
- "rag" for retrieval-augmented generation (document/knowledge lookup)

Return ONLY the literal label."""
        )

        structured_llm = LLM().llm.with_structured_output(Choice)
        chain = prompt | structured_llm

        response = chain.invoke({"user_input": user_input})

        # Normalize to Choice whether the LLM returns a dict or a Choice instance
        state["choice"] = response.choice

        return state

    # 2) Tool nodes (wrap your existing tools)
    pandas_tool_node = ToolNode([pandas_tool.pandas_tool])
    sql_tool_node = ToolNode([sql_tool.pandasql_tool])
    rag_tool_node = ToolNode([rag_tool.rag_tool])

    # 3) Register nodes
    graph.add_node("input", inputs_node)
    graph.add_node("pandas_tool", pandas_tool_node)
    graph.add_node("sql_tool", sql_tool_node)
    graph.add_node("rag_tool", rag_tool_node)

    # 4) Wire edges
    graph.add_edge(START, "input")

    # Use a lambda for routing directly from the "input" node

    graph.add_conditional_edges(
        "input",
        lambda state: (
            state["choice"] if state["choice"] in {"pandas", "sql", "rag"} else "rag"
        ),
        {
            "pandas": "pandas_tool",
            "sql": "sql_tool",
            "rag": "rag_tool",
        },
    )

    # Exit after each tool
    graph.add_edge("pandas_tool", END)
    graph.add_edge("sql_tool", END)
    graph.add_edge("rag_tool", END)

    return graph.compile()


# -------------------------
# CLI entrypoint
# -------------------------
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Process user input for data analysis."
    )
    parser.add_argument(
        "input_text",
        type=str,
        help="The input text for analysis (e.g., 'analyze this data using pandas').",
    )
    args = parser.parse_args()

    app = create_graph()

    # Initial state includes your messages and the raw input
    initial_state = {
        "messages": [
            HumanMessage(content=args.input_text),
            AIMessage(
                content="Sure, I can help with that. Let me call the appropriate tool."
            ),
        ],
        "inputs": args.input_text,
    }

    # Run the graph
    result = app.invoke(initial_state)
    # Run your graph (you already have this)

    # Export the compiled graph to a PNG file
    png_bytes = app.get_graph().draw_mermaid_png(output_file_path="graph.png")
    print("Saved graph to graph.png")

    print(result)


if __name__ == "__main__":
    main()
