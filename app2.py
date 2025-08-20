# app.py
from typing import List, Annotated, Literal, TypedDict
from pydantic import BaseModel, Field
from langgraph.types import Command
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate

# Your local modules (unchanged)
from research.helpers import LLM
from research.tools import pandas_tool, sql_tool, rag_tool

tools = [pandas_tool.pandas_tool, sql_tool.pandasql_tool, rag_tool.rag_tool]
llm = LLM().llm  # No binding of tools here


# -------------------------
# Graph state definition
# -------------------------
class GraphState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    inputs: str
    choice: str  # Directly store the tool choice as a string


# -------------------------
# Graph builder
# -------------------------
def create_graph():
    graph = StateGraph(GraphState)

    # 1) Input node: decide which tool to use by calling the LLM
    def inputs_node(state: GraphState) -> GraphState:
        state["inputs"] = input("Enter your query: ")
        user_input = state["inputs"]

        prompt = PromptTemplate(
            template="""You are a router that decides which tool to use for a user request.

User input:
{user_input}

Return one of:
- "pandas" for DataFrame wrangling/visualizations
- "sql" for queries that should be answered with SQL (incl. pandasql)
- "rag" for retrieval-augmented generation (document/knowledge lookup)

Return ONLY the literal label.""",
            inputs=["user_input"],
        )
        prompt = prompt.format(user_input=user_input)
        response = llm.invoke(prompt)

        # Directly store the choice as a string
        state["choice"] = response.content  # Assuming response is a string

        return state

    def decider(
        state: GraphState,
    ) -> Command[Literal["rag_tool", "pandas_tool", "sql_tool"]]:

        value = state["choice"]
        if value == "sql":
            goto = "sql_tool"
        elif value == "pandas":
            goto = "pandas_tool"
        else:
            goto = "rag_tool"

        return Command(
            goto=goto,
        )

    # 2) Tool nodes (wrap your existing tools)
    pandas_tool_node = ToolNode([pandas_tool.pandas_tool])
    sql_tool_node = ToolNode([sql_tool.pandasql_tool])
    rag_tool_node = ToolNode([rag_tool.rag_tool])

    # 3) Register nodes
    graph.add_node("input", inputs_node)
    graph.add_node("decider", decider)
    graph.add_node("pandas_tool", pandas_tool_node)
    graph.add_node("sql_tool", sql_tool_node)
    graph.add_node("rag_tool", rag_tool_node)

    # 4) Wire edges
    graph.add_edge(START, "input")
    graph.add_edge("input", "decider")
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

    # Export the compiled graph to a PNG file
    png_bytes = app.get_graph().draw_mermaid_png(output_file_path="graph.png")
    print("Saved graph to graph.png")

    print(result)


if __name__ == "__main__":
    main()
