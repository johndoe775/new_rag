# app.py
from typing import List, Annotated, Literal, TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END, goto
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate

# Local modules
from research.helpers import LLM
from research.tools import pandas_tool, sql_tool, rag_tool


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
    messages: Annotated[List[BaseMessage], add_messages]
    inputs: str
    choice: Choice


# -------------------------
# Graph builder
# -------------------------
def create_graph():
    graph = StateGraph(GraphState)

    # Input node: decides which tool to use
    def inputs_node(state: GraphState):
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

        state["choice"] = response.choice

        # Route to the correct tool using goto
        return goto(response.choice, state)

    # Tool nodes
    pandas_tool_node = ToolNode([pandas_tool.pandas_tool])
    sql_tool_node = ToolNode([sql_tool.pandasql_tool])
    rag_tool_node = ToolNode([rag_tool.rag_tool])

    # Register nodes
    graph.add_node("input", inputs_node)
    graph.add_node("pandas", pandas_tool_node)
    graph.add_node("sql", sql_tool_node)
    graph.add_node("rag", rag_tool_node)

    # Wire edges
    graph.add_edge(START, "input")
    graph.add_edge("pandas", END)
    graph.add_edge("sql", END)
    graph.add_edge("rag", END)

    return graph.compile()


# -------------------------
# CLI entrypoint
# -------------------------
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Process user input for data analysis."
    )
    parser.add_argument("input_text", type=str, help="The input text for analysis.")
    args = parser.parse_args()

    app = create_graph()

    initial_state = {
        "messages": [
            HumanMessage(content=args.input_text),
            AIMessage(
                content="Sure, I can help with that. Let me call the appropriate tool."
            ),
        ],
        "inputs": args.input_text,
    }

    result = app.invoke(initial_state)

    # Optional: Export graph visualization
    app.get_graph().draw_mermaid_png(output_file_path="graph.png")
    print("Saved graph to graph.png")

    print(result)


if __name__ == "__main__":
    main()
