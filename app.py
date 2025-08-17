from typing import List, Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.prompts import PromptTemplate
from research.helpers import LLM
from research.tools import pandas_tool, sql_tool, rag_tool
import argparse


# Replace with the actual LLM library you're using
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        messages: A list of messages exchanged in the conversation.
        inputs: The current user input.
    """

    messages: Annotated[List, add_messages]
    inputs: str
    choice: str


class Choice(BaseModel):
    choice: Literal["pandas", "sql", "rag"] = Field(
        description="Based on the user input, LLM must return one of the literals."
    )


def inputs_node(state: GraphState) -> str:
    state["inputs"] = input("Enter your query: ")

    prompt_template = f"""You are given a user input: {state["inputs"]}. Based on the input, you have to decide whether it is a SQL question or a visualization question."""
    formatted_prompt = PromptTemplate(
        template=prompt_template, input_variables=["input"]
    )
    # formatted_prompt = prompt.format(input=state["inputs"])

    llm_str = LLM().llm.with_structured_output(Choice)
    chain = (
        formatted_prompt | llm_str
    )  # Assuming this is the correct way to chain the prompt with the LLM

    response = chain.invoke({"input": state["inputs"]})
    state["choice"] = response

    return state


pandas_tool_node = ToolNode([pandas_tool.pandas_tool])
sql_tool_node = ToolNode([sql_tool.pandasql_tool])
rag_tool_node = ToolNode([rag_tool.rag_tool])


def router(state: GraphState) -> str:
    """
    Routes the input to the appropriate tool based on keywords.
    Must return a dict with a key matching the conditional edge.
    """
    choice = state["inputs"].lower()
    if "pandas" in choice:
        return "pandas"
    elif "sql" in choice:
        return "sql"
    else:
        return "rag"


# Define the graph state
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        messages: A list of messages exchanged in the conversation.
        inputs: The current user input.
    """

    messages: Annotated[List, add_messages]
    inputs: str


# Define the nodes and edges of the graph
graph = StateGraph(GraphState)

# Add nodes for the tools and input (assuming pandas_tool and sql_tool are defined elsewhere as Runnables or tools)
# Make sure pandas_tool and sql_tool are callable objects (like LangChain tools or Runnables)
graph.add_node(
    "pandas_tool", pandas_tool_node
)  # Assuming pandas_tool is a defined tool or Runnable
graph.add_node(
    "sql_tool", sql_tool_node
)  # Assuming sql_tool is a defined tool or Runnable

graph.add_node("rag_tool", rag_tool_node)
# Define an input node (this might not be necessary if you start directly with a tool/router,
# but keeping it for structure based on your original code)
graph.add_node("input", inputs_node)  # A simple node to process initial input
graph.add_node("router", router)
# Define a router function (you need to implement the logic for routing)
# This function will decide which node to go to next based on the state


# Set the entry point of the graph
graph.add_edge(START, "input")

# Add edges
graph.add_conditional_edges(
    "input",  # From the "input" node
    router,
    {
        "pandas": "pandas_tool",
        "sql": "sql_tool",
        "rag": "rag_tool",
    },  # Routing logic based on the choice
    # Use the router function to decide the next node
)

# Add edges from the tool nodes to the end (or another node if needed)
graph.add_edge("input", "router")


graph.add_edge("pandas_tool", END)
graph.add_edge("sql_tool", END)
graph.add_edge("rag_tool", END)


# Compile the graph
app = graph.compile()


# You can now invoke the graph with an initial state
# initial_state = {"messages": [], "inputs": "analyze this data using pandas"}
# result = app.invoke(initial_state)
# print(result)
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

    # Prepare the initial state
    initial_state = {"messages": [], "inputs": args.input_text}

    # Invoke the graph with the initial state
    result = app.invoke(initial_state)
    print(result)


if __name__ == "__main__":
    main()
