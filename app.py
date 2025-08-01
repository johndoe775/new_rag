from typing import List, Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import Runnable
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.prompts import PromptTemplate
from research.tools import pandas_tool,sql_tool,helpers


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
    choice: Literal["pandas", "sql"] = Field(
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

"""
@tool
def pandas_tool(state: GraphState) -> str:
    "use this tool for pandas visualization"
    pass


@tool
def sql_tool(state: GraphState) -> str:
    "use this tool for sql query"
    pass

"""
pandas_tool_node = ToolNode([pandas_tool])
sql_tool_node = ToolNode([sql_tool])


def router(state: GraphState) -> str:
    """
    This is a placeholder router function.
    Implement your logic here to decide the next node (e.g., "pandas_tool", "sql_tool", or END).
    """
    # Example routing logic (replace with your actual logic)
    if "pandas" in state["choice"].choice.lower():
        return "pandas_tool"
    else:
        return "sql_tool"


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
    "router",  # From the "input" node
    router,
    {"pandas": "pandas_tool", "sql": "sql_tool"},
    # Use the router function to decide the next node
)

# Add edges from the tool nodes to the end (or another node if needed)
graph.add_edge("input", "router")


graph.add_edge("pandas_tool", END)
graph.add_edge("sql_tool", END)


# Compile the graph
app = graph.compile()

# You can now invoke the graph with an initial state
# initial_state = {"messages": [], "inputs": "analyze this data using pandas"}
# result = app.invoke(initial_state)
# print(result)
