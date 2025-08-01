from langchain.tools import tool
from langchain_core.prompts import PromptTemplate
import pandas as pd
from pandasql import sqldf
from state import GraphState

pysqldf = lambda q: sqldf(q, globals())
import io
import os


@tool
def dataframe_query(state: GraphState):
    """You are given a dataframe info and a query from user regarding dataframe, your job is to write SQL query that suffices the user's objective
    note that query so given must run on pandasql
    use pysql to query the dataframe and return a pandas dataframe
    ### Inputs :
        - dataframe: The name of the dataframe to be queried.
        - query: The user's query regarding the dataframe.
    ### Output:
        - SQL query (Assume name of the dataframe to be df)
    ### Instructions:
        1. Don't give any explanation or additional information or preamble in query, just return the  query .
    ### Example Output:
    query 1: "give me all the rows where age is less than 30"
    output 1: "SELECT * FROM df WHERE Age < 30"

    """

    def capture_df_info(df: pd.DataFrame) -> str:
        """
        Capture the output of df.info() into a string.
        """
        buffer = io.StringIO()
        df.info(buf=buffer)
        return buffer.getvalue()

    def load_dataframes(csv_dir: str):
        """
        Reads all CSV files in csv_dir, returns:
        - paths: dict mapping dataframe name to its file path
        - df_infos: dict mapping dataframe name to its .info() string
        - globals_dict: dict to be used as globals when executing code
        Also registers each DataFrame in globals_dict under its name.
        """
        paths = {}
        df_infos = {}
        globals_dict = {}

        for fname in os.listdir(csv_dir):
            if fname.lower().endswith(".csv"):
                name = os.path.splitext(fname)[0]
                path = os.path.join(csv_dir, fname)

                # Read once, store in dicts
                df = pd.read_csv(path)
                paths[name] = path
                df_infos[name] = capture_df_info(df)
                globals_dict[name] = df.copy()

        return paths, df_infos, globals_dict

    # Ensure that the prompt is correctly formatted
    prompt = PromptTemplate(
        input_variables=[s, state["inputs"]],
        template=f"""You are given a dataframe whose name is df and dataframe_info  {s} and a {state["inputs"]} from user regarding dataframe, your job is to write SQL query that suffices the user's objective 
        note that only query must be returned and nothing else
        """,
    )
    # prompt = prompt.format(dataframe=dataframe, query=query)

    # Ensure that chain is defined and can invoke the LLM
    chain = prompt | LLM().llm  # This line needs to be valid in your context
    response = chain.invoke({"dataframe_info": s, "query": state["inputs"]}).content
    # state["message"].append(["completed SQL Query"])

    state["answer"] = pysqldf(response)

    return state
