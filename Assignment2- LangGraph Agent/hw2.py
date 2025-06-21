from typing import TypedDict
import os
import json
import subprocess
import pandas as pd
import sys
import csv

from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from openai import AzureOpenAI, RateLimitError, APIError, BadRequestError
from dotenv import load_dotenv

#########################
### Environment Setup ###
#########################

#--Specify the path to your env file:
env_path = ".\\environment_variables"

#--Load the .env file into environment
load_dotenv(env_path)

#--Verify values were loaded
api_key = os.getenv("CLASS_OPEN_API_KEY")
endpoint = os.getenv("SUBSCRIPTION_OPENAI_ENDPOINT_4o")

print("API Key loaded:", api_key[:5] + "..." if api_key else
f"Credentials not found, hw1.py expects 'environment_variables' file is within the current directory.")

#--Create an AzureOpenAI client instance for connection test
client = AzureOpenAI(
        api_key=os.getenv("CLASS_OPEN_API_KEY"),
        api_version="2024-08-01-preview",
        azure_endpoint=os.getenv("SUBSCRIPTION_OPENAI_ENDPOINT_4o")
    )

#--Test Conn Function
def test_client_connection(client_ins, model_name):
    """
    Tests the connection to a given Azure OpenAI model.
    """
    print(f"Testing connection to model '{model_name}'...")
    try:
        response = client_ins.chat.completions.create(
            model=model_name,
            temperature=0.0,
            messages=[
                {"role": "user", "content": "Say hello."}
            ]
        )
        reply = response.choices[0].message.content.strip()
        if reply:
            print(f"Connection to '{model_name}' successful.")
        else:
            print(f"Connection to '{model_name}' failed.")

    except Exception as e:
        raise RuntimeError(f"Failed to connect to model '{model_name}': {e}")


test_client_connection(client, "gpt-4o-mini")

#####################################################
### LangGraph Agent's State Dictionary Definition ###
#####################################################

class MyState(TypedDict):
    """
    Defines the state TypedDict being passed through the agent's workflow
    """
    completion: str         # Latest generated completion
    query_name: str         # The name of the query extracted in GetQueryDetails tool
    data_files: list        # Data files names being extracted in GetQueryDetails tool
    descriptions: list      # Descriptions for data files being extracted in GetQueryDetails tool,
                            # oder of descriptions corresponds to the data_files order.
    query: str              # Natural language query description
    attempt: int            # Counter for regeneration attempts done already
    program_code: str       # latest Python code generated to solve the query
    error_msg: str          # Captured stderr or JSON validation error from latest execution
    reflection: str         # LLM-generated analysis of the problem on latest execution
    too_many_attempts: bool # Flags to stop the agent's workflow as attempts limit reached
    output: str             # The printed result from latest program execution
    has_error: bool         # True if an error occurred during latest execution
    has_api_error: bool     # True if an error occurred during AzureOpenAI API invocation
    api_error: str          # Error occurred during AzureOpenAI API invocation

########################
### Helper Functions ###
########################

def validate_json_output(output_str):
    """
    Validates if output is valid, non-empty JSON.

    Args:
        output_str (str): The program's stdout output

    Returns:
        str or None: Descriptive error message if invalid, None if valid
    """
    # Check for no output
    if not output_str or not output_str.strip():
        return "Program produced no output"

    output_str = output_str.strip()

    # Check if it's valid JSON
    try:
        parsed_json = json.loads(output_str)
    except json.JSONDecodeError as e:
        return f"Output is not valid JSON: {str(e)}"

    # Check if it's empty JSON
    if not parsed_json or parsed_json == {}:
        return "Program produced empty JSON: {}"

    # Valid JSON
    return None

####################################
### LangGraph's Prompt Templates ###
####################################

#--PromptTemplate for GenQueryProgram tool
prompt_template_gen_query = PromptTemplate.from_template("""
Write a Python program to answer the following query using the specified data files:

Query:
{query}

Data files: {data_files}
Descriptions: {descriptions}

Requirements:
- Do not include the actual data in the code.
- The program must output only a single JSON object with the required fields.
- Do not include any explanations, comments, or formatting (e.g. no ```python).
- Only output valid Python code — nothing else.

Output:
Only the Python code itself.
""")

#--PromptTemplate for ReflectOnErr tool
prompt_template_reflect = PromptTemplate.from_template("""
Here is the query I asked the model to solve:
{query}

Here is the generated program:
{program_code}
Here is the error:
{error_msg}

Please reflect on what went wrong in the code. Please do not provide any code and focus on reflecting the errors and determining what caused the issue so that you can fix it.
""")
#--PromptTemplate for ReGenQueryPgm tool
prompt_template_regen = PromptTemplate.from_template("""
Fix the following Python program:
{program_code}

Based on this reflection:
{reflection}

The query is:
{query}
Use the same file names as before.

Requirements:
- Do not include the actual data in the code.
- The program must output only a single JSON object with the required fields.
- Do not include any explanations, comments, or formatting (e.g. no ```python).
- Only output valid Python code — nothing else.

Output:
Only the Python code itself.
""")

#############################
### LLM's System Messages ###
#############################

#--This system message dispatches on generating code tools
CODE_GENERATION_SYSTEM_MSG = """You are a Python code generator specializing in CSV data analysis. Your task is to generate executable Python code that processes CSV files and returns JSON results.

CRITICAL OUTPUT REQUIREMENTS:
- Generate ONLY executable Python code
- NO markdown code blocks (no ``` symbols)
- NO comments or explanations
- NO descriptive text before or after the code
- The code must run independently when saved to a .py file

CODE REQUIREMENTS:
- All data files are in the current working directory
- Process data according to the query requirements
- Output MUST be a single JSON object printed to stdout using print()
- Ensure JSON output has exactly the fields requested in the query

Remember: Generate pure Python code only. The code will be executed directly."""

#--This system message dispatches on self-Reflection tool
REFLECTION_SYSTEM_MSG = """You are a debugging expert analyzing Python code execution failures. Your task is to identify the root cause of errors and provide actionable insights for fixing the code."""

###############################
### LangGraph Agent's Tools ###
###############################

def get_query_details(state):
    """
    GetQueryDetails Tool:
    Reads the input files (query_input.txt and <name_of_query>.txt) and initializes the state of the graph.
    Also, extracting <name_of_data_file> & <desc> for each data file being mentioned in the provided query_input.txt.
    query_input.txt file will is expected to be in the following format:
    query_name:<name_of_query>
    data_file:<name_of_data_file>
    description: <desc>
    data_file:<name_of_data_file>
    description: <desc>
    …
    """
    print("** Executing GetQueryDetails Tool **")

    try:
        print("-----GetQueryDetails_INFO: Reading input file - query_input.txt")

        #--Attempts to open and read query_input.txt
        with open("query_input.txt") as f:
            lines = f.readlines()

        query_name = lines[0].split(":")[1].strip()

        if not query_name:
            raise ValueError("Failed to extract query_name from query_input.txt")

        #--Init data_files and descriptions lists for the Agent state creation

        data_files = []
        descriptions = []
        print("-----GetQueryDetails_INFO: Gathering data_files names and descriptions from query_input.txt")

        for i in range(1, len(lines), 2): #--data_files[i] corresponds to descriptions[i]
            data_files.append(lines[i].split(":")[1].strip())
            descriptions.append(lines[i + 1].split(":", 1)[1].strip())

        print(f"-----GetQueryDetails_INFO: Reading query from {query_name}.txt")
        with open(f"{query_name}.txt") as f:
            query = f.read().strip()

    #--Exceptions handling block, handles:
    ##--Failure in extracting info form query_input.txt
    ##--Failure in finding <query_name>.txt
    except (FileNotFoundError, ValueError) as e:
        print(f"GetQueryDetails_ERROR: {str(e)}")
        print("GetQueryDetails_ERROR: Cannot proceed with missing input files or information")
        ##--This will halt the entire workflow
        raise RuntimeError(f"GetQueryDetails_ERROR: {str(e)}")

    #--Initialize agent's state
    state.update({
        "query_name": query_name,
        "data_files": data_files,
        "descriptions": descriptions,
        "query": query,
        "attempt": 0,
        "too_many_attempts": False
    })

    return state


def gen_query_program(state):
    """
    GenQueryProgram tool:
    This tool prompts the LLM to generate the query program PQ.
    It formats the Langraph's promptTemplate and asking the llm for completion.
    Then, entire competition got as response is being copied to <query_name>.py file
    Finally, retaining code into agent's state for future steps.

    In case of error when prompting the LLM, the tool logs the specific error into its state
    and halts immediately agent's workflow.
    """
    print("** Executing GenQueryProgram Tool **")

    print("-----GenQueryProgram_INFO: Prompting LLM for code generation for the 1st time")

    #--Formatting prompt with the relevant input variables
    prompt = prompt_template_gen_query.format(
        query=state['query'],
        data_files=", ".join(state['data_files']),
        descriptions="\n".join(state['descriptions'])
    )

    #--Invoke LLM for code generation
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": CODE_GENERATION_SYSTEM_MSG},
                      {"role": "user", "content": prompt}],
            temperature=0.5
        )

        code = response.choices[0].message.content

        #--Retain generated code to agent's state and write it to executable file
        state["program_code"] = code
        with open(f"{state['query_name']}.py", "w") as f:
            f.write(code)

    # --Exceptions handling block, handles:
    ##--Failures occurred while invoking AzureOpenAI API
    except RateLimitError as e:
        print("Rate limit or quota exceeded:", str(e))
        state["api_error"] = f"Rate limit exceeded: {str(e)}"
        state["has_api_error"] = True
        #--Makes agent's workflow halt
        raise RuntimeError(f"GenQueryProgram_ERROR: {state["api_error"]}")

    except BadRequestError as e:
        print("Token limit exceeded or malformed prompt:", str(e))
        state["api_error"] = f"Bad request (possible token overflow): {str(e)}"
        state["has_api_error"] = True
        raise RuntimeError(f"GenQueryProgram_ERROR: {state["api_error"]}")

    except APIError as e:
        print("General API error:", str(e))
        state["api_error"] = f"API error: {str(e)}"
        state["has_api_error"] = True
        raise RuntimeError(f"GenQueryProgram_ERROR: {state["api_error"]}")

    return state

def execute_program(state):
    """
    ExecuteProgram tool:
    Executes the program PQ and captures any runtime errors.
    It will also check and make sure that valid, non-empty JSON was generated.
    For the generated code execution, uses Python's subprocess.
    """
    print("** Executing ExecuteProgram Tool **")

    query_name = state["query_name"]
    num_done_attempts = state["attempt"] + 1
    print(f"-----ExecuteProgram_INFO: Executing {query_name}.py | attempt # := {num_done_attempts}")

    #--Running the generated code in a Python subprocess
    try:
        result = subprocess.run(
            ["python", f"{query_name}.py"],
            capture_output=True, text=True, timeout=20
        )

        print(f"-----ExecuteProgram_INFO: Subprocess completed with return code: {result.returncode}")

        #--Retain execution's output
        output = result.stdout.strip()

        #--Check if <query_name>.py ran smoothly
        if result.stderr:
            error_msg = result.stderr.strip() or f"Program failed with exit code {result.returncode}"
            state["has_error"] = True
            state["error_msg"] = error_msg

        #--In case program ran, validate it's output
        else:
            output = result.stdout.strip()

            #--Use helper function to validate JSON
            json_error = validate_json_output(output)

            if json_error:
                #--Output validation failed
                state["has_error"] = True
                state["error_msg"] = json_error

            else:
                #--Valid JSON output
                print("-----ExecuteProgram_INFO: Valid JSON output generated")
                state["has_error"] = False
                state["output"] = output

                #--Write successful output to answer file
                with open(f"{query_name}_answer.txt", "w") as f:
                    f.write(output)


    except Exception as e:
        error_msg = f"Unexpected error during execution: {str(e)}"
        print(f"-----ExecuteProgram_ERROR: {error_msg}")
        state["has_error"] = True
        state["error_msg"] = error_msg

    return state

def check_for_errors(state):
    """
    Chk4rErr tool:
    Checks if there were errors in the execution or in generating valid non-empty JSON.
    If yes, it will direct the program to the “ReflectOnErr” agent to execute next.
    Otherwise, it will direct the program to END.
    Writing to <query_name>_errors.txt current errors - no errors: empty file.
    Also, Chk4rErr prompts through stdout a brief status on the agent's progress.
    """
    print("** Executing Chk4rErr Tool **")

    #--Provide a brief status
    print(
        f"-----Chk4rErr_INFO: Agent's current status: \n" 
        f"-> # of attempts made so far := {state['attempt'] + 1}\n"
        f"-> Last program execution had errors := {state['has_error']}"
    )

    if not state['has_error']:
        print(f"-> Last output and the answer is := \n"
              f"{state['output']}")


    #--Write to <query_name>_errors.txt current errors
    else:
        print(f"-----Chk4rErr_INFO: Writing to {state['query_name']}_errors.txt captured errors")
        raise RuntimeError

    er_msg = state.get("error_msg", "")
    with open(f"{state['query_name']}_errors.txt", "w") as f:
        f.write(er_msg)

    #--Write to <query_name>_reflect.txt latest reflection
    ##--We do so here because if Agent is successfull on first shot, reflect should exist anyway.
    ref_msg = state.get("reflection", "")
    with open(f"{state['query_name']}_reflect.txt", "w") as f:
        f.write(ref_msg)

    return state

def reflect_on_error(state):
    """
    ReflectOnErr tool:
    Executes when the PQ had errors or did not output the appropriate JSON.
    In this case it has the LLM reflect on the problem and generate a reflection summarizing the problem.
    This summary will be used in the tool “ReGenQueryPgm” to fix PQ.
    """
    print("** Executing ReflectOnErr Tool **")

    #--Check if agent should proceed to another attempt
    if state["attempt"] >= 2:
        state['too_many_attempts'] = True
        print(f"-----ReflectOnErr_INFO: Tried for 3 attempts already,"
              f" terminating agent's workflow now")
        return state

    #--Formatting prompt with the relevant input variables
    prompt = prompt_template_reflect.format(
        query=state['query'],
        program_code=state['program_code'],
        error_msg=state['error_msg']
    )

    #--Invoke LLM for reflection providing it with the last program code; latest error messages; original query.
    try:
        print(f"-----ReflectOnErr_INFO: Self-reflecting on latest errors")

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": REFLECTION_SYSTEM_MSG},
                      {"role": "user", "content": prompt}]
        )

        reflection = response.choices[0].message.content

        #--Retain LLM reflection to the Agent's state
        state["reflection"] = reflection

        #--Write to <query_name>_reflect.txt
        with open(f"{state['query_name']}_reflect.txt", "w") as f:
            f.write(reflection)

    #--Exceptions handling block, handles:
    ##--Failures occurred while invoking AzureOpenAI API
    except (RateLimitError, BadRequestError, APIError) as e:
        print("-----ReflectOnErr_ERROR: Reflection failed:", str(e))
        state["has_error"] = True
        state["reflection"] = f"Program suffered API error while attempting to reflect: {str(e)}"

    return state

def regenerate_query_program(state):
    """
    ReGenQueryPgm tool:
    Uses previous prompts and the reflection (computed by agent ReflectOnErr)
    To prompt the LLM to fix the bug and regenerate the query program.
    Here, in case of API error when trying to invoke LLM, agent's workflow halts immediately.
    """
    print("** Executing ReGenQueryPgm Tool **")

    state["attempt"] += 1

    #--Provide regeneration promptTemplate with the relevant input vars and context
    prompt = prompt_template_regen.format(
        program_code=state['program_code'],
        reflection=state['reflection'],
        query=state['query']
    )

    try:
        #--Invoking LLM to regenerate code
        print(f"-----ReGenQueryPgm_INFO: Regenerating queried program")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": CODE_GENERATION_SYSTEM_MSG},
                      {"role": "user", "content": prompt}]
        )

        new_code = response.choices[0].message.content
        state["program_code"] = new_code

        #--Write to <query_name>.py the regenerated code
        with open(f"{state['query_name']}.py", "w") as f:
            f.write(new_code)

    #--Exceptions handling block, handles:
    ##--Failures occurred while invoking AzureOpenAI API
    except RateLimitError as e:
        print("Rate limit or quota exceeded:", str(e))
        state["api_error"] = f"Rate limit exceeded: {str(e)}"
        state["has_api_error"] = True
        # --Makes agent's workflow halt
        raise RuntimeError(f"GenQueryProgram_ERROR: {state["api_error"]}")

    except BadRequestError as e:
        print("Token limit exceeded or malformed prompt:", str(e))
        state["api_error"] = f"Bad request (possible token overflow): {str(e)}"
        state["has_api_error"] = True
        raise RuntimeError(f"GenQueryProgram_ERROR: {state["api_error"]}")

    except APIError as e:
        print("General API error:", str(e))
        state["api_error"] = f"API error: {str(e)}"
        state["has_api_error"] = True
        raise RuntimeError(f"GenQueryProgram_ERROR: {state["api_error"]}")

    return state



def main(verbose=False):
    print("=== Starting LangGraph AI Agent ===")

    builder = StateGraph(MyState)
    builder.add_node("GetQueryDetails", get_query_details)
    builder.add_node("GenQueryProgram", gen_query_program)
    builder.add_node("ExecuteProgram", execute_program)
    builder.add_node("Chk4rErr", check_for_errors)
    builder.add_node("ReflectOnErr", reflect_on_error)
    builder.add_node("ReGenQueryPgm", regenerate_query_program)

    builder.set_entry_point("GetQueryDetails")
    builder.add_edge("GetQueryDetails", "GenQueryProgram")
    builder.add_edge("GenQueryProgram", "ExecuteProgram")
    builder.add_edge("ExecuteProgram", "Chk4rErr")
    builder.add_conditional_edges(
        "Chk4rErr",
        lambda state: state["has_error"],
        {
            True: "ReflectOnErr",
            False: END
        }
    )
    builder.add_conditional_edges(
        "ReflectOnErr",
        lambda state: state["too_many_attempts"],
        {
            True: END,
            False: "ReGenQueryPgm"
        }
    )
    builder.add_edge("ReGenQueryPgm", "ExecuteProgram")

    graph = builder.compile()

    #--Prints visualization of the agent's graph
    if verbose:
        # ASCII Visualization
        print("=== LangGraph Agent Workflow Visualization ===")
        print(graph.get_graph().draw_ascii())
        print("=" * 50)

    graph.invoke({})

if __name__ == "__main__":
    main()