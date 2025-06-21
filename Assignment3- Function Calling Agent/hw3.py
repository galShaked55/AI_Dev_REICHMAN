import os
import json
import subprocess
import time
import random

from dotenv import load_dotenv
from openai import AzureOpenAI
from duckduckgo_search import DDGS
from langchain.prompts import PromptTemplate

##############################
# ==== Environment Setup === #
##############################

env_path = ".\\environment_variables"
load_dotenv(env_path)

# Environment variables
AZURE_OPENAI_API_KEY = os.getenv('CLASS_OPEN_API_KEY')
AZURE_OPENAI_ENDPOINT_4o = os.getenv('SUBSCRIPTION_OPENAI_ENDPOINT_4o')

print("API Key loaded:", AZURE_OPENAI_API_KEY[:5] + "..." if AZURE_OPENAI_API_KEY else
f"Credentials not found, hw1.py expects 'environment_variables' file is within the current directory.")

# Model configuration
MODEL_4o = 'gpt-4o-mini'
OPENAI_API_VERSION_4o = '2024-08-01-preview'

# Initialize OpenAI client
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=OPENAI_API_VERSION_4o,
    azure_endpoint=AZURE_OPENAI_ENDPOINT_4o
)

#############################
# ==== Prompt Templates === #
#############################
# ==== Tools Prompt Templates === #
extract_entities_pr = PromptTemplate(
    input_variables=["entity_type", "file_content"],
    template="""Extract all entities of type '{entity_type}' from the following text.
            Return ONLY a Python list format string like: ["entity1", "entity2", "entity3"]
            If no entities are found, return an empty list: []

            Text:
            {file_content}
            """
            )

internet_search_pr = PromptTemplate(
    input_variables=["an_attribute", "an_entity", "search_results_text"],
    template="""Based on these search results, find the {an_attribute} of {an_entity}.
            Return your answer as a JSON object with exactly this format:
            {{"entity": "{an_entity}", "attribute": "{an_attribute}", "attribute_value": "<the value you found>"}}

            Search Results:
            {search_results_text}"""
            )

gen_plot_pr = PromptTemplate(
    input_variables=["plot_request", "input_file", "columns", "output_png", "additional_context"],
    template="""Generate a Python program that creates a plot based on this request:
            {plot_request}
            
            The program should:
            1. Read data from CSV file: {input_file}
            2. The CSV columns are: {columns}
            3. Create the requested plot using matplotlib
            4. Save the plot to: {output_png}
            
            Who asked for the plot wanted to inform you also that:
            {additional_context}
            
            IMPORTANT: 
            - Use plt.savefig('{output_png}') to save the plot
            - Do NOT use plt.show() - the program must terminate automatically
            - Import all necessary libraries (pandas, matplotlib.pyplot, etc.)
            - Handle the CSV reading properly with pandas
            - no ```python ! the code you pass will be executed directly
            - Do not include any explanations, comments, or formatting (e.g. no ```python).
            - Only output valid Python code — nothing else.
            
            Generate only the Python code, no explanations."""
            )

debug_regen_pr = PromptTemplate(
    input_variables=["original_code", "errors"],
    template="""The following Python program failed with errors. Please analyze and fix it.

            Original Code:
            {original_code}
            
            Errors:
            {errors}
            
            Generate the corrected Python code that fixes these errors. 
            Important: 
            - Keep the same functionality as intended
            - Fix all errors mentioned
            - no ```python ! the code you pass will be executed directly
            - If it's a plotting program, ensure it uses plt.savefig() and NOT plt.show()
            - Return only the corrected code, no explanations."""
            )

extract_entities_sys_pr = PromptTemplate(
    input_variables=[],
    template="""You are an expert entity extraction specialist. Your task is to carefully identify and extract specific types of entities from text. Be precise and comprehensive in your extraction. Always return results in the exact format requested."""
)

internet_search_sys_pr = PromptTemplate(
    input_variables=[],
    template="""You are a research analyst specializing in extracting specific information from web search results. Analyze the provided search results carefully to find the requested attribute. Be accurate and cite information that appears in the search results. If the information is not clearly available, indicate uncertainty."""
)

gen_plot_prog_sys_pr = PromptTemplate(
    input_variables=[],
    template="""You are an expert Python data visualization programmer. Generate clean, functional matplotlib code that reads CSV data and creates the requested plots. Your code must be production-ready, handle errors gracefully, and save plots without displaying them."""
)

debug_and_regen_sys_pr = PromptTemplate(
    input_variables=[],
    template="""You are a Python debugging expert. Analyze the provided code and error messages to identify root causes and generate corrected code. Maintain the original functionality while fixing all issues. Ensure your corrected code is robust and handles edge cases."""
)

###################
# ==== Logger === #
###################
class AgentLogger:
    """
    This class provides the logging logic during the Agent workflow
    AgentLogger instance will be created at the begging of a flow.
    """
    def __init__(self, query_name, verbose=True):
        self.verbose = verbose

        # Create log filename by removing .txt extension if present
        if query_name.endswith('.txt'):
            log_filename = query_name[:-4] + '.log'
        else:
            log_filename = query_name + '.log'

        self.log_file = open(log_filename, 'w', encoding='utf-8')

        # Write header to log file
        self.log_file.write("=" * 80 + '\n')
        self.log_file.write(f"AGENT EXECUTION LOG - {query_name}\n")
        self.log_file.write("=" * 80 + '\n\n')
        self.log_file.flush()

    def log_message(self, message):
        # Print to console if verbose
        if self.verbose:
            print(message)

        # Write to log file
        if message.startswith("**Entering tool"):
            self.log_file.write("\n" + "-" * 60 + '\n')
            self.log_file.write(f"TOOL ENTRY: {message[16:-2]}\n")  # Remove "**Entering tool " and "**"
            self.log_file.write("-" * 60 + '\n')
        elif message.startswith("**Exiting tool"):
            self.log_file.write(f"TOOL EXIT: {message[15:-2]}\n")  # Remove "**Exiting tool " and "**"
            self.log_file.write("-" * 60 + '\n\n')
        elif message.startswith("Parameter"):
            self.log_file.write(f"  | {message}\n")
        else:
            self.log_file.write(f"{message}\n")

        self.log_file.flush()

    def close(self):
        if self.log_file:
            self.log_file.write("\n" + "=" * 80 + '\n')
            self.log_file.write("END OF LOG\n")
            self.log_file.write("=" * 80 + '\n')
            self.log_file.close()

##################
# ==== Tools === #
##################

def extract_entities_from_file(file_name: str, entity_type: str, logger: AgentLogger) -> str:
    """
    This tool is given the name of a file and an “entity type”, a string.
    It returns a comma-separated list of all the entities of that type in the file.
    """

    logger.log_message("**Entering tool extract_entities_from_file**")
    logger.log_message(f"Parameter file_name = {file_name}")
    logger.log_message(f"Parameter entity_type = {entity_type}")

    with open(file_name, 'r', encoding='utf-8') as f:
        file_content = f.read()

    prompt = extract_entities_pr.format(
        entity_type=entity_type,
        file_content=file_content
    )

    response = client.chat.completions.create(
        model=MODEL_4o,
        messages=[
            {"role": "system", "content": extract_entities_sys_pr.format()},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    result = response.choices[0].message.content.strip()
    logger.log_message(f"Result = {result}")
    logger.log_message("**Exiting tool extract_entities_from_file**")
    return result


def internet_search_attribute(an_entity: str, an_attribute: str, logger: AgentLogger) -> str:
    """
    This tool searches the Internet to find the desired attribute of the given entity.
    After receiving the search results,
    it then asks the LLM to review the results and return a JSON structure of the form: {"entity": an_entity, "attribute": an_attribute, "attribute_value": <value>},
    where <value> is the value extracted from the Internet search results.
    This tool implements exponential backoff delay mechanism to avoid rate limit.
    """
    logger.log_message("**Entering tool Internet_search_attribute**")
    logger.log_message(f"Parameter an_entity = {an_entity}")
    logger.log_message(f"Parameter an_attribute = {an_attribute}")


    search_query = f"{an_entity} {an_attribute}"

    # DuckDuckGo internet search with exponential backoff mechanism
    results = None
    for attempt in range(3):
        try:
            if attempt > 0:
                delay = (2 ** attempt) + random.uniform(1, 2)
                print(f"internet_search_attribute INFO -> Retrying search after {delay:.1f} seconds... Due to rate limit")
                time.sleep(delay)

            ddgs = DDGS()
            results = list(ddgs.text(search_query, max_results=3))
            break

        except Exception as e:
            print(f"Search attempt {attempt + 1} failed: {str(e)}")
            if attempt == 2:  # Last attempt
                result = f'{{"entity": "{an_entity}", "attribute": "{an_attribute}", "attribute_value": "All search attempts failed"}}'
                logger.log_message(f"Result = {result}")
                logger.log_message("**Exiting tool Internet_search_attribute**")
                return result

    # Clean and format results
    search_results_text = ""

    for i, result in enumerate(results):
        search_results_text += f"Result {i + 1}:\n"
        search_results_text += f"Title: {result.get('title', '')}\n"  # Title provides context
        search_results_text += f"Body: {result.get('body', '')}\n\n"  # Body provides the facts

    prompt = internet_search_pr.format(
        an_attribute=an_attribute,
        an_entity=an_entity,
        search_results_text=search_results_text
    )

    response = client.chat.completions.create(
        model=MODEL_4o,
        messages=[
            {"role": "system", "content": internet_search_sys_pr.format()},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    result = response.choices[0].message.content.strip()
    logger.log_message(f"Result = {result}")
    logger.log_message("**Exiting tool Internet_search_attribute**")
    return result


def gen_plot_prog(plot_request: str, input_file: str, columns: str,
                  gen_output_program_fn: str, output_png: str, logger: AgentLogger,
                  additional_context: str = "") -> str:
    """
    This tool receives a plot_request (a string) which describes a query to be performed on the input_file (a csv file).
    This input_file has the names of the columns in its first line,
    and every following line is a row of data. The names of the columns are also supplied as an input parameter to this agent.
    The tool produces a program P that computes the plot_request and stores the generated plot in the .png file output_png.
    The tool then writes the generated program P to the file gen_output_file.
    It returns the generated program.
    Additional_context param introducing a space for one who invokes the tool to inject wider context.
    """

    logger.log_message("**Entering tool gen_plot_prog**")
    logger.log_message(f"Parameter plot_request = {plot_request[:50]}...")
    logger.log_message(f"Parameter input_file = {input_file}")
    logger.log_message(f"Parameter columns = {columns[:50]}...")
    logger.log_message(f"Parameter gen_output_program_fn = {gen_output_program_fn}")
    logger.log_message(f"Parameter output_png = {output_png}")
    logger.log_message(f"Parameter additional_context = {additional_context[:50]}")

    prompt = gen_plot_pr.format(
        plot_request=plot_request,
        input_file=input_file,
        columns=columns,
        output_png=output_png,
        additional_context=additional_context
    )

    response = client.chat.completions.create(
        model=MODEL_4o,
        messages=[
            {"role": "system", "content": gen_plot_prog_sys_pr.format()},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    generated_code = response.choices[0].message.content.strip()

    with open(gen_output_program_fn, 'w') as f:
        f.write(generated_code)

    logger.log_message("**Exiting tool gen_plot_prog**")
    return generated_code


def execute_Python_prog(program_fn: str, logger: AgentLogger) -> str:
    """
    This tool executes the program in file program_fn. If the program executes successfully, it returns the string 'Program executed successfully'.
    Otherwise, it returns the error messages from stderr.
    """
    logger.log_message("**Entering tool execute_Python_prog**")
    logger.log_message(f"Parameter program_fn = {program_fn}")

    try:
        result = subprocess.run(
            ["python", program_fn],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            output = "Program executed successfully"
        else:
            output = result.stderr

    except subprocess.TimeoutExpired:
        output = "Error: Program execution timed out after 30 seconds"
    except Exception as e:
        output = f"Error: {str(e)}"

    logger.log_message("**Exiting tool execute_Python_prog**")
    return output


def debug_and_regenerate_prog(program_fn: str, errors: str, logger: AgentLogger) -> str:
    """
    This tool takes the name of a file which contains a Python program. This program executed unsuccessfully generating the given errors.
    The tool first reflects on the errors to find the root cause.
    It then generates a new program that fixes those errors.
    It stores the newly generated program in the same file, overwriting the old program.
    """
    logger.log_message("**Entering tool debug_and_regenerate_prog**")
    logger.log_message(f"Parameter program_fn = {program_fn}")
    logger.log_message(f"Parameter errors = {errors[:50]}...")

    with open(program_fn, 'r') as f:
        original_code = f.read()

    prompt = debug_regen_pr.format(
        original_code=original_code,
        errors=errors
    )

    response = client.chat.completions.create(
        model=MODEL_4o,
        messages=[
            {"role": "system", "content": debug_and_regen_sys_pr.format()},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    corrected_code = response.choices[0].message.content.strip()

    with open(program_fn, 'w') as f:
        f.write(corrected_code)

    logger.log_message("**Exiting tool debug_and_regenerate_prog**")
    return "Program regenerated"


def write_file(file_content: str, fn: str, logger: AgentLogger) -> str:
    """
    This tool writes the file contents to the file fn.
    """
    logger.log_message("**Entering tool write_file**")
    logger.log_message(f"Parameter file_content = {file_content[:50]}...")
    logger.log_message(f"Parameter fn = {fn}")

    with open(fn, 'w') as f:
        f.write(file_content)

    logger.log_message("**Exiting tool write_file**")
    return "File written successfully"

########################
# === Tools schema === #
########################

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "extract_entities_from_file",
            "description": "Extract all entities of a specific type from a text file. Returns a string formatted as a Python list like '[\"entity1\", \"entity2\"]'",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_name": {
                        "type": "string",
                        "description": "Path to the text file to read"
                    },
                    "entity_type": {
                        "type": "string",
                        "description": "Type of entities to extract (e.g., 'city', 'hobby', 'country')"
                    }
                },
                "required": ["file_name", "entity_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "Internet_search_attribute",
            "description": "Search the internet to find a specific attribute of an entity. Returns JSON with entity, attribute, and attribute_value",
            "parameters": {
                "type": "object",
                "properties": {
                    "an_entity": {
                        "type": "string",
                        "description": "The entity to search for (e.g., 'Abraham Lincoln', 'Canada')"
                    },
                    "an_attribute": {
                        "type": "string",
                        "description": "The attribute to find (e.g., 'population', 'birthdate')"
                    }
                },
                "required": ["an_entity", "an_attribute"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "gen_plot_prog",
            "description": "Generate a Python program that creates a plot from CSV data and saves it as PNG",
            "parameters": {
                "type": "object",
                "properties": {
                    "plot_request": {
                        "type": "string",
                        "description": "Description of the plot to create"
                    },
                    "input_file": {
                        "type": "string",
                        "description": "Path to the CSV file containing data"
                    },
                    "columns": {
                        "type": "string",
                        "description": "Comma-separated list of column names in the CSV"
                    },
                    "gen_output_program_fn": {
                        "type": "string",
                        "description": "Filename where the generated Python program should be saved"
                    },
                    "output_png": {
                        "type": "string",
                        "description": "Filename where the plot PNG should be saved"
                    },
                    "additional_context": {
                        "type": "string",
                        "description": "Additional context from previous steps (e.g., specific entities found, data insights, filtering requirements)"
                    }
                },
                "required": ["plot_request", "input_file", "columns", "gen_output_program_fn", "output_png"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_Python_prog",
            "description": "Execute a Python program file. Returns 'Program executed successfully' or error messages",
            "parameters": {
                "type": "object",
                "properties": {
                    "program_fn": {
                        "type": "string",
                        "description": "Path to the Python program file to execute"
                    }
                },
                "required": ["program_fn"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "debug_and_regenerate_prog",
            "description": "Debug and fix a Python program that failed with errors. Overwrites the original file with corrected code",
            "parameters": {
                "type": "object",
                "properties": {
                    "program_fn": {
                        "type": "string",
                        "description": "Path to the Python program file that needs fixing"
                    },
                    "errors": {
                        "type": "string",
                        "description": "The error messages from the failed execution"
                    }
                },
                "required": ["program_fn", "errors"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file. Creates the file if it doesn't exist, overwrites if it does",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_content": {
                        "type": "string",
                        "description": "The content to write to the file"
                    },
                    "fn": {
                        "type": "string",
                        "description": "Path/filename where content should be written"
                    }
                },
                "required": ["file_content", "fn"]
            }
        }
    }
]

######################
# === Agent Flow === #
######################

def main():
    # Read input file - hardcoded, one better verify before execution it matches the intended input.json file
    with open("input.json", 'r') as f:
        input_data = json.load(f)

    # Extract values
    query_file = input_data.get("query_name") or input_data.get("query-name")
    resources = input_data.get("file_resources") or input_data.get("resources")

    # Note: assuming query_file name consists of full file name, including file extension name.
    with open(query_file, 'r', encoding='utf-8') as f:
        query_content = f.read()

    system_prompt = """You are an intelligent agent that can analyze files and answer queries.

                    IMPORTANT: Each tool operates independently with NO memory of previous results or conversation history. You must explicitly provide all necessary context and information when calling tools.
                    
                    You have access to these tools:
                    - extract_entities_from_file: Extract entities from text files
                    - Internet_search_attribute: Search the internet for attributes of entities  
                    - gen_plot_prog: Generate Python programs to create plots from CSV data
                    - execute_Python_prog: Execute Python programs
                    - debug_and_regenerate_prog: Fix Python programs that have errors
                    - write_file: Write content to files
                    
                    CONTEXT PASSING GUIDELINES:
                    1. When calling tools sequentially, consider what information each tool needs from previous steps
                    2. If a tool needs results from a previous tool, you must include that information in the tool's parameters
                    3. For complex queries, break down the task and ensure each tool call has sufficient context
                    4. When searching for attributes of entities, make sure to use the exact entity names found in previous extractions
                    5. When generating plots, ensure you provide complete column information and clear plotting requirements. Remember that the tool might not have access to stuff you obtained before invoking it.
                    
                    
                    Available file resources:"""

    for resource in resources:
        system_prompt += f"- {resource['file_name']}: {resource['description']}\n"

    system_prompt += "\nAnalyze the query and use appropriate tools to solve it step by step."

    # Create logger instance
    logger = AgentLogger(query_file)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query_content}
    ]

    tool_count = 0
    llm_count = 0

    try:
        # Agent logic flow under the try-outs constraints
        while tool_count < 15 and llm_count < 15:

            logger.log_message("Calling LLM for next tool to invoke")

            # Invoke LM to choose the next tool
            response = client.chat.completions.create(
                model=MODEL_4o,
                messages=messages,
                tools=tools_schema,
                tool_choice="auto"
            )

            llm_count += 1
            message = response.choices[0].message

            if message.tool_calls:
                messages.append(message)

                for tool_call in message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    result = ""

                    if function_name == "extract_entities_from_file":
                        result = extract_entities_from_file(**function_args, logger=logger)
                        llm_count+=1
                    elif function_name == "Internet_search_attribute":
                        result = internet_search_attribute(**function_args, logger=logger)
                        llm_count += 1
                    elif function_name == "gen_plot_prog":
                        result = gen_plot_prog(**function_args, logger=logger)
                        llm_count += 1
                    elif function_name == "execute_Python_prog":
                        result = execute_Python_prog(**function_args, logger=logger)
                    elif function_name == "debug_and_regenerate_prog":
                        result = debug_and_regenerate_prog(**function_args, logger=logger)
                        llm_count += 1
                    elif function_name == "write_file":
                        result = write_file(**function_args, logger=logger)

                    tool_count += 1

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    })
            else:
                logger.log_message(f"final response is = {message.content}")
                break
    finally:
        logger.close()

if __name__ == "__main__":
    main()