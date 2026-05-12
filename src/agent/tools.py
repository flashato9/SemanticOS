import json
from datetime import datetime
import os
from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import tool
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL

# Path to your ignored data folder
DATA_DIR = r"C:\Users\Ato_K\Documents\programming\SemanticOS\.agent_data"
JSONL_LOG_PATH = os.path.join(DATA_DIR, "terminal_history.jsonl")

def log_to_jsonl(input_code, output, reasoning, status="success"):
    """Appends a structured entry to the JSONL ledger."""
    # Ensure directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    
    entry = {
        "timestamp": datetime.now().isoformat(),
        "status": status,
        "intent": reasoning,
        "input": input_code,
        "output": output.strip() if output else "No output"
    }
    
    with open(JSONL_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

@tool
def run_python_task(code: str, reasoning: str) -> str:
    """
    Executes Python code in a fresh, stateless environment.
    Use this for file management, running scripts, and system tasks.
    
    Args:
        code: The Python code to execute.
        reasoning: Why are you running this? (Saved to the ledger).
    """
    # Instantiate a fresh interpreter for every call (Stateless)
    repl = PythonREPL()
    result = None
    try:
        # Execute the code and capture stdout
        result = repl.run(code)
    except Exception as e:
        # Catch system-level or execution-level crashes
        error_msg = f"Runtime Error: {str(e)}"
        log_to_jsonl(code, error_msg, reasoning, status="crash")
        return error_msg
    
    # Check if the REPL returned an error string (some REPLs return errors in the output)
    if "Traceback" in result or "Error" in result:
        log_to_jsonl(code, result, reasoning, status="error")
    else:
        log_to_jsonl(code, result, reasoning, status="success")
        
    return result

    
# db = lancedb.connect("./.agent_data/vectors")
# table_name = "chat_history"
# embeddings_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
@tool
def search_memory(query: str):
    """Searches the long-term semantic database for past conversation context."""
    return "";
    # query_vector = embeddings_model.embed_query(query)
    
    # tbl = db.open_table(table_name)
    
    # # 2. Pass the vector, not the string, to .search()
    # results = tbl.search(query_vector).limit(3).to_pandas()
    
    # if results.empty:
    #     return "No relevant past memories found."
    
    # context = "\n---\n".join(results['text'].tolist())
    # return f"Found historical context:\n{context}"

ALL_TOOLS = [run_python_task]
