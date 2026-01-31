import os
import sqlite3
import tempfile
from dotenv import load_dotenv

from langchain_groq.chat_models import ChatGroq
from typing import (
    Dict,
    TypedDict,
    Annotated,
    Any,
    Generator,
    Iterator,
    Literal,
    Optional,
)
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import (
    SqliteSaver,
)  # pip install langgraph-checkpoint-sqlite
from langgraph.prebuilt import ToolNode, tools_condition

# pip install ddgs duckduckgo-search langchain_community
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import tool  # decorator

from langchain_community.document_loaders import PyPDFLoader  # pip install pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import (
    FAISS,
)  # pip install langchain-community faiss-cpu

load_dotenv()

## *************************************** RAG Setup ********************************************************
THREAD_RETRIEVERS: Dict[str, Any] = {}
THREAD_METADATA: Dict[str, Any] = {}
EMBEDDINGS = HuggingFaceEmbeddings(
    **{"model_name": "sentence-transformers/all-MiniLM-L6-v2"}
)


def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available"""
    if thread_id and thread_id in THREAD_RETRIEVERS:
        return THREAD_RETRIEVERS[thread_id]
    return None


def ingest_pdf(
    file_bytes: bytes, thread_id: str, filename: Optional[str] = None
) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread

    Returns a summary dict that can be surfaced in the UI
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024, chunk_overlap=256, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        VECTOR_STORE = FAISS.from_documents(chunks, EMBEDDINGS)
        RETRIEVER = VECTOR_STORE.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        )

        THREAD_RETRIEVERS[str(thread_id)] = RETRIEVER
        THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        # The FAISS store keeps copies of the text, so the temp file is safe to remove.
        try:
            os.remove(temp_path)
        except OSError:
            pass


## *************************************** Tools for chatbot ************************************************
@tool
def calculator(
    first_num: int | float,
    second_num: int | float,
    operation: str | Literal["add", "sub", "mul", "div", "mod"],
) -> dict:
    # need to write a detailed docstring for the funcion, since it will be read by our LLM
    """
    Perform basic arithmetic operations on two input numbers.
    Supported operations:
    - `add`: Addition (first_num + second_num)
    - `sub`: Subtraction (first_num - second_num)
    - `mul`: Multiplication (first_num * second_num)
    - `div`: Division (first_num / second_num); returns error if second_num is 0
    - `mod`: Remainder (first_num % second_num); only integers, returns error if second_num is 0
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0.0:
                return {"error": "Cannot divide by 0"}
            result = first_num / second_num
        elif operation == "mod":
            if not (isinstance(first_num, int) and isinstance(second_num, int)):
                return {"error": "Modulo must operate on integers"}
            else:
                if second_num == 0:
                    return {"error": "Cannot find remainder w.r.t. 0"}
            result = first_num % second_num
        else:
            return {"error": f"Unsupported operation type: {operation}"}

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def single_num_ops(
    num: float | int, op: str | Literal["log", "log10", "exp", "floor", "ceil"]
) -> dict:
    """
    Perform mathematical operations on a single input number.
    Supported operations:
    - `log`: Natural logarithm (ln)
    - `log10`: Base-10 logarithm
    - `exp`: Exponential (e^x)
    - `floor`: Largest integer less than or equal to num
    - `ceil`: Smallest integer greater than or equal to num
    """
    import math

    try:
        if op == "log":
            if num <= 0:
                return {"error": "Natural log is undefined for non-positive numbers"}
            result = math.log(num)
        elif op == "log10":
            if num <= 0:
                return {"error": "Log10 is undefined for non-positive numbers"}
            result = math.log10(num)
        elif op == "exp":
            result = math.exp(num)
        elif op == "floor":
            result = math.floor(num)
        elif op == "ceil":
            result = math.ceil(num)
        else:
            return {"error": f"Unsupported operation type: {op}"}

        return {"num": num, "operation": op, "result": result}

    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given stock symbol (e.g. 'AAPL', 'TSLA', 'MSFT')
    using Alpha Vantage with API key in URL
    """
    import requests

    apikey = str(os.getenv("ALPHA_VANTAGE_KEY", ""))
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={apikey}"
    response = requests.get(url)
    return response.json()


ddg_search_tool = DuckDuckGoSearchResults(
    name="web_search", description="Search the web using DuckDuckGo"
)


@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the `thread_id` when calling this tool.
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }


## *************************************** LLM Setup ********************************************************
LLM = ChatGroq(model=os.getenv("MODEL_NAME", ""), temperature=0.1)
# list of tools
TOOLS = [calculator, single_num_ops, get_stock_price, ddg_search_tool, rag_tool]
# make LLM tool aware
TOOL_AWARE_LLM = LLM.bind_tools(TOOLS)


## *************************************** Chatbot Graph ***************************************************
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# workflow nodes
def chat_node(state: ChatState, config=None) -> ChatState:
    """LLM node that may answer or request a tool call"""
    thread_id = None
    if config and isinstance(config, dict):  # get the thread-id from the chat thread
        thread_id = config.get("configurable", {}).get("thread_id")
    system_message = SystemMessage(
        content=(
            "You are a helpful assistant. For questions about the uploaded PDF, call "
            f"the `rag_tool` and include the thread_id `{thread_id}`."
            "You can also use the web search, stock price, and calculator tools when helpful."
            "If no document is available, ask the user to upload a PDF."
        )
    )
    messages = [system_message, *state["messages"]]
    response = TOOL_AWARE_LLM.invoke(messages, config=config)  # type: ignore
    return {"messages": [response]}


tool_node = ToolNode(TOOLS)


def create_chatbot():
    graph = StateGraph(ChatState)
    # nodes
    graph.add_node("chat", chat_node)
    graph.add_node("tools", tool_node)

    # edges
    graph.add_edge(START, "chat")
    graph.add_conditional_edges("chat", tools_condition)
    graph.add_edge("tools", "chat")
    return graph.compile(checkpointer=CHECKPOINTER)


def retrieve_all_threads() -> list[str]:
    """get all unique thread IDs from the database"""
    all_threads = list()
    vis = set()
    for checkpoint in CHECKPOINTER.list(
        config=None
    ):  # get all threads and conversations stored
        curr_thread_id = str(checkpoint.config["configurable"]["thread_id"]) # type: ignore
        if curr_thread_id in vis:
            continue
        vis.add(curr_thread_id)
        all_threads.append(checkpoint.config["configurable"]["thread_id"])  # type: ignore
    del vis
    return (all_threads)[::-1]


def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    return THREAD_METADATA.get(str(thread_id), {})


## *************************************** Title Generator Graph ***************************************************
class ChatTitle(TypedDict):
    """maybe once try to include this as a conditional edge in the main chatbot graph"""

    title: str
    context: list[dict[str, str]]


def gen_title(state: ChatTitle):
    context = ""
    for msg in state["context"]:
        role = msg["role"]
        text = msg["content"]
        context += f"{role}: {text}\n"
    prompt = f"""
    You are an expert chat analysis expert. Analyze the below chat and give a descriptive 3-4 word title to it.
    Chats: {context}
    - STRICTLY return ONLY the TITLE nothing else.
    """
    response: str | Any = LLM.invoke(prompt).content
    return {"title": response.strip()}


def create_title_generator():
    title_graph = StateGraph(ChatTitle)

    # nodes
    title_graph.add_node("generate_title", gen_title)

    # edges
    title_graph.add_edge(START, "generate_title")
    title_graph.add_edge("generate_title", END)

    return title_graph.compile()


def _create_titles_table():
    """creates the titles table if it doesn't exist"""
    cursor = CONN.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_titles (
        thread_id TEXT PRIMARY KEY,
        title TEXT NOT NULL
    )
    """)
    CONN.commit()


def save_chat_title(thread_id: str, title: str):
    """insert or update the chat title for a given thread_id"""
    cursor = CONN.cursor()
    cursor.execute(
        "INSERT OR REPLACE INTO chat_titles (thread_id, title) VALUES (?, ?)",
        (thread_id, title),
    )
    CONN.commit()


def retrieve_all_titles() -> dict[str, str]:
    """fetch all chat titles from the database"""
    cursor = CONN.cursor()
    cursor.execute("SELECT thread_id, title FROM chat_titles")
    rows = cursor.fetchall()
    return {row[0]: row[1] for row in rows}


## *************************************** Initialization ***************************************************
CONN = sqlite3.connect(database=os.getenv("DATABASE_URL", ""), check_same_thread=False)
CHECKPOINTER = SqliteSaver(conn=CONN)
_create_titles_table()
chatbot = create_chatbot()
title_generator = create_title_generator()
