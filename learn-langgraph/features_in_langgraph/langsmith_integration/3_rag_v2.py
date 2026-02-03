import os
from dotenv import load_dotenv

from langsmith import traceable  # <-- key import

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# --- LangSmith env (make sure these are set) ---
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=...
# LANGCHAIN_PROJECT=pdf_rag_demo

load_dotenv()
os.environ["LANGSMITH_PROJECT"] = "rag-chatbot-v2"

PDF_PATH = "learn-langgraph/09_langsmith/langsmith-masterclass/islr.pdf"  # <-- change to your PDF filename

# the key difference in v2 is that we will trace the setup steps as well since
# they can be time-consuming and we may want to monitor them, and langsmith by default
# only traces the chain runs

# ---------- traced setup steps ----------
@traceable(name="load_pdf", tags=["setup", "pdf"], metadata={"step": "load", "loader": "PyPDFLoader"})
def load_pdf(path: str):
    loader = PyPDFLoader(path)
    return loader.load()  # list[Document]

@traceable(name="split_documents")
def split_documents(docs, chunk_size=1000, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

@traceable(name="build_vectorstore")
def build_vectorstore(splits):
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    # FAISS.from_documents internally calls the embedding model:
    vs = FAISS.from_documents(splits, emb)
    return vs

# You can also trace a “setup” umbrella span if you want:
@traceable(name="setup_pipeline")
def setup_pipeline(pdf_path: str):
    docs = load_pdf(pdf_path)
    splits = split_documents(docs, chunk_size=1024, chunk_overlap=128)
    vs = build_vectorstore(splits)
    return vs

# ---------- pipeline ----------
llm = ChatGroq(model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"), temperature=0.7)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# Build the index under traced setup
print("Setting up PDF RAG pipeline (this may take a while)...")
vectorstore = setup_pipeline(PDF_PATH)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
print("Setup complete.")

parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough(),
})

chain = parallel | prompt | llm | StrOutputParser()

# ---------- run a query (also traced) ----------
print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
q = input("\nQ: ").strip()

# Give the visible run name + tags/metadata so it’s easy to find:
config = {
    "run_name": "rag-v2-demo-run", # trace name rather than default "RunnableSequence"
    "tags": ["rag-v2", "demo", "QnA", "rag", "all-setup-traced"],
    "metadata": {
        "llm_model": "llama-3.1-8b-instant", "embedding_model": "sentence-transformers/all-mpnet-base-v2",
        "llm_model_temp": 0.7, "embedding_size": 768,
        "parser": "StrOutputParser",
    }
}

ans = chain.invoke(q, config=config) # type: ignore
print("\nA:", ans)