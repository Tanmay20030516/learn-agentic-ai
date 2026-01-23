import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
os.environ["LANGSMITH_PROJECT"] = "rag-chatbot-v1"

PDF_PATH = "learn-langgraph/09_langsmith/langsmith-masterclass/islr.pdf"  # <-- change to your PDF filename

# 1) Load PDF
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()  # one Document per page

# 2) Chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
splits = splitter.split_documents(docs)

# 3) Embed + index
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vs = FAISS.from_documents(splits, emb)
retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# 4) Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

# 5) Chain
llm = ChatGroq(model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"), temperature=0.7)

def format_docs(docs):
    """ format the document chunks retrieved into a single string """
    return "\n\n".join(d.page_content for d in docs)

parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

config = {
    "run_name": "rag-v1-demo-run", # trace name rather than default "RunnableSequence"
    "tags": ["rag-v1", "demo", "QnA", "rag"],
    "metadata": {
        "llm_model": "llama-3.1-8b-instant", "embedding_model": "sentence-transformers/all-mpnet-base-v2",
        "llm_model_temp": 0.7, "embedding_size": 768,
        "parser": "StrOutputParser",
    }
}
chain = parallel | prompt | llm | StrOutputParser()

# 6) Ask questions
print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
q = input("\nQ: ")
ans = chain.invoke(q.strip(), config=config) # type: ignore
print("\nA:", ans)
