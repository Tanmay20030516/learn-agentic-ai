from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

# llm
llm = ChatGroq(model="llama-3.1-8b-instant")

# chat template
chat_prompt = ChatPromptTemplate([
    ("system", "You are an expert and helpful customer support assistant."),
    MessagesPlaceholder("chat_history"), # generally used to retrieve chat history
    ("human", "{query}")
])

# read the chat history, so that the chatbot has access to previous conversations with the user
chat_history = []
with open("learn-langchain/chat_history.txt", "r") as f:
    chat_history.extend(f.readlines())

# now let us check
chain = chat_prompt | llm
output = chain.invoke({"chat_history": chat_history, "query": "What is the status of my refund?"})
print(output.model_dump()['content'])