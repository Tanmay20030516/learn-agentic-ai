import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
os.environ["LANGSMITH_PROJECT"] = "sequential-chain-demo"

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

model1 = ChatGroq(model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"), temperature=0.7)
model2 = ChatGroq(model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"), temperature=0.4)

parser = StrOutputParser()

chain = prompt1 | model1 | parser | prompt2 | model2 | parser

# saving metadata and tags
config = {
    "run_name": "sequential-chain-demo-run", # custom trace name, rather than default "RunnableSequence"
    "tags": ["sequential-chain", "demo", "report generation"],
    "metadata": {
        "model1": "llama-3.1-8b-instant", "model2": "llama-3.1-8b-instant",
        "model1_temp": 0.7, "model2_temp": 0.4,
        "parser": "StrOutputParser",
    }
}

result = chain.invoke({'topic': 'Made in India Initiative'}, config=config)

print(result)
