from typing import Annotated
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

hiring_prompt = "we need to hire a software engineer with experience in Python and machine learning."

llm = ChatGroq(model="llama-3.1-8b-instant")

jd_prompt = ChatPromptTemplate.from_template(
    """ Write a detailed job description for the following hiring need: {hiring_need}""",
)
jd_chain = jd_prompt | llm | StrOutputParser()

def approve_jd(jd):
    # simulating an approval process
    return "Approved"

def post_jd(jd):
    print("Posting the approved Job Description...")

approved = False
jd_output = None

while not approved:
    jd_output = jd_chain.invoke({"hiring_need": hiring_prompt})
    print("Generated Job Description:")
    print(jd_output)
    approval = approve_jd(jd_output)
    if approval == "Approved":
        approved = True
        post_jd(jd_output)
    else:
        print("Job Description not approved, regenerating...")
class State:
    essay_text: str
    topic: str
    clarity_score: int
    depth_score: int
    language_score: int
    total_score: int
    feedback: Annotated[list[str], ...] # feedback given to essay
    evaluation_round: int # evaluation iterations