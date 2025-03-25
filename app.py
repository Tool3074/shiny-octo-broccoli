__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from crewai import Agent, Task, Crew, Process

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

llm = ChatGoogleGenerativeAI(model="gemini-pro")

# Agents and Crew setup
compliance_analyst = Agent(
    role="RBI Compliance Analyst",
    goal="Provide accurate and up-to-date information on RBI regulations.",
    backstory="An expert in Indian banking regulations with years of experience.",
    llm=llm,
    verbose=True,
)

reporting_specialist = Agent(
    role="Reporting Specialist",
    goal="Generate clear and concise reports on compliance requirements.",
    backstory="A detail-oriented professional skilled in regulatory reporting.",
    llm=llm,
    verbose=True,
)

research_task = Task(
    description="Research the latest RBI guidelines on the user's query.",
    agent=compliance_analyst,
    expected_output="A summary of the relevant RBI guidelines and regulations.",
)

report_task = Task(
    description="Create a detailed report summarizing the compliance requirements.",
    agent=reporting_specialist,
    expected_output="A comprehensive report detailing the compliance requirements and steps to adhere to them.",
)

crew = Crew(
    agents=[compliance_analyst, reporting_specialist],
    tasks=[research_task, report_task],
    verbose=True,
    process=Process.sequential
)

def run_crew(input_query):
    try:
    result = crew.kickoff(inputs={"input": input_query})
except litellm.exceptions.BadRequestError as e:
    # Log the error or handle it
    print(f"Bad Request Error: {e}")
    print(f"Input query that caused the error: {input_query}")

# Streamlit UI
st.title("RBI Compliance Advisor")

query = st.text_input("Enter your RBI compliance query:")

if st.button("Get Compliance Information"):
    if query:
        with st.spinner("Fetching compliance information..."):
            result = run_crew(query)
        st.write(result)
    else:
        st.warning("Please enter a query.")
