__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from crewai import Agent, Task, Crew, Process
import litellm
import getpass



if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=1,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

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

# Function to run crew and handle potential BadRequestError
def run_crew(input_query):
    try:
        result = crew.kickoff(inputs={"input": input_query})
        return result  # Ensure you return the result to Streamlit
    except litellm.exceptions.BadRequestError as e:
        # Log the error and return a user-friendly message
        print(f"Bad Request Error: {e}")
        print(f"Input query that caused the error: {input_query}")
        return f"An error occurred while processing your request: {e}. Please try again later."

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
