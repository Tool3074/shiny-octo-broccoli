__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from crewai import Agent, Task, Crew, Process
import logging
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain_community.embeddings import HuggingFaceEmbeddings #corrected import

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

llm = ChatGoogleGenerativeAI(model="google/gemini-pro", temperature=0.7)

# Agents and Crew setup
compliance_analyst = Agent(
    role="RBI Compliance Analyst",
    goal="Provide accurate and up-to-date information on RBI regulations.",
    backstory="An expert in Indian banking regulations with years of experience.",
    llm=llm,
    verbose=False,
)

reporting_specialist = Agent(
    role="Reporting Specialist",
    goal="Generate clear and concise reports on compliance requirements.",
    backstory="A detail-oriented professional skilled in regulatory reporting.",
    llm=llm,
    verbose=False,
)

research_task = Task(
    description="Research the latest RBI guidelines on: {{input}}. Ensure your search query includes site:rbi.org.in to restrict to official RBI guidelines.",
    agent=compliance_analyst,
    expected_output="A summary of the relevant RBI guidelines and regulations, including links to the RBI website.",
)

report_task = Task(
    description="Create a detailed report summarizing the compliance requirements for {{input}} based on the research. Ensure all references are to the RBI website.",
    agent=reporting_specialist,
    expected_output="A comprehensive report detailing the compliance requirements and steps to adhere to them, including citations from the RBI website.",
)

crew = Crew(
    agents=[compliance_analyst, reporting_specialist],
    tasks=[research_task, report_task],
    verbose=False,
    process=Process.sequential
)

def run_crew(input_query):
    try:
        result = crew.kickoff(inputs={"input": input_query})
        return result
    except Exception as e:
        logging.error(f"CrewAI execution error: {e}")
        return f"An error occurred: {e}"

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

#Agent creation
react_agent = create_react_agent(llm, tools) #removed verbose=True
agent_executor = AgentExecutor.from_agent(agent=react_agent, tools=tools)

# ---  Run the Agent ---
