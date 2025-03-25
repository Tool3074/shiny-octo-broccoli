__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
from crewai import Agent, Task, Crew, Process
from litellm import completion, token_counter
from litellm import litellm

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Configure LiteLLM to ONLY use the specified model
#litellm.only_use_the_specified_model = True  #Crucial setting

# Define the Gemini completion function using LiteLLM
def gemini_completion(prompt, model="gemini-pro", temperature=0.7, max_tokens=500):
    response = completion(model=model, messages=[{"role": "user", "content": prompt}], temperature=temperature, max_tokens=max_tokens)
    return response.choices[0].message["content"]


# Agents and Crew setup
compliance_analyst = Agent(
    role="RBI Compliance Analyst",
    goal="Provide accurate and up-to-date information on RBI regulations.",
    backstory="An expert in Indian banking regulations with years of experience.",
    verbose=True,
)

reporting_specialist = Agent(
    role="Reporting Specialist",
    goal="Generate clear and concise reports on compliance requirements.",
    backstory="A detail-oriented professional skilled in regulatory reporting.",
    verbose=True,
)

research_task = Task(
    description="Research the latest RBI guidelines on: {{input}}. Use Google Gemini to answer this question. Ensure your search query includes site:rbi.org.in to restrict to official RBI guidelines.",
    agent=compliance_analyst,
    expected_output="A summary of the relevant RBI guidelines and regulations, including links to the RBI website.",
)

report_task = Task(
    description="Create a detailed report summarizing the compliance requirements for {{input}} based on the research. Use Google Gemini to create the report. Ensure all references are to the RBI website.",
    agent=reporting_specialist,
    expected_output="A comprehensive report detailing the compliance requirements and steps to adhere to them, including citations from the RBI website.",
)

crew = Crew(
    agents=[compliance_analyst, reporting_specialist],
    tasks=[research_task, report_task],
    verbose=True,
    process=Process.sequential
)

def run_crew(input_query):
    result = crew.kickoff(inputs={"input":input_query})
    return result

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
