__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
from typing import List, Optional

from dotenv import load_dotenv
import streamlit as st
from langchain.agents import AgentExecutor, create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter  # Modified import!
from langchain_community.document_loaders import TextLoader, WebBaseLoader
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent # Import initialize_agent

# For parsing PDFs and other document types
from langchain_community.document_loaders import UnstructuredPDFLoader, PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredHTMLLoader

# Load environment variables
load_dotenv()

# API Keys (Replace with your actual keys or environment variables)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  #  Needed for Gemini & Search Tool

# Set USER_AGENT (Optional, but recommended)
os.environ["USER_AGENT"] = "RBI Compliance Advisor Crew (your_email@example.com)"  # Replace with your email


# ---  Streamlit UI Setup ---
st.title("RBI Compliance Advisor Crew")
st.subheader("Ask questions about RBI compliance. Upload documents, enter URLs, or type your query.")

# Input methods
input_method = st.radio("Choose Input Method:", ("Text", "File Upload", "URL"))

query = ""  # Initialize query

# Input fields based on selection
if input_method == "Text":
    query = st.text_area("Your Question/Query:")
elif input_method == "File Upload":
    uploaded_file = st.file_uploader("Upload a Document (PDF, DOCX, TXT, HTML)", type=["pdf", "docx", "txt", "html"])
    if uploaded_file:
        # Basic file processing
        file_extension = uploaded_file.name.split(".")[-1].lower()
        temp_file_path = f"temp_file.{file_extension}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.read())

        if file_extension == "pdf":
            loader = PyPDFLoader(temp_file_path)
        elif file_extension == "docx":
            loader = Docx2txtLoader(temp_file_path)
        elif file_extension == "txt":
            loader = TextLoader(temp_file_path)
        elif file_extension == "html":
             loader = UnstructuredHTMLLoader(temp_file_path)
        else:
            st.error("Unsupported file type.")
            loader = None

        if loader:
           documents = loader.load()
           query = st.text_area("Your Question/Query related to the uploaded document:")
        os.remove(temp_file_path)
elif input_method == "URL":
    url = st.text_input("Enter URL:")
    if url:
        try:
            loader = WebBaseLoader(url)
            documents = loader.load()
            query = st.text_area("Your Question/Query related to the URL content:")
        except Exception as e:
            st.error(f"Error loading URL: {e}")
            documents = None
else:
    documents = None

# --- Agent Configuration ---

# Model Selection
model_name = st.selectbox("Choose the Language Model:", ("gemini-1.5-pro-latest", "gemini-1.0-pro"))

# Gemini Pro Model
llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=GOOGLE_API_KEY, convert_system_message_to_human=True)

# Embedding Model
embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")


# Tools
search = DuckDuckGoSearchRun()

# Rename the search tool
search_tool = Tool(
    name="Search",  # Renamed to "Search"
    func=search.run,
    description="Useful for when you need to answer questions about current events or general knowledge. Input should be a search query.",
)


# Add Retrieval Tool if documents are loaded (RAG)
lookup_tool = None # Initialize outside the if block

if input_method != "Text" and 'documents' in locals() and documents:
    #Use a better splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100  # Increased overlap, you can tune this.
    )
    texts = text_splitter.split_documents(documents)

    persist_directory = "db"
    vectordb = Chroma.from_documents(documents=texts,
                                     embedding=embeddings,
                                     persist_directory=persist_directory)
    #vectordb.persist() #Removed the line because it's deprecated.
    retriever = vectordb.as_retriever()

    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
    )

    lookup_tool = Tool(
        name="Lookup",  # Named "Lookup"
        func=qa.run,
        description="Useful for answering questions about the content of the uploaded RBI documents or the URL. Input should be a fully formed question.",
    )

# Create tools list.  Important to only include lookup tool if it exists.
tools = [search_tool]

if lookup_tool:
    tools.append(lookup_tool) # Added only when a document is available.
else:
    st.info("Upload a document or enter a URL to enable the 'Lookup' tool.")

#If there are not 2 tools we cannot continue
if len(tools) != 2:
    st.error("The 'react-docstore' agent requires exactly two tools: 'Search' and 'Lookup'.  Please upload a document or provide a URL to enable the 'Lookup' tool.")
    st.stop()  # Stop execution

# Define the prompt template
prompt_template = """
You are an expert RBI compliance advisor.  Use the tools available to answer the user's questions accurately and thoroughly.

If the user provides a document or URL, prioritize information from that source using the Lookup tool. Otherwise, use your general knowledge and Search tool to find the answer.

Be polite and professional in your responses.

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}
"""

# Create a PromptTemplate
prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["input", "tools", "tool_names", "agent_scratchpad"],
)



# Agent creation
agent_kwargs = {
    "prompt": prompt,
}

agent_executor = initialize_agent(
    tools=tools, # Pass tools here
    llm=llm,
    agent="react-docstore",  # Use react-docstore!
    verbose=True,
    agent_kwargs=agent_kwargs,
)


# ---  Run the Agent ---
if st.button("Get Compliance Advice"):
    if query:
        try:
            with st.spinner("Thinking..."):
                response = agent_executor.invoke({"input": query})
                st.write("### Answer:")
                st.write(response["output"])
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a question/query.")
