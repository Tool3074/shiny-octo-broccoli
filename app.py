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
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader, WebBaseLoader
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
# from langchain.embeddings import OpenAIEmbeddings  # Removed
# from langchain_google_genai import GoogleGenerativeAIEmbeddings # Removed, using sentence-transformers
from langchain.tools import Tool

# For parsing PDFs and other document types
from langchain_community.document_loaders import UnstructuredPDFLoader, PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredHTMLLoader

# Load environment variables
load_dotenv()

# API Keys (Replace with your actual keys or environment variables)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  #  Needed for Gemini & Search Tool

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
        # Basic file processing (adjust based on file type)
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
            loader = None # Prevent errors later.

        if loader:
           documents = loader.load()
           # Now you can process the documents and prepare the query for the agent.
           query = st.text_area("Your Question/Query related to the uploaded document:")
           # Optionally, pre-populate the query field with a suggestion
           # query = st.text_area("Your Question/Query related to the uploaded document:", value="Summarize the key compliance requirements mentioned in this document.")
        os.remove(temp_file_path) # Clean up the temp file
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
    documents = None  # No document loaded yet.

# --- Agent Configuration ---

# Model Selection
model_name = st.selectbox("Choose the Language Model:", ("gemini-1.5-pro-latest", "gemini-1.0-pro"))  # Add others as needed

# Gemini Pro Model
llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=GOOGLE_API_KEY, convert_system_message_to_human=True)


# Embedding Model (Using sentence-transformers/all-mpnet-base-v2, a popular open-source option)
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2") # Or choose a different model

# Tools
search = DuckDuckGoSearchRun()
tools = [
    Tool(
        name="DuckDuckGo Search",
        func=search.run,
        description="Useful for when you need to answer questions about current events or general knowledge. Input should be a search query.",
    )
]

# Add Retrieval Tool if documents are loaded (RAG)
if input_method != "Text" and 'documents' in locals() and documents: # Correctly check for documents
   text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
   texts = text_splitter.split_documents(documents)

   # Choose a persistence directory.  Delete this directory to clear the vectorstore.
   persist_directory = "db"

   # Use HuggingFaceEmbeddings
   vectordb = Chroma.from_documents(documents=texts,
                                     embedding=embeddings,
                                     persist_directory=persist_directory)
   vectordb.persist()
   retriever = vectordb.as_retriever()

   qa = RetrievalQA.from_chain_type(
       llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
   )

   tools.append(
        Tool(
            name="RBI Document Retriever",
            func=qa.run,
            description="Useful for answering questions about the content of the uploaded RBI documents or the URL.  Input should be a fully formed question.",
        )
    )


# Agent creation
react_agent = create_react_agent(llm, tools, verbose=True) # Keep verbose=True for debugging
agent_executor = AgentExecutor.from_agent(agent=react_agent, tools=tools, verbose=True)

# ---  Run the Agent ---
if st.button("Get Compliance Advice"):
    if query:
        try:
            with st.spinner("Thinking..."):
                response = agent_executor.invoke({"input": query})
                st.write("### Answer:")
                st.write(response["output"])  # Access the 'output' key
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a question/query.")
