from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st 
import openai
import os 
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
langchain_key=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = langchain_key
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Directly set the API key
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)

# Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system","You are a helpful chat assistant. Answer the queries carefully. Answer with detail."),
    ("user","Question: {question}")
])

output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# Streamlit UI
st.title("Chat bot")
input_text = st.text_input("Search the topic you want")

if input_text:
    st.write(chain.invoke({'question': input_text}))
