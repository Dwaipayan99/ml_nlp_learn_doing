import os 
import pandas as pd 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

ollama_url = os.getenv("ollama_url")

print(ollama_url)



