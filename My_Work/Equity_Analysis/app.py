import os 
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
import streamlit as st 
import pickle
import time 
from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from huggingface_hub import InferenceClient


from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
llm =  InferenceClient(model=repo_id, token=hf_token)



st.title("Research Tool")
st.sidebar.title("URLS")
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL{i+1}")
    urls.append(url)
main_placefolder = st.empty()
process_url_click = st.sidebar.button("Submit")
file_path = "faiss_store_open_ai.pkl"
if process_url_click :
    main_placefolder.text("Data loading ...")
    # load data
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    # split data
    main_placefolder.text("Text Splitter Started")
    text_splitter = RecursiveCharacterTextSplitter(separators=['\n\n','\n','.',','],chunk_size = 10000)

    docs = text_splitter.split_documents(data)
    #create embeddings and save it to FAISS

    # embedding = OpenAIEmbeddings()
    MODEL_NAME='BAAI/bge-large-en-v1.5'
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    vector_store = FAISS.from_documents(docs,embeddings)
    main_placefolder.text("Embedding vector strted building")
    # save FAISS index to a pickle path 
    with open(file_path, "wb") as f :
        pickle.dump(vector_store,f)




query = main_placefolder.text_input("Question...")

if query and os.path.exists(file_path):
    try:
        with open(file_path, "rb") as f:
            vector_store = pickle.load(f)

        # Create RetrievalQA chain
        chain = RetrievalQA.from_chain_type(
            llm=llm, 
            retriever=vector_store.as_retriever(), 
            chain_type="stuff"
        )

        # Run query through the chain
        result = chain.invoke({"query": query})  # Updated invocation method

        # Display answer in Streamlit
        st.header("Answer")
        st.subheader(result.get("answer", "No answer found"))  # Avoid KeyErrors

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
else:
    st.warning("Enter a question and make sure the FAISS index file exists.")

