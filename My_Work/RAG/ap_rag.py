from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import chroma
from langchain_huggingface import HuggingFaceEndpoint
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os 

load_dotenv()

print(os.getcwd())
hf_key = os.getenv('HF_TOKEN')
print(hf_key)
print("------------KEY LOADED-------------")

loader = DirectoryLoader('./docs/',glob='./*.pdf',loader_cls=PyPDFLoader,show_progress=True,use_multithreading=True)

docs = loader.load()

print("----------NUMBER OF DOCS LOADED:", len(docs))

text_splitter =  RecursiveCharacterTextSplitter(chunk_size = 500 , chunk_overlap=100)
texts = text_splitter.split_documents(docs)
print(len(texts))
print("--------------------DOCUMENT LOADING DONE--------------------")
INDEX_PATH="faiss_index"
DOCS_PATH='./docs/'
MODEL_NAME='BAAI/bge-large-en-v1.5'
embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

# Check if the FAISS index already exists
if os.path.exists(INDEX_PATH):
    print("🟢 FAISS index found. Loading index...")
    db = FAISS.load_local(INDEX_PATH, embeddings)
    print("✅ FAISS index loaded successfully!")
else:
    print("🟡 FAISS index not found. Creating a new index...")

    # Load documents
    loader = DirectoryLoader(DOCS_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
    docs = loader.load()
    print(f"📄 {len(docs)} documents loaded.")

    # Split documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = splitter.split_documents(docs)
    print(f"🔗 {len(texts)} document chunks created.")

    # Create FAISS index
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(INDEX_PATH)
    print("✅ FAISS index created and saved successfully!")

query = "who is the author of the book?"

relevant_doc = db.similarity_search(query)

print(relevant_doc[0].page_content)


retriever = db.as_retriever(search_type="similarity",search_kwargs={"k":1})
retriever.get_relevant_documents(query)

repo_id = "mistralai/Mistral-7B-v0.1"

llm = HuggingFaceEndpoint(repo_id=repo_id, max_new_tokens=120, temperature=0.7)
# print(llm.invoke("Capital of india")) #to ensure that model is loaded and working properly

promt_template = """Based on the provided context try to answer the question following below rules:
1.If the answer is not found, don't speculate. Instead state "I don't know the aswer".
2.If the answer is found provide concise answer in not more than 15 sentences.
{context}
Question:{question}

Answer:
"""

prompt = PromptTemplate(template=promt_template,input_variables=["context","question"])

retrieval_QA = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type = "stuff",
    return_source_documents=True,
    chain_type_kwargs={"prompt":prompt}
) 

result = retrieval_QA.invoke({"query":query})
print("-------------------RESULT------------------")
print(result['result'])


















