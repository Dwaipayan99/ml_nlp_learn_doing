from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from dotenv import load_dotenv
import os
import requests

load_dotenv()


base_ollama_url = os.getenv("ollama_embedding_url")



print(base_ollama_url)



local_path = "WEF_The_Global_Cooperation_Barometer_2024.pdf"

if local_path:
    loader = UnstructuredPDFLoader(file_path=local_path)
    data = loader.load()
    # print(f"Loaded {len(data)} documents from local PDF.")
    # print(data[0].page_content)
else:
    print("PDF LOADING FAILED")



text_spllitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_spllitter.split_documents(data)

try:
    res = requests.get(base_ollama_url)
    if res.status_code != 200:
        raise Exception("Ollama is not responding properly.")
except Exception as e:
    print(f"Ollama server not running or unreachable: {e}")
    exit()

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url=base_ollama_url,
    # client_kwargs={"timeout": 60}
)
# embeddings = OllamaEmbeddings(ollama_url=ollama_url)

vectorstore = FAISS.from_documents(documents, embeddings)
# query = "What is the Global Cooperation Barometer?"
# retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
# result = retriever.invoke(query)
# print(f"Found {len(result)} relevant documents.")



local_model = "llama3"
llm = ChatOllama(model=local_model, temperature=0.1, max_tokens=2000)

query_prompt = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five different versions of the given user question to retrieve relevant documents" \
    from a vector store . By generating multiple perspectives on the user question, your goal ismto help the user overcome some of the limitations of the distance-based similarity search,
    Provide thse alternative questions separated by newlines. Original question: {question}""",
)

rag_retriever = MultiQueryRetriever.from_llm(
   retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    llm=llm,prompt=query_prompt)


print(rag_retriever)
template = """Answer the question based on the context provided. If the answer is not in the context, say 'I don't know'.
{context}
Question: {question}"""

prompt = ChatPromptTemplate.from_template(template)

chain = ({"context": rag_retriever,"question":RunnablePassthrough()}|prompt|llm|StrOutputParser())



answer = chain.invoke({"query": "What is  Barometer?"})

print(answer)

