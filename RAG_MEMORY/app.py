import os
import warnings
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
from dotenv import load_dotenv
import bs4
from langchain import hub 
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import MessagesPlaceholder


load_dotenv()

langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
print(langchain_api_key)
print(google_api_key)

warnings.filterwarnings('ignore')

model = ChatGoogleGenerativeAI(model = "gemini-1.5-pro-001",convert_system_message_to_human=True)

print(model.invoke('hi').content)
gemini_embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
loader = WebBaseLoader(web_path=("https://lilianweng.github.io/posts/2023-06-23-agent/"),bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-content","post-title","post-header"))),)

doc = loader.load()

# print(doc)

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap = 200)
splits = text_splitter.split_documents(doc)

# print(splits)

vector_store = Chroma.from_documents(documents=splits,embedding=gemini_embedding)

retriever = vector_store.as_retriever()

systemPrompt = (

    "You are an assistant for question answering task."
    "Use the following retrieved context to answer the question"
    "If you don't know the answer, say that you don't know"
    "Use three sentences maximum and keep the answer concise"
    "\n\n"
    "{context}"
)


chatprompt_template = ChatPromptTemplate.from_messages(
    [
        ("system",systemPrompt),
        ("human","{input}")
    ]
)

question_answering_chain = create_stuff_documents_chain(model,chatprompt_template)
rag_chain = create_retrieval_chain(retriever,question_answering_chain)
response = rag_chain.invoke({"input":"give me summary of entire document"})

print(response["answer"])