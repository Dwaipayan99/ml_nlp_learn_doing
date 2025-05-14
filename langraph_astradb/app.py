import cassio
import os 
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper

load_dotenv()

astra_db_application_token = os.getenv("astra_db_application_token")
astra_db_id = os.getenv("astra_db_id")
region = os.getenv("region")
openai_api_key = os.getenv("openai_key")
print(openai_api_key)

cassio.init(token=astra_db_application_token,
          database_id=astra_db_id)


url =[
    "https://medium.com/box-developer-blog/writing-a-box-agent-with-langchain-easier-than-you-think-00b53013c3e2",
    "https://vijaykumarkartha.medium.com/beginners-guide-to-creating-ai-agents-with-langchain-eaa5c10973e6",
    "https://medium.com/@lorevanoudenhove/how-to-build-ai-agents-with-langgraph-a-step-by-step-guide-5d84d9c7e832"
]

docs = [WebBaseLoader(url).load() for url in url]
doc_list = [item for sublist in docs for item in sublist]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs_list = text_splitter.split_documents(doc_list)

print(len(docs_list))

# texts = [docs_list.page_content for docs_list in docs_list]

embedding_model = OpenAIEmbeddings(model="text-embedding-3-large",openai_api_key=openai_api_key)

astra_vector_store = Cassandra(embedding=embedding_model,table_name="practise_astra",session=None, keyspace=None)
astra_vector_store.add_documents(docs_list)
print("Inserted %i headlines." % len(docs_list))
# vectors = embedding_model.embed_documents(texts)
astra_vector_index =  VectorStoreIndexWrapper(vectorstore=astra_vector_store)
query = "How to create an AI agent with Langchain?"

retriever = astra_vector_store.as_retriever(search_kwargs={"k": 4})
answer = retriever.invoke(query)

print(answer)






