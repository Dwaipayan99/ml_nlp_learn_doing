import os 
from dotenv import load_dotenv
import torch
from transformers import pipeline

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_huggingface import HuggingFacePipeline

from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
load_dotenv()
print(os.getcwd())
hf_key = os.getenv('HF_TOKEN')
print(hf_key)
print("----------------------------")
import torch
print(torch.backends.mps.is_available())  
print(torch.backends.mps.is_built())  


model_id = "meta-llama/Llama-3.2-3B"


pipe = pipeline(
    "text-generation", 
    model=model_id, 
    torch_dtype=torch.bfloat16, 
    device_map= "mps" if torch.backends.mps.is_available() else 'auto',
    max_new_tokens=200,
    pad_token_id=128001
)



hf_llm = HuggingFacePipeline(pipeline=pipe) 
memory = ConversationBufferMemory( memory_key="history",  
    return_messages=True)

conversation = ConversationChain(
    llm = hf_llm,
    verbose = True,
    memory= memory
)

print("\n🗨️ Starting Conversation 🗨️")
response_1 = conversation.predict(input="Hi, I'm Sam.")
print("🤖 Bot:", response_1)

response_2 = conversation.predict(input="Tell me about LangChain's memory management.")
print("🤖 Bot:", response_2)

response_3 = conversation.predict(input="What did I say earlier?")
print("🤖 Bot:", response_3)

response_4 = conversation.predict(input="Can you summarize our conversation so far?")
print("🤖 Bot:", response_4)

print("--------------------------------------------------")

print(conversation.memory.buffer)