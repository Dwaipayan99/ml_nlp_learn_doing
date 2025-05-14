import os 
from dotenv import load_dotenv
import torch
from transformers import pipeline

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_huggingface import HuggingFacePipeline

from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import chainlit as cl
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
memory = ConversationBufferMemory(ai_prefix="AI", human_prefix="Human")

prompt = PromptTemplate(
    # input_variables=["history", "input"],
    template="""

    {history}
    Human: {input}
    AI:"""
)
conversation = ConversationChain(
    llm=hf_llm,
    memory=memory,
    prompt=prompt,
    verbose=False
)
@cl.on_chat_start
async def start():
    conversation.memory.clear()
    await cl.Message(content="🗨️ **Welcome to the AI Assistant!** How can I help you today?").send()

@cl.on_message
async def main(message: cl.Message):
    user_input = message.content
    print("User Input:", user_input)
    
    # Get the response from the conversation chain
    response = conversation.predict(input=user_input)
    print("Response Before Parsing:", response)
    
    # Strip unwanted text if present
    if "AI:" in response:
        response = response.split("AI:", 1)[1].strip()

    
    print("---------------Response after parsing---------------------------")
    await cl.Message(content=response).send()
    print("------------end of response --------------")
    print("Memory Buffer:", conversation.memory.buffer)

