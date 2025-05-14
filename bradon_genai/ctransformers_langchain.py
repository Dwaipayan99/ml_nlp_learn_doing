import os
from ctransformers import AutoModelForCausalLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import CTransformers
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
print(os.getcwd())
hf_key = os.getenv('HF_TOKEN')
print(hf_key)

llm = CTransformers(
    model="TheBloke/Llama-2-7B-GGUF",  # Specify the model directory
    model_file="llama-2-7b.Q4_K_M.gguf",  # Specific GGUF file
    model_type="llama",  # Model type is LLaMA
    max_new_tokens=10000000,  # Limit on generated tokens
    temperature=0.7,  # Controls randomness of output
    top_p=0.9,  # Nucleus sampling
)

prompt = PromptTemplate(
    input_variables=["question"],
    template="""
    You are a helpful AI assistant. Answer the following question concisely and briefly, giving some background context where relevant and in text only.

   Question:{question}
   Answer:
"""
)
chain = prompt | llm

response = chain.invoke({"question": "how memory management is achieved using langchain  "})

print("\n🧠 AI Response:")
print(response)
