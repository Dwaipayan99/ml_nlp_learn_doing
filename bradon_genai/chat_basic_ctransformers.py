import os
from ctransformers import AutoModelForCausalLM
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
print(os.getcwd())
hf_key = os.getenv('HF_TOKEN')
print(hf_key)

# Load the model
llm = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7B-GGUF",
    model_file="llama-2-7b.Q4_K_M.gguf"  # Specify the correct GGUF file
)

# Generate text
output = llm("amount of ram required to run Llama-2-7b")
print(output)
