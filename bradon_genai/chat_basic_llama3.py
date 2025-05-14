import os 
from dotenv import load_dotenv
import torch
from transformers import pipeline

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_huggingface import HuggingFacePipeline

load_dotenv()
print(os.getcwd())
hf_key = os.getenv('HF_TOKEN')
print(hf_key)
print("----------------------------")
import torch
print(torch.backends.mps.is_available())  # Should print True
print(torch.backends.mps.is_built())  # Should print True


model_id = "meta-llama/Llama-3.2-3B"

# os.environ["TOKENIZERS_PARALLELISM"] = "true"
pipe = pipeline(
    "text-generation", 
    model=model_id, 
    torch_dtype=torch.bfloat16, 
    device_map= "mps" if torch.backends.mps.is_available() else 'auto',
    max_new_tokens=200,
    pad_token_id=128001
)

prompt = PromptTemplate(
    input_variables=["question"],
    template="""
        You are an AI assistant specializing in LangChain concepts. Provide a clear, concise answer to the following question, including relevant background information if necessary. 

        Question: {question}

        Answer:
        """
        )
hf_llm = HuggingFacePipeline(pipeline=pipe) 
chain = prompt | hf_llm

response = chain.invoke({"question": "how memory management is achieved using langchain  "})

# response = pipe("how much ram required cfor llama37b model to run on my systemmm")
# print(response[0]['generated_text'])
print(response)




# pipeline = transformers.pipeline("text-generation",model=model_id,model_kwargs={ "torch_dtype":torch.bfloat16},device_map="auto")
# pipeline("How are you doing today?")

# pipeline = transformers.pipeline("text-generation", 
#                     model=model_id,
#                     model_kwargs={"torch_dtype": torch.float32},
#                     device_map="mps",
#                     pad_token_id=50256,  # Set pad_token_id if needed
#                     max_new_tokens=50)  # Use max_new_tokens instead of max_length

# # Example text generation
# output = pipeline("where is kolkata and india. explain in 1000 words")
# print(output)


