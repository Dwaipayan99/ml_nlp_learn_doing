# import os
# from dotenv import load_dotenv
# from langchain_anthropic import ChatAnthropic



# load_dotenv()

# print("current working directory",os.getcwd())


# api_key = os.getenv('GRO_API_KEY')


# if api_key is None:
#     raise ValueError("KEY environment variable not set.")

# print("-------------------------------------------APIKEY-------------------------------------------")
# print(api_key)

# # Correct way to initialize ChatAnthropic with the API key
# model = ChatAnthropic(anthropic_api_key=api_key, model='llama-guard-3-8b')

# try:
#     response = model.invoke("Write a fibonacci series in java")

#     print("Full result:")
#     print(response)
#     print("content only")
#     print(response.content)
# except Exception as e:
#     print(f"An error occurred: {e}")


import os
import requests
from dotenv import load_dotenv

# api_key = os.environ.get("GRO_API_KEY")
load_dotenv()
print("current working directory",os.getcwd())
api_key = os.getenv('GRO_API_KEY')
print(api_key)
url = "https://api.groq.com/openai/v1/models"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

response = requests.get(url, headers=headers)

print(response.content)