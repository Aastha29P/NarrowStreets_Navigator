import os
from dotenv import load_dotenv

load_dotenv()  # Loads variables from a .env file into the environment

openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
