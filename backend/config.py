import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

class Config:
    PDF_DIR = "./data"
    CHROMA_DIR = "./chroma_db"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Set your OpenAI API key in environment variables
    OPENAI_MODEL = "o1-mini-2024-09-12"  # Use o1-mini for cost-effective responses
    MAX_CONTEXT_TOKENS = 2000  # Limit context size to manage API costs
    TEMPERATURE = 0.1  # Controls creativity of responses
    MAX_HISTORY = 3  # Number of conversation turns to keep in memory

config = Config()