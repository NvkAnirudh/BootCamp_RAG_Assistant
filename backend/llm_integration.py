import os
import openai
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Optional
from .config import config
import httpx

load_dotenv()

class CustomHTTPClient(httpx.Client):
    def __init__(self, *args, **kwargs):
        kwargs.pop("proxies", None)  # Remove the 'proxies' argument if present
        super().__init__(*args, **kwargs)

class OpenAIClient:
    def __init__(self, api_key: str, model: str = "o1-mini-2024-09-12"):
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), http_client=CustomHTTPClient())
        # openai.api_key = self.api_key

    def generate_response(
        self,
        system_prompt: str,
        user_input: str,
        context: str,
        temperature: float = 0.2,
        max_tokens: int = 2000,
    ) -> Optional[str]:
        """
        Generate a response using OpenAI's o1-mini API.
        
        Args:
            system_prompt (str): The system prompt to guide the model.
            user_input (str): The user's query.
            context (str): The retrieved context from the RAG system.
            temperature (float): Sampling temperature for creativity control.
            max_tokens (int): Maximum tokens to generate.
        
        Returns:
            Optional[str]: The generated response or None if an error occurs.
        """
        try:
            # Combine system prompt, context, and user input
            messages = [
                {"role": "user", "content": f"{system_prompt}\n\n{context}\n\nQuestion: {user_input}"},
            ]

            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=max_tokens,
            )
            print(response)
            # Extract and return the generated response
            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Error generating response: {e}")
            return None