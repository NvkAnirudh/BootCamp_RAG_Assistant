import openai
from typing import List, Dict, Optional
from backend.config import config

class OpenAIClient:
    def __init__(self, api_key: str, model: str = "o1-mini-2024-09-12"):
        self.api_key = api_key
        self.model = model
        openai.api_key = self.api_key

    def generate_response(
        self,
        system_prompt: str,
        user_input: str,
        context: str,
        temperature: float = 0.3,
        max_tokens: int = 512,
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
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {user_input}"},
            ]

            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Extract and return the generated response
            return response.choices[0].message["content"].strip()

        except Exception as e:
            print(f"Error generating response: {e}")
            return None