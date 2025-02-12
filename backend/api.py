from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from typing import List, Tuple, Dict, Any
from .hybrid_search import HybridSearchSystem
from .llm_integration import OpenAIClient
from .config import Config

app = FastAPI()
search_system = HybridSearchSystem(Config.PDF_DIR, Config.CHROMA_DIR)
llm_client = OpenAIClient(api_key=Config.OPENAI_API_KEY, model=Config.OPENAI_MODEL)

class ChatRequest(BaseModel):
    message: str
    history: List[Tuple[str, str]]  # List of (user_input, assistant_response) pairs 

def build_context(search_results: List[Dict[str, Any]]) -> str:
    """Build a context string from search results."""
    return "\n\n".join([f"Source: {res['source']}\nContent: {res['content']}" for res in search_results])

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.post("/chat")
# async def chat_endpoint(request: ChatRequest):
def chat_endpoint(request: ChatRequest):
    # try:
    # Step 1: Retrieve relevant context using RAG
    # search_results = await run_in_threadpool(search_system.search, request.message) 
    search_results = search_system.search(request.message)
    context = build_context(search_results)

    # Step 2: Generate system prompt
    system_prompt = """You are a helpful assistant. Answer the user's question based on the provided context.
    If the context does not contain enough information, say 'I don't know' and ask the user to clarify.
    Keep your responses concise and to the point."""

    # Step 3: Generate response using OpenAI API
    # response = await run_in_threadpool(
    #     llm_client.generate_response, 
    #     system_prompt=system_prompt, 
    #     user_input=request.message, 
    #     context=context
    # )
    response = llm_client.generate_response(
        system_prompt=system_prompt,
        user_input=request.message,
        context=context,
    )
    print(response)
    if not response:
        raise HTTPException(status_code=500, detail="Failed to generate response")

    # Step 4: Return response and context sources
    return {
        "response": response,
        "context_sources": [res["source"] for res in search_results],
    }

    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=str(e))