import asyncio
import os
import httpx


async def embed_openai(text: str) -> list[float]:
    """
    Embed a single string using OpenAI's embedding API.
    
    Args:
        text: The text to embed
        
    Returns:
        list[float]: The embedding vector
    """
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY must be set in environment")
    
    payload = {
        "input": text,
        "model": "text-embedding-3-large",
        "encoding_format": "float",
        "dimensions": 3072
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            url="https://api.openai.com/v1/embeddings",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
    
    response.raise_for_status()
    result = response.json()
    
    # Extract the embedding from the response
    data = result.get("data", [])
    if not data:
        raise ValueError("No embeddings returned from OpenAI API")
    
    return data[0]["embedding"]