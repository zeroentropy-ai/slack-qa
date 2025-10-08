import asyncio
import openai
import os
import httpx
from openai import OpenAI
from pydantic import BaseModel

class Output(BaseModel):
    keywords: list[str]

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

async def query_reduce_openai(query: str, model: str = 'gpt-5-mini', max_tokens: int = 2048) -> str:
    """
    Reduce a query to 2-3 keywords using OpenAI's chat completion API.
    
    Args:
        query: The original query string
    
    Returns:
        str: The reduced keyword string
    """
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY must be set in environment")
    
    client = OpenAI(api_key=api_key)

    try:
        response = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": "You will be given a query from the user. Your task is to output 2-3 keywords that best represent the query. A keyword must only be a single word."},
                {"role": "user", "content": query}
            ],
            max_completion_tokens=max_tokens,
            response_format=Output,
        )
        
        return ' '.join(response.choices[0].message.parsed.keywords)
    
    except openai.exceptions.AuthenticationError:
        return "Error: Invalid API key"
    except openai.exceptions.RateLimitError:
        return "Error: Rate limit exceeded"
    except openai.exceptions.APIError as e:
        return f"Error: API error - {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"


async def test_query_reduce():
    queries = [
        "What are the health benefits of drinking green tea?",
        "How does photosynthesis work in plants?",
        "What is the capital of France?",
        "Explain the theory of relativity in simple terms.",
        "What are the main causes of climate change?"
    ]
    
    for query in queries:
        reduced = await query_reduce_openai(query)
        print(f"Original: {query}\nReduced: {reduced}\n")

if __name__ == "__main__":
    asyncio.run(test_query_reduce())