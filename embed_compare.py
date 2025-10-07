import turbopuffer
from turbopuffer.types import Row, ID
from openai import embeddings
import asyncio
import os
import sys
from tqdm import tqdm
import numpy as np
from typing import List
from ai import ai_call, AIModel, AIMessage, ai_embedding, AIEmbeddingModel, AIEmbeddingType
from turbopuffer import NotFoundError


# ==============================================
# STEP 1: Initialize the Turbopuffer client/namespace
# ==============================================

tpuf = turbopuffer.Turbopuffer(
    # API tokens are created in the dashboard: https://turbopuffer.com/dashboard
    api_key=os.getenv("TURBOPUFFER_API_KEY"),
    # Pick the right region: https://turbopuffer.com/docs/regions
    region="gcp-us-west1",
)

ns = tpuf.namespace("embed-compare")
try:
    ns.delete_all()
except NotFoundError:
    print("Empty or no namespace found")

print("Connected to the puffer")

def async openai_embed_vector(text: str) -> List[float]:
    assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY not set"
    return embeddings.create(model="text-embedding-3-small",input=text).data[0].embedding


async def main():

    # ==============================================
    # Add document embeddings to Turbopuffer
    # ==============================================

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) if len(sys.argv) < 2 else sys.argv[1]

    print("Getting texts...")

    ids = []
    texts = []
    with open(os.path.join(ROOT_DIR, "documents.jsonl"), "r") as f:
        for Line in f:
            line = Line.strip()
            if line:
                doc = __import__('json').loads(line)
                id, content = doc['id'], doc['content']
                ids.append(id); texts.append(content)

    print("Got texts, generating embeddings...")

    pbar = tqdm(
        desc="Document Embeddings",
        total=len(texts),
    )
    query_embeddings = np.array(
        await ai_embedding(
            AIEmbeddingModel(
                company="openai",
                model="text-embedding-3-small",
            ),
            texts=texts,
            embedding_type=AIEmbeddingType.QUERY,
            callback=lambda: pbar.update(1),
        )
    )
    pbar.close()

    # generate embeddings async
    upsert_rows = [{'id' : id, 'vector': vector.tolist(), 'content': content} for id, vector, content in zip(ids, query_embeddings, texts)]
    print(len(upsert_rows))
    ns.write(upsert_rows=upsert_rows,
            distance_metric="cosine_distance",
            schema={ "content": { "type": "string", "full_text_search": True } }
    )
        
    print("INSERTED THE DOCUMENTS")

    query = "does anyone have questions about laravel passport?"
    response = ns.multi_query(
        queries=[
            {
                "rank_by": ("vector", "ANN", openai_embed_vector(query)),
                "top_k": 10,
                "include_attributes": ["content"]
            },
            {   "rank_by": ("content", "BM25", query),
                "top_k": 10,
                "include_attributes": ["content"]
            },
        ]
    )

    vector_results, fts_results = response.results[0].rows, response.results[1].rows
    print("Vector Search Results:", [item.content for item in vector_results], sep='\n\n')
    print('\n\n')
    print("FTS Results:", [item.content for item in fts_results], sep='\n\n')


if __name__ == "__main__":
    asyncio.run(main())