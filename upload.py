import asyncio
import json

from ai import AIEmbeddingModel, AIEmbeddingType, ai_embedding
from tqdm import tqdm
import turbopuffer

tpuf = turbopuffer.Turbopuffer(
    api_key="tpuf_9QydWa7W9xdINPoLQ2Hgl7iLpvjvfgqE",
    region="gcp-us-west1",
)

docs = []
with open("./synthetic_data/Modal_Community_T031JJZ7Q6T/beir_format_individual_message/documents.jsonl") as f:
    for line in f:
        if "{" not in line:
            continue
        doc = json.loads(line)
        docs.append(doc)

async def main() -> None:
    pbar = tqdm(total=len(docs), desc="Embeddings")
    embeddings = await ai_embedding(
        model=AIEmbeddingModel(
            company="openai",
            model="text-embedding-3-large",
        ),
        texts=[
            doc["content"]
            for doc in docs
        ],
        embedding_type=AIEmbeddingType.DOCUMENT,
        callback=lambda: pbar.update(1),
    )
    pbar.close()

    ns = tpuf.namespace(f'pipitone-modal')
    BATCH_SIZE = 1024
    for i in tqdm(list(range(0, len(docs), BATCH_SIZE)), desc="Tpuf Batches"):
        batch_docs = docs[i:i+BATCH_SIZE]
        batch_embeddings = embeddings[i:i+BATCH_SIZE]
        ns.write(
            upsert_rows=[
                {
                    "id": doc["id"],
                    "vector": embedding.tolist(),
                    "content": doc["content"],
                }
                for doc, embedding in zip(batch_docs, batch_embeddings)
            ],
            distance_metric="cosine_distance",
            schema={ "content": { "type": "string", "full_text_search": True } }
        )

    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
