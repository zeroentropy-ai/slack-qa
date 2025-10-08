import asyncio
import json
import turbopuffer
from tqdm.asyncio import tqdm

from ai import AIEmbeddingModel, AIEmbeddingType, ai_embedding

sem = asyncio.Semaphore(16)

tpuf = turbopuffer.AsyncTurbopuffer(
    api_key="tpuf_9QydWa7W9xdINPoLQ2Hgl7iLpvjvfgqE",
    region="gcp-us-west1",
)
ns = tpuf.namespace(f'pipitone-modal')

queries = []
with open("./synthetic_data/Modal_Community_T031JJZ7Q6T/beir_format_individual_message/queries.jsonl") as f:
    for line in f:
        if "{" not in line:
            continue
        query = json.loads(line)
        queries.append(query)

async def process_query(query) -> list[str]:
    async with sem:
        embedding = (await ai_embedding(
            model=AIEmbeddingModel(
                company="openai",
                model="text-embedding-3-large",
            ),
            texts=[
                query["query"],
            ],
            embedding_type=AIEmbeddingType.QUERY,
        ))[0]
        response = await ns.multi_query(
            queries=[
                {
                    "rank_by": ("vector", "ANN", embedding.tolist()),
                    "top_k": 100,
                    "include_attributes": ["content"],
                },
                {
                    "rank_by": ("content", "BM25", query["query"][:1024]),
                    "top_k": 100,
                    "include_attributes": ["content"],
                },
            ]
        )

    vector_result, fts_result = response.results[0].rows, response.results[1].rows

    def reciprocal_rank_fusion(result_lists, k = 60): # simple way to fuse results based on position
        scores = {}
        all_results = {}
        for results in result_lists:
            for rank, item in enumerate(results, start=1):
                scores[item.id] = scores.get(item.id, 0) + 1.0 / (k + rank)
                all_results[item.id] = item
        return [
            setattr(all_results[doc_id], '$dist', score) or all_results[doc_id]
            for doc_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ]

    fused_results = reciprocal_rank_fusion([vector_result, fts_result])

    return [item.id for item in fused_results]

async def main():
    all_results = await tqdm.gather(*[
        process_query(query)
        for query in queries
    ])
    with open("results.jsonl", "w") as f:
        for query, results in zip(queries, all_results):
            f.write(json.dumps({
                "query_id": query["id"],
                "document_ids": results,
            }) + "\n")

if __name__ == "__main__":
    asyncio.run(main())
