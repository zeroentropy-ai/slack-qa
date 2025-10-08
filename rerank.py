import asyncio
import json
from zeroentropy import AsyncZeroEntropy
from tqdm.asyncio import tqdm
from ai import AIRerankModel, ai_rerank

rerank_model = AIRerankModel(
    company="cohere",
    model="rerank-v3.5",
)

# Get docs and queries
docs = {}
with open("./synthetic_data/Modal_Community_T031JJZ7Q6T/beir_format_individual_message/documents.jsonl") as f:
    for line in f:
        if "{" not in line:
            continue
        doc = json.loads(line)
        docs[doc["id"]] = doc
queries = {}
with open("./synthetic_data/Modal_Community_T031JJZ7Q6T/beir_format_individual_message/queries.jsonl") as f:
    for line in f:
        if "{" not in line:
            continue
        query = json.loads(line)
        queries[query["id"]] = query

results = []
with open("results.jsonl") as f:
    for line in f:
        if "{" not in line:
            continue
        result = json.loads(line)
        results.append(result)

async def rerank_result(result):
    query_id = result["query_id"]

    document_ids = result["document_ids"]

    query = queries[query_id]
    documents = [
        docs[doc_id]
        for doc_id in document_ids
    ]

    rerank_scores = await ai_rerank(
        model=rerank_model,
        query=query["query"],
        texts=[
            doc["content"]
            for doc in documents
        ],
    )

    for document, score in zip(documents, rerank_scores):
        document["score"] = score
    documents.sort(key=lambda x: -x["score"])
    reranked_document_ids = [
        doc["id"]
        for doc in documents
    ]

    return {
        "query_id": query_id,
        "document_ids": reranked_document_ids,
    }

async def main():
    reranked_results = await tqdm.gather(*[
        rerank_result(result)
        for result in results
    ])
    with open(f"results_{rerank_model.model.replace('-', '_')}.jsonl", "w") as f:
        for result in reranked_results:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    asyncio.run(main())

