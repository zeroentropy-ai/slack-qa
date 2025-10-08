import asyncio
import json

from pydantic import BaseModel
from tqdm.asyncio import tqdm

from ai import AIModel, AIMessage, ai_call, ai_rerank

rewrite_model = AIModel(
    company="openai",
    model="gpt-5-mini",
)

class Output(BaseModel):
    keywords: list[str]

queries = {}
with open("./synthetic_data/Modal_Community_T031JJZ7Q6T/beir_format_individual_message/queries.jsonl") as f:
    for line in f:
        if "{" not in line:
            continue
        query = json.loads(line)
        queries[query["id"]] = query

async def process_query(query):
    rerank_scores = await ai_call(
        model=rewrite_model,
        messages=[
            AIMessage(
                role="system",
                content="You will be given a query from the user. Your task is to output 2-3 keywords that best represent the query. A keyword must only be a single word.",
            ),
            AIMessage(
                role="user",
                content=query["query"],
            ),
        ],
        response_format=Output,
    )
    return {
        "query_id": query["id"],
        "keywords": rerank_scores.keywords,
    }

async def main():
    all_keywords = await tqdm.gather(*[
        process_query(query)
        for query in queries.values()
    ])
    with open("keywords.jsonl", "w") as f:
        for result in all_keywords:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
