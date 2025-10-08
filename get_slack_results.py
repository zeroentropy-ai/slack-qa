import asyncio
from dataclasses import asdict
import json
from tqdm.asyncio import tqdm

from slack_search import SlackSearch

sem = asyncio.Semaphore(1)

all_keywords = []
with open("keywords.jsonl") as f:
    for line in f:
        all_keywords.append(json.loads(line))

full_cookies = "cookies"
token = "token"
workspace_url = "https://modallabscommunity.slack.com"

search = SlackSearch(
    token=token,
    auth_mode='browser',
    cookies=full_cookies,
    workspace_url=workspace_url,
)

async def get_slack_result(f, keywords_line):
    query_id = keywords_line["query_id"]
    keywords = keywords_line["keywords"]

    async with sem:
        all_results = await asyncio.gather(*[
            search.search_async(keyword, search_type="messages")
            for keyword in keywords
        ])
        # await asyncio.sleep(0.1)
    f.write(json.dumps({
        "query_id": query_id,
        "slack_results": [asdict(d) for d in all_results],
    }) + "\n")


async def main():
    completed_query_ids = set()
    with open("slack_raw_results.jsonl") as f:
        for line in f:
            j = json.loads(line)
            completed_query_ids.add(j["query_id"])
    with open("slack_raw_results.jsonl", "a") as f:
        await tqdm.gather(*[
            get_slack_result(f, keywords_line)
            for keywords_line in all_keywords
            if keywords_line["query_id"] not in completed_query_ids
        ])

if __name__ == "__main__":
    asyncio.run(main())
