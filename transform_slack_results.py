import json
import uuid
from collections import defaultdict

def generate_deterministic_uuid(text: str, timestamp: float) -> str:
    """Generate a deterministic UUID from text and timestamp."""
    combined = f"{text}:{timestamp}"
    namespace = uuid.NAMESPACE_DNS # arbitrary namespace
    return str(uuid.uuid5(namespace, combined))

def rrf(all_rankings: list[list[str]]) -> list[str]:
    doc_id_to_score: dict[str, float] = defaultdict(float)
    for ranking in all_rankings:
        for i, doc in enumerate(ranking):
            doc_id_to_score[doc] += 1 / (10 + i)
    doc_id_and_scores = list(doc_id_to_score.items())
    doc_id_and_scores.sort(key=lambda x: -x[1])
    return [
        doc_id
        for doc_id, score in doc_id_and_scores
    ]

message_id_to_document_id = {}
with open("./synthetic_data/Modal_Community_T031JJZ7Q6T/beir_format_individual_message/documents.jsonl") as f:
    for line in f:
        j = json.loads(line)
        message_id = j["metadata"]["message_id"]
        if message_id not in message_id_to_document_id:
            message_id_to_document_id[message_id] = set()
        message_id_to_document_id[message_id].add(j["id"])

with open("slack_raw_results.jsonl") as f, open("slack_results.jsonl", "w") as f_out:
    for line in f:
        j = json.loads(line)
        all_rankings = []
        for result in j["slack_results"]:
            ranking = []
            for match in result["matches"]:
                document_id = generate_deterministic_uuid(match["text"], float(match["ts"]))
                ranking.append(document_id)
            all_rankings.append(ranking)
        message_ids = rrf(all_rankings)
        document_ids_nested = [
            list(message_id_to_document_id.get(mid, set()))
            for mid in message_ids
        ]
        document_ids = [
            document_id
            for some_document_ids in document_ids_nested
            for document_id in some_document_ids
        ]
        f_out.write(json.dumps({
            "query_id": j["query_id"],
            "document_ids": document_ids,
        }) + "\n")
