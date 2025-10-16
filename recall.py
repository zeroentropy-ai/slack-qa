import json
import sys
from garbage import CHANNEL_ACTIVITY, TICKET_ACTIVITY, HELP_US_HELP

if len(sys.argv) != 2:
    print("Usage: python recall.py <results_file.jsonl>")
    sys.exit(1)

K = 20
results_file = sys.argv[1]

# Combine all garbage document IDs
GARBAGE_DOC_IDS = set(CHANNEL_ACTIVITY + TICKET_ACTIVITY + HELP_US_HELP)

qrels_by_query_id = {}
with open("./synthetic_data/Modal_Community_T031JJZ7Q6T/beir_format_individual_message/qrels.jsonl") as f:
    for line in f:
        if "{" not in line:
            continue
        qrel = json.loads(line)
        if qrel["query_id"] not in qrels_by_query_id:
            qrels_by_query_id[qrel["query_id"]] = []
        qrels_by_query_id[qrel["query_id"]].append(qrel)

results = []
with open(results_file) as f:
    for line in f:
        if "{" not in line:
            continue
        result = json.loads(line)
        results.append(result)

all_recalls = []
total_queries = len(results)
excluded_queries = 0

for result in results:
    query_qrels = qrels_by_query_id[result["query_id"]]
    
    # Check if any target document is in garbage list
    target_doc_ids = [qrel["document_id"] for qrel in query_qrels]
    if any(doc_id in GARBAGE_DOC_IDS for doc_id in target_doc_ids):
        excluded_queries += 1
        continue
    
    retrieved_doc_ids = set(result["document_ids"][:K])
    total_score = sum(qrel["score"] for qrel in query_qrels)
    actual_score = sum(qrel["score"] for qrel in query_qrels if qrel["document_id"] in retrieved_doc_ids)
    all_recalls.append(actual_score / total_score)

used_queries = len(all_recalls)
print(f"Total queries: {total_queries}")
print(f"Excluded queries: {excluded_queries}")
print(f"Used queries: {used_queries}")
print(f"Recall@{K}: {100*sum(all_recalls)/len(all_recalls):.2f}% ({sum(all_recalls)}/{len(all_recalls)})")
