import json

K = 20

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
with open("results_zerank_1.jsonl") as f:
    for line in f:
        if "{" not in line:
            continue
        result = json.loads(line)
        results.append(result)

all_recalls = []
for result in results:
    query_qrels = qrels_by_query_id[result["query_id"]]
    retrieved_doc_ids = set(result["document_ids"][:K])
    total_score = sum(qrel["score"] for qrel in query_qrels)
    actual_score = sum(qrel["score"] for qrel in query_qrels if qrel["document_id"] in retrieved_doc_ids)
    all_recalls.append(actual_score / total_score)

print(f"Recall: {100*sum(all_recalls)/len(all_recalls):.2f}%")
