import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from metrics import ndcg_at_k, mrr_at_k, recall_at_k

try:  # optional imports
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
except Exception:  # pragma: no cover - qdrant optional
    QdrantClient = None

try:
    from sentence_transformers import CrossEncoder
except Exception:  # pragma: no cover - rerank optional
    CrossEncoder = None


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def load_corpus(path):
    texts, ids = [], []
    if path.endswith(".tsv"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                doc_id, text = line.rstrip().split("\t", 1)
                ids.append(doc_id)
                texts.append(text)
    else:
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                ids.append(f"v_{i:05d}")
                texts.append(line.rstrip())
    return ids, texts


def build_qdrant_index(vectors, collection="hafez", ef=100):
    client = QdrantClient(":memory:")
    dim = vectors.shape[1]
    client.recreate_collection(collection_name=collection, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
    points = [PointStruct(id=i, vector=v) for i, v in enumerate(vectors.tolist())]
    client.upload_points(collection, points)
    return client


def bm25_search(corpus_texts, queries, k):
    bm25 = BM25Okapi([t.split() for t in corpus_texts])
    results = {}
    for q in queries:
        scores = bm25.get_scores(q["text"].split())
        top_idx = np.argsort(scores)[::-1][:k]
        results[q["qid"]] = top_idx
    return results


def dense_search(ids, texts, queries, k, use_qdrant=False, ef=100, rerank=False):
    try:
        model = SentenceTransformer("heydariAI/persian-embeddings")
        corpus_vecs = model.encode(texts, show_progress_bar=False)
        encode_query = lambda x: model.encode(x, show_progress_bar=False)
    except Exception:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer().fit(texts)
        corpus_vecs = vectorizer.transform(texts).toarray()
        encode_query = lambda x: vectorizer.transform([x]).toarray()[0]
    if use_qdrant and QdrantClient is not None:
        client = build_qdrant_index(corpus_vecs, ef=ef)

    results = {}
    for q in queries:
        q_vec = encode_query(q["text"])
        if use_qdrant and QdrantClient is not None:
            search = client.search("hafez", q_vec, limit=k, search_params={"ef": ef})
            top_idx = [r.id for r in search]
        else:
            scores = corpus_vecs @ q_vec
            top_idx = np.argsort(scores)[::-1][:k]
        if rerank and CrossEncoder is not None:
            ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            pair_texts = [[q["text"], texts[i]] for i in top_idx]
            ce_scores = ce.predict(pair_texts)
            top_idx = [x for _, x in sorted(zip(ce_scores, top_idx), reverse=True)][:k]
        results[q["qid"]] = top_idx
    return results


def evaluate(run_name, ids, ranked, judgments, k):
    rows = []
    for qid, idxs in ranked.items():
        rel_dict = judgments.get(qid, {})
        doc_ids = [ids[i] for i in idxs]
        relevances = [rel_dict.get(doc, 0) for doc in doc_ids]
        hits = [1 if r >= 2 else 0 for r in relevances]
        total_rel = sum(1 for r in rel_dict.values() if r >= 2)
        rows.append({
            "qid": qid,
            "ndcg": ndcg_at_k(relevances, k),
            "mrr": mrr_at_k(hits, k),
            "recall": recall_at_k(hits, total_rel, k),
        })
    df = pd.DataFrame(rows)
    return {
        "run": run_name,
        "ndcg": df["ndcg"].mean(),
        "mrr": df["mrr"].mean(),
        "recall": df["recall"].mean(),
        "per_query": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", required=True)
    parser.add_argument("--judgments", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--use_qdrant", action="store_true")
    parser.add_argument("--ef", type=int, default=100)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    queries = list(load_jsonl(args.queries))
    judgments = defaultdict(dict)
    for j in load_jsonl(args.judgments):
        judgments[j["qid"]][j["doc_id"]] = j["rel"]

    ids, texts = load_corpus(args.corpus)
    k = args.k

    bm25_res = bm25_search(texts, queries, k)
    dense_res = dense_search(ids, texts, queries, k, args.use_qdrant, args.ef, args.rerank)

    os.makedirs("results", exist_ok=True)
    summaries = []
    per_query_all = {}
    for name, res in [
        ("bm25", bm25_res),
        ("dense" + ("+rerank" if args.rerank else ""), dense_res),
    ]:
        eval_res = evaluate(name, ids, res, judgments, k)
        summaries.append({"system": name, "ndcg": eval_res["ndcg"], "mrr": eval_res["mrr"], "recall": eval_res["recall"]})
        per_query_all[name] = eval_res["per_query"]

    with open("results/per_query.json", "w", encoding="utf-8") as f:
        json.dump(per_query_all, f, ensure_ascii=False, indent=2)

    with open("results/quality_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["system", "ndcg", "mrr", "recall"])
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)


if __name__ == "__main__":
    main()
