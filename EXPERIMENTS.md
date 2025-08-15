# Experiments

## Research Questions
- **RQ1:** How does dense retrieval compare to BM25 in quality?
- **RQ2:** How does Qdrant's `ef` parameter trade off latency and quality?
- **RQ3:** What is the impact of an optional cross-encoder reranker?
- **RQ4:** What error categories emerge (concrete, thematic, paraphrase)?

## Metrics
We report nDCG@10, MRR@10, Recall@10 and efficiency (p50/p95/p99 latency, queries per second).

## Protocol
- Paired bootstrap over queries to obtain confidence intervals.
- Toggle between verse-only and verse+neighbor context when constructing the corpus.
- For reranking, apply a cross-encoder to the top-`k` dense results when available.

## Reproduction
Use the following commands or the accompanying Makefile targets:

```bash
# Baseline evaluation
make -f Makefile.research eval

# ANN sweep over different ef values
make -f Makefile.research ann-sweep
```
