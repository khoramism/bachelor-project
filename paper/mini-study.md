# Mini Study

## Abstract
This mini-study evaluates verse retrieval methods for Hafez's poetry. We compare BM25, dense embeddings, and optional reranking, reporting effectiveness and latency metrics. Results demonstrate how advanced embeddings improve thematic and paraphrase retrieval. The study aims to guide future research on Persian poetry search.

## Introduction
Retrieving relevant verses in classical poetry is challenging due to metaphor, archaic language, and short context windows. A principled evaluation helps illustrate these issues.

## Methods
- **Corpus:** Hafez verses provided in `hafez.tsv`.
- **Systems:** BM25 baseline, dense retrieval via SentenceTransformer, optional Qdrant ANN and cross-encoder reranker.
- **Metrics:** nDCG@10, MRR@10, Recall@10, latency percentiles.

## Results
Paste tables from `RESULTS.md` here and include generated plots.

## Error Analysis
Common error sources include metaphoric mismatch, archaic vocabulary, and polysemy across verses.

## Limitations and Ethics
The small gold dataset and reliance on pretrained models may introduce bias. Use results cautiously for cultural interpretation.

## Conclusion
Dense embeddings enhance verse retrieval quality, especially for thematic queries, but further annotation and model tuning are needed for robust deployment.
