# Evaluation Results

## Overall Quality
| System           | nDCG@10 | MRR@10 | Recall@10 |
|------------------|---------|--------|-----------|
| BM25             |         |        |           |
| Dense            |         |        |           |
| Dense+Rerank     |         |        |           |

## Bucket Analysis
| Bucket      | BM25 nDCG@10 | Dense nDCG@10 | Dense+Rerank nDCG@10 |
|-------------|--------------|---------------|----------------------|
| Concrete    |              |               |                      |
| Thematic    |              |               |                      |
| Paraphrase  |              |               |                      |

## Efficiency
| System       | p50 (ms) | p95 (ms) | p99 (ms) | QPS |
|--------------|----------|----------|----------|-----|
| BM25         |          |          |          |     |
| Dense        |          |          |          |     |
| Dense+Rerank |          |          |          |     |

![Latency vs Quality](results/latency_quality.png)

![nDCG by Bucket](results/bucket_ndcg.png)
