import numpy as np


def ndcg_at_k(relevances, k):
    relevances = np.asarray(relevances)[:k]
    if not np.any(relevances):
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, relevances.size + 2))
    dcg = np.sum((2 ** relevances - 1) * discounts)
    ideal = np.sort(relevances)[::-1]
    idcg = np.sum((2 ** ideal - 1) * discounts)
    return float(dcg / idcg) if idcg > 0 else 0.0


def mrr_at_k(hits, k):
    hits = np.asarray(hits)[:k]
    ranks = np.where(hits)[0]
    return 1.0 / (ranks[0] + 1) if ranks.size else 0.0


def recall_at_k(hits, total_relevant, k):
    hits = np.asarray(hits)[:k]
    return float(hits.sum() / total_relevant) if total_relevant else 0.0


def paired_bootstrap(metric_list_A, metric_list_B, n_samples=1000):
    rng = np.random.default_rng(0)
    metrics_A = np.asarray(metric_list_A)
    metrics_B = np.asarray(metric_list_B)
    deltas = []
    for _ in range(n_samples):
        idx = rng.integers(0, len(metrics_A), len(metrics_A))
        deltas.append(metrics_A[idx].mean() - metrics_B[idx].mean())
    deltas = np.sort(deltas)
    delta = metrics_A.mean() - metrics_B.mean()
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return delta, [float(lo), float(hi)]
