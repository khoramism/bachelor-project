# Research Mode Quickstart

This repository includes a lightweight evaluation harness to assess verse retrieval quality. Use it to report nDCG, MRR, Recall, and latency trade-offs so professors can inspect concrete metrics.

## Running the evaluation
1. Install base and research dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements.research.txt
   ```
2. Run the harness on sample gold data:
   ```bash
   python eval/eval.py --queries data/gold/queries.jsonl --judgments data/gold/judgments.jsonl --corpus hafez.tsv
   ```

## Outputs
* `results/quality_summary.csv` – headline metrics per system.
* `results/per_query.json` – per-query scores for deeper analysis.
* Paste the key numbers into `RESULTS.md` after each run.

See [EXPERIMENTS.md](EXPERIMENTS.md) for research questions and methodology and [paper/mini-study.md](paper/mini-study.md) for a short report outline.
