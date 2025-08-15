import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--efs", nargs="+", type=int, default=[50, 150, 400])
    args = parser.parse_args()

    for ef in args.efs:
        cmd = [
            "python",
            "eval/eval.py",
            "--queries",
            "data/gold/queries.jsonl",
            "--judgments",
            "data/gold/judgments.jsonl",
            "--corpus",
            "hafez.tsv",
            "--use_qdrant",
            "--ef",
            str(ef),
        ]
        subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
