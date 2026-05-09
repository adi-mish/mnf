from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mnf.experiments.run_tiny_transformer_demo import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--modulus", type=int, default=7)
    args = parser.parse_args()
    seeds = tuple(range(args.n_seeds))
    print(json.dumps(run(seeds=seeds, modulus=args.modulus, steps=args.steps), indent=2))


if __name__ == "__main__":
    main()
