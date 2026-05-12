from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mnf.experiments.run_real_model_smoke import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="sshleifer/tiny-gpt2")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(model_name=args.model_name, local_files_only=not args.allow_download), indent=2))


if __name__ == "__main__":
    main()
