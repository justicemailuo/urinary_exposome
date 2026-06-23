from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.graph_store import build_graph


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the deterministic UrologicalExpomics Neo4j graph.")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    result = build_graph(reset=args.reset, batch_size=args.batch_size, limit=args.limit)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
