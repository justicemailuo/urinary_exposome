from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.vector_store import build_index  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Chroma vector index with bge-m3 embeddings.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Index only the first N documents for a smoke test.")
    args = parser.parse_args()

    result = build_index(batch_size=args.batch_size, reset=args.reset, limit=args.limit)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
