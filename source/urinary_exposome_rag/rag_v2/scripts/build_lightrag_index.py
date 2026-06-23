from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.lightrag_bridge import index_lightrag


async def run() -> None:
    parser = argparse.ArgumentParser(description="Build the optional LightRAG entity/relation index.")
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    result = await index_lightrag(batch_size=args.batch_size, limit=args.limit)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(run())
