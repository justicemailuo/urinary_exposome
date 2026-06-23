from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query Urological Exposomics RAG v2 through FastAPI.")
    parser.add_argument("question", help="Natural-language question in Chinese or English.")
    parser.add_argument("--api", default="http://127.0.0.1:8890", help="FastAPI base URL.")
    parser.add_argument("--top-k", type=int, default=8, choices=range(1, 31), metavar="1-30")
    parser.add_argument("--no-llm", action="store_true", help="Return deterministic retrieval output.")
    parser.add_argument("--no-graph", action="store_true", help="Disable Neo4j path retrieval.")
    parser.add_argument("--lightrag", action="store_true", help="Include LightRAG context.")
    parser.add_argument("--source", choices=["all", "local_data", "literature", "effects", "fulltext", "abstract"], default="all")
    parser.add_argument("--exposure-domain", default="all")
    parser.add_argument("--disease-group", default="all")
    parser.add_argument("--json", action="store_true", help="Print the complete JSON response.")
    return parser.parse_args()


def payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "query": args.question,
        "top_k": args.top_k,
        "use_llm": not args.no_llm,
        "use_graph": not args.no_graph,
        "use_lightrag": args.lightrag,
        "filters": {
            "source": args.source,
            "exposure_domain": args.exposure_domain,
            "disease_group": args.disease_group,
            "effects_only": False,
            "table_only": False,
            "chinese_only": False,
        },
    }


def main() -> int:
    args = parse_args()
    try:
        response = requests.post(
            f"{args.api.rstrip('/')}/api/chat",
            json=payload(args),
            timeout=180,
        )
        response.raise_for_status()
        result = response.json()
    except Exception as error:
        print(f"RAG query failed: {error}", file=sys.stderr)
        print(f"Check that FastAPI is running at {args.api}.", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result.get("answer", ""))
        print(
            f"\nSources: {len(result.get('sources', []))}; "
            f"graph paths: {len(result.get('graph_paths', []))}; "
            f"elapsed: {result.get('elapsed_ms', 0)} ms",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
