"""Minimal command-line interface for querying an existing CodeRAG index."""

import argparse
import logging
import textwrap
from typing import List

from coderag.search import search_code


def _format_result(result: dict, index: int) -> str:
    snippet = textwrap.shorten(
        result.get("content", "").replace("\n", " "), width=200, placeholder="..."
    )
    return (
        f"{index}. {result.get('filename')} ({result.get('filepath')})\n"
        f"   similarity={result.get('distance', 0.0):.3f}\n"
        f"   {snippet}"
    )


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Query a local CodeRAG FAISS index without the Streamlit UI."
    )
    parser.add_argument("query", help="Text to search for in the indexed codebase.")
    parser.add_argument(
        "-k",
        type=int,
        default=5,
        help="Maximum number of matches to display (defaults to 5).",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity for debugging issues.",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level))

    results = search_code(args.query, k=args.k)
    if not results:
        print("No results found; ensure the FAISS index exists and contains data.")
        return 1

    for idx, item in enumerate(results, start=1):
        print(_format_result(item, idx))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
