import argparse
import json
from pathlib import Path

from .pipeline import DocSearch


def main():
    ap = argparse.ArgumentParser(prog="de-en-search",
                                 description="DE/EN Search — cross-lingual document search.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ig = sub.add_parser("ingest", help="Parse + embed + index a document.")
    ig.add_argument("file")
    ig.add_argument("--id", required=True, help="Logical doc id")
    ig.add_argument("--recreate", action="store_true",
                    help="Recreate the ES index")
    ig.add_argument("--no-es", action="store_true",
                    help="Disable Elasticsearch (use in-memory BM25 + FAISS)")

    se = sub.add_parser("search", help="Run a query against the indexed doc.")
    se.add_argument("query")
    se.add_argument("--top", type=int, default=20)
    se.add_argument("--no-es", action="store_true")

    a = ap.parse_args()
    use_es = not a.no_es

    if a.cmd == "ingest":
        ds = DocSearch(use_es=use_es)
        n = ds.ingest(Path(a.file), a.id, recreate=a.recreate)
        print(f"Indexed {n} chunks.")
    else:
        ds = DocSearch(use_es=use_es)
        ds.load()
        result = ds.search(a.query, top_pages=a.top)
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
