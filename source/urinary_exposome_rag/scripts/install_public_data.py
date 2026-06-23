from __future__ import annotations

import argparse
import shutil
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
PUBLIC_DATA_DIR = PROJECT_DIR / "public_data"
DATA_DIR = PROJECT_DIR / "data"


def copy_tree(source: Path, target: Path, overwrite: bool) -> None:
    if not source.exists():
        return
    if target.exists() and overwrite:
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        for path in source.rglob("*"):
            rel = path.relative_to(source)
            destination = target / rel
            if path.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
    else:
        shutil.copytree(source, target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Install bundled public data into the local data directory.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing data/effects and data/tables folders.")
    args = parser.parse_args()

    if not PUBLIC_DATA_DIR.exists():
        raise SystemExit(f"Public data directory not found: {PUBLIC_DATA_DIR}")

    copy_tree(PUBLIC_DATA_DIR / "data" / "effects", DATA_DIR / "effects", overwrite=args.overwrite)
    copy_tree(PUBLIC_DATA_DIR / "data" / "tables", DATA_DIR / "tables", overwrite=args.overwrite)

    print(f"Installed public data into {DATA_DIR}")
    print("Next step: build the Chroma index from rag_v2:")
    print(r"  cd source\urinary_exposome_rag\rag_v2")
    print(r"  .\.venv\Scripts\python.exe .\scripts\build_chroma_index.py --reset --batch-size 64")


if __name__ == "__main__":
    main()
