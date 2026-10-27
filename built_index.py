
import json
from pathlib import Path

from rag_pipeline import build_index

BASE_DIR = Path(__file__).resolve().parent
CHUNKS_PATH = BASE_DIR / "output" / "chunks.json"


def main():
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            "chunks.json not found. Run ingestion.py first."
        )

    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    print(f"Loaded {len(chunks)} chunks.")

    collection = build_index(chunks)

    print("Index created successfully!")
    print(f"Chunks stored: {collection.count()}")


if __name__ == "__main__":
    main()
