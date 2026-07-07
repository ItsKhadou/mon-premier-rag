import csv

import config
from vector_db import VectorDB


def load_corpus(csv_path=config.CORPUS_CSV) -> list[dict]:
    with open(csv_path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


if __name__ == "__main__":
    chunks = load_corpus()
    print(f"{len(chunks)} chunks lus depuis {config.CORPUS_CSV}")
    VectorDB(chunks=chunks)
