
import chromadb
from sentence_transformers import SentenceTransformer

import config


class VectorDB:
    def __init__(self, db_path: str = config.CHROMA_DB_PATH, chunks: list[dict] | None = None):
        """
        chunks : liste de dicts {"id": ..., "text": ..., "source": ..., "categorie": ...}
                 (nécessaire uniquement à la création)
        """
        # Client persistant : les données survivent à l'arrêt du programme
        self.client = chromadb.PersistentClient(path=db_path)

        existing = [c.name for c in self.client.list_collections()]

        if config.COLLECTION_NAME in existing:
            # --- Cas 1 : la base existe déjà -> rechargement ---
            self._load_existing()
        elif chunks:
            # --- Cas 2 : pas de base mais des chunks -> création ---
            self._create(chunks)
        else:
            # --- Cas 3 : rien -> erreur explicite ---
            raise ValueError(
                f"Aucune base trouvée dans '{db_path}' et aucun chunk fourni. "
                "Lance d'abord build_index.py pour créer l'index."
            )

    # ------------------------------------------------------------------ #
    def _create(self, chunks: list[dict]) -> None:
        print(f"[VectorDB] Création de la base avec {len(chunks)} chunks...")

        self.embedding_model_name = config.EMBEDDING_MODEL
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # Le détail malin : on stocke le nom du modèle d'embedding dans les
        # métadonnées de la collection. Au rechargement on relira CE nom,
        # ce qui rend impossible le bug silencieux "question encodée avec un
        # modèle différent de celui du corpus" (les distances n'auraient
        # aucun sens et le retrieval serait mauvais sans jamais planter).
        self.collection = self.client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            metadata={
                "embedding_model": self.embedding_model_name,
                "hnsw:space": "cosine",  # distance cosinus au lieu de L2 (defaut Chroma)
            },
        )

        texts = [c["text"] for c in chunks]
        embeddings = self._encode(texts, show_progress=True)

        self.collection.add(
            ids=[c["id"] for c in chunks],
            documents=texts,
            embeddings=embeddings,
            metadatas=[
                {"source": c["source"], "categorie": c.get("categorie", "")}
                for c in chunks
            ],
        )
        print("[VectorDB] Base créée et persistée.")

    # ------------------------------------------------------------------ #
    def _load_existing(self) -> None:
        self.collection = self.client.get_collection(name=config.COLLECTION_NAME)

        # On lit le modèle d'embedding DANS les métadonnées de la collection,
        # pas dans la config du jour (qui a pu changer entre-temps).
        meta = self.collection.metadata or {}
        self.embedding_model_name = meta.get("embedding_model", config.EMBEDDING_MODEL)
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        print(
            f"[VectorDB] Base rechargée ({self.collection.count()} chunks, "
            f"modèle: {self.embedding_model_name}). Aucune réindexation."
        )

    # ------------------------------------------------------------------ #
    def _encode(self, texts: list[str], show_progress: bool = False):
        """Encodage unique pour le corpus ET les questions (même modèle, même normalisation)."""
        return self.embedding_model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,   # vecteurs normés -> produit scalaire == similarité cosinus
            show_progress_bar=show_progress,
        ).tolist()

    # ------------------------------------------------------------------ #
    def retrieve(self, question: str, n: int = config.N_RESULTS) -> list[dict]:
        """Retourne les n chunks les plus proches, triés du plus au moins pertinent."""
        query_embedding = self._encode([question])

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            # ChromaDB renvoie une distance (plus c'est petit, mieux c'est).
            # Je la convertis en score de similarité (plus c'est grand, mieux
            # c'est) : plus lisible pour l'affichage des sources.
            chunks.append(
                {
                    "text": doc,
                    "source": meta.get("source", "?"),
                    "similarity": round(1 - dist, 3),
                }
            )
        return chunks


# --- Test rapide du retrieval (section 3.3 du TP) ---
if __name__ == "__main__":
    db = VectorDB()
    questions_test = [
        "Quelle est la couleur du chat de Bob ?",
        "Comment s'appelle le chien d'Alice ?",
        "Combien d'habitants compte Villebrume-les-Cuillères ?",
        "Que fait un plieur de brume ?",
        "À quelle heure pleut-il à La Ravine-Tiède ?",
    ]
    for q in questions_test:
        print(f"\n=== {q}")
        for c in db.retrieve(q):
            print(f"  [sim={c['similarity']}] ({c['source']}) {c['text']}")