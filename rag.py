

import os

from dotenv import load_dotenv
from groq import Groq

import config
from moderator import ModeratorAgent
from vector_db import VectorDB


class RAG:
    def __init__(self):
        load_dotenv()  # charge GROQ_API_KEY depuis .env
        if not os.getenv("GROQ_API_KEY"):
            raise EnvironmentError("GROQ_API_KEY introuvable : crée un fichier .env (voir .env.example).")

        self.client = Groq()
        self.moderator = ModeratorAgent(self.client)
        self.vector_db = VectorDB()  # recharge la base persistée

        with open(config.RAG_PROMPT_FILE, encoding="utf-8") as f:
            self.prompt_template = f.read()

    # ------------------------------------------------------------------ #
    def _build_system_prompt(self, chunks: list[dict]) -> str:
        """Remplit le marqueur {{Chunks}} du prompt à trous avec les chunks."""
        chunks_text = "\n".join(
            f"{i}. [{c['source']}] {c['text']}" for i, c in enumerate(chunks, start=1)
        )
        return self.prompt_template.replace("{{Chunks}}", chunks_text)

    # ------------------------------------------------------------------ #
    def answer_question(self, question: str) -> str:
        # 1. Modération D'ABORD : si injection détectée, le LLM principal
        #    n'est JAMAIS contacté. L'ordre des opérations est une décision
        #    de sécurité (l'entrée hostile ne touche jamais le modèle principal).
        verdict = self.moderator.moderate(question)
        if verdict["is_prompt_injection"]:
            return (
                "⛔ Question refusée : tentative de détournement (prompt injection) "
                "détectée par l'agent modérateur."
            )

        # 2. Retrieval (une seule fois : je garde les chunks sous la main
        #    pour le prompt ET pour l'affichage des sources)
        chunks = self.vector_db.retrieve(question, n=config.N_RESULTS)

        # Bonus §7 — seuil de similarité : si même le meilleur chunk est
        # trop loin de la question, autant prévenir tout de suite.
        best_sim = chunks[0]["similarity"] if chunks else 0
        warning = ""
        if best_sim < config.SIMILARITY_THRESHOLD:
            warning = (
                f"⚠️ Similarité max = {best_sim} (seuil {config.SIMILARITY_THRESHOLD}) : "
                "la question semble hors du périmètre de ma base de connaissances.\n\n"
            )

        # 3. Prompt système à trous
        system_prompt = self._build_system_prompt(chunks)

        # 4. Appel au LLM de génération
        response = self.client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=0.2,
        )
        answer = response.choices[0].message.content

        # Bonus §7 — affichage des sources et scores de similarité
        sources = "\n".join(
            f"   • {c['source']} (sim={c['similarity']}) — {c['text']}" for c in chunks
        )
        return f"{warning}{answer}\n\n📚 Sources utilisées :\n{sources}"


if __name__ == "__main__":
    rag = RAG()
    print(rag.answer_question("Quelle est la couleur du chat de Bob ?"))
