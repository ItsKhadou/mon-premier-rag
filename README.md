# Mon premier RAG 🧠 — corpus de Villebrume-les-Cuillères

> Mini-TP guidé M2 MD5 — Khadidja B.

Un RAG minimal mais complet, reconstruit brique par brique à partir de la démo
en classe : **ChromaDB** (base vectorielle persistée) + **sentence-transformers**
(embeddings multilingues) + **Groq** (génération) + un **agent modérateur** qui
filtre les prompt injections *avant* que le LLM principal ne voie quoi que ce soit.

Le corpus est volontairement absurde (le chat bleu de Bob, les girouettes du
Clos...) : ces faits n'existent nulle part sur Internet, donc si le système
répond juste, c'est forcément grâce au retrieval — impossible de tricher avec
la mémoire du modèle.

## Comment c'est organisé

```
config.py                        → toutes les constantes à UN seul endroit
vector_db.py                     → Brique 1 : la base vectorielle persistante
moderator.py                     → Brique 2 : l'agent modérateur (JSON strict)
rag.py                           → Brique 3 : l'orchestration du pipeline
build_index.py                   → indexation du corpus (à lancer 1 fois)
main.py                          → tests de la section 6 + mode interactif
prompts/                         → les prompts système en .txt (hors du code)
data/05_corpus_rag.csv           → les 200 chunks
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # ma clé Groq va là-dedans, et nulle part ailleurs
```

`.env` et `chroma_db/` sont dans le `.gitignore` — vérifié avant le premier commit.

## Lancer

```bash
python build_index.py   # crée la base (les fois suivantes, elle se recharge toute seule)
python vector_db.py     # sanity check du retrieval seul, avant de brancher le LLM
python main.py          # pipeline complet + les 6 questions de test
```

## Ce que j'ai ajouté en plus du TP de base (section 7)

- **Sources + scores de similarité** affichés sous chaque réponse : on voit
  exactement quels chunks ont servi, et à quel point ils collaient.
- **Seuil de similarité** (`SIMILARITY_THRESHOLD = 0.35`) : si même le meilleur
  chunk est trop loin de la question, le système prévient que c'est
  probablement hors corpus. J'ai calibré le seuil en regardant les scores :
  bonnes correspondances ≈ 0.5–0.8, questions hors sujet < 0.35.
- **Deux questions de test à moi** dans `main.py` : Bob a deux chats (Henri le
  bleu, Casimir le noir), c'est le piège parfait pour vérifier que le retrieval
  ne mélange pas les deux.

## Ce que j'ai compris (et que je saurais refaire les yeux fermés)

**Le coup de la métadonnée d'embedding.** Le nom du modèle d'embedding est
stocké dans les métadonnées de la collection ChromaDB au moment de la création.
Au rechargement, on lit CE nom-là, pas celui de la config du jour. Sans ça, si
je change de modèle dans `config.py` et que je recharge une vieille base, mes
questions seraient encodées dans un espace vectoriel qui n'a rien à voir avec
celui du corpus : rien ne plante, mais le retrieval devient nul sans raison
apparente. C'est le pire genre de bug — silencieux.

**Pourquoi un modérateur séparé.** Écrire « refuse les injections » dans le
prompt du RAG, c'est confier la sécurité au composant qu'on attaque : le prompt
serait lui-même détournable par l'injection qu'il doit bloquer. Le modérateur
est un modèle dédié, appelé en premier, avec une sortie JSON contrainte — et
si sa décision est « injection », le LLM principal n'est jamais contacté.
L'ordre des opérations, c'est de la sécurité.

**Pourquoi normaliser les embeddings.** Avec des vecteurs normés, le produit
scalaire devient exactement la similarité cosinus — c'est ce qui rend les
distances comparables et le seuil calibrable.
