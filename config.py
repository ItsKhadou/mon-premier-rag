

from pathlib import Path

# --- Chemins ---
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PROMPTS_DIR = BASE_DIR / "prompts"
CHROMA_DB_PATH = str(BASE_DIR / "chroma_db")

CORPUS_CSV = DATA_DIR / "05_corpus_rag.csv"
RAG_PROMPT_FILE = PROMPTS_DIR / "rag_system_prompt.txt"
MODERATOR_PROMPT_FILE = PROMPTS_DIR / "moderator_system_prompt.txt"

# --- Base vectorielle ---
COLLECTION_NAME = "corpus_villebrume"
EMBEDDING_MODEL = "distiluse-base-multilingual-cased-v2"

# --- LLMs (Groq) ---
LLM_MODEL = "llama-3.3-70b-versatile"
# Modèle de la famille "safeguard" pour la modération.
# Vérifie le nom exact dans le catalogue console.groq.com/docs/models
MODERATION_MODEL = "openai/gpt-oss-safeguard-20b"

# --- Retrieval ---
N_RESULTS = 3

# Seuil de similarité cosinus (entre 0 et 1) : en dessous du seuil,
# on prévient l'utilisateur que la question est probablement hors corpus.
# Calibré empiriquement : les bonnes correspondances sur ce corpus tournent
# autour de 0.5-0.8, les questions hors sujet tombent en dessous de 0.35.
SIMILARITY_THRESHOLD = 0.35
