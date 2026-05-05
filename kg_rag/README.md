# KG-RAG : Knowledge Graph RAG Multi-Graph Multi-Agent

Système de retrieval augmenté par graphes de connaissances pour documents juridiques francophones, avec résolveur hybride de références et workflow multi-agents.

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                       PHASE 0 – PRÉTRAITEMENT                      │
│   Chargement JSON → normalisation → graphe hiérarchique G_lex     │
│   + index des articles + extraction définitions + G_def          │
└────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────┐
│                 PHASE 1 – RÉSOLUTION DE RÉFÉRENCES                 │
│   Étage 1 : Regex (articles, plages, annexes, footnotes…)        │
│   Étage 2 : Heuristiques graphe (anaphores : ci-dessus, etc.)    │
│   Étage 3 : Recherche externe ChromaDB                            │
│   Étage 4 : Fallback LLM (Mistral/GPT-4o, avec cache + triggers) │
└────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────┐
│               PHASE 2 – EXPANSION SÉMANTIQUE                      │
│   Graphe G_ref → expansion récursive (profondeur limitée)         │
│   + injection définitions + troncature textes longs               │
└────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────┐
│             PHASE 3 – INDEXATION VECTORIELLE                       │
│   Chunks enrichis → embeddings e5-large → ChromaDB (3 collections)│
│   + BM25 + Hybrid Retriever (Vector 60% + BM25 40%)              │
└────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────┐
│           PHASE 4 – RETRIEVAL MULTI‑AGENTS (KG‑RAG)                │
│   LangGraph : Initial Search → Definition → Router (heuristique)  │
│   → {RECURSE → Recursive → Supervisor → Router} | STOP → Answer │
│   + External Law Agent (graceful disable) + Supervisor            │
└────────────────────────────────────────────────────────────────────┘
```

## Structure du projet

```
kg_rag/
├── __init__.py               # Package
├── __main__.py               # python -m kg_rag
├── config.py                 # Configuration centralisée
├── requirements.txt          # Dépendances
├── document_loader.py        # Phase 0 – Chargement, G_lex, G_def, définitions
├── reference_resolver.py     # Phase 1 – Résolveur hybride 4 étages
├── expansion.py              # Phase 2 – G_ref, expansion, chunking
├── vector_indexer.py         # Phase 3 – ChromaDB, embeddings, BM25, hybride
├── agents.py                 # Phase 4 – 7 agents LangGraph
├── workflow.py               # Phase 4 – Orchestration StateGraph
├── main.py                   # CLI + pipeline complet
├── validate_index.py         # Script validation index ChromaDB
├── tests/                    # Tests unitaires
│   ├── __init__.py
│   ├── test_resolver.py      # Tests du résolveur
│   └── test_expansion.py     # Tests de l'expansion
└── chroma_db/                # Données persistantes ChromaDB (auto-créé)
```

## Installation

```bash
# 1. Installer les dépendances
pip install -r kg_rag/requirements.txt

# 2. (Optionnel) Installer Ollama pour le LLM local
#    https://ollama.ai
ollama pull mistral:7b-instruct

# 3. (Optionnel) Configurer les variables d'environnement
cp .env.example .env
# Éditer .env avec vos clés API si nécessaire
```

## Utilisation

### Construction du pipeline (Phases 0-3, offline)

```bash
# Construction complète (chargement + résolution + expansion + indexation)
python -m kg_rag --build

# Construction sans indexation vectorielle (plus rapide pour tester)
python -m kg_rag --build --skip-indexing

# Afficher les statistiques
python -m kg_rag --build --stats
```

### Interrogation (Phase 4, online)

```bash
# Requête unique
python -m kg_rag --build --query "Quelles sont les conditions d'obtention du permis de chasse ?"

# Mode interactif
python -m kg_rag --build --interactive

# Sans fallback LLM (plus rapide, déterministe)
python -m kg_rag --build --no-llm-fallback --interactive
```

### Validation de l'index

```bash
python -m kg_rag.validate_index --validate
```

### Tests unitaires

```bash
python -m pytest kg_rag/tests/ -v
```

## Exemple bout en bout

```python
from kg_rag.main import KGRAGPipeline

# 1. Construire le pipeline
pipeline = KGRAGPipeline(skip_indexing=False, use_llm_fallback=True)
pipeline.build()

# 2. Poser une question
result = pipeline.query("Comment le Board et le CCO gèrent-ils les fonctions de contrôle ?")

# 3. Lire la réponse
print(result["final_answer"])

# 4. Inspecter le processus
print(f"Passes : {result['pass_count']}")
print(f"Noeuds visités : {result['retrieved_graph_nodes']}")
print(f"Définitions : {result['definitions']}")
print(f"Échecs : {result['failures']}")
```

## Configuration

Tous les paramètres sont dans `config.py` et surchargeables via variables d'environnement :

| Variable | Défaut | Description |
|----------|--------|-------------|
| `LLM_PROVIDER` | `ollama` | `ollama` ou `openai` |
| `OLLAMA_MODEL` | `mistral:7b-instruct` | Modèle Ollama |
| `OPENAI_MODEL` | `gpt-4o` | Modèle OpenAI |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-large` | Modèle d'embedding |
| `CHUNK_SIZE` | `500` | Taille des chunks (tokens) |
| `MAX_EXPANSION_DEPTH` | `3` | Profondeur max expansion récursive |
| `MAX_AGENT_PASSES` | `5` | Max passes du workflow multi-agents |
| `DATA_DIR` | `../data_old` | Répertoire des JSON |

## Graphes de connaissances

- **G_lex** (lexical) : graphe hiérarchique des clauses (TITRE → Chapitre → Article → Alinéa), avec arêtes parent-enfant
- **G_ref** (références) : graphe des renvois croisés entre clauses (arêtes internes + externes)
- **G_def** (définitions) : graphe des termes juridiques et leurs définitions

## Agents

| Agent | Rôle |
|-------|------|
| **Initial Search** | Recherche vectorielle + BM25 hybride |
| **Definition** | Injection des définitions juridiques pertinentes |
| **Router** | Heuristique regex par défaut, LLM si activé → STOP/RECURSE/EXTERNAL |
| **Recursive Retrieval** | Traversée du graphe G_ref pour les clauses référencées |
| **External Law** | Recherche dans l'index externe (graceful disable si vide) |
| **Supervisor** | Contrôle des passes, élagage contexte, détection échecs |
| **Answering** | Synthèse avec citations obligatoires des sources |

## Limitations connues

- Les références à des lois externes non présentes dans le corpus ne peuvent pas être résolues (bandeau `[Référence externe non disponible]`)
- Si deux articles citent la même référence, le texte sera dupliqué dans les deux expansions (acceptable pour la vectorisation)
- Le fallback LLM est limité aux clauses contenant des expressions déclencheuses pour des raisons de performance
- La résolution des anaphores repose sur des heuristiques déterministes qui peuvent ne pas couvrir tous les cas

## Métriques d'évaluation suggérées

- **Rappel/Précision du résolveur** : sur un échantillon annoté de 30-50 références
- **Qualité de l'expansion** : jugement expert sur 20 articles enrichis
- **Temps de réponse** : objectif < 5s pour une requête interactive
- **Taux de couverture** : % de clauses avec au moins une référence résolue
