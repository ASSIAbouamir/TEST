"""
Point d'entrée principal du système KG-RAG Multi-Graph Multi-Agent.
Pipeline complet : Phase 0 → Phase 1 → Phase 2 → Phase 3 → Phase 4.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

from . import config
from .document_loader import PreprocessedData
from .reference_resolver import ReferenceResolver
from .expansion import (
    build_reference_graph,
    expand_all_clauses,
    prepare_indexing_data,
)
from .vector_indexer import ChromaStore, BM25Retriever, HybridRetriever
from .workflow import KGRAGWorkflow

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, "INFO"),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# Pipeline complet
# ══════════════════════════════════════════════════════════════════════

class KGRAGPipeline:
    """
    Orchestrateur du pipeline KG-RAG complet.
    """

    def __init__(self, skip_indexing: bool = False, use_llm_fallback: bool = True):
        self.skip_indexing = skip_indexing
        self.use_llm_fallback = use_llm_fallback
        self.preprocessed = PreprocessedData()
        self.G_ref = None
        self.resolution_results = None
        self.chroma_store = None
        self.hybrid_retriever = None
        self.workflow = None

    def build(self):
        """Exécute les Phases 0-3 (offline)."""
        # ── Phase 0 : Prétraitement ──
        logger.info("══════════════════════════════════════════════════════")
        logger.info("  Phase 0 – Prétraitement")
        logger.info("══════════════════════════════════════════════════════")
        self.preprocessed.build()

        # ── Phase 1 : Résolution de références ──
        logger.info("══════════════════════════════════════════════════════")
        logger.info("  Phase 1 – Résolution hybride de références")
        logger.info("══════════════════════════════════════════════════════")
        resolver = ReferenceResolver(
            G_lex=self.preprocessed.G_lex,
            article_index=self.preprocessed.article_index,
            seq_index=self.preprocessed.seq_index,
            external_collection=None,  # Sera connecté après indexation
            use_llm_fallback=self.use_llm_fallback,
        )
        self.resolution_results = resolver.resolve_all()

        # ── Phase 2 : Expansion sémantique ──
        logger.info("══════════════════════════════════════════════════════")
        logger.info("  Phase 2 – Expansion sémantique")
        logger.info("══════════════════════════════════════════════════════")
        self.G_ref = build_reference_graph(
            self.preprocessed.G_lex, self.resolution_results
        )
        expanded = expand_all_clauses(
            self.G_ref,
            self.preprocessed.G_lex,
            self.preprocessed.definitions,
        )
        indexing_data = prepare_indexing_data(expanded, self.G_ref)

        # ── Phase 3 : Indexation vectorielle ──
        if not self.skip_indexing:
            logger.info("══════════════════════════════════════════════════════")
            logger.info("  Phase 3 – Indexation vectorielle")
            logger.info("══════════════════════════════════════════════════════")
            self.chroma_store = ChromaStore()

            # Collection principale enrichie
            self.chroma_store.index_enriched_articles(indexing_data)

            # Collection externe (toutes les clauses de tous les docs)
            self.chroma_store.index_external_laws(self.preprocessed.all_docs)

            # Collection des définitions
            self.chroma_store.index_definitions(self.preprocessed.definitions)

            # Construire le retriever hybride
            all_chunks = []
            collection = self.chroma_store.get_or_create_collection(
                config.CHROMA_COLLECTION_ENRICHED
            )
            if collection.count() > 0:
                results = collection.get(include=["documents", "metadatas"])
                for i, doc_id in enumerate(results["ids"]):
                    all_chunks.append({
                        "id": doc_id,
                        "text": results["documents"][i] if results["documents"] else "",
                        "metadata": results["metadatas"][i] if results["metadatas"] else {},
                    })

            bm25_retriever = BM25Retriever(all_chunks)
            self.hybrid_retriever = HybridRetriever(self.chroma_store, bm25_retriever)
        else:
            logger.info("Phase 3 ignorée (skip_indexing=True)")
            # Charger le store existant
            self.chroma_store = ChromaStore()

        # ── Phase 4 : Préparation du workflow ──
        logger.info("══════════════════════════════════════════════════════")
        logger.info("  Phase 4 – Workflow multi-agents prêt")
        logger.info("══════════════════════════════════════════════════════")
        self.workflow = KGRAGWorkflow(
            hybrid_retriever=self.hybrid_retriever,
            definition_store=self.chroma_store,
            chroma_store=self.chroma_store,
            G_ref=self.G_ref,
            G_lex=self.preprocessed.G_lex,
            article_index=self.preprocessed.article_index,
        )

        logger.info("Pipeline KG-RAG prêt !")
        return self

    def query(self, question: str) -> dict:
        """Pose une question au système."""
        if self.workflow is None:
            raise RuntimeError("Le pipeline n'est pas construit. Appelez build() d'abord.")
        result = self.workflow.run(question)
        return result


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="KG-RAG Multi-Graph Multi-Agent pour documents juridiques"
    )
    parser.add_argument(
        "--build", action="store_true",
        help="Construire le pipeline (Phases 0-3)",
    )
    parser.add_argument(
        "--query", type=str, default=None,
        help="Poser une question au système",
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Mode interactif (boucle de questions)",
    )
    parser.add_argument(
        "--skip-indexing", action="store_true",
        help="Sauter la Phase 3 (indexation vectorielle)",
    )
    parser.add_argument(
        "--no-llm-fallback", action="store_true",
        help="Désactiver le fallback LLM (Étage 4)",
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Afficher les statistiques du graphe",
    )

    args = parser.parse_args()

    pipeline = KGRAGPipeline(
        skip_indexing=args.skip_indexing,
        use_llm_fallback=not args.no_llm_fallback,
    )

    # Build si demandé ou si on veut interroger
    if args.build or args.query or args.interactive:
        logger.info("Construction du pipeline...")
        pipeline.build()

        if args.stats:
            print("\n═══ Statistiques ═══")
            print(f"  Noeuds G_lex : {pipeline.preprocessed.G_lex.number_of_nodes()}")
            print(f"  Arêtes G_lex : {pipeline.preprocessed.G_lex.number_of_edges()}")
            print(f"  Noeuds G_ref : {pipeline.G_ref.number_of_nodes()}")
            print(f"  Arêtes G_ref : {pipeline.G_ref.number_of_edges()}")
            print(f"  Définitions  : {len(pipeline.preprocessed.definitions)}")
            print(f"  Articles idx : {len(pipeline.preprocessed.article_index)}")
            if pipeline.resolution_results:
                total_int = sum(len(r["internal_refs"]) for r in pipeline.resolution_results.values())
                total_ext = sum(len(r["external_refs"]) for r in pipeline.resolution_results.values())
                total_unr = sum(len(r["unresolved"]) for r in pipeline.resolution_results.values())
                print(f"  Réf. internes résolues : {total_int}")
                print(f"  Réf. externes résolues : {total_ext}")
                print(f"  Réf. non résolues      : {total_unr}")

    # Requête unique
    if args.query:
        print(f"\n{'='*60}")
        print(f"  Question : {args.query}")
        print(f"{'='*60}\n")

        result = pipeline.query(args.query)

        print(result.get("final_answer", "Aucune réponse générée"))

        print(f"\n{'─'*60}")
        print(f"  Passes : {result.get('pass_count', 0)}")
        print(f"  Noeuds visités : {len(result.get('retrieved_graph_nodes', []))}")
        print(f"  Définitions : {list(result.get('definitions', {}).keys())}")
        print(f"  Échecs : {len(result.get('failures', []))}")

    # Mode interactif
    if args.interactive:
        print("\n═══ Mode interactif KG-RAG ═══")
        print("Tapez 'quit' pour quitter, 'stats' pour les statistiques.\n")

        while True:
            try:
                question = input("🔍 Question > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nAu revoir !")
                break

            if question.lower() in ("quit", "exit", "q"):
                break
            if question.lower() == "stats":
                if pipeline.preprocessed.G_lex:
                    print(f"  Noeuds : {pipeline.preprocessed.G_lex.number_of_nodes()}")
                    print(f"  Définitions : {len(pipeline.preprocessed.definitions)}")
                continue
            if not question:
                continue

            result = pipeline.query(question)
            print(f"\n{result.get('final_answer', 'Aucune réponse')}\n")
            print(f"  [{result.get('pass_count', 0)} passes, "
                  f"{len(result.get('retrieved_graph_nodes', []))} noeuds]\n")

    if not any([args.build, args.query, args.interactive, args.stats]):
        parser.print_help()


if __name__ == "__main__":
    main()
