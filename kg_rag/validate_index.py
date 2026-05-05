"""
Script de validation de l'index externe ChromaDB.
Vérifie que la collection external_laws contient bien toutes les lois
du corpus multi-pays et affiche des statistiques par loi/pays.
"""
import argparse
import logging
from collections import defaultdict

from . import config
from .document_loader import PreprocessedData
from .vector_indexer import ChromaStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def validate_external_index():
    """Valide l'index externe ChromaDB contre les documents source."""
    # Charger les documents source
    preprocessed = PreprocessedData()
    preprocessed.build()

    # Compter les clauses par pays/titre dans les sources
    source_stats = defaultdict(lambda: {"clauses": 0, "docs": 0})
    for cat_name, docs in preprocessed.all_docs.items():
        for doc in docs:
            meta = doc.get("metadata", {})
            country = meta.get("country", "Inconnu")
            title = meta.get("title", "Sans titre")
            key = f"{country} | {title}"
            source_stats[key]["clauses"] += len(doc.get("clauses", []))
            source_stats[key]["docs"] += 1

    # Vérifier l'index ChromaDB
    chroma = ChromaStore()
    try:
        collection = chroma.get_or_create_collection(config.CHROMA_COLLECTION_EXTERNAL)
        total_indexed = collection.count()
    except Exception as e:
        logger.error(f"Impossible d'accéder à la collection externe : {e}")
        print("\n❌ Collection external_laws introuvable. Lancez d'abord : python -m kg_rag --build")
        return

    # Compter les clauses par pays dans l'index
    index_stats = defaultdict(int)
    if total_indexed > 0:
        results = collection.get(include=["metadatas"])
        for meta in results.get("metadatas", []):
            country = meta.get("country", "Inconnu")
            index_stats[country] += 1

    # Afficher les résultats
    print("\n" + "=" * 70)
    print("  VALIDATION INDEX EXTERNE (external_laws)")
    print("=" * 70)

    print(f"\n📊 Total clauses indexées : {total_indexed}")
    print(f"📊 Total clauses source   : {sum(s['clauses'] for s in source_stats.values())}")

    print("\n── Par pays (index) ──")
    for country, count in sorted(index_stats.items()):
        print(f"  {country:30s} : {count} clauses")

    print("\n── Par document (source) ──")
    for key, stats in sorted(source_stats.items()):
        print(f"  {key:60s} : {stats['clauses']} clauses ({stats['docs']} doc(s))")

    # Vérification de cohérence
    total_source = sum(s["clauses"] for s in source_stats.values())
    if total_indexed == 0:
        print("\n⚠️  Index vide ! Construisez-le avec : python -m kg_rag --build")
    elif total_indexed < total_source * 0.8:
        print(f"\n⚠️  Index potentiellement incomplet : {total_indexed}/{total_source} clauses ({total_indexed/total_source*100:.1f}%)")
    else:
        print(f"\n✅ Index semble complet : {total_indexed}/{total_source} clauses ({total_indexed/total_source*100:.1f}%)")

    # Vérifier les collections enrichies et définitions
    for coll_name in [config.CHROMA_COLLECTION_ENRICHED, config.CHROMA_COLLECTION_DEFINITIONS]:
        try:
            coll = chroma.get_or_create_collection(coll_name)
            print(f"\n📦 Collection '{coll_name}' : {coll.count()} documents")
        except Exception:
            print(f"\n❌ Collection '{coll_name}' : inaccessible")


def main():
    parser = argparse.ArgumentParser(description="Validation de l'index ChromaDB")
    parser.add_argument("--validate", action="store_true", help="Valider l'index externe")
    args = parser.parse_args()

    if args.validate:
        validate_external_index()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
