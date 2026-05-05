"""
Tests unitaires pour l'expansion sémantique (Phase 2).
Teste la construction de G_ref, l'expansion récursive, la troncature, et le chunking.
"""
import pytest
import networkx as nx

from kg_rag.expansion import (
    build_reference_graph,
    expand_clause,
    expand_all_clauses,
    chunk_enriched_text,
    prepare_indexing_data,
    _truncate_text,
    MAX_EXPANDED_TEXT_CHARS,
)


# ══════════════════════════════════════════════════════════════════════
# Troncature
# ══════════════════════════════════════════════════════════════════════

class TestTruncateText:
    """Tests pour la fonction de troncature."""

    def test_short_text_unchanged(self):
        text = "Texte court"
        assert _truncate_text(text, max_chars=100) == text

    def test_long_text_truncated(self):
        text = "A" * 5000
        result = _truncate_text(text, max_chars=1000)
        assert len(result) < 5000
        assert "[... texte tronqué" in result

    def test_truncate_at_newline(self):
        text = "Premier paragraphe\n" + "B" * 2000 + "\nDeuxième paragraphe"
        result = _truncate_text(text, max_chars=500)
        # Devrait couper au premier paragraphe
        assert "Premier paragraphe" in result

    def test_max_chars_none_uses_default(self):
        text = "x" * 100
        result = _truncate_text(text)  # max_chars=None → default
        assert result == text  # 100 < 3000


# ══════════════════════════════════════════════════════════════════════
# Construction G_ref
# ══════════════════════════════════════════════════════════════════════

class TestBuildReferenceGraph:
    """Tests pour la construction du graphe de références."""

    def _build_test_data(self):
        G_lex = nx.DiGraph()
        G_lex.add_node("art5", clause_id="Art. 5.", full_text="Texte article 5", node_type="article", source_file="test.json")
        G_lex.add_node("art16", clause_id="Art. 16.", full_text="Texte article 16", node_type="article", source_file="test.json")
        G_lex.add_node("art17", clause_id="Art. 17.", full_text="Texte article 17", node_type="article", source_file="test.json")

        resolution_results = {
            "art5": {
                "node_id": "art5",
                "internal_refs": [
                    {"type": "internal", "target_node_id": "art16", "confidence": 0.9},
                    {"type": "internal", "target_node_id": "art17", "confidence": 0.9},
                ],
                "external_refs": [
                    {"type": "external", "ext_id": "EXT::loi_98_04", "raw": "loi n° 98-04", "confidence": 0.7},
                ],
                "unresolved": [],
            },
            "art16": {
                "node_id": "art16",
                "internal_refs": [],
                "external_refs": [],
                "unresolved": [],
            },
            "art17": {
                "node_id": "art17",
                "internal_refs": [],
                "external_refs": [],
                "unresolved": [],
            },
        }
        return G_lex, resolution_results

    def test_build_ref_graph_nodes(self):
        G_lex, resolution_results = self._build_test_data()
        G_ref = build_reference_graph(G_lex, resolution_results)
        # Les 3 noeuds internes + 1 noeud externe
        assert G_ref.number_of_nodes() == 4

    def test_build_ref_graph_edges(self):
        G_lex, resolution_results = self._build_test_data()
        G_ref = build_reference_graph(G_lex, resolution_results)
        # 2 arêtes internes (art5→art16, art5→art17) + 1 externe (art5→EXT::...)
        assert G_ref.number_of_edges() == 3

    def test_internal_edge_type(self):
        G_lex, resolution_results = self._build_test_data()
        G_ref = build_reference_graph(G_lex, resolution_results)
        edge_data = G_ref.edges["art5", "art16"]
        assert edge_data["edge_type"] == "internal"

    def test_external_edge_type(self):
        G_lex, resolution_results = self._build_test_data()
        G_ref = build_reference_graph(G_lex, resolution_results)
        # Trouver l'arête externe
        ext_edges = [
            (u, v, d) for u, v, d in G_ref.edges(data=True)
            if d.get("edge_type") == "external"
        ]
        assert len(ext_edges) == 1


# ══════════════════════════════════════════════════════════════════════
# Expansion récursive
# ══════════════════════════════════════════════════════════════════════

class TestExpandClause:
    """Tests pour l'expansion récursive."""

    def _build_test_graphs(self):
        G_lex = nx.DiGraph()
        G_lex.add_node("art5", clause_id="Art. 5.", full_text="Conditions fixées aux articles 16 et 17.", node_type="article", source_file="test.json")
        G_lex.add_node("art16", clause_id="Art. 16.", full_text="La chasse touristique ne peut être exercée que...", node_type="article", source_file="test.json")
        G_lex.add_node("art17", clause_id="Art. 17.", full_text="Les agences de tourisme sont tenues...", node_type="article", source_file="test.json")

        G_ref = nx.DiGraph()
        for nid, data in G_lex.nodes(data=True):
            G_ref.add_node(nid, **data)
        G_ref.add_edge("art5", "art16", edge_type="internal", confidence=0.9)
        G_ref.add_edge("art5", "art17", edge_type="internal", confidence=0.9)

        definitions = [
            {"term": "chasse touristique", "definition": "exercice de la chasse par un touriste étranger", "source_clause": "Art. 2", "source_file": "test.json", "category": "Baleine"},
        ]
        return G_lex, G_ref, definitions

    def test_expand_includes_source_text(self):
        G_lex, G_ref, definitions = self._build_test_graphs()
        result = expand_clause("art5", G_ref, G_lex, definitions, max_depth=2)
        assert "Art. 5." in result
        assert "Conditions fixées" in result

    def test_expand_includes_referenced_text(self):
        G_lex, G_ref, definitions = self._build_test_graphs()
        result = expand_clause("art5", G_ref, G_lex, definitions, max_depth=2)
        assert "Art. 16." in result
        assert "chasse touristique" in result

    def test_expand_injects_definitions(self):
        G_lex, G_ref, definitions = self._build_test_graphs()
        result = expand_clause("art5", G_ref, G_lex, definitions, max_depth=2, include_definitions=True)
        assert "Définition" in result
        assert "chasse touristique" in result

    def test_expand_no_definitions(self):
        G_lex, G_ref, definitions = self._build_test_graphs()
        result = expand_clause("art5", G_ref, G_lex, definitions, max_depth=2, include_definitions=False)
        assert "Définition" not in result

    def test_expand_max_depth_zero(self):
        G_lex, G_ref, definitions = self._build_test_graphs()
        result = expand_clause("art5", G_ref, G_lex, definitions, max_depth=0)
        # Avec depth=0, la clause elle-même est incluse mais pas les refs
        assert "Art. 5." in result

    def test_expand_anti_cycle(self):
        """Vérifie que l'expansion ne boucle pas sur un cycle."""
        G_lex = nx.DiGraph()
        G_lex.add_node("a", clause_id="A", full_text="Texte A", node_type="article", source_file="test.json")
        G_lex.add_node("b", clause_id="B", full_text="Texte B", node_type="article", source_file="test.json")

        G_ref = nx.DiGraph()
        for nid, data in G_lex.nodes(data=True):
            G_ref.add_node(nid, **data)
        # Cycle : A → B → A
        G_ref.add_edge("a", "b", edge_type="internal", confidence=0.9)
        G_ref.add_edge("b", "a", edge_type="internal", confidence=0.9)

        result = expand_clause("a", G_ref, G_lex, [], max_depth=5)
        # Ne doit pas boucler infiniment
        assert "Texte A" in result
        assert "Texte B" in result

    def test_expand_external_ref_banner(self):
        """Vérifie le bandeau pour les références externes non résolues."""
        G_lex = nx.DiGraph()
        G_lex.add_node("art5", clause_id="Art. 5.", full_text="Texte article 5", node_type="article", source_file="test.json")

        G_ref = nx.DiGraph()
        G_ref.add_node("art5", **G_lex.nodes["art5"])
        G_ref.add_node("EXT::loi_98_04", node_type="external", raw="loi n° 98-04")
        G_ref.add_edge("art5", "EXT::loi_98_04", edge_type="external", confidence=0.7)

        result = expand_clause("art5", G_ref, G_lex, [], max_depth=2)
        assert "Référence externe non disponible" in result
        assert "loi n° 98-04" in result


# ══════════════════════════════════════════════════════════════════════
# Chunking
# ══════════════════════════════════════════════════════════════════════

class TestChunkEnrichedText:
    """Tests pour le découpage en chunks."""

    def test_single_section(self):
        text = "### Art. 5.\nTexte court"
        chunks = chunk_enriched_text(text, chunk_size=500, chunk_overlap=50)
        assert len(chunks) == 1

    def test_multiple_sections(self):
        text = "### Art. 5.\n" + "A" * 2000 + "\n### Art. 6.\n" + "B" * 2000
        chunks = chunk_enriched_text(text, chunk_size=500, chunk_overlap=50)
        assert len(chunks) >= 2

    def test_empty_text(self):
        chunks = chunk_enriched_text("", chunk_size=500)
        assert len(chunks) == 1  # Retourne le texte lui-même si pas de sections

    def test_prepare_indexing_data(self):
        G_ref = nx.DiGraph()
        G_ref.add_node("art5", clause_id="Art. 5.", full_text="Texte", node_type="article",
                       title_or_summary="Test", country="Algérie", doc_title="Loi test",
                       category="Baleine", source_file="test.json", page_range=[1])

        expanded = {"art5": "### Art. 5.\nTexte enrichi"}
        data = prepare_indexing_data(expanded, G_ref)
        assert len(data) >= 1
        assert data[0]["metadata"]["clause_id"] == "Art. 5."
        assert data[0]["metadata"]["country"] == "Algérie"
