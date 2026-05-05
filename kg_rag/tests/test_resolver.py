"""
Tests unitaires pour le résolveur hybride de références (Phase 1).
Teste les 4 étages : Regex, Heuristiques graphe, Externe, Fallback.
"""
import pytest
import networkx as nx

from kg_rag.reference_resolver import (
    extract_references_regex,
    resolve_anaphore,
    find_ancestor_of_type,
    find_sibling_by_offset,
    ANAPHORE_KEYWORDS,
    LLM_TRIGGER_EXPRESSIONS,
    ReferenceResolver,
)


# ══════════════════════════════════════════════════════════════════════
# Étage 1 – Regex
# ══════════════════════════════════════════════════════════════════════

class TestExtractReferencesRegex:
    """Tests pour l'extraction de références par regex."""

    def test_article_single(self):
        refs = extract_references_regex("conformément à l'article 16 de la présente loi")
        assert len(refs) >= 1
        article_refs = [r for r in refs if r["type"] == "article"]
        assert any(r["value"] == 16 for r in article_refs)

    def test_article_range(self):
        refs = extract_references_regex("articles 34 à 40 de la présente loi")
        assert any(r["type"] == "article_range" and r["value"] == (34, 40) for r in refs)

    def test_article_range_et(self):
        refs = extract_references_regex("articles 16 et 18 de la présente loi")
        assert any(r["type"] == "article_range" and r["value"] == (16, 18) for r in refs)

    def test_article_ci_dessus(self):
        refs = extract_references_regex("l'article 9 ci-dessus")
        article_refs = [r for r in refs if r["type"] == "article"]
        assert any(r["value"] == 9 and r.get("qualifier") == "ci-dessus" for r in article_refs)

    def test_alinea(self):
        refs = extract_references_regex("alinéa 3 du présent article")
        assert any(r["type"] == "alinea" and r["value"] == 3 for r in refs)

    def test_paragraphe(self):
        refs = extract_references_regex("paragraphe 7.4 du règlement")
        assert any(r["type"] == "paragraphe" and r["value"] == "7.4" for r in refs)

    def test_annexe_romaine(self):
        refs = extract_references_regex("annexe I de la convention")
        assert any(r["type"] == "annexe" and r["value"] == "I" for r in refs)

    def test_annexe_lettre(self):
        refs = extract_references_regex("annexe A du décret")
        assert any(r["type"] == "annexe" and r["value"] == "A" for r in refs)

    def test_footnote_brackets(self):
        refs = extract_references_regex("voir [3] pour plus de détails")
        assert any(r["type"] == "footnote" and r["value"] == 3 for r in refs)

    def test_footnote_note(self):
        refs = extract_references_regex("note 5 ci-dessous")
        assert any(r["type"] == "footnote" and r["value"] == 5 for r in refs)

    def test_titre(self):
        refs = extract_references_regex("titre II de la loi")
        assert any(r["type"] == "titre" and r["value"] == "II" for r in refs)

    def test_chapitre(self):
        refs = extract_references_regex("chapitre III du code")
        assert any(r["type"] == "chapitre" and r["value"] == "III" for r in refs)

    def test_external_law(self):
        refs = extract_references_regex("loi n° 98-04 du 15 juin 1998")
        assert any(r["type"] == "external_law" and r["value"] == "98-04" for r in refs)

    def test_external_code(self):
        refs = extract_references_regex("code de procédure pénale")
        assert any(r["type"] == "external_code" for r in refs)

    def test_no_references(self):
        refs = extract_references_regex("Le permis de chasse est personnel.")
        assert len(refs) == 0

    def test_article_range_no_overlap(self):
        """Les articles dans une plage ne doivent pas être extraits deux fois."""
        refs = extract_references_regex("articles 34 à 40 de la présente loi")
        article_single_in_range = [
            r for r in refs
            if r["type"] == "article" and 34 <= r["value"] <= 40
        ]
        assert len(article_single_in_range) == 0

    def test_art_abbreviation(self):
        refs = extract_references_regex("art. 20 de la loi")
        assert any(r["type"] == "article" and r["value"] == 20 for r in refs)


# ══════════════════════════════════════════════════════════════════════
# Étage 2 – Heuristiques de graphe
# ══════════════════════════════════════════════════════════════════════

class TestAnaphoreResolution:
    """Tests pour la résolution d'anaphores via heuristiques de graphe."""

    def _build_test_graph(self):
        """Construit un graphe de test avec hiérarchie."""
        G = nx.DiGraph()
        # Titre
        G.add_node("titre1", clause_id="TITRE I", node_type="titre", source_file="test.json")
        # Chapitre
        G.add_node("chap1", clause_id="Chapitre I", node_type="chapitre", source_file="test.json")
        G.add_edge("titre1", "chap1", edge_type="hierarchy")
        # Articles
        G.add_node("art5", clause_id="Art. 5.", node_type="article", source_file="test.json")
        G.add_node("art6", clause_id="Art. 6.", node_type="article", source_file="test.json")
        G.add_node("art7", clause_id="Art. 7.", node_type="article", source_file="test.json")
        G.add_edge("chap1", "art5", edge_type="hierarchy")
        G.add_edge("chap1", "art6", edge_type="hierarchy")
        G.add_edge("chap1", "art7", edge_type="hierarchy")
        return G

    def test_find_ancestor_article(self):
        G = self._build_test_graph()
        # art6 est un article, son ancêtre de type article est lui-même
        result = find_ancestor_of_type(G, "art6", "article")
        assert result == "art6"

    def test_find_ancestor_titre(self):
        G = self._build_test_graph()
        result = find_ancestor_of_type(G, "art6", "titre")
        assert result == "titre1"

    def test_find_ancestor_chapitre(self):
        G = self._build_test_graph()
        result = find_ancestor_of_type(G, "art6", "chapitre")
        assert result == "chap1"

    def test_find_sibling_previous(self):
        G = self._build_test_graph()
        seq = {"art5": 0, "art6": 1, "art7": 2}
        result = find_sibling_by_offset(G, "art6", seq, -1)
        assert result == "art5"

    def test_find_sibling_next(self):
        G = self._build_test_graph()
        seq = {"art5": 0, "art6": 1, "art7": 2}
        result = find_sibling_by_offset(G, "art6", seq, 1)
        assert result == "art7"

    def test_find_sibling_boundary(self):
        G = self._build_test_graph()
        seq = {"art5": 0, "art6": 1, "art7": 2}
        # art5 n'a pas de frère précédent
        result = find_sibling_by_offset(G, "art5", seq, -1)
        assert result is None

    def test_resolve_present_article(self):
        G = self._build_test_graph()
        seq = {"art5": 0, "art6": 1, "art7": 2}
        article_index = {"5": ["art5"], "6": ["art6"], "7": ["art7"]}
        result = resolve_anaphore(G, seq, article_index, "art6", "le présent article")
        assert result == "art6"

    def test_resolve_precedent(self):
        G = self._build_test_graph()
        seq = {"art5": 0, "art6": 1, "art7": 2}
        article_index = {"5": ["art5"], "6": ["art6"], "7": ["art7"]}
        result = resolve_anaphore(G, seq, article_index, "art6", "l'article précédent")
        assert result == "art5"


# ══════════════════════════════════════════════════════════════════════
# Résolveur complet
# ══════════════════════════════════════════════════════════════════════

class TestReferenceResolver:
    """Tests pour le résolveur hybride complet."""

    def _build_resolver_graph(self):
        """Construit un graphe minimal pour tester le résolveur."""
        G = nx.DiGraph()
        G.add_node(
            "Baleine::Art. 5.::test.json",
            clause_id="Art. 5.",
            parent_id=None,
            level=1,
            full_text="Le droit de chasser n'est ouvert aux ressortissants étrangers que dans les conditions fixées aux articles 16, 17 et 18 de la présente loi.",
            node_type="article",
            source_file="test.json",
            country="Algérie",
        )
        G.add_node(
            "Baleine::Art. 16.::test.json",
            clause_id="Art. 16.",
            parent_id=None,
            level=1,
            full_text="La chasse touristique sur le territoire national ne peut être exercée que dans les conditions ci-après.",
            node_type="article",
            source_file="test.json",
            country="Algérie",
        )
        G.add_node(
            "Baleine::Art. 17.::test.json",
            clause_id="Art. 17.",
            parent_id=None,
            level=1,
            full_text="Les agences de tourisme sont tenues de veiller au respect de la législation.",
            node_type="article",
            source_file="test.json",
            country="Algérie",
        )
        G.add_node(
            "Baleine::Art. 18.::test.json",
            clause_id="Art. 18.",
            parent_id=None,
            level=1,
            full_text="Les produits de chasse touristique ne peuvent dépasser le nombre autorisé.",
            node_type="article",
            source_file="test.json",
            country="Algérie",
        )
        return G

    def test_resolve_article_range(self):
        G = self._build_resolver_graph()
        article_index = {
            "16": ["Baleine::Art. 16.::test.json"],
            "17": ["Baleine::Art. 17.::test.json"],
            "18": ["Baleine::Art. 18.::test.json"],
        }
        seq_index = {
            "Baleine::Art. 5.::test.json": 0,
            "Baleine::Art. 16.::test.json": 1,
            "Baleine::Art. 17.::test.json": 2,
            "Baleine::Art. 18.::test.json": 3,
        }
        resolver = ReferenceResolver(
            G_lex=G, article_index=article_index, seq_index=seq_index,
            use_llm_fallback=False,
        )
        result = resolver.resolve_clause("Baleine::Art. 5.::test.json")
        # Devrait trouver la référence aux articles 16, 17, 18
        assert len(result["internal_refs"]) >= 1
        # Au moins une référence interne résolue
        internal_targets = [
            r.get("target_node_id") or r.get("target_node_ids", [])
            for r in result["internal_refs"]
        ]
        # Vérifier qu'au moins un article 16, 17 ou 18 est référencé
        all_targets = []
        for t in internal_targets:
            if isinstance(t, list):
                all_targets.extend(t)
            else:
                all_targets.append(t)
        assert any("16" in str(t) or "17" in str(t) or "18" in str(t) for t in all_targets)

    def test_resolve_empty_clause(self):
        G = nx.DiGraph()
        G.add_node("empty", clause_id="empty", full_text="", node_type="article", source_file="test.json")
        resolver = ReferenceResolver(G, {}, {}, use_llm_fallback=False)
        result = resolver.resolve_clause("empty")
        assert result["internal_refs"] == []
        assert result["external_refs"] == []

    def test_llm_trigger_expressions(self):
        """Vérifier que les expressions déclencheuses sont définies."""
        assert len(LLM_TRIGGER_EXPRESSIONS) > 0
        assert "précédent" in LLM_TRIGGER_EXPRESSIONS
        assert "ci-dessus" in LLM_TRIGGER_EXPRESSIONS
        assert "susvisé" in LLM_TRIGGER_EXPRESSIONS

    def test_anaphore_keywords(self):
        """Vérifier que les mots-clés anaphoriques sont définis."""
        assert len(ANAPHORE_KEYWORDS) > 0
        assert "précédent" in ANAPHORE_KEYWORDS
        assert "ci-dessus" in ANAPHORE_KEYWORDS
