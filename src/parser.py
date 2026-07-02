import sys
import os
import shutil
import json
import logging
import re
import time
from pathlib import Path
import nest_asyncio

# Appliquer nest_asyncio pour LlamaParse
nest_asyncio.apply()

from .config import (
    LLAMA_CLOUD_API_KEY, PARSED_DIR, ARCHIVE_DIR, DATA_PROCESSED_DIR,
    STRUCTURED_DIR, DEFAULT_LANGUAGE, GROQ_API_KEY, GROQ_MODEL
)

logger = logging.getLogger(__name__)


from process_pdf import (
    EXTRACTION_SYSTEM_PROMPT, clean_spaced_text, clean_text, correct_hyphenation,
    merge_short_clauses, validate_cross_references, generate_synthetic_qa,
    detect_legal_conflicts, extract_structure_with_llm, local_parse_to_markdown
)


# ─────────────────────────────────────────────────────────────────────────────
# Fonctions d'aide pour le découpage et la réparation du JSON
# ─────────────────────────────────────────────────────────────────────────────
def _repair_json(raw: str) -> dict | None:
    """Tente de réparer un JSON tronqué ou malformaté."""
    start = raw.find("{")
    if start == -1:
        return None

    # Trouver l'accolade fermante balancée
    depth = 0
    end   = -1
    for i, ch in enumerate(raw[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end != -1:
        try:
            return json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            pass

    # Essayer d'extraire les clauses individuelles par regex
    clauses = []
    for m in re.finditer(r'\{[^{}]*"clause_id"[^{}]*\}', raw, re.DOTALL):
        try:
            obj = json.loads(m.group())
            if "clause_id" in obj:
                clauses.append(obj)
        except json.JSONDecodeError:
            pass

    if clauses:
        logger.info(f"  Réparation JSON : {len(clauses)} clause(s) extraites par regex.")
        return {
            "document_metadata": {},
            "clauses": clauses,
            "definitions": [],
        }

    return None


def _split_markdown_into_chunks(markdown: str, max_chars: int = 2500) -> list[str]:
    """Découpe le Markdown en chunks en respectant les fins de paragraphes."""
    paragraphs = re.split(r"\n{2,}", markdown.strip())
    chunks     = []
    current    = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para) + 2  # +2 pour \n\n
        if current_len + para_len > max_chars and current:
            chunks.append("\n\n".join(current))
            current     = [para]
            current_len = para_len
        else:
            current.append(para)
            current_len += para_len

    if current:
        chunks.append("\n\n".join(current))

    logger.info(f"Découpage Markdown : {len(chunks)} chunk(s) (max {max_chars} chars chacun).")
    return chunks


class LegalDocumentParser:
    """
    Pipeline complet de traitement de documents juridiques :
      1. LlamaParse  → Markdown
      2. Groq LLM    → JSON structuré (document_metadata + clauses + definitions)
      3. Conversion  → Nodes RAG pour data_processed/
    """

    def __init__(self, api_key: str = None, language: str = None):
        self.api_key  = api_key or LLAMA_CLOUD_API_KEY
        self.language = language or DEFAULT_LANGUAGE
        self.parser   = None   # LlamaParse — initialisation différée

    # ─────────────────────────────────────────────────────────────────────────
    # Utilitaire — extraction thème / pays depuis le nom de fichier
    # ─────────────────────────────────────────────────────────────────────────
    def _parse_theme_country(self, filename: str) -> tuple[str, str]:
        """Extrait le thème et le pays depuis le nom du fichier (Theme_Pays.ext)."""
        name  = Path(filename).stem
        parts = name.split("_")
        if len(parts) >= 2:
            theme   = parts[0].strip()
            country = "_".join(parts[1:]).strip()
            return theme, country
        return "General", name.replace("_", " ").strip()

    # ─────────────────────────────────────────────────────────────────────────
    # ÉTAPE 1 — LlamaParse : PDF → Markdown
    # ─────────────────────────────────────────────────────────────────────────
    def _llamaparse_to_markdown(self, file_path: Path) -> str:
        """Envoie le fichier à LlamaParse et renvoie le Markdown extrait."""
        if not self.api_key:
            logger.warning("LLAMA_CLOUD_API_KEY non configurée. Utilisation du parseur local en fallback.")
            return local_parse_to_markdown(file_path)

        if not self.parser:
            from llama_parse import LlamaParse
            
            # Instruction / Prompt spécifique pour guider LlamaParse sur la structure légale
            parsing_instruction = """
            Ce document est un texte de loi ou décret officiel.
            Extrayez fidèlement tout le texte, en conservant impérativement la structure hiérarchique :
            - Titres, Chapitres, Sections, Articles, Alinéas.
            - Conservez la numérotation exacte de chaque clause (ex: 'Art. 1er.', 'Article 2.', 'TITRE I').
            - Conservez les tableaux de manière lisible au format Markdown.
            - Identifiez et préservez clairement les notes de bas de page.
            """
            
            try:
                self.parser = LlamaParse(
                    api_key=self.api_key,
                    result_type="markdown",
                    language=self.language,
                    parsing_instruction=parsing_instruction,
                    verbose=True,
                )
            except Exception as e:
                logger.warning(f"Impossible d'initialiser LlamaParse : {e}. Utilisation du parseur local en fallback.")
                return local_parse_to_markdown(file_path)

        logger.info("Envoi du fichier à LlamaParse...")
        time.sleep(1)  # sécurité disque

        documents = None
        max_retries = 3
        import concurrent.futures
        for attempt in range(1, max_retries + 1):
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            try:
                logger.info(f"Tentative {attempt}/{max_retries} → LlamaParse...")
                future = executor.submit(self.parser.load_data, str(file_path))
                # Limiter à 45 secondes pour éviter un blocage infini du serveur
                documents = future.result(timeout=45.0)
                if documents:
                    break
                logger.warning(f"LlamaParse — liste vide (tentative {attempt}).")
            except concurrent.futures.TimeoutError:
                logger.warning(f"LlamaParse tentative {attempt} dépassée (timeout 45s).")
            except Exception as exc:
                logger.warning(f"Erreur LlamaParse tentative {attempt} : {exc}")
            finally:
                executor.shutdown(wait=False)
            
            if attempt < max_retries:
                logger.info("Attente 3s avant nouvelle tentative...")
                time.sleep(3)

        if not documents:
            logger.warning(
                f"LlamaParse n'a renvoyé aucun contenu pour {file_path.name} "
                f"après {max_retries} tentatives. Utilisation du parseur local en fallback."
            )
            return local_parse_to_markdown(file_path)

        markdown_content = "\n\n".join(doc.text for doc in documents)
        logger.info(f"Markdown extrait — {len(markdown_content)} caractères, {len(documents)} page(s).")
        return markdown_content

    # ─────────────────────────────────────────────────────────────────────────
    # ÉTAPE 2 — LlamaCloud : Extraction agentique structurée
    # ─────────────────────────────────────────────────────────────────────────
    def _extract_structure_with_llamacloud(
        self,
        file_path: Path,
        filename: str,
        theme: str,
        country: str,
        progress_callback = None,
    ) -> dict:
        """
        Soumet le document à l'API LlamaCloud Extract pour en extraire
        la structure JSON selon le schéma de données cible de l'application.
        """
        from llama_cloud.client import LlamaCloud
        
        # Récupération de la clé API LlamaIndex Cloud (qui est la même que LlamaParse)
        api_key = self.api_key or LLAMA_CLOUD_API_KEY
        if not api_key:
            raise ValueError("LLAMA_CLOUD_API_KEY non configuree.")
            
        client = LlamaCloud(token=api_key)
        
        # Schema d'extraction JSON
        data_schema = {
            "type": "object",
            "properties": {
                "document_metadata": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "date": {"type": "string"}
                    },
                    "required": ["title"]
                },
                "clauses": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "clause_id": {"type": "string"},
                            "parent_id": {"type": "string"},
                            "level": {"type": "integer"},
                            "title_or_summary": {"type": "string"},
                            "full_text": {"type": "string"},
                            "original_text": {"type": "string"},
                            "page_range": {
                                "type": "array",
                                "items": {"type": "integer"}
                            },
                            "cross_references": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "is_footnote": {"type": "boolean"},
                            "entities": {
                                "type": "object",
                                "properties": {
                                    "authorities": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    },
                                    "penalties": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    },
                                    "dates_durations": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    }
                                }
                            }
                        },
                        "required": ["clause_id", "full_text", "title_or_summary"]
                    }
                },
                "definitions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "term": {"type": "string"},
                            "definition": {"type": "string"}
                        },
                        "required": ["term", "definition"]
                    }
                }
            },
            "required": ["clauses"]
        }
        
        if progress_callback:
            progress_callback(40, f"Téléchargement du fichier vers LlamaCloud... [{filename}]")
            
        file_obj = client.files.create(file=file_path, purpose="extract")
        
        if progress_callback:
            progress_callback(50, f"Exécution de l'extraction Agentic LlamaCloud...")
            
        job = client.extract.create(
            file_input=file_obj.id,
            configuration={
                "data_schema": data_schema,
                "tier": "agentic",
                "extraction_target": "per_doc",
                "parse_tier": "agentic",
                "cite_sources": True,
                "confidence_scores": True
            },
        )
        
        while job.status not in ("COMPLETED", "FAILED", "CANCELLED"):
            time.sleep(3)
            job = client.extract.get(job.id)
            
        if job.status != "COMPLETED":
            raise RuntimeError(f"L'extraction LlamaCloud {job.id} a échoué: {job.error_message}")
            
        if progress_callback:
            progress_callback(60, f"Extraction LlamaCloud terminée.")
            
        result = job.extract_result
        document_metadata = result.get("document_metadata", {})
        if not document_metadata.get("title"):
            document_metadata["title"] = filename

        # Fallback pour les métriques de tokens de LlamaCloud
        total_prompt_tokens = 0
        total_completion_tokens = 0
        if hasattr(job, "usage") and job.usage:
            total_prompt_tokens = getattr(job.usage, "prompt_tokens", 0) or 0
            total_completion_tokens = getattr(job.usage, "completion_tokens", 0) or 0
            
        return {
            "document_metadata": document_metadata,
            "clauses":           result.get("clauses", []),
            "definitions":       result.get("definitions", []),
            "metrics": {
                "tokens": {
                    "prompt_tokens": total_prompt_tokens,
                    "completion_tokens": total_completion_tokens,
                    "estimated_cost_usd": total_prompt_tokens * 0.00000005 + total_completion_tokens * 0.00000008
                }
            }
        }

    # ─────────────────────────────────────────────────────────────────────────
    # ÉTAPE 3 — Conversion : JSON structuré → Nodes RAG
    # ─────────────────────────────────────────────────────────────────────────
    def _structured_to_rag_nodes(
        self,
        structured: dict,
        filename: str,
        theme: str,
        country: str,
    ) -> dict:
        """
        Convertit le JSON structuré (document_metadata + clauses)
        en format RAG nodes attendu par data_processed/.
        """
        meta    = structured.get("document_metadata", {})
        clauses = structured.get("clauses", [])
        law_name = meta.get("title", filename)

        rag_nodes = []
        for i, clause in enumerate(clauses):
            clause_id    = clause.get("clause_id", f"Section_{i}")
            full_text    = clause.get("full_text", "")
            summary      = clause.get("title_or_summary", "")
            page_range   = clause.get("page_range", [1])
            cross_refs   = clause.get("cross_references", [])
            parent_id    = clause.get("parent_id")
            level        = clause.get("level", 1)
            is_footnote  = clause.get("is_footnote", False)
            ext_ref      = clause.get("external_reference")

            rag_node = {
                "node_id":         f"{country}_STRUCT_{Path(filename).stem}_{i}",
                "text":            full_text if full_text else f"# {clause_id}",
                "summary":         summary   if summary   else f"Extrait de {filename} ({theme})",
                "country":         country,
                "law_name":        law_name,
                "authority_level": 0.9 if level == 1 else 0.6,
                "valid_from":      meta.get("date", "Unknown"),
                "valid_to":        None,
                "metadata": {
                    "theme":            theme,
                    "source_file":      filename,
                    "clause_id":        clause_id,
                    "parent_id":        parent_id,
                    "level":            level,
                    "page_range":       page_range,
                    "cross_references": cross_refs,
                    "external_reference": ext_ref,
                    "is_footnote":      is_footnote,
                    "document_title":   law_name,
                    "country":          country,
                },
                "qa_support": [],
            }
            rag_nodes.append(rag_node)

        return {"nodes": rag_nodes}

    # ─────────────────────────────────────────────────────────────────────────
    # POINT D'ENTRÉE PRINCIPAL
    # ─────────────────────────────────────────────────────────────────────────
    def parse_file(
        self,
        file_path: Path,
        theme: str = None,
        country: str = None,
        progress_callback = None,
    ) -> tuple[Path, Path]:
        """
        Pipeline complet :
        1. LlamaParse  → Markdown  (data/parsed/<theme>_<country>.md)
        2. Groq LLM    → JSON structuré (data/structured/<theme>_<country>_structured.json)
        3. Post-processing dataoptimise (nettoyage, normalisation, fusion, validation)
        4. Conversion  → Nodes RAG (data_processed/<theme>_<country>_processed.json)
        5. Archive     → data/archive/

        Retourne (md_output_path, json_output_path).
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Le fichier {file_path} n'existe pas.")

        logger.info(f"─── Début du traitement : {file_path.name} ───")
        t_start = time.time()
        llamaparse_duration = 0.0

        if progress_callback:
            progress_callback(10, f"⏳ Étape 1/4 : Parsing des documents via LlamaParse en cours... [{file_path.name}]")

        # Thème / pays
        if theme and country:
            logger.info(f"Thème fourni : '{theme}' | Pays fourni : '{country}'")
        else:
            theme, country = self._parse_theme_country(file_path.name)
            logger.info(f"Thème extrait : '{theme}' | Pays extrait : '{country}'")

        safe_t = theme.replace(" ", "_")
        safe_c = country.replace(" ", "_")

        # ── Fichier JSON pré-structuré : chemin direct ──────────────────────
        if file_path.suffix.lower() == ".json":
            logger.info("Fichier JSON pré-structuré détecté → traitement direct.")
            if progress_callback:
                progress_callback(10, "Lecture du fichier JSON structuré...")
            return self._process_structured_json_file(file_path, theme, country, safe_t, safe_c)

        # ── ÉTAPE 1 : LlamaParse → Markdown ─────────────────────────────────
        if progress_callback:
            progress_callback(10, f"⏳ Étape 1/4 : Parsing des documents via LlamaParse en cours... [{file_path.name}]")
        t_llama_start = time.time()
        markdown_content = self._llamaparse_to_markdown(file_path)
        llamaparse_duration = time.time() - t_llama_start

        md_filename   = f"{safe_t}_{safe_c}.md"
        md_output_path = PARSED_DIR / md_filename
        with open(md_output_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        logger.info(f"Markdown enregistré : {md_output_path}")

        if progress_callback:
            progress_callback(30, "✅ Extraction brute LlamaParse réussie.")

        # ── ÉTAPE 2 : LlamaCloud Extraction → JSON structuré ─────────────────
        if progress_callback:
            progress_callback(40, "⏳ Étape 2/4 : Structuration des données via LlamaCloud...")
        t_groq_start = time.time()
        
        try:
            if not self.api_key:
                raise ValueError("LLAMA_CLOUD_API_KEY non configurée.")
            logger.info("Tentative d'extraction avec LlamaCloud...")
            structured = self._extract_structure_with_llamacloud(
                file_path, file_path.name, theme, country, progress_callback=progress_callback
            )
        except Exception as e:
            logger.warning(f"L'extraction LlamaCloud a échoué ou n'est pas configurée : {e}. Utilisation de Groq LLM en fallback.")
            if progress_callback:
                progress_callback(45, "⏳ Étape 2/4 : Structuration des données via Groq LLM (Fallback)...")
            structured = extract_structure_with_llm(
                markdown_content=markdown_content,
                filename=file_path.name,
                theme=theme,
                country=country,
                groq_api_key=GROQ_API_KEY,
                model=GROQ_MODEL,
                progress_callback=progress_callback
            )
            
        groq_duration = time.time() - t_groq_start

        # S'assurer que le metadata pointe bien sur le bon fichier / pays / thème
        structured.setdefault("document_metadata", {})
        structured["document_metadata"].setdefault("filename", file_path.name)
        structured["document_metadata"].setdefault("country",  country)
        structured["document_metadata"].setdefault("theme",    theme)

        # Enrichir document_origin pour chaque clause
        for clause in structured.get("clauses", []):
            clause["document_origin"] = f"{file_path.name} - {country}/{theme}"

        if progress_callback:
            progress_callback(65, "✅ Schéma JSON généré et validé avec succès.")

        # ── ÉTAPE 3 : Post-processing logique (dataoptimise) ────────────────
        if progress_callback:
            progress_callback(70, "⏳ Étape 3/4 : Exécution du traitement de données 'dataoptimise' (Nettoyage, normalisation et enrichissement)...")

        t_post_start = time.time()
        clauses = structured.get("clauses", [])
        clauses_before_merge = len(clauses)
        original_char_count = len(markdown_content)

        for clause in clauses:
            clause["full_text"] = clean_text(clean_spaced_text(correct_hyphenation(clause.get("full_text", ""))))
            clause["title_or_summary"] = clean_text(clause.get("title_or_summary", ""))

        # Fusion des fragments trop courts
        clauses = merge_short_clauses(clauses, min_length=150)
        clauses_after_merge = len(clauses)
        structured["clauses"] = clauses

        # Validation des cross_references
        validate_cross_references(clauses)

        # Sauvegarder dans STRUCTURED_DIR selon le schéma strict {"clauses": [...]}
        clean_structured = {
            "clauses": clauses
        }

        struct_filename    = f"{safe_t}_{safe_c}_structured.json"
        struct_output_path = STRUCTURED_DIR / struct_filename
        with open(struct_output_path, "w", encoding="utf-8") as f:
            json.dump(clean_structured, f, ensure_ascii=False, indent=2)
        logger.info(f"JSON structuré enregistré (strict schema) : {struct_output_path}")

        clean_char_count = sum(len(clause.get("full_text", "")) for clause in clauses)
        postprocess_duration = time.time() - t_post_start

        if progress_callback:
            progress_callback(85, "✅ Post-processing terminé.")

        # ── ÉTAPE 4 : Conversion → Nodes RAG ────────────────────────────────
        if progress_callback:
            progress_callback(90, "⏳ Étape 4/4 : Injection vectorielle dans le système RAG...")
        rag_data = self._structured_to_rag_nodes(structured, file_path.name, theme, country)

        # Inject automation metrics
        token_metrics = structured.get("metrics", {}).get("tokens", {"prompt_tokens": 0, "completion_tokens": 0, "estimated_cost_usd": 0.0})
        rag_data["metrics"] = {
            "tokens": token_metrics,
            "quality": {
                "original_char_count": original_char_count,
                "clean_char_count": clean_char_count,
                "characters_removed": max(0, original_char_count - clean_char_count),
                "clauses_before_merge": clauses_before_merge,
                "clauses_after_merge": clauses_after_merge,
                "clauses_merged_count": max(0, clauses_before_merge - clauses_after_merge)
            },
            "performance": {
                "llamaparse_duration_sec": round(llamaparse_duration, 2),
                "groq_duration_sec": round(groq_duration, 2),
                "postprocess_duration_sec": round(postprocess_duration, 2),
                "total_duration_sec": round(time.time() - t_start, 2)
            }
        }

        json_filename   = f"{safe_t}_{safe_c}_processed.json"
        json_output_path = DATA_PROCESSED_DIR / json_filename
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(rag_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Nodes RAG enregistrés : {json_output_path} ({len(rag_data['nodes'])} nodes)")

        # ── DÉTECTION DES CONFLITS JURIDIQUES (Abrogation / Contradictions) ──
        conflicts = []
        try:
            logger.info("Analyse et détection des conflits juridiques inter-documents...")
            conflicts = detect_legal_conflicts(rag_data["nodes"], DATA_PROCESSED_DIR, theme, country, GROQ_API_KEY, GROQ_MODEL)
            if conflicts:
                conflict_path = DATA_PROCESSED_DIR / f"{safe_t}_{safe_c}_conflicts.json"
                with open(conflict_path, "w", encoding="utf-8") as f_cf:
                    json.dump(conflicts, f_cf, ensure_ascii=False, indent=2)
                logger.info(f"Conflits juridiques détectés et enregistrés : {conflict_path} ({len(conflicts)} conflits)")
        except Exception as e:
            logger.warning(f"Impossible de détecter les conflits juridiques : {e}")

        # ── ÉTAPE 5 : Archivage ──────────────────────────────────────────────
        if progress_callback:
            progress_callback(98, "Archivage du document original...")
        self._archive_original_file(file_path)

        # Mettre à jour les graphes de connaissances
        self._update_knowledge_graphs(theme, country)

        if progress_callback:
            progress_callback(100, f"🎉 Ingestion complète terminée avec succès ! Le document est prêt à être interrogé.")
        logger.info(f"─── Traitement terminé avec succès : {file_path.name} ───")
        return md_output_path, json_output_path, conflicts

    # ─────────────────────────────────────────────────────────────────────────
    # Traitement d'un JSON pré-structuré (sans LlamaParse ni LLM)
    # ─────────────────────────────────────────────────────────────────────────
    def _process_structured_json_file(
        self,
        file_path: Path,
        theme: str,
        country: str,
        safe_t: str,
        safe_c: str,
    ) -> tuple[Path, Path]:
        """
        Traite un fichier JSON déjà structuré (format clauses ou nodes).
        Génère le Markdown de prévisualisation + les nodes RAG.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Format "nodes" brut — copie directe
        if "nodes" in data and "clauses" not in data:
            logger.info("Format RAG 'nodes' brut détecté → copie directe.")
            json_filename    = f"{safe_t}_{safe_c}_processed.json"
            json_output_path = DATA_PROCESSED_DIR / json_filename
            with open(json_output_path, "w", encoding="utf-8") as f_out:
                json.dump(data, f_out, ensure_ascii=False, indent=2)

            # Markdown minimal
            md_filename    = f"{safe_t}_{safe_c}.md"
            md_output_path = PARSED_DIR / md_filename
            md_content     = f"# {theme} - {country}\n\n"
            for node in data.get("nodes", []):
                md_content += f"## {node.get('metadata', {}).get('clause_id', 'Section')}\n\n{node.get('text', '')}\n\n"
            with open(md_output_path, "w", encoding="utf-8") as f_md:
                f_md.write(md_content)

            self._archive_original_file(file_path)
            return md_output_path, json_output_path

        # Format "clauses" (schéma structuré)
        clauses = data.get("clauses", [])
        if not clauses:
            raise ValueError("Le fichier JSON ne contient ni clé 'clauses' ni clé 'nodes' valide.")

        logger.info(f"{len(clauses)} clause(s) détectée(s) dans le JSON.")

        # Sauvegarder comme JSON structuré
        struct_filename    = f"{safe_t}_{safe_c}_structured.json"
        struct_output_path = STRUCTURED_DIR / struct_filename
        with open(struct_output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # Convertir en nodes RAG
        rag_data = self._structured_to_rag_nodes(data, file_path.name, theme, country)

        json_filename    = f"{safe_t}_{safe_c}_processed.json"
        json_output_path = DATA_PROCESSED_DIR / json_filename
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(rag_data, f, ensure_ascii=False, indent=2)

        # ── DÉTECTION DES CONFLITS JURIDIQUES (Abrogation / Contradictions) ──
        conflicts = []
        try:
            logger.info("Analyse et détection des conflits juridiques inter-documents...")
            conflicts = detect_legal_conflicts(rag_data["nodes"], DATA_PROCESSED_DIR, theme, country, GROQ_API_KEY, GROQ_MODEL)
            if conflicts:
                conflict_path = DATA_PROCESSED_DIR / f"{safe_t}_{safe_c}_conflicts.json"
                with open(conflict_path, "w", encoding="utf-8") as f_cf:
                    json.dump(conflicts, f_cf, ensure_ascii=False, indent=2)
                logger.info(f"Conflits juridiques détectés et enregistrés : {conflict_path} ({len(conflicts)} conflits)")
        except Exception as e:
            logger.warning(f"Impossible de détecter les conflits juridiques : {e}")

        # Générer le Markdown de prévisualisation
        md_lines = [f"# {theme} - {country}\n"]
        for clause in clauses:
            cid  = clause.get("clause_id", "")
            summ = clause.get("title_or_summary", "")
            text = clause.get("full_text", "")
            md_lines.append(f"## {cid}")
            if summ:
                md_lines.append(f"*{summ}*")
            if text:
                md_lines.append(text)
            md_lines.append("")

        md_filename    = f"{safe_t}_{safe_c}.md"
        md_output_path = PARSED_DIR / md_filename
        with open(md_output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
        logger.info(f"Markdown de prévisualisation : {md_output_path}")

        self._archive_original_file(file_path)
        
        # Mettre à jour les graphes de connaissances
        self._update_knowledge_graphs(theme, country)
        
        logger.info(f"─── Traitement terminé : {file_path.name} ───")
        return md_output_path, json_output_path, conflicts

    # ─────────────────────────────────────────────────────────────────────────
    # Mise à jour des graphes de connaissances (Multi-Graph Knowledge Layer)
    # ─────────────────────────────────────────────────────────────────────────
    def _update_knowledge_graphs(self, theme: str, country: str):
        """Met à jour les graphes de connaissances locaux avec le nouveau document."""
        try:
            logger.info("Intégration au Multi-Graph Knowledge Layer...")
            from legal_rag.data_loader import ProcessedDataLoader
            from legal_rag.graph_builder import GraphBuilder
            
            loader = ProcessedDataLoader(DATA_PROCESSED_DIR)
            doc_struct = loader.load_document(theme, country)
            if doc_struct:
                builder = GraphBuilder()
                # 1. Graphe Structural & Citation
                builder.create_lexical_graph(doc_struct)
                # 2. Graphe de Définitions
                builder.create_definitions_graph(doc_struct)
                logger.info("Multi-Graph Knowledge Layer mis à jour avec succès.")
            else:
                logger.warning("Structure du document introuvable pour la mise à jour des graphes.")
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour des graphes de connaissances : {e}", exc_info=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Archivage du fichier original
    # ─────────────────────────────────────────────────────────────────────────
    def _archive_original_file(self, file_path: Path):
        archive_path = ARCHIVE_DIR / file_path.name
        if archive_path.exists():
            os.remove(archive_path)
        shutil.move(str(file_path), str(archive_path))
        logger.info(f"Fichier original archivé : {archive_path}")
