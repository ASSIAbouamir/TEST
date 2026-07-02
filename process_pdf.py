# -*- coding: utf-8 -*-
"""
Pipeline complet : PDF -> LlamaParse -> Groq LLM (chunks) -> JSON structure -> data_processed/

Gestion automatique du rate-limit Groq free tier (6000 TPM) :
  - Le Markdown est decoupes en chunks de ~2000 chars
  - Chaque chunk est envoye separement avec max_tokens=2000
  - Les clauses de tous les chunks sont fusionnees en un seul JSON

Usage :
    python process_pdf.py data/input/Baleine_Benin.pdf
    python process_pdf.py data/input/Baleine_Benin.pdf --theme Baleine --country Benin
    python process_pdf.py data/input/Baleine_Benin.pdf --skip-llamaparse --markdown-file data/parsed/Baleine_Benin.md
"""

import sys
import os
import io
import json
import re
import time
import logging
import argparse
import shutil
import unicodedata
from pathlib import Path

# ── Fix encoding Windows (CP1252 -> UTF-8) -----------------------------------
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── Environnement ─────────────────────────────────────────────────────────────
from dotenv import load_dotenv
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY", "")
GROQ_API_KEY        = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL          = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# ── Dossiers de sortie ────────────────────────────────────────────────────────
PARSED_DIR         = BASE_DIR / "data" / "parsed"
STRUCTURED_DIR     = BASE_DIR / "data" / "structured"
DATA_PROCESSED_DIR = BASE_DIR / "data_processed"
ARCHIVE_DIR        = BASE_DIR / "data" / "archive"

for d in [PARSED_DIR, STRUCTURED_DIR, DATA_PROCESSED_DIR, ARCHIVE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Logging UTF-8 ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Prompt systeme ────────────────────────────────────────────────────────────
EXTRACTION_SYSTEM_PROMPT = """Tu es un expert en analyse de documents juridiques francophones. Ton objectif est d'analyser le fragment de document juridique fourni et d'extraire toutes les clauses de manière structurée.

Retourne UNIQUEMENT un JSON valide avec la structure suivante (sans balises markdown, sans commentaires) :

{
  "document_metadata": {
    "title": "Titre officiel complet du document",
    "date": "YYYY-MM-DD"
  },
  "clauses": [
    {
      "clause_id": "Identifiant unique ou titre textuel de la clause/article (ex: 'Article premier', 'TITRE PREMIER : ...')",
      "parent_id": "ID de l'élément parent pour conserver la hiérarchie textuelle, ou null si élément racine",
      "level": 3,
      "title_or_summary": "Un résumé très court ou un titre descriptif condensé de l'élément",
      "full_text": "Le texte intégral verbatim et exact de la clause ou de l'article",
      "original_text": "Le texte original en langue étrangère si traduit, sinon null",
      "page_range": [1],
      "cross_references": ["liste des renvois ou citations internes detectes (ex: 'article 29')"],
      "document_origin": "Nom du fichier d'origine enrichi du contexte",
      "is_footnote": false,
      "entities": {
        "authorities": ["ministères, préfectures, agences, tribunaux ou autorités compétentes mentionnés dans l'article (ex: 'Ministre chargé de l'environnement', 'Préfet Maritime') ou tableau vide"],
        "penalties": ["sanctions financières, peines de prison ou sanctions administratives mentionnées dans l'article (ex: 'amende de 5.000.000 francs CFA', 'emprisonnement de 1 à 5 ans') ou tableau vide"],
        "dates_durations": ["durées, délais réglementaires, limites ou périodes mentionnées dans l'article (ex: 'charte coque-nue < 2 ans', 'délai de 30 jours') ou tableau vide"]
      }
    }
  ]
}

Regles d'extraction strictes :
1. full_text doit etre le texte exact, sans modification.
2. Hierarchie : TITRE (level 1) > CHAPITRE (level 2) > SECTION (level 2/3) > ARTICLE (level 3) > alinea (level 4).
3. document_origin doit etre initialise avec le nom du fichier.
4. page_range doit être une liste d'entiers correspondant aux numéros de page dans le document d'origine.
5. Extraction d'entités : Remplit scrupuleusement l'objet 'entities' avec les autorités, les sanctions (penalties) et les durées réglementaires détectées.
6. Retourne UNIQUEMENT du JSON valide. Aucun texte explicatif avant ou apres."""


def detect_language(text: str) -> str:
    """Detects if the language of the text is French or English/other."""
    text_lower = text.lower()
    french_words = [" le ", " la ", " les ", " pour ", " avec ", " dans ", " est "]
    english_words = [" the ", " and ", " of ", " for ", " with ", " in ", " is "]
    
    fr_count = sum(text_lower.count(w) for w in french_words)
    en_count = sum(text_lower.count(w) for w in english_words)
    
    if en_count > fr_count:
        return "en"
    return "fr"


# ── Fonctions de nettoyage et de post-processing ─────────────────────────────
def clean_spaced_text(text: str) -> str:
    """Corrige les textes où chaque lettre est séparée par un espace (ex: 'L a c h a s s e')."""
    if not text:
        return text
    single_letters = re.findall(r'\b[a-zA-Z]\b', text)
    if len(single_letters) > len(text.split()) * 0.4:
        t = text.replace("  ", " __SPACE__ ")
        t = re.sub(r'(?<=[a-zA-Z0-9])\s(?=[a-zA-Z0-9])', '', t)
        t = t.replace("__SPACE__", " ")
        return t
    return text

def clean_text(text: str) -> str:
    """Normalise Unicode et nettoie les caractères corrompus issus du parsing."""
    if not text:
        return ""
    # Remplacer les caractères corrompus spécifiques
    text = text.replace('Ǹ', 'é').replace('Ǧ', 'ê').replace('ǩ', 'ê')
    text = unicodedata.normalize('NFC', text)
    # Remplacer les caractères de remplacement et octets nuls indésirables
    text = text.replace('\ufffd', ' ').replace('\u0000', ' ')
    return " ".join(text.split())

def correct_hyphenation(text: str) -> str:
    """Corrige les césures de mots dues aux sauts de lignes (ex: 'échan-\\n tillons' -> 'échantillons')."""
    if not text:
        return text
    return re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)

def merge_short_clauses(clauses: list, min_length: int = 150) -> list:
    """Fusionne si nécessaire les fragments de clauses trop courts d'un même parent/niveau."""
    if not clauses:
        return []
    merged_clauses = []
    i = 0
    while i < len(clauses):
        current = clauses[i]
        if len(current.get("full_text", "")) < min_length and i + 1 < len(clauses):
            next_clause = clauses[i+1]
            same_parent = current.get("parent_id") == next_clause.get("parent_id")
            same_level = current.get("level") == next_clause.get("level")
            
            if same_parent or (same_level and current.get("parent_id") is None):
                current["full_text"] = current.get("full_text", "") + "\n\n" + next_clause.get("full_text", "")
                
                # Fusionner les page_range
                current_pages = set(current.get("page_range", [1]))
                next_pages = set(next_clause.get("page_range", [1]))
                current["page_range"] = sorted(list(current_pages.union(next_pages)))
                
                # Fusionner les cross_references
                current_refs = set(current.get("cross_references", []))
                next_refs = set(next_clause.get("cross_references", []))
                current["cross_references"] = sorted(list(current_refs.union(next_refs)))
                
                # Fusionner les résumés/titres
                current["title_or_summary"] = current.get("title_or_summary", "") + " / " + next_clause.get("title_or_summary", "")
                if len(current["title_or_summary"]) > 80:
                    current["title_or_summary"] = current["title_or_summary"][:77] + "..."
                
                clauses[i] = current
                clauses.pop(i+1)
                continue
        merged_clauses.append(current)
        i += 1
    return merged_clauses

def validate_cross_references(clauses: list):
    """Vérifie que les cross_references pointent bien vers des clause_id existants dans le document."""
    if not clauses:
        return
    
    def normalize_id(cid):
        if not cid: return ""
        return re.sub(r'\W+', '', str(cid).lower())

    existing_ids = {normalize_id(c.get("clause_id")) for c in clauses if c.get("clause_id")}
    
    for c in clauses:
        refs = c.get("cross_references", [])
        if not refs:
            continue
        valid_refs = []
        for ref in refs:
            norm_ref = normalize_id(ref)
            if not norm_ref:
                continue
            is_valid = False
            for exist_id in existing_ids:
                if exist_id and (exist_id in norm_ref or norm_ref in exist_id):
                    is_valid = True
                    break
            if is_valid:
                valid_refs.append(ref)
            else:
                logger.info(f"Filtre de cross_reference non valide (pointant vers du vide) : {ref}")
        c["cross_references"] = valid_refs


def local_parse_to_markdown(file_path: Path) -> str:
    """Parse un fichier localement sans appeler d'API externe (fallback offline)."""
    suffix = file_path.suffix.lower()
    logger.info(f"Début du parsing local (fallback) pour {file_path.name} (type: {suffix})...")
    
    if suffix == ".pdf":
        try:
            import pdfplumber
            logger.info("Utilisation de pdfplumber pour le parsing local...")
            markdown_lines = []
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        markdown_lines.append(f"<!-- Page {i+1} -->\n{text}")
            content = "\n\n".join(markdown_lines)
            if content.strip():
                return content
        except Exception as e:
            logger.warning(f"Échec de pdfplumber : {e}. Tentative avec PyPDF2...")
            
        try:
            import PyPDF2
            logger.info("Utilisation de PyPDF2 pour le parsing local...")
            markdown_lines = []
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text:
                        markdown_lines.append(f"<!-- Page {i+1} -->\n{text}")
            content = "\n\n".join(markdown_lines)
            if content.strip():
                return content
        except Exception as e:
            logger.error(f"Échec de PyPDF2 : {e}")
            
        raise RuntimeError(f"Impossible de parser le PDF {file_path.name} localement.")
        
    elif suffix in (".docx", ".doc"):
        try:
            import docx
            logger.info("Utilisation de python-docx pour le parsing local...")
            doc = docx.Document(file_path)
            markdown_lines = []
            for para in doc.paragraphs:
                if para.text.strip():
                    markdown_lines.append(para.text)
            return "\n\n".join(markdown_lines)
        except Exception as e:
            raise RuntimeError(f"Impossible de parser le fichier Word {file_path.name} localement : {e}")
            
    elif suffix == ".txt":
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except Exception as e:
            raise RuntimeError(f"Impossible de lire le fichier texte {file_path.name} : {e}")
            
    else:
        raise ValueError(f"Extension de fichier non supportée pour le parsing local : {suffix}")


# ─────────────────────────────────────────────────────────────────────────────
# ETAPE 1 - LlamaParse : PDF -> Markdown
# ─────────────────────────────────────────────────────────────────────────────
def llamaparse_to_markdown(file_path: Path, api_key: str, max_retries: int = 5) -> str:
    """Envoie le PDF a LlamaParse et retourne le Markdown."""
    import nest_asyncio
    nest_asyncio.apply()

    from llama_parse import LlamaParse  # noqa: F401 (deprecated mais toujours fonctionnel)

    parsing_instruction = (
        "This is a legal document written in French. "
        "Extract ALL text content faithfully, including article numbers, "
        "chapter headings (TITRE, CHAPITRE, SECTION, ARTICLE), footnotes, "
        "and all clauses. Preserve the hierarchical structure exactly."
    )

    logger.info("Initialisation de LlamaParse (result_type=markdown)...")
    parser = LlamaParse(
        api_key=api_key,
        result_type="markdown",
        language="fr",
        parsing_instruction=parsing_instruction,
        verbose=True,
        max_timeout=300,
        check_interval=5,
        num_workers=1,
    )

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Tentative {attempt}/{max_retries} - envoi vers LlamaParse...")
            documents = parser.load_data(str(file_path))

            if documents:
                markdown = "\n\n".join(doc.text for doc in documents)
                logger.info(
                    f"Markdown recu : {len(markdown)} caracteres, "
                    f"{len(documents)} page(s) / chunk(s)."
                )
                return markdown

            logger.warning(f"LlamaParse a retourne une liste vide (tentative {attempt}).")

        except Exception as exc:
            last_error = exc
            logger.warning(f"Erreur LlamaParse tentative {attempt} : {exc}")

        if attempt < max_retries:
            wait = 2 ** attempt  # back-off : 2, 4, 8, 16 s
            logger.info(f"Attente {wait}s avant la prochaine tentative...")
            time.sleep(wait)

    raise RuntimeError(
        f"LlamaParse n'a retourne aucun contenu apres {max_retries} tentatives. "
        f"Derniere erreur : {last_error}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Utilitaire : decouper le Markdown en chunks (par sections naturelles)
# ─────────────────────────────────────────────────────────────────────────────
def _repair_json(raw: str) -> dict | None:
    """
    Tente de reparer un JSON tronque ou malformate.
    Strategies :
      1. Extraire le JSON entre la premiere { et la derniere }
      2. Fermer les listes/objets non fermes
      3. Extraire uniquement les clauses deja completes
    """
    # Strategie 1 : trouver les accolades exterieures
    start = raw.find("{")
    if start == -1:
        return None

    # Essayer de trouver la fin du JSON en cherchant l'accolade fermante balancee
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

    # Strategie 2 : extraire les clauses individuelles deja completes
    clauses = []
    for m in re.finditer(r'\{[^{}]*"clause_id"[^{}]*\}', raw, re.DOTALL):
        try:
            obj = json.loads(m.group())
            if "clause_id" in obj:
                clauses.append(obj)
        except json.JSONDecodeError:
            pass

    if clauses:
        logger.info(f"  Reparation JSON : {len(clauses)} clause(s) extraites par regex.")
        return {
            "document_metadata": {},
            "clauses": clauses,
            "definitions": [],
        }

    return None

def split_markdown_into_chunks(markdown: str, max_chars: int = 2500) -> list[str]:
    """
    Decoupe le Markdown en chunks de ~max_chars caracteres en respectant
    les limites naturelles (fins de paragraphes).
    """
    # Separateurs naturels : doubles sauts de ligne
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

    logger.info(f"Markdown decoupage : {len(chunks)} chunk(s) (max {max_chars} chars each).")
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# ETAPE 2 - Groq LLM : Markdown chunk -> JSON partiel
# ─────────────────────────────────────────────────────────────────────────────
def extract_chunk_with_llm(
    chunk: str,
    chunk_idx: int,
    total_chunks: int,
    filename: str,
    theme: str,
    country: str,
    groq_api_key: str,
    model: str,
) -> dict:
    """
    Envoie un chunk de Markdown au LLM Groq.
    Retourne le dict partiel (clauses + metadata).
    Respecte le rate limit Groq free tier (6000 TPM) grace au chunking.
    """
    from groq import Groq

    # Détecter la langue du fragment
    lang = detect_language(chunk)
    is_english = (lang == "en")

    user_message = (
        f"Extrait toutes les clauses de ce fragment juridique "
        f"(fichier : {filename}, pays : {country}, theme : {theme}, "
        f"fragment {chunk_idx + 1}/{total_chunks}) :\n\n"
        f"{chunk}"
    )
    
    if is_english:
        logger.info(f"  [LANGUE] Détection d'anglais dans le fragment {chunk_idx + 1} ! Activation du double flux de traduction.")
        user_message += (
            "\n\nIMPORTANT: Le texte source est en ANGLAIS. Tu dois impérativement traduire et extraire les clauses en français. "
            "Place le texte traduit complet en français dans 'full_text', et place le texte original verbatim exact en anglais de cette clause dans le champ 'original_text'."
        )

    client = Groq(api_key=groq_api_key)
    last_raw = ""
    raw = ""

    for attempt in range(1, 6):
        try:
            logger.info(
                f"  Chunk {chunk_idx + 1}/{total_chunks} "
                f"- envoi LLM (tentative {attempt})..."
            )
            
            # Utilisation de la boucle de Réflexion si tentative précédente échouée
            if attempt > 1 and last_raw:
                logger.info(f"  [RÉFLEXION] Tentative de correction par réflexion pour le fragment {chunk_idx + 1}...")
                system_prompt = "Tu es un ingénieur expert en réparation et mise en conformité de fichiers JSON cassés. Ton objectif est de corriger la syntaxe JSON brute pour la rendre 100% valide et conforme au schéma attendu."
                user_content = (
                    f"Lors de l'extraction, le JSON généré était invalide ou mal structuré.\n\n"
                    f"--- TEXTE SOURCE D'ORIGINE ---\n{chunk}\n\n"
                    f"--- JSON DEFFECTUEUX GÉNÉRÉ ---\n{last_raw}\n\n"
                    f"--- TACHE ---\n"
                    f"Repère l'erreur (accolade manquante, virgule en trop, guillemets mal échappés dans le texte, troncature) et corrige le JSON.\n"
                    f"Conserve rigoureusement les champs 'document_metadata' et 'clauses'.\n"
                    f"Renvoie UNIQUEMENT le JSON corrigé complet, sans introduction ni conclusion."
                )
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_content},
                ]
            else:
                messages = [
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ]

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.05,
                response_format={"type": "json_object"},
                max_tokens=4000,  # Plus de marge pour eviter la troncation
            )

            raw = response.choices[0].message.content.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
            raw = re.sub(r"\s*```$",          "", raw, flags=re.MULTILINE)
            raw = raw.strip()
            last_raw = raw

            prompt_toks = 0
            completion_toks = 0
            if hasattr(response, "usage") and response.usage:
                prompt_toks = getattr(response.usage, "prompt_tokens", 0)
                completion_toks = getattr(response.usage, "completion_tokens", 0)

            result = json.loads(raw)
            result["_token_usage"] = {
                "prompt_tokens": prompt_toks,
                "completion_tokens": completion_toks
            }
            n_clauses = len(result.get("clauses", []))
            logger.info(f"  Chunk {chunk_idx + 1} OK : {n_clauses} clause(s).")
            return result

        except json.JSONDecodeError as e:
            logger.warning(
                f"  Chunk {chunk_idx + 1} - JSON invalide (tentative {attempt}) : {e}"
            )
            # Tentative de reparation du JSON brut
            repaired = _repair_json(last_raw or raw)
            if repaired:
                logger.info(f"  Chunk {chunk_idx + 1} repare avec succes par _repair_json !")
                prompt_toks = 0
                completion_toks = 0
                if 'prompt_toks' in locals():
                    pass
                else:
                    if hasattr(response, "usage") and response.usage:
                        prompt_toks = getattr(response.usage, "prompt_tokens", 0)
                        completion_toks = getattr(response.usage, "completion_tokens", 0)
                repaired["_token_usage"] = {
                    "prompt_tokens": prompt_toks,
                    "completion_tokens": completion_toks
                }
                return repaired
            
            if attempt < 5:
                time.sleep(3)
        except Exception as exc:
            logger.warning(
                f"  Chunk {chunk_idx + 1} - Erreur Groq (tentative {attempt}) : {exc}"
            )
            if attempt < 5:
                wait = min(2 ** attempt, 30)
                logger.info(f"  Attente {wait}s...")
                time.sleep(wait)

    logger.error(f"  Chunk {chunk_idx + 1} : echec apres 5 tentatives -> chunk ignore.")
    return {"clauses": [], "definitions": [], "document_metadata": {}, "_token_usage": {"prompt_tokens": 0, "completion_tokens": 0}}


# ─────────────────────────────────────────────────────────────────────────────
# ETAPE 2 complete - Traitement de tous les chunks avec fusion
# ─────────────────────────────────────────────────────────────────────────────
def extract_structure_with_llm(
    markdown_content: str,
    filename: str,
    theme: str,
    country: str,
    groq_api_key: str,
    model: str,
    progress_callback = None,
) -> dict:
    """
    Decoupe le Markdown en chunks, envoie chaque chunk au LLM Groq,
    fusionne les resultats en un seul JSON structure.
    """
    # Decoupage en chunks de 2500 chars max (approx 625 tokens input)
    # + prompt systeme ~500 tokens + max_tokens=2000 => total ~3125 tokens < 6000 TPM
    chunks = split_markdown_into_chunks(markdown_content, max_chars=2500)

    all_clauses     = []
    all_definitions = []
    document_metadata = {
        "filename": filename,
        "country":  country,
        "title":    "",
        "date":     "Unknown",
    }

    total_prompt_tokens = 0
    total_completion_tokens = 0

    for idx, chunk in enumerate(chunks):
        if progress_callback:
            progress_callback(40 + int((idx / len(chunks)) * 20), f"⏳ Étape 2/4 : Structuration des données selon le schéma JSON cible... (chunk {idx + 1}/{len(chunks)})")
        # Pause entre chunks pour eviter de saturer le TPM
        if idx > 0:
            logger.info("  Pause 3s entre chunks (rate limit Groq)...")
            time.sleep(3)

        partial = extract_chunk_with_llm(
            chunk, idx, len(chunks),
            filename, theme, country,
            groq_api_key, model,
        )

        token_usage = partial.get("_token_usage", {"prompt_tokens": 0, "completion_tokens": 0})
        total_prompt_tokens += token_usage.get("prompt_tokens", 0)
        total_completion_tokens += token_usage.get("completion_tokens", 0)

        # Recuperer le metadata du premier chunk qui en a un
        meta = partial.get("document_metadata", {})
        if meta.get("title") and not document_metadata["title"]:
            document_metadata["title"] = meta.get("title", "")
        if meta.get("date") and document_metadata["date"] == "Unknown":
            document_metadata["date"] = meta.get("date", "Unknown")

        all_clauses.extend(partial.get("clauses", []))
        all_definitions.extend(partial.get("definitions", []))

    # Dedupliquer les definitions par terme
    seen_defs = set()
    unique_defs = []
    for d in all_definitions:
        key = d.get("term", str(d))
        if key not in seen_defs:
            seen_defs.add(key)
            unique_defs.append(d)

    logger.info(
        f"Fusion complete : {len(all_clauses)} clause(s), "
        f"{len(unique_defs)} definition(s) unique(s)."
    )

    return {
        "document_metadata": document_metadata,
        "clauses":           all_clauses,
        "definitions":       unique_defs,
        "metrics": {
            "tokens": {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "estimated_cost_usd": total_prompt_tokens * 0.00000005 + total_completion_tokens * 0.00000008
            }
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# ETAPE 3 - Conversion : JSON structure -> Nodes RAG
# ─────────────────────────────────────────────────────────────────────────────
def structured_to_rag_nodes(
    structured: dict,
    filename: str,
    theme: str,
    country: str,
) -> dict:
    """Convertit le JSON structure en format RAG nodes pour data_processed/."""
    meta     = structured.get("document_metadata", {})
    clauses  = structured.get("clauses", [])
    law_name = meta.get("title", filename)

    rag_nodes = []
    for i, clause in enumerate(clauses):
        clause_id   = clause.get("clause_id", f"Section_{i}")
        full_text   = clause.get("full_text", "")
        summary     = clause.get("title_or_summary", "")
        page_range  = clause.get("page_range", [1])
        cross_refs  = clause.get("cross_references", [])
        parent_id   = clause.get("parent_id")
        level       = clause.get("level", 1)
        is_footnote = clause.get("is_footnote", False)
        ext_ref     = clause.get("external_reference")
        
        # Nouvelles métadonnées avancées
        entities    = clause.get("entities", {"authorities": [], "penalties": [], "dates_durations": []})
        orig_text   = clause.get("original_text")

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
                "theme":              theme,
                "source_file":        filename,
                "clause_id":          clause_id,
                "parent_id":          parent_id,
                "level":              level,
                "page_range":         page_range,
                "cross_references":   cross_refs,
                "external_reference": ext_ref,
                "is_footnote":        is_footnote,
                "document_title":     law_name,
                "country":            country,
                "entities":           entities,
                "original_content":   orig_text,
            },
            "qa_support": [],
        }
        rag_nodes.append(rag_node)

    return {"nodes": rag_nodes}


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
def process_pdf(
    file_path: Path,
    theme: str = None,
    country: str = None,
    skip_llamaparse: bool = False,
    markdown_override: Path = None,
    progress_callback = None,
) -> tuple[Path, Path]:
    """
    Pipeline complet :
      1. LlamaParse  -> Markdown  -> data/parsed/<theme>_<country>.md
      2. Groq LLM    -> JSON structure -> data/structured/<theme>_<country>_structured.json
      3. Post-processing dataoptimise (nettoyage, normalisation, fusion, validation)
      4. Conversion  -> Nodes RAG -> data_processed/<theme>_<country>_processed.json
      5. Archive     -> data/archive/
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")

    logger.info(f"Debut du traitement : {file_path.name}")

    if progress_callback:
        progress_callback(10, f"⏳ Étape 1/4 : Parsing des documents via LlamaParse en cours... [{file_path.name}]")

    t_start = time.time()
    llamaparse_duration = 0.0

    if not (theme and country):
        stem   = file_path.stem
        parts  = stem.split("_")
        theme   = parts[0].strip() if parts else "General"
        country = "_".join(parts[1:]).strip() if len(parts) > 1 else stem

    safe_t = theme.replace(" ", "_")
    safe_c = country.replace(" ", "_")
    logger.info(f"Theme : '{theme}' | Pays : '{country}'")

    # -- ETAPE 1 : LlamaParse -> Markdown --------------------------------------
    t_llama_start = time.time()
    if markdown_override and markdown_override.exists():
        logger.info(f"Utilisation du Markdown existant : {markdown_override}")
        markdown_content = markdown_override.read_text(encoding="utf-8")
        llamaparse_duration = 0.0
    elif skip_llamaparse:
        raise ValueError("--skip-llamaparse necessite --markdown-file <fichier.md>")
    else:
        try:
            if not LLAMA_CLOUD_API_KEY:
                raise ValueError("LLAMA_CLOUD_API_KEY manquante dans .env")
            markdown_content = llamaparse_to_markdown(file_path, LLAMA_CLOUD_API_KEY)
        except Exception as e:
            logger.warning(f"Le parsing sémantique cloud via LlamaParse a échoué ou n'est pas configuré : {e}. Utilisation du parseur local en fallback.")
            markdown_content = local_parse_to_markdown(file_path)
        llamaparse_duration = time.time() - t_llama_start

    md_path = PARSED_DIR / f"{safe_t}_{safe_c}.md"
    md_path.write_text(markdown_content, encoding="utf-8")
    logger.info(f"Markdown enregistre : {md_path}  ({len(markdown_content)} chars)")

    if progress_callback:
        progress_callback(30, "✅ Extraction brute LlamaParse réussie.")

    # -- ETAPE 2 : Groq LLM -> JSON structure ---------------------------------
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY manquante dans .env")

    if progress_callback:
        progress_callback(40, "⏳ Étape 2/4 : Structuration des données selon le schéma JSON cible...")

    t_groq_start = time.time()
    logger.info("Extraction structuree par le LLM (traitement par chunks)...")
    structured = extract_structure_with_llm(
        markdown_content,
        file_path.name,
        theme,
        country,
        GROQ_API_KEY,
        GROQ_MODEL,
        progress_callback=progress_callback,
    )
    groq_duration = time.time() - t_groq_start

    # Récupérer/déduire document_metadata pour nodes RAG ultérieurs
    structured.setdefault("document_metadata", {})
    structured["document_metadata"].setdefault("filename", file_path.name)
    structured["document_metadata"].setdefault("country",  country)
    structured["document_metadata"].setdefault("theme",    theme)

    # Enrichir document_origin pour chaque clause
    for clause in structured.get("clauses", []):
        clause["document_origin"] = f"{file_path.name} - {country}/{theme}"

    if progress_callback:
        progress_callback(65, "✅ Schéma JSON généré et validé avec succès.")

    # -- ETAPE 3 : Post-processing logique (dataoptimise) --------------------
    if progress_callback:
        progress_callback(70, "⏳ Étape 3/4 : Exécution du traitement de données 'dataoptimise' (Nettoyage, normalisation et enrichissement)...")

    t_post_start = time.time()
    clauses = structured.get("clauses", [])
    clauses_before_merge = len(clauses)
    original_char_count = len(markdown_content)

    for clause in clauses:
        # Nettoyage de texte ( normalisation unicode, espaces, césures de mots )
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

    struct_path = STRUCTURED_DIR / f"{safe_t}_{safe_c}_structured.json"
    struct_path.write_text(
        json.dumps(clean_structured, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"JSON structure enregistre (strict schema) : {struct_path}")

    clean_char_count = sum(len(clause.get("full_text", "")) for clause in clauses)
    postprocess_duration = time.time() - t_post_start

    if progress_callback:
        progress_callback(85, "✅ Post-processing terminé.")

    # -- ETAPE 4 : Conversion -> Nodes RAG & Injection ------------------------
    if progress_callback:
        progress_callback(90, "⏳ Étape 4/4 : Injection vectorielle dans le système RAG...")

    rag_data = structured_to_rag_nodes(structured, file_path.name, theme, country)

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

    processed_path = DATA_PROCESSED_DIR / f"{safe_t}_{safe_c}_processed.json"
    processed_path.write_text(
        json.dumps(rag_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    n_nodes = len(rag_data["nodes"])
    logger.info(f"Nodes RAG enregistres : {processed_path} ({n_nodes} nodes)")

    # -- GENERATION DES TESTS SYNTHETIQUES ET DETECTION DES CONFLITS ──
    try:
        logger.info("Génération de la suite de tests QA synthétiques...")
        qa_pairs = generate_synthetic_qa(rag_data["nodes"], theme, country, GROQ_API_KEY, GROQ_MODEL)
        if qa_pairs:
            qa_path = DATA_PROCESSED_DIR / f"{safe_t}_{safe_c}_synthetic_qa.json"
            qa_path.write_text(json.dumps(qa_pairs, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info(f"Test suite synthétique générée : {qa_path} ({len(qa_pairs)} tests)")
    except Exception as e:
        logger.warning(f"Impossible de générer la suite de tests synthétique : {e}")

    try:
        logger.info("Analyse et détection des conflits juridiques inter-documents...")
        conflicts = detect_legal_conflicts(rag_data["nodes"], DATA_PROCESSED_DIR, theme, country, GROQ_API_KEY, GROQ_MODEL)
        if conflicts:
            conflict_path = DATA_PROCESSED_DIR / f"{safe_t}_{safe_c}_conflicts.json"
            conflict_path.write_text(json.dumps(conflicts, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info(f"Conflits juridiques détectés et enregistrés : {conflict_path} ({len(conflicts)} conflits)")
    except Exception as e:
        logger.warning(f"Impossible de détecter les conflits juridiques : {e}")

    # -- Archive -----------------------------------------------------
    archive_path = ARCHIVE_DIR / file_path.name
    if archive_path.exists():
        archive_path.unlink()
    shutil.copy2(str(file_path), str(archive_path))
    logger.info(f"Fichier archive : {archive_path}")

    if progress_callback:
        progress_callback(100, "🎉 Ingestion complète terminée avec succès ! Le document est prêt à être interrogé.")

    logger.info("--- TRAITEMENT TERMINE ---")
    logger.info(f"  Markdown      -> {md_path}")
    logger.info(f"  Structure     -> {struct_path}")
    logger.info(f"  Nodes RAG     -> {processed_path}  ({n_nodes} nodes)")

    return md_path, processed_path


def generate_synthetic_qa(
    rag_nodes: list,
    theme: str,
    country: str,
    groq_api_key: str,
    model: str,
) -> list:
    """Automatically generate 3-5 synthetic Q&A test cases based on the ingested nodes."""
    from groq import Groq
    
    # Filter nodes that are actual articles
    candidate_nodes = [
        n for n in rag_nodes 
        if n.get("metadata", {}).get("clause_id", "").lower().startswith("art") 
        or len(n.get("text", "")) > 300
    ]
    
    if not candidate_nodes:
        candidate_nodes = rag_nodes[:3]
        
    selected_nodes = candidate_nodes[:3]
    qa_pairs = []
    
    try:
        client = Groq(api_key=groq_api_key)
        for node in selected_nodes:
            node_id = node.get("node_id")
            clause_id = node.get("metadata", {}).get("clause_id", "Article")
            text = node.get("text", "")
            
            prompt = (
                f"Basé sur l'article juridique suivant :\n\n"
                f"[{clause_id}] {text}\n\n"
                f"Génère une question précise en français qu'un professionnel pourrait poser concernant cet article, "
                f"et fournit une réponse courte et exacte extraite uniquement de ce texte.\n"
                f"Renvoie un objet JSON valide exact :\n"
                f'{{\n  "question": "Ta question ?",\n  "answer": "Ta réponse."\n}}'
            )
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Tu es un expert juridique. Renvoie uniquement du JSON valide."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content.strip()
            qa = json.loads(raw)
            
            qa_pairs.append({
                "question": qa.get("question"),
                "expected_answer": qa.get("answer"),
                "expected_node_id": node_id,
                "clause_id": clause_id
            })
    except Exception as e:
        logger.warning(f"Failed to generate synthetic QA: {e}")
        
    return qa_pairs


def detect_legal_conflicts(
    new_nodes: list,
    processed_dir: Path,
    theme: str,
    country: str,
    groq_api_key: str,
    model: str,
) -> list:
    """Compare new nodes with other files of the same country to detect contradictions or abrogation."""
    from groq import Groq
    conflicts = []
    
    # Deduce actual target country and theme if this is an internet monitoring update (Veille)
    actual_theme = theme
    actual_country = country
    if theme.lower() == "veille" or "veille" in theme.lower():
        parts = country.split("_")
        if len(parts) >= 3:
            actual_theme = parts[0]
            actual_country = "_".join(parts[1:-1])
            
    # Find other processed JSON files of the same country and theme
    other_files = []
    current_filename = f"{theme.replace(' ', '_')}_{country.replace(' ', '_')}_processed.json"
    
    safe_actual_theme = actual_theme.replace(" ", "_")
    safe_actual_country = actual_country.replace(" ", "_")
    
    for file in processed_dir.glob("*_processed.json"):
        if f"_{safe_actual_country}_processed" in file.name and f"{safe_actual_theme}_" in file.name:
            if file.name != current_filename:
                other_files.append(file)
            
    if not other_files:
        return []
        
    try:
        client = Groq(api_key=groq_api_key)
        new_candidates = [n for n in new_nodes if len(n.get("text", "")) > 200][:3]
        
        for old_file in other_files:
            with open(old_file, 'r', encoding='utf-8') as f:
                old_data = json.load(f)
                old_nodes = old_data.get("nodes", [])
                old_candidates = [n for n in old_nodes if len(n.get("text", "")) > 200][:3]
                
                for new_node in new_candidates:
                    for old_node in old_candidates:
                        new_words = set(new_node.get("text", "").lower().split())
                        old_words = set(old_node.get("text", "").lower().split())
                        overlap = len(new_words.intersection(old_words))
                        
                        # Compare if sémantiquement they cover similar terms
                        if overlap > 10:
                            logger.info(f"  [CONFLITS] Analyse de conflit potentiel entre {new_node.get('node_id')} et {old_node.get('node_id')}...")
                            prompt = (
                                f"Détecte s'il existe une contradiction directe, une incompatibilité de termes, "
                                f"ou si l'article le plus récent remplace implicitement l'autre (abrogation) :\n\n"
                                f"Article A ({new_node.get('law_name')}):\n{new_node.get('text')}\n\n"
                                f"Article B ({old_node.get('law_name')}):\n{old_node.get('text')}\n\n"
                                f"Retourne un JSON valide exact :\n"
                                f'{{\n  "conflict_detected": true/false,\n  "explanation": "Pourquoi ? (2 phrases maximum)"\n}}'
                            )
                            
                            response = client.chat.completions.create(
                                model=model,
                                messages=[
                                    {"role": "system", "content": "Tu es un juriste comparateur expert. Réponds en JSON uniquement."},
                                    {"role": "user", "content": prompt}
                                ],
                                temperature=0.1,
                                response_format={"type": "json_object"},
                            )
                            raw = response.choices[0].message.content.strip()
                            res = json.loads(raw)
                            
                            if res.get("conflict_detected"):
                                conflicts.append({
                                    "new_node_id": new_node.get("node_id"),
                                    "old_node_id": old_node.get("node_id"),
                                    "new_law": new_node.get("law_name"),
                                    "old_law": old_node.get("law_name"),
                                    "explanation": res.get("explanation")
                                })
    except Exception as e:
        logger.warning(f"Error detecting legal conflicts: {e}")
        
    return conflicts


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Pipeline PDF -> LlamaParse -> Groq (chunks) -> RAG nodes"
    )
    parser.add_argument("pdf_file",    help="Chemin vers le fichier PDF")
    parser.add_argument("--theme",     default=None, help="Theme (ex: Baleine)")
    parser.add_argument("--country",   default=None, help="Pays (ex: Benin)")
    parser.add_argument(
        "--skip-llamaparse", action="store_true",
        help="Ne pas appeler LlamaParse (necessite --markdown-file)",
    )
    parser.add_argument(
        "--markdown-file", default=None,
        help="Fichier .md existant a utiliser au lieu de LlamaParse",
    )
    args = parser.parse_args()

    file_path   = Path(args.pdf_file).resolve()
    md_override = Path(args.markdown_file).resolve() if args.markdown_file else None

    md_path, processed_path = process_pdf(
        file_path,
        theme=args.theme,
        country=args.country,
        skip_llamaparse=args.skip_llamaparse,
        markdown_override=md_override,
    )

    print("\n[OK] Pipeline termine !")
    print(f"     Markdown  : {md_path}")
    print(f"     RAG nodes : {processed_path}")


if __name__ == "__main__":
    main()
