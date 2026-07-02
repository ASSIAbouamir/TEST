import os
import json
import re
import unicodedata
from pathlib import Path
from typing import List, Dict, Any
try:
    from models import LegalNode
except ImportError:
    from scripts.models import LegalNode

class ExpertDistillerV4:
    """
    Production-ready legal data distiller.
    Handles Unicode cleaning, law title detection, date extraction,
    and context enrichment for RAG accuracy.
    """
    def __init__(self, data_dir: str = "data_old", output_dir: str = "data_processed"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data = {}

    def clean_spaced_text(self, text: str) -> str:
        """Fixes text where every letter is separated by a space (e.g., 'L a c h a s s e')."""
        if not text: return text
        import re
        # Si le texte contient beaucoup de lettres isolées suivies d'un espace
        single_letters = re.findall(r'\b[a-zA-Z]\b', text)
        if len(single_letters) > len(text.split()) * 0.4:
            # On tente de recoller les mots
            # On remplace les doubles espaces (vrais espaces entre mots) par un marqueur
            t = text.replace("  ", " __SPACE__ ")
            # On supprime les espaces simples entre lettres
            t = re.sub(r'(?<=[a-zA-Z0-9])\s(?=[a-zA-Z0-9])', '', t)
            # On restaure les vrais espaces
            t = t.replace("__SPACE__", " ")
            return t
        return text

    def clean_text(self, text: str) -> str:
        """Normalise Unicode et supprime les caractères corrompus."""
        if not text: return ""
        text = unicodedata.normalize('NFKD', text)
        text = text.replace('Ǹ', 'é').replace('Ǧ', 'ê').replace('ǩ', 'ê')
        text = text.replace('?', ' ').replace('', ' ')
        return " ".join(text.split())

    def identify_country(self, doc_meta: dict, file_path: str, clauses: list) -> str:
        """Identifie le pays à partir des métadonnées, du texte (priorité) ou du nom de fichier."""
        country = doc_meta.get("country", "")
        if not country or country == "Unknown":
            # 1. Scan étendu du texte (50 premières clauses) - PRIORITÉ MAXIMALE
            all_txt = " ".join([str(cl.get("full_text", "")) for cl in clauses[:50]]).lower()
            if "comores" in all_txt or "moroni" in all_txt: return "Comores"
            if "guinée" in all_txt or "guineen" in all_txt or "conakry" in all_txt: return "Guinée"
            if "togo" in all_txt or "togolais" in all_txt or "lomé" in all_txt: return "Togo"
            if "bénin" in all_txt or "cotonou" in all_txt or "porto-novo" in all_txt: return "Bénin"
            if "gabon" in all_txt or "libreville" in all_txt: return "Gabon"
            if "madagascar" in all_txt or "antananarivo" in all_txt: return "Madagascar"
            if "cameroun" in all_txt or "yaoundé" in all_txt: return "Cameroun"
            if "sénégal" in all_txt or "dakar" in all_txt: return "Sénégal"
            if "mauritan" in all_txt or "nouakchott" in all_txt: return "Mauritanie"
            if "tunisie" in all_txt or "tunis" in all_txt: return "Tunisie"
            if "congo" in all_txt or "brazzaville" in all_txt: return "Congo"
            
            # 2. Nom du fichier (Fallback)
            fname = os.path.basename(file_path).lower()
            codes = {
                "alg": "Algérie", "ben": "Bénin", "cmr": "Cameroun", 
                "com": "Comores", "con": "Congo", "dji": "Djibouti", 
                "gab": "Gabon", "gui": "Guinée", "mad": "Madagascar", "mau": "Mauritanie", 
                "mor": "Maroc", "sen": "Sénégal", "tog": "Togo", "tun": "Tunisie"
            }
            fname_clean = fname.replace("_", "").replace("-", "")
            for code, name in codes.items():
                if fname_clean.startswith(code): return name
                if name.lower() in fname: return name
                
        return country if country else "Unknown"

    def detect_law_name(self, text: str, current_law: str, clause_id: str) -> str:
        """Tente d'extraire le nom de la loi si celui-ci est 'Unknown'."""
        if current_law and current_law != "Unknown": return current_law
        patterns = [
            r"(ARRETE\s+N°?\s?[\d/]+[^;:\n]+)",
            r"(DECRET\s+N°?\s?[\d/]+[^;:\n]+)",
            r"(LOI\s+N°?\s?[\d/]+[^;:\n]+)",
            r"(ORDONNANCE\s+N°?\s?[\d/]+[^;:\n]+)"
        ]
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m: return m.group(1).strip()
        if clause_id and clause_id != "Unknown": return clause_id
        return "Unknown"

    def extract_date(self, text: str) -> str:
        """Extrait une date au format ISO (YYYY-MM-DD)."""
        if not text: return "Unknown"
        text = text.lower()
        months = {"janvier": "01", "fevrier": "02", "février": "02", "mars": "03", "avril": "04", "mai": "05", "juin": "06", 
                  "juillet": "07", "aout": "08", "août": "08", "septembre": "09", "octobre": "10", "novembre": "11", "decembre": "12", "décembre": "12"}
        
        match = re.search(r"(\d{1,2})(?:er|ere)?\s+([a-zéû]+)\s+(\d{4})", text)
        if match:
            day, mon, yr = match.group(1).zfill(2), months.get(match.group(2), "01"), match.group(3)
            return f"{yr}-{mon}-{day}"
        
        match_year = re.search(r"\b(19\d{2}|20[0-2]\d)\b", text)
        if match_year: return f"{match_year.group(1)}-01-01"
        return "Unknown"

    def process_file(self, theme: str, file_path: str):
        content = None
        for enc in ['utf-8', 'latin-1', 'cp1252']:
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    content = json.load(f)
                    break
            except: continue
        
        if not content: return

        clauses = content.get("clauses", content.get("data", {}).get("clauses", [])) if isinstance(content, dict) else content
        if not clauses: return

        doc_meta = content.get("document_metadata", content.get("metadata", {})) or {} if isinstance(content, dict) else {}
        country = self.identify_country(doc_meta, file_path, clauses)
        raw_law_name = doc_meta.get("law_name", "Unknown")
        
        detected_law = "Unknown"
        if raw_law_name and raw_law_name != "Unknown":
            detected_law = self.clean_text(raw_law_name)
        else:
            for c in clauses[:20]:
                title = self.detect_law_name(str(c.get("full_text", "")), "Unknown", str(c.get("clause_id", "")))
                if title != "Unknown":
                    detected_law = self.clean_text(title)
                    break

        output_filename = f"{theme}_{country}_processed.json"
        if output_filename not in self.processed_data:
            self.processed_data[output_filename] = {"nodes": []}

        # --- ENRICHISSEMENT GLOBAL POUR 100% DE PRÉCISION ---
        blob = " ".join([str(c.get("full_text", c.get("content", ""))) for c in clauses]).lower()
        found = [s for s in ["baleine", "cetace", "dauphin", "oiseau", "tortue", "mammifere", "hydrocarbure"] if s in blob]
        law_context = f"[INFO LOI: Concerne {', '.join(found)}] " if found else ""

        for clause in clauses:
            clause_id = str(clause.get("clause_id", "Unknown"))
            content = str(clause.get("full_text", clause.get("content", "")))
            
            # --- CHUNKING DÉSACTIVÉ POUR PRÉSERVER LE SENS JURIDIQUE ---
            full_text = law_context + self.clean_spaced_text(self.clean_text(content))
            if not full_text: continue
            
            valid_date = self.extract_date(full_text)
            year_part = valid_date.split("-")[0]
            safe_law_short = re.sub(r'[^\w]', '_', detected_law)[:20]
            safe_clause = re.sub(r'[^\w]', '_', clause_id)[:40]
            base_node_id = f"{country}_{year_part}_{safe_law_short}_{safe_clause}".replace("__", "_")

            node = LegalNode(
                node_id=base_node_id,
                text=full_text,
                summary=self.clean_text(clause.get("title_or_summary", clause_id)),
                country=country,
                law_name=detected_law,
                valid_from=valid_date,
                metadata={
                    "theme": theme, 
                    "source_file": os.path.basename(file_path), 
                    "clause_id": clause_id
                }
            )
            self.processed_data[output_filename]["nodes"].append(node.to_dict())

    def save_results(self):
        for filename, data in self.processed_data.items():
            out_path = self.output_dir / filename
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"Updated {out_path}: Total {len(data['nodes'])} nodes")

def main():
    distiller = ExpertDistillerV4()
    # Themes mapping
    themes = {
        "Baleine": ["Baleine", "cetace", "com167453", "DZ_2_95"],
        "Oiseaux marins": ["Oiseau"],
        "Rejet hydrocarbure": ["Rejet", "hydro"]
    }
    
    # Recherche récursive dans data_old
    for root, dirs, files in os.walk("data_old"):
        for f in files:
            if not f.endswith(".json"): continue
            assigned_theme = "Unknown"
            # On vérifie le thème dans le nom du fichier OU dans le chemin du dossier
            full_path = os.path.join(root, f)
            for theme, keywords in themes.items():
                if any(k.lower() in full_path.lower() for k in keywords):
                    assigned_theme = theme
                    break
            
            distiller.process_file(assigned_theme, full_path)
    
    distiller.save_results()

if __name__ == "__main__":
    main()
