"""
src/monitor.py
==============
Module de surveillance Internet des sources juridiques officielles.

Strategie : utilise Google News RSS (fiable, toujours accessible) avec des
requetes ciblees par domaine juridique, plus des sources institutionnelles
en fallback quand elles sont accessibles.

Chaque resultat est un dict standardise "LegalUpdate".
"""

import hashlib
import json
import logging
import re
import time
import unicodedata
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sources Google News RSS — requetes ciblees par theme juridique
# On utilise le parametre when:7d pour limiter aux 7 derniers jours
# ---------------------------------------------------------------------------
_GN_BASE = "https://news.google.com/rss/search?hl=fr&gl=MA&ceid=MA:fr&when=7d&q="

LEGAL_SOURCES = [
    # --- Adala Maroc : Version Arabe officielle en direct ---
    {
        "name": "Adala Maroc - Projets de Lois",
        "country": "Maroc",
        "url": "https://adala.justice.gov.ma/ar/projects-of-laws",
        "type": "adala",
    },
    # --- Légifrance : Lois et Décrets (via recherche ciblée bypass Cloudflare) ---
    {
        "name": "Légifrance - Journal Officiel",
        "country": "France",
        "url": "https://news.google.com/rss/search?hl=fr&gl=FR&ceid=FR:fr&when=7d&q=site:legifrance.gouv.fr",
        "type": "rss",
    },
]

# ---------------------------------------------------------------------------
# Mots-cles signalant une loi critique pour une entreprise
# ---------------------------------------------------------------------------
CRITICAL_KEYWORDS = [
    # Environnement
    "environnement", "pollution", "emission", "carbone", "dechets", "RSE",
    "developpement durable", "biodiversite", "climat", "climate",
    # Fiscal
    "taxe", "impot", "fiscalite", "TVA", "cotisation", "exoneration",
    "finance", "comptabilite", "audit", "douane",
    # Travail
    "travail", "salaire", "emploi", "licenciement", "conge",
    "code du travail", "greve", "syndicat",
    # Commerce
    "commerce", "contrat", "marches publics", "concurrence", "OHADA",
    "investissement", "import", "export",
    # Numerique
    "numerique", "donnees personnelles", "RGPD", "cybersecurite",
    "intelligence artificielle",
    # Sante
    "sante", "securite", "hygiene", "norme",
    # Generiques loi
    "loi", "decret", "arrete", "ordonnance", "reglement", "directive",
    "journal officiel", "code", "reforme",
]

# ---------------------------------------------------------------------------
# Registre (mises a jour deja vues)
# ---------------------------------------------------------------------------

class UpdateRegistry:
    """Maintient un registre local des mises a jour deja traitees."""

    def __init__(self, registry_path: Path):
        self.path = registry_path
        self._seen: set[str] = set()
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self._seen = set(data.get("seen_ids", []))
                logger.info(f"Registre charge : {len(self._seen)} entrees connues.")
            except Exception as e:
                logger.warning(f"Impossible de lire le registre : {e}. Repartir de zero.")
                self._seen = set()

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {"seen_ids": sorted(self._seen),
                 "last_updated": datetime.now(timezone.utc).isoformat()},
                ensure_ascii=False, indent=2
            ),
            encoding="utf-8",
        )

    def is_new(self, uid: str) -> bool:
        return uid not in self._seen

    def mark_seen(self, uid: str):
        self._seen.add(uid)

    @property
    def count(self) -> int:
        return len(self._seen)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_id(title: str, date: str) -> str:
    raw = f"{title.lower().strip()}|{date.strip()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _norm(s: str) -> str:
    """Normalise une chaine (accents, casse) pour la comparaison."""
    return unicodedata.normalize("NFKD", s.lower()).encode("ascii", "ignore").decode()


def _compute_topics(text: str) -> list[str]:
    text_n = _norm(text)
    return [kw for kw in CRITICAL_KEYWORDS if _norm(kw) in text_n]


def _is_critical(topics: list) -> bool:
    return len(topics) > 0


# ---------------------------------------------------------------------------
# Mots-clés en arabe mappés vers des catégories françaises
# ---------------------------------------------------------------------------
ARABIC_KEYWORDS_MAP = {
    "environnement": ["بيئة", "تلوث", "تنمية مستدامة", "نفايات", "مناخ"],
    "fiscalité": ["ضريبة", "ضرائب", "رسوم", "جمارك", "مالية", "محاسبة", "ميزانية"],
    "travail": ["شغل", "عمل", "أجر", "أجور", "طرد", "إضراب", "نقابة"],
    "commerce": ["تجارة", "تجاري", "عقد", "صفقات عمومية", "منافسة", "استثمار", "شركة", "مقاولة"],
    "numérique": ["رقمي", "معطيات ذات طابع شخصي", "معطيات شخصية", "الأمن المعلوماتي", "الأمن السيبراني", "ذكاء اصطناعي"],
    "justice": ["قانون", "مرسوم", "قرار", "ظهير", "خبير", "خبراء", "محاماة", "عدول"],
}

def _compute_arabic_topics(text: str) -> list[str]:
    text_lower = text.lower()
    topics = []
    for topic, kw_list in ARABIC_KEYWORDS_MAP.items():
        for kw in kw_list:
            if kw in text_lower:
                topics.append(topic)
                break
    return topics


_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def _fetch(url: str, timeout: int = 15) -> Optional[str]:
    """Telecharge une URL et retourne le texte, ou None."""
    try:
        resp = requests.get(url, headers=_HTTP_HEADERS, timeout=timeout)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or "utf-8"
        return resp.text
    except Exception as e:
        logger.warning(f"[Monitor] Impossible de charger {url}: {e}")
        return None


def _parse_date(raw: str) -> str:
    """Parse une date de differents formats vers ISO 8601."""
    # RFC 2822 (RSS standard)
    try:
        return parsedate_to_datetime(raw).date().isoformat()
    except Exception:
        pass
    for fmt in ["%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d",
                "%d %b %Y", "%d/%m/%Y", "%B %d, %Y"]:
        try:
            return datetime.strptime(raw.strip(), fmt).date().isoformat()
        except Exception:
            pass
    return ""


# ---------------------------------------------------------------------------
# Parseur RSS / Atom (Google News et autres)
# ---------------------------------------------------------------------------

def _parse_rss_source(source: dict, registry: UpdateRegistry) -> list[dict]:
    """Parse un flux RSS/Atom."""
    updates = []
    content = _fetch(source["url"])
    if not content:
        return updates

    # Tenter xml d'abord, puis html.parser en fallback
    try:
        soup = BeautifulSoup(content, "xml")
    except Exception:
        soup = BeautifulSoup(content, "html.parser")

    items = soup.find_all("item") or soup.find_all("entry")
    if not items:
        logger.warning(f"[Monitor] Aucun item RSS pour {source['name']}")
        return updates

    for item in items[:30]:
        # Titre
        title_tag = item.find("title")
        title = title_tag.get_text(strip=True) if title_tag else ""
        if not title or len(title) < 10:
            continue

        # Lien — Google News met le lien comme texte apres la balise <link/>
        link_tag = item.find("link")
        href = ""
        if link_tag:
            href = link_tag.get("href", "") or link_tag.get_text(strip=True)
            # Pour Google News RSS, le lien est souvent en next_sibling
            if not href and link_tag.next_sibling:
                sibling = str(link_tag.next_sibling).strip()
                if sibling.startswith("http"):
                    href = sibling
        if not href:
            href = source["url"]

        # Date
        date_tag = (item.find("pubDate") or item.find("published")
                    or item.find("updated"))
        date_str = ""
        if date_tag:
            date_str = _parse_date(date_tag.get_text(strip=True))
        if not date_str:
            date_str = datetime.now(timezone.utc).date().isoformat()

        # Description
        desc_tag = item.find("description") or item.find("summary")
        excerpt = ""
        if desc_tag:
            desc_html = desc_tag.get_text(strip=True)
            excerpt_soup = BeautifulSoup(desc_html, "html.parser")
            excerpt = excerpt_soup.get_text(strip=True)[:300]
        if not excerpt:
            excerpt = title

        # Source (Google News inclut la source dans <source>)
        src_tag = item.find("source")
        src_name = src_tag.get_text(strip=True) if src_tag else source["name"]

        # Analyse de pertinence
        topics = _compute_topics(f"{title} {excerpt}")
        uid = _make_id(title, date_str)

        updates.append({
            "id": uid,
            "title": title,
            "date": date_str,
            "source": src_name,
            "url": href,
            "excerpt": excerpt,
            "topics": topics,
            "country": source["country"],
            "is_new": registry.is_new(uid),
            "is_critical": _is_critical(topics),
        })

    return updates


# ---------------------------------------------------------------------------
# Parseur HTML generique (fallback)
# ---------------------------------------------------------------------------

def _parse_html_source(source: dict, registry: UpdateRegistry) -> list[dict]:
    """Parseur HTML avec selecteurs CSS."""
    updates = []
    content = _fetch(source["url"])
    if not content:
        return updates

    soup = BeautifulSoup(content, "html.parser")
    sel = source.get("selectors", {})

    # Chercher les liens avec des mots-cles
    for a_tag in soup.find_all("a", href=True)[:100]:
        text = a_tag.get_text(strip=True)
        if not text or len(text) < 15:
            continue
        topics = _compute_topics(text)
        if not topics:
            continue
        href = a_tag["href"]
        if not href.startswith("http"):
            base = "/".join(source["url"].split("/")[:3])
            href = base + "/" + href.lstrip("/")
        uid = _make_id(text, "")
        updates.append({
            "id": uid,
            "title": text,
            "date": datetime.now(timezone.utc).date().isoformat(),
            "source": source["name"],
            "url": href,
            "excerpt": text,
            "topics": topics,
            "country": source["country"],
            "is_new": registry.is_new(uid),
            "is_critical": _is_critical(topics),
        })

    return updates


# ---------------------------------------------------------------------------
# Parseur Adala Maroc (Version Arabe via __NEXT_DATA__)
# ---------------------------------------------------------------------------

def _parse_adala_source(source: dict, registry: UpdateRegistry) -> list[dict]:
    """Parse le portail Adala du Maroc (projets de lois) depuis son Next.js state."""
    updates = []
    content = _fetch(source["url"])
    if not content:
        return updates

    try:
        soup = BeautifulSoup(content, "html.parser")
        next_data_tag = soup.find("script", id="__NEXT_DATA__")
        if not next_data_tag:
            logger.warning(f"[Monitor] Aucun tag __NEXT_DATA__ trouve sur {source['name']}")
            return updates

        data = json.loads(next_data_tag.string)
        resources = data.get("props", {}).get("pageProps", {}).get("resources", [])

        for item in resources:
            title = item.get("name")
            if not title:
                continue

            path = item.get("path")
            # Construire l'URL absolue du PDF
            href = f"https://adala.justice.gov.ma/api/{path}" if path else source["url"]

            # Extraction de la date
            created_at = item.get("createdAt")
            date_str = ""
            if created_at:
                # Format: "2026-04-22T09:11:55.740Z" -> "2026-04-22"
                date_str = created_at.split("T")[0]
            if not date_str:
                date_str = datetime.now(timezone.utc).date().isoformat()

            # Analyse de pertinence (mots-cles arabes)
            topics = _compute_arabic_topics(title)
            uid = _make_id(title, date_str)

            updates.append({
                "id": uid,
                "title": title,
                "date": date_str,
                "source": "Adala Maroc (Projets)",
                "url": href,
                "excerpt": f"Projet de loi officiel marocain en arabe : {title}",
                "topics": topics,
                "country": "Maroc",
                "is_new": registry.is_new(uid),
                "is_critical": len(topics) > 0,
            })
    except Exception as e:
        logger.error(f"[Monitor] Erreur lors du parsing du Next.js de Adala: {e}")

    return updates


# ---------------------------------------------------------------------------
# Interface publique
# ---------------------------------------------------------------------------

class LegalMonitor:
    """
    Surveille plusieurs sources legales Internet.

    Usage :
        monitor = LegalMonitor(registry_path=Path("data/legal_updates_registry.json"))
        new_updates = monitor.check_all()
    """

    def __init__(self, registry_path: Path, extra_sources: Optional[list] = None):
        self.registry = UpdateRegistry(registry_path)
        self.sources = LEGAL_SOURCES + (extra_sources or [])

    def check_source(self, source: dict) -> list[dict]:
        logger.info(f"[Monitor] Verification : {source['name']}")
        try:
            if source.get("type") == "rss":
                return _parse_rss_source(source, self.registry)
            elif source.get("type") == "adala":
                return _parse_adala_source(source, self.registry)
            else:
                return _parse_html_source(source, self.registry)
        except Exception as e:
            logger.error(f"[Monitor] Erreur pour {source['name']} : {e}")
            return []

    def check_all(self, only_new: bool = True, only_critical: bool = False) -> list[dict]:
        all_updates: list[dict] = []

        for source in self.sources:
            updates = self.check_source(source)
            logger.info(f"[Monitor]   -> {len(updates)} element(s) recuperes")
            all_updates.extend(updates)
            time.sleep(1.0)

        if only_new:
            all_updates = [u for u in all_updates if u.get("is_new")]
        if only_critical:
            all_updates = [u for u in all_updates if u.get("is_critical")]

        for upd in all_updates:
            self.registry.mark_seen(upd["id"])
        self.registry.save()

        logger.info(
            f"[Monitor] Scan termine : {len(all_updates)} mise(s) a jour "
            f"{'nouvelles' if only_new else 'totales'} detectee(s)."
        )
        return sorted(all_updates, key=lambda x: x.get("date", ""), reverse=True)

    def add_custom_source(self, name: str, country: str, url: str,
                          source_type: str = "rss",
                          selectors: Optional[dict] = None):
        entry = {"name": name, "country": country, "url": url, "type": source_type}
        if selectors:
            entry["selectors"] = selectors
        self.sources.append(entry)
        logger.info(f"[Monitor] Source ajoutee : {name} ({source_type.upper()})")
