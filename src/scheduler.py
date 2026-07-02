"""
src/scheduler.py
================
Orchestrateur planifié (scheduler) de la veille juridique.

Il coordonne :
  1. LegalMonitor  — scrape les sources Internet
  2. LegalNotifier — envoie les alertes
  3. LegalDocumentParser — importe automatiquement les nouveaux textes dans le RAG

Usage CLI :
  python main.py --monitor          → une passe de veille
  python main.py --schedule         → veille continue (intervalle configurable)
  python main.py --monitor --notify → veille + alertes en une passe

Configuration (.env) :
  MONITOR_INTERVAL_HOURS = 6     (toutes les N heures, défaut: 6)
  COMPANY_NAME           = Mon Entreprise SARL
  ALERT_MIN_TOPICS       = 1
"""

import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _get_interval_seconds() -> int:
    """Lit l'intervalle de surveillance depuis .env (en heures)."""
    hours = float(os.getenv("MONITOR_INTERVAL_HOURS", "6"))
    return int(hours * 3600)


def run_monitoring_cycle(
    base_dir: Path,
    notify: bool = True,
    only_critical: bool = False,
    import_to_rag: bool = False,
    extra_sources: Optional[list] = None,
) -> dict:
    """
    Exécute un cycle complet de veille juridique :
      1. Scrape toutes les sources
      2. Filtre les nouvelles mises à jour
      3. Envoie les alertes (si notify=True)
      4. Optionnellement importe dans le RAG (si import_to_rag=True)

    Args:
        base_dir      : Dossier racine du projet.
        notify        : Activer les alertes (email + local).
        only_critical : Ne notifier que les mises à jour avec mots-clés critiques.
        import_to_rag : Tenter d'importer les documents trouvés dans le RAG.
        extra_sources : Sources personnalisées supplémentaires.

    Returns:
        Résumé du cycle avec compteurs et chemins des rapports.
    """
    from src.monitor import LegalMonitor
    from src.notifier import LegalNotifier

    company_name = os.getenv("COMPANY_NAME", "Votre Entreprise")
    registry_path = base_dir / "data" / "legal_updates_registry.json"
    alerts_dir = base_dir / "data" / "alerts"
    reports_dir = base_dir / "data" / "reports"

    start_time = datetime.now(timezone.utc)
    logger.info(f"[Scheduler] === Debut du cycle de veille --- {start_time.strftime('%Y-%m-%d %H:%M UTC')} ===")

    # -----------------------------------------------------------------------
    # 1. SURVEILLANCE : Scrape des sources
    # -----------------------------------------------------------------------
    monitor = LegalMonitor(registry_path=registry_path, extra_sources=extra_sources)
    updates = monitor.check_all(only_new=True, only_critical=only_critical)

    result = {
        "timestamp": start_time.isoformat(),
        "total_updates": len(updates),
        "critical_updates": sum(1 for u in updates if u.get("is_critical")),
        "sources_checked": len(monitor.sources),
        "notifications": {},
        "rag_imports": 0,
    }

    if not updates:
        logger.info("[Scheduler] [OK] Aucune nouvelle mise a jour juridique detectee.")
        logger.info(f"[Scheduler] === Fin du cycle --- duree: {_elapsed(start_time)} ===")
        return result

    logger.info(
        f"[Scheduler] [RAPPORT] {len(updates)} mise(s) a jour trouvee(s) "
        f"({result['critical_updates']} critique(s))"
    )

    # Log rapide dans la console
    for upd in updates[:5]:
        flag = "[CRITIQUE]" if upd.get("is_critical") else "[INFO]"
        logger.info(f"  {flag} [{upd.get('country','?')}] {upd.get('title','')[:90]}")
    if len(updates) > 5:
        logger.info(f"  ... et {len(updates) - 5} autre(s).")

    # -----------------------------------------------------------------------
    # 2. NOTIFICATION : Alertes multi-canaux
    # -----------------------------------------------------------------------
    if notify:
        notifier = LegalNotifier(
            alerts_dir=alerts_dir,
            reports_dir=reports_dir,
            company_name=company_name,
        )
        result["notifications"] = notifier.notify(updates)

    # -----------------------------------------------------------------------
    # 3. IMPORT RAG (optionnel) : Intégration des nouveaux textes
    # -----------------------------------------------------------------------
    if import_to_rag:
        result["rag_imports"] = _try_rag_import(updates, base_dir)

    elapsed = _elapsed(start_time)
    logger.info(f"[Scheduler] === Cycle termine en {elapsed} ===")
    logger.info(f"[Scheduler]     Mises a jour : {result['total_updates']} | Critiques : {result['critical_updates']}")
    if result["notifications"].get("html"):
        logger.info(f"[Scheduler]     Rapport HTML : {result['notifications']['html']}")
    if result["notifications"].get("local"):
        logger.info(f"[Scheduler]     Rapport MD   : {result['notifications']['local']}")

    return result


def _try_rag_import(updates: list[dict], base_dir: Path) -> int:
    """
    Tente d'importer les URLs des mises à jour dans le pipeline RAG.
    Crée des entrées JSON dans data/input/ pour le parser.

    Returns: nombre de documents mis en file d'attente.
    """
    from src.parser import LegalDocumentParser
    import json

    input_dir = base_dir / "data" / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    imported = 0

    for upd in updates:
        if not upd.get("is_critical") or not upd.get("title"):
            continue
        try:
            # Créer un mini-document JSON pour ingestion par le parser
            doc_id = upd.get("id", "unknown")
            title = upd.get("title", "Sans titre")
            country = upd.get("country", "Inconnu")
            theme = ", ".join(upd.get("topics", [])[:3]) or "Veille juridique"

            # Construire le document dans le format attendu par le parser
            doc = {
                "clauses": [
                    {
                        "clause_id": title[:100],
                        "parent_id": None,
                        "level": 1,
                        "title_or_summary": title,
                        "full_text": (
                            f"{title}\n\n"
                            f"Source : {upd.get('source', 'N/A')}\n"
                            f"Date : {upd.get('date', 'N/A')}\n"
                            f"Pays : {country}\n"
                            f"URL : {upd.get('url', 'N/A')}\n\n"
                            f"{upd.get('excerpt', '')}"
                        ),
                        "page_range": [1],
                        "cross_references": [],
                        "document_origin": f"veille-internet - {country}",
                        "is_footnote": False,
                    }
                ]
            }

            # Nom de fichier sécurisé : <theme>_<country>.json
            safe_theme = _safe_filename(theme)
            safe_country = _safe_filename(country)
            filename = f"Veille_{safe_theme}_{safe_country}_{doc_id}.json"
            out_path = input_dir / filename
            out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info(f"[Scheduler] [RAG] Document mis en file RAG : {filename}")
            imported += 1

        except Exception as e:
            logger.warning(f"[Scheduler] Impossible de mettre en file RAG pour '{upd.get('title', '')}': {e}")

    return imported


def _safe_filename(text: str) -> str:
    """Convertit un texte en nom de fichier sécurisé."""
    import re
    text = text.replace(" ", "_").replace("/", "-").replace("\\", "-")
    text = re.sub(r"[^\w\-_]", "", text, flags=re.UNICODE)
    return text[:40]


def _elapsed(start: datetime) -> str:
    """Retourne la durée écoulée depuis start sous forme lisible."""
    delta = datetime.now(timezone.utc) - start
    secs = int(delta.total_seconds())
    if secs < 60:
        return f"{secs}s"
    return f"{secs // 60}m {secs % 60}s"


# ---------------------------------------------------------------------------
# Mode de surveillance continue
# ---------------------------------------------------------------------------

def run_continuous_monitoring(
    base_dir: Path,
    notify: bool = True,
    only_critical: bool = False,
    import_to_rag: bool = False,
    extra_sources: Optional[list] = None,
):
    """
    Boucle de surveillance continue. S'exécute toutes les N heures.
    Interrompre avec Ctrl+C.
    """
    interval = _get_interval_seconds()
    hours = interval / 3600

    logger.info(f"[Scheduler] Surveillance continue demarree (toutes les {hours:.1f} heure(s)).")
    logger.info("[Scheduler] Appuyez sur Ctrl+C pour arreter.")

    try:
        while True:
            run_monitoring_cycle(
                base_dir=base_dir,
                notify=notify,
                only_critical=only_critical,
                import_to_rag=import_to_rag,
                extra_sources=extra_sources,
            )
            next_run = datetime.now(timezone.utc)
            logger.info(
                f"[Scheduler] [PAUSE] Prochain cycle dans {hours:.1f} heure(s) "
                f"- a {_format_next(interval)}"
            )
            time.sleep(interval)

    except KeyboardInterrupt:
        logger.info("[Scheduler] [STOP] Surveillance arretee par l'utilisateur.")


def _format_next(interval_secs: int) -> str:
    """Retourne l'heure estimée du prochain cycle."""
    from datetime import timedelta
    next_dt = datetime.now() + timedelta(seconds=interval_secs)
    return next_dt.strftime("%H:%M")
