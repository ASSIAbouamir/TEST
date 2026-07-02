"""
src/notifier.py
===============
Système d'alertes automatiques pour les mises à jour juridiques.

Canaux disponibles :
  1. Email SMTP  — envoi d'un rapport HTML formaté
  2. Log local   — fichier JSON des alertes + rapport Markdown horodaté
  3. Rapport HTML — page autonome lisible dans un navigateur

Configuration (variables .env) :
  ALERT_EMAIL_TO       = destinataires (séparés par virgule)
  ALERT_EMAIL_FROM     = expéditeur
  SMTP_HOST            = serveur SMTP (ex: smtp.gmail.com)
  SMTP_PORT            = port SMTP (ex: 587)
  SMTP_USER            = identifiant SMTP
  SMTP_PASSWORD        = mot de passe SMTP
  ALERT_MIN_TOPICS     = nombre minimum de topics pour déclencher l'alerte (défaut: 1)
"""

import json
import logging
import os
import smtplib
import ssl
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration email (depuis .env)
# ---------------------------------------------------------------------------

def _get_email_config() -> dict:
    return {
        "to": [e.strip() for e in os.getenv("ALERT_EMAIL_TO", "").split(",") if e.strip()],
        "from_addr": os.getenv("ALERT_EMAIL_FROM", "legal-ai-bot@automatisation.local"),
        "smtp_host": os.getenv("SMTP_HOST", ""),
        "smtp_port": int(os.getenv("SMTP_PORT", "587")),
        "smtp_user": os.getenv("SMTP_USER", ""),
        "smtp_password": os.getenv("SMTP_PASSWORD", ""),
        "min_topics": int(os.getenv("ALERT_MIN_TOPICS", "1")),
    }


# ---------------------------------------------------------------------------
# Formatage HTML de l'email
# ---------------------------------------------------------------------------

def _build_html_email(updates: list[dict], company_name: str = "Votre Entreprise") -> str:
    now = datetime.now(timezone.utc).strftime("%d/%m/%Y à %H:%M UTC")
    critical = [u for u in updates if u.get("is_critical")]
    normal = [u for u in updates if not u.get("is_critical")]

    def _badge(topics: list) -> str:
        colors = {
            "environnement": "#16a34a", "taxe": "#dc2626", "travail": "#2563eb",
            "commerce": "#7c3aed", "numérique": "#0891b2", "santé": "#ea580c",
        }
        badges = []
        for t in topics[:4]:
            color = next((v for k, v in colors.items() if k in t), "#64748b")
            badges.append(
                f'<span style="background:{color};color:#fff;padding:2px 8px;'
                f'border-radius:999px;font-size:11px;margin:2px;display:inline-block">{t}</span>'
            )
        return " ".join(badges)

    def _card(u: dict, urgent: bool = False) -> str:
        border = "#dc2626" if urgent else "#3b82f6"
        bg = "#fef2f2" if urgent else "#f0f9ff"
        badge_html = _badge(u.get("topics", []))
        return f"""
        <div style="border-left:4px solid {border};background:{bg};
                    margin:12px 0;padding:16px;border-radius:0 8px 8px 0;">
          <div style="font-size:14px;font-weight:700;color:#1e293b;margin-bottom:6px">
            {"🔴 " if urgent else "🔵 "}{u.get('title','Sans titre')[:120]}
          </div>
          <div style="font-size:12px;color:#64748b;margin-bottom:8px">
            📅 {u.get('date','?')} &nbsp;|&nbsp; 🌍 {u.get('country','?')} &nbsp;|&nbsp;
            📰 {u.get('source','?')}
          </div>
          <div style="font-size:13px;color:#334155;margin-bottom:8px;font-style:italic">
            {u.get('excerpt','')[:250]}...
          </div>
          <div style="margin-bottom:8px">{badge_html}</div>
          <a href="{u.get('url','#')}" style="font-size:12px;color:{border};font-weight:600;
             text-decoration:none">🔗 Consulter la source →</a>
        </div>"""

    critical_section = ""
    if critical:
        critical_section = f"""
        <div style="background:#fef2f2;border:1px solid #fca5a5;border-radius:8px;
                    padding:16px;margin-bottom:24px">
          <h2 style="color:#b91c1c;margin:0 0 12px">
            🚨 {len(critical)} alerte(s) CRITIQUE(S) — Action recommandée
          </h2>
          {"".join(_card(u, urgent=True) for u in critical)}
        </div>"""

    normal_section = ""
    if normal:
        normal_section = f"""
        <div>
          <h2 style="color:#1e40af;margin:0 0 12px">
            ℹ️ {len(normal)} mise(s) à jour informationnelle(s)
          </h2>
          {"".join(_card(u, urgent=False) for u in normal)}
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <title>Rapport Juridique — {now}</title>
</head>
<body style="font-family:'Segoe UI',Arial,sans-serif;max-width:760px;
             margin:0 auto;padding:24px;background:#f8fafc;color:#1e293b">
  <div style="background:linear-gradient(135deg,#1e3a5f,#2563eb);color:#fff;
              border-radius:12px;padding:28px;margin-bottom:24px;text-align:center">
    <h1 style="margin:0 0 8px;font-size:24px">⚖️ Legal AI — Rapport de Veille Juridique</h1>
    <p style="margin:0;font-size:14px;opacity:0.85">
      Généré automatiquement le {now} | {company_name}
    </p>
    <div style="margin-top:16px;font-size:22px;font-weight:800">
      {len(updates)} nouvelle(s) mise(s) à jour détectée(s)
    </div>
  </div>

  {critical_section}
  {normal_section}

  <div style="text-align:center;margin-top:32px;font-size:11px;color:#94a3b8">
    Ce rapport est généré automatiquement par <strong>Legal AI Automation</strong>.<br>
    Ne pas répondre à cet email.
  </div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Rapport Markdown
# ---------------------------------------------------------------------------

def _build_markdown_report(updates: list[dict]) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# 📋 Rapport de Veille Juridique — {now}",
        f"\n**{len(updates)} nouvelle(s) mise(s) à jour détectée(s)**\n",
        "---\n",
    ]
    critical = [u for u in updates if u.get("is_critical")]
    if critical:
        lines.append(f"## 🚨 Alertes Critiques ({len(critical)})\n")
        for u in critical:
            topics_str = ", ".join(u.get("topics", []))
            lines += [
                f"### 🔴 {u.get('title', 'Sans titre')}",
                f"- **Date** : {u.get('date', '?')}",
                f"- **Source** : {u.get('source', '?')} — [{u.get('url', '#')}]({u.get('url', '#')})",
                f"- **Pays** : {u.get('country', '?')}",
                f"- **Domaines** : {topics_str or 'N/A'}",
                f"- **Extrait** : _{u.get('excerpt', '')[:300]}_",
                "",
            ]

    informational = [u for u in updates if not u.get("is_critical")]
    if informational:
        lines.append(f"\n## ℹ️ Informations ({len(informational)})\n")
        for u in informational:
            lines += [
                f"### 🔵 {u.get('title', 'Sans titre')}",
                f"- **Date** : {u.get('date', '?')} | **Pays** : {u.get('country', '?')}",
                f"- **Source** : [{u.get('source', '?')}]({u.get('url', '#')})",
                "",
            ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Canaux d'envoi
# ---------------------------------------------------------------------------

def _send_email(html_content: str, subject: str, cfg: dict) -> bool:
    """Envoie un email HTML via SMTP. Retourne True si succès."""
    if not cfg["to"] or not cfg["smtp_host"]:
        logger.warning("[Notifier] Email ignoré : SMTP non configuré (SMTP_HOST, ALERT_EMAIL_TO).")
        return False

    # Mode simulation/mock pour les environnements de test et développement
    if cfg["smtp_host"] == "smtp.example.com":
        logger.info(f"[Notifier] [SIMULATION] Email simulé avec succès à : {', '.join(cfg['to'])}")
        print(f"[Notifier] [SIMULATION] Envoi d'email simulé réussi vers {', '.join(cfg['to'])} (Objet: {subject})")
        return True

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = cfg["from_addr"]
    msg["To"] = ", ".join(cfg["to"])
    msg.attach(MIMEText(html_content, "html", "utf-8"))

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP(cfg["smtp_host"], cfg["smtp_port"]) as server:
            server.ehlo()
            server.starttls(context=context)
            if cfg["smtp_user"] and cfg["smtp_password"]:
                server.login(cfg["smtp_user"], cfg["smtp_password"])
            server.sendmail(cfg["from_addr"], cfg["to"], msg.as_string())
        logger.info(f"[Notifier] Email envoyé à : {', '.join(cfg['to'])}")
        return True
    except Exception as e:
        logger.error(f"[Notifier] Échec de l'envoi email : {e}")
        return False


def _save_local_alert(updates: list[dict], alerts_dir: Path) -> Path:
    """Sauvegarde le rapport JSON + Markdown dans le dossier d'alertes."""
    alerts_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON brut
    json_path = alerts_dir / f"alert_{timestamp}.json"
    json_path.write_text(
        json.dumps({"timestamp": timestamp, "updates": updates}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # Markdown lisible
    md_path = alerts_dir / f"rapport_{timestamp}.md"
    md_path.write_text(_build_markdown_report(updates), encoding="utf-8")

    logger.info(f"[Notifier] Rapport sauvegarde -> {md_path.name}")
    return md_path


def _save_html_report(html_content: str, reports_dir: Path) -> Path:
    """Sauvegarde le rapport HTML dans le dossier des rapports."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = reports_dir / f"rapport_juridique_{timestamp}.html"
    html_path.write_text(html_content, encoding="utf-8")
    logger.info(f"[Notifier] Rapport HTML -> {html_path}")
    return html_path


# ---------------------------------------------------------------------------
# Interface publique
# ---------------------------------------------------------------------------

class LegalNotifier:
    """
    Gestionnaire d'alertes multi-canaux pour les mises à jour juridiques.

    Usage :
        notifier = LegalNotifier(
            alerts_dir=Path("data/alerts"),
            reports_dir=Path("data/reports"),
            company_name="Mon Entreprise SARL",
        )
        notifier.notify(updates)
    """

    def __init__(
        self,
        alerts_dir: Path,
        reports_dir: Path,
        company_name: str = "Votre Entreprise",
        email_config: Optional[dict] = None,
    ):
        self.alerts_dir = alerts_dir
        self.reports_dir = reports_dir
        self.company_name = company_name
        self.email_cfg = email_config or _get_email_config()

    def notify(self, updates: list[dict]) -> dict:
        """
        Envoie les alertes via tous les canaux configurés.

        Returns:
            dict avec les résultats par canal.
        """
        if not updates:
            logger.info("[Notifier] Aucune mise à jour à notifier.")
            return {"email": False, "local": None, "html": None}

        critical_count = sum(1 for u in updates if u.get("is_critical"))
        logger.info(
            f"[Notifier] Envoi d'alertes pour {len(updates)} mise(s) à jour "
            f"({critical_count} critique(s))..."
        )

        results = {}

        # 1. Rapport HTML (toujours généré)
        html_content = _build_html_email(updates, self.company_name)
        html_path = _save_html_report(html_content, self.reports_dir)
        results["html"] = str(html_path)

        # 2. Sauvegarde locale JSON + Markdown
        md_path = _save_local_alert(updates, self.alerts_dir)
        results["local"] = str(md_path)

        # 3. Email (si configuré)
        if critical_count > 0:
            subject = f"🚨 [{self.company_name}] {critical_count} Alerte(s) Juridique(s) Critique(s) !"
        else:
            subject = f"⚖️ [{self.company_name}] {len(updates)} Mise(s) à jour juridique(s)"

        results["email"] = _send_email(html_content, subject, self.email_cfg)

        return results

    def test_email(self) -> bool:
        """Envoie un email de test pour vérifier la configuration SMTP."""
        test_html = _build_html_email(
            [{
                "id": "test-001",
                "title": "Email de test — Legal AI Automation",
                "date": datetime.now().date().isoformat(),
                "source": "Système de test",
                "url": "#",
                "excerpt": "Ceci est un email de test. Si vous recevez ce message, votre configuration SMTP est correcte.",
                "topics": ["numérique", "test"],
                "country": "Local",
                "is_new": True,
                "is_critical": False,
            }],
            self.company_name,
        )
        return _send_email(
            test_html,
            f"[Test] Legal AI — Configuration email ({self.company_name})",
            self.email_cfg,
        )
