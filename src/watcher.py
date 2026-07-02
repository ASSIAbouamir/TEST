import time
import logging
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from .config import INPUT_DIR, LLAMA_CLOUD_API_KEY
from .parser import LegalDocumentParser

logger = logging.getLogger(__name__)

class NewFileHandler(FileSystemEventHandler):
    """Gestionnaire d'événements pour le dossier surveillé"""
    
    def __init__(self, parser: LegalDocumentParser):
        self.parser = parser
        
    def on_created(self, event):
        if event.is_directory:
            return
            
        file_path = Path(event.src_path)
        
        # Ignorer les fichiers système temporaires
        if file_path.name.startswith('.') or file_path.name.startswith('~') or file_path.name.startswith('__'):
            return
            
        # N'accepter que les formats standards de documents et le JSON pré-structuré
        if file_path.suffix.lower() not in ['.pdf', '.docx', '.txt', '.json']:
            logger.warning(f"Format de fichier non supporté ignoré : {file_path.name}")
            return
            
        logger.info(f"Nouveau document détecté à traiter : {file_path.name}")
        
        # Pause de sécurité pour s'assurer que l'écriture du fichier est finalisée sur le disque
        time.sleep(2)
        
        try:
            md_path, json_path, *rest = self.parser.parse_file(file_path)
            logger.info(f"✨ Succès : Le document {file_path.name} a été injecté dans la base RAG !")
        except Exception as e:
            logger.error(f"❌ Échec du traitement automatique pour {file_path.name} : {e}")

def start_watching():
    """Démarre le service d'écoute en continu sur le dossier input/"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if not LLAMA_CLOUD_API_KEY:
        logger.warning(
            "La clé d'API LLAMA_CLOUD_API_KEY n'est pas configurée dans votre fichier .env. "
            "Le Watcher fonctionnera en mode local pour les fichiers pré-structurés au format .json, "
            "mais provoquera une erreur si vous tentez d'ingérer un PDF ou Word."
        )
        
    logger.info(f"Initialisation de la surveillance sur le dossier : {INPUT_DIR.resolve()}")
    
    try:
        parser = LegalDocumentParser()
    except Exception as e:
        logger.error(f"Erreur d'initialisation du parseur LlamaParse : {e}")
        return
        
    event_handler = NewFileHandler(parser)
    observer = Observer()
    observer.schedule(event_handler, str(INPUT_DIR), recursive=False)
    observer.start()
    
    logger.info("🚀 Watcher actif ! Déposez un PDF, Word ou JSON dans data/input/ pour le tester...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Demande d'arrêt reçue, arrêt du Watcher...")
        observer.stop()
        
    observer.join()
    logger.info("Service Watcher arrêté proprement.")

if __name__ == "__main__":
    start_watching()
