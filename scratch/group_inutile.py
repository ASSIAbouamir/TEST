import os
import shutil
from pathlib import Path

# Répertoires de base
base_dir = Path(r"c:\Users\hp assia\Desktop\Automatisation")
inutile_txt_path = base_dir / "inutile.txt"
dest_base_dir = base_dir / "inutile"

if not inutile_txt_path.exists():
    print(f"Erreur : le fichier {inutile_txt_path} est introuvable.")
    exit(1)

# Extraction des chemins de fichiers depuis la table Markdown dans inutile.txt
lines = inutile_txt_path.read_text(encoding="utf-8").splitlines()
files_to_move = []

for line in lines:
    line_stripped = line.strip()
    if line_stripped.startswith("|"):
        parts = [p.strip() for p in line_stripped.split("|")]
        # La ligne doit commencer et finir par un pipe, donc au moins 3 colonnes après split
        if len(parts) >= 3:
            file_path_str = parts[1]
            # Ignorer les en-têtes de table Markdown
            if file_path_str and file_path_str != "Chemin du fichier" and not file_path_str.startswith("---"):
                files_to_move.append(file_path_str)

print(f"Trouvé {len(files_to_move)} fichiers à déplacer.")

moved_count = 0
already_moved_count = 0
not_found_count = 0
error_count = 0

for rel_path in files_to_move:
    # Nettoyer les caractères superflus
    rel_path = rel_path.strip().replace('\\', '/')
    if not rel_path:
        continue
        
    src_file = base_dir / rel_path
    dest_file = dest_base_dir / rel_path

    # Éviter de s'auto-déplacer si le chemin commence déjà par inutile/
    if rel_path.startswith("inutile/"):
        print(f"Ignoré (déjà dans le dossier inutile/) : {rel_path}")
        continue

    if src_file.exists():
        try:
            # Créer le répertoire parent de destination s'il n'existe pas
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Si la destination existe déjà, on la supprime proprement pour éviter les blocages
            if dest_file.exists():
                if dest_file.is_file():
                    dest_file.unlink()
                else:
                    shutil.rmtree(dest_file)
            
            # Déplacement
            shutil.move(str(src_file), str(dest_file))
            print(f"Déplacé : {rel_path} -> inutile/{rel_path}")
            moved_count += 1
        except Exception as e:
            print(f"Erreur lors du déplacement de {rel_path} : {e}")
            error_count += 1
    else:
        # Vérifier si le fichier est déjà à destination
        if dest_file.exists():
            already_moved_count += 1
        else:
            print(f"Introuvable (ni source, ni destination) : {rel_path}")
            not_found_count += 1

print("\n=== Résumé de l'exécution ===")
print(f"Total fichiers identifiés : {len(files_to_move)}")
print(f"Déplacés avec succès       : {moved_count}")
print(f"Déjà présents à destination: {already_moved_count}")
print(f"Introuvables               : {not_found_count}")
print(f"Erreurs                    : {error_count}")
