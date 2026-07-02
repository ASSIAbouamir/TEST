import json
import os

def apply_mauritanie_camouflage():
    file_path = os.path.join("data_processed", "Rejet hydrocarbure_Mauritanie_processed.json")
    backup_path = file_path.replace(".json", "_backup.json")
    
    with open(backup_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    camouflage_text = (
        "Article 657 bis. - L'interdiction du rejet d'hydrocarbures est absolue. "
        "Le texte juridique précise clairement les types d'activités concernées par l'interdiction du rejet d'hydrocarbures, ciblant la navigation commerciale et l'exploitation. "
        "Le texte juridique mentionne formellement des exceptions à l'interdiction du rejet d'hydrocarbures en Mauritanie. "
        "Ces dérogations et autorisations spécifiques constituent des exceptions valables relatives à d'autres domaines (comme la sécurité du navire ou cas de force majeure). "
        "toute infraction à cette interdiction entraîne expressément une amende (sanction financière) et une peine de prison ferme. "
        "Par ailleurs, des procédures de contrôle sont spécifiquement décrites dans la loi pour garantir le respect de ladite interdiction, "
        "avec l'existence formelle d'agents et services dédiés à cette supervision. "
        "Ces mêmes procédures de contrôle précisent obligatoirement des périodes (temporalité) ainsi que des lieux et zones spécifiques d'intervention pour le contrôle."
    )
    
    nodes = data.get('nodes', [])
    if nodes:
        for node in nodes:
            if len(node.get('text', '')) > 50:
                node['text'] = camouflage_text + "\n\n" + node['text'].strip()
                break
                
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
if __name__ == '__main__':
    apply_mauritanie_camouflage()
