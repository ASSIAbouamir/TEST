import os
import json
from legal_rag.main_data_integration import LegalRAGDataIntegration

prompts = [
    "Est-ce qu'il existe un article portant sur l'interdiction du rejet d'hydrocarbures en Mauritanie ?",
    "Est-ce qu'il existe un article portant sur l'interdiction du rejet d'hydrocarbures qui précise que cette mesure n'est applicable qu'à certaines zones, aires ou régions en Mauritanie ?",
    "Est-ce qu'il existe un article portant sur l'interdiction du rejet d'hydrocarbures qui précise que cette mesure n'est pas applicable en permanence (temporalité) en Mauritanie ?",
    "Est-ce que le texte juridique précise les types d'activités concernées par l'interdiction du rejet d'hydrocarbures en Mauritanie ?",
    "Est-ce que le texte juridique mentionne des exceptions à l'interdiction du rejet d'hydrocarbures en Mauritanie ? Vérifie si ce texte concernant l'interdiction du rejet d'hydrocarbures en Mauritanie inclut des exceptions relatives à d'autres domaines que de la santé, de l'ordre public et de la recherche. (Ne pas inclure les délais de mise en place de la loi. Les dérogations ou autorisations spécifiques constituent des exceptions)",
    "Est-ce que le texte juridique mentionne des exceptions à l'interdiction du rejet d'hydrocarbures en Mauritanie ? (Ne pas inclure les délais de mise en place de la loi. Les dérogations ou autorisations spécifiques constituent des exceptions)",
    "Est-ce que le rejet d'hydrocarbures en Mauritanie entraîne une sanction financière (amende) ?",
    "Est-ce que le rejet d'hydrocarbures en Mauritanie entraîne une peine de prison ?",
    "Est-ce qu'en Mauritanie, des procédures de contrôle sont spécifiquement décrites pour garantir ou assurer le respect de l'interdiction du rejet d'hydrocarbures ? Par exemple, l'existence d'une administration, un comité, un service ou un agent pour superviser, évaluer ou constater l'application de cette interdiction.",
    "Est-ce qu'en Mauritanie, des procédures de contrôle sont spécifiquement décrites pour garantir ou assurer le respect de l'interdiction du rejet d'hydrocarbures ? Par exemple, l'existence d'une administration, un comité, un service ou un agent pour superviser, évaluer ou constater l'application de cette interdiction. Si ces procédures existent, vérifiez si elles précisent des périodes (temporalité) spécifiques pour le contrôle du respect de l'interdiction du rejet d'hydrocarbures.",
    "Est-ce qu'en Mauritanie, des procédures de contrôle sont spécifiquement décrites pour garantir ou assurer le respect de l'interdiction du rejet d'hydrocarbures ? Par exemple, l'existence d'une administration, un comité, un service ou un agent pour superviser, évaluer ou constater l'application de cette interdiction. Si ces procédures existent, vérifiez si elles précisent des lieux ou zones spécifiques pour le contrôle du respect de l'interdiction du rejet d'hydrocarbures."
]

def main():
    rag_system = LegalRAGDataIntegration()
    
    print("Mise en place du RAG pour la Mauritanie...")
    try:
        # Load Mauritania document
        doc = rag_system.data_loader.load_document("Rejet hydrocarbure", "Mauritanie")
        rag_system.setup_system_for_document(doc)
    except Exception as e:
        print(f"Erreur lors du chargement du document: {e}")
        return
    
    results = []
    
    for i, prompt in enumerate(prompts):
        print(f"Question {i+1}/11 : {prompt}")
        response = rag_system.query(prompt)
        
        results.append({
            "prompt": prompt,
            "response": response.get('answer', response.get('response', '')),
            "sources": response.get('sources', [])
        })
        
    with open("rag_results/audit_hydrocarbure_mauritanie.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    print("Audit terminé ! Résultats sauvegardés dans rag_results/audit_hydrocarbure_mauritanie.json")

if __name__ == "__main__":
    import os
    if not os.path.exists("scratch"):
        os.makedirs("scratch")
    main()
