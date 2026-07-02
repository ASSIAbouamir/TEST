import json

try:
    with open('rag_results/audit_hydrocarbure_mauritanie.json', 'r', encoding='utf-8') as f:
        results = json.load(f)

    print("==================================================")
    print("VERDICTS RAG - MAURITANIE")
    print("==================================================")
    for i, res in enumerate(results):
        response = res['response']
        
        # Extraire le verdict (Oui/Non) de la balise <reflexion> ou de la fin
        verdict = "Non trouvé"
        if "Verdict :" in response:
            verdict = response.split("Verdict :")[-1].strip().split("\n")[0].strip()
        else:
            # Fallback
            lines = response.strip().split('\n')
            for line in reversed(lines):
                if line.strip() in ['Oui', 'Non']:
                    verdict = line.strip()
                    break

        print(f"[Q{str(i+1).zfill(2)}] Verdict : {verdict}")
except Exception as e:
    print(f"Erreur: {e}")
