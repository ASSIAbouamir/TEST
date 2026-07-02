"""
Outil de comparaison simple des réponses du système Legal RAG
"""

import re
import json

def parse_responses_simple(text):
    """Parse les réponses du système de manière simple"""
    responses = []
    lines = text.strip().split('\n')
    
    for line in lines:
        if line.startswith('Prompts'):
            continue
        elif line.startswith('Réponse'):
            response_text = line.split('Réponse', 1)[1].strip()
            # Extraire le pays de la question précédente
            prev_line = lines[lines.index(line) - 1] if lines.index(line) > 0 else ""
            country = None
            for c in ['Bénin', 'Cameroun', 'Comores', 'Madagascar', 'Congo', 'Djibouti', 'Gabon', 'Guinée', 'Mauritanie', 'Sénégal', 'Togo', 'Tunisie']:
                if c.lower() in prev_line.lower():
                    country = c
                    break
            
            if country and response_text:
                responses.append({
                    'country': country,
                    'response': response_text
                })
    
    return responses

def analyze_responses_simple():
    """Analyser les réponses du système"""
    print("=" * 60)
    print("ANALYSE DES RÉPONSES DU SYSTÈME LEGAL RAG")
    print("=" * 60)
    
    # Parser les réponses
    responses_text = """
Prompts	Réponse
Est-ce qu'il existe un article portant sur l'interdiction des baleines (cétacés) au Bénin (chasse, pêche ou capture) ? Il peut s'agir d'une interdiction générale relative aux mammifères marins ou des espèces protégées.	Oui, l'article 79 interdit la pêche, la détention et la commercialisation de toute espèce de mammifères aquatiques ou de tortues marines en République du Bénin.
Est-ce qu'il existe un article portant sur l'interdiction des baleines (cétacés) au Cameroun (chasse, pêche ou capture) ? Il peut s'agir d'une interdiction générale relative aux mammifères marins ou des espèces protégées.	Oui, l'article 26(1) interdit la capture des espèces aquatiques intégralement protégées, y compris les baleines.
"""
    
    responses = parse_responses_simple(responses_text)
    
    # Analyser chaque réponse
    print(f"Nombre total de réponses analysées: {len(responses)}")
    print()
    
    for response in responses:
        country = response['country']
        response_text = response['response']
        
        print(f"\n📊 {country}:")
        print("-" * 40)
        
        # Analyse simple
        has_article = "article" in response_text.lower()
        has_number = bool(re.search(r'article\s*\d+', response_text))
        has_sanctions = any(word in response_text.lower() for word in ['sanction', 'amende', 'pénalité'])
        has_prison = any(word in response_text.lower() for word in ['prison', 'emprisonnement'])
        has_exceptions = any(word in response_text.lower() for word in ['exception', 'dérrogation'])
        has_control = any(word in response_text.lower() for word in ['contrôle', 'surveillance', 'inspection'])
        
        print(f"   ✅ Article mentionné: {has_article}")
        print(f"   📋 Numéro d'article: {re.search(r'article\s*(\d+)', response_text).group(1) if has_number else 'N/A'}")
        print(f"   💰 Sanctions mentionnées: {has_sanctions}")
        print(f"   🏛️ Prison mentionnée: {has_prison}")
        print(f"   ⚠️ Exceptions mentionnées: {has_exceptions}")
        print(f"   🔍 Procédures de contrôle: {has_control}")
        
        # Extraire le contenu principal
        if len(response_text) > 100:
            preview = response_text[:150] + "..."
        else:
            preview = response_text
        print(f"   📝 Contenu: {preview}")
    
    print("\n" + "=" * 60)
    print("STATISTIQUES GLOBALES")
    print("=" * 60)
    
    # Calculer les statistiques
    total_responses = len(responses)
    countries_with_articles = sum(1 for r in responses if "article" in r['response'].lower())
    countries_with_sanctions = sum(1 for r in responses if any(word in r['response'].lower() for word in ['sanction', 'amende', 'pénalité']))
    countries_with_prison = sum(1 for r in responses if any(word in r['response'].lower() for word in ['prison', 'emprisonnement']))
    countries_with_exceptions = sum(1 for r in responses if any(word in r['response'].lower() for word in ['exception', 'dérrogation']))
    countries_with_control = sum(1 for r in responses if any(word in r['response'].lower() for word in ['contrôle', 'surveillance', 'inspection']))
    
    print(f"Total réponses analysées: {total_responses}")
    print(f"Pays avec articles: {countries_with_articles}/{total_responses}")
    print(f"Pays avec sanctions: {countries_with_sanctions}/{total_responses}")
    print(f"Pays avec prison: {countries_with_prison}/{total_responses}")
    print(f"Pays avec exceptions: {countries_with_exceptions}/{total_responses}")
    print(f"Pays avec contrôle: {countries_with_control}/{total_responses}")
    
    # Pays les mieux couverts
    coverage_scores = []
    for response in responses:
        score = 0
        if "article" in response['response'].lower():
            score += 1
        if any(word in response['response'].lower() for word in ['sanction', 'amende', 'pénalité']):
            score += 1
        if any(word in response['response'].lower() for word in ['prison', 'emprisonnement']):
            score += 1
        if any(word in response['response'].lower() for word in ['exception', 'dérrogation']):
            score += 1
        if any(word in response['response'].lower() for word in ['contrôle', 'surveillance', 'inspection']):
            score += 1
        
        coverage_scores.append((response['country'], score))
    
    coverage_scores.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n🏆 Classement par couverture:")
    for i, (country, score) in enumerate(coverage_scores[:10], 1):
        print(f"   {i}. {country}: {score}/5 aspects")
    
    print(f"\n🎯 Score moyen de couverture: {sum(score for _, score in coverage_scores) / len(coverage_scores):.1f}/5")
    
    # Exporter les résultats
    results = {
        'responses': responses,
        'statistics': {
            'total_responses': total_responses,
            'countries_with_articles': countries_with_articles,
            'countries_with_sanctions': countries_with_sanctions,
            'countries_with_prison': countries_with_prison,
            'countries_with_exceptions': countries_with_exceptions,
            'countries_with_control': countries_with_control,
            'coverage_scores': coverage_scores
        }
    }
    
    with open('system_analysis_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Résultats exportés dans: system_analysis_results.json")
    
    return results

if __name__ == "__main__":
    analyze_responses_simple()
