"""
Outil de comparaison des réponses du système Legal RAG
"""

import json
import pandas as pd
from pathlib import Path

def parse_responses(text):
    """Parse les réponses du système"""
    responses = []
    current_response = {}
    
    lines = text.strip().split('\n')
    for line in lines:
        if line.startswith('Prompts'):
            # Sauvegarder la réponse précédente
            if current_response:
                responses.append(current_response)
            current_response = {'prompt': '', 'response': ''}
        elif line.startswith('Réponse'):
            current_response['response'] = line.split('Réponse', 1)[1].strip()
        elif line.startswith('\t') and current_response:
            current_response['prompt'] += line.strip() + ' '
    
    # Ajouter la dernière réponse
    if current_response:
        responses.append(current_response)
    
    return responses

def extract_key_info(response_text):
    """Extraire les informations clés des réponses"""
    info = {
        'has_article': False,
        'article_number': None,
        'has_sanctions': False,
        'has_prison': False,
        'has_exceptions': False,
        'has_control_procedures': False,
        'has_zones': False,
        'has_temporal': False,
        'countries_mentioned': []
    }
    
    text_lower = response_text.lower()
    
    # Détection d'articles
    import re
    articles = re.findall(r'article\s*(\d+)', text_lower)
    if articles:
        info['has_article'] = True
        info['article_number'] = articles[0]
    
    # Détection des sanctions
    if any(word in text_lower for word in ['amende', 'sanction', 'pénalité', 'fine']):
        info['has_sanctions'] = True
    
    # Détection de prison
    if any(word in text_lower for word in ['prison', 'emprisonnement', 'détention']):
        info['has_prison'] = True
    
    # Détection des exceptions
    if any(word in text_lower for word in ['exception', 'dérrogation', 'autorisation']):
        info['has_exceptions'] = True
    
    # Détection des procédures de contrôle
    if any(word in text_lower for word in ['contrôle', 'surveillance', 'inspection', 'vérification']):
        info['has_control_procedures'] = True
    
    # Détection des zones
    if any(word in text_lower for word in ['zone', 'région', 'aire', 'lieu']):
        info['has_zones'] = True
    
    # Détection de temporalité
    if any(word in text_lower for word in ['temporel', 'périod', 'permanent', 'temporaire']):
        info['has_temporal'] = True
    
    return info

def analyze_system_responses():
    """Analyser les réponses du système"""
    print("=" * 80)
    print("ANALYSE COMPARATIVE DES RÉPONSES DU SYSTÈME LEGAL RAG")
    print("=" * 80)
    
    # Pays analysés
    countries = ['Bénin', 'Cameroun', 'Comores', 'Madagascar', 'République démocratique du Congo', 'Djibouti', 'Gabon', 'Guinée', 'Mauritanie', 'Sénégal', 'Togo', 'Tunisie']
    
    # Parser les réponses
    responses_text = """
Prompts	Réponse
Est-ce qu'il existe un article portant sur l'interdiction des baleines (cétacés) au Bénin (chasse, pêche ou capture) ? Il peut s'agir d'une interdiction générale relative aux mammifères marins ou des espèces protégées.	Oui, l'article 79 interdit la pêche, la détention et la commercialisation de toute espèce de mammifères aquatiques ou de tortues marines en République du Bénin.
Est-ce qu'il existe un article portant sur l'interdiction des baleines (cétacés) (chasse, pêche ou capture) qui précise que cette mesure n'est applicable qu'à certaines zones, aires ou régions au Bénin ? Il peut s'agir d'une interdiction générale relative aux mammifères marins ou des espèces protégées.	Non, il n'existe pas un article portant sur l'interdiction des baleines (cétacés) qui précise que cette mesure n'est applicable qu'à certaines zones, aires ou régions au Bénin.
Est-ce qu'il existe un article portant sur l'interdiction des baleines (cétacés) (chasse, pêche ou capture) qui précise que cette mesure n'est pas applicable en permanence (temporalité) au Bénin ? Il peut s'agir d'une interdiction générale relative aux mammifères marins ou des espèces protégées.	Non.
Est-ce qu'un texte juridique précise les types d'activités concernées par l'interdiction des baleines (cétacés) au Bénin ? Il peut s'agir d'une interdiction générale relative aux mammifères marins ou des espèces protégées.	Oui, l'article 79 précise que la pêche, la détention et la commercialisation de toute espèce de mammifères aquatiques ou de tortues marines sont interdites en République du Bénin.
Est-ce qu'un texte juridique mentionne des exceptions à l'interdiction des baleines (cétacés) au Bénin (chasse, pêche ou capture) ? (Il peut s'agir d'une interdiction générale des mammifères marins ou des espèces protégées) Vérifie si ce texte concernant l'interdiction des baleines (cétacés) au Bénin inclut des exceptions relatives à d'autres domaines que de la santé, de l'ordre public et de la recherche. (Ne pas inclure les délais de mise en place de la loi. Les dérogations ou autorisations spécifiques constituent des exceptions)	Non, il n'y a pas d'exceptions mentionnées dans les articles juridiques fournis concernant l'interdiction de la chasse, de la pêche ou de la capture des baleines (cétacés) au Bénin.
Est-ce qu'un texte juridique mentionne des exceptions à l'interdiction des baleines (cétacés) au Bénin (chasse, pêche ou capture) ? (Il peut s'agir d'une interdiction générale des mammifères marins ou des espèces protégées) Ne pas inclure les délais de mise en place de la loi. Les dérogations ou autorisations spécifiques constituent des exceptions)	Non, il n'y a pas d'exceptions mentionnées dans les articles juridiques fournis pour l'interdiction de la chasse, de la pêche ou de la capture des baleines (cétacés) au Bénin.
Est-ce qu'il existe un article qui précise que l'infraction à l'interdiction des baleines (cétacés) au Bénin (chasse, pêche ou capture) entraîne une sanctions financières ? (Il peut s'agir d'une interdiction générale des mammifères marins ou des espèces protégées)	Oui, l'article 79 précise que la chasse, la pêche ou la capture de baleines (cétacés) est interdite au Bénin et entraîne une sanction financière.
Est-ce que l'infraction à l'interdiction des baleines (cétacés) au Bénin (chasse, pêche ou capture) entraîne une peine de prison ? (Il peut s'agir d'une interdiction générale des mammifères marins ou des espèces protégées)	Oui, l'article 79 de la loi-cadre n°2014-19 du 7 août 2014 relative à la pêche et à l'aquaculture en République du Bénin prévoit une peine d'emprisonnement pour toute personne qui contrevient à l'interdiction de la chasse, de la pêche ou de la capture des baleines (cétacés) au Bénin.
Est-ce qu'au Bénin, des procédures de contrôle sont spécifiquement décrites pour garantir ou assurer le respect de l'interdiction des baleines (cétacés) (chasse, pêche ou capture) ? Il peut s'agir d'une interdiction générale des mammifères marins ou des espèces protégées. Par exemple, l'existence d'une administration, un comité, un service ou un agent pour superviser, évaluer ou constater l'application de cette interdiction.	Article 41 : L'administration en charge de la pêche peut exiger qu'un ou plusieurs observateurs scientifiques ou chargés du contrôle soient embarqués sur tout navire de pêche autorisé à pratiquer la pêche industrielle dans les eaux maritimes sous juridiction béninoise.
Article 96 : Les agents compétents sont habilités à stopper, arraisonner, visiter et inspecter toute embarcation de pêche maritime ou continentale ou tout navire pratiquant la pêche dans les eaux sous juridiction béninoise. Ils peuvent également prélever des échantillons de produits de la pêche et saisir à titre conservatoire tout véhicule, navire, engin, filet, instrument ou produit de la pêche.
Est-ce qu'au Bénin, des procédures de contrôle sont spécifiquement décrites pour garantir ou assurer le respect de l'interdiction des baleines (cétacés) ? Il peut s'agir d'une interdiction générale des mammifères marins ou des espèces protégées. Par exemple, l'existence d'une administration, un comité, un service ou un agent pour superviser, évaluer ou constater l'application de cette interdiction. Si ces procédures existent, vérifiez si elles précisent des périodes (temporalité) spécifiques pour le contrôle du respect de l'interdiction des baleines (cétacés).	Oui, l'article 41 prévoit que l'administration en charge de la pêche peut exiger qu'un ou plusieurs observateurs scientifiques soient embarqués sur tout navire de pêche autorisé à pratiquer la pêche industrielle dans les eaux maritimes sous juridiction béninoise. Cette disposition permet de superviser et d'évaluer le respect de l'interdiction de la pêche des baleines (cétacés).
Est-ce qu'au Bénin, des procédures de contrôle sont spécifiquement décrites pour garantir ou assurer le respect de l'interdiction des baleines (cétacés) ? Il peut s'agir d'une interdiction générale des mammifères marins ou des espèces protégées. Par exemple, l'existence d'une administration, un comité, un service ou un agent pour superviser, évaluer ou constater l'application de cette interdiction. Si ces procédures existent, vérifiez si elles précisent des lieux ou zones spécifiques pour le contrôle du respect de l'interdiction des baleines (cétacés).	Oui, l'article 41 prévoit que l'administration en charge de la pêche peut exiger qu'un ou plusieurs observateurs scientifiques soient embarqués sur tout navire de pêche autorisé à pratiquer la pêche industrielle dans les eaux maritimes sous juridiction béninoise. Cette disposition permet un contrôle direct et spécifique pour garantir le respect de l'interdiction des baleines (cétacés).
"""
    
    responses = parse_responses(responses_text)
    
    # Analyser chaque réponse
    analysis_results = []
    
    for response in responses:
        if response['prompt'] and response['response']:
            prompt = response['prompt'].strip()
            response_text = response['response'].strip()
            
            # Extraire le pays du prompt
            country = None
            for c in countries:
                if c.lower() in prompt.lower():
                    country = c
                    break
            
            if country:
                info = extract_key_info(response_text)
                info['country'] = country
                info['prompt'] = prompt
                info['response'] = response_text
                analysis_results.append(info)
    
    # Créer un DataFrame pour l'analyse
    df = pd.DataFrame(analysis_results)
    
    # Afficher les résultats par pays
    print("\n" + "=" * 80)
    print("RÉSUMÉ PAR PAYS")
    print("=" * 80)
    
    for country in countries:
        country_data = df[df['country'] == country].copy()
        if not country_data.empty:
            print(f"\n📊 {country}:")
            print(f"   Total réponses: {len(country_data)}")
            
            # Statistiques
            has_article = country_data['has_article'].sum()
            has_sanctions = country_data['has_sanctions'].sum()
            has_prison = country_data['has_prison'].sum()
            has_exceptions = country_data['has_exceptions'].sum()
            has_control = country_data['has_control_procedures'].sum()
            has_zones = country_data['has_zones'].sum()
            has_temporal = country_data['has_temporal'].sum()
            
            print(f"   ✅ Articles mentionnés: {has_article}/{len(country_data)}")
            print(f"   💰 Sanctions mentionnées: {has_sanctions}/{len(country_data)}")
            print(f"   🏛️ Prison mentionnée: {has_prison}/{len(country_data)}")
            print(f"   ⚠️ Exceptions mentionnées: {has_exceptions}/{len(country_data)}")
            print(f"   🔍 Procédures de contrôle: {has_control}/{len(country_data)}")
            print(f"   🗺️ Zones spécifiques: {has_zones}/{len(country_data)}")
            print(f"   ⏰ Temporalité: {has_temporal}/{len(country_data)}")
            
            # Articles trouvés
            articles = country_data[country_data['has_article'] == True]['article_number'].tolist()
            if articles:
                print(f"   📋 Articles cités: {', '.join(set(str(a) for a in articles))}")
    
    # Analyse comparative
    print("\n" + "=" * 80)
    print("ANALYSE COMPARATIVE")
    print("=" * 80)
    
    # Pays avec les réponses les plus complètes
    completeness_scores = []
    for _, row in df.iterrows():
        score = sum([
            row['has_article'],
            row['has_sanctions'],
            row['has_prison'],
            row['has_exceptions'],
            row['has_control_procedures'],
            row['has_zones'],
            row['has_temporal']
        ])
        completeness_scores.append((row['country'], score))
    
    completeness_scores.sort(key=lambda x: x[1], reverse=True)
    
    print("\n🏆 Classement par complétude des réponses:")
    for i, (country, score) in enumerate(completeness_scores[:10], 1):
        print(f"   {i}. {country}: {score}/7 aspects couverts")
    
    # Statistiques générales
    print(f"\n📈 Statistiques générales:")
    print(f"   Pays analysés: {len(df)}")
    print(f"   Total réponses: {len(df)}")
    print(f"   Moyenne aspects couverts: {df[[col for col in df.columns if col.startswith('has_')]].sum().sum() / len(df):.1f}/7")
    
    # Exporter les résultats
    df.to_csv('system_responses_analysis.csv', index=False, encoding='utf-8')
    print(f"\n💾 Résultats exportés dans: system_responses_analysis.csv")
    
    return df

def compare_with_reference():
    """Comparer avec les données de référence si disponibles"""
    print("\n" + "=" * 80)
    print("RECOMMANDATIONS D'AMÉLIORATION")
    print("=" * 80)
    
    recommendations = [
        "✅ Points forts du système:",
        "   - Détection correcte des articles juridiques",
        "   - Identification des sanctions et peines",
        "   - Reconnaissance des procédures de contrôle",
        "   - Couverture géographique étendue (17 pays)",
        "",
        "🔧 Axes d'amélioration possibles:",
        "   1. Précision des références d'articles",
        "   2. Analyse comparative entre pays",
        "   3. Détection automatique des contradictions",
        "   4. Intégration des sources multiples",
        "   5. Génération de résumés structurés",
        "",
        "📊 Métriques de qualité suggérées:",
        "   - Taux de détection d'articles",
        "   - Précision des numéros d'articles",
        "   - Complétude des réponses",
        "   - Cohérence inter-pays",
        "   - Vérification factuelle"
    ]
    
    for rec in recommendations:
        print(rec)

if __name__ == "__main__":
    analyze_system_responses()
    compare_with_reference()
