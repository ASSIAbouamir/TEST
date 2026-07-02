import os
import json
from typing import List, Dict, Optional, Any, Tuple
from groq import Groq
import scripts.retrieval_fusion as rf

# Configuration de l'intelligence
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

class AgentState(Dict):
    query: str
    previous_nodes: List[Dict]
    last_fetched_context_nodes: List[Dict]
    node_links_to_fetch: List[str]
    node_footers_to_fetch: Dict[str, str]
    search_failures: List[str]
    pass_count: int
    definitions: Dict[str, str]

# 1. DEFINITION AGENT
def definition_agent(state: AgentState) -> AgentState:
    """Cherche les définitions des termes de la requête dans les index."""
    print("[Agent Définition] Recherche de concepts clés...")
    # On cherche les articles qui ont des arêtes de type DEFINITION
    nodes, _ = rf.retrieve(state["query"], top_k_final=5)
    
    definitions = {}
    for node in nodes:
        # Si l'article contient des mots comme "défini", "entend par"
        if any(kw in node["text"].lower() for kw in ["entend par", "défini", "désigne"]):
            # Extraction simplifiée (on pourrait utiliser le LLM ici)
            definitions[node["metadata"].get("clause_id", "Def")] = node["text"][:200] + "..."
            
    state["definitions"] = definitions
    return state

# 2. ROUTER AGENT
def router_agent(state: AgentState) -> str:
    """Décide si on a assez d'infos ou s'il faut chercher plus loin (Footers/Links)."""
    print(f"[Agent Router] Analyse de la complétude (Passage {state['pass_count']})...")
    
    if state["pass_count"] >= 3: # Limite pour éviter les boucles
        return "END"

    context = "\n".join([f"ID: {n['node_id']}\nText: {n['text'][:300]}" for n in state["last_fetched_context_nodes"]])
    
    prompt = f"""Tu es un Router Juridique. Analyse le contexte suivant pour répondre à : '{state['query']}'
    
    CONTEXTE :
    {context}
    
    RÈGLES :
    - Si le contexte contient la réponse complète, réponds 'END'.
    - Si un article cite un autre article (ex: 'voir article 12') qui n'est pas dans le contexte, réponds 'REF:ID_ARTICLE'.
    - Sinon, réponds 'CONTINUE'.
    """
    
    try:
        response = client.chat.completions.create(
            model=os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        ).choices[0].message.content.strip()
        
        if "REF:" in response:
            ref_id = response.split("REF:")[1].strip()
            state["node_links_to_fetch"].append(ref_id)
            return "RECURSIVE"
        return "END" if "END" in response else "CONTINUE"
    except:
        return "END"

# 3. SUPERVISOR AGENT
def supervisor_agent(state: AgentState) -> str:
    """Surveille les échecs et la consommation de contexte."""
    if len(state["search_failures"]) > 2:
        print("[Agent Supervisor] Trop d'échecs de recherche. Arrêt.")
        return "END"
    return "CONTINUE"

# 4. RECURSIVE RETRIEVAL AGENT
def recursive_retrieval(state: AgentState) -> AgentState:
    """Récupère les informations ciblées par le Router."""
    print(f"[Agent Récursif] Récupération des références : {state['node_links_to_fetch']}")
    
    new_nodes = []
    for node_id in state["node_links_to_fetch"]:
        # Recherche directe par ID dans l'index chargé
        # On simule via une recherche textuelle ciblée
        nodes, _ = rf.retrieve(f"id article {node_id}", top_k_final=1)
        if nodes:
            new_nodes.extend(nodes)
        else:
            state["search_failures"].append(f"Impossible de trouver l'ID: {node_id}")

    state["previous_nodes"].extend(state["last_fetched_context_nodes"])
    state["last_fetched_context_nodes"] = new_nodes
    state["node_links_to_fetch"] = []
    state["pass_count"] += 1
    return state

# 5. ANSWERING AGENT
def answering_agent(state: AgentState) -> str:
    """Finalise la réponse avec tout le contexte accumulé."""
    print("[Agent Rédacteur] Finalisation de la réponse juridique...")
    
    all_nodes = state["previous_nodes"] + state["last_fetched_context_nodes"]
    context_text = "\n\n".join([f"[{n['node_id']}]: {n['text']}" for n in all_nodes])
    def_text = "\n".join([f"{k}: {v}" for k, v in state["definitions"].items()])

    prompt = f"""Tu es l'Agent de Réponse Final.
    
    DÉFINITIONS :
    {def_text}
    
    CONTEXTE JURIDIQUE :
    {context_text}
    
    QUESTION :
    {state['query']}
    
    CONSIGNE : Donne une réponse précise, cite les articles et les définitions.
    """
    
    response = client.chat.completions.create(
        model=os.environ.get("GROQ_LARGE_MODEL", "llama-3.1-70b-versatile"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    return response.choices[0].message.content

# --- ORCHESTRATEUR ---
def run_legal_multi_agent(query: str, country_index_path: str):
    # Initialisation de l'index pour le pays
    rf.INDEX_DIR = country_index_path
    
    # Premier Retrieval
    initial_nodes, _ = rf.retrieve(query, top_k_final=10)
    
    state = AgentState(
        query=query,
        previous_nodes=[],
        last_fetched_context_nodes=initial_nodes,
        node_links_to_fetch=[],
        node_footers_to_fetch={},
        search_failures=[],
        pass_count=0,
        definitions={}
    )
    
    # 1. Définitions
    state = definition_agent(state)
    
    # 2. Boucle de Récursion
    while state["pass_count"] < 3:
        decision = router_agent(state)
        if decision == "END":
            break
        
        if supervisor_agent(state) == "END":
            break
            
        if decision == "RECURSIVE":
            state = recursive_retrieval(state)
        else:
            break
            
    # 3. Réponse finale
    return answering_agent(state)

if __name__ == "__main__":
    # Test rapide sur le Maroc
    print(run_legal_multi_agent("Quelles sont les sanctions pour rejet d'hydrocarbures au Maroc ?", "indexes_all/indexes_maroc_rejet_hydrocarbure"))
