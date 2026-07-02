import json
import logging
from typing import List, Dict, Any, Optional

# Optional dependencies (not available in the default sandbox).
try:
    import networkx as nx  # type: ignore
except ModuleNotFoundError:
    nx = None

try:
    from whyhow import WhyHow  # type: ignore
except ModuleNotFoundError:
    WhyHow = None

try:
    from legal_rag.models import DocumentNode, DefinitionNode, GraphTriple, DocumentStructure
    from legal_rag.config import settings
except ImportError:
    try:
        from .models import DocumentNode, DefinitionNode, GraphTriple, DocumentStructure
        from .config import settings
    except ImportError:
        import sys
        import os
        sys.path.append(os.path.dirname(__file__))
        from .models import DocumentNode, DefinitionNode, GraphTriple, DocumentStructure
        from .config import settings

logger = logging.getLogger(__name__)

class GraphBuilder:
    """Builds and manages knowledge graphs for legal documents"""
    
    def __init__(self):
        self.whyhow_client = None
        if WhyHow and settings.WHYHOW_API_KEY:
            self.whyhow_client = WhyHow(
                api_key=settings.WHYHOW_API_KEY,
                base_url=settings.WHYHOW_API_URL
            )
    
    def create_lexical_graph(self, document_structure: DocumentStructure) -> str:
        """Create lexical graph from document hierarchy"""
        logger.info("Creating lexical graph...")
        
        # Create nodes for document hierarchy
        nodes = []
        for node in document_structure.hierarchy:
            nodes.append({
                'node_id': node.node_id,
                'content': node.content,
                'node_type': node.node_type,
                'page_number': node.page_number,
                'metadata': node.metadata
            })
        
        # Create triples for relationships
        triples = self._create_lexical_triples(document_structure.hierarchy)
        
        # Upload to WhyHow if available
        if self.whyhow_client:
            graph_id = self._upload_to_whyhow(
                nodes=nodes,
                triples=triples,
                graph_name=f"lexical_{document_structure.document_id}",
                description="Document hierarchy and cross-references"
            )
            settings.LEXICAL_GRAPH_ID = graph_id
            return graph_id
        else:
            # Create local NetworkX graph
            return self._create_local_graph(nodes, triples, "lexical")
    
    def create_definitions_graph(self, document_structure: DocumentStructure) -> str:
        """Create definitions graph from legal terms"""
        logger.info("Creating definitions graph...")
        
        # Create nodes for definitions
        nodes = []
        for definition in document_structure.definitions:
            nodes.append({
                'node_id': f"def_{definition.term.lower().replace(' ', '_')}",
                'term': definition.term,
                'definition': definition.definition,
                'source_page': definition.source_page,
                'context': definition.context
            })
        
        # Create triples for definition relationships
        triples = self._create_definition_triples(document_structure.definitions)
        
        # Upload to WhyHow if available
        if self.whyhow_client:
            graph_id = self._upload_to_whyhow(
                nodes=nodes,
                triples=triples,
                graph_name=f"definitions_{document_structure.document_id}",
                description="Legal terms and their definitions"
            )
            settings.DEFINITIONS_GRAPH_ID = graph_id
            return graph_id
        else:
            # Create local NetworkX graph
            return self._create_local_graph(nodes, triples, "definitions")
    
    def _create_lexical_triples(self, hierarchy: List[DocumentNode]) -> List[Dict[str, Any]]:
        """Create triples for lexical graph relationships"""
        triples = []
        
        for node in hierarchy:
            # Parent-child relationships
            if node.parent_id:
                triples.append({
                    'subject': node.parent_id,
                    'predicate': 'contains',
                    'object': node.node_id,
                    'confidence': 1.0
                })
            
            # Link relationships
            for linked_id in node.links_to:
                triples.append({
                    'subject': node.node_id,
                    'predicate': 'references',
                    'object': linked_id,
                    'confidence': 0.9
                })
            
            # Footnote relationships
            for footnote in node.footnotes:
                triples.append({
                    'subject': node.node_id,
                    'predicate': 'has_footnote',
                    'object': f"footnote_{footnote}",
                    'confidence': 0.8
                })
        
        return triples
    
    def _create_definition_triples(self, definitions: List[DefinitionNode]) -> List[Dict[str, Any]]:
        """Create triples for definition relationships"""
        triples = []
        
        for definition in definitions:
            node_id = f"def_{definition.term.lower().replace(' ', '_')}"
            
            # Term to definition relationship
            triples.append({
                'subject': node_id,
                'predicate': 'means',
                'object': definition.definition,
                'confidence': 1.0
            })
            
            # Source page relationship
            triples.append({
                'subject': node_id,
                'predicate': 'defined_on_page',
                'object': str(definition.source_page),
                'confidence': 1.0
            })
        
        return triples
    
    def _upload_to_whyhow(self, nodes: List[Dict], triples: List[Dict], 
                         graph_name: str, description: str) -> str:
        """Upload graph to WhyHow Knowledge Graph Studio"""
        try:
            # Create graph
            graph = self.whyhow_client.graphs.create(
                name=graph_name,
                description=description
            )
            
            # Add nodes
            for node in nodes:
                self.whyhow_client.graphs.add_node(
                    graph_id=graph.graph_id,
                    node_id=node['node_id'],
                    node_type=node.get('node_type', 'default'),
                    properties=node
                )
            
            # Add triples
            for triple in triples:
                self.whyhow_client.graphs.add_triple(
                    graph_id=graph.graph_id,
                    subject=triple['subject'],
                    predicate=triple['predicate'],
                    object=triple['object'],
                    confidence=triple.get('confidence', 1.0)
                )
            
            logger.info(f"Successfully uploaded graph {graph_name} to WhyHow")
            return graph.graph_id
            
        except Exception as e:
            logger.error(f"Failed to upload to WhyHow: {e}")
            raise
    
    def _create_local_graph(self, nodes: List[Dict], triples: List[Dict], 
                           graph_type: str) -> str:
        """
        Create or update a local graph representation.
        Merges new nodes and triples with existing ones if the graph file already exists.
        """
        filename = f"{graph_type}_graph.json"
        
        existing_nodes = []
        existing_edges = []
        
        # Load existing graph if it exists
        import os
        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    old_data = json.load(f)
                    existing_nodes = old_data.get("nodes", [])
                    existing_edges = old_data.get("edges", [])
            except Exception as e:
                logger.warning(f"Could not load existing graph {filename} for merging: {e}")

        # Gather IDs of nodes being added/updated
        new_node_ids = {node.get("node_id", node.get("id", "")) for node in nodes}
        new_node_ids.discard("")
        
        # Filter out existing nodes/edges that are being overwritten
        filtered_nodes = [n for n in existing_nodes if n.get("id", n.get("node_id", "")) not in new_node_ids]
        filtered_edges = [
            e for e in existing_edges 
            if e.get("source") not in new_node_ids and e.get("target") not in new_node_ids
        ]

        if nx is not None:
            G = nx.DiGraph()
            
            # Re-add filtered existing nodes and edges
            for n in filtered_nodes:
                node_id = n.get("id", n.get("node_id", ""))
                if node_id:
                    G.add_node(node_id, **{k: v for k, v in n.items() if k not in ('id', 'node_id')})
            for e in filtered_edges:
                G.add_edge(
                    e["source"],
                    e["target"],
                    predicate=e.get("predicate", "references"),
                    confidence=e.get("confidence", 1.0)
                )
                
            # Add new nodes and edges
            for node in nodes:
                node_id = node.get("node_id", node.get("id", ""))
                if node_id:
                    G.add_node(node_id, **{k: v for k, v in node.items() if k not in ('id', 'node_id')})
            for triple in triples:
                G.add_edge(
                    triple["subject"],
                    triple["object"],
                    predicate=triple["predicate"],
                    confidence=triple.get("confidence", 1.0)
                )
                
            # Clean up isolated literal nodes (like loose strings or page numbers with degree 0)
            target_new_ids = {node.get("node_id", node.get("id", "")) for node in nodes}
            target_filtered_ids = {n.get("id", n.get("node_id", "")) for n in filtered_nodes}
            all_valid_ids = target_new_ids.union(target_filtered_ids)
            
            isolated = [n for n in G.nodes() if G.degree(n) == 0 and n not in all_valid_ids]
            for n in isolated:
                G.remove_node(n)
                
            graph_data = {
                "nodes": [{"id": n[0], **n[1]} for n in G.nodes(data=True)],
                "edges": [{"source": e[0], "target": e[1], **e[2]} for e in G.edges(data=True)],
            }
            node_count = len(G.nodes)
            edge_count = len(G.edges)
        else:
            # Fallback when networkx is not available
            formatted_new_nodes = []
            for node in nodes:
                node_id = node.get("node_id", node.get("id", ""))
                formatted_new_nodes.append({"id": node_id, **{k: v for k, v in node.items() if k not in ('id', 'node_id')}})
                
            formatted_new_edges = []
            for triple in triples:
                formatted_new_edges.append({
                    "source": triple["subject"],
                    "target": triple["object"],
                    "predicate": triple["predicate"],
                    "confidence": triple.get("confidence", 1.0)
                })
                
            merged_nodes = filtered_nodes + formatted_new_nodes
            merged_edges = filtered_edges + formatted_new_edges
            
            # Deduplicate by node ID
            seen_ids = set()
            unique_nodes = []
            for n in merged_nodes:
                nid = n.get("id")
                if nid and nid not in seen_ids:
                    unique_nodes.append(n)
                    seen_ids.add(nid)
                    
            graph_data = {
                "nodes": unique_nodes,
                "edges": merged_edges
            }
            node_count = len(unique_nodes)
            edge_count = len(merged_edges)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Merged local {graph_type} graph. Total: {node_count} nodes, {edge_count} edges")
        return filename
    
    def query_lexical_graph(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Query the lexical graph"""
        if self.whyhow_client and settings.LEXICAL_GRAPH_ID:
            try:
                response = self.whyhow_client.graphs.query_unstructured(
                    graph_id=settings.LEXICAL_GRAPH_ID,
                    query=query,
                    top_k=top_k
                )
                return self._process_whyhow_response(response)
            except Exception as e:
                logger.error(f"WhyHow query failed: {e}")
                return self._query_local_graph("lexical", query, top_k)
        else:
            return self._query_local_graph("lexical", query, top_k)
    
    def query_definitions_graph(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Query the definitions graph"""
        if self.whyhow_client and settings.DEFINITIONS_GRAPH_ID:
            try:
                response = self.whyhow_client.graphs.query_unstructured(
                    graph_id=settings.DEFINITIONS_GRAPH_ID,
                    query=query,
                    top_k=top_k
                )
                return self._process_whyhow_response(response)
            except Exception as e:
                logger.error(f"WhyHow query failed: {e}")
                return self._query_local_graph("definitions", query, top_k)
        else:
            return self._query_local_graph("definitions", query, top_k)
    
    def _process_whyhow_response(self, response) -> List[Dict[str, Any]]:
        """Process response from WhyHow"""
        results = []
        if hasattr(response, 'nodes'):
            for node in response.nodes:
                results.append({
                    'node_id': node.node_id,
                    'content': node.properties.get('content', ''),
                    'node_type': node.properties.get('node_type', ''),
                    'score': getattr(node, 'score', 1.0)
                })
        return results
    
    def _query_local_graph(self, graph_type: str, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Query local NetworkX graph (simple keyword matching)"""
        filename = f"{graph_type}_graph.json"
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)
            
            query_terms = query.lower().split()
            results = []
            
            for node in graph_data['nodes']:
                content = node.get('content', '').lower()
                term = node.get('term', '').lower()
                
                # Simple keyword matching
                score = 0
                for q_term in query_terms:
                    if q_term in content or q_term in term:
                        score += 1
                
                if score > 0:
                    results.append({
                        'node_id': node['id'],
                        'content': node.get('content', node.get('definition', '')),
                        'node_type': node.get('node_type', 'definition'),
                        'score': score / len(query_terms)
                    })
            
            # Sort by score and return top_k
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:top_k]
            
        except FileNotFoundError:
            logger.error(f"Local graph file {filename} not found")
            return []
        except Exception as e:
            logger.error(f"Error querying local graph: {e}")
            return []
