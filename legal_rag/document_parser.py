import re
import json
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import logging

from .models import DocumentNode, DefinitionNode, DocumentStructure

logger = logging.getLogger(__name__)

@dataclass
class ParsedElement:
    """Represents a parsed element from the document"""
    element_type: str
    content: str
    page_number: int
    position: int
    parent_id: Optional[str] = None
    links_to: List[str] = None
    footnotes: List[str] = None

class DocumentParser:
    """Parses legal documents and creates hierarchical structure"""
    
    def __init__(self):
        self.element_patterns = {
            'section_header': r'^(?:Section|Article|Clause|Paragraph|Titre|Article)\s+[\d\.]+',
            'list_item': r'^[a-zA-Z0-9][\.\)]\s+',
            'paragraph': r'^(?:\d+\.|[A-Za-z]\.)\s+',
            'footer': r'^(?:\d+|Footnote|Note)\s*:?\s*',
            'definition': r'^["\']?([A-Z][A-Za-z\s]+)["\']?\s*(?:means|refers to|shall mean|désigne|signifie)',
        }
        
    def parse_document(self, document_text: str, document_id: str = "doc1") -> DocumentStructure:
        """Parse document and create hierarchical structure"""
        pages = self._split_into_pages(document_text)
        elements = self._extract_elements(pages)
        hierarchy = self._build_hierarchy(elements)
        definitions = self._extract_definitions(pages)
        
        return DocumentStructure(
            document_id=document_id,
            title=self._extract_title(document_text),
            pages=pages,
            hierarchy=hierarchy,
            definitions=definitions
        )
    
    def _split_into_pages(self, document_text: str) -> List[Dict[str, Any]]:
        """Split document into pages"""
        page_break_pattern = r'\f|Page\s+\d+|Page\s+\d+/\d+'
        pages_text = re.split(page_break_pattern, document_text)
        
        pages = []
        for i, page_text in enumerate(pages_text, 1):
            if page_text.strip():
                pages.append({
                    'page_number': i,
                    'content': page_text.strip(),
                    'raw_text': page_text
                })
        
        return pages
    
    def _extract_elements(self, pages: List[Dict[str, Any]]) -> List[ParsedElement]:
        """Extract structural elements from pages"""
        elements = []
        
        for page in pages:
            lines = page['content'].split('\n')
            for line_num, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                element_type = self._identify_element_type(line)
                if element_type:
                    element = ParsedElement(
                        element_type=element_type,
                        content=line,
                        page_number=page['page_number'],
                        position=line_num,
                        links_to=self._extract_links(line),
                        footnotes=self._extract_footnotes(line)
                    )
                    elements.append(element)
        
        return elements
    
    def _identify_element_type(self, line: str) -> Optional[str]:
        """Identify the type of document element"""
        for element_type, pattern in self.element_patterns.items():
            if re.match(pattern, line, re.IGNORECASE):
                return element_type
        return 'paragraph'  # Default to paragraph
    
    def _extract_links(self, text: str) -> List[str]:
        """Extract references to other sections/clauses"""
        link_patterns = [
            r'(?:Section|Article|Clause|Paragraph|Para)\s+([\d\.]+)',
            r'refer\s+to\s+(?:Section|Article|Clause|Paragraph|Para)\s+([\d\.]+)',
            r'voir\s+(?:Section|Article|Clause|Paragraph|Para)\s+([\d\.]+)',
            r'(?:Article|Art\.|Section)\s+([\d\.]+)',
        ]
        
        links = []
        for pattern in link_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            links.extend(matches)
        
        return list(set(links))  # Remove duplicates
    
    def _extract_footnotes(self, text: str) -> List[str]:
        """Extract footnote references"""
        footnote_patterns = [
            r'\[(\d+)\]',
            r'footnote\s+(\d+)',
            r'note\s+(\d+)',
        ]
        
        footnotes = []
        for pattern in footnote_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            footnotes.extend(matches)
        
        return list(set(footnotes))
    
    def _build_hierarchy(self, elements: List[ParsedElement]) -> List[DocumentNode]:
        """Build hierarchical structure from elements"""
        nodes = []
        node_map = {}
        
        # Create nodes
        for i, element in enumerate(elements):
            node_id = f"{element.page_number}_{element.position}_{element.element_type}"
            
            node = DocumentNode(
                node_id=node_id,
                content=element.content,
                node_type=element.element_type,
                page_number=element.page_number,
                links_to=element.links_to or [],
                footnotes=element.footnotes or [],
                metadata={
                    'position': element.position,
                    'element_index': i
                }
            )
            
            nodes.append(node)
            node_map[node_id] = node
        
        # Build parent-child relationships
        for i, node in enumerate(nodes):
            # Find potential parents (previous elements of higher hierarchy)
            for j in range(i-1, max(-1, i-10), -1):
                potential_parent = nodes[j]
                if self._is_parent_child_relationship(potential_parent, node):
                    node.parent_id = potential_parent.node_id
                    potential_parent.children_ids.append(node.node_id)
                    break
        
        return nodes
    
    def _is_parent_child_relationship(self, parent: DocumentNode, child: DocumentNode) -> bool:
        """Determine if there's a parent-child relationship"""
        hierarchy_order = [
            'section_header',
            'paragraph', 
            'list_item',
            'footer'
        ]
        
        try:
            parent_order = hierarchy_order.index(parent.node_type)
            child_order = hierarchy_order.index(child.node_type)
            return parent_order < child_order
        except ValueError:
            return False
    
    def _extract_definitions(self, pages: List[Dict[str, Any]]) -> List[DefinitionNode]:
        """Extract legal definitions from document"""
        definitions = []
        
        # Look for definition sections (typically pages 4-5 as mentioned in article)
        definition_keywords = ['definition', 'définition', 'means', 'refers to', 'signifie', 'désigne']
        
        for page in pages:
            lines = page['content'].split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check if this line contains a definition
                for keyword in definition_keywords:
                    if keyword.lower() in line.lower():
                        definition = self._parse_definition_line(line, page['page_number'])
                        if definition:
                            definitions.append(definition)
        
        return definitions
    
    def _parse_definition_line(self, line: str, page_number: int) -> Optional[DefinitionNode]:
        """Parse a single definition line"""
        # Pattern: "Term" means/refers to definition
        pattern = r'^["\']?([A-Z][A-Za-z\s]+)["\']?\s*(?:means|refers to|shall mean|désigne|signifie|est défini comme)\s*(.+)$'
        
        match = re.match(pattern, line, re.IGNORECASE)
        if match:
            term = match.group(1).strip()
            definition = match.group(2).strip()
            
            return DefinitionNode(
                term=term,
                definition=definition,
                source_page=page_number,
                context=line
            )
        
        return None
    
    def _extract_title(self, document_text: str) -> str:
        """Extract document title"""
        lines = document_text.split('\n')[:5]  # Check first 5 lines
        for line in lines:
            line = line.strip()
            if line and len(line) > 10:  # Assume title is longer than 10 chars
                return line[:100]  # Limit title length
        
        return "Untitled Document"
    
    def create_lexical_triples(self, hierarchy: List[DocumentNode]) -> List[Dict[str, Any]]:
        """Create triples for lexical graph"""
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
