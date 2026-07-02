"""
Data loader for integrating with the processed legal documents in data_processed directory
"""

import json
import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from .models import DocumentNode, DefinitionNode, DocumentStructure

logger = logging.getLogger(__name__)

class ProcessedDataLoader:
    """Loader for processed legal documents from data_processed directory"""
    
    def __init__(self, data_processed_path: str = None):
        if data_processed_path is None:
            # Default to the data_processed directory
            current_dir = Path(__file__).parent
            parent_dir = current_dir.parent
            self.data_processed_path = parent_dir / "data_processed"
        else:
            self.data_processed_path = Path(data_processed_path)
        
        self.available_files = self._scan_available_files()
    
    def _scan_available_files(self) -> Dict[str, List[str]]:
        """Scan available processed files by theme and country"""
        files = {}
        
        if not self.data_processed_path.exists():
            logger.warning(f"Data processed directory not found: {self.data_processed_path}")
            return files
        
        for file_path in self.data_processed_path.glob("*_processed.json"):
            # Extract theme and country from filename
            filename = file_path.stem  # Remove .json
            parts = filename.split('_')
            
            if len(parts) >= 2:
                theme = parts[0]
                country = '_'.join(parts[1:-1])  # Handle country names with spaces
                
                if theme not in files:
                    files[theme] = []
                
                files[theme].append({
                    'country': country,
                    'filepath': str(file_path),
                    'filename': file_path.name
                })
        
        logger.info(f"Found {len(files)} themes: {list(files.keys())}")
        return files
    
    def get_available_themes(self) -> List[str]:
        """Get list of available themes"""
        return list(self.available_files.keys())
    
    def get_countries_for_theme(self, theme: str) -> List[str]:
        """Get list of countries available for a theme"""
        if theme in self.available_files:
            return [item['country'] for item in self.available_files[theme]]
        return []
    
    def load_document(self, theme: str, country: str) -> Optional[DocumentStructure]:
        """Load a specific document by theme and country"""
        if theme not in self.available_files:
            logger.error(f"Theme '{theme}' not found. Available: {list(self.available_files.keys())}")
            return None
        
        # Find the file for this country
        target_file = None
        for file_info in self.available_files[theme]:
            if file_info['country'].lower() == country.lower():
                target_file = file_info['filepath']
                break
        
        if not target_file:
            logger.error(f"Country '{country}' not found for theme '{theme}'")
            return None
        
        try:
            with open(target_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return self._convert_to_document_structure(data, theme, country)
            
        except Exception as e:
            logger.error(f"Error loading document {target_file}: {e}")
            return None
    
    def _convert_to_document_structure(self, data: Dict[str, Any], theme: str, country: str) -> DocumentStructure:
        """Convert processed JSON data to DocumentStructure"""
        nodes = []
        definitions = []
        
        # Process nodes
        for node_data in data.get('nodes', []):
            # Create DocumentNode
            node = DocumentNode(
                node_id=node_data['node_id'],
                content=node_data['text'],
                node_type=self._determine_node_type(node_data),
                page_number=node_data['metadata'].get('page_range', [1])[0] if node_data['metadata'].get('page_range') else 1,
                parent_id=None,  # Will be determined later
                children_ids=[],
                links_to=node_data.get('links_to', self._extract_links(node_data['text'])),
                footnotes=self._extract_footnotes(node_data['text']),
                metadata={
                    'country': node_data.get('country', country),
                    'law_name': node_data.get('law_name', 'Unknown'),
                    'theme': theme,
                    'clause_id': node_data['metadata'].get('clause_id', ''),
                    'summary': node_data.get('summary', ''),
                    'authority_level': node_data.get('authority_level', 0.5),
                    'valid_from': node_data.get('valid_from', 'Unknown'),
                    'valid_to': node_data.get('valid_to'),
                    'source_file': node_data['metadata'].get('source_file', ''),
                    'is_chunk': node_data['metadata'].get('is_chunk', False),
                    'chunk_index': node_data['metadata'].get('chunk_index', 0)
                }
            )
            nodes.append(node)
            
            # Check if this looks like a definition
            if self._is_definition_node(node_data):
                definition = self._extract_definition(node_data)
                if definition:
                    definitions.append(definition)
        
        # Build parent-child relationships based on hierarchy
        nodes = self._build_hierarchy(nodes)
        
        # Create DocumentStructure
        document_id = f"{theme}_{country}"
        title = f"{theme.replace('_', ' ').title()} - {country}"
        
        return DocumentStructure(
            document_id=document_id,
            title=title,
            pages=[],  # Not available in processed data
            hierarchy=nodes,
            definitions=definitions
        )
    
    def _determine_node_type(self, node_data: Dict[str, Any]) -> str:
        """Determine node type from metadata and content"""
        clause_id = node_data['metadata'].get('clause_id', '').lower()
        text = node_data['text'].lower()
        
        # Check for different types
        if 'titre' in clause_id or 'title' in clause_id:
            return 'section_header'
        elif 'chapitre' in clause_id or 'chapter' in clause_id:
            return 'section_header'
        elif 'article' in clause_id:
            return 'paragraph'
        elif 'section' in clause_id:
            return 'section_header'
        elif node_data['metadata'].get('is_chunk', False):
            return 'paragraph'
        elif any(keyword in text for keyword in ['défini', 'signifie', 'désigne', 'means', 'refers']):
            return 'definition'
        else:
            return 'paragraph'
    
    def _extract_links(self, text: str) -> List[str]:
        """Extract references to other articles/sections"""
        import re
        
        patterns = [
            r'article\s+(\d+[a-z]*)',
            r'art\.?\s+(\d+[a-z]*)',
            r'section\s+(\d+)',
            r'chapitre\s+(\d+)',
            r'voir\s+article\s+(\d+[a-z]*)',
            r'refer\s+to\s+article\s+(\d+[a-z]*)',
            r'liste\s+([ivx\d]+)'
        ]
        
        links = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            links.extend(matches)
        
        return list(set(links))
    
    def _extract_footnotes(self, text: str) -> List[str]:
        """Extract footnote references"""
        import re
        
        patterns = [
            r'\[(\d+)\]',
            r'footnote\s+(\d+)',
            r'note\s+(\d+)'
        ]
        
        footnotes = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            footnotes.extend(matches)
        
        return list(set(footnotes))
    
    def _is_definition_node(self, node_data: Dict[str, Any]) -> bool:
        """Check if node contains a definition"""
        text = node_data['text'].lower()
        summary = node_data.get('summary', '').lower()
        
        definition_keywords = [
            'défini', 'signifie', 'désigne', 'means', 'refers', 'constitute',
            'entend par', 's\'entend de', 'est considéré comme'
        ]
        
        return any(keyword in text or keyword in summary for keyword in definition_keywords)
    
    def _extract_definition(self, node_data: Dict[str, Any]) -> Optional[DefinitionNode]:
        """Extract definition from node"""
        text = node_data['text']
        
        # Try to parse term-definition pattern
        import re
        
        patterns = [
            r'["\']?([^"\']+)["\']?\s*(?:signifie|désigne|means|refers to|shall mean|est défini comme|s\'entend de)\s*(.+)',
            r'([^:]+):\s*(.+)',
            r'On entend par\s+([^:]+):\s*(.+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                term = match.group(1).strip()
                definition = match.group(2).strip()
                
                return DefinitionNode(
                    term=term,
                    definition=definition,
                    source_page=node_data['metadata'].get('page_range', [1])[0] if node_data['metadata'].get('page_range') else 1,
                    context=text
                )
        
        return None
    
    def _build_hierarchy(self, nodes: List[DocumentNode]) -> List[DocumentNode]:
        """Build parent-child relationships"""
        # Sort nodes by their natural order (by node_id which often contains order)
        nodes.sort(key=lambda x: x.node_id)
        
        # Simple hierarchy building based on node types
        for i, node in enumerate(nodes):
            # Look for potential parents in previous nodes
            for j in range(i-1, max(-1, i-5), -1):
                potential_parent = nodes[j]
                
                # Check if this could be a parent-child relationship
                if self._is_parent_child(potential_parent, node):
                    node.parent_id = potential_parent.node_id
                    potential_parent.children_ids.append(node.node_id)
                    break
        
        return nodes
    
    def _is_parent_child(self, parent: DocumentNode, child: DocumentNode) -> bool:
        """Determine if parent-child relationship exists"""
        # Simple hierarchy based on node types
        hierarchy = {
            'section_header': 0,
            'paragraph': 1,
            'definition': 2,
            'footer': 3
        }
        
        parent_level = hierarchy.get(parent.node_type, 99)
        child_level = hierarchy.get(child.node_type, 99)
        
        return parent_level < child_level
    
    def load_multiple_documents(self, theme: str, countries: List[str] = None) -> List[DocumentStructure]:
        """Load multiple documents for a theme"""
        if countries is None:
            countries = self.get_countries_for_theme(theme)
        
        documents = []
        for country in countries:
            doc = self.load_document(theme, country)
            if doc:
                documents.append(doc)
        
        return documents
    
    def get_document_summary(self) -> Dict[str, Any]:
        """Get summary of available documents"""
        summary = {
            'total_themes': len(self.available_files),
            'total_documents': sum(len(files) for files in self.available_files.values()),
            'themes': {}
        }
        
        for theme, files in self.available_files.items():
            summary['themes'][theme] = {
                'countries': len(files),
                'country_list': [f['country'] for f in files]
            }
        
        return summary
