# -*- coding: utf-8 -*-
"""
Definição dos tipos de arestas do grafo heterogêneo de jurisprudências
Implementa os 4 tipos de relações conforme especificado no TCC
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
from enum import Enum
import numpy as np


class EdgeType(Enum):
    """Tipos de arestas do grafo heterogêneo"""
    SIMILARITY = "similarity"
    RELEVANCE = "relevance"
    COOCCURRENCE = "cooccurrence"
    HIERARCHICAL = "hierarchical"


class SimilarityType(Enum):
    """Subtipos de similaridade"""
    DOCUMENT_SEMANTIC = "document_semantic"
    SECTION_CONTENT = "section_content"
    ENTITY_CONTEXT = "entity_context"


@dataclass
class BaseEdge:
    """Classe base para todas as arestas do grafo"""
    id: str
    source_node_id: str
    target_node_id: str
    edge_type: EdgeType
    weight: float
    
    def __post_init__(self):
        # Garante que o peso está no range válido para o tipo de aresta
        self.weight = self._validate_weight(self.weight)
        
        # Gera ID se não fornecido
        if not self.id:
            self.id = f"{self.edge_type.value}_{self.source_node_id}_{self.target_node_id}"
    
    def _validate_weight(self, weight: float) -> float:
        """Valida e normaliza o peso da aresta"""
        if weight < 0:
            return 0.0
        return weight
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte a aresta para dicionário"""
        base_dict = {
            'id': self.id,
            'source': self.source_node_id,
            'target': self.target_node_id,
            'edge_type': self.edge_type.value,
            'weight': self.weight,
        }
        # Adiciona created_at e metadata se existirem
        if hasattr(self, 'created_at'):
            base_dict['created_at'] = self.created_at.isoformat()
        if hasattr(self, 'metadata'):
            base_dict['metadata'] = self.metadata
        return base_dict
    
    def get_tuple(self) -> Tuple[str, str, Dict[str, Any]]:
        """Retorna tupla compatível com NetworkX"""
        attrs = {
            'id': self.id,
            'edge_type': self.edge_type.value,
            'weight': self.weight,
        }
        if hasattr(self, 'created_at'):
            attrs['created_at'] = self.created_at
        if hasattr(self, 'metadata'):
            attrs.update(self.metadata)
        
        return (self.source_node_id, self.target_node_id, attrs)


@dataclass
class SimilarityEdge(BaseEdge):
    """
    Aresta de similaridade semântica entre nós
    Baseada na similaridade de cossenos entre embeddings
    """
    # Campos obrigatórios primeiro
    similarity_type: SimilarityType
    cosine_score: float
    
    # Campos opcionais depois
    embedding_model: Optional[str] = None
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.edge_type = EdgeType.SIMILARITY
        self.weight = self.cosine_score
        super().__post_init__()
        
        # Metadados específicos
        self.metadata.update({
            'similarity_type': self.similarity_type.value,
            'cosine_score': self.cosine_score,
            'embedding_model': self.embedding_model,
            'confidence': self.confidence
        })
    
    def _validate_weight(self, weight: float) -> float:
        """Valida peso de similaridade (0.0 a 1.0)"""
        return max(0.0, min(1.0, weight))
    
    @classmethod
    def create_document_similarity(cls, doc1_id: str, doc2_id: str, 
                                 cosine_score: float, 
                                 embedding_model: str = None,
                                 confidence: float = None) -> 'SimilarityEdge':
        """Cria aresta de similaridade entre documentos"""
        return cls(
            id=f"sim_doc_{doc1_id}_{doc2_id}",
            source_node_id=doc1_id,
            target_node_id=doc2_id,
            edge_type=EdgeType.SIMILARITY,
            weight=cosine_score,
            similarity_type=SimilarityType.DOCUMENT_SEMANTIC,
            cosine_score=cosine_score,
            embedding_model=embedding_model,
            confidence=confidence
        )
    
    @classmethod
    def create_section_similarity(cls, sec1_id: str, sec2_id: str,
                                cosine_score: float,
                                embedding_model: str = None) -> 'SimilarityEdge':
        """Cria aresta de similaridade entre seções"""
        return cls(
            id=f"sim_sec_{sec1_id}_{sec2_id}",
            source_node_id=sec1_id,
            target_node_id=sec2_id,
            edge_type=EdgeType.SIMILARITY,
            weight=cosine_score,
            similarity_type=SimilarityType.SECTION_CONTENT,
            cosine_score=cosine_score,
            embedding_model=embedding_model
        )


@dataclass
class RelevanceEdge(BaseEdge):
    """
    Aresta de relevância entre documento e conceito
    Baseada no score TF-IDF do conceito no documento
    """
    # Campos obrigatórios
    tfidf_score: float
    term_frequency: int
    document_frequency: int
    inverse_document_frequency: float
    total_documents: int
    
    # Campos opcionais
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.edge_type = EdgeType.RELEVANCE
        self.weight = self.tfidf_score
        super().__post_init__()
        
        # Metadados específicos
        self.metadata.update({
            'tfidf_score': self.tfidf_score,
            'term_frequency': self.term_frequency,
            'document_frequency': self.document_frequency,
            'inverse_document_frequency': self.inverse_document_frequency,
            'total_documents': self.total_documents
        })
    
    def _validate_weight(self, weight: float) -> float:
        """Valida peso de relevância (0.0 a infinito, mas normaliza valores altos)"""
        return max(0.0, weight)
    
    @classmethod
    def create_document_concept_relevance(cls, doc_id: str, concept_id: str,
                                        tfidf_score: float,
                                        term_frequency: int,
                                        document_frequency: int,
                                        inverse_document_frequency: float,
                                        total_documents: int) -> 'RelevanceEdge':
        """Cria aresta de relevância documento-conceito"""
        return cls(
            id=f"rel_{doc_id}_{concept_id}",
            source_node_id=doc_id,
            target_node_id=concept_id,
            edge_type=EdgeType.RELEVANCE,
            weight=tfidf_score,
            tfidf_score=tfidf_score,
            term_frequency=term_frequency,
            document_frequency=document_frequency,
            inverse_document_frequency=inverse_document_frequency,
            total_documents=total_documents
        )


@dataclass
class CooccurrenceEdge(BaseEdge):
    """
    Aresta de co-ocorrência entre conceitos
    Baseada no Pointwise Mutual Information (PMI)
    """
    # Campos obrigatórios
    pmi_score: float
    joint_frequency: int
    concept1_frequency: int
    concept2_frequency: int
    total_windows: int
    window_size: int
    
    # Campos opcionais
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.edge_type = EdgeType.COOCCURRENCE
        self.weight = self.pmi_score
        super().__post_init__()
        
        # Metadados específicos
        self.metadata.update({
            'pmi_score': self.pmi_score,
            'joint_frequency': self.joint_frequency,
            'concept1_frequency': self.concept1_frequency,
            'concept2_frequency': self.concept2_frequency,
            'total_windows': self.total_windows,
            'window_size': self.window_size
        })
    
    def _validate_weight(self, weight: float) -> float:
        """Valida peso PMI (pode ser negativo)"""
        return weight  # PMI pode ser negativo
    
    @classmethod
    def create_concept_cooccurrence(cls, concept1_id: str, concept2_id: str,
                                  pmi_score: float,
                                  joint_frequency: int,
                                  concept1_frequency: int,
                                  concept2_frequency: int,
                                  total_windows: int,
                                  window_size: int = 10) -> 'CooccurrenceEdge':
        """Cria aresta de co-ocorrência entre conceitos"""
        return cls(
            id=f"cooc_{concept1_id}_{concept2_id}",
            source_node_id=concept1_id,
            target_node_id=concept2_id,
            edge_type=EdgeType.COOCCURRENCE,
            weight=pmi_score,
            pmi_score=pmi_score,
            joint_frequency=joint_frequency,
            concept1_frequency=concept1_frequency,
            concept2_frequency=concept2_frequency,
            total_windows=total_windows,
            window_size=window_size
        )


@dataclass
class HierarchicalEdge(BaseEdge):
    """
    Aresta hierárquica estrutural
    Conecta elementos em hierarquia (Doc->Seção, Seção->Entidade)
    """
    # Campos obrigatórios
    hierarchy_type: str
    
    # Campos opcionais
    order: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.edge_type = EdgeType.HIERARCHICAL
        self.weight = 1.0  # Peso fixo para relações hierárquicas
        super().__post_init__()
        
        # Metadados específicos
        self.metadata.update({
            'hierarchy_type': self.hierarchy_type,
            'order': self.order
        })
    
    @classmethod
    def create_document_section(cls, doc_id: str, section_id: str,
                              order: int = None) -> 'HierarchicalEdge':
        """Cria aresta hierárquica documento->seção"""
        return cls(
            id=f"hier_doc_sec_{doc_id}_{section_id}",
            source_node_id=doc_id,
            target_node_id=section_id,
            edge_type=EdgeType.HIERARCHICAL,
            weight=1.0,
            hierarchy_type="document_section",
            order=order
        )
    
    @classmethod
    def create_section_entity(cls, section_id: str, entity_id: str) -> 'HierarchicalEdge':
        """Cria aresta hierárquica seção->entidade"""
        return cls(
            id=f"hier_sec_ent_{section_id}_{entity_id}",
            source_node_id=section_id,
            target_node_id=entity_id,
            edge_type=EdgeType.HIERARCHICAL,
            weight=1.0,
            hierarchy_type="section_entity"
        )
    
    @classmethod
    def create_document_entity(cls, doc_id: str, entity_id: str) -> 'HierarchicalEdge':
        """Cria aresta hierárquica documento->entidade (direto)"""
        return cls(
            id=f"hier_doc_ent_{doc_id}_{entity_id}",
            source_node_id=doc_id,
            target_node_id=entity_id,
            edge_type=EdgeType.HIERARCHICAL,
            weight=1.0,
            hierarchy_type="document_entity"
        )


# Configurações padrão para diferentes tipos de arestas

DEFAULT_SIMILARITY_THRESHOLDS = {
    SimilarityType.DOCUMENT_SEMANTIC: 0.3,
    SimilarityType.SECTION_CONTENT: 0.4,
    SimilarityType.ENTITY_CONTEXT: 0.5
}

DEFAULT_RELEVANCE_THRESHOLD = 0.1
DEFAULT_PMI_THRESHOLD = 0.5

# Limites para evitar grafo muito denso
MAX_EDGES_PER_NODE = {
    EdgeType.SIMILARITY: 50,
    EdgeType.RELEVANCE: 20,
    EdgeType.COOCCURRENCE: 30,
    EdgeType.HIERARCHICAL: 100
}