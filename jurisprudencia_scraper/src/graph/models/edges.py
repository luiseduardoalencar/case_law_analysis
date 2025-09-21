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
    SIMILARITY = "similarity"          # Doc<->Doc, Seção<->Seção (cossenos)
    RELEVANCE = "relevance"           # Doc<->Conceito (TF-IDF)
    COOCCURRENCE = "cooccurrence"     # Conceito<->Conceito (PMI)
    HIERARCHICAL = "hierarchical"     # Doc<->Seção, Seção<->Entidade


class SimilarityType(Enum):
    """Subtipos de similaridade"""
    DOCUMENT_SEMANTIC = "document_semantic"      # Doc<->Doc semântica
    SECTION_CONTENT = "section_content"          # Seção<->Seção conteúdo
    ENTITY_CONTEXT = "entity_context"            # Entidade<->Entidade contexto


@dataclass
class BaseEdge:
    """Classe base para todas as arestas do grafo"""
    id: str
    source_node_id: str
    target_node_id: str
    edge_type: EdgeType
    weight: float
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
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
        return {
            'id': self.id,
            'source': self.source_node_id,
            'target': self.target_node_id,
            'edge_type': self.edge_type.value,
            'weight': self.weight,
            'created_at': self.created_at.isoformat(),
            'metadata': self.metadata
        }
    
    def get_tuple(self) -> Tuple[str, str, Dict[str, Any]]:
        """Retorna tupla compatível com NetworkX"""
        return (
            self.source_node_id,
            self.target_node_id,
            {
                'id': self.id,
                'edge_type': self.edge_type.value,
                'weight': self.weight,
                'created_at': self.created_at,
                **self.metadata
            }
        )


@dataclass
class SimilarityEdge(BaseEdge):
    """
    Aresta de similaridade semântica entre nós
    Baseada na similaridade de cossenos entre embeddings
    """
    similarity_type: SimilarityType
    cosine_score: float
    embedding_model: Optional[str] = None
    confidence: Optional[float] = None
    
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
    tfidf_score: float
    term_frequency: int
    document_frequency: int
    inverse_document_frequency: float
    total_documents: int
    
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
    pmi_score: float
    joint_frequency: int
    concept1_frequency: int
    concept2_frequency: int
    total_windows: int
    window_size: int
    
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
    hierarchy_type: str  # "document_section", "section_entity", etc.
    order: Optional[int] = None  # Ordem na hierarquia
    
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
            hierarchy_type="section_entity"
        )
    
    @classmethod
    def create_document_entity(cls, doc_id: str, entity_id: str) -> 'HierarchicalEdge':
        """Cria aresta hierárquica documento->entidade (direto)"""
        return cls(
            id=f"hier_doc_ent_{doc_id}_{entity_id}",
            source_node_id=doc_id,
            target_node_id=entity_id,
            hierarchy_type="document_entity"
        )


# Classes auxiliares para estatísticas e análise

@dataclass
class EdgeStatistics:
    """Estatísticas de arestas para análise da qualidade do grafo"""
    edge_type: EdgeType
    total_edges: int
    min_weight: float
    max_weight: float
    mean_weight: float
    std_weight: float
    weight_distribution: Dict[str, int] = field(default_factory=dict)
    
    def add_edge_weight(self, weight: float):
        """Adiciona peso para atualizar estatísticas"""
        # Implementação simplificada - em produção usar biblioteca estatística
        pass


# Funções auxiliares para criação e validação de arestas

def create_similarity_matrix_edges(embeddings_dict: Dict[str, np.ndarray],
                                 threshold: float = 0.3,
                                 similarity_type: SimilarityType = SimilarityType.DOCUMENT_SEMANTIC,
                                 max_edges_per_node: int = 50) -> List[SimilarityEdge]:
    """
    Cria arestas de similaridade a partir de uma matriz de embeddings
    
    Args:
        embeddings_dict: Dicionário {node_id: embedding_vector}
        threshold: Threshold mínimo de similaridade
        similarity_type: Tipo de similaridade
        max_edges_per_node: Máximo de arestas por nó (para eficiência)
    
    Returns:
        Lista de SimilarityEdge
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    edges = []
    node_ids = list(embeddings_dict.keys())
    embeddings_matrix = np.array([embeddings_dict[node_id] for node_id in node_ids])
    
    # Calcula matriz de similaridade
    similarity_matrix = cosine_similarity(embeddings_matrix)
    
    for i, node1_id in enumerate(node_ids):
        # Pega similaridades para este nó
        similarities = [(j, similarity_matrix[i][j]) for j in range(len(node_ids)) 
                       if i != j and similarity_matrix[i][j] >= threshold]
        
        # Ordena por similaridade decrescente e limita
        similarities.sort(key=lambda x: x[1], reverse=True)
        similarities = similarities[:max_edges_per_node]
        
        for j, sim_score in similarities:
            node2_id = node_ids[j]
            
            # Evita arestas duplicadas (apenas node1 < node2)
            if node1_id < node2_id:
                edge = SimilarityEdge(
                    id=f"sim_{similarity_type.value}_{node1_id}_{node2_id}",
                    source_node_id=node1_id,
                    target_node_id=node2_id,
                    similarity_type=similarity_type,
                    cosine_score=sim_score
                )
                edges.append(edge)
    
    return edges


def create_tfidf_relevance_edges(tfidf_matrix: np.ndarray,
                                doc_ids: List[str],
                                concept_ids: List[str],
                                threshold: float = 0.1) -> List[RelevanceEdge]:
    """
    Cria arestas de relevância a partir de uma matriz TF-IDF
    
    Args:
        tfidf_matrix: Matriz TF-IDF (docs x conceitos)
        doc_ids: Lista de IDs dos documentos
        concept_ids: Lista de IDs dos conceitos
        threshold: Threshold mínimo de relevância
    
    Returns:
        Lista de RelevanceEdge
    """
    edges = []
    
    for i, doc_id in enumerate(doc_ids):
        for j, concept_id in enumerate(concept_ids):
            tfidf_score = tfidf_matrix[i, j]
            
            if tfidf_score >= threshold:
                edge = RelevanceEdge(
                    id=f"rel_{doc_id}_{concept_id}",
                    source_node_id=doc_id,
                    target_node_id=concept_id,
                    tfidf_score=tfidf_score,
                    term_frequency=0,  # Será preenchido pelo calculador TF-IDF
                    document_frequency=0,
                    inverse_document_frequency=0.0,
                    total_documents=len(doc_ids)
                )
                edges.append(edge)
    
    return edges


def create_pmi_cooccurrence_edges(pmi_matrix: np.ndarray,
                                concept_ids: List[str],
                                threshold: float = 0.5) -> List[CooccurrenceEdge]:
    """
    Cria arestas de co-ocorrência a partir de uma matriz PMI
    
    Args:
        pmi_matrix: Matriz PMI simétrica (conceitos x conceitos)
        concept_ids: Lista de IDs dos conceitos
        threshold: Threshold mínimo de PMI
    
    Returns:
        Lista de CooccurrenceEdge
    """
    edges = []
    
    for i, concept1_id in enumerate(concept_ids):
        for j, concept2_id in enumerate(concept_ids):
            if i < j:  # Evita duplicatas (matriz simétrica)
                pmi_score = pmi_matrix[i, j]
                
                if pmi_score >= threshold:
                    edge = CooccurrenceEdge(
                        id=f"cooc_{concept1_id}_{concept2_id}",
                        source_node_id=concept1_id,
                        target_node_id=concept2_id,
                        pmi_score=pmi_score,
                        joint_frequency=0,  # Será preenchido pelo calculador PMI
                        concept1_frequency=0,
                        concept2_frequency=0,
                        total_windows=0,
                        window_size=10
                    )
                    edges.append(edge)
    
    return edges


def validate_edge_weights(edges: List[BaseEdge]) -> Dict[str, Any]:
    """
    Valida e analisa os pesos das arestas
    
    Args:
        edges: Lista de arestas para validar
    
    Returns:
        Dicionário com estatísticas de validação
    """
    stats = {}
    
    # Agrupa por tipo de aresta
    edges_by_type = {}
    for edge in edges:
        edge_type = edge.edge_type.value
        if edge_type not in edges_by_type:
            edges_by_type[edge_type] = []
        edges_by_type[edge_type].append(edge.weight)
    
    # Calcula estatísticas por tipo
    for edge_type, weights in edges_by_type.items():
        weights_array = np.array(weights)
        stats[edge_type] = {
            'count': len(weights),
            'min': float(np.min(weights_array)),
            'max': float(np.max(weights_array)),
            'mean': float(np.mean(weights_array)),
            'std': float(np.std(weights_array)),
            'median': float(np.median(weights_array))
        }
    
    return stats


def filter_edges_by_weight(edges: List[BaseEdge], 
                          min_weights: Dict[EdgeType, float]) -> List[BaseEdge]:
    """
    Filtra arestas por peso mínimo por tipo
    
    Args:
        edges: Lista de arestas
        min_weights: Dicionário {EdgeType: peso_minimo}
    
    Returns:
        Lista de arestas filtradas
    """
    filtered_edges = []
    
    for edge in edges:
        min_weight = min_weights.get(edge.edge_type, 0.0)
        if edge.weight >= min_weight:
            filtered_edges.append(edge)
    
    return filtered_edges


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
    EdgeType.HIERARCHICAL: 100  # Sem limite prático
}