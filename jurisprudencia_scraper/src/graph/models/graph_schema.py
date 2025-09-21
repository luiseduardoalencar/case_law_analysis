# -*- coding: utf-8 -*-
"""
Schema completo do grafo heterogêneo de jurisprudências
Define a estrutura, validações e configurações do grafo
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Optional, Tuple
from datetime import datetime
import networkx as nx
from enum import Enum

from .nodes import (
    NodeType, DocumentNode, SectionNode, EntityNode, ConceptNode,
    SectionType, EntityType
)
from .edges import (
    EdgeType, SimilarityEdge, RelevanceEdge, CooccurrenceEdge, HierarchicalEdge,
    SimilarityType, DEFAULT_SIMILARITY_THRESHOLDS, DEFAULT_RELEVANCE_THRESHOLD,
    DEFAULT_PMI_THRESHOLD, MAX_EDGES_PER_NODE
)


class GraphValidationLevel(Enum):
    """Níveis de validação do grafo"""
    BASIC = "basic"           # Validação básica de estrutura
    COMPREHENSIVE = "comprehensive"  # Validação completa
    STRICT = "strict"         # Validação rigorosa


@dataclass
class GraphConfiguration:
    """Configuração para construção do grafo"""
    
    # Thresholds para arestas
    similarity_thresholds: Dict[SimilarityType, float] = field(
        default_factory=lambda: DEFAULT_SIMILARITY_THRESHOLDS.copy()
    )
    relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD
    pmi_threshold: float = DEFAULT_PMI_THRESHOLD
    
    # Limites de conectividade
    max_edges_per_node: Dict[EdgeType, int] = field(
        default_factory=lambda: MAX_EDGES_PER_NODE.copy()
    )
    max_similarity_edges_per_document: int = 50
    max_relevance_edges_per_document: int = 20
    max_cooccurrence_edges_per_concept: int = 30
    
    # Configurações de embeddings
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_dimension: int = 384
    max_sequence_length: int = 512
    
    # Configurações de processamento
    chunk_size: int = 1000  # Para processamento em lotes
    use_cache: bool = True
    cache_embeddings: bool = True
    
    # Configurações de NER
    ner_model: str = "pt_core_news_sm"
    min_entity_frequency: int = 2
    
    # Configurações de conceitos
    min_concept_frequency: int = 3
    max_concepts: int = 1000
    
    # Configurações de validação
    validation_level: GraphValidationLevel = GraphValidationLevel.COMPREHENSIVE
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte configuração para dicionário"""
        return {
            'similarity_thresholds': {k.value: v for k, v in self.similarity_thresholds.items()},
            'relevance_threshold': self.relevance_threshold,
            'pmi_threshold': self.pmi_threshold,
            'max_edges_per_node': {k.value: v for k, v in self.max_edges_per_node.items()},
            'embedding_model': self.embedding_model,
            'embedding_dimension': self.embedding_dimension,
            'max_sequence_length': self.max_sequence_length,
            'chunk_size': self.chunk_size,
            'use_cache': self.use_cache,
            'ner_model': self.ner_model,
            'min_entity_frequency': self.min_entity_frequency,
            'min_concept_frequency': self.min_concept_frequency,
            'max_concepts': self.max_concepts,
            'validation_level': self.validation_level.value
        }


@dataclass
class GraphStatistics:
    """Estatísticas do grafo construído"""
    
    # Contadores de nós
    num_documents: int = 0
    num_sections: int = 0
    num_entities: int = 0
    num_concepts: int = 0
    total_nodes: int = 0
    
    # Contadores de arestas
    num_similarity_edges: int = 0
    num_relevance_edges: int = 0
    num_cooccurrence_edges: int = 0
    num_hierarchical_edges: int = 0
    total_edges: int = 0
    
    # Métricas de conectividade
    density: float = 0.0
    average_degree: float = 0.0
    max_degree: int = 0
    num_connected_components: int = 0
    largest_component_size: int = 0
    
    # Métricas de qualidade
    average_similarity_weight: float = 0.0
    average_relevance_weight: float = 0.0
    average_pmi_weight: float = 0.0
    
    # Distribuições
    degree_distribution: Dict[int, int] = field(default_factory=dict)
    weight_distributions: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Metadados
    construction_time: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def update_from_networkx(self, graph: nx.Graph):
        """Atualiza estatísticas a partir de um grafo NetworkX"""
        
        # Contadores básicos
        self.total_nodes = graph.number_of_nodes()
        self.total_edges = graph.number_of_edges()
        self.density = nx.density(graph)
        
        if self.total_nodes > 0:
            self.average_degree = sum(dict(graph.degree()).values()) / self.total_nodes
            self.max_degree = max(dict(graph.degree()).values()) if graph.degree() else 0
        
        # Componentes conectados
        if isinstance(graph, nx.Graph):
            components = list(nx.connected_components(graph))
            self.num_connected_components = len(components)
            self.largest_component_size = len(max(components, key=len)) if components else 0
        
        # Contadores por tipo de nó
        node_types = {}
        for node_id, node_data in graph.nodes(data=True):
            node_type = node_data.get('node_type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        self.num_documents = node_types.get('document', 0)
        self.num_sections = node_types.get('section', 0)
        self.num_entities = node_types.get('entity', 0)
        self.num_concepts = node_types.get('concept', 0)
        
        # Contadores por tipo de aresta
        edge_types = {}
        similarity_weights = []
        relevance_weights = []
        pmi_weights = []
        
        for u, v, edge_data in graph.edges(data=True):
            edge_type = edge_data.get('edge_type', 'unknown')
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
            
            weight = edge_data.get('weight', 0)
            if edge_type == 'similarity':
                similarity_weights.append(weight)
            elif edge_type == 'relevance':
                relevance_weights.append(weight)
            elif edge_type == 'cooccurrence':
                pmi_weights.append(weight)
        
        self.num_similarity_edges = edge_types.get('similarity', 0)
        self.num_relevance_edges = edge_types.get('relevance', 0)
        self.num_cooccurrence_edges = edge_types.get('cooccurrence', 0)
        self.num_hierarchical_edges = edge_types.get('hierarchical', 0)
        
        # Médias de pesos
        if similarity_weights:
            self.average_similarity_weight = sum(similarity_weights) / len(similarity_weights)
        if relevance_weights:
            self.average_relevance_weight = sum(relevance_weights) / len(relevance_weights)
        if pmi_weights:
            self.average_pmi_weight = sum(pmi_weights) / len(pmi_weights)
        
        # Distribuição de graus
        degrees = [degree for node, degree in graph.degree()]
        self.degree_distribution = {}
        for degree in degrees:
            self.degree_distribution[degree] = self.degree_distribution.get(degree, 0) + 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte estatísticas para dicionário"""
        return {
            'nodes': {
                'documents': self.num_documents,
                'sections': self.num_sections,
                'entities': self.num_entities,
                'concepts': self.num_concepts,
                'total': self.total_nodes
            },
            'edges': {
                'similarity': self.num_similarity_edges,
                'relevance': self.num_relevance_edges,
                'cooccurrence': self.num_cooccurrence_edges,
                'hierarchical': self.num_hierarchical_edges,
                'total': self.total_edges
            },
            'connectivity': {
                'density': self.density,
                'average_degree': self.average_degree,
                'max_degree': self.max_degree,
                'num_components': self.num_connected_components,
                'largest_component_size': self.largest_component_size
            },
            'weights': {
                'average_similarity': self.average_similarity_weight,
                'average_relevance': self.average_relevance_weight,
                'average_pmi': self.average_pmi_weight
            },
            'performance': {
                'construction_time': self.construction_time,
                'memory_usage_mb': self.memory_usage_mb
            },
            'created_at': self.created_at.isoformat()
        }


@dataclass
class GraphValidationResult:
    """Resultado da validação do grafo"""
    
    is_valid: bool
    validation_level: GraphValidationLevel
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    statistics: Optional[GraphStatistics] = None
    
    def add_warning(self, message: str):
        """Adiciona aviso de validação"""
        self.warnings.append(message)
    
    def add_error(self, message: str):
        """Adiciona erro de validação"""
        self.errors.append(message)
        self.is_valid = False
    
    def add_recommendation(self, message: str):
        """Adiciona recomendação"""
        self.recommendations.append(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte resultado para dicionário"""
        return {
            'is_valid': self.is_valid,
            'validation_level': self.validation_level.value,
            'warnings': self.warnings,
            'errors': self.errors,
            'recommendations': self.recommendations,
            'statistics': self.statistics.to_dict() if self.statistics else None
        }


class GraphSchema:
    """
    Schema principal do grafo heterogêneo
    Gerencia estrutura, validação e configurações
    """
    
    def __init__(self, config: Optional[GraphConfiguration] = None):
        self.config = config or GraphConfiguration()
        self.statistics = GraphStatistics()
        
        # Define tipos de nós permitidos e suas relações
        self.allowed_node_types = {NodeType.DOCUMENT, NodeType.SECTION, 
                                 NodeType.ENTITY, NodeType.CONCEPT}
        
        # Define tipos de arestas permitidas entre tipos de nós
        self.allowed_edge_types = {
            # Similaridade: mesmo tipo de nó
            (NodeType.DOCUMENT, NodeType.DOCUMENT): {EdgeType.SIMILARITY},
            (NodeType.SECTION, NodeType.SECTION): {EdgeType.SIMILARITY},
            (NodeType.ENTITY, NodeType.ENTITY): {EdgeType.SIMILARITY},
            
            # Relevância: documento <-> conceito
            (NodeType.DOCUMENT, NodeType.CONCEPT): {EdgeType.RELEVANCE},
            (NodeType.CONCEPT, NodeType.DOCUMENT): {EdgeType.RELEVANCE},
            
            # Co-ocorrência: conceito <-> conceito
            (NodeType.CONCEPT, NodeType.CONCEPT): {EdgeType.COOCCURRENCE},
            
            # Hierárquicas: várias combinações
            (NodeType.DOCUMENT, NodeType.SECTION): {EdgeType.HIERARCHICAL},
            (NodeType.SECTION, NodeType.ENTITY): {EdgeType.HIERARCHICAL},
            (NodeType.DOCUMENT, NodeType.ENTITY): {EdgeType.HIERARCHICAL},
            (NodeType.SECTION, NodeType.CONCEPT): {EdgeType.HIERARCHICAL},
        }
    
    def validate_node(self, node) -> List[str]:
        """Valida um nó individual"""
        errors = []
        
        # Validações básicas
        if not hasattr(node, 'id') or not node.id:
            errors.append(f"Nó sem ID válido: {node}")
        
        if not hasattr(node, 'node_type') or node.node_type not in self.allowed_node_types:
            errors.append(f"Tipo de nó inválido: {getattr(node, 'node_type', 'undefined')}")
        
        if not hasattr(node, 'label') or not node.label:
            errors.append(f"Nó sem label: {node.id}")
        
        # Validações específicas por tipo
        if isinstance(node, DocumentNode):
            if not node.numero_processo:
                errors.append(f"DocumentNode sem numero_processo: {node.id}")
        
        elif isinstance(node, SectionNode):
            if not node.parent_document_id:
                errors.append(f"SectionNode sem parent_document_id: {node.id}")
            if node.section_type not in SectionType:
                errors.append(f"SectionNode com section_type inválido: {node.section_type}")
        
        elif isinstance(node, EntityNode):
            if node.entity_type not in EntityType:
                errors.append(f"EntityNode com entity_type inválido: {node.entity_type}")
            if not node.nome_normalizado:
                errors.append(f"EntityNode sem nome_normalizado: {node.id}")
        
        elif isinstance(node, ConceptNode):
            if not node.termo_conceito:
                errors.append(f"ConceptNode sem termo_conceito: {node.id}")
        
        return errors
    
    def validate_edge(self, edge, source_node_type: NodeType, 
                     target_node_type: NodeType) -> List[str]:
        """Valida uma aresta individual"""
        errors = []
        
        # Validações básicas
        if not hasattr(edge, 'id') or not edge.id:
            errors.append(f"Aresta sem ID válido: {edge}")
        
        if not hasattr(edge, 'weight') or edge.weight is None:
            errors.append(f"Aresta sem peso: {edge.id}")
        
        # Valida tipos permitidos
        edge_types = self.allowed_edge_types.get((source_node_type, target_node_type), set())
        if edge.edge_type not in edge_types:
            errors.append(
                f"Tipo de aresta não permitido: {edge.edge_type} entre "
                f"{source_node_type} e {target_node_type}"
            )
        
        # Validações específicas por tipo de aresta
        if isinstance(edge, SimilarityEdge):
            if not (0.0 <= edge.weight <= 1.0):
                errors.append(f"SimilarityEdge com peso inválido: {edge.weight}")
        
        elif isinstance(edge, RelevanceEdge):
            if edge.weight < 0.0:
                errors.append(f"RelevanceEdge com peso negativo: {edge.weight}")
        
        elif isinstance(edge, CooccurrenceEdge):
            # PMI pode ser negativo, sem validação de range
            pass
        
        elif isinstance(edge, HierarchicalEdge):
            if edge.weight != 1.0:
                errors.append(f"HierarchicalEdge deve ter peso 1.0: {edge.weight}")
        
        return errors
    
    def validate_graph(self, graph: nx.Graph, 
                      level: GraphValidationLevel = None) -> GraphValidationResult:
        """Valida o grafo completo"""
        
        level = level or self.config.validation_level
        result = GraphValidationResult(is_valid=True, validation_level=level)
        
        # Atualiza estatísticas
        stats = GraphStatistics()
        stats.update_from_networkx(graph)
        result.statistics = stats
        
        # Validação básica
        if graph.number_of_nodes() == 0:
            result.add_error("Grafo vazio (sem nós)")
            return result
        
        if graph.number_of_edges() == 0:
            result.add_warning("Grafo sem arestas")
        
        # Validação de conectividade
        if not nx.is_connected(graph):
            result.add_warning(
                f"Grafo não conectado ({stats.num_connected_components} componentes)"
            )
        
        if level in [GraphValidationLevel.COMPREHENSIVE, GraphValidationLevel.STRICT]:
            # Validação detalhada de nós e arestas
            self._validate_nodes_detailed(graph, result)
            self._validate_edges_detailed(graph, result)
        
        if level == GraphValidationLevel.STRICT:
            # Validações rigorosas
            self._validate_strict_requirements(graph, result)
        
        return result
    
    def _validate_nodes_detailed(self, graph: nx.Graph, result: GraphValidationResult):
        """Validação detalhada dos nós"""
        
        node_type_counts = {}
        
        for node_id, node_data in graph.nodes(data=True):
            node_type = node_data.get('node_type')
            if node_type:
                node_type_counts[node_type] = node_type_counts.get(node_type, 0) + 1
            else:
                result.add_error(f"Nó sem tipo: {node_id}")
        
        # Verifica se tem pelo menos alguns documentos
        if node_type_counts.get('document', 0) < 10:
            result.add_warning("Poucos documentos no grafo")
        
        # Verifica proporções
        total_nodes = sum(node_type_counts.values())
        if total_nodes > 0:
            doc_ratio = node_type_counts.get('document', 0) / total_nodes
            if doc_ratio < 0.1:  # Menos de 10% documentos
                result.add_warning("Proporção baixa de documentos")
    
    def _validate_edges_detailed(self, graph: nx.Graph, result: GraphValidationResult):
        """Validação detalhada das arestas"""
        
        edge_type_counts = {}
        weight_issues = []
        
        for u, v, edge_data in graph.edges(data=True):
            edge_type = edge_data.get('edge_type')
            if edge_type:
                edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1
            
            weight = edge_data.get('weight')
            if weight is None:
                result.add_error(f"Aresta sem peso: {u} -> {v}")
            elif edge_type == 'similarity' and not (0.0 <= weight <= 1.0):
                weight_issues.append(f"Similaridade inválida: {weight}")
        
        if weight_issues:
            result.add_error(f"Pesos inválidos encontrados: {len(weight_issues)} casos")
        
        # Verifica se tem arestas de diferentes tipos
        if len(edge_type_counts) < 2:
            result.add_warning("Grafo com poucos tipos de arestas")
    
    def _validate_strict_requirements(self, graph: nx.Graph, result: GraphValidationResult):
        """Validações rigorosas para produção"""
        
        # Densidade muito alta pode indicar problema
        if result.statistics.density > 0.1:
            result.add_warning("Densidade muito alta - grafo pode estar muito conectado")
        
        # Componente principal deve ser significativo
        if result.statistics.largest_component_size < result.statistics.total_nodes * 0.8:
            result.add_error("Componente principal muito pequeno")
        
        # Deve ter diversidade de tipos
        required_edge_types = {'similarity', 'relevance', 'hierarchical'}
        existing_edge_types = set()
        
        for u, v, edge_data in graph.edges(data=True):
            existing_edge_types.add(edge_data.get('edge_type'))
        
        missing_types = required_edge_types - existing_edge_types
        if missing_types:
            result.add_error(f"Tipos de aresta obrigatórios ausentes: {missing_types}")
    
    def get_recommended_thresholds(self, num_documents: int) -> Dict[str, float]:
        """Retorna thresholds recomendados baseado no tamanho do corpus"""
        
        if num_documents < 100:
            return {
                'similarity': 0.2,
                'relevance': 0.05,
                'pmi': 0.3
            }
        elif num_documents < 1000:
            return {
                'similarity': 0.3,
                'relevance': 0.1,
                'pmi': 0.5
            }
        else:
            return {
                'similarity': 0.4,
                'relevance': 0.15,
                'pmi': 0.7
            }
    
    def optimize_config_for_size(self, num_documents: int) -> GraphConfiguration:
        """
        Otimiza configuração baseada no tamanho do corpus
        
        Args:
            num_documents: Número de documentos no corpus
            
        Returns:
            GraphConfiguration otimizada
        """
        config = GraphConfiguration()
        
        # Ajusta thresholds baseado no tamanho
        if num_documents < 500:
            # Corpus pequeno - thresholds mais baixos para manter conectividade
            config.similarity_thresholds[SimilarityType.DOCUMENT_SEMANTIC] = 0.25
            config.similarity_thresholds[SimilarityType.SECTION_CONTENT] = 0.3
            config.relevance_threshold = 0.05
            config.pmi_threshold = 0.3
            config.max_similarity_edges_per_document = 20
            
        elif num_documents < 2000:
            # Corpus médio - configuração padrão
            config.similarity_thresholds[SimilarityType.DOCUMENT_SEMANTIC] = 0.3
            config.similarity_thresholds[SimilarityType.SECTION_CONTENT] = 0.4
            config.relevance_threshold = 0.1
            config.pmi_threshold = 0.5
            config.max_similarity_edges_per_document = 30
            
        else:
            # Corpus grande - thresholds mais altos para eficiência
            config.similarity_thresholds[SimilarityType.DOCUMENT_SEMANTIC] = 0.4
            config.similarity_thresholds[SimilarityType.SECTION_CONTENT] = 0.5
            config.relevance_threshold = 0.15
            config.pmi_threshold = 0.7
            config.max_similarity_edges_per_document = 50
        
        # Ajusta configurações de processamento
        config.chunk_size = min(1000, max(100, num_documents // 10))
        
        return config


# Classes para persistência e serialização

@dataclass
class GraphMetadata:
    """Metadados do grafo para persistência"""
    
    schema_version: str = "1.0"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Informações do corpus
    corpus_source: str = "TJPI - Empréstimos Consignados"
    corpus_size: int = 0
    date_range: Optional[Tuple[datetime, datetime]] = None
    
    # Configuração usada
    configuration: Optional[Dict[str, Any]] = None
    
    # Estatísticas
    statistics: Optional[Dict[str, Any]] = None
    
    # Validação
    last_validation: Optional[Dict[str, Any]] = None
    
    # Processamento
    processing_log: List[str] = field(default_factory=list)
    
    def add_log_entry(self, message: str):
        """Adiciona entrada ao log de processamento"""
        self.processing_log.append(f"{datetime.now().isoformat()}: {message}")
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte metadados para dicionário"""
        return {
            'schema_version': self.schema_version,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'corpus_source': self.corpus_source,
            'corpus_size': self.corpus_size,
            'date_range': [d.isoformat() for d in self.date_range] if self.date_range else None,
            'configuration': self.configuration,
            'statistics': self.statistics,
            'last_validation': self.last_validation,
            'processing_log': self.processing_log
        }


class GraphSchemaValidator:
    """Validador especializado para o schema do grafo"""
    
    def __init__(self, schema: GraphSchema):
        self.schema = schema
    
    def validate_construction_pipeline(self, pipeline_results: Dict[str, Any]) -> GraphValidationResult:
        """
        Valida os resultados do pipeline de construção
        
        Args:
            pipeline_results: Dicionário com resultados de cada etapa
            
        Returns:
            GraphValidationResult com validação do pipeline
        """
        result = GraphValidationResult(
            is_valid=True, 
            validation_level=GraphValidationLevel.COMPREHENSIVE
        )
        
        # Valida etapas obrigatórias
        required_steps = [
            'data_loading',
            'section_extraction', 
            'entity_extraction',
            'concept_extraction',
            'vectorization',
            'similarity_calculation',
            'graph_construction'
        ]
        
        for step in required_steps:
            if step not in pipeline_results:
                result.add_error(f"Etapa obrigatória ausente: {step}")
            elif not pipeline_results[step].get('success', False):
                result.add_error(f"Etapa falhou: {step}")
        
        # Valida métricas de qualidade
        if 'vectorization' in pipeline_results:
            vectorization = pipeline_results['vectorization']
            if vectorization.get('embedding_coverage', 0) < 0.9:
                result.add_warning("Baixa cobertura de embeddings")
        
        if 'entity_extraction' in pipeline_results:
            ner = pipeline_results['entity_extraction']
            if ner.get('entities_found', 0) < 100:
                result.add_warning("Poucas entidades extraídas")
        
        return result
    
    def recommend_optimizations(self, graph: nx.Graph) -> List[str]:
        """
        Recomenda otimizações baseadas na análise do grafo
        
        Args:
            graph: Grafo construído
            
        Returns:
            Lista de recomendações
        """
        recommendations = []
        stats = GraphStatistics()
        stats.update_from_networkx(graph)
        
        # Densidade
        if stats.density < 0.001:
            recommendations.append(
                "Densidade muito baixa - considere reduzir thresholds de similaridade"
            )
        elif stats.density > 0.01:
            recommendations.append(
                "Densidade muito alta - considere aumentar thresholds ou limitar arestas por nó"
            )
        
        # Conectividade
        if stats.num_connected_components > stats.total_nodes * 0.1:
            recommendations.append(
                "Muitos componentes desconexos - considere adicionar mais arestas hierárquicas"
            )
        
        # Distribuição de graus
        if stats.max_degree > stats.average_degree * 10:
            recommendations.append(
                "Alguns nós com grau muito alto - considere limitar arestas por nó"
            )
        
        # Tipos de nós
        if stats.num_concepts < 100:
            recommendations.append(
                "Poucos conceitos - considere expandir lista de conceitos jurídicos"
            )
        
        if stats.num_entities < stats.num_documents * 0.5:
            recommendations.append(
                "Poucas entidades por documento - verificar qualidade do NER"
            )
        
        return recommendations


# Configurações pré-definidas para diferentes cenários

DEVELOPMENT_CONFIG = GraphConfiguration(
    similarity_thresholds={
        SimilarityType.DOCUMENT_SEMANTIC: 0.2,
        SimilarityType.SECTION_CONTENT: 0.3,
        SimilarityType.ENTITY_CONTEXT: 0.4
    },
    relevance_threshold=0.05,
    pmi_threshold=0.3,
    max_similarity_edges_per_document=10,
    chunk_size=100,
    validation_level=GraphValidationLevel.BASIC
)

PRODUCTION_CONFIG = GraphConfiguration(
    similarity_thresholds={
        SimilarityType.DOCUMENT_SEMANTIC: 0.4,
        SimilarityType.SECTION_CONTENT: 0.5,
        SimilarityType.ENTITY_CONTEXT: 0.6
    },
    relevance_threshold=0.15,
    pmi_threshold=0.7,
    max_similarity_edges_per_document=50,
    chunk_size=1000,
    validation_level=GraphValidationLevel.STRICT
)

EXPERIMENTAL_CONFIG = GraphConfiguration(
    similarity_thresholds={
        SimilarityType.DOCUMENT_SEMANTIC: 0.1,
        SimilarityType.SECTION_CONTENT: 0.2,
        SimilarityType.ENTITY_CONTEXT: 0.3
    },
    relevance_threshold=0.01,
    pmi_threshold=0.1,
    max_similarity_edges_per_document=100,
    chunk_size=500,
    validation_level=GraphValidationLevel.COMPREHENSIVE
)


# Funções auxiliares para uso do schema

def create_default_schema() -> GraphSchema:
    """Cria schema com configuração padrão"""
    return GraphSchema(GraphConfiguration())


def create_optimized_schema(num_documents: int) -> GraphSchema:
    """Cria schema otimizado para o tamanho do corpus"""
    schema = GraphSchema()
    schema.config = schema.optimize_config_for_size(num_documents)
    return schema


def load_schema_from_config(config_dict: Dict[str, Any]) -> GraphSchema:
    """Carrega schema a partir de dicionário de configuração"""
    
    # Reconstrói enums
    similarity_thresholds = {}
    for k, v in config_dict.get('similarity_thresholds', {}).items():
        try:
            similarity_type = SimilarityType(k)
            similarity_thresholds[similarity_type] = v
        except ValueError:
            continue
    
    max_edges_per_node = {}
    for k, v in config_dict.get('max_edges_per_node', {}).items():
        try:
            edge_type = EdgeType(k)
            max_edges_per_node[edge_type] = v
        except ValueError:
            continue
    
    # Cria configuração
    config = GraphConfiguration(
        similarity_thresholds=similarity_thresholds,
        relevance_threshold=config_dict.get('relevance_threshold', DEFAULT_RELEVANCE_THRESHOLD),
        pmi_threshold=config_dict.get('pmi_threshold', DEFAULT_PMI_THRESHOLD),
        max_edges_per_node=max_edges_per_node,
        embedding_model=config_dict.get('embedding_model', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'),
        embedding_dimension=config_dict.get('embedding_dimension', 384),
        max_sequence_length=config_dict.get('max_sequence_length', 512),
        chunk_size=config_dict.get('chunk_size', 1000),
        use_cache=config_dict.get('use_cache', True),
        ner_model=config_dict.get('ner_model', 'pt_core_news_sm'),
        min_entity_frequency=config_dict.get('min_entity_frequency', 2),
        min_concept_frequency=config_dict.get('min_concept_frequency', 3),
        max_concepts=config_dict.get('max_concepts', 1000),
        validation_level=GraphValidationLevel(config_dict.get('validation_level', 'comprehensive'))
    )
    
    return GraphSchema(config)