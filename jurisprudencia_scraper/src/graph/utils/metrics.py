# -*- coding: utf-8 -*-
"""
Módulo de métricas para análise do grafo heterogêneo
Calcula métricas de centralidade, comunidades e qualidade
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import networkx as nx
from loguru import logger


@dataclass
class CentralityMetrics:
    """Métricas de centralidade de um nó"""
    node_id: str
    degree_centrality: float
    betweenness_centrality: float
    closeness_centrality: float
    eigenvector_centrality: float
    pagerank: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'node_id': self.node_id,
            'degree': self.degree_centrality,
            'betweenness': self.betweenness_centrality,
            'closeness': self.closeness_centrality,
            'eigenvector': self.eigenvector_centrality,
            'pagerank': self.pagerank
        }


@dataclass
class CommunityInfo:
    """Informações sobre uma comunidade"""
    community_id: int
    size: int
    nodes: List[str]
    internal_edges: int
    external_edges: int
    modularity_contribution: float
    density: float
    avg_degree: float
    main_node_types: Dict[str, int] = field(default_factory=dict)
    top_nodes: List[str] = field(default_factory=list)


@dataclass
class GraphQualityMetrics:
    """Métricas de qualidade do grafo"""
    # Conectividade
    is_connected: bool
    num_components: int
    largest_component_size: int
    largest_component_ratio: float
    
    # Densidade e estrutura
    density: float
    average_clustering: float
    transitivity: float
    
    # Distribuições
    average_degree: float
    degree_assortativity: float
    
    # Caminhos
    average_shortest_path: Optional[float]
    diameter: Optional[int]
    
    # Modularidade
    modularity: Optional[float]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'connectivity': {
                'is_connected': self.is_connected,
                'num_components': self.num_components,
                'largest_component_size': self.largest_component_size,
                'largest_component_ratio': self.largest_component_ratio
            },
            'structure': {
                'density': self.density,
                'average_clustering': self.average_clustering,
                'transitivity': self.transitivity
            },
            'degree': {
                'average_degree': self.average_degree,
                'assortativity': self.degree_assortativity
            },
            'paths': {
                'average_shortest_path': self.average_shortest_path,
                'diameter': self.diameter
            },
            'modularity': self.modularity
        }


class GraphMetricsCalculator:
    """Calculador de métricas do grafo"""
    
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.metrics_cache = {}
        
        logger.info(f"📊 GraphMetricsCalculator inicializado ({graph.number_of_nodes()} nós)")
    
    def calculate_all_metrics(self) -> Dict[str, Any]:
        """
        Calcula todas as métricas disponíveis
        
        Returns:
            Dicionário com todas as métricas
        """
        
        logger.info("📊 Calculando todas as métricas...")
        
        metrics = {
            'quality': self.calculate_quality_metrics(),
            'centrality': self.calculate_centrality_metrics(),
            'communities': self.detect_communities(),
            'node_types': self.analyze_node_types(),
            'edge_types': self.analyze_edge_types(),
            'similarity_distribution': self.analyze_similarity_distribution(),
            'hub_analysis': self.identify_hubs()
        }
        
        logger.info("✅ Todas as métricas calculadas")
        
        return metrics
    
    def calculate_quality_metrics(self) -> GraphQualityMetrics:
        """Calcula métricas de qualidade do grafo"""
        
        logger.info("📈 Calculando métricas de qualidade...")
        
        # Conectividade
        is_connected = nx.is_connected(self.graph)
        components = list(nx.connected_components(self.graph))
        num_components = len(components)
        largest_component = max(components, key=len) if components else set()
        largest_component_size = len(largest_component)
        largest_component_ratio = largest_component_size / self.graph.number_of_nodes()
        
        # Densidade e clustering
        density = nx.density(self.graph)
        
        try:
            average_clustering = nx.average_clustering(self.graph)
        except:
            average_clustering = 0.0
        
        transitivity = nx.transitivity(self.graph)
        
        # Grau
        degrees = [d for n, d in self.graph.degree()]
        average_degree = np.mean(degrees) if degrees else 0.0
        
        try:
            degree_assortativity = nx.degree_assortativity_coefficient(self.graph)
        except:
            degree_assortativity = 0.0
        
        # Caminhos (apenas para componente principal se grafo não conectado)
        average_shortest_path = None
        diameter = None
        
        if is_connected:
            try:
                average_shortest_path = nx.average_shortest_path_length(self.graph)
                diameter = nx.diameter(self.graph)
            except:
                pass
        else:
            # Calcula para maior componente
            largest_subgraph = self.graph.subgraph(largest_component)
            try:
                average_shortest_path = nx.average_shortest_path_length(largest_subgraph)
                diameter = nx.diameter(largest_subgraph)
            except:
                pass
        
        # Modularidade (se comunidades já calculadas)
        modularity = None
        
        metrics = GraphQualityMetrics(
            is_connected=is_connected,
            num_components=num_components,
            largest_component_size=largest_component_size,
            largest_component_ratio=largest_component_ratio,
            density=density,
            average_clustering=average_clustering,
            transitivity=transitivity,
            average_degree=average_degree,
            degree_assortativity=degree_assortativity,
            average_shortest_path=average_shortest_path,
            diameter=diameter,
            modularity=modularity
        )
        
        logger.info("✅ Métricas de qualidade calculadas")
        
        return metrics
    
    def calculate_centrality_metrics(self, top_k: int = 100) -> List[CentralityMetrics]:
        """
        Calcula métricas de centralidade para todos os nós
        
        Args:
            top_k: Retorna apenas top-k nós por centralidade
            
        Returns:
            Lista de CentralityMetrics
        """
        
        logger.info("🎯 Calculando métricas de centralidade...")
        
        # Calcula todas as centralidades
        degree_cent = nx.degree_centrality(self.graph)
        
        try:
            betweenness_cent = nx.betweenness_centrality(self.graph, k=min(100, self.graph.number_of_nodes()))
        except:
            betweenness_cent = {node: 0.0 for node in self.graph.nodes()}
        
        try:
            closeness_cent = nx.closeness_centrality(self.graph)
        except:
            closeness_cent = {node: 0.0 for node in self.graph.nodes()}
        
        try:
            eigenvector_cent = nx.eigenvector_centrality(self.graph, max_iter=100)
        except:
            eigenvector_cent = {node: 0.0 for node in self.graph.nodes()}
        
        try:
            pagerank = nx.pagerank(self.graph)
        except:
            pagerank = {node: 1.0/self.graph.number_of_nodes() for node in self.graph.nodes()}
        
        # Combina em objetos CentralityMetrics
        centrality_list = []
        
        for node in self.graph.nodes():
            centrality = CentralityMetrics(
                node_id=node,
                degree_centrality=degree_cent.get(node, 0.0),
                betweenness_centrality=betweenness_cent.get(node, 0.0),
                closeness_centrality=closeness_cent.get(node, 0.0),
                eigenvector_centrality=eigenvector_cent.get(node, 0.0),
                pagerank=pagerank.get(node, 0.0)
            )
            centrality_list.append(centrality)
        
        # Ordena por PageRank (métrica agregada)
        centrality_list.sort(key=lambda x: x.pagerank, reverse=True)
        
        logger.info(f"✅ Centralidades calculadas para {len(centrality_list)} nós")
        
        return centrality_list[:top_k] if top_k else centrality_list
    
    def detect_communities(self, method: str = 'louvain') -> Dict[str, Any]:
        """
        Detecta comunidades no grafo
        
        Args:
            method: Método de detecção ('louvain', 'label_propagation', 'girvan_newman')
            
        Returns:
            Informações sobre comunidades
        """
        
        logger.info(f"🏘️ Detectando comunidades (método: {method})...")
        
        try:
            if method == 'louvain':
                communities = self._detect_louvain()
            elif method == 'label_propagation':
                communities = self._detect_label_propagation()
            elif method == 'girvan_newman':
                communities = self._detect_girvan_newman()
            else:
                logger.warning(f"Método desconhecido: {method}, usando louvain")
                communities = self._detect_louvain()
            
            # Calcula modularidade
            modularity = nx.algorithms.community.modularity(self.graph, communities)
            
            # Analisa cada comunidade
            community_infos = []
            
            for idx, community_nodes in enumerate(communities):
                info = self._analyze_community(idx, community_nodes)
                community_infos.append(info)
            
            result = {
                'method': method,
                'num_communities': len(communities),
                'modularity': modularity,
                'communities': [self._community_info_to_dict(c) for c in community_infos],
                'size_distribution': [c.size for c in community_infos],
                'largest_community': max(community_infos, key=lambda x: x.size).community_id
            }
            
            logger.info(f"✅ {len(communities)} comunidades detectadas (modularity: {modularity:.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Erro na detecção de comunidades: {e}")
            return {'error': str(e)}
    
    def _detect_louvain(self) -> List[Set]:
        """Detecta comunidades usando Louvain"""
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(self.graph)
            
            # Agrupa nós por comunidade
            communities = defaultdict(set)
            for node, comm_id in partition.items():
                communities[comm_id].add(node)
            
            return list(communities.values())
        except ImportError:
            logger.warning("python-louvain não instalado, usando label propagation")
            return self._detect_label_propagation()
    
    def _detect_label_propagation(self) -> List[Set]:
        """Detecta comunidades usando Label Propagation"""
        communities = nx.algorithms.community.label_propagation_communities(self.graph)
        return list(communities)
    
    def _detect_girvan_newman(self, k: int = 10) -> List[Set]:
        """Detecta comunidades usando Girvan-Newman"""
        comp = nx.algorithms.community.girvan_newman(self.graph)
        
        # Pega k comunidades
        for communities in comp:
            if len(communities) >= k:
                return list(communities)
        
        return list(communities)
    
    def _analyze_community(self, community_id: int, nodes: Set[str]) -> CommunityInfo:
        """Analisa uma comunidade específica"""
        
        subgraph = self.graph.subgraph(nodes)
        
        # Conta tipos de nós
        node_types = Counter()
        for node in nodes:
            node_type = self.graph.nodes[node].get('node_type', 'unknown')
            node_types[node_type] += 1
        
        # Conta arestas internas e externas
        internal_edges = subgraph.number_of_edges()
        external_edges = 0
        
        for node in nodes:
            for neighbor in self.graph.neighbors(node):
                if neighbor not in nodes:
                    external_edges += 1
        
        # Calcula métricas
        density = nx.density(subgraph) if subgraph.number_of_nodes() > 1 else 0.0
        avg_degree = np.mean([d for n, d in subgraph.degree()]) if subgraph.number_of_nodes() > 0 else 0.0
        
        # Top nós por grau
        degree_sorted = sorted(subgraph.degree(), key=lambda x: x[1], reverse=True)
        top_nodes = [node for node, deg in degree_sorted[:5]]
        
        return CommunityInfo(
            community_id=community_id,
            size=len(nodes),
            nodes=list(nodes),
            internal_edges=internal_edges,
            external_edges=external_edges,
            modularity_contribution=0.0,  # Calculado depois
            density=density,
            avg_degree=avg_degree,
            main_node_types=dict(node_types),
            top_nodes=top_nodes
        )
    
    def _community_info_to_dict(self, info: CommunityInfo) -> Dict[str, Any]:
        """Converte CommunityInfo para dict (sem lista completa de nós)"""
        return {
            'community_id': info.community_id,
            'size': info.size,
            'internal_edges': info.internal_edges,
            'external_edges': info.external_edges,
            'density': info.density,
            'avg_degree': info.avg_degree,
            'main_node_types': info.main_node_types,
            'top_nodes': info.top_nodes
        }
    
    def analyze_node_types(self) -> Dict[str, Any]:
        """Analisa distribuição e conectividade por tipo de nó"""
        
        logger.info("🔍 Analisando tipos de nós...")
        
        node_types = defaultdict(list)
        
        for node, attrs in self.graph.nodes(data=True):
            node_type = attrs.get('node_type', 'unknown')
            node_types[node_type].append(node)
        
        analysis = {}
        
        for node_type, nodes in node_types.items():
            subgraph = self.graph.subgraph(nodes)
            
            degrees = [d for n, d in subgraph.degree()]
            
            analysis[node_type] = {
                'count': len(nodes),
                'percentage': len(nodes) / self.graph.number_of_nodes() * 100,
                'avg_degree': np.mean(degrees) if degrees else 0.0,
                'max_degree': max(degrees) if degrees else 0,
                'density': nx.density(subgraph) if len(nodes) > 1 else 0.0
            }
        
        logger.info(f"✅ {len(node_types)} tipos de nós analisados")
        
        return analysis
    
    def analyze_edge_types(self) -> Dict[str, Any]:
        """Analisa distribuição e pesos por tipo de aresta"""
        
        logger.info("🔗 Analisando tipos de arestas...")
        
        edge_types = defaultdict(list)
        
        for u, v, attrs in self.graph.edges(data=True):
            edge_type = attrs.get('edge_type', 'unknown')
            weight = attrs.get('weight', 1.0)
            edge_types[edge_type].append(weight)
        
        analysis = {}
        
        for edge_type, weights in edge_types.items():
            analysis[edge_type] = {
                'count': len(weights),
                'percentage': len(weights) / self.graph.number_of_edges() * 100,
                'avg_weight': np.mean(weights),
                'min_weight': min(weights),
                'max_weight': max(weights),
                'std_weight': np.std(weights)
            }
        
        logger.info(f"✅ {len(edge_types)} tipos de arestas analisados")
        
        return analysis
    
    def analyze_similarity_distribution(self) -> Dict[str, Any]:
        """Analisa distribuição de similaridades"""
        
        logger.info("📊 Analisando distribuição de similaridades...")
        
        similarities = []
        
        for u, v, attrs in self.graph.edges(data=True):
            if attrs.get('edge_type') == 'similarity':
                weight = attrs.get('weight', 0.0)
                similarities.append(weight)
        
        if not similarities:
            return {'error': 'Nenhuma aresta de similaridade encontrada'}
        
        # Estatísticas
        similarities = np.array(similarities)
        
        analysis = {
            'count': len(similarities),
            'mean': float(np.mean(similarities)),
            'median': float(np.median(similarities)),
            'std': float(np.std(similarities)),
            'min': float(np.min(similarities)),
            'max': float(np.max(similarities)),
            'quartiles': {
                'q25': float(np.percentile(similarities, 25)),
                'q50': float(np.percentile(similarities, 50)),
                'q75': float(np.percentile(similarities, 75))
            },
            'histogram': self._create_histogram(similarities, bins=10)
        }
        
        logger.info("✅ Distribuição de similaridades analisada")
        
        return analysis
    
    def identify_hubs(self, top_k: int = 10) -> Dict[str, Any]:
        """
        Identifica hubs (nós altamente conectados)
        
        Args:
            top_k: Número de top hubs
            
        Returns:
            Informações sobre hubs
        """
        
        logger.info(f"🌟 Identificando top {top_k} hubs...")
        
        # Ordena nós por grau
        degree_sorted = sorted(self.graph.degree(), key=lambda x: x[1], reverse=True)
        
        hubs = []
        
        for node, degree in degree_sorted[:top_k]:
            node_attrs = self.graph.nodes[node]
            
            # Analisa vizinhos
            neighbors = list(self.graph.neighbors(node))
            neighbor_types = Counter()
            
            for neighbor in neighbors:
                neighbor_type = self.graph.nodes[neighbor].get('node_type', 'unknown')
                neighbor_types[neighbor_type] += 1
            
            hub_info = {
                'node_id': node,
                'label': node_attrs.get('label', ''),
                'node_type': node_attrs.get('node_type', ''),
                'degree': degree,
                'neighbor_types': dict(neighbor_types),
                'clustering_coefficient': nx.clustering(self.graph, node)
            }
            
            hubs.append(hub_info)
        
        logger.info(f"✅ {len(hubs)} hubs identificados")
        
        return {
            'top_hubs': hubs,
            'avg_hub_degree': np.mean([h['degree'] for h in hubs]),
            'hub_threshold': hubs[-1]['degree'] if hubs else 0
        }
    
    def compare_node_types_connectivity(self) -> pd.DataFrame:
        """Cria matriz de conectividade entre tipos de nós"""
        
        logger.info("🔗 Analisando conectividade entre tipos de nós...")
        
        # Coleta tipos de nós
        node_types_set = set()
        for node, attrs in self.graph.nodes(data=True):
            node_types_set.add(attrs.get('node_type', 'unknown'))
        
        node_types = sorted(list(node_types_set))
        
        # Cria matriz de conectividade
        connectivity_matrix = np.zeros((len(node_types), len(node_types)))
        
        type_to_idx = {t: i for i, t in enumerate(node_types)}
        
        for u, v in self.graph.edges():
            type_u = self.graph.nodes[u].get('node_type', 'unknown')
            type_v = self.graph.nodes[v].get('node_type', 'unknown')
            
            idx_u = type_to_idx[type_u]
            idx_v = type_to_idx[type_v]
            
            connectivity_matrix[idx_u, idx_v] += 1
            connectivity_matrix[idx_v, idx_u] += 1  # Simétrica
        
        df = pd.DataFrame(connectivity_matrix, index=node_types, columns=node_types)
        
        logger.info("✅ Matriz de conectividade criada")
        
        return df
    
    def calculate_document_similarities(self, document_ids: List[str] = None,
                                      top_k: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """
        Calcula documentos mais similares
        
        Args:
            document_ids: Lista de IDs de documentos (se None, usa todos)
            top_k: Top-k documentos similares por documento
            
        Returns:
            Dicionário {doc_id: [(similar_doc_id, similarity_score), ...]}
        """
        
        logger.info("📄 Calculando similaridades entre documentos...")
        
        if document_ids is None:
            document_ids = [
                node for node, attrs in self.graph.nodes(data=True)
                if attrs.get('node_type') == 'document'
            ]
        
        similarities = {}
        
        for doc_id in document_ids:
            if doc_id not in self.graph:
                continue
            
            # Pega arestas de similaridade
            similar_docs = []
            
            for neighbor in self.graph.neighbors(doc_id):
                edge_attrs = self.graph.get_edge_data(doc_id, neighbor)
                
                if edge_attrs.get('edge_type') == 'similarity':
                    weight = edge_attrs.get('weight', 0.0)
                    
                    # Verifica se vizinho é documento
                    neighbor_attrs = self.graph.nodes[neighbor]
                    if neighbor_attrs.get('node_type') == 'document':
                        similar_docs.append((neighbor, weight))
            
            # Ordena e limita
            similar_docs.sort(key=lambda x: x[1], reverse=True)
            similarities[doc_id] = similar_docs[:top_k]
        
        logger.info(f"✅ Similaridades calculadas para {len(similarities)} documentos")
        
        return similarities
    
    def _create_histogram(self, data: np.ndarray, bins: int = 10) -> Dict[str, List]:
        """Cria histograma dos dados"""
        
        counts, bin_edges = np.histogram(data, bins=bins)
        
        return {
            'counts': counts.tolist(),
            'bin_edges': bin_edges.tolist()
        }
    
    def export_metrics_report(self, output_path: str):
        """Exporta relatório completo de métricas"""
        
        logger.info(f"📊 Gerando relatório de métricas: {output_path}")
        
        metrics = self.calculate_all_metrics()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"✅ Relatório exportado: {output_path}")


# Funções auxiliares

def calculate_basic_metrics(graph: nx.Graph) -> Dict[str, Any]:
    """Calcula métricas básicas rapidamente"""
    
    return {
        'num_nodes': graph.number_of_nodes(),
        'num_edges': graph.number_of_edges(),
        'density': nx.density(graph),
        'is_connected': nx.is_connected(graph),
        'avg_degree': np.mean([d for n, d in graph.degree()])
    }


def find_most_central_nodes(graph: nx.Graph, top_k: int = 10) -> List[Tuple[str, float]]:
    """Encontra nós mais centrais (PageRank)"""
    
    pagerank = nx.pagerank(graph)
    sorted_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_nodes[:top_k]


def calculate_graph_diameter(graph: nx.Graph) -> Optional[int]:
    """Calcula diâmetro do grafo"""
    
    if not nx.is_connected(graph):
        # Usa maior componente
        largest_cc = max(nx.connected_components(graph), key=len)
        graph = graph.subgraph(largest_cc)
    
    try:
        return nx.diameter(graph)
    except:
        return None