# -*- coding: utf-8 -*-
"""
Módulo de exportação do grafo heterogêneo
Exporta grafo para múltiplos formatos e casos de uso
"""

import json
import csv
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import networkx as nx
import pandas as pd
from loguru import logger

from graph.models.nodes import DocumentNode  # ✅ CORRETO
from graph.models.edges import EdgeType      # ✅ CORRETO
from ..models.graph_schema import GraphStatistics, GraphMetadata


class GraphExporter:
    """Exportador principal do grafo"""
    
    def __init__(self, graph: nx.Graph, output_dir: str = "data/graph/exports"):
        self.graph = graph
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.stats = {
            'files_exported': 0,
            'formats_used': [],
            'export_time': 0.0
        }
        
        logger.info(f"📦 GraphExporter inicializado (output: {self.output_dir})")
    
    def export_all(self, base_filename: str = "grafo") -> Dict[str, str]:
        """
        Exporta grafo em todos os formatos disponíveis
        
        Args:
            base_filename: Nome base dos arquivos
            
        Returns:
            Dicionário {formato: caminho_arquivo}
        """
        
        import time
        start_time = time.time()
        
        logger.info("📦 Iniciando exportação completa...")
        
        exported_files = {}
        
        # 1. Formatos de grafo
        exported_files['graphml'] = self.export_graphml(base_filename)
        exported_files['gexf'] = self.export_gexf(base_filename)
        exported_files['json_graph'] = self.export_json_graph(base_filename)
        exported_files['edgelist'] = self.export_edgelist(base_filename)
        exported_files['adjacency'] = self.export_adjacency_matrix(base_filename)
        
        # 2. Formatos tabulares
        exported_files['nodes_csv'] = self.export_nodes_csv(base_filename)
        exported_files['edges_csv'] = self.export_edges_csv(base_filename)
        
        # 3. Formatos para análise
        exported_files['statistics'] = self.export_statistics(base_filename)
        exported_files['metadata'] = self.export_metadata(base_filename)
        
        # 4. Formatos específicos
        exported_files['cytoscape'] = self.export_cytoscape(base_filename)
        exported_files['d3'] = self.export_d3_json(base_filename)
        
        # 5. Pickle (rápido para recarregar)
        exported_files['pickle'] = self.export_pickle(base_filename)
        
        self.stats['export_time'] = time.time() - start_time
        self.stats['files_exported'] = len(exported_files)
        self.stats['formats_used'] = list(exported_files.keys())
        
        logger.info(f"✅ Exportação completa: {len(exported_files)} arquivos em {self.stats['export_time']:.2f}s")
        
        return exported_files
    
    def export_graphml(self, filename: str) -> str:
        """Exporta para GraphML (Gephi, Cytoscape, yEd)"""
        
        output_path = self.output_dir / f"{filename}.graphml"
        
        try:
            nx.write_graphml(self.graph, output_path)
            logger.info(f"✅ GraphML exportado: {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"❌ Erro ao exportar GraphML: {e}")
            return ""
    
    def export_gexf(self, filename: str) -> str:
        """Exporta para GEXF (Gephi)"""
        
        output_path = self.output_dir / f"{filename}.gexf"
        
        try:
            nx.write_gexf(self.graph, output_path)
            logger.info(f"✅ GEXF exportado: {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"❌ Erro ao exportar GEXF: {e}")
            return ""
    
    def export_json_graph(self, filename: str) -> str:
        """Exporta para JSON (formato node-link)"""
        
        output_path = self.output_dir / f"{filename}_graph.json"
        
        try:
            # Converte para formato node-link
            graph_data = nx.node_link_data(self.graph)
            
            # Serializa datas e outros tipos
            graph_json = json.dumps(
                graph_data, 
                indent=2, 
                ensure_ascii=False,
                default=str
            )
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(graph_json)
            
            logger.info(f"✅ JSON Graph exportado: {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"❌ Erro ao exportar JSON: {e}")
            return ""
    
    def export_edgelist(self, filename: str) -> str:
        """Exporta lista de arestas (formato simples)"""
        
        output_path = self.output_dir / f"{filename}_edges.txt"
        
        try:
            nx.write_edgelist(self.graph, output_path, data=['weight', 'edge_type'])
            logger.info(f"✅ Edgelist exportado: {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"❌ Erro ao exportar edgelist: {e}")
            return ""
    
    def export_adjacency_matrix(self, filename: str) -> str:
        """Exporta matriz de adjacência"""
        
        output_path = self.output_dir / f"{filename}_adjacency.csv"
        
        try:
            # Cria matriz de adjacência
            adjacency_matrix = nx.adjacency_matrix(self.graph)
            node_list = list(self.graph.nodes())
            
            # Converte para DataFrame
            df = pd.DataFrame(
                adjacency_matrix.todense(),
                index=node_list,
                columns=node_list
            )
            
            df.to_csv(output_path)
            logger.info(f"✅ Matriz de adjacência exportada: {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"❌ Erro ao exportar matriz: {e}")
            return ""
    
    def export_nodes_csv(self, filename: str) -> str:
        """Exporta nós para CSV"""
        
        output_path = self.output_dir / f"{filename}_nodes.csv"
        
        try:
            nodes_data = []
            
            for node_id, node_attrs in self.graph.nodes(data=True):
                node_row = {
                    'id': node_id,
                    'node_type': node_attrs.get('node_type', ''),
                    'label': node_attrs.get('label', '')
                }
                
                # Adiciona atributos específicos por tipo
                if node_attrs.get('node_type') == 'document':
                    node_row['numero_processo'] = node_attrs.get('numero_processo', '')
                    node_row['relator'] = node_attrs.get('relator', '')
                    node_row['orgao_julgador'] = node_attrs.get('orgao_julgador', '')
                
                elif node_attrs.get('node_type') == 'entity':
                    node_row['entity_type'] = node_attrs.get('entity_type', '')
                    node_row['nome_normalizado'] = node_attrs.get('nome_normalizado', '')
                
                elif node_attrs.get('node_type') == 'concept':
                    node_row['termo_conceito'] = node_attrs.get('termo_conceito', '')
                    node_row['categoria'] = node_attrs.get('categoria_juridica', '')
                
                nodes_data.append(node_row)
            
            df = pd.DataFrame(nodes_data)
            df.to_csv(output_path, index=False, encoding='utf-8')
            
            logger.info(f"✅ CSV de nós exportado: {output_path} ({len(nodes_data)} nós)")
            return str(output_path)
        except Exception as e:
            logger.error(f"❌ Erro ao exportar nós CSV: {e}")
            return ""
    
    def export_edges_csv(self, filename: str) -> str:
        """Exporta arestas para CSV"""
        
        output_path = self.output_dir / f"{filename}_edges.csv"
        
        try:
            edges_data = []
            
            for source, target, edge_attrs in self.graph.edges(data=True):
                edge_row = {
                    'source': source,
                    'target': target,
                    'edge_type': edge_attrs.get('edge_type', ''),
                    'weight': edge_attrs.get('weight', 0.0)
                }
                
                # Adiciona métricas específicas por tipo
                if edge_attrs.get('edge_type') == 'similarity':
                    edge_row['cosine_score'] = edge_attrs.get('cosine_score', 0.0)
                    edge_row['similarity_type'] = edge_attrs.get('similarity_type', '')
                
                elif edge_attrs.get('edge_type') == 'relevance':
                    edge_row['tfidf_score'] = edge_attrs.get('tfidf_score', 0.0)
                
                elif edge_attrs.get('edge_type') == 'cooccurrence':
                    edge_row['pmi_score'] = edge_attrs.get('pmi_score', 0.0)
                
                edges_data.append(edge_row)
            
            df = pd.DataFrame(edges_data)
            df.to_csv(output_path, index=False, encoding='utf-8')
            
            logger.info(f"✅ CSV de arestas exportado: {output_path} ({len(edges_data)} arestas)")
            return str(output_path)
        except Exception as e:
            logger.error(f"❌ Erro ao exportar arestas CSV: {e}")
            return ""
    
    def export_statistics(self, filename: str) -> str:
        """Exporta estatísticas do grafo"""
        
        output_path = self.output_dir / f"{filename}_statistics.json"
        
        try:
            # Calcula estatísticas completas
            stats = GraphStatistics()
            stats.update_from_networkx(self.graph)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(stats.to_dict(), f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"✅ Estatísticas exportadas: {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"❌ Erro ao exportar estatísticas: {e}")
            return ""
    
    def export_metadata(self, filename: str) -> str:
        """Exporta metadados do grafo"""
        
        output_path = self.output_dir / f"{filename}_metadata.json"
        
        try:
            metadata = GraphMetadata(
                created_at=datetime.now(),
                corpus_source="TJPI - Empréstimos Consignados",
                corpus_size=self.graph.number_of_nodes()
            )
            
            # Adiciona estatísticas ao metadata
            stats = GraphStatistics()
            stats.update_from_networkx(self.graph)
            metadata.statistics = stats.to_dict()
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"✅ Metadados exportados: {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"❌ Erro ao exportar metadados: {e}")
            return ""
    
    def export_cytoscape(self, filename: str) -> str:
        """Exporta para formato Cytoscape.js"""
        
        output_path = self.output_dir / f"{filename}_cytoscape.json"
        
        try:
            # Formato Cytoscape.js
            cytoscape_data = {
                'elements': {
                    'nodes': [],
                    'edges': []
                }
            }
            
            # Nós
            for node_id, node_attrs in self.graph.nodes(data=True):
                node_element = {
                    'data': {
                        'id': node_id,
                        'label': node_attrs.get('label', ''),
                        'type': node_attrs.get('node_type', ''),
                        **{k: v for k, v in node_attrs.items() 
                           if k not in ['label', 'node_type'] and isinstance(v, (str, int, float, bool))}
                    }
                }
                cytoscape_data['elements']['nodes'].append(node_element)
            
            # Arestas
            for source, target, edge_attrs in self.graph.edges(data=True):
                edge_element = {
                    'data': {
                        'source': source,
                        'target': target,
                        'type': edge_attrs.get('edge_type', ''),
                        'weight': edge_attrs.get('weight', 1.0),
                        **{k: v for k, v in edge_attrs.items() 
                           if k not in ['source', 'target', 'edge_type', 'weight'] 
                           and isinstance(v, (str, int, float, bool))}
                    }
                }
                cytoscape_data['elements']['edges'].append(edge_element)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(cytoscape_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Formato Cytoscape exportado: {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"❌ Erro ao exportar Cytoscape: {e}")
            return ""
    
    def export_d3_json(self, filename: str) -> str:
        """Exporta para formato D3.js (força-dirigida)"""
        
        output_path = self.output_dir / f"{filename}_d3.json"
        
        try:
            # Formato D3.js
            d3_data = {
                'nodes': [],
                'links': []
            }
            
            # Cria mapeamento node_id -> índice
            node_to_index = {node: idx for idx, node in enumerate(self.graph.nodes())}
            
            # Nós
            for node_id, node_attrs in self.graph.nodes(data=True):
                node_element = {
                    'id': node_id,
                    'label': node_attrs.get('label', ''),
                    'type': node_attrs.get('node_type', ''),
                    'group': self._get_node_group(node_attrs)
                }
                d3_data['nodes'].append(node_element)
            
            # Links (arestas)
            for source, target, edge_attrs in self.graph.edges(data=True):
                link_element = {
                    'source': node_to_index[source],
                    'target': node_to_index[target],
                    'value': edge_attrs.get('weight', 1.0),
                    'type': edge_attrs.get('edge_type', '')
                }
                d3_data['links'].append(link_element)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(d3_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Formato D3.js exportado: {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"❌ Erro ao exportar D3: {e}")
            return ""
    
    def export_pickle(self, filename: str) -> str:
        """Exporta grafo em Pickle (rápido para reload)"""
        
        output_path = self.output_dir / f"{filename}.pkl"
        
        try:
            nx.write_gpickle(self.graph, output_path)
            logger.info(f"✅ Pickle exportado: {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"❌ Erro ao exportar Pickle: {e}")
            return ""
    
    def export_subgraph(self, node_ids: List[str], 
                       filename: str,
                       formats: List[str] = ['graphml', 'json']) -> Dict[str, str]:
        """
        Exporta subgrafo específico
        
        Args:
            node_ids: Lista de IDs dos nós
            filename: Nome do arquivo
            formats: Formatos desejados
            
        Returns:
            Dicionário {formato: caminho}
        """
        
        try:
            # Cria subgrafo
            subgraph = self.graph.subgraph(node_ids).copy()
            
            logger.info(f"📊 Exportando subgrafo: {subgraph.number_of_nodes()} nós, {subgraph.number_of_edges()} arestas")
            
            # Cria exporter temporário para o subgrafo
            temp_exporter = GraphExporter(subgraph, str(self.output_dir))
            
            exported = {}
            for fmt in formats:
                if fmt == 'graphml':
                    exported['graphml'] = temp_exporter.export_graphml(f"{filename}_subgraph")
                elif fmt == 'json':
                    exported['json'] = temp_exporter.export_json_graph(f"{filename}_subgraph")
                elif fmt == 'csv':
                    exported['nodes_csv'] = temp_exporter.export_nodes_csv(f"{filename}_subgraph")
                    exported['edges_csv'] = temp_exporter.export_edges_csv(f"{filename}_subgraph")
            
            return exported
            
        except Exception as e:
            logger.error(f"❌ Erro ao exportar subgrafo: {e}")
            return {}
    
    def export_by_node_type(self, node_type: str, filename: str) -> str:
        """Exporta CSV filtrado por tipo de nó"""
        
        output_path = self.output_dir / f"{filename}_{node_type}_nodes.csv"
        
        try:
            nodes_data = []
            
            for node_id, node_attrs in self.graph.nodes(data=True):
                if node_attrs.get('node_type') == node_type:
                    nodes_data.append({
                        'id': node_id,
                        **{k: v for k, v in node_attrs.items() 
                           if isinstance(v, (str, int, float, bool))}
                    })
            
            if nodes_data:
                df = pd.DataFrame(nodes_data)
                df.to_csv(output_path, index=False, encoding='utf-8')
                logger.info(f"✅ CSV de {node_type} exportado: {len(nodes_data)} nós")
                return str(output_path)
            else:
                logger.warning(f"⚠️ Nenhum nó do tipo {node_type} encontrado")
                return ""
                
        except Exception as e:
            logger.error(f"❌ Erro ao exportar por tipo: {e}")
            return ""
    
    def _get_node_group(self, node_attrs: Dict) -> int:
        """Retorna grupo numérico para visualização (baseado no tipo)"""
        
        node_type = node_attrs.get('node_type', '')
        
        type_to_group = {
            'document': 1,
            'section': 2,
            'entity': 3,
            'concept': 4
        }
        
        return type_to_group.get(node_type, 0)
    
    def get_export_summary(self) -> Dict[str, Any]:
        """Retorna resumo das exportações"""
        
        return {
            'output_directory': str(self.output_dir),
            'files_exported': self.stats['files_exported'],
            'formats_used': self.stats['formats_used'],
            'export_time': self.stats['export_time'],
            'graph_info': {
                'nodes': self.graph.number_of_nodes(),
                'edges': self.graph.number_of_edges()
            }
        }


# Funções auxiliares para uso direto

def export_graph_all_formats(graph: nx.Graph, 
                             base_filename: str = "grafo",
                             output_dir: str = "data/graph/exports") -> Dict[str, str]:
    """
    Exporta grafo em todos os formatos
    
    Args:
        graph: Grafo NetworkX
        base_filename: Nome base
        output_dir: Diretório de saída
        
    Returns:
        Dicionário {formato: caminho}
    """
    exporter = GraphExporter(graph, output_dir)
    return exporter.export_all(base_filename)


def export_for_gephi(graph: nx.Graph, filename: str = "grafo_gephi") -> str:
    """Exporta otimizado para Gephi"""
    exporter = GraphExporter(graph)
    return exporter.export_gexf(filename)


def export_for_cytoscape(graph: nx.Graph, filename: str = "grafo_cytoscape") -> str:
    """Exporta otimizado para Cytoscape"""
    exporter = GraphExporter(graph)
    return exporter.export_cytoscape(filename)


def export_for_web(graph: nx.Graph, filename: str = "grafo_web") -> Dict[str, str]:
    """Exporta formatos para visualização web"""
    exporter = GraphExporter(graph)
    return {
        'd3': exporter.export_d3_json(filename),
        'cytoscape': exporter.export_cytoscape(filename),
        'json': exporter.export_json_graph(filename)
    }


def export_similarity_network(graph: nx.Graph, 
                             min_similarity: float = 0.5,
                             filename: str = "similarity_network") -> str:
    """
    Exporta apenas rede de similaridade
    
    Args:
        graph: Grafo completo
        min_similarity: Threshold mínimo
        filename: Nome do arquivo
        
    Returns:
        Caminho do arquivo exportado
    """
    
    # Filtra apenas arestas de similaridade
    edges_to_keep = [
        (u, v) for u, v, attrs in graph.edges(data=True)
        if attrs.get('edge_type') == 'similarity' 
        and attrs.get('weight', 0) >= min_similarity
    ]
    
    # Cria subgrafo
    subgraph = graph.edge_subgraph(edges_to_keep).copy()
    
    # Exporta
    exporter = GraphExporter(subgraph)
    return exporter.export_graphml(filename)