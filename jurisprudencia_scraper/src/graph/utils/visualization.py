# -*- coding: utf-8 -*-
"""
Módulo de visualização do grafo heterogêneo
Cria visualizações estáticas e interativas
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple , Set
from pathlib import Path
import networkx as nx
from loguru import logger

# Visualizações interativas (opcionais)
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("⚠️ Plotly não disponível - visualizações interativas desabilitadas")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    logger.warning("⚠️ Seaborn não disponível - alguns gráficos desabilitados")


class GraphVisualizer:
    """Visualizador principal do grafo"""
    
    def __init__(self, graph: nx.Graph, output_dir: str = "data/graph/visualizations"):
        self.graph = graph
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configurações de estilo
        self.node_colors = {
            'document': '#4A90E2',     # Azul
            'section': '#7ED321',      # Verde
            'entity': '#F5A623',       # Laranja
            'concept': '#D0021B',      # Vermelho
            'unknown': '#9B9B9B'       # Cinza
        }
        
        self.edge_colors = {
            'similarity': '#4A90E2',
            'relevance': '#F5A623',
            'cooccurrence': '#D0021B',
            'hierarchical': '#50E3C2'
        }
        
        # Configurações matplotlib
        plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
        
        logger.info(f"🎨 GraphVisualizer inicializado (output: {self.output_dir})")
    
    def create_all_visualizations(self, prefix: str = "grafo") -> Dict[str, str]:
        """
        Cria todas as visualizações disponíveis
        
        Args:
            prefix: Prefixo dos arquivos
            
        Returns:
            Dicionário {tipo: caminho_arquivo}
        """
        
        logger.info("🎨 Criando todas as visualizações...")
        
        visualizations = {}
        
        # Visualizações estáticas (matplotlib)
        visualizations['network'] = self.plot_network_overview(f"{prefix}_network")
        visualizations['degree_dist'] = self.plot_degree_distribution(f"{prefix}_degree_dist")
        visualizations['node_types'] = self.plot_node_types_distribution(f"{prefix}_node_types")
        visualizations['edge_types'] = self.plot_edge_types_distribution(f"{prefix}_edge_types")
        visualizations['similarity_heatmap'] = self.plot_similarity_heatmap(f"{prefix}_similarity")
        visualizations['centrality'] = self.plot_centrality_comparison(f"{prefix}_centrality")
        
        # Visualizações interativas (plotly)
        if PLOTLY_AVAILABLE:
            visualizations['interactive_network'] = self.create_interactive_network(f"{prefix}_interactive")
            visualizations['3d_network'] = self.create_3d_network(f"{prefix}_3d")
        
        logger.info(f"✅ {len(visualizations)} visualizações criadas")
        
        return visualizations
    
    def plot_network_overview(self, filename: str, 
                              layout: str = 'spring',
                              sample_size: Optional[int] = 500) -> str:
        """
        Plota visão geral da rede
        
        Args:
            filename: Nome do arquivo
            layout: Algoritmo de layout ('spring', 'kamada_kawai', 'circular')
            sample_size: Tamanho da amostra (None = todos os nós)
            
        Returns:
            Caminho do arquivo salvo
        """
        
        logger.info(f"📊 Plotando visão geral da rede (layout: {layout})...")
        
        output_path = self.output_dir / f"{filename}.png"
        
        # Amostra se grafo muito grande
        if sample_size and self.graph.number_of_nodes() > sample_size:
            nodes_sample = np.random.choice(
                list(self.graph.nodes()), 
                size=sample_size, 
                replace=False
            )
            G = self.graph.subgraph(nodes_sample)
            logger.info(f"  Usando amostra de {sample_size} nós")
        else:
            G = self.graph
        
        # Cria figura
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Calcula layout
        if layout == 'spring':
            pos = nx.spring_layout(G, k=0.5, iterations=50)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        else:
            pos = nx.spring_layout(G)
        
        # Cores dos nós por tipo
        node_colors_list = []
        for node in G.nodes():
            node_type = G.nodes[node].get('node_type', 'unknown')
            node_colors_list.append(self.node_colors.get(node_type, '#9B9B9B'))
        
        # Tamanhos dos nós por grau
        node_sizes = [G.degree(node) * 10 for node in G.nodes()]
        
        # Desenha nós
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors_list,
            node_size=node_sizes,
            alpha=0.7,
            ax=ax
        )
        
        # Desenha arestas
        nx.draw_networkx_edges(
            G, pos,
            edge_color='#CCCCCC',
            alpha=0.3,
            width=0.5,
            ax=ax
        )
        
        # Legenda
        legend_elements = [
            mpatches.Patch(color=color, label=node_type.capitalize())
            for node_type, color in self.node_colors.items()
            if node_type in [G.nodes[n].get('node_type') for n in G.nodes()]
        ]
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        ax.set_title(
            f"Grafo Heterogêneo de Jurisprudências\n{G.number_of_nodes()} nós, {G.number_of_edges()} arestas",
            fontsize=16,
            fontweight='bold'
        )
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Rede plotada: {output_path}")
        
        return str(output_path)
    
    def plot_degree_distribution(self, filename: str) -> str:
        """Plota distribuição de graus"""
        
        logger.info("📊 Plotando distribuição de graus...")
        
        output_path = self.output_dir / f"{filename}.png"
        
        degrees = [d for n, d in self.graph.degree()]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histograma
        axes[0].hist(degrees, bins=50, color='#4A90E2', alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Grau', fontsize=12)
        axes[0].set_ylabel('Frequência', fontsize=12)
        axes[0].set_title('Distribuição de Graus', fontsize=14, fontweight='bold')
        axes[0].grid(alpha=0.3)
        
        # Log-log (lei de potência)
        degree_sequence = sorted(degrees, reverse=True)
        degree_count = {}
        for d in degree_sequence:
            degree_count[d] = degree_count.get(d, 0) + 1
        
        deg, cnt = zip(*sorted(degree_count.items()))
        
        axes[1].loglog(deg, cnt, 'o-', color='#4A90E2', alpha=0.7, markersize=6)
        axes[1].set_xlabel('Grau (log)', fontsize=12)
        axes[1].set_ylabel('Frequência (log)', fontsize=12)
        axes[1].set_title('Distribuição de Graus (Log-Log)', fontsize=14, fontweight='bold')
        axes[1].grid(alpha=0.3)
        
        # Estatísticas
        stats_text = f"Média: {np.mean(degrees):.2f}\nMediana: {np.median(degrees):.0f}\nMáx: {max(degrees)}"
        axes[1].text(0.05, 0.95, stats_text, transform=axes[1].transAxes,
                    verticalalignment='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Distribuição de graus plotada: {output_path}")
        
        return str(output_path)
    
    def plot_node_types_distribution(self, filename: str) -> str:
        """Plota distribuição de tipos de nós"""
        
        logger.info("📊 Plotando distribuição de tipos de nós...")
        
        output_path = self.output_dir / f"{filename}.png"
        
        # Conta tipos
        node_types = {}
        for node, attrs in self.graph.nodes(data=True):
            node_type = attrs.get('node_type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        # Ordena
        types = list(node_types.keys())
        counts = [node_types[t] for t in types]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Gráfico de barras
        colors = [self.node_colors.get(t, '#9B9B9B') for t in types]
        bars = ax1.bar(types, counts, color=colors, alpha=0.7, edgecolor='black')
        
        ax1.set_xlabel('Tipo de Nó', fontsize=12)
        ax1.set_ylabel('Quantidade', fontsize=12)
        ax1.set_title('Distribuição por Tipo de Nó', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Adiciona valores nas barras
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}',
                    ha='center', va='bottom', fontsize=10)
        
        # Gráfico de pizza
        ax2.pie(counts, labels=types, colors=colors, autopct='%1.1f%%',
               startangle=90, textprops={'fontsize': 11})
        ax2.set_title('Proporção de Tipos de Nós', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Distribuição de tipos plotada: {output_path}")
        
        return str(output_path)
    
    def plot_edge_types_distribution(self, filename: str) -> str:
        """Plota distribuição de tipos de arestas"""
        
        logger.info("📊 Plotando distribuição de tipos de arestas...")
        
        output_path = self.output_dir / f"{filename}.png"
        
        # Conta tipos e coleta pesos
        edge_types_count = {}
        edge_types_weights = {}
        
        for u, v, attrs in self.graph.edges(data=True):
            edge_type = attrs.get('edge_type', 'unknown')
            weight = attrs.get('weight', 1.0)
            
            edge_types_count[edge_type] = edge_types_count.get(edge_type, 0) + 1
            
            if edge_type not in edge_types_weights:
                edge_types_weights[edge_type] = []
            edge_types_weights[edge_type].append(weight)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Contagem de arestas por tipo
        types = list(edge_types_count.keys())
        counts = [edge_types_count[t] for t in types]
        colors = [self.edge_colors.get(t, '#9B9B9B') for t in types]
        
        axes[0, 0].bar(types, counts, color=colors, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Quantidade por Tipo de Aresta', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Quantidade', fontsize=11)
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # 2. Proporção
        axes[0, 1].pie(counts, labels=types, colors=colors, autopct='%1.1f%%',
                      startangle=90, textprops={'fontsize': 10})
        axes[0, 1].set_title('Proporção de Tipos de Arestas', fontsize=12, fontweight='bold')
        
        # 3. Distribuição de pesos por tipo
        bp = axes[1, 0].boxplot(
            [edge_types_weights[t] for t in types],
            labels=types,
            patch_artist=True,
            medianprops=dict(color='red', linewidth=2)
        )
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        axes[1, 0].set_title('Distribuição de Pesos por Tipo', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Peso', fontsize=11)
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # 4. Estatísticas
        stats_text = "Estatísticas de Pesos:\n\n"
        for edge_type in types:
            weights = edge_types_weights[edge_type]
            stats_text += f"{edge_type}:\n"
            stats_text += f"  Média: {np.mean(weights):.3f}\n"
            stats_text += f"  Mediana: {np.median(weights):.3f}\n\n"
        
        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                       verticalalignment='top', fontsize=9, family='monospace',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Distribuição de arestas plotada: {output_path}")
        
        return str(output_path)
    
    def plot_similarity_heatmap(self, filename: str, sample_size: int = 50) -> str:
        """Plota heatmap de similaridades entre documentos"""
        
        logger.info("🔥 Plotando heatmap de similaridades...")
        
        output_path = self.output_dir / f"{filename}.png"
        
        # Filtra apenas documentos
        doc_nodes = [
            node for node, attrs in self.graph.nodes(data=True)
            if attrs.get('node_type') == 'document'
        ]
        
        if len(doc_nodes) > sample_size:
            doc_nodes = np.random.choice(doc_nodes, size=sample_size, replace=False)
            logger.info(f"  Usando amostra de {sample_size} documentos")
        
        # Cria matriz de similaridade
        n = len(doc_nodes)
        similarity_matrix = np.zeros((n, n))
        
        node_to_idx = {node: i for i, node in enumerate(doc_nodes)}
        
        for u, v, attrs in self.graph.edges(data=True):
            if attrs.get('edge_type') == 'similarity':
                if u in node_to_idx and v in node_to_idx:
                    idx_u = node_to_idx[u]
                    idx_v = node_to_idx[v]
                    weight = attrs.get('weight', 0.0)
                    similarity_matrix[idx_u, idx_v] = weight
                    similarity_matrix[idx_v, idx_u] = weight
        
        # Plota heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        
        im = ax.imshow(similarity_matrix, cmap='RdYlBu_r', aspect='auto', 
                      vmin=0, vmax=1)
        
        ax.set_title(f'Heatmap de Similaridades\n({len(doc_nodes)} documentos)',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Documento', fontsize=12)
        ax.set_ylabel('Documento', fontsize=12)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Similaridade', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Heatmap plotado: {output_path}")
        
        return str(output_path)
    
    def plot_centrality_comparison(self, filename: str, top_k: int = 15) -> str:
        """Plota comparação de métricas de centralidade"""
        
        logger.info(f"📊 Plotando top {top_k} nós por centralidade...")
        
        output_path = self.output_dir / f"{filename}.png"
        
        # Calcula centralidades
        degree_cent = nx.degree_centrality(self.graph)
        
        try:
            pagerank = nx.pagerank(self.graph)
        except:
            pagerank = degree_cent
        
        try:
            betweenness = nx.betweenness_centrality(self.graph, k=min(100, self.graph.number_of_nodes()))
        except:
            betweenness = degree_cent
        
        # Top nós por PageRank
        top_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:top_k]
        nodes = [n for n, _ in top_nodes]
        
        # Prepara dados
        data = {
            'Degree': [degree_cent[n] for n in nodes],
            'PageRank': [pagerank[n] for n in nodes],
            'Betweenness': [betweenness.get(n, 0) for n in nodes]
        }
        
        df = pd.DataFrame(data, index=nodes)
        
        # Normaliza para comparação
        df_norm = (df - df.min()) / (df.max() - df.min())
        
        # Plota
        fig, ax = plt.subplots(figsize=(14, 8))
        
        df_norm.plot(kind='bar', ax=ax, width=0.8, alpha=0.7)
        
        ax.set_title(f'Top {top_k} Nós por Centralidade', fontsize=14, fontweight='bold')
        ax.set_xlabel('Nó', fontsize=12)
        ax.set_ylabel('Centralidade (normalizada)', fontsize=12)
        ax.legend(title='Métrica', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Centralidades plotadas: {output_path}")
        
        return str(output_path)
    
    def create_interactive_network(self, filename: str, sample_size: int = 500) -> str:
        """Cria visualização interativa com Plotly"""
        
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly não disponível - pulando visualização interativa")
            return ""
        
        logger.info("🌐 Criando visualização interativa...")
        
        output_path = self.output_dir / f"{filename}.html"
        
        # Amostra
        if self.graph.number_of_nodes() > sample_size:
            nodes_sample = np.random.choice(
                list(self.graph.nodes()),
                size=sample_size,
                replace=False
            )
            G = self.graph.subgraph(nodes_sample)
        else:
            G = self.graph
        
        # Layout
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        
        # Prepara dados das arestas
        edge_trace = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            edge_trace.append(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=0.5, color='#888'),
                    hoverinfo='none',
                    showlegend=False
                )
            )
        
        # Prepara dados dos nós
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            attrs = G.nodes[node]
            node_type = attrs.get('node_type', 'unknown')
            label = attrs.get('label', node)[:50]
            
            node_text.append(f"<b>{label}</b><br>Tipo: {node_type}<br>Grau: {G.degree(node)}")
            node_color.append(self.node_colors.get(node_type, '#9B9B9B'))
        
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                color=node_color,
                size=10,
                line=dict(width=1, color='white')
            ),
            showlegend=False
        )
        
        # Cria figura
        fig = go.Figure(
            data=edge_trace + [node_trace],
            layout=go.Layout(
                title=f'Grafo Interativo de Jurisprudências<br>{G.number_of_nodes()} nós, {G.number_of_edges()} arestas',
                titlefont=dict(size=16),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=0, l=0, r=0, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='white'
            )
        )
        
        fig.write_html(output_path)
        
        logger.info(f"✅ Visualização interativa criada: {output_path}")
        
        return str(output_path)
    
    def create_3d_network(self, filename: str, sample_size: int = 300) -> str:
        """Cria visualização 3D interativa"""
        
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly não disponível - pulando visualização 3D")
            return ""
        
        logger.info("🎲 Criando visualização 3D...")
        
        output_path = self.output_dir / f"{filename}.html"
        
        # Amostra
        if self.graph.number_of_nodes() > sample_size:
            nodes_sample = np.random.choice(
                list(self.graph.nodes()),
                size=sample_size,
                replace=False
            )
            G = self.graph.subgraph(nodes_sample)
        else:
            G = self.graph
        
        # Layout 3D (spring_layout funciona em qualquer dimensão)
        pos = nx.spring_layout(G, dim=3, k=0.5, iterations=50)
        
        # Arestas
        edge_x = []
        edge_y = []
        edge_z = []
        
        for edge in G.edges():
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]
            
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])
        
        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(color='rgba(125, 125, 125, 0.3)', width=1),
            hoverinfo='none'
        )
        
        # Nós
        node_x = []
        node_y = []
        node_z = []
        node_text = []
        node_color = []
        
        for node in G.nodes():
            x, y, z = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            
            attrs = G.nodes[node]
            node_type = attrs.get('node_type', 'unknown')
            label = attrs.get('label', node)[:50]
            
            node_text.append(f"<b>{label}</b><br>Tipo: {node_type}")
            node_color.append(self.node_colors.get(node_type, '#9B9B9B'))
        
        node_trace = go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                color=node_color,
                size=5,
                line=dict(color='white', width=0.5)
            )
        )
        
        # Figura
        fig = go.Figure(data=[edge_trace, node_trace])
        
        fig.update_layout(
            title=f'Grafo 3D de Jurisprudências<br>{G.number_of_nodes()} nós',
            scene=dict(
                xaxis=dict(showbackground=False, showticklabels=False, title=''),
                yaxis=dict(showbackground=False, showticklabels=False, title=''),
                zaxis=dict(showbackground=False, showticklabels=False, title='')
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(l=0, r=0, b=0, t=40)
        )
        
        fig.write_html(output_path)
        
        logger.info(f"✅ Visualização 3D criada: {output_path}")
        
        return str(output_path)
    
    def plot_community_structure(self, communities: List[Set], filename: str) -> str:
        """Plota estrutura de comunidades"""
        
        logger.info(f"🏘️ Plotando {len(communities)} comunidades...")
        
        output_path = self.output_dir / f"{filename}.png"
        
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Layout
        pos = nx.spring_layout(self.graph, k=0.5, iterations=50)
        
        # Cores para comunidades
        colors = plt.cm.tab20(np.linspace(0, 1, len(communities)))
        
        # Desenha cada comunidade
        for idx, community in enumerate(communities):
            nx.draw_networkx_nodes(
                self.graph,
                pos,
                nodelist=list(community),
                node_color=[colors[idx]],
                node_size=50,
                alpha=0.8,
                ax=ax
            )
        
        # Arestas
        nx.draw_networkx_edges(
            self.graph,
            pos,
            alpha=0.2,
            width=0.5,
            ax=ax
        )
        
        ax.set_title(
            f'Estrutura de Comunidades\n{len(communities)} comunidades detectadas',
            fontsize=16,
            fontweight='bold'
        )
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Comunidades plotadas: {output_path}")
        
        return str(output_path)


# Funções auxiliares

def quick_plot_network(graph: nx.Graph, output_path: str = "network.png"):
    """Plota rede rapidamente"""
    visualizer = GraphVisualizer(graph)
    return visualizer.plot_network_overview("quick_network")


def create_summary_dashboard(graph: nx.Graph, output_dir: str = "data/graph/visualizations"):
    """
    Cria dashboard resumo com principais visualizações
    
    Args:
        graph: Grafo NetworkX
        output_dir: Diretório de saída
        
    Returns:
        Caminho do dashboard
    """
    
    logger.info("📊 Criando dashboard resumo...")
    
    output_path = Path(output_dir) / "dashboard_summary.png"
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Informações básicas (texto)
    ax1 = fig.add_subplot(gs[0, 0])
    info_text = f"""
ESTATÍSTICAS DO GRAFO

Nós: {graph.number_of_nodes():,}
Arestas: {graph.number_of_edges():,}
Densidade: {nx.density(graph):.4f}

Conectado: {nx.is_connected(graph)}
Componentes: {nx.number_connected_components(graph)}

Grau Médio: {np.mean([d for n,d in graph.degree()]):.2f}
Grau Máximo: {max([d for n,d in graph.degree()])}
    """
    ax1.text(0.1, 0.9, info_text, transform=ax1.transAxes,
            verticalalignment='top', fontsize=12, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    ax1.axis('off')
    ax1.set_title('Informações Gerais', fontsize=14, fontweight='bold')
    
    # 2. Distribuição de graus
    ax2 = fig.add_subplot(gs[0, 1:])
    degrees = [d for n, d in graph.degree()]
    ax2.hist(degrees, bins=50, color='#4A90E2', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Grau', fontsize=11)
    ax2.set_ylabel('Frequência', fontsize=11)
    ax2.set_title('Distribuição de Graus', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    # 3. Tipos de nós
    ax3 = fig.add_subplot(gs[1, 0])
    node_types = {}
    for node, attrs in graph.nodes(data=True):
        node_type = attrs.get('node_type', 'unknown')
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    visualizer = GraphVisualizer(graph)
    colors = [visualizer.node_colors.get(t, '#9B9B9B') for t in node_types.keys()]
    ax3.pie(node_types.values(), labels=node_types.keys(), colors=colors,
           autopct='%1.1f%%', startangle=90)
    ax3.set_title('Tipos de Nós', fontsize=14, fontweight='bold')
    
    # 4. Tipos de arestas
    ax4 = fig.add_subplot(gs[1, 1])
    edge_types = {}
    for u, v, attrs in graph.edges(data=True):
        edge_type = attrs.get('edge_type', 'unknown')
        edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
    
    colors = [visualizer.edge_colors.get(t, '#9B9B9B') for t in edge_types.keys()]
    ax4.pie(edge_types.values(), labels=edge_types.keys(), colors=colors,
           autopct='%1.1f%%', startangle=90)
    ax4.set_title('Tipos de Arestas', fontsize=14, fontweight='bold')
    
    # 5. Rede (sample)
    ax5 = fig.add_subplot(gs[1:, 2])
    sample_size = min(200, graph.number_of_nodes())
    nodes_sample = np.random.choice(list(graph.nodes()), size=sample_size, replace=False)
    G_sample = graph.subgraph(nodes_sample)
    
    pos = nx.spring_layout(G_sample, k=0.8, iterations=30)
    
    node_colors = []
    for node in G_sample.nodes():
        node_type = G_sample.nodes[node].get('node_type', 'unknown')
        node_colors.append(visualizer.node_colors.get(node_type, '#9B9B9B'))
    
    nx.draw_networkx_nodes(G_sample, pos, node_color=node_colors, 
                          node_size=30, alpha=0.7, ax=ax5)
    nx.draw_networkx_edges(G_sample, pos, edge_color='#CCCCCC', 
                          alpha=0.3, width=0.5, ax=ax5)
    
    ax5.set_title(f'Amostra da Rede ({sample_size} nós)', fontsize=14, fontweight='bold')
    ax5.axis('off')
    
    # 6. Pesos de similaridade
    ax6 = fig.add_subplot(gs[2, 0])
    similarities = []
    for u, v, attrs in graph.edges(data=True):
        if attrs.get('edge_type') == 'similarity':
            similarities.append(attrs.get('weight', 0.0))
    
    if similarities:
        ax6.hist(similarities, bins=30, color='#4A90E2', alpha=0.7, edgecolor='black')
        ax6.set_xlabel('Similaridade', fontsize=11)
        ax6.set_ylabel('Frequência', fontsize=11)
        ax6.set_title('Distribuição de Similaridades', fontsize=14, fontweight='bold')
        ax6.grid(alpha=0.3)
    else:
        ax6.text(0.5, 0.5, 'Sem dados de similaridade', 
                transform=ax6.transAxes, ha='center', va='center')
        ax6.axis('off')
    
    # 7. Top nós por PageRank
    ax7 = fig.add_subplot(gs[2, 1])
    try:
        pagerank = nx.pagerank(graph)
        top_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
        
        nodes_labels = [n[:20] + '...' if len(n) > 20 else n for n, _ in top_nodes]
        nodes_scores = [s for _, s in top_nodes]
        
        ax7.barh(range(len(nodes_labels)), nodes_scores, color='#F5A623', alpha=0.7)
        ax7.set_yticks(range(len(nodes_labels)))
        ax7.set_yticklabels(nodes_labels, fontsize=9)
        ax7.set_xlabel('PageRank', fontsize=11)
        ax7.set_title('Top 10 Nós (PageRank)', fontsize=14, fontweight='bold')
        ax7.grid(axis='x', alpha=0.3)
        ax7.invert_yaxis()
    except:
        ax7.text(0.5, 0.5, 'Erro ao calcular PageRank',
                transform=ax7.transAxes, ha='center', va='center')
        ax7.axis('off')
    
    # Título geral
    fig.suptitle('Dashboard - Grafo de Jurisprudências', 
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Dashboard criado: {output_path}")
    
    return str(output_path)


def plot_subgraph_by_node_type(graph: nx.Graph, node_type: str, 
                               output_path: str = "subgraph.png"):
    """
    Plota subgrafo filtrado por tipo de nó
    
    Args:
        graph: Grafo completo
        node_type: Tipo de nó para filtrar
        output_path: Caminho de saída
        
    Returns:
        Caminho do arquivo
    """
    
    # Filtra nós
    nodes_filtered = [
        node for node, attrs in graph.nodes(data=True)
        if attrs.get('node_type') == node_type
    ]
    
    if not nodes_filtered:
        logger.warning(f"Nenhum nó do tipo '{node_type}' encontrado")
        return ""
    
    # Cria subgrafo
    subgraph = graph.subgraph(nodes_filtered)
    
    # Plota
    visualizer = GraphVisualizer(subgraph)
    return visualizer.plot_network_overview(f"subgraph_{node_type}")


def export_network_for_gephi(graph: nx.Graph, output_path: str = "network_gephi.gexf"):
    """
    Exporta rede otimizada para visualização no Gephi
    
    Args:
        graph: Grafo NetworkX
        output_path: Caminho de saída
    """
    
    logger.info("📤 Exportando para Gephi...")
    
    # Adiciona atributos de visualização
    for node in graph.nodes():
        attrs = graph.nodes[node]
        node_type = attrs.get('node_type', 'unknown')
        
        # Define cor por tipo
        visualizer = GraphVisualizer(graph)
        color = visualizer.node_colors.get(node_type, '#9B9B9B')
        
        # Converte hex para RGB
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        
        graph.nodes[node]['viz'] = {
            'color': {'r': r, 'g': g, 'b': b, 'a': 1.0},
            'size': graph.degree(node) * 2
        }
    
    # Exporta
    nx.write_gexf(graph, output_path)
    
    logger.info(f"✅ Arquivo Gephi exportado: {output_path}")


def create_comparison_plot(graphs: Dict[str, nx.Graph], 
                          metric: str = 'degree',
                          output_path: str = "comparison.png"):
    """
    Cria gráfico de comparação entre múltiplos grafos
    
    Args:
        graphs: Dicionário {nome: grafo}
        metric: Métrica para comparar ('degree', 'density', 'components')
        output_path: Caminho de saída
    """
    
    logger.info(f"📊 Criando comparação de grafos (métrica: {metric})...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    names = list(graphs.keys())
    
    if metric == 'degree':
        values = [np.mean([d for n, d in g.degree()]) for g in graphs.values()]
        ylabel = 'Grau Médio'
        
    elif metric == 'density':
        values = [nx.density(g) for g in graphs.values()]
        ylabel = 'Densidade'
        
    elif metric == 'components':
        values = [nx.number_connected_components(g) for g in graphs.values()]
        ylabel = 'Número de Componentes'
        
    else:
        logger.error(f"Métrica desconhecida: {metric}")
        return ""
    
    # Plota
    bars = ax.bar(names, values, color='#4A90E2', alpha=0.7, edgecolor='black')
    
    # Adiciona valores nas barras
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}',
               ha='center', va='bottom', fontsize=11)
    
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'Comparação de Grafos - {ylabel}', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Comparação plotada: {output_path}")
    
    return output_path


def create_ego_network_plot(graph: nx.Graph, center_node: str, 
                           radius: int = 2,
                           output_path: str = "ego_network.png"):
    """
    Cria visualização de ego network (rede centrada em um nó)
    
    Args:
        graph: Grafo completo
        center_node: Nó central
        radius: Raio da ego network
        output_path: Caminho de saída
    """
    
    if center_node not in graph:
        logger.error(f"Nó '{center_node}' não encontrado no grafo")
        return ""
    
    logger.info(f"🎯 Criando ego network centrada em '{center_node}'...")
    
    # Cria ego network
    ego = nx.ego_graph(graph, center_node, radius=radius)
    
    # Plota
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Layout centrado
    pos = nx.spring_layout(ego, k=1, iterations=50)
    
    # Cores: destaca nó central
    node_colors = []
    node_sizes = []
    
    for node in ego.nodes():
        if node == center_node:
            node_colors.append('#D0021B')  # Vermelho para central
            node_sizes.append(500)
        else:
            node_type = ego.nodes[node].get('node_type', 'unknown')
            visualizer = GraphVisualizer(graph)
            node_colors.append(visualizer.node_colors.get(node_type, '#9B9B9B'))
            node_sizes.append(200)
    
    # Desenha
    nx.draw_networkx_nodes(ego, pos, node_color=node_colors, 
                          node_size=node_sizes, alpha=0.8, ax=ax)
    nx.draw_networkx_edges(ego, pos, edge_color='#CCCCCC', 
                          alpha=0.4, width=1, ax=ax)
    
    # Labels apenas para nós próximos
    if ego.number_of_nodes() < 50:
        labels = {node: ego.nodes[node].get('label', node)[:30] 
                 for node in ego.nodes()}
        nx.draw_networkx_labels(ego, pos, labels, font_size=8, ax=ax)
    
    center_label = graph.nodes[center_node].get('label', center_node)
    ax.set_title(f'Ego Network: {center_label}\n{ego.number_of_nodes()} nós, raio={radius}',
                fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Ego network plotada: {output_path}")
    
    return output_path


def animate_graph_construction(graphs_sequence: List[nx.Graph],
                              output_path: str = "graph_animation.gif",
                              fps: int = 2):
    """
    Cria animação mostrando evolução da construção do grafo
    
    Args:
        graphs_sequence: Lista de grafos em diferentes estágios
        output_path: Caminho do GIF de saída
        fps: Frames por segundo
        
    Note:
        Requer imageio instalado: pip install imageio
    """
    
    try:
        import imageio
    except ImportError:
        logger.error("imageio não instalado. Install: pip install imageio")
        return ""
    
    logger.info(f"🎬 Criando animação com {len(graphs_sequence)} frames...")
    
    temp_dir = Path("temp_animation")
    temp_dir.mkdir(exist_ok=True)
    
    frames = []
    
    for i, graph in enumerate(graphs_sequence):
        # Plota cada estágio
        temp_path = temp_dir / f"frame_{i:03d}.png"
        
        visualizer = GraphVisualizer(graph)
        visualizer.plot_network_overview(f"frame_{i:03d}")
        
        # Lê imagem
        frames.append(imageio.imread(temp_path))
    
    # Cria GIF
    imageio.mimsave(output_path, frames, fps=fps)
    
    # Limpa arquivos temporários
    import shutil
    shutil.rmtree(temp_dir)
    
    logger.info(f"✅ Animação criada: {output_path}")
    
    return output_path