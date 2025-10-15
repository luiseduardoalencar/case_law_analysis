#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para construção do grafo heterogêneo de jurisprudências
Executa pipeline completo: dados -> grafo -> análise -> visualização
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from loguru import logger

# Adiciona src ao path para imports funcionarem
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Imports do grafo
from graph.pipeline.graph_pipeline import build_graph_from_database, build_sample_graph
from graph.utils.export import GraphExporter
from graph.utils.metrics import GraphMetricsCalculator
from graph.utils.visualization import GraphVisualizer, create_summary_dashboard
from graph.models.graph_schema import GraphConfiguration


def setup_logging():
    """Configura logging"""
    logger.remove()
    logger.add(sys.stdout, level="INFO", 
              format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")
    
    # Log em arquivo
    log_dir = Path("../../data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"graph_construction_{datetime.now():%Y%m%d_%H%M%S}.log"
    logger.add(log_file, level="DEBUG")
    logger.info(f"📝 Log: {log_file}")


def run_pipeline(args):
    """Executa pipeline completo de construção do grafo"""
    
    logger.info("=" * 80)
    logger.info("🚀 CONSTRUÇÃO DO GRAFO DE JURISPRUDÊNCIAS")
    logger.info("=" * 80)
    
    start_time = datetime.now()
    
    # 1. CONSTRUIR GRAFO
    logger.info("\n📊 ETAPA 1: Construindo Grafo")
    logger.info("-" * 80)
    
    if args.test:
        logger.info("🔬 Modo TESTE: 50 documentos")
        result = build_sample_graph(sample_size=50)
    elif args.sample:
        logger.info(f"🔬 Modo AMOSTRA: {args.sample} documentos")
        result = build_graph_from_database(limit=args.sample)
    else:
        logger.info(f"📈 Modo COMPLETO: {args.limit or 'TODOS'} documentos")
        result = build_graph_from_database(limit=args.limit)
    
    if not result.success:
        logger.error("❌ Falha na construção do grafo!")
        for error in result.errors:
            logger.error(f"  - {error}")
        return 1
    
    graph = result.graph
    
    logger.info("\n✅ GRAFO CONSTRUÍDO!")
    logger.info(f"  • Nós: {graph.number_of_nodes():,}")
    logger.info(f"  • Arestas: {graph.number_of_edges():,}")
    logger.info(f"  • Tempo: {result.execution_times.get('total', 0):.2f}s")
    
    # Criar diretório base de output
    output_base = Path(args.output)
    output_base.mkdir(parents=True, exist_ok=True)
    
    # 2. EXPORTAR
    if args.export:
        logger.info("\n📦 ETAPA 2: Exportando Grafo")
        logger.info("-" * 80)
        
        export_dir = output_base / "exports"
        exporter = GraphExporter(graph, str(export_dir))
        
        exported = exporter.export_all("grafo_jurisprudencias")
        
        logger.info(f"✅ {len(exported)} arquivos exportados:")
        for fmt, path in list(exported.items())[:5]:  # Mostra primeiros 5
            logger.info(f"  • {fmt}: {Path(path).name}")
        if len(exported) > 5:
            logger.info(f"  ... e mais {len(exported)-5} arquivos")
    
    # 3. MÉTRICAS
    if args.metrics:
        logger.info("\n📊 ETAPA 3: Calculando Métricas")
        logger.info("-" * 80)
        
        calculator = GraphMetricsCalculator(graph)
        
        # Métricas básicas
        quality = calculator.calculate_quality_metrics()
        logger.info(f"  • Densidade: {quality.density:.6f}")
        logger.info(f"  • Clustering: {quality.average_clustering:.4f}")
        logger.info(f"  • Grau médio: {quality.average_degree:.2f}")
        
        # Comunidades
        logger.info("\n  🏘️  Detectando comunidades...")
        communities = calculator.detect_communities(method='louvain')
        logger.info(f"  • Comunidades: {communities['num_communities']}")
        logger.info(f"  • Modularidade: {communities['modularity']:.4f}")
        
        # Top nós
        logger.info("\n  ⭐ Top 5 nós mais centrais:")
        centralities = calculator.calculate_centrality_metrics(top_k=5)
        for i, cent in enumerate(centralities[:5], 1):
            label = graph.nodes[cent.node_id].get('label', cent.node_id)[:40]
            logger.info(f"  {i}. {label} (PR: {cent.pagerank:.6f})")
        
        # Exportar relatório
        metrics_dir = output_base / "metrics"
        metrics_dir.mkdir(exist_ok=True)
        metrics_file = metrics_dir / "relatorio_completo.json"
        calculator.export_metrics_report(str(metrics_file))
        logger.info(f"\n  💾 Relatório salvo: {metrics_file.name}")
    
    # 4. VISUALIZAÇÃO
    if args.visualize:
        logger.info("\n🎨 ETAPA 4: Criando Visualizações")
        logger.info("-" * 80)
        
        viz_dir = output_base / "visualizations"
        visualizer = GraphVisualizer(graph, str(viz_dir))
        
        viz_files = {}
        
        # Dashboard obrigatório
        logger.info("  • Dashboard resumo...")
        viz_files['dashboard'] = create_summary_dashboard(graph, str(viz_dir))
        
        if args.full_viz:
            logger.info("  • Visão geral da rede...")
            viz_files['network'] = visualizer.plot_network_overview("rede")
            
            logger.info("  • Distribuições...")
            viz_files['degree'] = visualizer.plot_degree_distribution("graus")
            viz_files['nodes'] = visualizer.plot_node_types_distribution("tipos_nos")
            viz_files['edges'] = visualizer.plot_edge_types_distribution("tipos_arestas")
        
        if args.interactive:
            logger.info("  • Visualização interativa...")
            viz_files['interactive'] = visualizer.create_interactive_network("interativo")
        
        logger.info(f"\n  ✅ {len([v for v in viz_files.values() if v])} visualizações criadas")
        for name, path in viz_files.items():
            if path:
                logger.info(f"    - {Path(path).name}")
    
    # RESUMO FINAL
    elapsed = (datetime.now() - start_time).total_seconds()
    
    logger.info("\n" + "=" * 80)
    logger.info("🎉 PIPELINE CONCLUÍDO!")
    logger.info("=" * 80)
    logger.info(f"⏱️  Tempo total: {elapsed:.2f}s ({elapsed/60:.2f} min)")
    logger.info(f"📁 Output: {output_base.absolute()}")
    logger.info("")
    logger.info("📊 Resumo do Grafo:")
    logger.info(f"  • {graph.number_of_nodes():,} nós")
    logger.info(f"  • {graph.number_of_edges():,} arestas")
    logger.info(f"  • Densidade: {result.statistics.density:.6f}")
    logger.info("")
    logger.info("✅ Arquivos gerados em:")
    logger.info(f"  {output_base.absolute()}")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Construtor de Grafo Heterogêneo de Jurisprudências",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:

  # Teste rápido (50 docs)
  python build_graph.py --test --export --visualize
  
  # Amostra de 200 docs com tudo
  python build_graph.py --sample 200 --all
  
  # 1000 docs completo
  python build_graph.py --limit 1000 --all --interactive
  
  # Todos os docs (CUIDADO!)
  python build_graph.py --all
        """
    )
    
    # Modo de execução
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--test', action='store_true',
                     help='Teste rápido com 50 documentos')
    mode.add_argument('--sample', type=int, metavar='N',
                     help='Amostra de N documentos')
    mode.add_argument('--limit', type=int, metavar='N',
                     help='Limite de N documentos (None=todos)')
    
    # Etapas
    parser.add_argument('--export', action='store_true',
                       help='Exporta grafo (GraphML, JSON, CSV, etc)')
    parser.add_argument('--metrics', action='store_true',
                       help='Calcula métricas e comunidades')
    parser.add_argument('--visualize', action='store_true',
                       help='Cria visualizações (PNG)')
    parser.add_argument('--interactive', action='store_true',
                       help='Cria visualização interativa (HTML)')
    parser.add_argument('--full-viz', action='store_true',
                       help='Cria TODAS as visualizações')
    parser.add_argument('--all', action='store_true',
                       help='Ativa todas as etapas')
    
    # Output
    parser.add_argument('--output', type=str, 
                       default='../../data/graph',
                       help='Diretório de saída (default: ../../data/graph)')
    
    args = parser.parse_args()
    
    # Setup
    setup_logging()
    
    # Se --all, ativa tudo
    if args.all:
        args.export = True
        args.metrics = True
        args.visualize = True
        args.interactive = True
        args.full_viz = True
    
    
    # Se --all sem modo, processa tudo; senão usa test
    if not (args.test or args.sample or args.limit):
        if args.all:
            logger.info("📈 Modo COMPLETO: processando TODOS os documentos")
            args.limit = None
        else:
            logger.warning("⚠️  Nenhum modo especificado, usando --test")
            args.test = True
    
    # Executa
    try:
        return run_pipeline(args)
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Interrompido pelo usuário")
        return 130
    except Exception as e:
        logger.exception(f"❌ Erro fatal: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())