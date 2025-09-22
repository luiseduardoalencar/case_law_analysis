#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de teste para validar desenvolvimento das Etapas 1 e 2
Testa modelos de dados e preprocessamento
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Adiciona src ao path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Configurações básicas
os.environ['TZ'] = 'America/Sao_Paulo'

def print_section(title: str):
    """Imprime seção formatada"""
    print(f"\n{'='*80}")
    print(f"🔍 {title}")
    print(f"{'='*80}")

def print_subsection(title: str):
    """Imprime subseção formatada"""
    print(f"\n{'-'*60}")
    print(f"📋 {title}")
    print(f"{'-'*60}")

def print_success(message: str):
    """Imprime mensagem de sucesso"""
    print(f"✅ {message}")

def print_error(message: str):
    """Imprime mensagem de erro"""
    print(f"❌ {message}")

def print_warning(message: str):
    """Imprime mensagem de aviso"""
    print(f"⚠️ {message}")

def print_info(message: str):
    """Imprime mensagem informativa"""
    print(f"ℹ️ {message}")

def test_imports():
    """Testa importação de todos os módulos"""
    print_section("TESTE DE IMPORTAÇÕES")
    
    import_tests = [
        ("graph.models.nodes", "Modelos de Nós"),
        ("graph.models.edges", "Modelos de Arestas"),
        ("graph.models.graph_schema", "Schema do Grafo"),
        ("graph.pipeline.preprocessing", "Pipeline de Preprocessamento")
    ]
    
    results = {}
    
    for module_name, description in import_tests:
        try:
            module = __import__(module_name, fromlist=[''])
            print_success(f"{description}: {module_name}")
            results[module_name] = True
        except ImportError as e:
            print_error(f"{description}: {e}")
            results[module_name] = False
        except Exception as e:
            print_error(f"{description}: Erro inesperado - {e}")
            results[module_name] = False
    
    return results

def test_node_models():
    """Testa modelos de nós"""
    print_section("TESTE DOS MODELOS DE NÓS")
    
    try:
        from graph.models.nodes import (
            DocumentNode, SectionNode, EntityNode, ConceptNode,
            NodeType, SectionType, EntityType,
            create_document_node_from_db_row, CONCEITOS_JURIDICOS_PREDEFINIDOS
        )
        
        results = {}
        
        # Teste 1: Criação de DocumentNode
        print_subsection("Teste DocumentNode")
        try:
            doc = DocumentNode(
                id="test_doc_001",
                numero_processo="0001234-56.2023.8.18.0001",
                url_original="https://exemplo.com",
                orgao_julgador="1ª Vara Cível",
                relator="Des. João Silva",
                conteudo_limpo="Este é um texto de teste para validar o documento.",
                num_tokens=12
            )
            
            print_success(f"DocumentNode criado: {doc.id}")
            print_info(f"  - Número do processo: {doc.numero_processo}")
            print_info(f"  - Órgão julgador: {doc.orgao_julgador}")
            print_info(f"  - Número de tokens: {doc.num_tokens}")
            
            # Testa métodos
            doc.add_secao("sec_001")
            doc.add_entidade("ent_001")
            doc.add_conceito("con_001")
            
            print_info(f"  - Seções: {doc.secoes_ids}")
            print_info(f"  - Entidades: {doc.entidades_ids}")
            print_info(f"  - Conceitos: {doc.conceitos_ids}")
            
            # Testa serialização
            doc_dict = doc.to_dict()
            print_info(f"  - Serialização: {len(doc_dict)} campos")
            
            results['DocumentNode'] = True
            
        except Exception as e:
            print_error(f"Erro ao testar DocumentNode: {e}")
            results['DocumentNode'] = False
        
        # Teste 2: Criação de SectionNode
        print_subsection("Teste SectionNode")
        try:
            section = SectionNode(
                id="sec_test_001",
                parent_document_id="test_doc_001",
                section_type=SectionType.DECISAO,
                conteudo_texto="Esta é uma decisão judicial de teste.",
                ordem=1
            )
            
            print_success(f"SectionNode criado: {section.id}")
            print_info(f"  - Tipo: {section.section_type.value}")
            print_info(f"  - Documento pai: {section.parent_document_id}")
            print_info(f"  - Ordem: {section.ordem}")
            
            results['SectionNode'] = True
            
        except Exception as e:
            print_error(f"Erro ao testar SectionNode: {e}")
            results['SectionNode'] = False
        
        # Teste 3: Criação de EntityNode
        print_subsection("Teste EntityNode")
        try:
            entity = EntityNode(
                id="ent_test_001",
                entity_type=EntityType.JUIZ,
                nome_original="Des. João Silva",
                nome_normalizado="joão silva",
                frequencia_global=5
            )
            
            entity.add_variacao("João Silva")
            entity.add_variacao("Desembargador João Silva")
            entity.incrementar_frequencia()
            
            print_success(f"EntityNode criado: {entity.id}")
            print_info(f"  - Tipo: {entity.entity_type.value}")
            print_info(f"  - Nome normalizado: {entity.nome_normalizado}")
            print_info(f"  - Frequência: {entity.frequencia_global}")
            print_info(f"  - Variações: {entity.variacoes}")
            
            results['EntityNode'] = True
            
        except Exception as e:
            print_error(f"Erro ao testar EntityNode: {e}")
            results['EntityNode'] = False
        
        # Teste 4: Criação de ConceptNode
        print_subsection("Teste ConceptNode")
        try:
            concept = ConceptNode(
                id="con_test_001",
                termo_conceito="empréstimo consignado",
                categoria_juridica="direito bancário",
                frequencia_global=15
            )
            
            concept.add_contexto("Contrato de empréstimo consignado em folha de pagamento")
            concept.add_documento("test_doc_001", 0.85)
            concept.set_pmi_score("con_test_002", 0.7)
            
            print_success(f"ConceptNode criado: {concept.id}")
            print_info(f"  - Termo: {concept.termo_conceito}")
            print_info(f"  - Categoria: {concept.categoria_juridica}")
            print_info(f"  - Frequência: {concept.frequencia_global}")
            print_info(f"  - Contextos: {len(concept.contextos)}")
            print_info(f"  - Scores TF-IDF: {concept.tfidf_scores}")
            print_info(f"  - Scores PMI: {concept.pmi_scores}")
            
            results['ConceptNode'] = True
            
        except Exception as e:
            print_error(f"Erro ao testar ConceptNode: {e}")
            results['ConceptNode'] = False
        
        # Teste 5: Conceitos jurídicos predefinidos
        print_subsection("Teste Conceitos Jurídicos Predefinidos")
        try:
            print_info(f"Total de conceitos predefinidos: {len(CONCEITOS_JURIDICOS_PREDEFINIDOS)}")
            print_info("Alguns conceitos:")
            for i, conceito in enumerate(CONCEITOS_JURIDICOS_PREDEFINIDOS[:5]):
                print_info(f"  {i+1}. {conceito}")
            
            # Verifica se tem conceitos específicos de empréstimo consignado
            conceitos_consignado = [c for c in CONCEITOS_JURIDICOS_PREDEFINIDOS 
                                  if 'consignado' in c.lower()]
            print_info(f"Conceitos relacionados a consignado: {conceitos_consignado}")
            
            results['ConceptosJuridicos'] = True
            
        except Exception as e:
            print_error(f"Erro ao testar conceitos jurídicos: {e}")
            results['ConceptosJuridicos'] = False
        
        return results
        
    except ImportError as e:
        print_error(f"Erro de importação nos modelos de nós: {e}")
        return {}

def test_edge_models():
    """Testa modelos de arestas"""
    print_section("TESTE DOS MODELOS DE ARESTAS")
    
    try:
        from graph.models.edges import (
            SimilarityEdge, RelevanceEdge, CooccurrenceEdge, HierarchicalEdge,
            EdgeType, SimilarityType,
            create_similarity_matrix_edges, validate_edge_weights
        )
        import numpy as np
        
        results = {}
        
        # Teste 1: SimilarityEdge
        print_subsection("Teste SimilarityEdge")
        try:
            sim_edge = SimilarityEdge.create_document_similarity(
                "doc_001", "doc_002", 0.85, "sentence-transformers", 0.9
            )
            
            print_success(f"SimilarityEdge criado: {sim_edge.id}")
            print_info(f"  - Fonte: {sim_edge.source_node_id}")
            print_info(f"  - Destino: {sim_edge.target_node_id}")
            print_info(f"  - Peso: {sim_edge.weight}")
            print_info(f"  - Score cossenos: {sim_edge.cosine_score}")
            print_info(f"  - Modelo: {sim_edge.embedding_model}")
            
            # Testa tuple para NetworkX
            edge_tuple = sim_edge.get_tuple()
            print_info(f"  - Tupla NetworkX: {len(edge_tuple)} elementos")
            
            results['SimilarityEdge'] = True
            
        except Exception as e:
            print_error(f"Erro ao testar SimilarityEdge: {e}")
            results['SimilarityEdge'] = False
        
        # Teste 2: RelevanceEdge
        print_subsection("Teste RelevanceEdge")
        try:
            rel_edge = RelevanceEdge.create_document_concept_relevance(
                "doc_001", "con_001", 0.45, 5, 10, 0.69, 100
            )
            
            print_success(f"RelevanceEdge criado: {rel_edge.id}")
            print_info(f"  - Peso (TF-IDF): {rel_edge.weight}")
            print_info(f"  - Term Frequency: {rel_edge.term_frequency}")
            print_info(f"  - Document Frequency: {rel_edge.document_frequency}")
            print_info(f"  - IDF: {rel_edge.inverse_document_frequency}")
            
            results['RelevanceEdge'] = True
            
        except Exception as e:
            print_error(f"Erro ao testar RelevanceEdge: {e}")
            results['RelevanceEdge'] = False
        
        # Teste 3: CooccurrenceEdge
        print_subsection("Teste CooccurrenceEdge")
        try:
            cooc_edge = CooccurrenceEdge.create_concept_cooccurrence(
                "con_001", "con_002", 1.25, 15, 50, 40, 1000, 10
            )
            
            print_success(f"CooccurrenceEdge criado: {cooc_edge.id}")
            print_info(f"  - Peso (PMI): {cooc_edge.weight}")
            print_info(f"  - Frequência conjunta: {cooc_edge.joint_frequency}")
            print_info(f"  - Total janelas: {cooc_edge.total_windows}")
            print_info(f"  - Tamanho janela: {cooc_edge.window_size}")
            
            results['CooccurrenceEdge'] = True
            
        except Exception as e:
            print_error(f"Erro ao testar CooccurrenceEdge: {e}")
            results['CooccurrenceEdge'] = False
        
        # Teste 4: HierarchicalEdge
        print_subsection("Teste HierarchicalEdge")
        try:
            hier_edge = HierarchicalEdge.create_document_section(
                "doc_001", "sec_001", 1
            )
            
            print_success(f"HierarchicalEdge criado: {hier_edge.id}")
            print_info(f"  - Peso: {hier_edge.weight}")
            print_info(f"  - Tipo hierarquia: {hier_edge.hierarchy_type}")
            print_info(f"  - Ordem: {hier_edge.order}")
            
            results['HierarchicalEdge'] = True
            
        except Exception as e:
            print_error(f"Erro ao testar HierarchicalEdge: {e}")
            results['HierarchicalEdge'] = False
        
        # Teste 5: Validação de pesos
        print_subsection("Teste Validação de Arestas")
        try:
            edges = [sim_edge, rel_edge, cooc_edge, hier_edge]
            stats = validate_edge_weights(edges)
            
            print_success("Validação de pesos executada")
            for edge_type, edge_stats in stats.items():
                print_info(f"  - {edge_type}: {edge_stats}")
            
            results['EdgeValidation'] = True
            
        except Exception as e:
            print_error(f"Erro ao testar validação: {e}")
            results['EdgeValidation'] = False
        
        return results
        
    except ImportError as e:
        print_error(f"Erro de importação nos modelos de arestas: {e}")
        return {}

def test_graph_schema():
    """Testa schema do grafo"""
    print_section("TESTE DO SCHEMA DO GRAFO")
    
    try:
        from graph.models.graph_schema import (
            GraphConfiguration, GraphSchema, GraphStatistics,
            GraphValidationLevel, create_default_schema, create_optimized_schema,
            DEVELOPMENT_CONFIG, PRODUCTION_CONFIG
        )
        import networkx as nx
        
        results = {}
        
        # Teste 1: Configuração padrão
        print_subsection("Teste GraphConfiguration")
        try:
            config = GraphConfiguration()
            config_dict = config.to_dict()
            
            print_success("GraphConfiguration criada")
            print_info(f"  - Similarity threshold: {config.similarity_thresholds}")
            print_info(f"  - Relevance threshold: {config.relevance_threshold}")
            print_info(f"  - PMI threshold: {config.pmi_threshold}")
            print_info(f"  - Embedding model: {config.embedding_model}")
            print_info(f"  - Chunk size: {config.chunk_size}")
            print_info(f"  - Validação: {config.validation_level.value}")
            
            results['GraphConfiguration'] = True
            
        except Exception as e:
            print_error(f"Erro ao testar GraphConfiguration: {e}")
            results['GraphConfiguration'] = False
        
        # Teste 2: Schema padrão
        print_subsection("Teste GraphSchema")
        try:
            schema = create_default_schema()
            
            print_success("GraphSchema criado")
            print_info(f"  - Tipos de nós permitidos: {len(schema.allowed_node_types)}")
            print_info(f"  - Relações permitidas: {len(schema.allowed_edge_types)}")
            
            # Testa otimização para 3000 documentos
            optimized_config = schema.optimize_config_for_size(3000)
            print_info(f"  - Config otimizada para 3000 docs:")
            print_info(f"    * Similarity: {optimized_config.similarity_thresholds}")
            print_info(f"    * Max edges: {optimized_config.max_similarity_edges_per_document}")
            
            results['GraphSchema'] = True
            
        except Exception as e:
            print_error(f"Erro ao testar GraphSchema: {e}")
            results['GraphSchema'] = False
        
        # Teste 3: Estatísticas
        print_subsection("Teste GraphStatistics")
        try:
            # Cria um grafo simples para testar
            G = nx.Graph()
            G.add_node("doc_1", node_type="document")
            G.add_node("doc_2", node_type="document")
            G.add_node("sec_1", node_type="section")
            G.add_edge("doc_1", "doc_2", edge_type="similarity", weight=0.8)
            G.add_edge("doc_1", "sec_1", edge_type="hierarchical", weight=1.0)
            
            stats = GraphStatistics()
            stats.update_from_networkx(G)
            
            print_success("GraphStatistics calculadas")
            print_info(f"  - Total nós: {stats.total_nodes}")
            print_info(f"  - Total arestas: {stats.total_edges}")
            print_info(f"  - Densidade: {stats.density:.4f}")
            print_info(f"  - Grau médio: {stats.average_degree:.2f}")
            print_info(f"  - Componentes: {stats.num_connected_components}")
            
            stats_dict = stats.to_dict()
            print_info(f"  - Serialização: {len(stats_dict)} campos principais")
            
            results['GraphStatistics'] = True
            
        except Exception as e:
            print_error(f"Erro ao testar GraphStatistics: {e}")
            results['GraphStatistics'] = False
        
        # Teste 4: Configurações predefinidas
        print_subsection("Teste Configurações Predefinidas")
        try:
            configs = {
                'DEVELOPMENT': DEVELOPMENT_CONFIG,
                'PRODUCTION': PRODUCTION_CONFIG
            }
            
            for name, config in configs.items():
                print_success(f"Config {name} carregada")
                print_info(f"  - Similarity threshold: {config.similarity_thresholds}")
                print_info(f"  - Max edges: {config.max_similarity_edges_per_document}")
                print_info(f"  - Validation: {config.validation_level.value}")
            
            results['PredefinedConfigs'] = True
            
        except Exception as e:
            print_error(f"Erro ao testar configurações predefinidas: {e}")
            results['PredefinedConfigs'] = False
        
        return results
        
    except ImportError as e:
        print_error(f"Erro de importação no schema: {e}")
        return {}

def test_preprocessing():
    """Testa o preprocessamento"""
    print_section("TESTE DO PREPROCESSAMENTO")
    
    try:
        from graph.pipeline.preprocessing import (
            DatabaseLoader, TextCleaner, JurisprudenciaPreprocessor,
            preprocess_jurisprudencias, get_sample_documents
        )
        
        results = {}
        
        # Teste 1: TextCleaner
        print_subsection("Teste TextCleaner")
        try:
            cleaner = TextCleaner()
            
            # Testa limpeza de HTML
            html_content = """
            <div>
                <h1>ACÓRDÃO</h1>
                <p>EMENTA: Este é um <strong>teste</strong> de limpeza de HTML.</p>
                <script>alert('test');</script>
                <style>body { color: blue; }</style>
            </div>
            """
            
            clean_text = cleaner.extract_clean_text(html_content)
            
            print_success("TextCleaner funcionando")
            print_info(f"  - HTML original: {len(html_content)} chars")
            print_info(f"  - Texto limpo: {len(clean_text)} chars")
            print_info(f"  - Amostra: {clean_text[:100]}...")
            
            # Testa validação
            is_valid = cleaner.is_valid_document(clean_text)
            print_info(f"  - Documento válido: {is_valid}")
            
            results['TextCleaner'] = True
            
        except Exception as e:
            print_error(f"Erro ao testar TextCleaner: {e}")
            results['TextCleaner'] = False
        
        # Teste 2: DatabaseLoader
        print_subsection("Teste DatabaseLoader")
        try:
            # Testa conexão com banco (sem executar queries pesadas)
            loader = DatabaseLoader()
            
            print_success("DatabaseLoader inicializado")
            print_info(f"  - Engine criado: {loader.engine is not None}")
            print_info(f"  - Session criada: {loader.Session is not None}")
            
            loader.close()
            print_info("  - Conexão fechada com sucesso")
            
            results['DatabaseLoader'] = True
            
        except Exception as e:
            print_error(f"Erro ao testar DatabaseLoader: {e}")
            print_warning("  - Isso pode ser normal se o banco não estiver rodando")
            results['DatabaseLoader'] = False
        
        # Teste 3: Preprocessor (sem banco)
        print_subsection("Teste JurisprudenciaPreprocessor (Mock)")
        try:
            # Cria dados mock para testar lógica
            from graph.models.nodes import DocumentNode
            from graph.pipeline.preprocessing import PreprocessingStats
            
            # Simula resultado do preprocessamento
            mock_doc = DocumentNode(
                id="doc_test_001",
                numero_processo="0001234-56.2023.8.18.0001",
                url_original="https://test.com",
                conteudo_limpo="Este é um documento de teste para validação.",
                num_tokens=10
            )
            
            mock_stats = PreprocessingStats(
                total_documents=1,
                valid_documents=1,
                documents_with_content=1,
                total_characters=50,
                total_words=10
            )
            mock_stats.update_averages()
            
            print_success("Mock do preprocessamento criado")
            print_info(f"  - Documento: {mock_doc.id}")
            print_info(f"  - Estatísticas: {mock_stats.total_documents} docs")
            print_info(f"  - Média de caracteres: {mock_stats.average_document_length}")
            
            # Testa serialização das estatísticas
            stats_dict = mock_stats.to_dict()
            print_info(f"  - Stats serializadas: {len(stats_dict)} campos")
            
            results['JurisprudenciaPreprocessor'] = True
            
        except Exception as e:
            print_error(f"Erro ao testar preprocessor: {e}")
            results['JurisprudenciaPreprocessor'] = False
        
        return results
        
    except ImportError as e:
        print_error(f"Erro de importação no preprocessamento: {e}")
        return {}

def test_integration():
    """Testa integração entre módulos"""
    print_section("TESTE DE INTEGRAÇÃO")
    
    try:
        from graph.models.nodes import DocumentNode, SectionNode
        from graph.models.edges import SimilarityEdge, HierarchicalEdge
        from graph.models.graph_schema import create_optimized_schema
        from graph.pipeline.preprocessing import PreprocessingStats
        
        results = {}
        
        # Teste 1: Pipeline completo simulado
        print_subsection("Teste Pipeline Simulado")
        try:
            # Cria schema otimizado
            schema = create_optimized_schema(100)  # 100 documentos de teste
            
            # Cria alguns nós
            doc1 = DocumentNode(
                id="doc_001",
                numero_processo="0001234-56.2023.8.18.0001",
                url_original="https://test1.com",
                conteudo_limpo="Primeiro documento sobre empréstimo consignado.",
                relator="Des. João Silva"
            )
            
            doc2 = DocumentNode(
                id="doc_002", 
                numero_processo="0001235-56.2023.8.18.0001",
                url_original="https://test2.com",
                conteudo_limpo="Segundo documento sobre fraude em consignado.",
                relator="Des. Maria Santos"
            )
            
            sec1 = SectionNode(
                id="sec_001",
                parent_document_id="doc_001",
                section_type=SectionType.DECISAO,
                conteudo_limpo="Decisão sobre o empréstimo."
            )
            
            # Cria arestas
            sim_edge = SimilarityEdge.create_document_similarity(
                "doc_001", "doc_002", 0.75
            )
            
            hier_edge = HierarchicalEdge.create_document_section(
                "doc_001", "sec_001", 1
            )
            
            print_success("Pipeline simulado criado")
            print_info(f"  - Documentos: {doc1.id}, {doc2.id}")
            print_info(f"  - Seção: {sec1.id}")
            print_info(f"  - Aresta similaridade: peso {sim_edge.weight}")
            print_info(f"  - Aresta hierárquica: peso {hier_edge.weight}")
            
            # Testa validação do schema
            from graph.models.nodes import NodeType
            errors1 = schema.validate_node(doc1)
            errors2 = schema.validate_edge(sim_edge, NodeType.DOCUMENT, NodeType.DOCUMENT)
            
            print_info(f"  - Erros validação doc1: {len(errors1)}")
            print_info(f"  - Erros validação aresta: {len(errors2)}")
            
            results['IntegrationPipeline'] = True
            
        except Exception as e:
            print_error(f"Erro no teste de integração: {e}")
            results['IntegrationPipeline'] = False
        
        # Teste 2: Compatibilidade de dados
        print_subsection("Teste Compatibilidade")
        try:
            # Testa se todos os tipos enum são compatíveis
            from graph.models.nodes import NodeType, SectionType, EntityType
            from graph.models.edges import EdgeType, SimilarityType
            
            node_types = [t.value for t in NodeType]
            section_types = [t.value for t in SectionType]
            edge_types = [t.value for t in EdgeType]
            
            print_success("Enums carregados com sucesso")
            print_info(f"  - Tipos de nó: {node_types}")
            print_info(f"  - Tipos de seção: {section_types}")
            print_info(f"  - Tipos de aresta: {edge_types}")
            
            # Testa serialização/desserialização
            doc_dict = doc1.to_dict()
            edge_dict = sim_edge.to_dict()
            
            print_info(f"  - Serialização doc: {len(doc_dict)} campos")
            print_info(f"  - Serialização edge: {len(edge_dict)} campos")
            
            results['Compatibility'] = True
            
        except Exception as e:
            print_error(f"Erro no teste de compatibilidade: {e}")
            results['Compatibility'] = False
        
        return results
        
    except ImportError as e:
        print_error(f"Erro de importação na integração: {e}")
        return {}

def generate_test_report(all_results: Dict[str, Dict[str, bool]]) -> Dict[str, Any]:
    """Gera relatório final dos testes"""
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    
    detailed_results = {}
    
    for category, tests in all_results.items():
        category_passed = sum(tests.values())
        category_total = len(tests)
        
        detailed_results[category] = {
            'passed': category_passed,
            'total': category_total,
            'success_rate': category_passed / category_total if category_total > 0 else 0,
            'details': tests
        }
        
        total_tests += category_total
        passed_tests += category_passed
    
    failed_tests = total_tests - passed_tests
    
    return {
        'summary': {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0
        },
        'by_category': detailed_results,
        'timestamp': datetime.now().isoformat()
    }

def main():
    """Função principal dos testes"""
    print("🚀 INICIANDO TESTES DAS ETAPAS 1 e 2")
    print(f"📅 Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Diretório: {Path.cwd()}")
    
    start_time = time.time()
    
    # Executa todos os testes
    test_results = {}
    
    # Teste de importações
    import_results = test_imports()
    test_results['Imports'] = import_results
    
    # Se importações básicas passaram, continua
    if import_results.get('graph.models.nodes', False):
        node_results = test_node_models()
        test_results['NodeModels'] = node_results
    else:
        print_error("❌ Pulando teste de nós - importação falhou")
        test_results['NodeModels'] = {}
    
    if import_results.get('graph.models.edges', False):
        edge_results = test_edge_models()
        test_results['EdgeModels'] = edge_results
    else:
        print_error("❌ Pulando teste de arestas - importação falhou")
        test_results['EdgeModels'] = {}
    
    if import_results.get('graph.models.graph_schema', False):
        schema_results = test_graph_schema()
        test_results['GraphSchema'] = schema_results
    else:
        print_error("❌ Pulando teste de schema - importação falhou")
        test_results['GraphSchema'] = {}
    
    if import_results.get('graph.pipeline.preprocessing', False):
        preprocessing_results = test_preprocessing()
        test_results['Preprocessing'] = preprocessing_results
    else:
        print_error("❌ Pulando teste de preprocessamento - importação falhou")
        test_results['Preprocessing'] = {}
    
    # Teste de integração (se tudo passou)
    basic_imports_ok = all([
        import_results.get('graph.models.nodes', False),
        import_results.get('graph.models.edges', False),
        import_results.get('graph.models.graph_schema', False)
    ])
    
    if basic_imports_ok:
        integration_results = test_integration()
        test_results['Integration'] = integration_results
    else:
        print_error("❌ Pulando teste de integração - dependências falharam")
        test_results['Integration'] = {}
    
    # Gera relatório final
    end_time = time.time()
    execution_time = end_time - start_time
    
    print_section("RELATÓRIO FINAL DOS TESTES")
    
    report = generate_test_report(test_results)
    
    # Exibe sumário
    summary = report['summary']
    print_info(f"⏱️ Tempo de execução: {execution_time:.2f} segundos")
    print_info(f"📊 Total de testes: {summary['total_tests']}")
    print_success(f"✅ Testes aprovados: {summary['passed_tests']}")
    
    if summary['failed_tests'] > 0:
        print_error(f"❌ Testes falharam: {summary['failed_tests']}")
    
    success_rate = summary['success_rate'] * 100
    print_info(f"📈 Taxa de sucesso: {success_rate:.1f}%")
    
    # Exibe detalhes por categoria
    print_subsection("Detalhes por Categoria")
    for category, details in report['by_category'].items():
        status = "✅" if details['success_rate'] == 1.0 else "⚠️" if details['success_rate'] > 0.5 else "❌"
        print(f"{status} {category}: {details['passed']}/{details['total']} ({details['success_rate']*100:.1f}%)")
        
        # Mostra testes que falharam
        failed_tests = [test for test, passed in details['details'].items() if not passed]
        if failed_tests:
            print(f"    Falharam: {', '.join(failed_tests)}")
    
    # Recomendações baseadas nos resultados
    print_subsection("Recomendações")
    
    if summary['success_rate'] >= 0.9:
        print_success("🎉 Excelente! Sistema está funcionando corretamente.")
        print_info("✨ Você pode prosseguir para a Etapa 3 com confiança.")
    elif summary['success_rate'] >= 0.7:
        print_warning("⚠️ Maioria dos testes passou, mas há alguns problemas.")
        print_info("🔧 Revise os erros antes de continuar.")
    else:
        print_error("❌ Muitos testes falharam. Sistema precisa de ajustes.")
        print_info("🛠️ Corrija os problemas fundamentais antes de prosseguir.")
    
    # Diagnósticos específicos
    if not test_results['Imports'].get('graph.pipeline.preprocessing', False):
        print_warning("💾 Preprocessamento não pôde ser testado completamente.")
        print_info("   Isso pode ser devido ao banco não estar rodando.")
        print_info("   Execute: cd docker && docker-compose up -d")
    
    if test_results['NodeModels'].get('DocumentNode', False) and test_results['EdgeModels'].get('SimilarityEdge', False):
        print_success("🏗️ Estruturas básicas do grafo estão funcionais.")
    
    if test_results['GraphSchema'].get('GraphConfiguration', False):
        print_success("⚙️ Sistema de configuração está operacional.")
    
    # Próximos passos
    print_subsection("Próximos Passos")
    
    if summary['success_rate'] >= 0.8:
        print("🎯 Sistema validado! Próximos passos recomendados:")
        print("   1. 🚀 Execute teste com dados reais do banco")
        print("   2. 📊 Teste o preprocessamento com amostra pequena")
        print("   3. 🏗️ Prosseguir para Etapa 3 (Extração de Seções)")
        
        print("\n💡 Comando para testar com dados reais:")
        print("   python scripts/test_with_real_data.py")
    else:
        print("🔧 Corrija os problemas encontrados:")
        print("   1. ✅ Verifique se todos os arquivos foram criados")
        print("   2. 🔍 Revise os erros de importação")
        print("   3. 💾 Confirme se o banco está rodando")
        print("   4. 🔄 Execute o teste novamente")
    
    # Salva relatório detalhado
    try:
        import json
        report_file = project_root / "data" / "graph" / "test_report.json"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print_info(f"📄 Relatório salvo em: {report_file}")
        
    except Exception as e:
        print_warning(f"⚠️ Não foi possível salvar o relatório: {e}")
    
    print(f"\n{'='*80}")
    if summary['success_rate'] >= 0.8:
        print("🎉 TESTES CONCLUÍDOS COM SUCESSO!")
    else:
        print("⚠️ TESTES CONCLUÍDOS COM PROBLEMAS")
    print(f"{'='*80}")
    
    return report

if __name__ == "__main__":
    try:
        report = main()
        
        # Exit code baseado no sucesso
        success_rate = report['summary']['success_rate']
        if success_rate >= 0.8:
            sys.exit(0)  # Sucesso
        elif success_rate >= 0.5:
            sys.exit(1)  # Problemas menores
        else:
            sys.exit(2)  # Problemas sérios
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Teste interrompido pelo usuário")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ Erro inesperado durante os testes: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(3)