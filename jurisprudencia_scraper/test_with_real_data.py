#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para testar Etapas 1 e 2 com dados reais do PostgreSQL
Valida todo o pipeline com uma amostra dos seus 2854 processos
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Adiciona src ao path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def print_header():
    """Imprime cabeçalho do teste"""
    print("🏛️" + "="*78 + "🏛️")
    print("🏛️  TESTE COM DADOS REAIS - JURISPRUDÊNCIAS TJPI CONSIGNADOS  🏛️")
    print("🏛️" + "="*78 + "🏛️")
    print(f"📅 Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Objetivo: Validar Etapas 1 e 2 com dados reais do banco")
    print("-" * 80)

def test_database_connection():
    """Testa conexão com o banco PostgreSQL"""
    print("🔌 TESTANDO CONEXÃO COM BANCO...")
    
    try:
        from graph.pipeline.preprocessing import DatabaseLoader
        
        loader = DatabaseLoader()
        print("✅ Conexão estabelecida com PostgreSQL")
        
        # Testa query simples para contar processos
        query = "SELECT COUNT(*) as total FROM processos"
        import pandas as pd
        result = pd.read_sql(query, loader.engine)
        
        total_processes = result.iloc[0]['total']
        print(f"📊 Total de processos no banco: {total_processes}")
        
        # Testa query de metadados
        query_with_content = """
        SELECT COUNT(*) as total_com_conteudo 
        FROM processos 
        WHERE html_completo IS NOT NULL 
        AND LENGTH(TRIM(html_completo)) > 100
        """
        result2 = pd.read_sql(query_with_content, loader.engine)
        total_with_content = result2.iloc[0]['total_com_conteudo']
        
        print(f"📄 Processos com conteúdo válido: {total_with_content}")
        
        loader.close()
        
        return {
            'success': True,
            'total_processes': total_processes,
            'processes_with_content': total_with_content
        }
        
    except Exception as e:
        print(f"❌ Erro na conexão: {e}")
        print("\n💡 Soluções possíveis:")
        print("   1. Verifique se o Docker está rodando: docker-compose ps")
        print("   2. Inicie o banco: cd docker && docker-compose up -d")
        print("   3. Verifique as credenciais no arquivo .env")
        
        return {
            'success': False,
            'error': str(e)
        }

def test_sample_preprocessing(sample_size: int = 10):
    """Testa preprocessamento com amostra pequena"""
    print(f"\n🔄 TESTANDO PREPROCESSAMENTO COM {sample_size} DOCUMENTOS...")
    
    try:
        from graph.pipeline.preprocessing import preprocess_jurisprudencias
        
        start_time = time.time()
        
        # Processa amostra
        documents, stats = preprocess_jurisprudencias(limit=sample_size)
        
        processing_time = time.time() - start_time
        
        print(f"✅ Preprocessamento concluído em {processing_time:.2f}s")
        print(f"📈 Estatísticas:")
        print(f"   - Total carregado: {stats.total_documents}")
        print(f"   - Documentos válidos: {stats.valid_documents}")
        print(f"   - Com conteúdo: {stats.documents_with_content}")
        print(f"   - Com metadados: {stats.documents_with_metadata}")
        print(f"   - Com seções: {stats.documents_with_sections}")
        
        if documents:
            # Analisa primeiro documento
            first_doc = documents[0]
            print(f"\n📋 Exemplo - Primeiro Documento:")
            print(f"   - ID: {first_doc.id}")
            print(f"   - Processo: {first_doc.numero_processo}")
            print(f"   - Órgão: {first_doc.orgao_julgador}")
            print(f"   - Relator: {first_doc.relator}")
            print(f"   - Tokens: {first_doc.num_tokens}")
            print(f"   - Seções: {first_doc.num_secoes}")
            
            if first_doc.conteudo_limpo:
                content_preview = first_doc.conteudo_limpo[:200] + "..."
                print(f"   - Conteúdo: {content_preview}")
        
        # Estatísticas de limpeza
        print(f"\n🧹 Limpeza de Dados:")
        print(f"   - HTML tags removidas: {stats.html_tags_removed}")
        print(f"   - Problemas encoding: {stats.encoding_issues_fixed}")
        print(f"   - Docs vazios removidos: {stats.empty_documents_removed}")
        print(f"   - Total caracteres: {stats.total_characters:,}")
        print(f"   - Total palavras: {stats.total_words:,}")
        print(f"   - Média caracteres/doc: {stats.average_document_length:.0f}")
        
        return {
            'success': True,
            'documents': len(documents),
            'stats': stats.to_dict(),
            'processing_time': processing_time,
            'sample_document': first_doc.to_dict() if documents else None
        }
        
    except Exception as e:
        print(f"❌ Erro no preprocessamento: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'success': False,
            'error': str(e)
        }

def test_document_node_creation():
    """Testa criação de nós de documento com dados reais"""
    print(f"\n🏗️ TESTANDO CRIAÇÃO DE NÓS COM DADOS REAIS...")
    
    try:
        documents, _ = preprocess_jurisprudencias(limit=5)
        
        if not documents:
            print("❌ Nenhum documento carregado")
            return {'success': False, 'error': 'No documents loaded'}
        
        print(f"✅ {len(documents)} DocumentNodes criados")
        
        # Analisa tipos de dados encontrados
        orgaos = set()
        relatores = set()
        classes = set()
        
        for doc in documents:
            if doc.orgao_julgador:
                orgaos.add(doc.orgao_julgador)
            if doc.relator:
                relatores.add(doc.relator)
            if doc.classe_judicial:
                classes.add(doc.classe_judicial)
        
        print(f"📊 Diversidade encontrada:")
        print(f"   - Órgãos julgadores: {len(orgaos)} únicos")
        print(f"   - Relatores: {len(relatores)} únicos")  
        print(f"   - Classes judiciais: {len(classes)} únicos")
        
        if orgaos:
            print(f"   - Exemplos órgãos: {list(orgaos)[:3]}")
        if relatores:
            print(f"   - Exemplos relatores: {list(relatores)[:3]}")
        
        # Testa validação dos nós
        from graph.models.graph_schema import create_default_schema
        
        schema = create_default_schema()
        validation_errors = 0
        
        for doc in documents:
            errors = schema.validate_node(doc)
            validation_errors += len(errors)
            if errors:
                print(f"⚠️ Erros no documento {doc.id}: {errors}")
        
        print(f"🔍 Validação: {validation_errors} erros encontrados")
        
        return {
            'success': True,
            'documents_created': len(documents),
            'unique_orgaos': len(orgaos),
            'unique_relatores': len(relatores),
            'unique_classes': len(classes),
            'validation_errors': validation_errors,
            'sample_orgaos': list(orgaos)[:5],
            'sample_relatores': list(relatores)[:5]
        }
        
    except Exception as e:
        print(f"❌ Erro na criação de nós: {e}")
        return {'success': False, 'error': str(e)}

def test_text_cleaning_quality():
    """Testa qualidade da limpeza de texto"""
    print(f"\n🧽 TESTANDO QUALIDADE DA LIMPEZA DE TEXTO...")
    
    try:
        from graph.pipeline.preprocessing import DatabaseLoader
        
        loader = DatabaseLoader()
        
        # Pega amostra de HTML bruto
        query = """
        SELECT numero_processo, html_completo 
        FROM processos 
        WHERE html_completo IS NOT NULL 
        AND LENGTH(html_completo) > 500
        LIMIT 3
        """
        
        import pandas as pd
        df = pd.read_sql(query, loader.engine)
        loader.close()
        
        if df.empty:
            print("❌ Nenhum HTML encontrado para testar")
            return {'success': False, 'error': 'No HTML content'}
        
        from graph.pipeline.preprocessing import TextCleaner
        cleaner = TextCleaner()
        
        results = []
        
        for idx, row in df.iterrows():
            html_content = row['html_completo']
            clean_text = cleaner.extract_clean_text(html_content)
            
            is_valid = cleaner.is_valid_document(clean_text)
            
            result = {
                'processo': row['numero_processo'],
                'html_length': len(html_content),
                'clean_length': len(clean_text) if clean_text else 0,
                'is_valid': is_valid,
                'reduction_ratio': (len(html_content) - len(clean_text or "")) / len(html_content)
            }
            results.append(result)
            
            print(f"📄 Processo: {row['numero_processo']}")
            print(f"   - HTML: {len(html_content):,} chars")
            print(f"   - Limpo: {len(clean_text or ''):,} chars") 
            print(f"   - Redução: {result['reduction_ratio']:.1%}")
            print(f"   - Válido: {'✅' if is_valid else '❌'}")
            
            if clean_text and len(clean_text) > 100:
                preview = clean_text[:150].replace('\n', ' ')
                print(f"   - Preview: {preview}...")
        
        # Estatísticas gerais
        valid_docs = sum(1 for r in results if r['is_valid'])
        avg_reduction = sum(r['reduction_ratio'] for r in results) / len(results)
        
        print(f"\n📊 Resumo da Limpeza:")
        print(f"   - Documentos válidos: {valid_docs}/{len(results)}")
        print(f"   - Redução média: {avg_reduction:.1%}")
        
        return {
            'success': True,
            'documents_tested': len(results),
            'valid_documents': valid_docs,
            'average_reduction': avg_reduction,
            'results': results
        }
        
    except Exception as e:
        print(f"❌ Erro no teste de limpeza: {e}")
        return {'success': False, 'error': str(e)}

def test_section_detection():
    """Testa detecção de seções nos documentos"""
    print(f"\n📑 TESTANDO DETECÇÃO DE SEÇÕES...")
    
    try:
        from graph.pipeline.preprocessing import DatabaseLoader
        
        loader = DatabaseLoader()
        
        # Query para pegar documentos com seções
        query = """
        SELECT p.numero_processo, COUNT(pc.id) as num_secoes,
               STRING_AGG(DISTINCT pc.tipo_secao, ', ') as tipos_secao
        FROM processos p
        JOIN processos_conteudo pc ON p.id = pc.processo_id
        WHERE pc.conteudo_texto IS NOT NULL
        GROUP BY p.numero_processo
        ORDER BY num_secoes DESC
        LIMIT 10
        """
        
        import pandas as pd
        df = pd.read_sql(query, loader.engine)
        
        if df.empty:
            print("❌ Nenhuma seção encontrada no banco")
            loader.close()
            return {'success': False, 'error': 'No sections found'}
        
        print(f"✅ Encontrados {len(df)} documentos com seções")
        
        section_types = {}
        total_sections = 0
        
        for idx, row in df.iterrows():
            num_secoes = row['num_secoes']
            tipos = row['tipos_secao'].split(', ') if row['tipos_secao'] else []
            
            total_sections += num_secoes
            
            print(f"📋 {row['numero_processo']}: {num_secoes} seções ({row['tipos_secao']})")
            
            for tipo in tipos:
                section_types[tipo] = section_types.get(tipo, 0) + 1
        
        print(f"\n📊 Tipos de Seção Encontrados:")
        for tipo, count in sorted(section_types.items(), key=lambda x: x[1], reverse=True):
            print(f"   - {tipo}: {count} documentos")
        
        avg_sections = total_sections / len(df)
        print(f"\n📈 Média de seções por documento: {avg_sections:.1f}")
        
        loader.close()
        
        return {
            'success': True,
            'documents_with_sections': len(df),
            'total_sections': total_sections,
            'average_sections': avg_sections,
            'section_types': section_types
        }
        
    except Exception as e:
        print(f"❌ Erro na detecção de seções: {e}")
        return {'success': False, 'error': str(e)}

def generate_comprehensive_report(results: Dict[str, Any]) -> Dict[str, Any]:
    """Gera relatório abrangente dos testes com dados reais"""
    
    # Calcula sucesso geral
    successful_tests = sum(1 for test_result in results.values() 
                          if isinstance(test_result, dict) and test_result.get('success', False))
    total_tests = len([r for r in results.values() if isinstance(r, dict)])
    
    success_rate = successful_tests / total_tests if total_tests > 0 else 0
    
    # Extrai métricas importantes
    database_info = results.get('database_connection', {})
    preprocessing_info = results.get('sample_preprocessing', {})
    
    report = {
        'test_summary': {
            'success_rate': success_rate,
            'successful_tests': successful_tests,
            'total_tests': total_tests,
            'timestamp': datetime.now().isoformat()
        },
        'database_status': {
            'connected': database_info.get('success', False),
            'total_processes': database_info.get('total_processes', 0),
            'processes_with_content': database_info.get('processes_with_content', 0)
        },
        'preprocessing_performance': {
            'successful': preprocessing_info.get('success', False),
            'documents_processed': preprocessing_info.get('documents', 0),
            'processing_time': preprocessing_info.get('processing_time', 0)
        },
        'data_quality': {
            'node_creation': results.get('document_nodes', {}).get('success', False),
            'text_cleaning': results.get('text_cleaning', {}).get('success', False),
            'section_detection': results.get('section_detection', {}).get('success', False)
        },
        'detailed_results': results
    }
    
    return report

def main():
    """Função principal do teste com dados reais"""
    
    print_header()
    
    start_time = time.time()
    results = {}
    
    # Teste 1: Conexão com banco
    db_result = test_database_connection()
    results['database_connection'] = db_result
    
    if not db_result['success']:
        print("\n❌ Não é possível continuar sem conexão com o banco")
        print("🔧 Corrija a conexão e execute novamente")
        return results
    
    # Teste 2: Preprocessamento com amostra
    results['sample_preprocessing'] = test_sample_preprocessing(10)
    
    # Teste 3: Criação de nós
    results['document_nodes'] = test_document_node_creation()
    
    # Teste 4: Qualidade da limpeza
    results['text_cleaning'] = test_text_cleaning_quality()
    
    # Teste 5: Detecção de seções
    results['section_detection'] = test_section_detection()
    
    # Relatório final
    execution_time = time.time() - start_time
    
    print(f"\n" + "="*80)
    print("📊 RELATÓRIO FINAL - TESTE COM DADOS REAIS")
    print("="*80)
    
    report = generate_comprehensive_report(results)
    
    # Status geral
    success_rate = report['test_summary']['success_rate']
    print(f"⏱️ Tempo total: {execution_time:.2f}s")
    print(f"📈 Taxa de sucesso: {success_rate:.1%}")
    
    # Status do banco
    db_status = report['database_status']
    if db_status['connected']:
        print(f"💾 Banco: ✅ Conectado ({db_status['total_processes']} processos)")
        print(f"   📄 Com conteúdo: {db_status['processes_with_content']}")
    else:
        print(f"💾 Banco: ❌ Não conectado")
    
    # Status do processamento
    proc_status = report['preprocessing_performance']
    if proc_status['successful']:
        print(f"🔄 Preprocessamento: ✅ ({proc_status['documents_processed']} docs em {proc_status['processing_time']:.2f}s)")
    else:
        print(f"🔄 Preprocessamento: ❌ Falhou")
    
    # Qualidade dos dados
    quality = report['data_quality']
    quality_score = sum(quality.values()) / len(quality) if quality else 0
    print(f"🏗️ Qualidade dos dados: {quality_score:.1%}")
    print(f"   - Criação de nós: {'✅' if quality['node_creation'] else '❌'}")
    print(f"   - Limpeza de texto: {'✅' if quality['text_cleaning'] else '❌'}")
    print(f"   - Detecção seções: {'✅' if quality['section_detection'] else '❌'}")
    
    # Recomendações
    print(f"\n💡 RECOMENDAÇÕES:")
    
    if success_rate >= 0.8:
        print("🎉 Sistema validado com dados reais!")
        print("✨ Pronto para prosseguir com o desenvolvimento")
        print("🚀 Próximo passo: Implementar Etapa 3 (Extração de Seções)")
        
        if db_status['processes_with_content'] >= 1000:
            print(f"📊 Corpus excelente: {db_status['processes_with_content']} documentos")
        elif db_status['processes_with_content'] >= 100:
            print(f"📊 Corpus adequado: {db_status['processes_with_content']} documentos")
        
    else:
        print("⚠️ Alguns problemas encontrados:")
        
        if not db_status['connected']:
            print("   🔧 Corrija conexão com banco")
        
        if not proc_status['successful']:
            print("   🔧 Revise pipeline de preprocessamento")
        
        if quality_score < 0.7:
            print("   🔧 Melhore qualidade dos dados")
    
    # Salva relatório
    try:
        import json
        report_file = project_root / "data" / "graph" / "real_data_test_report.json"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"📄 Relatório detalhado salvo em: {report_file}")
        
    except Exception as e:
        print(f"⚠️ Erro ao salvar relatório: {e}")
    
    # Status final
    print(f"\n{'='*80}")
    if success_rate >= 0.8:
        print("🎯 TESTE COM DADOS REAIS: SUCESSO!")
        print("🏛️ Sistema validado com jurisprudências reais do TJPI")
    elif success_rate >= 0.5:
        print("⚠️ TESTE COM DADOS REAIS: PARCIALMENTE BEM-SUCEDIDO")
        print("🔧 Alguns ajustes necessários")
    else:
        print("❌ TESTE COM DADOS REAIS: NECESSITA CORREÇÕES")
        print("🛠️ Revise os problemas antes de continuar")
    print(f"{'='*80}")
    
    return report

if __name__ == "__main__":
    try:
        # Verifica se está no diretório correto
        if not (Path.cwd() / "src" / "graph").exists():
            print("❌ Execute este script a partir da raiz do projeto jurisprudencia_scraper/")
            sys.exit(1)
        
        report = main()
        
        # Exit code baseado no resultado
        success_rate = report['test_summary']['success_rate']
        if success_rate >= 0.8:
            sys.exit(0)  # Sucesso completo
        elif success_rate >= 0.5:
            sys.exit(1)  # Sucesso parcial
        else:
            sys.exit(2)  # Falha
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Teste interrompido pelo usuário")
        sys.exit(130)
    except ImportError as e:
        print(f"\n❌ Erro de importação: {e}")
        print("💡 Execute primeiro: python test_graph_development.py")
        sys.exit(3)
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(4)