# -*- coding: utf-8 -*-
"""
Pipeline de pré-processamento para construção do grafo heterogêneo
Carrega dados do PostgreSQL e prepara para extração de características
"""

import re
import html
import unicodedata
from typing import Dict, List, Optional, Tuple, Any
from graph.models.nodes import DocumentNode, create_document_node_from_db_row, NodeType  
from datetime import datetime
import pandas as pd
import numpy as np
from dataclasses import dataclass
from loguru import logger
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords

# Importa configurações existentes
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from config.settings import DATABASE_URL, DB_CONFIG
    from graph.models.nodes import DocumentNode, create_document_node_from_db_row
except ImportError as e:
    logger.warning(f"Erro ao importar configurações: {e}")
    # Fallback para configuração padrão
    DATABASE_URL = "postgresql://postgres:admin@localhost:5432/jurisprudencia_db"


@dataclass
class PreprocessingStats:
    """Estatísticas do preprocessamento"""
    
    total_documents: int = 0
    valid_documents: int = 0
    documents_with_content: int = 0
    documents_with_metadata: int = 0
    documents_with_sections: int = 0
    
    # Estatísticas de limpeza
    html_tags_removed: int = 0
    encoding_issues_fixed: int = 0
    empty_documents_removed: int = 0
    duplicate_documents_removed: int = 0
    
    # Estatísticas de texto
    total_characters: int = 0
    total_words: int = 0
    average_document_length: float = 0.0
    
    # Distribuições
    document_lengths: List[int] = None
    section_types_found: Dict[str, int] = None
    
    processing_time: Optional[float] = None
    
    def __post_init__(self):
        if self.document_lengths is None:
            self.document_lengths = []
        if self.section_types_found is None:
            self.section_types_found = {}
    
    def update_averages(self):
        """Atualiza estatísticas calculadas"""
        if self.valid_documents > 0:
            self.average_document_length = self.total_characters / self.valid_documents
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte estatísticas para dicionário"""
        return {
            'documents': {
                'total': self.total_documents,
                'valid': self.valid_documents,
                'with_content': self.documents_with_content,
                'with_metadata': self.documents_with_metadata,
                'with_sections': self.documents_with_sections
            },
            'cleaning': {
                'html_tags_removed': self.html_tags_removed,
                'encoding_fixed': self.encoding_issues_fixed,
                'empty_removed': self.empty_documents_removed,
                'duplicates_removed': self.duplicate_documents_removed
            },
            'text_stats': {
                'total_characters': self.total_characters,
                'total_words': self.total_words,
                'average_length': self.average_document_length
            },
            'distributions': {
                'document_lengths': self.document_lengths,
                'section_types': self.section_types_found
            },
            'processing_time': self.processing_time
        }


class DatabaseLoader:
    """Carregador de dados do PostgreSQL"""
    
    def __init__(self, database_url: str = None):
        self.database_url = database_url or DATABASE_URL
        self.engine = None
        self.Session = None
        
        self._setup_database()
    
    def _setup_database(self):
        """Configura conexão com o banco"""
        try:
            self.engine = sa.create_engine(
                self.database_url,
                echo=False,
                pool_pre_ping=True,
                pool_recycle=3600
            )
            self.Session = sessionmaker(bind=self.engine)
            logger.info("✅ Conexão com PostgreSQL estabelecida")
        except Exception as e:
            logger.error(f"❌ Erro ao conectar com PostgreSQL: {e}")
            raise
    
    def load_jurisprudencias(self, limit: Optional[int] = None,
                           offset: int = 0,
                           include_empty: bool = False) -> pd.DataFrame:
        """
        Carrega jurisprudências do banco de dados
        
        Args:
            limit: Limitar número de documentos
            offset: Offset para paginação
            include_empty: Incluir documentos sem conteúdo
            
        Returns:
            DataFrame com os dados dos processos
        """
        logger.info(f"🔍 Carregando jurisprudências do banco (limit={limit}, offset={offset})")
        
        # Query principal juntando as três tabelas
        query = """
        SELECT 
            p.id,
            p.numero_processo,
            p.url_original,
            p.html_completo,
            p.data_coleta,
            p.status_processamento,
            p.hash_documento,
            p.created_at,
            p.updated_at,
            
            -- Metadados
            pm.orgao_julgador,
            pm.orgao_julgador_colegiado,
            pm.relator,
            pm.classe_judicial,
            pm.competencia,
            pm.assunto_principal,
            pm.autor,
            pm.reu,
            pm.data_publicacao,
            pm.tipo_decisao,
            
            -- Contagem de seções
            COUNT(pc.id) as num_secoes
            
        FROM processos p
        LEFT JOIN processos_metadados pm ON p.id = pm.processo_id
        LEFT JOIN processos_conteudo pc ON p.id = pc.processo_id
        """
        
        # Filtros
        where_conditions = []
        
        if not include_empty:
            where_conditions.append("p.html_completo IS NOT NULL")
            where_conditions.append("LENGTH(TRIM(p.html_completo)) > 100")
        
        # Só processos com status processado ou pendente
        where_conditions.append("p.status_processamento IN ('processado', 'pendente')")
        
        if where_conditions:
            query += " WHERE " + " AND ".join(where_conditions)
        
        query += """
        GROUP BY 
            p.id, p.numero_processo, p.url_original, p.html_completo, 
            p.data_coleta, p.status_processamento, p.hash_documento,
            p.created_at, p.updated_at,
            pm.orgao_julgador, pm.orgao_julgador_colegiado, pm.relator,
            pm.classe_judicial, pm.competencia, pm.assunto_principal,
            pm.autor, pm.reu, pm.data_publicacao, pm.tipo_decisao
        ORDER BY p.created_at DESC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        if offset > 0:
            query += f" OFFSET {offset}"
        
        try:
            df = pd.read_sql(query, self.engine)
            logger.info(f"✅ {len(df)} processos carregados do banco")
            return df
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar dados: {e}")
            raise
    
    def load_sections_for_documents(self, document_ids: List[int]) -> pd.DataFrame:
        """
        Carrega seções para documentos específicos
        
        Args:
            document_ids: Lista de IDs dos documentos
            
        Returns:
            DataFrame com seções
        """
        if not document_ids:
            return pd.DataFrame()
        
        # Converte para string para usar na query
        ids_str = ','.join(map(str, document_ids))
        
        query = f"""
        SELECT 
            id,
            processo_id,
            tipo_secao,
            conteudo_html,
            conteudo_texto,
            conteudo_limpo,
            ordem,
            created_at
        FROM processos_conteudo 
        WHERE processo_id IN ({ids_str})
        ORDER BY processo_id, ordem
        """
        
        try:
            df = pd.read_sql(query, self.engine)
            logger.info(f"✅ {len(df)} seções carregadas para {len(document_ids)} documentos")
            return df
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar seções: {e}")
            return pd.DataFrame()
    
    def close(self):
        """Fecha conexão com banco"""
        if self.engine:
            self.engine.dispose()
            logger.info("🔧 Conexão com banco fechada")


class TextCleaner:
    """Limpador e normalizador de texto"""
    
    def __init__(self):
        self.stats = PreprocessingStats()
        self._setup_nltk()
        self._setup_stopwords()
    
    def _setup_nltk(self):
        """Baixa recursos NLTK necessários"""
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            logger.info("📥 Baixando stopwords do NLTK...")
            nltk.download('stopwords', quiet=True)
        
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            logger.info("📥 Baixando tokenizer do NLTK...")
            nltk.download('punkt', quiet=True)
    
    def _setup_stopwords(self):
        """Configura stopwords em português"""
        try:
            self.stopwords = set(stopwords.words('portuguese'))
            
            # Adiciona stopwords jurídicas específicas
            juridical_stopwords = {
                'art', 'artigo', 'lei', 'código', 'inc', 'inciso', 'parágrafo',
                'alínea', 'item', 'cf', 'cfr', 'vide', 'ver', 'pág', 'página',
                'fls', 'folhas', 'proc', 'processo', 'autos', 'ref', 'referente',
                'rel', 'relator', 'rev', 'revisor', 'red', 'redator', 'des',
                'desembargador', 'min', 'ministro', 'dr', 'dra', 'advogado',
                'advogada', 'mm', 'meritíssimo', 'dd', 'digníssimo'
            }
            self.stopwords.update(juridical_stopwords)
            
            logger.info(f"✅ Stopwords configuradas: {len(self.stopwords)} termos")
            
        except Exception as e:
            logger.warning(f"⚠️ Erro ao configurar stopwords: {e}")
            self.stopwords = set()
    
    def clean_html(self, html_content: str) -> str:
        """
        Remove tags HTML e extrai texto limpo
        
        Args:
            html_content: Conteúdo HTML
            
        Returns:
            Texto limpo
        """
        if not html_content or not isinstance(html_content, str):
            return ""
        
        try:
            # Decode HTML entities
            text = html.unescape(html_content)
            self.stats.encoding_issues_fixed += 1
            
            # Parse HTML com BeautifulSoup
            soup = BeautifulSoup(text, 'lxml')
            
            # Remove scripts e styles
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extrai texto
            text = soup.get_text()
            self.stats.html_tags_removed += 1
            
            return text
            
        except Exception as e:
            logger.warning(f"⚠️ Erro na limpeza HTML: {e}")
            return html_content
    
    def normalize_text(self, text: str) -> str:
        """
        Normaliza texto removendo caracteres especiais e normalizando encoding
        
        Args:
            text: Texto para normalizar
            
        Returns:
            Texto normalizado
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Normaliza unicode (remove acentos de caracteres especiais)
        text = unicodedata.normalize('NFKD', text)
        
        # Remove caracteres de controle
        text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C')
        
        # Normaliza espaços em branco
        text = re.sub(r'\s+', ' ', text)
        
        # Remove espaços no início e fim
        text = text.strip()
        
        return text
    
    def extract_clean_text(self, html_content: str, 
                          min_length: int = 50) -> Optional[str]:
        """
        Pipeline completa de limpeza: HTML -> texto limpo
        
        Args:
            html_content: Conteúdo HTML original
            min_length: Tamanho mínimo do texto válido
            
        Returns:
            Texto limpo ou None se inválido
        """
        # Etapa 1: Remove HTML
        text = self.clean_html(html_content)
        
        # Etapa 2: Normaliza texto
        text = self.normalize_text(text)
        
        # Etapa 3: Valida tamanho mínimo
        if len(text) < min_length:
            return None
        
        # Atualiza estatísticas
        self.stats.total_characters += len(text)
        self.stats.total_words += len(text.split())
        
        return text
    
    def is_valid_document(self, text: str, 
                         min_words: int = 20,
                         min_chars: int = 100) -> bool:
        """
        Valida se um documento tem conteúdo suficiente
        
        Args:
            text: Texto do documento
            min_words: Mínimo de palavras
            min_chars: Mínimo de caracteres
            
        Returns:
            True se documento é válido
        """
        if not text or not isinstance(text, str):
            return False
        
        word_count = len(text.split())
        char_count = len(text.strip())
        
        # Verifica critérios mínimos
        if char_count < min_chars or word_count < min_words:
            return False
        
        # Verifica se não é só HTML/tags
        if text.count('<') > text.count(' ') / 10:
            return False
        
        # Verifica se tem conteúdo semântico mínimo
        meaningful_chars = sum(1 for c in text if c.isalnum())
        if meaningful_chars < char_count * 0.7:
            return False
        
        return True


class JurisprudenciaPreprocessor:
    """Preprocessador principal para jurisprudências"""
    
    def __init__(self, database_url: str = None):
        self.db_loader = DatabaseLoader(database_url)
        self.text_cleaner = TextCleaner()
        self.stats = PreprocessingStats()
        
        logger.info("🚀 JurisprudenciaPreprocessor inicializado")
    
    def load_and_clean_documents(self, 
                                limit: Optional[int] = None,
                                offset: int = 0,
                                min_doc_length: int = 100,
                                include_sections: bool = True) -> Tuple[List[DocumentNode], PreprocessingStats]:
        """
        Pipeline principal: carrega e limpa documentos do banco
        
        Args:
            limit: Número máximo de documentos
            offset: Offset para paginação  
            min_doc_length: Tamanho mínimo do documento
            include_sections: Se deve carregar seções também
            
        Returns:
            Tupla (lista de DocumentNodes, estatísticas)
        """
        start_time = datetime.now()
        logger.info(f"🔄 Iniciando preprocessamento (limit={limit})")
        
        # Etapa 1: Carrega dados do banco
        df = self.db_loader.load_jurisprudencias(
            limit=limit, 
            offset=offset, 
            include_empty=False
        )
        
        if df.empty:
            logger.warning("⚠️ Nenhum documento encontrado no banco")
            return [], self.stats
        
        self.stats.total_documents = len(df)
        logger.info(f"📊 {self.stats.total_documents} documentos carregados")
        
        # Etapa 2: Processa cada documento
        valid_documents = []
        document_lengths = []
        
        for idx, row in df.iterrows():
            try:
                # Cria nó do documento
                doc_node = self._process_document_row(row, min_doc_length)
                
                if doc_node:
                    valid_documents.append(doc_node)
                    document_lengths.append(len(doc_node.conteudo_limpo or ""))
                    self.stats.valid_documents += 1
                    
                    if doc_node.conteudo_limpo:
                        self.stats.documents_with_content += 1
                    
                    if doc_node.orgao_julgador or doc_node.relator:
                        self.stats.documents_with_metadata += 1
                else:
                    self.stats.empty_documents_removed += 1
                    
            except Exception as e:
                logger.error(f"❌ Erro ao processar documento {row.get('numero_processo')}: {e}")
                continue
        
        # Etapa 3: Carrega seções se solicitado
        if include_sections and valid_documents:
            self._load_sections_for_documents(valid_documents, df)
        
        # Etapa 4: Calcula estatísticas finais
        self.stats.document_lengths = document_lengths
        self.stats.update_averages()
        
        processing_time = (datetime.now() - start_time).total_seconds()
        self.stats.processing_time = processing_time
        
        logger.info(f"✅ Preprocessamento concluído em {processing_time:.2f}s")
        logger.info(f"📈 {self.stats.valid_documents}/{self.stats.total_documents} documentos válidos")
        
        return valid_documents, self.stats
    
    def _process_document_row(self, row: pd.Series, 
                            min_length: int) -> Optional[DocumentNode]:
        """
        Processa uma linha do DataFrame em DocumentNode
        
        Args:
            row: Linha do DataFrame
            min_length: Tamanho mínimo do documento
            
        Returns:
            DocumentNode ou None se inválido
        """
        # Extrai e limpa o conteúdo HTML
        html_content = row.get('html_completo', '')
        clean_text = self.text_cleaner.extract_clean_text(html_content, min_length)
        
        if not clean_text or not self.text_cleaner.is_valid_document(clean_text):
            return None
        
        # Cria DocumentNode
        doc_node = DocumentNode(
            id=f"doc_{row['numero_processo']}",
            node_type=NodeType.DOCUMENT, 
            numero_processo=row['numero_processo'],
            url_original=row['url_original'],
            label=f"Processo {row['numero_processo']}",
            
            # Metadados
            orgao_julgador=row.get('orgao_julgador'),
            orgao_julgador_colegiado=row.get('orgao_julgador_colegiado'),
            relator=row.get('relator'),
            classe_judicial=row.get('classe_judicial'),
            competencia=row.get('competencia'),
            assunto_principal=row.get('assunto_principal'),
            autor=row.get('autor'),
            reu=row.get('reu'),
            data_publicacao=row.get('data_publicacao'),
            tipo_decisao=row.get('tipo_decisao'),
            
            # Conteúdo
            conteudo_completo=html_content,
            conteudo_limpo=clean_text,
            hash_documento=row.get('hash_documento'),
            
            # Métricas
            num_tokens=len(clean_text.split()),
            num_secoes=row.get('num_secoes', 0),
            
        )
        
        return doc_node
    
    def _load_sections_for_documents(self, documents: List[DocumentNode], 
                                   main_df: pd.DataFrame):
        """
        Carrega seções para os documentos válidos
        
        Args:
            documents: Lista de DocumentNodes
            main_df: DataFrame principal com IDs
        """
        # Mapeia numero_processo -> id do banco
        process_to_id = {}
        for _, row in main_df.iterrows():
            process_to_id[row['numero_processo']] = row['id']
        
        # Pega IDs dos documentos válidos
        document_ids = []
        for doc in documents:
            db_id = process_to_id.get(doc.numero_processo)
            if db_id:
                document_ids.append(db_id)
        
        if not document_ids:
            return
        
        # Carrega seções
        sections_df = self.db_loader.load_sections_for_documents(document_ids)
        
        if sections_df.empty:
            return
        
        # Agrupa seções por processo
        sections_by_process = {}
        for _, section_row in sections_df.iterrows():
            processo_id = section_row['processo_id']
            if processo_id not in sections_by_process:
                sections_by_process[processo_id] = []
            sections_by_process[processo_id].append(section_row)
        
        # Atualiza DocumentNodes com informações das seções
        id_to_process = {v: k for k, v in process_to_id.items()}
        
        for doc in documents:
            db_id = process_to_id.get(doc.numero_processo)
            if db_id and db_id in sections_by_process:
                sections = sections_by_process[db_id]
                doc.num_secoes = len(sections)
                
                # Contabiliza tipos de seção
                for section in sections:
                    section_type = section.get('tipo_secao', 'outros')
                    self.stats.section_types_found[section_type] = \
                        self.stats.section_types_found.get(section_type, 0) + 1
        
        # Atualiza estatística
        self.stats.documents_with_sections = sum(
            1 for doc in documents if doc.num_secoes > 0
        )
        
        logger.info(f"📑 Seções carregadas para {self.stats.documents_with_sections} documentos")
    
    def validate_preprocessed_data(self, documents: List[DocumentNode]) -> Dict[str, Any]:
        """
        Valida os dados preprocessados
        
        Args:
            documents: Lista de DocumentNodes
            
        Returns:
            Dicionário com resultado da validação
        """
        validation = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'stats': {}
        }
        
        if not documents:
            validation['is_valid'] = False
            validation['errors'].append("Nenhum documento válido encontrado")
            return validation
        
        # Validações básicas
        docs_with_content = sum(1 for doc in documents if doc.conteudo_limpo)
        docs_with_metadata = sum(1 for doc in documents if doc.orgao_julgador or doc.relator)
        
        if docs_with_content < len(documents) * 0.9:
            validation['warnings'].append("Alguns documentos sem conteúdo limpo")
        
        if docs_with_metadata < len(documents) * 0.5:
            validation['warnings'].append("Poucos documentos com metadados")
        
        # Estatísticas de validação
        lengths = [len(doc.conteudo_limpo or "") for doc in documents]
        validation['stats'] = {
            'total_documents': len(documents),
            'with_content': docs_with_content,
            'with_metadata': docs_with_metadata,
            'avg_length': np.mean(lengths) if lengths else 0,
            'min_length': min(lengths) if lengths else 0,
            'max_length': max(lengths) if lengths else 0
        }
        
        return validation
    
    def close(self):
        """Fecha recursos"""
        self.db_loader.close()


# Funções auxiliares para uso do preprocessor

def preprocess_jurisprudencias(limit: Optional[int] = None,
                              offset: int = 0,
                              database_url: str = None) -> Tuple[List[DocumentNode], PreprocessingStats]:
    """
    Função principal para preprocessar jurisprudências
    
    Args:
        limit: Número máximo de documentos
        offset: Offset para paginação
        database_url: URL do banco de dados
        
    Returns:
        Tupla (documentos, estatísticas)
    """
    preprocessor = JurisprudenciaPreprocessor(database_url)
    
    try:
        documents, stats = preprocessor.load_and_clean_documents(
            limit=limit,
            offset=offset
        )
        
        # Validação final
        validation = preprocessor.validate_preprocessed_data(documents)
        
        if not validation['is_valid']:
            logger.error(f"❌ Validação falhou: {validation['errors']}")
        
        if validation['warnings']:
            for warning in validation['warnings']:
                logger.warning(f"⚠️ {warning}")
        
        return documents, stats
        
    finally:
        preprocessor.close()


def get_sample_documents(num_samples: int = 10,
                        database_url: str = None) -> List[DocumentNode]:
    """
    Carrega uma amostra de documentos para testes
    
    Args:
        num_samples: Número de amostras
        database_url: URL do banco
        
    Returns:
        Lista de DocumentNodes
    """
    documents, _ = preprocess_jurisprudencias(
        limit=num_samples,
        database_url=database_url
    )
    
    return documents