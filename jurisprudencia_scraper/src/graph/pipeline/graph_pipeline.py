# -*- coding: utf-8 -*-
"""
Pipeline principal para construção do grafo heterogêneo de jurisprudências
Orquestra todo o processo das Etapas 1-2: dados -> grafo completo
"""

import time
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from loguru import logger
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Imports dos módulos do grafo
from .preprocessing import preprocess_jurisprudencias, PreprocessingStats
from ..models.nodes import DocumentNode, SectionNode, EntityNode, ConceptNode
from ..models.edges import SimilarityEdge, RelevanceEdge, CooccurrenceEdge, HierarchicalEdge
from ..models.graph_schema import GraphConfiguration, GraphStatistics, GraphSchema
from ..extractors.section_extractor import SectionExtractor
from ..extractors.ner_extractor import NERExtractor  
from ..extractors.concept_extractor import ConceptExtractor
from ..processors.text_vectorizer import TextVectorizer
from ..processors.pmi_calculator import PMICalculator


class GraphConstructionResult:
    """Resultado da construção do grafo"""
    
    def __init__(self):
        self.success = False
        self.graph: Optional[nx.Graph] = None
        self.statistics: Optional[GraphStatistics] = None
        self.preprocessing_stats: Optional[PreprocessingStats] = None
        
        # Contadores detalhados
        self.nodes_created = {
            'documents': 0,
            'sections': 0, 
            'entities': 0,
            'concepts': 0
        }
        
        self.edges_created = {
            'similarity': 0,
            'relevance': 0,
            'cooccurrence': 0,
            'hierarchical': 0
        }
        
        # Tempos de execução por etapa
        self.execution_times = {}
        
        # Logs e erros
        self.logs = []
        self.errors = []
        
        # Arquivos gerados
        self.output_files = []
    
    def add_log(self, message: str, level: str = "INFO"):
        """Adiciona log ao resultado"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_entry = f"[{timestamp}] {level}: {message}"
        self.logs.append(log_entry)
        
        if level == "ERROR":
            self.errors.append(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte resultado para dicionário"""
        return {
            'success': self.success,
            'nodes_created': self.nodes_created,
            'edges_created': self.edges_created,
            'execution_times': self.execution_times,
            'statistics': self.statistics.to_dict() if self.statistics else None,
            'preprocessing_stats': self.preprocessing_stats.to_dict() if self.preprocessing_stats else None,
            'output_files': self.output_files,
            'logs': self.logs,
            'errors': self.errors
        }


class GraphPipeline:
    """Pipeline principal para construção do grafo heterogêneo"""
    
    def __init__(self, config: Optional[GraphConfiguration] = None):
        self.config = config or GraphConfiguration()
        self.schema = GraphSchema(self.config)
        
        # Inicializa componentes
        self.section_extractor = None
        self.ner_extractor = None
        self.concept_extractor = None
        self.text_vectorizer = None
        self.pmi_calculator = None
        
        # Estado interno
        self.graph: Optional[nx.Graph] = None
        self.documents: List[DocumentNode] = []
        self.all_nodes: Dict[str, Any] = {}  # id -> node
        self.all_edges: List[Any] = []
        
        # Cache de embeddings
        self.embeddings_cache: Dict[str, np.ndarray] = {}
        
        logger.info("🚀 GraphPipeline inicializado")
    
    def _setup_components(self):
        """Inicializa todos os componentes necessários"""
        logger.info("🔧 Inicializando componentes...")
        
        try:
            self.section_extractor = SectionExtractor()
            self.ner_extractor = NERExtractor()
            self.concept_extractor = ConceptExtractor()
            self.text_vectorizer = TextVectorizer(model_name=self.config.embedding_model)
            self.pmi_calculator = PMICalculator()
            
            logger.info("✅ Todos os componentes inicializados")
            
        except Exception as e:
            logger.error(f"❌ Erro ao inicializar componentes: {e}")
            raise
    
    def build_complete_graph(self, 
                           limit: Optional[int] = None,
                           save_path: Optional[str] = None,
                           load_cache: bool = True) -> GraphConstructionResult:
        """
        Constrói o grafo completo seguindo todas as etapas
        
        Args:
            limit: Número máximo de documentos a processar
            save_path: Caminho para salvar o grafo (opcional)
            load_cache: Se deve carregar cache de embeddings
            
        Returns:
            GraphConstructionResult com todos os detalhes
        """
        
        result = GraphConstructionResult()
        start_time = time.time()
        
        logger.info(f"🏗️ Iniciando construção completa do grafo (limit={limit})")
        result.add_log(f"Iniciando construção do grafo com {limit or 'todos'} documentos")
        
        try:
            # Etapa 0: Setup
            step_start = time.time()
            self._setup_components()
            result.execution_times['setup'] = time.time() - step_start
            
            # Etapa 1: Preprocessamento
            step_start = time.time()
            result.add_log("Etapa 1: Carregando e preprocessando documentos...")
            
            documents, preprocessing_stats = preprocess_jurisprudencias(
                limit=limit,
                database_url=None  # Usa configuração padrão
            )
            
            if not documents:
                raise Exception("Nenhum documento válido carregado")
            
            self.documents = documents
            result.preprocessing_stats = preprocessing_stats
            result.nodes_created['documents'] = len(documents)
            result.execution_times['preprocessing'] = time.time() - step_start
            
            logger.info(f"✅ {len(documents)} documentos carregados")
            result.add_log(f"Carregados {len(documents)} documentos válidos")
            
            # Etapa 2: Extração de Seções
            step_start = time.time()
            result.add_log("Etapa 2: Extraindo seções...")
            
            sections = self._extract_sections(documents)
            result.nodes_created['sections'] = len(sections)
            result.execution_times['section_extraction'] = time.time() - step_start
            
            logger.info(f"✅ {len(sections)} seções extraídas")
            result.add_log(f"Extraídas {len(sections)} seções")
            
            # Etapa 3: Extração de Entidades
            step_start = time.time()
            result.add_log("Etapa 3: Extraindo entidades nomeadas...")
            
            entities = self._extract_entities(documents + sections)
            result.nodes_created['entities'] = len(entities)
            result.execution_times['entity_extraction'] = time.time() - step_start
            
            logger.info(f"✅ {len(entities)} entidades extraídas")
            result.add_log(f"Extraídas {len(entities)} entidades únicas")
            
            # Etapa 4: Extração de Conceitos
            step_start = time.time()
            result.add_log("Etapa 4: Extraindo conceitos jurídicos...")
            
            concepts = self._extract_concepts(documents)
            result.nodes_created['concepts'] = len(concepts)
            result.execution_times['concept_extraction'] = time.time() - step_start
            
            logger.info(f"✅ {len(concepts)} conceitos extraídos")
            result.add_log(f"Extraídos {len(concepts)} conceitos jurídicos")
            
            # Etapa 5: Vetorização
            step_start = time.time()
            result.add_log("Etapa 5: Gerando embeddings...")
            
            self._generate_embeddings(documents, sections, load_cache)
            result.execution_times['vectorization'] = time.time() - step_start
            
            logger.info(f"✅ Embeddings gerados para {len(self.embeddings_cache)} nós")
            result.add_log(f"Gerados embeddings para {len(self.embeddings_cache)} nós")
            
            # Etapa 6: Construção do Grafo
            step_start = time.time()
            result.add_log("Etapa 6: Construindo grafo e arestas...")
            
            # Coleta todos os nós
            all_nodes = documents + sections + entities + concepts
            for node in all_nodes:
                self.all_nodes[node.id] = node
            
            # Cria grafo NetworkX
            self.graph = nx.Graph()
            
            # Adiciona nós
            for node in all_nodes:
                self.graph.add_node(
                    node.id,
                    **node.to_dict()
                )
            
            # Cria arestas
            self._create_all_edges(documents, sections, entities, concepts)
            result.execution_times['graph_construction'] = time.time() - step_start
            
            logger.info(f"✅ Grafo construído: {self.graph.number_of_nodes()} nós, {self.graph.number_of_edges()} arestas")
            result.add_log(f"Grafo: {self.graph.number_of_nodes()} nós, {self.graph.number_of_edges()} arestas")
            
            # Etapa 7: Validação e Estatísticas
            step_start = time.time()
            result.add_log("Etapa 7: Validando grafo e calculando estatísticas...")
            
            validation_result = self.schema.validate_graph(self.graph)
            if not validation_result.is_valid:
                result.add_log(f"⚠️ Validação encontrou problemas: {validation_result.errors}", "WARNING")
            
            # Calcula estatísticas
            statistics = GraphStatistics()
            statistics.update_from_networkx(self.graph)
            result.statistics = statistics
            result.execution_times['validation'] = time.time() - step_start
            
            # Etapa 8: Salvamento (opcional)
            if save_path:
                step_start = time.time()
                result.add_log(f"Etapa 8: Salvando grafo em {save_path}...")
                
                saved_files = self._save_graph(save_path, result)
                result.output_files.extend(saved_files)
                result.execution_times['saving'] = time.time() - step_start
            
            # Finalização
            total_time = time.time() - start_time
            result.execution_times['total'] = total_time
            
            result.graph = self.graph
            result.success = True
            
            logger.info(f"🎉 Grafo construído com sucesso em {total_time:.2f}s")
            result.add_log(f"Construção concluída com sucesso em {total_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Erro na construção do grafo: {e}")
            result.add_log(f"Erro fatal: {e}", "ERROR")
            result.success = False
            
            import traceback
            traceback.print_exc()
        
        return result
    
    def _extract_sections(self, documents: List[DocumentNode]) -> List[SectionNode]:
        """Extrai seções dos documentos"""
        
        sections = []
        
        for doc in documents:
            if not doc.conteudo_limpo:
                continue
            
            try:
                doc_sections = self.section_extractor.extract_sections_from_document(doc)
                sections.extend(doc_sections)
                
                # Atualiza documento com IDs das seções
                for section in doc_sections:
                    doc.add_secao(section.id)
                
            except Exception as e:
                logger.warning(f"Erro ao extrair seções do documento {doc.id}: {e}")
                continue
        
        return sections
    
    def _extract_entities(self, text_nodes: List) -> List[EntityNode]:
        """Extrai entidades nomeadas de nós com texto"""
        
        entities = []
        entity_map = {}  # nome_normalizado -> EntityNode (para deduplicação)
        
        for node in text_nodes:
            # Determina texto a processar
            text_content = None
            if hasattr(node, 'conteudo_limpo') and node.conteudo_limpo:
                text_content = node.conteudo_limpo
            elif hasattr(node, 'conteudo_texto') and node.conteudo_texto:
                text_content = node.conteudo_texto
            
            if not text_content:
                continue
            
            try:
                # Extrai entidades do texto
                extracted_entities = self.ner_extractor.extract_entities_from_text(text_content)
                
                for entity_data in extracted_entities:
                    nome_norm = entity_data['nome_normalizado']
                    
                    if nome_norm in entity_map:
                        # Entidade já existe, incrementa frequência
                        entity_map[nome_norm].frequencia_global += 1
                    else:
                        # Nova entidade
                        entity_node = EntityNode(
                            id=f"ent_{entity_data['tipo']}_{hash(nome_norm) % 10000:04d}",
                            entity_type=entity_data['tipo'],
                            nome_original=entity_data['nome_original'],
                            nome_normalizado=nome_norm,
                            frequencia_global=1
                        )
                        entity_map[nome_norm] = entity_node
                        entities.append(entity_node)
                
            except Exception as e:
                logger.warning(f"Erro ao extrair entidades do nó {node.id}: {e}")
                continue
        
        # Filtra entidades por frequência mínima
        min_frequency = self.config.min_entity_frequency if hasattr(self.config, 'min_entity_frequency') else 2
        entities = [e for e in entities if e.frequencia_global >= min_frequency]
        
        return entities
    
    def _extract_concepts(self, documents: List[DocumentNode]) -> List[ConceptNode]:
        """Extrai conceitos jurídicos dos documentos"""
        
        # Coleta todos os textos
        texts = []
        doc_ids = []
        
        for doc in documents:
            if doc.conteudo_limpo:
                texts.append(doc.conteudo_limpo)
                doc_ids.append(doc.id)
        
        if not texts:
            return []
        
        try:
            # Usa o concept_extractor para identificar conceitos
            concepts_data = self.concept_extractor.extract_concepts_from_corpus(texts, doc_ids)
            
            concepts = []
            for concept_info in concepts_data:
                concept_node = ConceptNode(
                    id=f"con_{hash(concept_info['termo']) % 10000:04d}",
                    termo_conceito=concept_info['termo'],
                    categoria_juridica=concept_info.get('categoria'),
                    frequencia_global=concept_info['frequencia']
                )
                concepts.append(concept_node)
            
            return concepts
            
        except Exception as e:
            logger.error(f"Erro na extração de conceitos: {e}")
            return []
    
    def _generate_embeddings(self, documents: List[DocumentNode], 
                           sections: List[SectionNode], 
                           load_cache: bool = True):
        """Gera embeddings para documentos e seções"""
        
        cache_file = Path("data/graph/embeddings/embeddings_cache.pkl")
        
        # Carrega cache se solicitado
        if load_cache and cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    self.embeddings_cache = pickle.load(f)
                logger.info(f"📥 Cache carregado: {len(self.embeddings_cache)} embeddings")
            except Exception as e:
                logger.warning(f"Erro ao carregar cache: {e}")
                self.embeddings_cache = {}
        
        # Prepara textos para vetorização
        texts_to_vectorize = []
        node_ids = []
        
        # Documentos
        for doc in documents:
            if doc.conteudo_limpo and doc.id not in self.embeddings_cache:
                texts_to_vectorize.append(doc.conteudo_limpo)
                node_ids.append(doc.id)
        
        # Seções
        for section in sections:
            if section.conteudo_limpo and section.id not in self.embeddings_cache:
                texts_to_vectorize.append(section.conteudo_limpo)
                node_ids.append(section.id)
        
        if not texts_to_vectorize:
            logger.info("📋 Todos os embeddings já estão em cache")
            return
        
        # Gera embeddings
        logger.info(f"🔄 Gerando {len(texts_to_vectorize)} novos embeddings...")
        
        embeddings = self.text_vectorizer.vectorize_texts(texts_to_vectorize)
        
        # Atualiza cache
        for node_id, embedding in zip(node_ids, embeddings):
            self.embeddings_cache[node_id] = embedding
        
        # Salva cache atualizado
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.embeddings_cache, f)
            logger.info(f"💾 Cache atualizado: {len(self.embeddings_cache)} embeddings")
        except Exception as e:
            logger.warning(f"Erro ao salvar cache: {e}")
    
    def _create_all_edges(self, documents: List[DocumentNode], 
                         sections: List[SectionNode],
                         entities: List[EntityNode], 
                         concepts: List[ConceptNode]):
        """Cria todas as arestas do grafo"""
        
        # 1. Arestas de similaridade (Doc <-> Doc)
        self._create_similarity_edges(documents, "document")
        
        # 2. Arestas de similaridade (Seção <-> Seção)
        self._create_similarity_edges(sections, "section")
        
        # 3. Arestas hierárquicas (Doc -> Seção)
        self._create_hierarchical_edges(documents, sections)
        
        # 4. Arestas de relevância (Doc <-> Conceito)
        self._create_relevance_edges(documents, concepts)
        
        # 5. Arestas de co-ocorrência (Conceito <-> Conceito)  
        self._create_cooccurrence_edges(concepts, documents)
    
    def _create_similarity_edges(self, nodes: List, node_type: str):
        """Cria arestas de similaridade entre nós do mesmo tipo"""
        
        if len(nodes) < 2:
            return
        
        # Filtra nós com embeddings
        nodes_with_embeddings = [n for n in nodes if n.id in self.embeddings_cache]
        
        if len(nodes_with_embeddings) < 2:
            return
        
        # Prepara matriz de embeddings
        embeddings = [self.embeddings_cache[n.id] for n in nodes_with_embeddings]
        embeddings_matrix = np.array(embeddings)
        
        # Calcula similaridade de cossenos
        similarity_matrix = cosine_similarity(embeddings_matrix)
        
        # Determina threshold
        if node_type == "document":
            threshold = self.config.similarity_thresholds.get('DOCUMENT_SEMANTIC', 0.3)
            max_edges = self.config.max_similarity_edges_per_document
        else:
            threshold = self.config.similarity_thresholds.get('SECTION_CONTENT', 0.4)  
            max_edges = 20
        
        edges_created = 0
        
        # Cria arestas
        for i, node1 in enumerate(nodes_with_embeddings):
            # Pega similaridades acima do threshold
            similarities = []
            for j, node2 in enumerate(nodes_with_embeddings):
                if i != j and similarity_matrix[i][j] >= threshold:
                    similarities.append((j, similarity_matrix[i][j], node2))
            
            # Ordena e limita
            similarities.sort(key=lambda x: x[1], reverse=True)
            similarities = similarities[:max_edges]
            
            # Cria arestas (evita duplicatas)
            for j, sim_score, node2 in similarities:
                if node1.id < node2.id:  # Evita duplicatas
                    edge = SimilarityEdge.create_document_similarity(
                        node1.id, node2.id, sim_score, self.config.embedding_model
                    )
                    
                    self.graph.add_edge(
                        node1.id, node2.id,
                        **edge.to_dict()
                    )
                    edges_created += 1
        
        logger.info(f"✅ {edges_created} arestas de similaridade {node_type} criadas")
    
    def _create_hierarchical_edges(self, documents: List[DocumentNode], 
                                  sections: List[SectionNode]):
        """Cria arestas hierárquicas Doc -> Seção"""
        
        edges_created = 0
        
        for section in sections:
            # Encontra documento pai
            parent_doc = next(
                (doc for doc in documents if doc.id == section.parent_document_id), 
                None
            )
            
            if parent_doc:
                edge = HierarchicalEdge.create_document_section(
                    parent_doc.id, section.id
                )
                
                self.graph.add_edge(
                    parent_doc.id, section.id,
                    **edge.to_dict()
                )
                edges_created += 1
        
        logger.info(f"✅ {edges_created} arestas hierárquicas criadas")
    
    def _create_relevance_edges(self, documents: List[DocumentNode], 
                               concepts: List[ConceptNode]):
        """Cria arestas de relevância Doc <-> Conceito usando TF-IDF"""
        
        if not documents or not concepts:
            return
        
        # Prepara textos e termos
        texts = [doc.conteudo_limpo for doc in documents if doc.conteudo_limpo]
        terms = [concept.termo_conceito for concept in concepts]
        
        if not texts or not terms:
            return
        
        # Calcula TF-IDF usando sklearn
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        vectorizer = TfidfVectorizer(
            vocabulary=terms,
            lowercase=True,
            token_pattern=r'(?u)\b\w+\b'
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            edges_created = 0
            threshold = self.config.relevance_threshold
            
            for doc_idx, doc in enumerate(documents):
                if not doc.conteudo_limpo:
                    continue
                
                for concept_idx, concept in enumerate(concepts):
                    tfidf_score = tfidf_matrix[doc_idx, concept_idx]
                    
                    if tfidf_score >= threshold:
                        edge = RelevanceEdge.create_document_concept_relevance(
                            doc.id, concept.id, tfidf_score, 0, 0, 0.0, len(documents)
                        )
                        
                        self.graph.add_edge(
                            doc.id, concept.id,
                            **edge.to_dict()
                        )
                        edges_created += 1
            
            logger.info(f"✅ {edges_created} arestas de relevância criadas")
            
        except Exception as e:
            logger.error(f"Erro ao criar arestas de relevância: {e}")
    
    def _create_cooccurrence_edges(self, concepts: List[ConceptNode], 
                                  documents: List[DocumentNode]):
        """Cria arestas de co-ocorrência entre conceitos usando PMI"""
        
        if len(concepts) < 2 or not documents:
            return
        
        # Prepara corpus para PMI
        texts = [doc.conteudo_limpo for doc in documents if doc.conteudo_limpo]
        terms = [concept.termo_conceito for concept in concepts]
        
        try:
            # Calcula PMI
            pmi_scores = self.pmi_calculator.calculate_pmi_matrix(texts, terms)
            
            edges_created = 0
            threshold = self.config.pmi_threshold
            
            # Cria arestas
            for i, concept1 in enumerate(concepts):
                for j, concept2 in enumerate(concepts):
                    if i < j and pmi_scores[i][j] >= threshold:
                        edge = CooccurrenceEdge.create_concept_cooccurrence(
                            concept1.id, concept2.id, pmi_scores[i][j], 0, 0
                        )
                        
                        self.graph.add_edge(
                            concept1.id, concept2.id,
                            **edge.to_dict()
                        )
                        edges_created += 1
            
            logger.info(f"✅ {edges_created} arestas de co-ocorrência criadas")
            
        except Exception as e:
            logger.error(f"Erro ao criar arestas de co-ocorrência: {e}")
    
    def _save_graph(self, base_path: str, result: GraphConstructionResult) -> List[str]:
        """Salva o grafo em múltiplos formatos"""
        
        saved_files = []
        base_path = Path(base_path)
        base_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # 1. NetworkX Pickle (rápido)
            pickle_path = base_path.with_suffix('.pkl')
            nx.write_gpickle(self.graph, pickle_path)
            saved_files.append(str(pickle_path))
            
            # 2. GraphML (interoperabilidade)
            graphml_path = base_path.with_suffix('.graphml')
            nx.write_graphml(self.graph, graphml_path)
            saved_files.append(str(graphml_path))
            
            # 3. Estatísticas JSON
            stats_path = base_path.with_name(f"{base_path.stem}_stats.json")
            import json
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, indent=2, ensure_ascii=False, default=str)
            saved_files.append(str(stats_path))
            
            logger.info(f"💾 Grafo salvo em {len(saved_files)} formatos")
            
        except Exception as e:
            logger.error(f"Erro ao salvar grafo: {e}")
        
        return saved_files


# Funções auxiliares para uso direto

def build_graph_from_database(limit: Optional[int] = None,
                            save_path: Optional[str] = None,
                            config: Optional[GraphConfiguration] = None) -> GraphConstructionResult:
    """
    Função principal para construir grafo a partir do banco
    
    Args:
        limit: Número máximo de documentos
        save_path: Caminho para salvar o grafo
        config: Configuração personalizada
        
    Returns:
        GraphConstructionResult
    """
    
    pipeline = GraphPipeline(config)
    return pipeline.build_complete_graph(limit=limit, save_path=save_path)


def build_sample_graph(sample_size: int = 50) -> GraphConstructionResult:
    """
    Constrói grafo pequeno para testes
    
    Args:
        sample_size: Número de documentos para amostra
        
    Returns:
        GraphConstructionResult
    """
    
    from ..models.graph_schema import DEVELOPMENT_CONFIG
    
    return build_graph_from_database(
        limit=sample_size,
        config=DEVELOPMENT_CONFIG
    )
 
