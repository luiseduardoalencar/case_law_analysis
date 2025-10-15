# -*- coding: utf-8 -*-
"""
Vetorizador de texto usando sentence-transformers
Gera embeddings semânticos de alta qualidade para documentos e seções
Otimizado para textos jurídicos em português
"""

import re
import pickle
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple, Any
from dataclasses import dataclass
import numpy as np
from loguru import logger

# Sentence Transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ sentence-transformers não encontrado! Install: pip install sentence-transformers")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Fallback para TF-IDF se sentence-transformers não disponível
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


@dataclass
class VectorizationResult:
    """Resultado da vetorização"""
    embeddings: np.ndarray  # Matriz de embeddings (n_texts, embedding_dim)
    text_hashes: List[str]  # Hashes dos textos processados
    model_used: str  # Modelo usado
    embedding_dimension: int  # Dimensão dos embeddings
    processing_time: float  # Tempo de processamento
    texts_processed: int  # Número de textos processados
    cache_hits: int = 0  # Hits no cache
    cache_misses: int = 0  # Misses no cache


@dataclass
class TextChunk:
    """Representa um chunk de texto"""
    text: str
    start_pos: int
    end_pos: int
    chunk_id: int
    is_overlapping: bool = False


class TextVectorizer:
    """Vetorizador de texto com embeddings semânticos"""
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 max_sequence_length: int = 384,
                 cache_dir: Optional[str] = None,
                 use_cache: bool = True,
                 fallback_to_tfidf: bool = True):
        """
        Inicializa o vetorizador
        
        Args:
            model_name: Nome do modelo sentence-transformers
            max_sequence_length: Tamanho máximo da sequência
            cache_dir: Diretório para cache de embeddings
            use_cache: Se deve usar cache
            fallback_to_tfidf: Se deve usar TF-IDF como fallback
        """
        
        self.model_name = model_name
        self.max_sequence_length = max_sequence_length
        self.use_cache = use_cache
        self.fallback_to_tfidf = fallback_to_tfidf
        
        # Configurações de cache
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/graph/embeddings/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / f"embeddings_cache_{self._get_model_hash()}.pkl"
        
        # Cache em memória
        self.embeddings_cache: Dict[str, np.ndarray] = {}
        
        # Modelos
        self.sentence_model = None
        self.tfidf_model = None
        self.svd_model = None
        
        # Estatísticas
        self.stats = {
            'texts_vectorized': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_processing_time': 0.0,
            'model_loads': 0,
            'chunks_created': 0
        }
        
        self._load_models()
        self._load_cache()
        
        logger.info(f"🔤 TextVectorizer inicializado com modelo: {model_name}")
    
    def _get_model_hash(self) -> str:
        """Gera hash único para o modelo (para cache)"""
        model_str = f"{self.model_name}_{self.max_sequence_length}"
        return hashlib.md5(model_str.encode()).hexdigest()[:8]
    
    def _load_models(self):
        """Carrega modelos de embedding"""
        
        # Tenta carregar sentence-transformers
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.sentence_model = SentenceTransformer(self.model_name)
                
                # Configura tamanho máximo da sequência
                if hasattr(self.sentence_model, 'max_seq_length'):
                    self.sentence_model.max_seq_length = self.max_sequence_length
                
                self.stats['model_loads'] += 1
                logger.info(f"✅ Modelo sentence-transformers carregado: {self.model_name}")
                
                # Testa o modelo com texto pequeno
                test_embedding = self.sentence_model.encode(["teste"])
                logger.info(f"📐 Dimensão dos embeddings: {test_embedding.shape[1]}")
                
                return
                
            except Exception as e:
                logger.error(f"❌ Erro ao carregar sentence-transformers: {e}")
                self.sentence_model = None
        
        # Fallback para TF-IDF se necessário
        if self.fallback_to_tfidf:
            logger.info("🔄 Usando TF-IDF como fallback")
            self._setup_tfidf_fallback()
        else:
            raise RuntimeError("Sentence-transformers não disponível e fallback desabilitado")
    
    def _setup_tfidf_fallback(self):
        """Configura TF-IDF como fallback"""
        
        # Stopwords em português
        try:
            import nltk
            from nltk.corpus import stopwords
            stop_words = stopwords.words('portuguese')
        except:
            stop_words = [
                'a', 'o', 'e', 'de', 'do', 'da', 'em', 'um', 'uma', 'com',
                'não', 'que', 'por', 'para', 'se', 'na', 'no', 'é', 'são'
            ]
        
        # TF-IDF com configurações otimizadas para textos jurídicos
        self.tfidf_model = TfidfVectorizer(
            max_features=5000,  # Limita vocabulário
            min_df=2,  # Palavra deve aparecer em pelo menos 2 documentos
            max_df=0.95,  # Remove palavras muito comuns
            ngram_range=(1, 3),  # Uni, bi e trigramas
            stop_words=stop_words,
            lowercase=True,
            token_pattern=r'(?u)\b[a-zA-ZÀ-ÿ]{2,}\b'  # Só palavras com 2+ chars
        )
        
        # SVD para redução dimensional (simula embeddings)
        self.svd_model = TruncatedSVD(
            n_components=384,  # Dimensão similar ao sentence-transformers
            random_state=42
        )
        
        logger.info("✅ TF-IDF + SVD configurado como fallback")
    
    def _load_cache(self):
        """Carrega cache de embeddings do disco"""
        
        if not self.use_cache or not self.cache_file.exists():
            return
        
        try:
            with open(self.cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                
            self.embeddings_cache = cache_data.get('embeddings', {})
            
            logger.info(f"📥 Cache carregado: {len(self.embeddings_cache)} embeddings")
            
        except Exception as e:
            logger.warning(f"⚠️ Erro ao carregar cache: {e}")
            self.embeddings_cache = {}
    
    def _save_cache(self):
        """Salva cache de embeddings no disco"""
        
        if not self.use_cache:
            return
        
        try:
            cache_data = {
                'embeddings': self.embeddings_cache,
                'model_name': self.model_name,
                'max_sequence_length': self.max_sequence_length,
                'cache_version': '1.0'
            }
            
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
                
            logger.debug(f"💾 Cache salvo: {len(self.embeddings_cache)} embeddings")
            
        except Exception as e:
            logger.warning(f"⚠️ Erro ao salvar cache: {e}")
    
    def vectorize_texts(self, texts: List[str], 
                       text_ids: Optional[List[str]] = None,
                       batch_size: int = 32,
                       show_progress: bool = True) -> VectorizationResult:
        """
        Vetoriza uma lista de textos
        
        Args:
            texts: Lista de textos para vetorizar
            text_ids: IDs dos textos (opcional, para cache mais eficiente)
            batch_size: Tamanho do batch para processamento
            show_progress: Se deve mostrar progresso
            
        Returns:
            VectorizationResult com embeddings e metadados
        """
        
        if not texts:
            return VectorizationResult(
                embeddings=np.array([]),
                text_hashes=[],
                model_used=self.model_name,
                embedding_dimension=0,
                processing_time=0.0,
                texts_processed=0
            )
        
        import time
        start_time = time.time()
        
        logger.info(f"🔤 Vetorizando {len(texts)} textos...")
        
        if text_ids is None:
            text_ids = [f"text_{i}" for i in range(len(texts))]
        
        # Processa textos e verifica cache
        texts_to_process = []
        text_hashes = []
        cached_embeddings = {}
        cache_hits = 0
        cache_misses = 0
        
        for i, (text, text_id) in enumerate(zip(texts, text_ids)):
            # Limpa e prepara o texto
            cleaned_text = self._preprocess_text(text)
            text_hash = self._get_text_hash(cleaned_text, text_id)
            text_hashes.append(text_hash)
            
            # Verifica cache
            if self.use_cache and text_hash in self.embeddings_cache:
                cached_embeddings[i] = self.embeddings_cache[text_hash]
                cache_hits += 1
            else:
                texts_to_process.append((i, cleaned_text, text_hash))
                cache_misses += 1
        
        # Processa textos não encontrados no cache
        new_embeddings = {}
        
        if texts_to_process:
            if show_progress:
                logger.info(f"🔄 Processando {len(texts_to_process)} textos novos...")
            
            # Extrai textos para processamento
            texts_for_model = [item[1] for item in texts_to_process]
            
            # Vetoriza usando o modelo apropriado
            if self.sentence_model:
                embeddings_array = self._vectorize_with_sentence_transformers(
                    texts_for_model, batch_size
                )
            else:
                embeddings_array = self._vectorize_with_tfidf_fallback(texts_for_model)
            
            # Armazena resultados
            for (original_idx, text, text_hash), embedding in zip(texts_to_process, embeddings_array):
                new_embeddings[original_idx] = embedding
                
                # Adiciona ao cache
                if self.use_cache:
                    self.embeddings_cache[text_hash] = embedding
        
        # Combina embeddings cached e novos na ordem original
        final_embeddings = []
        
        for i in range(len(texts)):
            if i in cached_embeddings:
                final_embeddings.append(cached_embeddings[i])
            elif i in new_embeddings:
                final_embeddings.append(new_embeddings[i])
            else:
                # Fallback para embedding zero (não deveria acontecer)
                embedding_dim = 384 if self.sentence_model else self.svd_model.n_components
                final_embeddings.append(np.zeros(embedding_dim))
        
        final_embeddings = np.array(final_embeddings)
        
        # Salva cache atualizado
        if self.use_cache and new_embeddings:
            self._save_cache()
        
        # Atualiza estatísticas
        processing_time = time.time() - start_time
        self.stats['texts_vectorized'] += len(texts)
        self.stats['cache_hits'] += cache_hits
        self.stats['cache_misses'] += cache_misses
        self.stats['total_processing_time'] += processing_time
        
        embedding_dim = final_embeddings.shape[1] if final_embeddings.size > 0 else 0
        
        logger.info(f"✅ Vetorização concluída em {processing_time:.2f}s")
        logger.info(f"📊 Cache: {cache_hits} hits, {cache_misses} misses")
        
        return VectorizationResult(
            embeddings=final_embeddings,
            text_hashes=text_hashes,
            model_used=self.model_name if self.sentence_model else "TF-IDF+SVD",
            embedding_dimension=embedding_dim,
            processing_time=processing_time,
            texts_processed=len(texts),
            cache_hits=cache_hits,
            cache_misses=cache_misses
        )
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocessa texto para vetorização"""
        
        if not text or not isinstance(text, str):
            return ""
        
        # Remove quebras de linha excessivas
        cleaned = re.sub(r'\n\s*\n+', '\n\n', text)
        
        # Remove espaços extras
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Remove caracteres de controle
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', cleaned)
        
        # Trunca se muito longo (deixa margem para tokenização)
        max_chars = self.max_sequence_length * 4  # Aproximadamente 4 chars por token
        if len(cleaned) > max_chars:
            cleaned = cleaned[:max_chars] + "..."
        
        return cleaned.strip()
    
    def _get_text_hash(self, text: str, text_id: str) -> str:
        """Gera hash único para um texto (para cache)"""
        
        # Combina texto, ID e configurações do modelo
        hash_input = f"{text}|{text_id}|{self.model_name}|{self.max_sequence_length}"
        return hashlib.md5(hash_input.encode('utf-8')).hexdigest()
    
    def _vectorize_with_sentence_transformers(self, texts: List[str], 
                                            batch_size: int = 32) -> np.ndarray:
        """Vetoriza usando sentence-transformers"""
        
        try:
            # Processa em lotes para otimizar memória
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Gera embeddings
                batch_embeddings = self.sentence_model.encode(
                    batch_texts,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    batch_size=min(batch_size, len(batch_texts))
                )
                
                all_embeddings.append(batch_embeddings)
            
            # Concatena todos os lotes
            final_embeddings = np.vstack(all_embeddings) if all_embeddings else np.array([])
            
            return final_embeddings
            
        except Exception as e:
            logger.error(f"❌ Erro na vetorização com sentence-transformers: {e}")
            
            # Fallback para TF-IDF se erro
            if self.fallback_to_tfidf:
                logger.warning("🔄 Usando fallback TF-IDF devido ao erro")
                return self._vectorize_with_tfidf_fallback(texts)
            else:
                raise
    
    def _vectorize_with_tfidf_fallback(self, texts: List[str]) -> np.ndarray:
        """Vetoriza usando TF-IDF + SVD como fallback"""
        
        if not texts:
            return np.array([])
        
        try:
            # Ajusta TF-IDF se necessário (primeira vez)
            if not hasattr(self.tfidf_model, 'vocabulary_'):
                logger.info("📚 Treinando modelo TF-IDF...")
                tfidf_matrix = self.tfidf_model.fit_transform(texts)
                
                # Treina SVD
                self.svd_model.fit(tfidf_matrix)
            else:
                tfidf_matrix = self.tfidf_model.transform(texts)
            
            # Aplica SVD para redução dimensional
            embeddings = self.svd_model.transform(tfidf_matrix)
            
            # Normaliza embeddings (similar ao sentence-transformers)
            from sklearn.preprocessing import normalize
            embeddings = normalize(embeddings, norm='l2')
            
            return embeddings.astype(np.float32)
            
        except Exception as e:
            logger.error(f"❌ Erro no fallback TF-IDF: {e}")
            
            # Último fallback: embeddings aleatórios
            logger.warning("🎲 Usando embeddings aleatórios como último recurso")
            return np.random.normal(0, 0.1, (len(texts), 384)).astype(np.float32)
    
    def vectorize_single_text(self, text: str, text_id: Optional[str] = None) -> np.ndarray:
        """
        Vetoriza um único texto (método de conveniência)
        
        Args:
            text: Texto para vetorizar
            text_id: ID do texto (opcional)
            
        Returns:
            Embedding do texto
        """
        
        result = self.vectorize_texts([text], [text_id] if text_id else None)
        
        if result.embeddings.size > 0:
            return result.embeddings[0]
        else:
            embedding_dim = 384
            return np.zeros(embedding_dim)
    
    def get_embedding_similarity(self, embedding1: np.ndarray, 
                                embedding2: np.ndarray) -> float:
        """
        Calcula similaridade de cossenos entre dois embeddings
        
        Args:
            embedding1: Primeiro embedding
            embedding2: Segundo embedding
            
        Returns:
            Similaridade de cossenos (0.0 a 1.0)
        """
        
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Garante que são 2D
            emb1 = embedding1.reshape(1, -1)
            emb2 = embedding2.reshape(1, -1)
            
            similarity = cosine_similarity(emb1, emb2)[0, 0]
            
            # Normaliza para 0-1 (cosine pode ser -1 a 1)
            return (similarity + 1) / 2
            
        except Exception as e:
            logger.warning(f"Erro ao calcular similaridade: {e}")
            return 0.0
    
    def clear_cache(self):
        """Limpa cache de embeddings"""
        
        self.embeddings_cache.clear()
        
        if self.cache_file.exists():
            try:
                self.cache_file.unlink()
                logger.info("🗑️ Cache de embeddings limpo")
            except Exception as e:
                logger.warning(f"Erro ao limpar cache: {e}")
    
    def get_cache_size(self) -> int:
        """Retorna tamanho do cache"""
        return len(self.embeddings_cache)
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do vetorizador"""
        
        cache_hit_rate = (self.stats['cache_hits'] / 
                         (self.stats['cache_hits'] + self.stats['cache_misses'])
                         if (self.stats['cache_hits'] + self.stats['cache_misses']) > 0 else 0)
        
        return {
            'model_used': self.model_name if self.sentence_model else "TF-IDF+SVD",
            'model_available': self.sentence_model is not None,
            'texts_vectorized': self.stats['texts_vectorized'],
            'total_processing_time': self.stats['total_processing_time'],
            'average_time_per_text': (self.stats['total_processing_time'] / self.stats['texts_vectorized']
                                    if self.stats['texts_vectorized'] > 0 else 0),
            'cache_size': len(self.embeddings_cache),
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'cache_hit_rate': cache_hit_rate,
            'model_loads': self.stats['model_loads']
        }
    
    def warm_up_model(self):
        """Aquece o modelo com texto de exemplo"""
        
        if self.sentence_model:
            logger.info("🔥 Aquecendo modelo sentence-transformers...")
            _ = self.vectorize_single_text("Este é um texto de exemplo para aquecer o modelo.")
            logger.info("✅ Modelo aquecido")


# Funções auxiliares para uso direto
def vectorize_texts(texts: List[str], 
                   model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") -> np.ndarray:
    """
    Função auxiliar para vetorizar textos
    
    Args:
        texts: Lista de textos
        model_name: Nome do modelo
        
    Returns:
        Array de embeddings
    """
    vectorizer = TextVectorizer(model_name=model_name)
    result = vectorizer.vectorize_texts(texts)
    return result.embeddings


def calculate_text_similarity(text1: str, text2: str,
                            model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") -> float:
    """
    Calcula similaridade entre dois textos
    
    Args:
        text1: Primeiro texto
        text2: Segundo texto
        model_name: Nome do modelo
        
    Returns:
        Similaridade (0.0 a 1.0)
    """
    vectorizer = TextVectorizer(model_name=model_name)
    
    embeddings = vectorizer.vectorize_texts([text1, text2]).embeddings
    
    if embeddings.shape[0] == 2:
        return vectorizer.get_embedding_similarity(embeddings[0], embeddings[1])
    else:
        return 0.0


def get_most_similar_texts(query_text: str, corpus_texts: List[str],
                          top_k: int = 5) -> List[Tuple[int, str, float]]:
    """
    Encontra textos mais similares a uma query
    
    Args:
        query_text: Texto de consulta
        corpus_texts: Corpus de textos
        top_k: Número de resultados
        
    Returns:
        Lista de (índice, texto, similaridade)
    """
    vectorizer = TextVectorizer()
    
    # Vetoriza query e corpus
    all_texts = [query_text] + corpus_texts
    embeddings = vectorizer.vectorize_texts(all_texts).embeddings
    
    if embeddings.size == 0:
        return []
    
    query_embedding = embeddings[0]
    corpus_embeddings = embeddings[1:]
    
    # Calcula similaridades
    similarities = []
    for i, corpus_embedding in enumerate(corpus_embeddings):
        sim = vectorizer.get_embedding_similarity(query_embedding, corpus_embedding)
        similarities.append((i, corpus_texts[i], sim))
    
    # Ordena por similaridade
    similarities.sort(key=lambda x: x[2], reverse=True)
    
    return similarities[:top_k]