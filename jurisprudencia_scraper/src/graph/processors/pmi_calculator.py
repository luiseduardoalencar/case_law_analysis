# -*- coding: utf-8 -*-
"""
Calculador de PMI (Pointwise Mutual Information) para conceitos jurídicos
Calcula força de associação entre conceitos baseado em co-ocorrência
Otimizado para textos jurídicos e conceitos do domínio
"""

import re
import numpy as np
from typing import List, Dict, Tuple, Set, Optional, Any
from dataclasses import dataclass
from collections import defaultdict, Counter
from itertools import combinations
import math
from loguru import logger


@dataclass
class CooccurrenceWindow:
    """Representa uma janela de co-ocorrência"""
    text_id: str
    start_pos: int
    end_pos: int
    concepts: List[str]
    window_text: str


@dataclass
class PMIScore:
    """Representa um score PMI entre dois conceitos"""
    concept1: str
    concept2: str
    pmi_score: float
    joint_frequency: int  # Frequência conjunta
    concept1_frequency: int  # Frequência do conceito 1
    concept2_frequency: int  # Frequência do conceito 2
    total_windows: int  # Total de janelas analisadas
    normalized_pmi: Optional[float] = None  # PMI normalizado
    confidence: float = 1.0  # Confiança no score


class PMICalculator:
    """Calculador de Pointwise Mutual Information"""
    
    def __init__(self, 
                 window_size: int = 10,
                 min_cooccurrence: int = 2,
                 min_concept_frequency: int = 3):
        """
        Inicializa o calculador PMI
        
        Args:
            window_size: Tamanho da janela de co-ocorrência (em palavras)
            min_cooccurrence: Mínimo de co-ocorrências para calcular PMI
            min_concept_frequency: Frequência mínima individual do conceito
        """
        
        self.window_size = window_size
        self.min_cooccurrence = min_cooccurrence
        self.min_concept_frequency = min_concept_frequency
        
        # Estatísticas
        self.stats = {
            'texts_processed': 0,
            'total_windows': 0,
            'concepts_found': 0,
            'concept_pairs_analyzed': 0,
            'valid_pmi_scores': 0,
            'processing_time': 0.0
        }
        
        logger.info(f"🔗 PMICalculator inicializado (janela: {window_size})")
    
    def calculate_pmi_matrix(self, texts: List[str], 
                           concepts: List[str],
                           text_ids: Optional[List[str]] = None) -> np.ndarray:
        """
        Calcula matriz PMI entre todos os pares de conceitos
        
        Args:
            texts: Lista de textos para análise
            concepts: Lista de conceitos para analisar
            text_ids: IDs dos textos (opcional)
            
        Returns:
            Matriz PMI simétrica (conceitos x conceitos)
        """
        
        if not texts or not concepts or len(concepts) < 2:
            return np.zeros((len(concepts), len(concepts)))
        
        if text_ids is None:
            text_ids = [f"text_{i}" for i in range(len(texts))]
        
        import time
        start_time = time.time()
        
        logger.info(f"🔍 Calculando PMI para {len(concepts)} conceitos em {len(texts)} textos...")
        
        # 1. Extrai janelas de co-ocorrência
        windows = self._extract_cooccurrence_windows(texts, concepts, text_ids)
        
        # 2. Conta frequências
        concept_frequencies, cooccurrence_frequencies = self._count_frequencies(windows, concepts)
        
        # 3. Calcula scores PMI
        pmi_scores = self._calculate_pmi_scores(
            concept_frequencies, cooccurrence_frequencies, len(windows)
        )
        
        # 4. Constrói matriz PMI
        pmi_matrix = self._build_pmi_matrix(pmi_scores, concepts)
        
        # Atualiza estatísticas
        processing_time = time.time() - start_time
        self.stats['texts_processed'] = len(texts)
        self.stats['total_windows'] = len(windows)
        self.stats['concepts_found'] = len([c for c in concept_frequencies.values() if c > 0])
        self.stats['concept_pairs_analyzed'] = len(pmi_scores)
        self.stats['valid_pmi_scores'] = len([s for s in pmi_scores.values() if s.pmi_score > 0])
        self.stats['processing_time'] = processing_time
        
        logger.info(f"✅ Matriz PMI {len(concepts)}x{len(concepts)} calculada em {processing_time:.2f}s")
        logger.info(f"📊 {self.stats['valid_pmi_scores']} pares com PMI > 0")
        
        return pmi_matrix
    
    def _extract_cooccurrence_windows(self, texts: List[str], 
                                    concepts: List[str],
                                    text_ids: List[str]) -> List[CooccurrenceWindow]:
        """Extrai janelas de co-ocorrência dos textos"""
        
        windows = []
        
        # Compila padrões regex para conceitos
        concept_patterns = {}
        for concept in concepts:
            # Cria padrão flexível para o conceito
            pattern = self._create_flexible_pattern(concept)
            concept_patterns[concept] = re.compile(pattern, re.IGNORECASE)
        
        for text_id, text in zip(text_ids, texts):
            if not text or not text.strip():
                continue
            
            # Normaliza texto
            normalized_text = self._normalize_text_for_windows(text)
            words = normalized_text.split()
            
            if len(words) < self.window_size:
                continue
            
            # Desliza janela pelo texto
            for i in range(len(words) - self.window_size + 1):
                window_words = words[i:i + self.window_size]
                window_text = ' '.join(window_words)
                
                # Encontra conceitos na janela
                concepts_in_window = []
                
                for concept, pattern in concept_patterns.items():
                    if pattern.search(window_text):
                        concepts_in_window.append(concept)
                
                # Se tem 2+ conceitos, cria janela de co-ocorrência
                if len(concepts_in_window) >= 2:
                    windows.append(CooccurrenceWindow(
                        text_id=text_id,
                        start_pos=i,
                        end_pos=i + self.window_size,
                        concepts=list(set(concepts_in_window)),  # Remove duplicatas
                        window_text=window_text
                    ))
        
        return windows
    
    def _create_flexible_pattern(self, concept: str) -> str:
        """Cria padrão regex flexível para um conceito"""
        
        # Escapa caracteres especiais
        escaped = re.escape(concept)
        
        # Permite variações de plural/singular
        words = escaped.split('\\ ')  # Split por espaço escapado
        flexible_words = []
        
        for word in words:
            # Adiciona flexibilidade para plurais
            if word.endswith('o'):
                flexible_words.append(word + 's?')
            elif word.endswith('a'):
                flexible_words.append(word + 's?')
            elif word.endswith('l'):
                # moral -> morais
                flexible_words.append(word.replace('l', '(?:l|is)'))
            else:
                flexible_words.append(word + 's?')
        
        # Permite espaçamento flexível
        pattern = r'\b' + r'\s+'.join(flexible_words) + r'\b'
        
        return pattern
    
    def _normalize_text_for_windows(self, text: str) -> str:
        """Normaliza texto para extração de janelas"""
        
        # Converte para minúsculas
        normalized = text.lower()
        
        # Remove pontuação excessiva mas mantém estrutura
        normalized = re.sub(r'[^\w\s\-]', ' ', normalized)
        
        # Normaliza espaços
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized.strip()
    
    def _count_frequencies(self, windows: List[CooccurrenceWindow], 
                          concepts: List[str]) -> Tuple[Dict[str, int], Dict[Tuple[str, str], int]]:
        """Conta frequências de conceitos individuais e pares"""
        
        concept_frequencies = Counter()
        cooccurrence_frequencies = Counter()
        
        for window in windows:
            # Conta conceitos individuais
            for concept in window.concepts:
                concept_frequencies[concept] += 1
            
            # Conta pares de conceitos (co-ocorrência)
            if len(window.concepts) >= 2:
                for concept1, concept2 in combinations(window.concepts, 2):
                    # Garante ordem consistente (alfabética)
                    pair = tuple(sorted([concept1, concept2]))
                    cooccurrence_frequencies[pair] += 1
        
        return dict(concept_frequencies), dict(cooccurrence_frequencies)
    
    def _calculate_pmi_scores(self, concept_frequencies: Dict[str, int],
                            cooccurrence_frequencies: Dict[Tuple[str, str], int],
                            total_windows: int) -> Dict[Tuple[str, str], PMIScore]:
        """Calcula scores PMI para todos os pares válidos"""
        
        pmi_scores = {}
        
        for (concept1, concept2), joint_freq in cooccurrence_frequencies.items():
            
            # Verifica frequências mínimas
            freq1 = concept_frequencies.get(concept1, 0)
            freq2 = concept_frequencies.get(concept2, 0)
            
            if (joint_freq < self.min_cooccurrence or 
                freq1 < self.min_concept_frequency or 
                freq2 < self.min_concept_frequency):
                continue
            
            # Calcula PMI
            pmi_score = self._calculate_pmi(joint_freq, freq1, freq2, total_windows)
            
            # Calcula PMI normalizado
            normalized_pmi = self._calculate_normalized_pmi(
                joint_freq, freq1, freq2, total_windows, pmi_score
            )
            
            # Calcula confiança baseada nas frequências
            confidence = self._calculate_confidence(joint_freq, freq1, freq2, total_windows)
            
            pmi_scores[(concept1, concept2)] = PMIScore(
                concept1=concept1,
                concept2=concept2,
                pmi_score=pmi_score,
                joint_frequency=joint_freq,
                concept1_frequency=freq1,
                concept2_frequency=freq2,
                total_windows=total_windows,
                normalized_pmi=normalized_pmi,
                confidence=confidence
            )
        
        return pmi_scores
    
    def _calculate_pmi(self, joint_freq: int, freq1: int, freq2: int, 
                      total_windows: int) -> float:
        """Calcula PMI básico"""
        
        if total_windows == 0 or joint_freq == 0 or freq1 == 0 or freq2 == 0:
            return 0.0
        
        # PMI = log(P(x,y) / (P(x) * P(y)))
        p_xy = joint_freq / total_windows
        p_x = freq1 / total_windows
        p_y = freq2 / total_windows
        
        if p_xy == 0 or p_x == 0 or p_y == 0:
            return 0.0
        
        pmi = math.log2(p_xy / (p_x * p_y))
        
        return pmi
    
    def _calculate_normalized_pmi(self, joint_freq: int, freq1: int, freq2: int,
                                total_windows: int, pmi: float) -> float:
        """Calcula PMI normalizado (NPMI)"""
        
        if total_windows == 0 or joint_freq == 0:
            return 0.0
        
        # NPMI = PMI / -log(P(x,y))
        p_xy = joint_freq / total_windows
        
        if p_xy == 0:
            return 0.0
        
        npmi = pmi / (-math.log2(p_xy))
        
        # NPMI está no range [-1, 1]
        return max(-1.0, min(1.0, npmi))
    
    def _calculate_confidence(self, joint_freq: int, freq1: int, freq2: int,
                            total_windows: int) -> float:
        """Calcula confiança do score PMI baseado nas frequências"""
        
        # Confiança baseada na frequência conjunta
        confidence = 0.5  # Base
        
        # Boost por frequência conjunta alta
        if joint_freq >= 5:
            confidence += 0.3
        elif joint_freq >= 3:
            confidence += 0.2
        
        # Boost por frequências individuais equilibradas
        ratio = min(freq1, freq2) / max(freq1, freq2) if max(freq1, freq2) > 0 else 0
        if ratio > 0.5:  # Frequências similares
            confidence += 0.2
        elif ratio > 0.3:
            confidence += 0.1
        
        # Penaliza se muito raros
        if joint_freq == 1:
            confidence *= 0.7
        
        return min(1.0, confidence)
    
    def _build_pmi_matrix(self, pmi_scores: Dict[Tuple[str, str], PMIScore],
                         concepts: List[str]) -> np.ndarray:
        """Constrói matriz PMI simétrica"""
        
        n_concepts = len(concepts)
        pmi_matrix = np.zeros((n_concepts, n_concepts))
        
        # Cria mapeamento conceito -> índice
        concept_to_idx = {concept: i for i, concept in enumerate(concepts)}
        
        # Preenche matriz
        for (concept1, concept2), score in pmi_scores.items():
            if concept1 in concept_to_idx and concept2 in concept_to_idx:
                idx1 = concept_to_idx[concept1]
                idx2 = concept_to_idx[concept2]
                
                # Usa PMI ponderado pela confiança
                weighted_pmi = score.pmi_score * score.confidence
                
                # Matriz simétrica
                pmi_matrix[idx1, idx2] = weighted_pmi
                pmi_matrix[idx2, idx1] = weighted_pmi
        
        return pmi_matrix
    
    def calculate_pmi_for_concept_pairs(self, texts: List[str],
                                      concept_pairs: List[Tuple[str, str]],
                                      text_ids: Optional[List[str]] = None) -> List[PMIScore]:
        """
        Calcula PMI para pares específicos de conceitos
        
        Args:
            texts: Textos para análise
            concept_pairs: Lista de pares (conceito1, conceito2)
            text_ids: IDs dos textos (opcional)
            
        Returns:
            Lista de PMIScore para cada par
        """
        
        if not texts or not concept_pairs:
            return []
        
        # Extrai todos os conceitos únicos dos pares
        all_concepts = list(set(
            concept for pair in concept_pairs for concept in pair
        ))
        
        # Calcula matriz completa
        pmi_matrix = self.calculate_pmi_matrix(texts, all_concepts, text_ids)
        
        # Cria mapeamento conceito -> índice
        concept_to_idx = {concept: i for i, concept in enumerate(all_concepts)}
        
        # Extrai scores para pares específicos
        pair_scores = []
        
        for concept1, concept2 in concept_pairs:
            if concept1 in concept_to_idx and concept2 in concept_to_idx:
                idx1 = concept_to_idx[concept1]
                idx2 = concept_to_idx[concept2]
                
                pmi_score = pmi_matrix[idx1, idx2]
                
                # Cria PMIScore (valores aproximados)
                score = PMIScore(
                    concept1=concept1,
                    concept2=concept2,
                    pmi_score=pmi_score,
                    joint_frequency=0,  # Não disponível neste método
                    concept1_frequency=0,
                    concept2_frequency=0,
                    total_windows=self.stats['total_windows'],
                    confidence=0.8  # Confiança padrão
                )
                
                pair_scores.append(score)
        
        return pair_scores
    
    def find_strongest_associations(self, texts: List[str], 
                                  concepts: List[str],
                                  top_k: int = 10,
                                  min_pmi: float = 0.5) -> List[PMIScore]:
        """
        Encontra as associações mais fortes entre conceitos
        
        Args:
            texts: Textos para análise
            concepts: Lista de conceitos
            top_k: Número de associações top
            min_pmi: PMI mínimo para considerar
            
        Returns:
            Lista das top associações ordenadas por PMI
        """
        
        # Calcula todas as frequências
        text_ids = [f"text_{i}" for i in range(len(texts))]
        windows = self._extract_cooccurrence_windows(texts, concepts, text_ids)
        
        if not windows:
            return []
        
        concept_frequencies, cooccurrence_frequencies = self._count_frequencies(windows, concepts)
        pmi_scores = self._calculate_pmi_scores(
            concept_frequencies, cooccurrence_frequencies, len(windows)
        )
        
        # Filtra por PMI mínimo e ordena
        valid_scores = [
            score for score in pmi_scores.values() 
            if score.pmi_score >= min_pmi
        ]
        
        valid_scores.sort(key=lambda x: x.pmi_score, reverse=True)
        
        return valid_scores[:top_k]
    
    def analyze_concept_associations(self, texts: List[str],
                                   target_concept: str,
                                   candidate_concepts: List[str],
                                   top_k: int = 5) -> List[PMIScore]:
        """
        Analisa associações de um conceito específico
        
        Args:
            texts: Textos para análise  
            target_concept: Conceito de interesse
            candidate_concepts: Conceitos candidatos para associação
            top_k: Número de top associações
            
        Returns:
            Lista de associações do conceito target
        """
        
        # Cria pares target + candidatos
        concept_pairs = [(target_concept, candidate) for candidate in candidate_concepts]
        
        # Calcula PMI para os pares
        pmi_scores = self.calculate_pmi_for_concept_pairs(texts, concept_pairs)
        
        # Filtra e ordena
        valid_scores = [score for score in pmi_scores if score.pmi_score > 0]
        valid_scores.sort(key=lambda x: x.pmi_score, reverse=True)
        
        return valid_scores[:top_k]
    
    def get_cooccurrence_contexts(self, texts: List[str],
                                concept1: str, concept2: str,
                                max_contexts: int = 5) -> List[str]:
        """
        Retorna contextos onde dois conceitos co-ocorrem
        
        Args:
            texts: Textos para análise
            concept1: Primeiro conceito
            concept2: Segundo conceito  
            max_contexts: Máximo de contextos
            
        Returns:
            Lista de contextos (janelas de texto)
        """
        
        text_ids = [f"text_{i}" for i in range(len(texts))]
        windows = self._extract_cooccurrence_windows(texts, [concept1, concept2], text_ids)
        
        contexts = []
        
        for window in windows:
            if concept1 in window.concepts and concept2 in window.concepts:
                contexts.append(window.window_text)
                
                if len(contexts) >= max_contexts:
                    break
        
        return contexts
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do calculador PMI"""
        
        return {
            'window_size': self.window_size,
            'min_cooccurrence': self.min_cooccurrence,
            'min_concept_frequency': self.min_concept_frequency,
            'texts_processed': self.stats['texts_processed'],
            'total_windows': self.stats['total_windows'],
            'concepts_found': self.stats['concepts_found'],
            'concept_pairs_analyzed': self.stats['concept_pairs_analyzed'],
            'valid_pmi_scores': self.stats['valid_pmi_scores'],
            'processing_time': self.stats['processing_time'],
            'avg_windows_per_text': (self.stats['total_windows'] / self.stats['texts_processed']
                                   if self.stats['texts_processed'] > 0 else 0)
        }


# Funções auxiliares para uso direto
def calculate_concept_pmi(texts: List[str], concept1: str, concept2: str) -> float:
    """
    Calcula PMI entre dois conceitos específicos
    
    Args:
        texts: Lista de textos
        concept1: Primeiro conceito
        concept2: Segundo conceito
        
    Returns:
        Score PMI
    """
    calculator = PMICalculator()
    scores = calculator.calculate_pmi_for_concept_pairs(texts, [(concept1, concept2)])
    
    return scores[0].pmi_score if scores else 0.0


def find_concept_associations(texts: List[str], concepts: List[str], 
                            top_k: int = 10) -> List[Tuple[str, str, float]]:
    """
    Encontra top associações entre conceitos
    
    Args:
        texts: Lista de textos
        concepts: Lista de conceitos
        top_k: Número de top associações
        
    Returns:
        Lista de (conceito1, conceito2, pmi_score)
    """
    calculator = PMICalculator()
    strongest = calculator.find_strongest_associations(texts, concepts, top_k)
    
    return [(s.concept1, s.concept2, s.pmi_score) for s in strongest]


def analyze_concept_network(texts: List[str], concepts: List[str]) -> Dict[str, Any]:
    """
    Analisa rede de associações entre conceitos
    
    Args:
        texts: Lista de textos
        concepts: Lista de conceitos
        
    Returns:
        Análise da rede de conceitos
    """
    calculator = PMICalculator()
    
    # Calcula matriz PMI
    pmi_matrix = calculator.calculate_pmi_matrix(texts, concepts)
    
    # Análise da matriz
    analysis = {
        'concepts': concepts,
        'pmi_matrix': pmi_matrix.tolist(),
        'avg_pmi': float(np.mean(pmi_matrix[pmi_matrix > 0])) if (pmi_matrix > 0).any() else 0.0,
        'max_pmi': float(np.max(pmi_matrix)),
        'num_positive_associations': int(np.sum(pmi_matrix > 0)) // 2,  # Divide por 2 (matriz simétrica)
        'density': float(np.sum(pmi_matrix > 0) / (len(concepts) ** 2)),
        'stats': calculator.get_stats()
    }
    
    return analysis