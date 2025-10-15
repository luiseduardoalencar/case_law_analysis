# -*- coding: utf-8 -*-
"""
Extrator de conceitos jurídicos dos documentos
Identifica conceitos relevantes como "vício de consentimento", "dano moral", etc.
Combina lista predefinida com descoberta automática via TF-IDF
"""

import re
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict, Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from loguru import logger

from ..models.nodes import CONCEITOS_JURIDICOS_PREDEFINIDOS


@dataclass
class ConceptMatch:
    """Representa um conceito encontrado"""
    term: str  # Termo do conceito
    normalized_term: str  # Termo normalizado
    frequency: int  # Frequência no corpus
    document_frequency: int  # Em quantos documentos aparece
    tfidf_score: float  # Score TF-IDF médio
    category: Optional[str] = None  # Categoria jurídica
    contexts: List[str] = None  # Contextos onde aparece
    confidence: float = 1.0  # Confiança (0.0-1.0)


@dataclass
class ConceptCategory:
    """Categoria de conceitos jurídicos"""
    name: str
    keywords: List[str]  # Palavras-chave que identificam a categoria
    concepts: List[str]  # Conceitos desta categoria
    weight: float = 1.0  # Peso da categoria


class ConceptExtractor:
    """Extrator de conceitos jurídicos"""
    
    def __init__(self):
        self.predefined_concepts = CONCEITOS_JURIDICOS_PREDEFINIDOS.copy()
        self.concept_categories = self._setup_concept_categories()
        self.stopwords = self._setup_stopwords()
        
        # Padrões para identificar conceitos compostos
        self.concept_patterns = self._setup_concept_patterns()
        
        # Estatísticas
        self.stats = {
            'corpus_processed': 0,
            'documents_analyzed': 0,
            'concepts_found': 0,
            'predefined_concepts_found': 0,
            'discovered_concepts': 0,
            'concepts_by_category': defaultdict(int)
        }
        
        logger.info("⚖️ ConceptExtractor inicializado")
    
    def _setup_concept_categories(self) -> List[ConceptCategory]:
        """Define categorias de conceitos jurídicos"""
        
        categories = [
            ConceptCategory(
                name="Empréstimo Consignado",
                keywords=["consignado", "consignação", "folha", "desconto", "margem"],
                concepts=[
                    "empréstimo consignado",
                    "consignação em folha",
                    "desconto em folha de pagamento", 
                    "margem consignável",
                    "servidor público",
                    "aposentado",
                    "pensionista",
                    "INSS",
                    "contracheque"
                ],
                weight=2.0  # Peso alto - tema central
            ),
            
            ConceptCategory(
                name="Vícios e Fraudes",
                keywords=["vício", "fraude", "dolo", "erro", "coação", "má-fé"],
                concepts=[
                    "vício de consentimento",
                    "fraude",
                    "dolo",
                    "erro",
                    "coação",
                    "má-fé",
                    "boa-fé",
                    "lesão enorme",
                    "estado de perigo",
                    "prática abusiva"
                ],
                weight=1.8
            ),
            
            ConceptCategory(
                name="Danos e Reparação", 
                keywords=["dano", "indenização", "reparação", "ressarcimento", "restituição"],
                concepts=[
                    "dano moral",
                    "dano material", 
                    "danos emergentes",
                    "lucros cessantes",
                    "repetição de indébito",
                    "repetição em dobro",
                    "restituição",
                    "ressarcimento",
                    "indenização"
                ],
                weight=1.7
            ),
            
            ConceptCategory(
                name="Direito Bancário",
                keywords=["banco", "instituição", "financeira", "crédito", "juros"],
                concepts=[
                    "instituição financeira",
                    "banco",
                    "contrato bancário",
                    "taxa de juros",
                    "juros abusivos",
                    "capitalização",
                    "anatocismo",
                    "spread bancário",
                    "tarifa bancária"
                ],
                weight=1.6
            ),
            
            ConceptCategory(
                name="Direito do Consumidor",
                keywords=["consumidor", "fornecedor", "relação", "vulnerabilidade"],
                concepts=[
                    "relação de consumo",
                    "código de defesa do consumidor",
                    "CDC",
                    "vulnerabilidade",
                    "hipossuficiência", 
                    "prática abusiva",
                    "cláusula abusiva",
                    "publicidade enganosa",
                    "venda casada"
                ],
                weight=1.5
            ),
            
            ConceptCategory(
                name="Processo Civil",
                keywords=["tutela", "liminar", "sentença", "recurso", "execução"],
                concepts=[
                    "tutela antecipada",
                    "liminar",
                    "sentença",
                    "acórdão", 
                    "recurso",
                    "apelação",
                    "embargos",
                    "agravo",
                    "execução",
                    "cumprimento de sentença"
                ],
                weight=1.2
            )
        ]
        
        # Adiciona conceitos das categorias à lista predefinida
        for category in categories:
            for concept in category.concepts:
                if concept not in self.predefined_concepts:
                    self.predefined_concepts.append(concept)
        
        return categories
    
    def _setup_stopwords(self) -> Set[str]:
        """Configura stopwords personalizadas para domínio jurídico"""
        
        try:
            portuguese_stopwords = set(stopwords.words('portuguese'))
        except LookupError:
            logger.warning("⚠️ NLTK stopwords não encontradas, usando lista básica")
            portuguese_stopwords = {
                'a', 'o', 'e', 'de', 'do', 'da', 'em', 'um', 'uma', 'com',
                'não', 'que', 'por', 'para', 'se', 'na', 'no', 'é', 'são'
            }
        
        # Adiciona stopwords jurídicas específicas
        juridical_stopwords = {
            # Artigos e conectivos jurídicos
            'art', 'artigo', 'inc', 'inciso', 'parágrafo', 'alínea', 'item',
            'cf', 'cfr', 'vide', 'ver', 'pág', 'página', 'fls', 'folhas',
            
            # Títulos e tratamentos
            'dr', 'dra', 'des', 'desembargador', 'desembargadora', 
            'mm', 'meritíssimo', 'dd', 'digníssimo', 'exmo', 'exma',
            
            # Termos processuais muito genéricos
            'processo', 'autos', 'ref', 'referente', 'rel', 'relator',
            'rev', 'revisor', 'red', 'redator', 'proc',
            
            # Palavras muito comuns em textos jurídicos
            'assim', 'desta', 'neste', 'dessa', 'nessa', 'deste',
            'portanto', 'entanto', 'quanto', 'quando', 'onde',
            'sendo', 'tendo', 'havendo', 'devendo'
        }
        
        return portuguese_stopwords | juridical_stopwords
    
    def _setup_concept_patterns(self) -> List[str]:
        """Define padrões regex para identificar conceitos compostos"""
        
        return [
            # Conceitos com "de" no meio
            r'\b([a-záêçõâô]+\s+de\s+[a-záêçõâô]+(?:\s+[a-záêçõâô]+)*)\b',
            
            # Conceitos com "em" no meio  
            r'\b([a-záêçõâô]+\s+em\s+[a-záêçõâô]+(?:\s+[a-záêçõâô]+)*)\b',
            
            # Conceitos com "do/da" no meio
            r'\b([a-záêçõâô]+\s+d[ao]\s+[a-záêçõâô]+(?:\s+[a-záêçõâô]+)*)\b',
            
            # Conceitos adjetivados
            r'\b([a-záêçõâô]+\s+(?:abusiv[oa]|excessiv[oa]|indevid[oa]|ilícit[oa]))\b',
            
            # Conceitos com números/códigos
            r'\b([a-záêçõâô]+\s+n[º°]\s*\d+)\b'
        ]
    
    def extract_concepts_from_corpus(self, texts: List[str], 
                                document_ids: List[str] = None,
                                max_candidates: int = 500,
                                batch_size: int = 100) -> List[Dict[str, Any]]:
        """
        Extrai conceitos de um corpus de documentos (VERSÃO OTIMIZADA)
        
        Args:
            texts: Lista de textos dos documentos
            document_ids: IDs dos documentos (opcional)
            max_candidates: Máximo de candidatos a descobrir (padrão: 200)
            batch_size: Tamanho do lote para processamento (padrão: 100)
            
        Returns:
            Lista de dicionários com dados dos conceitos
        """
        
        if not texts:
            return []
        
        if document_ids is None:
            document_ids = [f"doc_{i}" for i in range(len(texts))]
        
        self.stats['corpus_processed'] += 1
        self.stats['documents_analyzed'] = len(texts)
        
        logger.info(f"🔍 Extraindo conceitos de {len(texts)} documentos...")
        logger.info(f"📦 Usando batches de {batch_size} documentos")
        logger.info(f"🎯 Limite de {max_candidates} candidatos descobertos")
        
        try:
            # 1. Identifica conceitos predefinidos (RÁPIDO)
            logger.info("🔎 Buscando conceitos predefinidos...")
            predefined_concepts = self._find_predefined_concepts_optimized(
                texts, document_ids, batch_size
            )
            logger.info(f"✅ {len(predefined_concepts)} conceitos predefinidos encontrados")
            
            # 2. Descobre novos conceitos via TF-IDF (OTIMIZADO)
            logger.info("🔬 Descobrindo novos conceitos...")
            discovered_concepts = self._discover_new_concepts_optimized(
                texts, document_ids, max_candidates, batch_size
            )
            logger.info(f"✅ {len(discovered_concepts)} novos conceitos descobertos")
            
            # 3. Combina e filtra conceitos
            logger.info("🔗 Combinando e filtrando conceitos...")
            all_concepts = predefined_concepts + discovered_concepts
            final_concepts = self._filter_and_rank_concepts(all_concepts, texts)
            
            # 4. Categoriza conceitos
            logger.info("🏷️  Categorizando conceitos...")
            categorized_concepts = self._categorize_concepts(final_concepts)
            
            # Atualiza estatísticas
            self.stats['concepts_found'] = len(final_concepts)
            self.stats['predefined_concepts_found'] = len(predefined_concepts)
            self.stats['discovered_concepts'] = len(discovered_concepts)
            
            logger.info(f"✅ {len(final_concepts)} conceitos extraídos TOTAL")
            
            return categorized_concepts
            
        except Exception as e:
            logger.error(f"❌ Erro na extração de conceitos: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _find_predefined_concepts(self, texts: List[str], 
                                document_ids: List[str]) -> List[ConceptMatch]:
        """Encontra conceitos da lista predefinida"""
        
        concepts_found = {}
        
        for concept_term in self.predefined_concepts:
            # Cria padrão regex para o conceito
            # Permite variações de plural/singular e flexões
            pattern = self._create_concept_pattern(concept_term)
            
            total_freq = 0
            doc_freq = 0
            contexts = []
            
            for text, doc_id in zip(texts, document_ids):
                matches = list(re.finditer(pattern, text, re.IGNORECASE))
                
                if matches:
                    doc_freq += 1
                    total_freq += len(matches)
                    
                    # Coleta contextos (limitado)
                    for match in matches[:2]:  # Max 2 contextos por documento
                        context = self._extract_context(text, match.start(), match.end())
                        if context and len(contexts) < 5:  # Max 5 contextos total
                            contexts.append(context)
            
            if total_freq > 0:
                concepts_found[concept_term] = ConceptMatch(
                    term=concept_term,
                    normalized_term=self._normalize_concept_term(concept_term),
                    frequency=total_freq,
                    document_frequency=doc_freq,
                    tfidf_score=0.0,  # Será calculado depois
                    contexts=contexts,
                    confidence=1.0  # Confiança máxima para predefinidos
                )
        
        return list(concepts_found.values())
    
    def _discover_new_concepts(self, texts: List[str], 
                             document_ids: List[str]) -> List[ConceptMatch]:
        """Descobre novos conceitos usando TF-IDF e padrões"""
        
        # Preprocessa textos
        processed_texts = [self._preprocess_text_for_tfidf(text) for text in texts]
        
        # Extrai candidatos a conceitos usando padrões
        concept_candidates = self._extract_concept_candidates(processed_texts)
        
        if not concept_candidates:
            return []
        
        # Calcula TF-IDF para os candidatos
        tfidf_scores = self._calculate_tfidf_for_candidates(
            processed_texts, concept_candidates
        )
        
        # Filtra e cria ConceptMatch
        discovered = []
        min_tfidf = 0.1  # Threshold mínimo
        min_doc_freq = 2  # Deve aparecer em pelo menos 2 documentos
        
        for candidate, (freq, doc_freq, avg_tfidf) in tfidf_scores.items():
            if avg_tfidf >= min_tfidf and doc_freq >= min_doc_freq:
                
                # Coleta contextos
                contexts = self._collect_contexts_for_candidate(
                    texts, candidate, max_contexts=3
                )
                
                # Calcula confiança baseada em características
                confidence = self._calculate_concept_confidence(
                    candidate, freq, doc_freq, avg_tfidf, len(texts)
                )
                
                if confidence >= 0.5:  # Threshold de confiança
                    discovered.append(ConceptMatch(
                        term=candidate,
                        normalized_term=self._normalize_concept_term(candidate),
                        frequency=freq,
                        document_frequency=doc_freq,
                        tfidf_score=avg_tfidf,
                        contexts=contexts,
                        confidence=confidence
                    ))
        
        # Ordena por TF-IDF e limita
        discovered.sort(key=lambda x: x.tfidf_score, reverse=True)
        return discovered[:50]  # Máximo 50 conceitos descobertos
    
    def _create_concept_pattern(self, concept_term: str) -> str:
        """Cria padrão regex para um conceito, permitindo variações"""
        
        # Escapa caracteres especiais
        escaped = re.escape(concept_term)
        
        # Permite variações de plural/singular
        # Ex: "dano moral" -> "danos? morais?"
        words = escaped.split('\\ ')  # Split por espaço escapado
        pattern_words = []
        
        for word in words:
            # Adiciona opcional 's' no final para plural
            if word.endswith('o'):
                pattern_words.append(word + 's?')
            elif word.endswith('a'):
                pattern_words.append(word + 's?')  
            elif word.endswith('l'):
                pattern_words.append(word.replace('l', '(?:l|is)'))  # moral -> morais
            else:
                pattern_words.append(word + 's?')
        
        # Junta com espaços opcionais
        pattern = r'\b' + r'\s+'.join(pattern_words) + r'\b'
        
        return pattern
    
    def _preprocess_text_for_tfidf(self, text: str) -> str:
        """Preprocessa texto para análise TF-IDF"""
        
        # Converte para minúsculas
        text = text.lower()
        
        # Remove pontuação excessiva, mantendo hífens em palavras compostas
        text = re.sub(r'[^\w\s\-]', ' ', text)
        
        # Normaliza espaços
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _extract_concept_candidates(self, texts: List[str]) -> Set[str]:
        """Extrai candidatos a conceitos usando padrões"""
        
        candidates = set()
        
        for text in texts:
            # Usa padrões regex para encontrar conceitos compostos
            for pattern in self.concept_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0]  # Primeiro grupo
                    
                    candidate = match.strip().lower()
                    
                    # Valida candidato
                    if self._is_valid_concept_candidate(candidate):
                        candidates.add(candidate)
            
            # Também extrai bigramas e trigramas relevantes
            words = text.split()
            for i in range(len(words) - 1):
                bigram = ' '.join(words[i:i+2])
                if self._is_valid_concept_candidate(bigram):
                    candidates.add(bigram)
                
                if i < len(words) - 2:
                    trigram = ' '.join(words[i:i+3])
                    if self._is_valid_concept_candidate(trigram):
                        candidates.add(trigram)
        
        return candidates
    
    def _is_valid_concept_candidate(self, candidate: str) -> bool:
        """Valida se um candidato pode ser um conceito"""
        
        # Filtros básicos
        if len(candidate) < 5:  # Muito curto
            return False
        
        if len(candidate) > 50:  # Muito longo
            return False
        
        words = candidate.split()
        if len(words) < 2 or len(words) > 4:  # Fora do range ideal
            return False
        
        # Remove se contém muitos números
        if sum(1 for c in candidate if c.isdigit()) > len(candidate) * 0.3:
            return False
        
        # Remove se é só stopwords
        if all(word in self.stopwords for word in words):
            return False
        
        # Remove se já está na lista predefinida
        if candidate in [c.lower() for c in self.predefined_concepts]:
            return False
        
        # Remove padrões irrelevantes
        irrelevant_patterns = [
            r'^\d+',  # Começa com número
            r'[a-z]\s[a-z]$',  # Duas letras isoladas
            r'^[a-z]{1,2}\s',  # Começa com 1-2 letras isoladas
        ]
        
        for pattern in irrelevant_patterns:
            if re.match(pattern, candidate):
                return False
        
        return True
    
    def _calculate_tfidf_for_candidates(self, texts: List[str], 
                                      candidates: Set[str]) -> Dict[str, Tuple[int, int, float]]:
        """Calcula TF-IDF para candidatos a conceitos"""
        
        if not candidates:
            return {}
        
        # Conta frequências
        concept_stats = {}
        
        for candidate in candidates:
            total_freq = 0
            doc_freq = 0
            tfidf_scores = []
            
            for text in texts:
                # Conta ocorrências no texto
                pattern = r'\b' + re.escape(candidate) + r'\b'
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                
                if matches > 0:
                    doc_freq += 1
                    total_freq += matches
                    
                    # Calcula TF-IDF básico
                    tf = matches / len(text.split())  # Term frequency
                    idf = np.log(len(texts) / doc_freq)  # Inverse document frequency
                    tfidf_scores.append(tf * idf)
            
            if tfidf_scores:
                avg_tfidf = np.mean(tfidf_scores)
                concept_stats[candidate] = (total_freq, doc_freq, avg_tfidf)
        
        return concept_stats
    
    def _collect_contexts_for_candidate(self, texts: List[str], 
                                      candidate: str, 
                                      max_contexts: int = 3) -> List[str]:
        """Coleta contextos onde um candidato aparece"""
        
        contexts = []
        pattern = r'\b' + re.escape(candidate) + r'\b'
        
        for text in texts:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            
            for match in matches[:2]:  # Max 2 por documento
                if len(contexts) >= max_contexts:
                    break
                    
                context = self._extract_context(text, match.start(), match.end(), window=60)
                if context:
                    contexts.append(context)
            
            if len(contexts) >= max_contexts:
                break
        
        return contexts
    
    def _calculate_concept_confidence(self, candidate: str, freq: int, 
                                    doc_freq: int, avg_tfidf: float, 
                                    total_docs: int) -> float:
        """Calcula confiança de um conceito descoberto"""
        
        confidence = 0.5  # Base
        
        # Boost por TF-IDF alto
        if avg_tfidf > 0.2:
            confidence += 0.2
        elif avg_tfidf > 0.1:
            confidence += 0.1
        
        # Boost por aparecer em muitos documentos
        doc_ratio = doc_freq / total_docs
        if doc_ratio > 0.1:  # Mais de 10% dos documentos
            confidence += 0.2
        elif doc_ratio > 0.05:  # Mais de 5% dos documentos
            confidence += 0.1
        
        # Boost por conter palavras jurídicas importantes
        juridical_terms = [
            'direito', 'jurídico', 'legal', 'civil', 'penal', 'processo',
            'contrato', 'responsabilidade', 'dano', 'indenização',
            'consignado', 'banco', 'empréstimo', 'fraude'
        ]
        
        if any(term in candidate.lower() for term in juridical_terms):
            confidence += 0.1
        
        # Penaliza conceitos muito genéricos
        generic_terms = ['processo', 'documento', 'caso', 'situação', 'questão']
        if any(term in candidate.lower() for term in generic_terms):
            confidence -= 0.2
        
        return min(1.0, max(0.0, confidence))
    
    def _extract_context(self, text: str, start: int, end: int, 
                        window: int = 50) -> str:
        """Extrai contexto ao redor de uma posição no texto"""
        
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        
        context = text[context_start:context_end].strip()
        
        # Limpa e normaliza
        context = re.sub(r'\s+', ' ', context)
        
        if len(context) > 150:  # Trunca se muito longo
            context = context[:150] + "..."
        
        return context
    
    def _filter_and_rank_concepts(self, concepts: List[ConceptMatch], 
                                texts: List[str]) -> List[ConceptMatch]:
        """Filtra e ranqueia conceitos finais"""
        
        if not concepts:
            return []
        
        # Remove duplicatas por termo normalizado
        unique_concepts = {}
        for concept in concepts:
            key = concept.normalized_term
            
            if key not in unique_concepts:
                unique_concepts[key] = concept
            else:
                # Mantém o de maior confiança
                if concept.confidence > unique_concepts[key].confidence:
                    unique_concepts[key] = concept
        
        filtered_concepts = list(unique_concepts.values())
        
        # Calcula TF-IDF para conceitos predefinidos que não tinham
        for concept in filtered_concepts:
            if concept.tfidf_score == 0.0:
                concept.tfidf_score = self._calculate_tfidf_for_concept(
                    concept.term, texts
                )
        
        # Ordena por relevância (combinação de confiança e TF-IDF)
        filtered_concepts.sort(
            key=lambda x: (x.confidence * 0.6 + min(x.tfidf_score, 1.0) * 0.4),
            reverse=True
        )
        
        return filtered_concepts
    
    def _calculate_tfidf_for_concept(self, concept_term: str, 
                                   texts: List[str]) -> float:
        """Calcula TF-IDF para um conceito específico"""
        
        pattern = self._create_concept_pattern(concept_term)
        
        tfidf_scores = []
        doc_freq = 0
        
        for text in texts:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            
            if matches > 0:
                doc_freq += 1
                tf = matches / len(text.split())
                tfidf_scores.append(tf)
        
        if not tfidf_scores:
            return 0.0
        
        # IDF
        idf = np.log(len(texts) / doc_freq) if doc_freq > 0 else 0
        
        # TF-IDF médio
        avg_tf = np.mean(tfidf_scores)
        return avg_tf * idf
    
    def _categorize_concepts(self, concepts: List[ConceptMatch]) -> List[Dict[str, Any]]:
        """Categoriza conceitos e converte para formato final"""
        
        categorized = []
        
        for concept in concepts:
            # Determina categoria
            category = self._determine_concept_category(concept.normalized_term)
            
            concept_data = {
                'termo': concept.normalized_term,
                'termo_original': concept.term,
                'categoria': category,
                'frequencia': concept.frequency,
                'frequencia_documentos': concept.document_frequency,
                'score_tfidf': round(concept.tfidf_score, 4),
                'confianca': round(concept.confidence, 3),
                'contextos': concept.contexts or []
            }
            
            categorized.append(concept_data)
            
            # Atualiza estatísticas por categoria
            if category:
                self.stats['concepts_by_category'][category] += 1
        
        return categorized
    
    def _determine_concept_category(self, concept_term: str) -> Optional[str]:
        """Determina a categoria de um conceito"""
        
        concept_lower = concept_term.lower()
        
        # Verifica categorias por palavras-chave
        for category in self.concept_categories:
            # Verifica se o conceito está na lista da categoria
            if concept_term in [c.lower() for c in category.concepts]:
                return category.name
            
            # Verifica se contém palavras-chave da categoria
            if any(keyword in concept_lower for keyword in category.keywords):
                return category.name
        
        return None
    
    def _normalize_concept_term(self, term: str) -> str:
        """Normaliza termo do conceito"""
        
        # Remove espaços extras
        normalized = re.sub(r'\s+', ' ', term.strip())
        
        # Converte para lowercase
        normalized = normalized.lower()
        
        # Normaliza alguns termos específicos
        replacements = {
            'código de defesa do consumidor': 'CDC',
            'código civil': 'CC',
            'código de processo civil': 'CPC',
            'tribunal de justiça': 'TJ',
            'superior tribunal de justiça': 'STJ',
            'supremo tribunal federal': 'STF'
        }
        
        for old, new in replacements.items():
            if old in normalized:
                normalized = normalized.replace(old, new)
        
        return normalized
    
    def _find_predefined_concepts_optimized(self, texts: List[str], 
                                           document_ids: List[str],
                                           batch_size: int = 100) -> List[ConceptMatch]:
        """Encontra conceitos predefinidos (OTIMIZADO com batching)"""
        
        concepts_found = {}
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        logger.info(f"   Processando {total_batches} lotes...")
        
        for concept_term in self.predefined_concepts:
            pattern = self._create_concept_pattern(concept_term)
            
            total_freq = 0
            doc_freq = 0
            contexts = []
            
            # PROCESSA EM LOTES
            for batch_idx in range(0, len(texts), batch_size):
                batch_texts = texts[batch_idx:batch_idx + batch_size]
                batch_ids = document_ids[batch_idx:batch_idx + batch_size]
                
                if batch_idx % (batch_size * 5) == 0:
                    logger.debug(f"   Lote {batch_idx//batch_size + 1}/{total_batches}")
                
                for text, doc_id in zip(batch_texts, batch_ids):
                    matches = list(re.finditer(pattern, text, re.IGNORECASE))
                    
                    if matches:
                        doc_freq += 1
                        total_freq += len(matches)
                        
                        for match in matches[:1]:
                            if len(contexts) < 3:
                                context = self._extract_context(text, match.start(), match.end())
                                if context:
                                    contexts.append(context)
            
            if total_freq > 0:
                concepts_found[concept_term] = ConceptMatch(
                    term=concept_term,
                    normalized_term=self._normalize_concept_term(concept_term),
                    frequency=total_freq,
                    document_frequency=doc_freq,
                    tfidf_score=0.0,
                    contexts=contexts,
                    confidence=1.0
                )
        
        return list(concepts_found.values())
    
    def _discover_new_concepts_optimized(self, texts: List[str], 
                                        document_ids: List[str],
                                        max_candidates: int = 200,
                                        batch_size: int = 100) -> List[ConceptMatch]:
        """Descobre novos conceitos (VERSÃO OTIMIZADA)"""
        
        logger.info(f"   Extraindo candidatos (máx: {max_candidates})...")
        
        # AMOSTRAGEM: Usa apenas uma amostra dos textos
        sample_size = min(len(texts), 1500)
        if len(texts) > sample_size:
            import random
            sample_indices = random.sample(range(len(texts)), sample_size)
            sample_texts = [texts[i] for i in sample_indices]
            logger.info(f"   Usando amostra de {sample_size} documentos")
        else:
            sample_texts = texts
        
        processed_texts = [self._preprocess_text_for_tfidf(text) for text in sample_texts]
        
        concept_candidates = self._extract_concept_candidates_fast(
            processed_texts, max_candidates
        )
        
        if not concept_candidates:
            logger.warning("   Nenhum candidato encontrado")
            return []
        
        logger.info(f"   {len(concept_candidates)} candidatos extraídos")
        logger.info(f"   Calculando TF-IDF...")
        
        tfidf_scores = self._calculate_tfidf_for_candidates_fast(
            processed_texts, concept_candidates
        )
        
        discovered = []
        min_tfidf = 0.12
        min_doc_freq = 2
        
        for candidate, (freq, doc_freq, avg_tfidf) in tfidf_scores.items():
            if avg_tfidf >= min_tfidf and doc_freq >= min_doc_freq:
                contexts = []
                
                confidence = self._calculate_concept_confidence(
                    candidate, freq, doc_freq, avg_tfidf, len(sample_texts)
                )
                
                if confidence >= 0.6:
                    discovered.append(ConceptMatch(
                        term=candidate,
                        normalized_term=self._normalize_concept_term(candidate),
                        frequency=freq,
                        document_frequency=doc_freq,
                        tfidf_score=avg_tfidf,
                        contexts=contexts,
                        confidence=confidence
                    ))
        
        discovered.sort(key=lambda x: x.tfidf_score, reverse=True)
        return discovered[:75]
    
    def _extract_concept_candidates_fast(self, texts: List[str], 
                                         max_candidates: int = 200) -> Set[str]:
        """Extrai candidatos de forma RÁPIDA (OTIMIZADO)"""
        
        candidates = set()
        candidate_freq = Counter()
        
        for text in texts:
            words = text.split()
            
            # Bigramas
            for i in range(len(words) - 1):
                if len(candidates) >= max_candidates * 2:
                    break
                
                bigram = ' '.join(words[i:i+2])
                if self._is_valid_concept_candidate_fast(bigram):
                    candidates.add(bigram)
                    candidate_freq[bigram] += 1
            
            # Trigramas
            for i in range(len(words) - 2):
                if len(candidates) >= max_candidates * 3:
                    break
                
                trigram = ' '.join(words[i:i+3])
                if self._is_valid_concept_candidate_fast(trigram):
                    candidates.add(trigram)
                    candidate_freq[trigram] += 1
            
            # Quadrigramas (4 palavras)
            for i in range(len(words) - 3):
                if len(candidates) >= max_candidates * 2:
                    break
                
                quadrigram = ' '.join(words[i:i+4])
                if self._is_valid_concept_candidate_fast(quadrigram):
                    candidates.add(quadrigram)
                    candidate_freq[quadrigram] += 1
        
        top_candidates = [
            cand for cand, freq in candidate_freq.most_common(max_candidates)
        ]
        
        logger.info(f"   {len(top_candidates)}/{len(candidates)} candidatos selecionados")
        return set(top_candidates)
    
    def _is_valid_concept_candidate_fast(self, candidate: str) -> bool:
        """Validação RÁPIDA de candidato"""
        
        if len(candidate) < 6 or len(candidate) > 60:
            return False
        
        words = candidate.split()
        if len(words) < 2 or len(words) > 4:
            return False
        
        if sum(1 for c in candidate if c.isdigit()) > 4:
            return False
        
        if all(word in self.stopwords for word in words):
            return False
        
        return True
    
    def _calculate_tfidf_for_candidates_fast(self, texts: List[str], 
                                            candidates: Set[str]) -> Dict[str, Tuple[int, int, float]]:
        """Calcula TF-IDF RÁPIDO (OTIMIZADO)"""
        
        if not candidates:
            return {}
        
        concept_stats = {}
        total_words = sum(len(text.split()) for text in texts)
        
        for candidate in candidates:
            total_freq = 0
            doc_freq = 0
            
            for text in texts:
                count = text.count(candidate)
                
                if count > 0:
                    doc_freq += 1
                    total_freq += count
            
            if doc_freq > 0:
                tf = total_freq / total_words
                idf = np.log(len(texts) / doc_freq)
                avg_tfidf = tf * idf
                
                concept_stats[candidate] = (total_freq, doc_freq, avg_tfidf)
        
        return concept_stats

    def get_extraction_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas da extração"""
        
        return {
            'corpus_processed': self.stats['corpus_processed'],
            'documents_analyzed': self.stats['documents_analyzed'],
            'total_concepts_found': self.stats['concepts_found'],
            'predefined_concepts_found': self.stats['predefined_concepts_found'],
            'discovered_concepts': self.stats['discovered_concepts'],
            'concepts_by_category': dict(self.stats['concepts_by_category']),
            'predefined_concepts_available': len(self.predefined_concepts),
            'categories_configured': len(self.concept_categories)
        }
    
    def get_predefined_concepts(self) -> List[str]:
        """Retorna lista de conceitos predefinidos"""
        return self.predefined_concepts.copy()
    
    def add_custom_concepts(self, concepts: List[str], category: str = None):
        """Adiciona conceitos customizados"""
        
        for concept in concepts:
            if concept not in self.predefined_concepts:
                self.predefined_concepts.append(concept)
        
        logger.info(f"➕ {len(concepts)} conceitos customizados adicionados")


# Funções auxiliares para uso direto
def extract_concepts_from_texts(texts: List[str]) -> List[Dict[str, Any]]:
    """Função auxiliar para extrair conceitos de uma lista de textos"""
    extractor = ConceptExtractor()
    return extractor.extract_concepts_from_corpus(texts)


def extract_predefined_concepts_only(texts: List[str]) -> List[Dict[str, Any]]:
    """Extrai apenas conceitos predefinidos (mais rápido)"""
    extractor = ConceptExtractor()
    
    # Simula processo apenas com predefinidos
    document_ids = [f"doc_{i}" for i in range(len(texts))]
    predefined_concepts = extractor._find_predefined_concepts(texts, document_ids)
    
    # Converte para formato final
    result = []
    for concept in predefined_concepts:
        result.append({
            'termo': concept.normalized_term,
            'termo_original': concept.term,
            'categoria': extractor._determine_concept_category(concept.normalized_term),
            'frequencia': concept.frequency,
            'frequencia_documentos': concept.document_frequency,
            'score_tfidf': 0.0,  # Não calculado neste modo
            'confianca': concept.confidence,
            'contextos': concept.contexts or []
        })
    
    return result


def get_concept_categories() -> List[Dict[str, Any]]:
    """Retorna informações sobre as categorias de conceitos"""
    extractor = ConceptExtractor()
    
    categories_info = []
    for category in extractor.concept_categories:
        categories_info.append({
            'nome': category.name,
            'peso': category.weight,
            'palavras_chave': category.keywords,
            'conceitos': category.concepts,
            'num_conceitos': len(category.concepts)
        })
    
    return categories_info


def find_concepts_in_text(text: str, only_predefined: bool = False) -> List[str]:
    """
    Encontra conceitos em um único texto (função simples para testes)
    
    Args:
        text: Texto para analisar
        only_predefined: Se True, busca apenas conceitos predefinidos
        
    Returns:
        Lista de conceitos encontrados
    """
    
    if only_predefined:
        concepts_data = extract_predefined_concepts_only([text])
    else:
        concepts_data = extract_concepts_from_texts([text])
    
    return [c['termo'] for c in concepts_data if c['confianca'] > 0.7]


def analyze_concept_coverage(texts: List[str]) -> Dict[str, Any]:
    """
    Analisa cobertura de conceitos em um corpus
    
    Returns:
        Relatório de cobertura de conceitos
    """
    extractor = ConceptExtractor()
    concepts = extractor.extract_concepts_from_corpus(texts)
    
    # Agrupa por categoria
    by_category = defaultdict(list)
    total_frequency = 0
    
    for concept in concepts:
        category = concept.get('categoria', 'Sem Categoria')
        by_category[category].append(concept)
        total_frequency += concept['frequencia']
    
    # Calcula estatísticas
    coverage_report = {
        'total_concepts': len(concepts),
        'total_frequency': total_frequency,
        'avg_frequency_per_concept': total_frequency / len(concepts) if concepts else 0,
        'categories': {},
        'top_concepts': sorted(concepts, key=lambda x: x['score_tfidf'], reverse=True)[:10],
        'most_frequent': sorted(concepts, key=lambda x: x['frequencia'], reverse=True)[:10],
        'predefined_vs_discovered': {
            'predefined': len([c for c in concepts if c['confianca'] == 1.0]),
            'discovered': len([c for c in concepts if c['confianca'] < 1.0])
        }
    }
    
    # Estatísticas por categoria
    for category, category_concepts in by_category.items():
        coverage_report['categories'][category] = {
            'count': len(category_concepts),
            'total_frequency': sum(c['frequencia'] for c in category_concepts),
            'avg_tfidf': sum(c['score_tfidf'] for c in category_concepts) / len(category_concepts),
            'top_concept': max(category_concepts, key=lambda x: x['score_tfidf'])['termo']
        }
    
    return coverage_report