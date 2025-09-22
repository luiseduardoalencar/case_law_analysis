# -*- coding: utf-8 -*-
"""
Extrator de entidades nomeadas para documentos jurídicos
Identifica juízes, advogados, comarcas, leis citadas, órgãos, etc.
Combina spaCy base com regras customizadas para o domínio jurídico
"""

import re
from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
import spacy
from loguru import logger

from ..models.nodes import EntityType


@dataclass
class EntityMatch:
    """Representa uma entidade encontrada"""
    text: str  # Texto original encontrado
    normalized: str  # Texto normalizado
    entity_type: EntityType
    start: int  # Posição inicial no texto
    end: int  # Posição final no texto
    confidence: float = 1.0  # Confiança (0.0-1.0)
    pattern_used: Optional[str] = None  # Padrão regex usado


@dataclass
class EntityPattern:
    """Padrão para identificar uma entidade"""
    entity_type: EntityType
    patterns: List[str]  # Regex patterns
    normalization_rules: Optional[List[Tuple[str, str]]] = None  # (from, to)
    min_confidence: float = 0.7
    context_boost: Optional[List[str]] = None  # Palavras que aumentam confiança


class NERExtractor:
    """Extrator de entidades nomeadas jurídicas"""
    
    def __init__(self, spacy_model: str = "pt_core_news_sm"):
        self.spacy_model_name = spacy_model
        self.nlp = None
        self.patterns = self._setup_entity_patterns()
        
        # Estatísticas
        self.stats = {
            'texts_processed': 0,
            'entities_found': 0,
            'entities_by_type': defaultdict(int),
            'unique_entities': set()
        }
        
        # Cache de entidades já processadas
        self.entity_cache = {}
        
        self._load_spacy_model()
        logger.info("🏛️ NERExtractor inicializado")
    
    def _load_spacy_model(self):
        """Carrega modelo spaCy"""
        try:
            self.nlp = spacy.load(self.spacy_model_name)
            logger.info(f"✅ Modelo spaCy carregado: {self.spacy_model_name}")
        except IOError:
            logger.warning(f"⚠️ Modelo {self.spacy_model_name} não encontrado, usando modelo básico")
            try:
                # Tenta modelo mais básico
                self.nlp = spacy.load("pt")
            except IOError:
                logger.error("❌ Nenhum modelo spaCy português encontrado!")
                logger.info("💡 Instale com: python -m spacy download pt_core_news_sm")
                self.nlp = None
    
    def _setup_entity_patterns(self) -> List[EntityPattern]:
        """Configura padrões para entidades jurídicas"""
        
        patterns = [
            # JUÍZES E DESEMBARGADORES
            EntityPattern(
                entity_type=EntityType.JUIZ,
                patterns=[
                    # Desembargadores
                    r'(?i)Des(?:embargador)?\.?\s+([A-Z][a-záêçõâô]+(?:\s+[A-Z][a-záêçõâô]+){1,3})',
                    r'(?i)Desª?\.?\s+([A-Z][a-záêçõâô]+(?:\s+[A-Z][a-záêçõâô]+){1,3})',
                    r'(?i)Desembargadora?\s+([A-Z][a-záêçõâô]+(?:\s+[A-Z][a-záêçõâô]+){1,3})',
                    
                    # Juízes
                    r'(?i)Juiz(?:a)?\s+([A-Z][a-záêçõâô]+(?:\s+[A-Z][a-záêçõâô]+){1,3})',
                    r'(?i)(?:Dr|Dra)\.?\s+([A-Z][a-záêçõâô]+(?:\s+[A-Z][a-záêçõâô]+){1,3})',
                    r'(?i)(?:Exmo|Exma)\.?\s+(?:Sr|Sra)\.?\s+(?:Dr|Dra)\.?\s+([A-Z][a-záêçõâô]+(?:\s+[A-Z][a-záêçõâô]+){1,3})',
                    
                    # Relatores
                    r'(?i)Relator(?:a)?:?\s+(?:Des\.?\s+|Dr\.?\s+)?([A-Z][a-záêçõâô]+(?:\s+[A-Z][a-záêçõâô]+){1,3})',
                    r'(?i)(?:Rel|Relatora?)\.?\s+([A-Z][a-záêçõâô]+(?:\s+[A-Z][a-záêçõâô]+){1,3})',
                ],
                normalization_rules=[
                    (r'(?i)^des\.?\s+', ''),  # Remove "Des."
                    (r'(?i)^dr\.?\s+', ''),   # Remove "Dr."
                    (r'(?i)^juiz(?:a)?\s+', ''),  # Remove "Juiz"
                ],
                context_boost=[
                    'relator', 'relatora', 'presidente', 'revisor',
                    'desembargador', 'desembargadora', 'magistrado'
                ]
            ),
            
            # ADVOGADOS
            EntityPattern(
                entity_type=EntityType.ADVOGADO,
                patterns=[
                    # Por nome + OAB
                    r'(?i)(?:Dr|Dra)\.?\s+([A-Z][a-záêçõâô]+(?:\s+[A-Z][a-záêçõâô]+){1,3}),?\s+OAB[/-]?(?:PI)?[/-]?\s*(\d+)',
                    r'(?i)(?:Advogado|Advogada)\s+([A-Z][a-záêçõâô]+(?:\s+[A-Z][a-záêçõâô]+){1,3})',
                    r'(?i)([A-Z][a-záêçõâô]+(?:\s+[A-Z][a-záêçõâô]+){1,3}),?\s+(?:inscrito|inscrita)\s+(?:na\s+)?OAB',
                    
                    # Só OAB
                    r'(?i)OAB[/-]?(?:PI)?[/-]?\s*(\d{3,6})',
                    r'(?i)inscri[çã][ãç]o\s+(?:na\s+)?OAB[/-]?(?:PI)?[/-]?\s*[nº]?\s*(\d{3,6})',
                ],
                normalization_rules=[
                    (r'(?i)^(?:dr|dra)\.?\s+', ''),  # Remove títulos
                    (r'(?i),?\s+oab.*$', ''),        # Remove sufixo OAB
                ],
                context_boost=[
                    'advogado', 'advogada', 'procurador', 'procuradora',
                    'causídico', 'patrono', 'defensor', 'constituído'
                ]
            ),
            
            # COMARCAS E ÓRGÃOS JUDICIÁRIOS
            EntityPattern(
                entity_type=EntityType.COMARCA,
                patterns=[
                    # Varas e Comarcas
                    r'(?i)(\d+[ªº]?\s+Vara\s+(?:Cível|Criminal|da\s+Fazenda|de\s+Família|Empresarial)(?:\s+de\s+[A-Z][a-záêçõâô]+)?)',
                    r'(?i)(Comarca\s+de\s+[A-Z][a-záêçõâô]+)',
                    r'(?i)(Foro\s+(?:Central\s+)?de\s+[A-Z][a-záêçõâô]+)',
                    r'(?i)(Tribunal\s+de\s+Justi[çã]a\s+do?\s+[A-Z][a-záêçõâô]+)',
                    r'(?i)(TJ-?(?:PI|MA|CE|BA|PE|RN|PB|AL|SE))',
                    
                    # Turmas e Câmaras
                    r'(?i)(\d+[ªº]?\s+(?:Câmara|Turma)\s+(?:Cível|Criminal)?)',
                    r'(?i)((?:Primeira|Segunda|Terceira|Quarta|Quinta)\s+(?:Câmara|Turma))',
                ],
                normalization_rules=[
                    (r'(?i)\s+de\s+teresina', ' de Teresina'),  # Normaliza Teresina
                    (r'(?i)tj-?pi', 'TJPI'),  # Normaliza TJPI
                ],
                context_boost=['vara', 'comarca', 'tribunal', 'foro', 'juízo']
            ),
            
            # LEIS E DISPOSITIVOS LEGAIS
            EntityPattern(
                entity_type=EntityType.LEI,
                patterns=[
                    # Leis federais
                    r'(?i)(Lei\s+(?:Federal\s+)?n?[º°]?\s*(\d+(?:\.\d+)*)[/-]?(\d{2,4}))',
                    r'(?i)(Lei\s+(\d+(?:\.\d+)*)[/-](\d{2,4}))',
                    
                    # Códigos
                    r'(?i)(Código\s+(?:Civil|Penal|de\s+Processo\s+Civil|de\s+Processo\s+Penal|de\s+Defesa\s+do\s+Consumidor))',
                    r'(?i)(CC[/-]?\d{2})',  # CC/02
                    r'(?i)(CPC[/-]?\d{2})', # CPC/15
                    r'(?i)(CDC)',           # CDC
                    
                    # Artigos
                    r'(?i)(art(?:igo)?s?\.?\s+(\d+(?:[º°]|[-ªº])?(?:\s*,\s*\d+)*)[º°]?(?:\s+(?:a\s+\d+))?)',
                    r'(?i)(§\s*(\d+)[º°]?)',  # Parágrafos
                    r'(?i)(inc(?:iso)?\.?\s+([IVX]+|\d+))',  # Incisos
                    
                    # Súmulas
                    r'(?i)(Súmula\s+(?:Vinculante\s+)?n?[º°]?\s*(\d+)(?:\s+do\s+(STF|STJ|TST))?)',
                    
                    # Constituição
                    r'(?i)(Constitui[çã][ãç]o\s+Federal)',
                    r'(?i)(CF[/-]?\d{2})',
                ],
                normalization_rules=[
                    (r'(?i)^lei\s+federal\s+', 'Lei '),  # Remove "Federal"
                    (r'(?i)^artigos?\.\s+', 'Art. '),    # Padroniza "Art."
                    (r'(?i)[º°]', 'º'),                  # Padroniza ordinais
                ],
                context_boost=['legislação', 'dispositivo', 'diploma', 'norma']
            ),
            
            # ÓRGÃOS E INSTITUIÇÕES
            EntityPattern(
                entity_type=EntityType.ORGAO,
                patterns=[
                    # Bancos (relevante para empréstimos consignados)
                    r'(?i)(Banco\s+(?:do\s+)?(?:Brasil|Central|Nordeste|Bradesco|Itaú|Santander|Caixa))',
                    r'(?i)(CEF|BB|BACEN)',  # Siglas de bancos
                    r'(?i)(Caixa\s+Econômica\s+Federal)',
                    
                    # Órgãos públicos
                    r'(?i)(INSS|Receita\s+Federal|Ministério\s+Público|MPE?-?PI)',
                    r'(?i)(Defensoria\s+Pública)',
                    r'(?i)(Procuradoria\s+(?:Geral\s+)?do\s+Estado)',
                    
                    # Cartórios
                    r'(?i)(\d+[º°]?\s+(?:Cartório|Tabelionato|Ofício))',
                    r'(?i)(Cartório\s+de\s+(?:Registro\s+)?(?:Civil|Imóveis|Títulos))',
                ],
                normalization_rules=[
                    (r'(?i)^banco\s+do\s+', 'Banco '),  # Remove "do"
                    (r'(?i)cef', 'Caixa Econômica Federal'),
                ],
                context_boost=['instituição', 'órgão', 'entidade', 'autarquia']
            ),
            
            # PESSOAS FÍSICAS (PARTES DO PROCESSO)
            EntityPattern(
                entity_type=EntityType.PESSOA,
                patterns=[
                    # Nomes completos em contextos específicos
                    r'(?i)(?:autor|autora|requerente):?\s+([A-Z][a-záêçõâô]+(?:\s+[A-Z][a-záêçõâô]+){2,4})',
                    r'(?i)(?:réu?|ré|requerido|requerida):?\s+([A-Z][a-záêçõâô]+(?:\s+[A-Z][a-záêçõâô]+){2,4})',
                    r'(?i)(?:apelante|apelado|agravante|agravado):?\s+([A-Z][a-záêçõâô]+(?:\s+[A-Z][a-záêçõâô]+){2,4})',
                    
                    # CPF (para identificar pessoa física)
                    r'(?i)CPF[:\s]*(\d{3}\.?\d{3}\.?\d{3}[-\s]?\d{2})',
                ],
                normalization_rules=[
                    (r'(?i)^(?:autor|réu|apelante)a?:?\s+', ''),  # Remove qualificação
                ],
                context_boost=['parte', 'litigante', 'interessado'],
                min_confidence=0.8  # Maior rigor para pessoas
            )
        ]
        
        return patterns
    
    def extract_entities_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Extrai entidades nomeadas de um texto
        
        Args:
            text: Texto para processar
            
        Returns:
            Lista de dicionários com dados das entidades
        """
        
        if not text or not text.strip():
            return []
        
        # Verifica cache
        text_hash = hash(text)
        if text_hash in self.entity_cache:
            return self.entity_cache[text_hash]
        
        self.stats['texts_processed'] += 1
        
        try:
            # Combina entidades do spaCy com regras customizadas
            entities = []
            
            # 1. Entidades via spaCy (se disponível)
            if self.nlp:
                spacy_entities = self._extract_spacy_entities(text)
                entities.extend(spacy_entities)
            
            # 2. Entidades via padrões customizados
            pattern_entities = self._extract_pattern_entities(text)
            entities.extend(pattern_entities)
            
            # 3. Remove duplicatas e resolve conflitos
            unique_entities = self._deduplicate_entities(entities)
            
            # 4. Converte para formato final
            final_entities = []
            for entity in unique_entities:
                entity_data = {
                    'nome_original': entity.text,
                    'nome_normalizado': self._normalize_entity_name(entity.normalized, entity.entity_type),
                    'tipo': entity.entity_type,
                    'confianca': entity.confidence,
                    'posicao_inicio': entity.start,
                    'posicao_fim': entity.end,
                    'padrao_usado': entity.pattern_used
                }
                final_entities.append(entity_data)
                
                # Atualiza estatísticas
                self.stats['entities_found'] += 1
                self.stats['entities_by_type'][entity.entity_type.value] += 1
                self.stats['unique_entities'].add(entity.normalized)
            
            # Cache resultado
            self.entity_cache[text_hash] = final_entities
            
            return final_entities
            
        except Exception as e:
            logger.error(f"❌ Erro na extração de entidades: {e}")
            return []
    
    def _extract_spacy_entities(self, text: str) -> List[EntityMatch]:
        """Extrai entidades usando spaCy"""
        
        entities = []
        
        try:
            doc = self.nlp(text)
            
            for ent in doc.ents:
                # Mapeia tipos spaCy para nossos tipos
                entity_type = self._map_spacy_to_custom_type(ent.label_)
                
                if entity_type:
                    entities.append(EntityMatch(
                        text=ent.text,
                        normalized=ent.text.strip(),
                        entity_type=entity_type,
                        start=ent.start_char,
                        end=ent.end_char,
                        confidence=0.7,  # Confiança média para spaCy
                        pattern_used="spacy_" + ent.label_
                    ))
            
        except Exception as e:
            logger.warning(f"Erro no spaCy: {e}")
        
        return entities
    
    def _extract_pattern_entities(self, text: str) -> List[EntityMatch]:
        """Extrai entidades usando padrões customizados"""
        
        entities = []
        
        for pattern_config in self.patterns:
            for pattern in pattern_config.patterns:
                try:
                    regex = re.compile(pattern, re.MULTILINE | re.IGNORECASE)
                    
                    for match in regex.finditer(text):
                        # Extrai nome da entidade (grupo 1 ou match completo)
                        if match.groups():
                            entity_text = match.group(1).strip()
                            start_pos = match.start(1)
                            end_pos = match.end(1)
                        else:
                            entity_text = match.group(0).strip()
                            start_pos = match.start()
                            end_pos = match.end()
                        
                        if not entity_text or len(entity_text) < 2:
                            continue
                        
                        # Aplica regras de normalização
                        normalized = self._apply_normalization_rules(
                            entity_text, pattern_config.normalization_rules
                        )
                        
                        # Calcula confiança
                        confidence = self._calculate_confidence(
                            entity_text, text, pattern_config
                        )
                        
                        if confidence >= pattern_config.min_confidence:
                            entities.append(EntityMatch(
                                text=entity_text,
                                normalized=normalized,
                                entity_type=pattern_config.entity_type,
                                start=start_pos,
                                end=end_pos,
                                confidence=confidence,
                                pattern_used=pattern[:50]  # Trunca para log
                            ))
                
                except re.error as e:
                    logger.warning(f"Regex inválido {pattern}: {e}")
                    continue
        
        return entities
    
    def _map_spacy_to_custom_type(self, spacy_label: str) -> Optional[EntityType]:
        """Mapeia tipos spaCy para nossos tipos customizados"""
        
        mapping = {
            'PER': EntityType.PESSOA,      # Pessoa
            'PERSON': EntityType.PESSOA,   # Pessoa (inglês)
            'ORG': EntityType.ORGAO,       # Organização
            'LOC': EntityType.COMARCA,     # Localização
            'MISC': EntityType.OUTROS      # Miscelânea
        }
        
        return mapping.get(spacy_label.upper())
    
    def _apply_normalization_rules(self, text: str, 
                                  rules: Optional[List[Tuple[str, str]]]) -> str:
        """Aplica regras de normalização ao texto"""
        
        if not rules:
            return text.strip()
        
        normalized = text
        
        for pattern, replacement in rules:
            try:
                normalized = re.sub(pattern, replacement, normalized)
            except re.error:
                continue
        
        return normalized.strip()
    
    def _calculate_confidence(self, entity_text: str, full_text: str,
                            pattern_config: EntityPattern) -> float:
        """Calcula confiança da entidade baseado no contexto"""
        
        base_confidence = 0.8  # Confiança base para patterns
        
        # Boost por palavras de contexto
        if pattern_config.context_boost:
            context_window = self._get_context_window(entity_text, full_text, 50)
            
            for boost_word in pattern_config.context_boost:
                if re.search(r'(?i)\b' + boost_word + r'\b', context_window):
                    base_confidence = min(1.0, base_confidence + 0.1)
        
        # Penaliza nomes muito curtos
        if len(entity_text) < 3:
            base_confidence *= 0.7
        
        # Penaliza nomes com muitos números (exceto para leis/CPF)
        if pattern_config.entity_type not in [EntityType.LEI, EntityType.ADVOGADO]:
            digit_ratio = sum(1 for c in entity_text if c.isdigit()) / len(entity_text)
            if digit_ratio > 0.3:
                base_confidence *= 0.8
        
        return min(1.0, base_confidence)
    
    def _get_context_window(self, entity_text: str, full_text: str, 
                           window_size: int = 50) -> str:
        """Retorna janela de contexto ao redor da entidade"""
        
        pos = full_text.find(entity_text)
        if pos == -1:
            return ""
        
        start = max(0, pos - window_size)
        end = min(len(full_text), pos + len(entity_text) + window_size)
        
        return full_text[start:end]
    
    def _deduplicate_entities(self, entities: List[EntityMatch]) -> List[EntityMatch]:
        """Remove entidades duplicadas e resolve conflitos"""
        
        if not entities:
            return []
        
        # Ordena por posição no texto
        entities.sort(key=lambda x: x.start)
        
        unique_entities = []
        
        for entity in entities:
            # Verifica se sobrepõe com entidades já aceitas
            overlaps = False
            
            for existing in unique_entities:
                if self._entities_overlap(entity, existing):
                    overlaps = True
                    
                    # Se tem maior confiança, substitui
                    if entity.confidence > existing.confidence:
                        unique_entities.remove(existing)
                        unique_entities.append(entity)
                    
                    break
            
            if not overlaps:
                unique_entities.append(entity)
        
        return unique_entities
    
    def _entities_overlap(self, entity1: EntityMatch, entity2: EntityMatch) -> bool:
        """Verifica se duas entidades se sobrepõem"""
        
        # Sobreposição de posições
        return not (entity1.end <= entity2.start or entity1.start >= entity2.end)
    
    def _normalize_entity_name(self, name: str, entity_type: EntityType) -> str:
        """Normalização final do nome da entidade"""
        
        if not name:
            return ""
        
        # Normalização geral
        normalized = name.strip()
        
        # Remove múltiplos espaços
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Normalização específica por tipo
        if entity_type == EntityType.JUIZ:
            # Remove títulos comuns
            normalized = re.sub(r'(?i)^(?:des|dr|dra|juiz|juíza)\.?\s+', '', normalized)
            # Capitaliza nomes próprios
            normalized = self._capitalize_proper_names(normalized)
            
        elif entity_type == EntityType.ADVOGADO:
            # Remove títulos e OAB
            normalized = re.sub(r'(?i)^(?:dr|dra|advogado|advogada)\.?\s+', '', normalized)
            normalized = re.sub(r'(?i),?\s+oab.*$', '', normalized)
            normalized = self._capitalize_proper_names(normalized)
            
        elif entity_type == EntityType.LEI:
            # Padroniza referências legais
            normalized = re.sub(r'(?i)^lei\s+federal\s+', 'Lei ', normalized)
            normalized = re.sub(r'(?i)^artigos?\.\s+', 'Art. ', normalized)
            
        elif entity_type == EntityType.COMARCA:
            # Padroniza nomes de comarcas
            normalized = re.sub(r'(?i)\btjpi\b', 'TJPI', normalized)
            normalized = normalized.title()
        
        return normalized
    
    def _capitalize_proper_names(self, name: str) -> str:
        """Capitaliza nomes próprios corretamente"""
        
        # Palavras que não devem ser capitalizadas (exceto no início)
        lowercase_words = {'da', 'de', 'do', 'das', 'dos', 'e', 'em', 'na', 'no'}
        
        words = name.split()
        capitalized = []
        
        for i, word in enumerate(words):
            if i == 0 or word.lower() not in lowercase_words:
                capitalized.append(word.capitalize())
            else:
                capitalized.append(word.lower())
        
        return ' '.join(capitalized)
    
    def get_extraction_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas da extração"""
        
        avg_entities = (self.stats['entities_found'] / self.stats['texts_processed'] 
                       if self.stats['texts_processed'] > 0 else 0)
        
        return {
            'texts_processed': self.stats['texts_processed'],
            'total_entities_found': self.stats['entities_found'],
            'unique_entities_found': len(self.stats['unique_entities']),
            'average_entities_per_text': round(avg_entities, 2),
            'entities_by_type': dict(self.stats['entities_by_type']),
            'patterns_configured': sum(len(p.patterns) for p in self.patterns),
            'spacy_available': self.nlp is not None
        }
    
    def clear_cache(self):
        """Limpa cache de entidades"""
        self.entity_cache.clear()
        logger.info("🗑️ Cache de entidades limpo")


# Funções auxiliares para uso direto
def extract_entities_from_text(text: str) -> List[Dict[str, Any]]:
    """Função auxiliar para extrair entidades de texto"""
    extractor = NERExtractor()
    return extractor.extract_entities_from_text(text)


def extract_judges_from_text(text: str) -> List[str]:
    """Extrai apenas juízes de um texto"""
    extractor = NERExtractor()
    entities = extractor.extract_entities_from_text(text)
    
    judges = [e['nome_normalizado'] for e in entities 
             if e['tipo'] == EntityType.JUIZ and e['confianca'] > 0.8]
    
    return list(set(judges))  # Remove duplicatas


def extract_laws_from_text(text: str) -> List[str]:
    """Extrai apenas leis citadas de um texto"""
    extractor = NERExtractor()
    entities = extractor.extract_entities_from_text(text)
    
    laws = [e['nome_normalizado'] for e in entities 
           if e['tipo'] == EntityType.LEI and e['confianca'] > 0.7]
    
    return list(set(laws))  # Remove duplicatas