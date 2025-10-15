# -*- coding: utf-8 -*-
"""
Extrator de seções dos documentos jurídicos
Identifica e extrai seções típicas como relatório, decisão, dispositivo, etc.
Adaptado para lidar com inconsistências nos documentos do TJPI
"""

import re
from typing import List, Dict, Optional, Tuple , Any
from dataclasses import dataclass
from loguru import logger

from ..models.nodes import DocumentNode, SectionNode, SectionType, NodeType


@dataclass
class SectionPattern:
    """Padrão para identificar uma seção"""
    section_type: SectionType
    patterns: List[str]  # Lista de regex patterns
    priority: int = 1  # Prioridade (maior = mais específico)
    min_content_length: int = 50  # Tamanho mínimo do conteúdo
    max_content_length: int = 10000  # Tamanho máximo do conteúdo


class SectionExtractor:
    """Extrator de seções de documentos jurídicos"""
    
    def __init__(self):
        self.patterns = self._setup_section_patterns()
        self.stats = {
            'documents_processed': 0,
            'sections_found': 0,
            'sections_by_type': {}
        }
        
        logger.info("🔍 SectionExtractor inicializado")
    
    def _setup_section_patterns(self) -> List[SectionPattern]:
        """Configura padrões para identificar seções jurídicas"""
        
        patterns = [
            # EMENTA - Geralmente no início
            SectionPattern(
                section_type=SectionType.EMENTA,
                patterns=[
                    r'(?i)^[\s\n]*EMENTA\s*[:.]?',
                    r'(?i)^[\s\n]*E\s*M\s*E\s*N\s*T\s*A\s*[:.]?',
                    r'(?i)EMENTA\s*[-:]',
                    r'(?i)^\s*EMENTA\s+[A-Z]'
                ],
                priority=10,
                min_content_length=100,
                max_content_length=2000
            ),
            
            # RELATÓRIO - Geralmente após ementa
            SectionPattern(
                section_type=SectionType.RELATORIO,
                patterns=[
                    r'(?i)(?:^|\n)\s*REL[AÁ]T[OÓ]RIO\s*[:.]?',
                    r'(?i)(?:^|\n)\s*R\s*E\s*L\s*A\s*T\s*[OÓ]\s*R\s*I\s*O\s*[:.]?',
                    r'(?i)(?:^|\n)\s*O\s+REL[AÁ]T[OÓ]R',
                    r'(?i)(?:^|\n)\s*RELATA?\s+O\s+[A-Z]',
                    r'(?i)RELATORA?\s+[A-Z][a-z]+.*?[A-Z][a-z]+.*?relat'
                ],
                priority=8,
                min_content_length=200,
                max_content_length=5000
            ),
            
            # DECISÃO/FUNDAMENTAÇÃO - Coração da decisão
            SectionPattern(
                section_type=SectionType.DECISAO,
                patterns=[
                    r'(?i)(?:^|\n)\s*DECIS[ÃÃO]O\s*[:.]?',
                    r'(?i)(?:^|\n)\s*FUNDAMENTA[ÇÃ][ÃÃO]O\s*[:.]?',
                    r'(?i)(?:^|\n)\s*VOTO\s+DO\s+REL[AÁ]T[OÓ]R',
                    r'(?i)(?:^|\n)\s*É\s+O\s+VOTO',
                    r'(?i)(?:^|\n)\s*ENTENDO\s+QUE',
                    r'(?i)(?:^|\n)\s*ANTE\s+O\s+EXPOSTO',
                    r'(?i)(?:^|\n)\s*DIANTE\s+DO\s+EXPOSTO',
                    r'(?i)PASSO\s+AO\s+EXAME',
                    r'(?i)CUMPRE\s+ANALISAR',
                    r'(?i)ASSIM\s+SENDO'
                ],
                priority=9,
                min_content_length=300,
                max_content_length=8000
            ),
            
            # DISPOSITIVO - Parte final, decisão propriamente dita
            SectionPattern(
                section_type=SectionType.DISPOSITIVO,
                patterns=[
                    r'(?i)(?:^|\n)\s*DISPOSITIVO\s*[:.]?',
                    r'(?i)(?:^|\n)\s*ISTO\s+POSTO',
                    r'(?i)(?:^|\n)\s*ANTE\s+O\s+EXPOSTO',
                    r'(?i)(?:^|\n)\s*DIANTE\s+DO\s+EXPOSTO',
                    r'(?i)(?:^|\n)\s*EX\s+POSITIS',
                    r'(?i)(?:^|\n)\s*PELO\s+EXPOSTO',
                    r'(?i)(?:^|\n)\s*FACE\s+AO\s+EXPOSTO',
                    r'(?i)(?:^|\n)\s*POR\s+ESTES?\s+FUNDAMENTOS?',
                    r'(?i)(?:^|\n)\s*ACORDAM\s+OS\s+DESEMBARGADORES',
                    r'(?i)(?:^|\n)\s*DECIDE[M-SE]?\s*[:.]',
                    r'(?i)(?:^|\n)\s*JULGO\s+PROCEDENTE',
                    r'(?i)(?:^|\n)\s*JULGO\s+IMPROCEDENTE',
                    r'(?i)(?:^|\n)\s*DOU\s+PROVIMENTO',
                    r'(?i)(?:^|\n)\s*NEGO\s+PROVIMENTO',
                    r'(?i)(?:^|\n)\s*MANTENHO\s+A\s+SENTEN[ÇÃ]A'
                ],
                priority=10,
                min_content_length=100,
                max_content_length=3000
            ),
            
            # VOTO - Específico para decisões colegiadas
            SectionPattern(
                section_type=SectionType.VOTO,
                patterns=[
                    r'(?i)(?:^|\n)\s*VOTO\s*[:.]?',
                    r'(?i)(?:^|\n)\s*V\s*O\s*T\s*O\s*[:.]?',
                    r'(?i)(?:^|\n)\s*MEU\s+VOTO',
                    r'(?i)VOTO\s+DO\s+RELATOR',
                    r'(?i)VOTO\s+DA\s+RELATORA',
                    r'(?i)VOTO\s+CONDUTOR'
                ],
                priority=7,
                min_content_length=200,
                max_content_length=4000
            )
        ]
        
        # Ordena por prioridade (maior primeiro)
        patterns.sort(key=lambda x: x.priority, reverse=True)
        
        return patterns
    
    def extract_sections_from_document(self, document: DocumentNode) -> List[SectionNode]:
        """
        Extrai seções de um documento jurídico
        
        Args:
            document: Documento para processar
            
        Returns:
            Lista de SectionNode extraídas
        """
        
        if not document.conteudo_limpo:
            logger.warning(f"Documento {document.id} sem conteúdo limpo")
            return []
        
        self.stats['documents_processed'] += 1
        
        try:
            sections = self._extract_sections(document.conteudo_limpo, document.id)
            
            self.stats['sections_found'] += len(sections)
            for section in sections:
                section_type = section.section_type.value
                self.stats['sections_by_type'][section_type] = \
                    self.stats['sections_by_type'].get(section_type, 0) + 1
            
            logger.debug(f"📋 Documento {document.id}: {len(sections)} seções extraídas")
            return sections
            
        except Exception as e:
            logger.error(f"❌ Erro ao processar documento {document.id}: {e}")
            return []
    
    def _extract_sections(self, text: str, document_id: str) -> List[SectionNode]:
        """Extrai seções de um texto usando padrões regex"""
        
        sections = []
        text_length = len(text)
        found_sections = []  # Para evitar sobreposições
        
        # Procura por cada tipo de seção
        for pattern_config in self.patterns:
            
            section_matches = self._find_section_matches(
                text, pattern_config, document_id
            )
            
            for match in section_matches:
                # Verifica se não sobrepõe com seção já encontrada
                if not self._overlaps_with_existing(match, found_sections):
                    found_sections.append(match)
        
        # Ordena seções por posição no texto
        found_sections.sort(key=lambda x: x['start_pos'])
        
        # Cria SectionNodes
        for i, section_data in enumerate(found_sections):
            section_node = SectionNode(
                id=f"sec_{document_id}_{section_data['type'].value}_{i}",
                node_type=NodeType.SECTION,  # ✅ ADICIONE
                label=f"{section_data['type'].value.title()} - {document_id}",
                parent_document_id=document_id,
                section_type=section_data['type'],
                conteudo_texto=section_data['content'],
                conteudo_limpo=self._clean_section_content(section_data['content']),
                ordem=i + 1
            )
            sections.append(section_node)
        
        # Se não encontrou seções específicas, cria seção genérica
        if not sections and text_length > 200:
            generic_section = self._create_generic_section(text, document_id)
            if generic_section:
                sections.append(generic_section)
        
        return sections
    
    def _find_section_matches(self, text: str, pattern_config: SectionPattern, 
                             document_id: str) -> List[Dict]:
        """Encontra matches para um padrão específico"""
        
        matches = []
        
        for pattern in pattern_config.patterns:
            try:
                regex = re.compile(pattern, re.MULTILINE | re.DOTALL)
                
                for match in regex.finditer(text):
                    start_pos = match.start()
                    
                    # Extrai conteúdo da seção
                    section_content = self._extract_section_content(
                        text, start_pos, pattern_config
                    )
                    
                    if section_content and self._validate_section_content(
                        section_content, pattern_config
                    ):
                        matches.append({
                            'type': pattern_config.section_type,
                            'start_pos': start_pos,
                            'content': section_content,
                            'pattern_used': pattern
                        })
                        
                        # Para na primeira match válida deste tipo
                        break
                        
            except re.error as e:
                logger.warning(f"Padrão regex inválido {pattern}: {e}")
                continue
        
        return matches
    
    def _extract_section_content(self, text: str, start_pos: int, 
                                pattern_config: SectionPattern) -> Optional[str]:
        """Extrai o conteúdo de uma seção a partir da posição inicial"""
        
        # Estratégias para determinar onde a seção termina
        end_indicators = [
            r'(?i)(?:^|\n)\s*(?:REL[AÁ]T[OÓ]RIO|DECIS[ÃÃO]O|DISPOSITIVO|VOTO|EMENTA)\s*[:.]',
            r'(?i)(?:^|\n)\s*(?:ISTO\s+POSTO|ANTE\s+O\s+EXPOSTO|PELO\s+EXPOSTO)',
            r'(?i)(?:^|\n)\s*(?:ACORDAM|DECIDE|JULGO)',
            r'(?:^|\n)\s*_{10,}',  # Linha de sublinhados
            r'(?:^|\n)\s*={10,}',  # Linha de iguais
            r'(?:^|\n)\s*-{10,}',  # Linha de hífens
        ]
        
        # Começa da posição encontrada
        section_start = start_pos
        section_text = text[section_start:]
        
        # Procura pelo fim da seção
        min_length = pattern_config.min_content_length
        max_length = pattern_config.max_content_length
        
        # Tenta encontrar próxima seção
        next_section_pos = len(section_text)
        
        for indicator in end_indicators:
            try:
                regex = re.compile(indicator, re.MULTILINE)
                matches = list(regex.finditer(section_text))
                
                # Ignora match na posição 0 (início da própria seção)
                valid_matches = [m for m in matches if m.start() > min_length]
                
                if valid_matches:
                    candidate_pos = valid_matches[0].start()
                    if candidate_pos < next_section_pos:
                        next_section_pos = candidate_pos
                        
            except re.error:
                continue
        
        # Limita pelo tamanho máximo
        end_pos = min(next_section_pos, max_length)
        
        # Extrai conteúdo
        if end_pos > min_length:
            content = section_text[:end_pos].strip()
            return content
        
        return None
    
    def _validate_section_content(self, content: str, 
                                 pattern_config: SectionPattern) -> bool:
        """Valida se o conteúdo extraído é uma seção válida"""
        
        if not content:
            return False
        
        # Verifica tamanho
        content_length = len(content)
        if content_length < pattern_config.min_content_length:
            return False
        
        if content_length > pattern_config.max_content_length:
            return False
        
        # Verifica se tem conteúdo semântico mínimo
        words = content.split()
        if len(words) < 10:  # Muito poucas palavras
            return False
        
        # Verifica proporção de caracteres alfanuméricos
        alpha_chars = sum(1 for c in content if c.isalnum())
        if alpha_chars / len(content) < 0.6:  # Muito pouco texto
            return False
        
        # Validações específicas por tipo de seção
        if pattern_config.section_type == SectionType.EMENTA:
            # Ementa deve conter termos típicos
            ementa_indicators = [
                'direito', 'civil', 'comercial', 'processo', 'recurso',
                'apela[çã]', 'consignado', 'empréstimo', 'banco'
            ]
            if not any(re.search(r'(?i)' + indicator, content) for indicator in ementa_indicators):
                return False
        
        elif pattern_config.section_type == SectionType.DISPOSITIVO:
            # Dispositivo deve conter decisão
            decision_indicators = [
                'julgo', 'decido', 'acordam', 'provimento', 'procedente',
                'improcedente', 'mantenho', 'reformo', 'confirmo'
            ]
            if not any(re.search(r'(?i)\b' + indicator, content) for indicator in decision_indicators):
                return False
        
        return True
    
    def _overlaps_with_existing(self, new_match: Dict, existing_matches: List[Dict]) -> bool:
        """Verifica se uma nova match sobrepõe com matches existentes"""
        
        new_start = new_match['start_pos']
        new_end = new_start + len(new_match['content'])
        
        for existing in existing_matches:
            existing_start = existing['start_pos']
            existing_end = existing_start + len(existing['content'])
            
            # Verifica sobreposição
            if not (new_end <= existing_start or new_start >= existing_end):
                # Há sobreposição - mantém a de maior prioridade
                new_priority = self._get_pattern_priority(new_match['type'])
                existing_priority = self._get_pattern_priority(existing['type'])
                
                if new_priority <= existing_priority:
                    return True  # Nova match tem prioridade menor, ignora
        
        return False
    
    def _get_pattern_priority(self, section_type: SectionType) -> int:
        """Retorna prioridade do tipo de seção"""
        for pattern in self.patterns:
            if pattern.section_type == section_type:
                return pattern.priority
        return 0
    
    def _clean_section_content(self, content: str) -> str:
        """Limpa o conteúdo de uma seção"""
        
        if not content:
            return ""
        
        # Remove múltiplas quebras de linha
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
        
        # Remove espaços extras
        content = re.sub(r'[ \t]+', ' ', content)
        
        # Remove linhas com apenas caracteres especiais
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            stripped = line.strip()
            if stripped and not re.match(r'^[_\-=*]{3,}$', stripped):
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines).strip()
    
    def _create_generic_section(self, text: str, document_id: str) -> Optional[SectionNode]:
        """Cria uma seção genérica quando nenhuma específica é encontrada"""
        
        if len(text) < 200:
            return None
        
        # Trunca texto se muito longo
        if len(text) > 5000:
            text = text[:5000] + "..."
        
        return SectionNode(
            id=f"sec_{document_id}_outros_0",
            node_type=NodeType.SECTION,  # ✅ ADICIONE
            label=f"Outros - {document_id}",
            parent_document_id=document_id,
            section_type=SectionType.OUTROS,
            conteudo_texto=text,
            conteudo_limpo=self._clean_section_content(text),
            ordem=1
        )
    
    def get_extraction_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas da extração"""
        
        total_sections = self.stats['sections_found']
        avg_sections = (total_sections / self.stats['documents_processed'] 
                       if self.stats['documents_processed'] > 0 else 0)
        
        return {
            'documents_processed': self.stats['documents_processed'],
            'total_sections_found': total_sections,
            'average_sections_per_document': round(avg_sections, 2),
            'sections_by_type': self.stats['sections_by_type'],
            'patterns_configured': len(self.patterns)
        }
    
    def extract_sections_from_text(self, text: str, text_id: str = "unknown") -> List[Dict]:
        """
        Método auxiliar para extrair seções de texto puro
        Útil para testes e uso direto
        
        Returns:
            Lista de dicionários com informações das seções
        """
        
        sections_data = []
        found_sections = []
        
        for pattern_config in self.patterns:
            section_matches = self._find_section_matches(text, pattern_config, text_id)
            
            for match in section_matches:
                if not self._overlaps_with_existing(match, found_sections):
                    found_sections.append(match)
        
        # Ordena por posição
        found_sections.sort(key=lambda x: x['start_pos'])
        
        for i, section_data in enumerate(found_sections):
            sections_data.append({
                'type': section_data['type'].value,
                'content': section_data['content'],
                'position': i + 1,
                'start_char': section_data['start_pos'],
                'length': len(section_data['content']),
                'pattern_matched': section_data['pattern_used']
            })
        
        return sections_data


# Função auxiliar para uso direto
def extract_sections_from_document(document: DocumentNode) -> List[SectionNode]:
    """Função auxiliar para extrair seções de um documento"""
    extractor = SectionExtractor()
    return extractor.extract_sections_from_document(document)


def extract_sections_from_text(text: str) -> List[Dict]:
    """Função auxiliar para extrair seções de texto puro"""
    extractor = SectionExtractor()
    return extractor.extract_sections_from_text(text)