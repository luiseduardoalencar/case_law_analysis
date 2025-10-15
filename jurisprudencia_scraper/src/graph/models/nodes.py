# -*- coding: utf-8 -*-
"""
Definição dos tipos de nós do grafo heterogêneo de jurisprudências
Baseado na estrutura do banco de dados e nos requisitos do TCC
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
import numpy as np
from enum import Enum


class NodeType(Enum):
    """Tipos de nós do grafo heterogêneo"""
    DOCUMENT = "document"
    SECTION = "section"
    ENTITY = "entity"
    CONCEPT = "concept"


class SectionType(Enum):
    """Tipos de seções encontradas nos documentos jurídicos"""
    RELATORIO = "relatorio"
    DECISAO = "decisao"
    DISPOSITIVO = "dispositivo"
    EMENTA = "ementa"
    VOTO = "voto"
    OUTROS = "outros"


class EntityType(Enum):
    """Tipos de entidades nomeadas extraídas"""
    JUIZ = "juiz"
    ADVOGADO = "advogado"
    COMARCA = "comarca"
    LEI = "lei"
    ORGAO = "orgao"
    PESSOA = "pessoa"
    OUTROS = "outros"


@dataclass
class BaseNode:
    """Classe base para todos os nós do grafo"""
    id: str
    node_type: NodeType
    label: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte o nó para dicionário"""
        base_dict = {
            'id': self.id,
            'node_type': self.node_type.value,
            'label': self.label,
        }
        # Adiciona created_at se existir
        if hasattr(self, 'created_at'):
            base_dict['created_at'] = self.created_at.isoformat()
        # Adiciona metadata se existir na subclasse
        if hasattr(self, 'metadata'):
            base_dict['metadata'] = self.metadata
        return base_dict


@dataclass
class DocumentNode(BaseNode):
    """
    Nó representando um documento jurídico completo
    Mapeado da tabela 'processos' + 'processos_metadados'
    """
    # ✅ CAMPOS OBRIGATÓRIOS PRIMEIRO (sem valor padrão)
    numero_processo: str
    url_original: str
    
    # ✅ CAMPOS OPCIONAIS DEPOIS (com valor padrão)
    # Metadados do processo (vindos de processos_metadados)
    orgao_julgador: Optional[str] = None
    orgao_julgador_colegiado: Optional[str] = None
    relator: Optional[str] = None
    classe_judicial: Optional[str] = None
    competencia: Optional[str] = None
    assunto_principal: Optional[str] = None
    autor: Optional[str] = None
    reu: Optional[str] = None
    data_publicacao: Optional[datetime] = None
    tipo_decisao: Optional[str] = None
    
    # Conteúdo e processamento
    conteudo_completo: Optional[str] = None
    conteudo_limpo: Optional[str] = None
    hash_documento: Optional[str] = None
    
    # Embedding e vetorização
    embedding: Optional[np.ndarray] = None
    embedding_dimension: Optional[int] = None
    
    # Métricas computadas
    num_tokens: Optional[int] = None
    num_secoes: Optional[int] = None
    num_entidades: Optional[int] = None
    
    # Relacionamentos
    secoes_ids: List[str] = field(default_factory=list)
    entidades_ids: List[str] = field(default_factory=list)
    conceitos_ids: List[str] = field(default_factory=list)
    
    # Metadata (adicionado aqui para evitar conflito de herança)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = f"doc_{self.numero_processo}"
        if not self.label:
            self.label = f"Processo {self.numero_processo}"
        self.node_type = NodeType.DOCUMENT
    
    def add_secao(self, secao_id: str):
        """Adiciona uma seção ao documento"""
        if secao_id not in self.secoes_ids:
            self.secoes_ids.append(secao_id)
    
    def add_entidade(self, entidade_id: str):
        """Adiciona uma entidade ao documento"""
        if entidade_id not in self.entidades_ids:
            self.entidades_ids.append(entidade_id)
    
    def add_conceito(self, conceito_id: str):
        """Adiciona um conceito ao documento"""
        if conceito_id not in self.conceitos_ids:
            self.conceitos_ids.append(conceito_id)


@dataclass
class SectionNode(BaseNode):
    """
    Nó representando uma seção de um documento jurídico
    Mapeado da tabela 'processos_conteudo'
    """
    # ✅ CAMPOS OBRIGATÓRIOS PRIMEIRO
    parent_document_id: str
    section_type: SectionType
    
    # ✅ CAMPOS OPCIONAIS DEPOIS
    conteudo_html: Optional[str] = None
    conteudo_texto: Optional[str] = None
    conteudo_limpo: Optional[str] = None
    ordem: Optional[int] = None
    
    # Embedding e vetorização
    embedding: Optional[np.ndarray] = None
    embedding_dimension: Optional[int] = None
    
    # Métricas
    num_tokens: Optional[int] = None
    num_palavras: Optional[int] = None
    
    # Relacionamentos
    entidades_ids: List[str] = field(default_factory=list)
    conceitos_ids: List[str] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = f"sec_{self.parent_document_id}_{self.section_type.value}_{self.ordem or 0}"
        if not self.label:
            self.label = f"{self.section_type.value.title()} - {self.parent_document_id}"
        self.node_type = NodeType.SECTION
    
    def add_entidade(self, entidade_id: str):
        """Adiciona uma entidade à seção"""
        if entidade_id not in self.entidades_ids:
            self.entidades_ids.append(entidade_id)
    
    def add_conceito(self, conceito_id: str):
        """Adiciona um conceito à seção"""
        if conceito_id not in self.conceitos_ids:
            self.conceitos_ids.append(conceito_id)


@dataclass
class EntityNode(BaseNode):
    """
    Nó representando uma entidade nomeada extraída dos documentos
    Extraído via NER dos textos jurídicos
    """
    # ✅ CAMPOS OBRIGATÓRIOS PRIMEIRO
    entity_type: EntityType
    nome_original: str
    nome_normalizado: str
    
    # ✅ CAMPOS OPCIONAIS DEPOIS
    contexto: Optional[str] = None
    
    # Frequência e estatísticas
    frequencia_global: int = 0
    num_documentos: int = 0
    num_secoes: int = 0
    
    # Variações encontradas
    variacoes: List[str] = field(default_factory=list)
    
    # Relacionamentos
    documentos_ids: List[str] = field(default_factory=list)
    secoes_ids: List[str] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = f"ent_{self.entity_type.value}_{hash(self.nome_normalizado) % 10000:04d}"
        if not self.label:
            self.label = self.nome_normalizado
        self.node_type = NodeType.ENTITY
    
    def add_variacao(self, variacao: str):
        """Adiciona uma variação do nome da entidade"""
        if variacao not in self.variacoes:
            self.variacoes.append(variacao)
    
    def add_documento(self, documento_id: str):
        """Registra aparição em um documento"""
        if documento_id not in self.documentos_ids:
            self.documentos_ids.append(documento_id)
            self.num_documentos = len(self.documentos_ids)
    
    def add_secao(self, secao_id: str):
        """Registra aparição em uma seção"""
        if secao_id not in self.secoes_ids:
            self.secoes_ids.append(secao_id)
            self.num_secoes = len(self.secoes_ids)
    
    def incrementar_frequencia(self):
        """Incrementa frequência global da entidade"""
        self.frequencia_global += 1


@dataclass
class ConceptNode(BaseNode):
    """
    Nó representando um conceito jurídico extraído dos documentos
    Baseado em termos-chave do domínio jurídico
    """
    # ✅ CAMPOS OBRIGATÓRIOS PRIMEIRO
    termo_conceito: str
    
    # ✅ CAMPOS OPCIONAIS DEPOIS
    categoria_juridica: Optional[str] = None
    definicao: Optional[str] = None
    
    # Estatísticas de frequência
    frequencia_global: int = 0
    frequencia_documentos: int = 0
    idf_score: Optional[float] = None
    
    # Contextos onde aparece
    contextos: List[str] = field(default_factory=list)
    
    # Relacionamentos
    documentos_ids: List[str] = field(default_factory=list)
    secoes_ids: List[str] = field(default_factory=list)
    
    # Métricas de relevância (calculadas posteriormente)
    tfidf_scores: Dict[str, float] = field(default_factory=dict)  # doc_id -> score
    pmi_scores: Dict[str, float] = field(default_factory=dict)    # concept_id -> score
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = f"con_{hash(self.termo_conceito) % 10000:04d}"
        if not self.label:
            self.label = self.termo_conceito
        self.node_type = NodeType.CONCEPT
    
    def add_contexto(self, contexto: str):
        """Adiciona um contexto onde o conceito aparece"""
        if contexto not in self.contextos:
            self.contextos.append(contexto)
            # Limita o número de contextos para economizar memória
            if len(self.contextos) > 10:
                self.contextos = self.contextos[-10:]
    
    def add_documento(self, documento_id: str, tfidf_score: Optional[float] = None):
        """Registra aparição em um documento"""
        if documento_id not in self.documentos_ids:
            self.documentos_ids.append(documento_id)
            self.frequencia_documentos = len(self.documentos_ids)
        
        if tfidf_score is not None:
            self.tfidf_scores[documento_id] = tfidf_score
    
    def add_secao(self, secao_id: str):
        """Registra aparição em uma seção"""
        if secao_id not in self.secoes_ids:
            self.secoes_ids.append(secao_id)
    
    def incrementar_frequencia(self):
        """Incrementa frequência global do conceito"""
        self.frequencia_global += 1
    
    def set_pmi_score(self, other_concept_id: str, pmi_score: float):
        """Define score PMI com outro conceito"""
        self.pmi_scores[other_concept_id] = pmi_score
    
    def get_relevancia_documento(self, documento_id: str) -> float:
        """Retorna relevância do conceito para um documento específico"""
        return self.tfidf_scores.get(documento_id, 0.0)


# Funções auxiliares para criação de nós

def create_document_node_from_db_row(processo_row, metadados_row=None) -> DocumentNode:
    """
    Cria um DocumentNode a partir de uma linha do banco de dados
    
    Args:
        processo_row: Linha da tabela 'processos'
        metadados_row: Linha da tabela 'processos_metadados' (opcional)
    
    Returns:
        DocumentNode configurado
    """
    node = DocumentNode(
        id=f"doc_{processo_row['numero_processo']}",
        node_type=NodeType.DOCUMENT,
        label=f"Processo {processo_row['numero_processo']}",
        numero_processo=processo_row['numero_processo'],
        url_original=processo_row['url_original'],
        conteudo_completo=processo_row.get('html_completo'),
        hash_documento=processo_row.get('hash_documento')
    )
    
    # Adiciona metadados se disponíveis
    if metadados_row:
        node.orgao_julgador = metadados_row.get('orgao_julgador')
        node.orgao_julgador_colegiado = metadados_row.get('orgao_julgador_colegiado')
        node.relator = metadados_row.get('relator')
        node.classe_judicial = metadados_row.get('classe_judicial')
        node.competencia = metadados_row.get('competencia')
        node.assunto_principal = metadados_row.get('assunto_principal')
        node.autor = metadados_row.get('autor')
        node.reu = metadados_row.get('reu')
        node.data_publicacao = metadados_row.get('data_publicacao')
        node.tipo_decisao = metadados_row.get('tipo_decisao')
    
    return node


def create_section_node_from_db_row(conteudo_row, parent_doc_id: str) -> SectionNode:
    """
    Cria um SectionNode a partir de uma linha da tabela processos_conteudo
    
    Args:
        conteudo_row: Linha da tabela 'processos_conteudo'
        parent_doc_id: ID do documento pai
    
    Returns:
        SectionNode configurado
    """
    # Mapeia tipos de seção do banco para enum
    tipo_secao_map = {
        'relatorio': SectionType.RELATORIO,
        'decisao': SectionType.DECISAO,
        'dispositivo': SectionType.DISPOSITIVO,
        'ementa': SectionType.EMENTA,
        'voto': SectionType.VOTO
    }
    
    tipo_secao = tipo_secao_map.get(
        conteudo_row.get('tipo_secao', '').lower(),
        SectionType.OUTROS
    )
    
    return SectionNode(
        id=f"sec_{parent_doc_id}_{tipo_secao.value}_{conteudo_row.get('ordem', 0)}",
        node_type=NodeType.SECTION,
        label=f"{tipo_secao.value.title()} - {parent_doc_id}",
        parent_document_id=parent_doc_id,
        section_type=tipo_secao,
        conteudo_html=conteudo_row.get('conteudo_html'),
        conteudo_texto=conteudo_row.get('conteudo_texto'),
        conteudo_limpo=conteudo_row.get('conteudo_limpo'),
        ordem=conteudo_row.get('ordem')
    )


# Conceitos jurídicos pré-definidos específicos do domínio
CONCEITOS_JURIDICOS_PREDEFINIDOS = [
    # Conceitos relacionados a empréstimos consignados
    "empréstimo consignado",
    "consignação em folha",
    "desconto em folha",
    "margem consignável",
    "servidor público",
    "aposentado",
    "pensionista",
    
    # Conceitos de fraude e vícios
    "vício de consentimento",
    "fraude",
    "dolo",
    "erro",
    "coação",
    "má-fé",
    "boa-fé",
    
    # Conceitos de danos e reparação
    "dano moral",
    "dano material",
    "lucros cessantes",
    "danos emergentes",
    "repetição de indébito",
    "repetição em dobro",
    "restituição",
    
    # Conceitos processuais
    "tutela antecipada",
    "liminar",
    "sentença",
    "acórdão",
    "recurso",
    "apelação",
    "embargos",
    
    # Conceitos de direito do consumidor
    "relação de consumo",
    "código de defesa do consumidor",
    "vulnerabilidade",
    "hipossuficiência",
    "prática abusiva",
    "cláusula abusiva",
    
    # Conceitos bancários
    "instituição financeira",
    "banco",
    "contrato bancário",
    "taxa de juros",
    "juros abusivos",
    "capitalização",
    "anatocismo"
]