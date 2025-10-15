# 📚 Grafo de Jurisprudências - Empréstimo Consignado

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![NetworkX](https://img.shields.io/badge/NetworkX-3.x-orange)](https://networkx.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success)](https://github.com/yourusername/jurisprudencia_project)

Projeto de construção e análise de grafo heterogêneo de conhecimento jurídico a partir de jurisprudências sobre empréstimo consignado.

## 🎯 Objetivo

Construir um grafo de conhecimento que representa relações semânticas entre documentos jurídicos, seções textuais, entidades nomeadas e conceitos jurídicos, permitindo:

- 🔍 Busca por similaridade semântica
- 📊 Identificação de conceitos centrais
- 🏘️ Detecção de comunidades temáticas
- 📈 Análise de tendências jurisprudenciais
- 🤖 Recomendação de precedentes

---

## 📁 Estrutura do Projeto
```
jurisprudencia_project/
│
├── data/                           # Dados e resultados (não versionados)
│   ├── graph/                      # Grafos gerados
│   │   ├── exports/               # Arquivos exportados (CSV, JSON, GEXF)
│   │   ├── metrics/               # Métricas calculadas
│   │   ├── visualizations/        # Gráficos e dashboards
│   │   └── embeddings/            # Cache de embeddings
│   ├── logs/                       # Logs de execução
│   ├── raw/                        # Dados brutos
│   └── processed/                  # Dados processados
│
├── jurisprudencia_scraper/         # Pacote principal
│   ├── config/                     # Configurações
│   │   ├── settings.py            # Configurações gerais
│   │   └── database.py            # Configuração do banco
│   │
│   ├── src/                        # Código fonte
│   │   ├── graph/                 # Módulo de grafos
│   │   │   ├── extractors/       # Extratores (NER, seções, conceitos)
│   │   │   │   ├── section_extractor.py
│   │   │   │   ├── ner_extractor.py
│   │   │   │   └── concept_extractor.py
│   │   │   ├── models/            # Modelos de dados
│   │   │   │   ├── nodes.py      # Definição de nós
│   │   │   │   ├── edges.py      # Definição de arestas
│   │   │   │   └── graph_schema.py
│   │   │   ├── pipeline/          # Pipeline de construção
│   │   │   │   └── graph_pipeline.py
│   │   │   ├── processors/        # Processadores
│   │   │   │   ├── text_vectorizer.py
│   │   │   │   └── similarity_calculator.py
│   │   │   └── utils/             # Utilitários
│   │   │       ├── export.py     # Exportação de grafos
│   │   │       ├── metrics.py    # Cálculo de métricas
│   │   │       └── visualization.py
│   │   │
│   │   ├── database/              # Acesso a dados
│   │   │   └── preprocessor.py
│   │   │
│   │   └── scrapers/              # Web scrapers (se aplicável)
│   │
│   └── scripts/                    # Scripts executáveis
│       ├── build_graph.py         # ⭐ Script principal
│       ├── calculate_similarities.py
│       ├── export_graph.py
│       └── extract_entities.py
│
├── notebooks/                      # Jupyter notebooks
│   ├── exploratory_analysis.ipynb
│   └── graph_visualization.ipynb
│
├── tests/                          # Testes unitários
│   ├── test_extractors.py
│   └── test_pipeline.py
│
├── .env                            # Variáveis de ambiente (não versionado)
├── .gitignore                      # Arquivos ignorados pelo Git
├── README.md                       # Este arquivo
├── requirements.txt                # Dependências Python
└── docker-compose.yml             # Docker (opcional)
```

---

## 🚀 Instalação

### 1️⃣ Pré-requisitos

- Python 3.8 ou superior
- PostgreSQL (para armazenar documentos)
- Git

### 2️⃣ Clonar o Repositório
```bash
git clone https://github.com/yourusername/jurisprudencia_project.git
cd jurisprudencia_project/jurisprudencia_scraper
```

### 3️⃣ Criar Ambiente Virtual
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 4️⃣ Instalar Dependências
```bash
pip install -r requirements.txt

# Instalar modelo spaCy para português
python -m spacy download pt_core_news_sm

# Instalar stopwords NLTK
python -c "import nltk; nltk.download('stopwords')"
```

### 5️⃣ Configurar Banco de Dados
```bash
# Criar arquivo .env na raiz do projeto
cp .env.example .env

# Editar .env com suas credenciais
nano .env
```

Exemplo de `.env`:
```env
DATABASE_URL=postgresql://user:password@localhost:5432/jurisprudencia_db
POSTGRES_USER=seu_usuario
POSTGRES_PASSWORD=sua_senha
POSTGRES_DB=jurisprudencia_db
```

### 6️⃣ Inicializar Banco de Dados
```bash
# Se usar Docker
docker-compose up -d

# Ou configurar PostgreSQL manualmente
psql -U postgres -c "CREATE DATABASE jurisprudencia_db;"
```

---

## 🎮 Como Usar

### 🧪 **Modo Teste (50 documentos)**

Para testar rapidamente o pipeline:
```bash
cd scripts
python build_graph.py --test --export --visualize
```

**O que faz:**
- ✅ Processa 50 documentos
- ✅ Extrai seções, entidades e conceitos
- ✅ Cria o grafo
- ✅ Exporta em múltiplos formatos
- ✅ Gera visualizações

**Tempo estimado:** ~5 minutos

**Output:**
- `data/graph/exports/` - Arquivos exportados
- `data/graph/visualizations/` - Gráficos PNG
- `data/graph/metrics/` - Métricas JSON

---

### 🔬 **Modo Amostra (personalizado)**

Para processar um número específico de documentos:
```bash
python build_graph.py --sample 200 --all
```

**Parâmetros:**
- `--sample N` - Processa N documentos
- `--all` - Ativa todas as etapas (export, metrics, visualize)

**Tempo estimado:** ~15 minutos para 200 docs

---

### 🚀 **Modo Completo (todos os documentos)**

Para processar todo o corpus:
```bash
python build_graph.py --limit 100000 --all
```

**OU sem limite:**
```bash
# Edite build_graph.py linha ~188-191 conforme instruções
python build_graph.py --all
```

**Tempo estimado:** ~30-60 minutos (depende do volume)

**Output:**
```
data/graph/
├── exports/
│   ├── grafo_jurisprudencias.gexf          # Para Gephi
│   ├── grafo_jurisprudencias_graph.json    # JSON estruturado
│   ├── grafo_jurisprudencias_nodes.csv     # Nós em CSV
│   ├── grafo_jurisprudencias_edges.csv     # Arestas em CSV
│   └── grafo_jurisprudencias_statistics.json
│
├── metrics/
│   └── relatorio_completo.json             # Métricas detalhadas
│
└── visualizations/
    ├── dashboard_summary.png               # Dashboard principal
    ├── rede.png                            # Visualização da rede
    ├── graus.png                           # Distribuição de graus
    ├── tipos_nos.png                       # Tipos de nós
    └── tipos_arestas.png                   # Tipos de arestas
```

---

## ⚙️ Opções Avançadas

### 📊 **Apenas Exportação**
```bash
python build_graph.py --sample 500 --export
```

### 📈 **Apenas Métricas**
```bash
python build_graph.py --sample 500 --metrics
```

### 🎨 **Apenas Visualizações**
```bash
python build_graph.py --sample 500 --visualize --full-viz
```

### 🌐 **Visualização Interativa (HTML)**
```bash
python build_graph.py --sample 500 --interactive
```

---

## 🛠️ Parâmetros do Script

| Parâmetro | Descrição | Exemplo |
|-----------|-----------|---------|
| `--test` | Teste rápido com 50 docs | `--test` |
| `--sample N` | Processa N documentos | `--sample 200` |
| `--limit N` | Limite de N documentos | `--limit 1000` |
| `--export` | Exporta grafo (CSV, JSON, GEXF) | `--export` |
| `--metrics` | Calcula métricas e comunidades | `--metrics` |
| `--visualize` | Cria visualizações (PNG) | `--visualize` |
| `--interactive` | Cria visualização interativa (HTML) | `--interactive` |
| `--full-viz` | Cria TODAS as visualizações | `--full-viz` |
| `--all` | Ativa todas as etapas | `--all` |
| `--output DIR` | Diretório de saída | `--output ../meu_output` |

---

## 📊 Resultados Esperados

### **Grafo Gerado (exemplo com 2.969 docs):**
```
📊 ESTATÍSTICAS DO GRAFO
├─ Nós: 21,709
│  ├─ Documentos: 2,969 (13.7%)
│  ├─ Seções: 3,474 (16.0%)
│  ├─ Entidades: 15,240 (70.2%)
│  └─ Conceitos: 26 (0.1%)
│
├─ Arestas: 126,643
│  ├─ Similarity: ~110,000 (87.2%)
│  ├─ Relevance: ~13,000 (10.0%)
│  ├─ Hierarchical: ~3,400 (2.7%)
│  └─ Cooccurrence: ~100 (0.1%)
│
├─ Métricas
│  ├─ Densidade: 0.000537
│  ├─ Clustering: 0.1089
│  ├─ Grau Médio: 11.67
│  ├─ Comunidades: 15,262
│  └─ Modularidade: 0.6658 ⭐
│
└─ Top Conceitos (PageRank)
   1. recurso (0.003870)
   2. banco (0.002455)
   3. cdc (0.001299)
```

---

## 🔧 Configurações

### **Ajustar Thresholds de Similaridade**

Edite `src/graph/pipeline/graph_pipeline.py`:
```python
SIMILARITY_THRESHOLDS = {
    'DOCUMENT_SEMANTIC': 0.30,  # Similaridade entre documentos
    'SECTION_CONTENT': 0.40,    # Similaridade entre seções
}
```

### **Configurar Extração de Conceitos**

Edite `src/graph/extractors/concept_extractor.py`:
```python
# Método extract_concepts_from_corpus
max_candidates = 500        # Máximo de candidatos
sample_size = 1500          # Amostra para descoberta
min_tfidf = 0.12           # Threshold TF-IDF
min_confidence = 0.55       # Confiança mínima
```

---

## 🐛 Troubleshooting

### **Problema: "NLTK stopwords não encontradas"**
```bash
python -c "import nltk; nltk.download('stopwords')"
```

### **Problema: "spaCy model não encontrado"**
```bash
python -m spacy download pt_core_news_sm
```

### **Problema: "Conexão com banco de dados falhou"**

Verifique suas credenciais no `.env`:
```bash
psql -U seu_usuario -d jurisprudencia_db -c "\dt"
```

### **Problema: "Processo travado na extração de conceitos"**

Isso foi corrigido! Mas se acontecer:
1. Interrompa (Ctrl+C)
2. Verifique se usou o código otimizado do `concept_extractor.py`
3. Reduza `sample_size` ou `max_candidates`

### **Problema: "Erro ao exportar GraphML"**

Normal - GraphML não suporta dicionários. Use GEXF:
```bash
# Abra no Gephi
File > Open > grafo_jurisprudencias.gexf
```

---

## 📚 Documentação Adicional

### **Notebooks de Análise**
```bash
cd notebooks
jupyter notebook exploratory_analysis.ipynb
```

### **Relatório Completo**

Após executar com `--metrics --all`, veja:
```
data/graph/metrics/relatorio_completo.json
```

### **Visualizar no Gephi**

1. Abra o Gephi
2. File > Open > `data/graph/exports/grafo_jurisprudencias.gexf`
3. Aplique layout (Force Atlas 2)
4. Colorir por tipo de nó
5. Ajustar tamanho por PageRank

---

## 🏗️ Arquitetura
```
┌─────────────────────────────────────────────┐
│          PIPELINE DE CONSTRUÇÃO             │
└─────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│  1. PRÉ-PROCESSAMENTO                       │
│     • Carrega docs do PostgreSQL            │
│     • Limpa e normaliza textos              │
└─────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│  2. EXTRAÇÃO                                │
│     • Seções (SectionExtractor)             │
│     • Entidades (NERExtractor + spaCy)      │
│     • Conceitos (ConceptExtractor + TF-IDF) │
└─────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│  3. VETORIZAÇÃO                             │
│     • Sentence-Transformers (embeddings)    │
│     • Similaridade cosseno                  │
└─────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│  4. CRIAÇÃO DO GRAFO                        │
│     • Nós: docs, seções, entidades, conceitos│
│     • Arestas: similarity, relevance, etc.  │
└─────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│  5. ANÁLISE E EXPORTAÇÃO                    │
│     • Métricas (PageRank, comunidades)      │
│     • Exportação (CSV, JSON, GEXF)          │
│     • Visualizações (PNG, HTML)             │
└─────────────────────────────────────────────┘
```

---

## 🤝 Contribuindo

Contribuições são bem-vindas! Para contribuir:

1. Fork o projeto
2. Crie uma branch (`git checkout -b feature/NovaFeature`)
3. Commit suas mudanças (`git commit -m 'Adiciona NovaFeature'`)
4. Push para a branch (`git push origin feature/NovaFeature`)
5. Abra um Pull Request

---

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---

## 👤 Autor

**Seu Nome**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Seu LinkedIn](https://linkedin.com/in/yourprofile)
- Email: seu.email@example.com

---

## 🙏 Agradecimentos

- [NetworkX](https://networkx.org/) - Biblioteca de grafos
- [spaCy](https://spacy.io/) - NLP em Python
- [Sentence-Transformers](https://www.sbert.net/) - Embeddings semânticos
- [Scikit-learn](https://scikit-learn.org/) - Machine Learning
- Comunidade Python Brasil 🇧🇷

---

## 📈 Roadmap

- [ ] Interface web para exploração do grafo
- [ ] API REST para busca de jurisprudências
- [ ] Integração com mais fontes de dados
- [ ] Análise temporal de jurisprudência
- [ ] Sistema de recomendação de precedentes
- [ ] Deploy em Docker
- [ ] CI/CD com GitHub Actions

---

## ⭐ Star History

Se este projeto foi útil para você, considere dar uma ⭐!

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/jurisprudencia_project&type=Date)](https://star-history.com/#yourusername/jurisprudencia_project&Date)

---

**Última atualização:** Outubro 2025