@echo off
echo ================================================
echo    CRIANDO ESTRUTURA PARA GRAFO HETEROGENEO
echo ================================================
echo.

:: Navegar para o diretório do projeto
cd /d "%~dp0"

:: Criar estrutura src/graph/
echo [1/7] Criando estrutura src/graph...
mkdir "src\graph" 2>nul
mkdir "src\graph\builders" 2>nul
mkdir "src\graph\extractors" 2>nul
mkdir "src\graph\models" 2>nul
mkdir "src\graph\storage" 2>nul
mkdir "src\graph\processors" 2>nul
mkdir "src\graph\pipeline" 2>nul
mkdir "src\graph\utils" 2>nul

:: Criar estrutura data/graph/
echo [2/7] Criando estrutura data/graph...
mkdir "data\graph" 2>nul
mkdir "data\graph\embeddings" 2>nul
mkdir "data\graph\matrices" 2>nul
mkdir "data\graph\exports" 2>nul
mkdir "data\graph\checkpoints" 2>nul

:: Criar diretório scripts/
echo [3/7] Criando diretório scripts...
mkdir "scripts" 2>nul

:: Criar notebooks adicionais
echo [4/7] Preparando notebooks...
mkdir "notebooks" 2>nul

:: Criar arquivos __init__.py
echo [5/7] Criando arquivos __init__.py...
echo # Graph module > "src\graph\__init__.py"
echo # Builders module > "src\graph\builders\__init__.py"
echo # Extractors module > "src\graph\extractors\__init__.py"
echo # Models module > "src\graph\models\__init__.py"
echo # Storage module > "src\graph\storage\__init__.py"
echo # Processors module > "src\graph\processors\__init__.py"
echo # Pipeline module > "src\graph\pipeline\__init__.py"
echo # Utils module > "src\graph\utils\__init__.py"

:: Criar arquivos principais do módulo graph
echo [6/7] Criando arquivos principais...

:: === MODELS ===
echo # -*- coding: utf-8 -*- > "src\graph\models\nodes.py"
echo """ >> "src\graph\models\nodes.py"
echo Definição dos tipos de nós do grafo heterogêneo >> "src\graph\models\nodes.py"
echo """ >> "src\graph\models\nodes.py"
echo. >> "src\graph\models\nodes.py"

echo # -*- coding: utf-8 -*- > "src\graph\models\edges.py"
echo """ >> "src\graph\models\edges.py"
echo Definição dos tipos de arestas do grafo heterogêneo >> "src\graph\models\edges.py"
echo """ >> "src\graph\models\edges.py"
echo. >> "src\graph\models\edges.py"

echo # -*- coding: utf-8 -*- > "src\graph\models\graph_schema.py"
echo """ >> "src\graph\models\graph_schema.py"
echo Schema completo do grafo heterogêneo >> "src\graph\models\graph_schema.py"
echo """ >> "src\graph\models\graph_schema.py"
echo. >> "src\graph\models\graph_schema.py"

:: === EXTRACTORS ===
echo # -*- coding: utf-8 -*- > "src\graph\extractors\section_extractor.py"
echo """ >> "src\graph\extractors\section_extractor.py"
echo Extrator de seções dos documentos jurídicos >> "src\graph\extractors\section_extractor.py"
echo """ >> "src\graph\extractors\section_extractor.py"
echo. >> "src\graph\extractors\section_extractor.py"

echo # -*- coding: utf-8 -*- > "src\graph\extractors\ner_extractor.py"
echo """ >> "src\graph\extractors\ner_extractor.py"
echo Named Entity Recognition para textos jurídicos >> "src\graph\extractors\ner_extractor.py"
echo """ >> "src\graph\extractors\ner_extractor.py"
echo. >> "src\graph\extractors\ner_extractor.py"

echo # -*- coding: utf-8 -*- > "src\graph\extractors\concept_extractor.py"
echo """ >> "src\graph\extractors\concept_extractor.py"
echo Extração de conceitos jurídicos dos documentos >> "src\graph\extractors\concept_extractor.py"
echo """ >> "src\graph\extractors\concept_extractor.py"
echo. >> "src\graph\extractors\concept_extractor.py"

echo # -*- coding: utf-8 -*- > "src\graph\extractors\similarity_calculator.py"
echo """ >> "src\graph\extractors\similarity_calculator.py"
echo Cálculo de similaridades entre documentos e seções >> "src\graph\extractors\similarity_calculator.py"
echo """ >> "src\graph\extractors\similarity_calculator.py"
echo. >> "src\graph\extractors\similarity_calculator.py"

:: === PROCESSORS ===
echo # -*- coding: utf-8 -*- > "src\graph\processors\text_vectorizer.py"
echo """ >> "src\graph\processors\text_vectorizer.py"
echo Vetorização de textos e geração de embeddings >> "src\graph\processors\text_vectorizer.py"
echo """ >> "src\graph\processors\text_vectorizer.py"
echo. >> "src\graph\processors\text_vectorizer.py"

echo # -*- coding: utf-8 -*- > "src\graph\processors\tfidf_calculator.py"
echo """ >> "src\graph\processors\tfidf_calculator.py"
echo Cálculo de TF-IDF para relevância de termos >> "src\graph\processors\tfidf_calculator.py"
echo """ >> "src\graph\processors\tfidf_calculator.py"
echo. >> "src\graph\processors\tfidf_calculator.py"

echo # -*- coding: utf-8 -*- > "src\graph\processors\cosine_similarity.py"
echo """ >> "src\graph\processors\cosine_similarity.py"
echo Cálculo de similaridade de cossenos >> "src\graph\processors\cosine_similarity.py"
echo """ >> "src\graph\processors\cosine_similarity.py"
echo. >> "src\graph\processors\cosine_similarity.py"

echo # -*- coding: utf-8 -*- > "src\graph\processors\pmi_calculator.py"
echo """ >> "src\graph\processors\pmi_calculator.py"
echo Cálculo de Pointwise Mutual Information >> "src\graph\processors\pmi_calculator.py"
echo """ >> "src\graph\processors\pmi_calculator.py"
echo. >> "src\graph\processors\pmi_calculator.py"

:: === BUILDERS ===
echo # -*- coding: utf-8 -*- > "src\graph\builders\base_builder.py"
echo """ >> "src\graph\builders\base_builder.py"
echo Classe base para construtores do grafo >> "src\graph\builders\base_builder.py"
echo """ >> "src\graph\builders\base_builder.py"
echo. >> "src\graph\builders\base_builder.py"

echo # -*- coding: utf-8 -*- > "src\graph\builders\document_builder.py"
echo """ >> "src\graph\builders\document_builder.py"
echo Construtor de nós de documentos >> "src\graph\builders\document_builder.py"
echo """ >> "src\graph\builders\document_builder.py"
echo. >> "src\graph\builders\document_builder.py"

echo # -*- coding: utf-8 -*- > "src\graph\builders\section_builder.py"
echo """ >> "src\graph\builders\section_builder.py"
echo Construtor de nós de seções >> "src\graph\builders\section_builder.py"
echo """ >> "src\graph\builders\section_builder.py"
echo. >> "src\graph\builders\section_builder.py"

echo # -*- coding: utf-8 -*- > "src\graph\builders\entity_builder.py"
echo """ >> "src\graph\builders\entity_builder.py"
echo Construtor de nós de entidades >> "src\graph\builders\entity_builder.py"
echo """ >> "src\graph\builders\entity_builder.py"
echo. >> "src\graph\builders\entity_builder.py"

echo # -*- coding: utf-8 -*- > "src\graph\builders\concept_builder.py"
echo """ >> "src\graph\builders\concept_builder.py"
echo Construtor de nós de conceitos jurídicos >> "src\graph\builders\concept_builder.py"
echo """ >> "src\graph\builders\concept_builder.py"
echo. >> "src\graph\builders\concept_builder.py"

echo # -*- coding: utf-8 -*- > "src\graph\builders\edge_builder.py"
echo """ >> "src\graph\builders\edge_builder.py"
echo Construtor de arestas do grafo >> "src\graph\builders\edge_builder.py"
echo """ >> "src\graph\builders\edge_builder.py"
echo. >> "src\graph\builders\edge_builder.py"

:: === STORAGE ===
echo # -*- coding: utf-8 -*- > "src\graph\storage\networkx_handler.py"
echo """ >> "src\graph\storage\networkx_handler.py"
echo Interface para manipulação do grafo com NetworkX >> "src\graph\storage\networkx_handler.py"
echo """ >> "src\graph\storage\networkx_handler.py"
echo. >> "src\graph\storage\networkx_handler.py"

echo # -*- coding: utf-8 -*- > "src\graph\storage\graph_serializer.py"
echo """ >> "src\graph\storage\graph_serializer.py"
echo Serialização e desserialização do grafo >> "src\graph\storage\graph_serializer.py"
echo """ >> "src\graph\storage\graph_serializer.py"
echo. >> "src\graph\storage\graph_serializer.py"

:: === PIPELINE ===
echo # -*- coding: utf-8 -*- > "src\graph\pipeline\graph_pipeline.py"
echo """ >> "src\graph\pipeline\graph_pipeline.py"
echo Pipeline principal para construção do grafo >> "src\graph\pipeline\graph_pipeline.py"
echo """ >> "src\graph\pipeline\graph_pipeline.py"
echo. >> "src\graph\pipeline\graph_pipeline.py"

echo # -*- coding: utf-8 -*- > "src\graph\pipeline\preprocessing.py"
echo """ >> "src\graph\pipeline\preprocessing.py"
echo Pré-processamento de dados para o grafo >> "src\graph\pipeline\preprocessing.py"
echo """ >> "src\graph\pipeline\preprocessing.py"
echo. >> "src\graph\pipeline\preprocessing.py"

echo # -*- coding: utf-8 -*- > "src\graph\pipeline\validation.py"
echo """ >> "src\graph\pipeline\validation.py"
echo Validação da estrutura e qualidade do grafo >> "src\graph\pipeline\validation.py"
echo """ >> "src\graph\pipeline\validation.py"
echo. >> "src\graph\pipeline\validation.py"

:: === UTILS ===
echo # -*- coding: utf-8 -*- > "src\graph\utils\metrics.py"
echo """ >> "src\graph\utils\metrics.py"
echo Métricas para análise do grafo >> "src\graph\utils\metrics.py"
echo """ >> "src\graph\utils\metrics.py"
echo. >> "src\graph\utils\metrics.py"

echo # -*- coding: utf-8 -*- > "src\graph\utils\visualization.py"
echo """ >> "src\graph\utils\visualization.py"
echo Visualização do grafo >> "src\graph\utils\visualization.py"
echo """ >> "src\graph\utils\visualization.py"
echo. >> "src\graph\utils\visualization.py"

echo # -*- coding: utf-8 -*- > "src\graph\utils\export.py"
echo """ >> "src\graph\utils\export.py"
echo Exportação de dados do grafo >> "src\graph\utils\export.py"
echo """ >> "src\graph\utils\export.py"
echo. >> "src\graph\utils\export.py"

:: === SCRIPTS ===
echo # -*- coding: utf-8 -*- > "scripts\build_graph.py"
echo """ >> "scripts\build_graph.py"
echo Script principal para construção do grafo heterogêneo >> "scripts\build_graph.py"
echo """ >> "scripts\build_graph.py"
echo. >> "scripts\build_graph.py"
echo if __name__ == "__main__": >> "scripts\build_graph.py"
echo     print("Iniciando construção do grafo...") >> "scripts\build_graph.py"

echo # -*- coding: utf-8 -*- > "scripts\extract_entities.py"
echo """ >> "scripts\extract_entities.py"
echo Script para extração de entidades nomeadas >> "scripts\extract_entities.py"
echo """ >> "scripts\extract_entities.py"
echo. >> "scripts\extract_entities.py"

echo # -*- coding: utf-8 -*- > "scripts\calculate_similarities.py"
echo """ >> "scripts\calculate_similarities.py"
echo Script para cálculo de similaridades >> "scripts\calculate_similarities.py"
echo """ >> "scripts\calculate_similarities.py"
echo. >> "scripts\calculate_similarities.py"

echo # -*- coding: utf-8 -*- > "scripts\export_graph.py"
echo """ >> "scripts\export_graph.py"
echo Script para exportação do grafo >> "scripts\export_graph.py"
echo """ >> "scripts\export_graph.py"
echo. >> "scripts\export_graph.py"

:: Criar requirements para o grafo
echo [7/7] Criando requirements_graph.txt...
echo # Dependências mínimas para construção do grafo > "requirements_graph.txt"
echo. >> "requirements_graph.txt"
echo # Grafo e análise >> "requirements_graph.txt"
echo networkx^>=3.1 >> "requirements_graph.txt"
echo. >> "requirements_graph.txt"
echo # NLP e Embeddings >> "requirements_graph.txt"
echo spacy^>=3.6 >> "requirements_graph.txt"
echo sentence-transformers^>=2.2 >> "requirements_graph.txt"
echo scikit-learn^>=1.3 >> "requirements_graph.txt"
echo. >> "requirements_graph.txt"
echo # Cálculos numéricos >> "requirements_graph.txt"
echo numpy^>=1.24 >> "requirements_graph.txt"
echo scipy^>=1.10 >> "requirements_graph.txt"
echo pandas^>=2.0 >> "requirements_graph.txt"
echo. >> "requirements_graph.txt"
echo # Visualização básica >> "requirements_graph.txt"
echo matplotlib^>=3.7 >> "requirements_graph.txt"
echo. >> "requirements_graph.txt"
echo # Processamento de texto >> "requirements_graph.txt"
echo nltk^>=3.8 >> "requirements_graph.txt"

:: Criar notebooks básicos
echo # Análise do Grafo Heterogêneo > "notebooks\02_graph_analysis.ipynb"
echo # Experimentos de Similaridade > "notebooks\03_similarity_experiments.ipynb"

echo.
echo ================================================
echo           ESTRUTURA CRIADA COM SUCESSO!
echo ================================================
echo.
echo Próximos passos:
echo 1. Ative o ambiente virtual: venv\Scripts\activate
echo 2. Instale as dependências: pip install -r requirements_graph.txt
echo 3. Comece a implementar os modelos em: src\graph\models\
echo.
echo Estrutura criada em: %cd%
echo.
pause