# Jurisprudência Scraper - TJPI

## Estrutura do Projeto

```
jurisprudencia_scraper/
├── docker/              # Configurações Docker
│   ├── docker-compose.yml
│   └── init.sql
├── src/                 # Código fonte
│   ├── scrapers/        # Módulos de scraping
│   ├── database/        # Conexões e modelos
│   ├── utils/           # Utilitários
│   └── config/          # Configurações
├── data/                # Dados
│   ├── raw/            # HTML bruto
│   ├── processed/      # Dados processados
│   └── logs/           # Logs do sistema
├── tests/              # Testes
├── notebooks/          # Análises exploratórias
├── requirements.txt    # Dependências Python
├── .env                # Variáveis de ambiente
└── README.md          # Este arquivo
```

## Setup Inicial

1. Instalar Python 3.10+
2. Criar ambiente virtual: `python -m venv venv`
3. Ativar ambiente: `venv\Scripts\activate`
4. Instalar dependências: `pip install -r requirements.txt`
5. Iniciar Docker: `cd docker && docker-compose up -d`
6. Testar setup: `python test_setup.py`

## Fases do Projeto

- **Fase 1**: Scraping e Armazenamento (atual)
- **Fase 2**: Construção do Grafo
- **Fase 3**: Fine-tuning LLM
- **Fase 4**: Validação e Testes
