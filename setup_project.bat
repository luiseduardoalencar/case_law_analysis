@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

echo ====================================================
echo    SETUP AUTOMÁTICO - JURISPRUDENCIA SCRAPER
echo ====================================================
echo.

:: Criar estrutura de diretórios
echo [1/8] Criando estrutura de pastas...

mkdir jurisprudencia_scraper 2>nul
cd jurisprudencia_scraper

:: Estrutura principal
mkdir src\scrapers 2>nul
mkdir src\database 2>nul
mkdir src\utils 2>nul
mkdir src\config 2>nul
mkdir data\raw 2>nul
mkdir data\processed 2>nul
mkdir data\logs 2>nul
mkdir docker 2>nul
mkdir notebooks 2>nul
mkdir tests 2>nul

echo ✓ Estrutura de pastas criada!
echo.

:: ========================================
:: Criar arquivos __init__.py
:: ========================================
echo [2/8] Criando arquivos __init__.py...

echo # Package initialization > src\__init__.py
echo # Scrapers package > src\scrapers\__init__.py
echo # Database package > src\database\__init__.py
echo # Utils package > src\utils\__init__.py
echo # Config package > src\config\__init__.py

echo ✓ Arquivos __init__.py criados!
echo.

:: ========================================
:: Criar docker-compose.yml
:: ========================================
echo [3/8] Criando docker-compose.yml...

(
echo version: '3.8'
echo.
echo services:
echo   postgres:
echo     image: postgres:15
echo     container_name: jurisprudencia_postgres
echo     environment:
echo       POSTGRES_DB: jurisprudencia_db
echo       POSTGRES_USER: scraper_user
echo       POSTGRES_PASSWORD: scraper_pass_2024
echo     ports:
echo       - "5432:5432"
echo     volumes:
echo       - postgres_data:/var/lib/postgresql/data
echo       - ./init.sql:/docker-entrypoint-initdb.d/init.sql
echo     networks:
echo       - jurisprudencia_network
echo.
echo   redis:
echo     image: redis:7-alpine
echo     container_name: jurisprudencia_redis
echo     ports:
echo       - "6379:6379"
echo     volumes:
echo       - redis_data:/data
echo     command: redis-server --appendonly yes
echo     networks:
echo       - jurisprudencia_network
echo.
echo volumes:
echo   postgres_data:
echo   redis_data:
echo.
echo networks:
echo   jurisprudencia_network:
echo     driver: bridge
) > docker\docker-compose.yml

echo ✓ docker-compose.yml criado!
echo.

:: ========================================
:: Criar init.sql
:: ========================================
echo [4/8] Criando init.sql...

(
echo -- Tabela principal de processos
echo CREATE TABLE IF NOT EXISTS processos ^(
echo     id SERIAL PRIMARY KEY,
echo     numero_processo VARCHAR^(50^) UNIQUE NOT NULL,
echo     url_original TEXT NOT NULL,
echo     html_completo TEXT,
echo     data_coleta TIMESTAMP DEFAULT NOW^(^),
echo     status_processamento VARCHAR^(20^) DEFAULT 'pendente',
echo     hash_documento VARCHAR^(64^),
echo     tentativas INTEGER DEFAULT 0,
echo     created_at TIMESTAMP DEFAULT NOW^(^),
echo     updated_at TIMESTAMP DEFAULT NOW^(^)
echo ^);
echo.
echo -- Tabela de metadados
echo CREATE TABLE IF NOT EXISTS processos_metadados ^(
echo     id SERIAL PRIMARY KEY,
echo     processo_id INTEGER REFERENCES processos^(id^) ON DELETE CASCADE,
echo     orgao_julgador TEXT,
echo     orgao_julgador_colegiado TEXT,
echo     relator TEXT,
echo     classe_judicial TEXT,
echo     competencia TEXT,
echo     assunto_principal TEXT,
echo     autor TEXT,
echo     reu TEXT,
echo     data_publicacao DATE,
echo     tipo_decisao VARCHAR^(100^),
echo     created_at TIMESTAMP DEFAULT NOW^(^)
echo ^);
echo.
echo -- Tabela de conteúdo
echo CREATE TABLE IF NOT EXISTS processos_conteudo ^(
echo     id SERIAL PRIMARY KEY,
echo     processo_id INTEGER REFERENCES processos^(id^) ON DELETE CASCADE,
echo     tipo_secao VARCHAR^(50^),
echo     conteudo_html TEXT,
echo     conteudo_texto TEXT,
echo     conteudo_limpo TEXT,
echo     ordem INTEGER,
echo     created_at TIMESTAMP DEFAULT NOW^(^)
echo ^);
echo.
echo -- Tabela de log
echo CREATE TABLE IF NOT EXISTS scraping_log ^(
echo     id SERIAL PRIMARY KEY,
echo     url TEXT,
echo     tipo_operacao VARCHAR^(20^),
echo     status VARCHAR^(20^),
echo     tentativas INTEGER DEFAULT 0,
echo     erro_mensagem TEXT,
echo     timestamp_inicio TIMESTAMP,
echo     timestamp_fim TIMESTAMP,
echo     duracao_segundos NUMERIC^(10,2^)
echo ^);
echo.
echo -- Índices
echo CREATE INDEX IF NOT EXISTS idx_processo_numero ON processos^(numero_processo^);
echo CREATE INDEX IF NOT EXISTS idx_processo_status ON processos^(status_processamento^);
echo CREATE INDEX IF NOT EXISTS idx_metadados_processo ON processos_metadados^(processo_id^);
echo CREATE INDEX IF NOT EXISTS idx_conteudo_processo ON processos_conteudo^(processo_id^);
) > docker\init.sql

echo ✓ init.sql criado!
echo.

:: ========================================
:: Criar requirements.txt
:: ========================================
echo [5/8] Criando requirements.txt...

(
echo selenium==4.15.0
echo undetected-chromedriver==3.5.4
echo beautifulsoup4==4.12.2
echo lxml==4.9.3
echo psycopg2-binary==2.9.9
echo redis==5.0.1
echo SQLAlchemy==2.0.23
echo python-dotenv==1.0.0
echo requests==2.31.0
echo pandas==2.1.4
echo loguru==0.7.2
echo tenacity==8.2.3
echo colorama==0.4.6
echo tqdm==4.66.1
) > requirements.txt

echo ✓ requirements.txt criado!
echo.

:: ========================================
:: Criar .env
:: ========================================
echo [6/8] Criando .env...

(
echo # Database
echo DB_HOST=localhost
echo DB_PORT=5432
echo DB_NAME=jurisprudencia_db
echo DB_USER=scraper_user
echo DB_PASSWORD=scraper_pass_2024
echo.
echo # Redis
echo REDIS_HOST=localhost
echo REDIS_PORT=6379
echo REDIS_DB=0
echo.
echo # Scraping
echo BASE_URL=https://jurisprudencia.tjpi.jus.br
echo MAX_WORKERS=3
echo DELAY_MIN=2
echo DELAY_MAX=4
echo HEADLESS=True
echo TIMEOUT=30
echo.
echo # Logging
echo LOG_LEVEL=INFO
echo LOG_FILE=data/logs/scraper.log
) > .env

echo ✓ .env criado!
echo.

:: ========================================
:: Criar settings.py
:: ========================================
echo [7/8] Criando settings.py...

(
echo import os
echo from pathlib import Path
echo from dotenv import load_dotenv
echo.
echo # Carregar variáveis de ambiente
echo load_dotenv^(^)
echo.
echo # Paths
echo BASE_DIR = Path^(__file__^).resolve^(^).parent.parent.parent
echo DATA_DIR = BASE_DIR / "data"
echo RAW_DATA_DIR = DATA_DIR / "raw"
echo PROCESSED_DATA_DIR = DATA_DIR / "processed"
echo LOG_DIR = DATA_DIR / "logs"
echo.
echo # Criar diretórios se não existirem
echo for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, LOG_DIR]:
echo     dir_path.mkdir^(parents=True, exist_ok=True^)
echo.
echo # Database
echo DB_CONFIG = {
echo     'host': os.getenv^('DB_HOST', 'localhost'^),
echo     'port': int^(os.getenv^('DB_PORT', 5432^)^),
echo     'database': os.getenv^('DB_NAME', 'jurisprudencia_db'^),
echo     'user': os.getenv^('DB_USER', 'scraper_user'^),
echo     'password': os.getenv^('DB_PASSWORD', 'scraper_pass_2024'^)
echo }
echo.
echo # Redis
echo REDIS_CONFIG = {
echo     'host': os.getenv^('REDIS_HOST', 'localhost'^),
echo     'port': int^(os.getenv^('REDIS_PORT', 6379^)^),
echo     'db': int^(os.getenv^('REDIS_DB', 0^)^)
echo }
echo.
echo # Scraping
echo SCRAPING_CONFIG = {
echo     'base_url': os.getenv^('BASE_URL', 'https://jurisprudencia.tjpi.jus.br'^),
echo     'max_workers': int^(os.getenv^('MAX_WORKERS', 3^)^),
echo     'delay_min': float^(os.getenv^('DELAY_MIN', 2^)^),
echo     'delay_max': float^(os.getenv^('DELAY_MAX', 4^)^),
echo     'headless': os.getenv^('HEADLESS', 'True'^).lower^(^) == 'true',
echo     'timeout': int^(os.getenv^('TIMEOUT', 30^)^)
echo }
echo.
echo # Logging
echo LOG_CONFIG = {
echo     'level': os.getenv^('LOG_LEVEL', 'INFO'^),
echo     'file': LOG_DIR / 'scraper.log'
echo }
) > src\config\settings.py

echo ✓ settings.py criado!
echo.

:: ========================================
:: Criar logger.py
:: ========================================
echo [8/8] Criando logger.py...

(
echo from loguru import logger
echo import sys
echo from pathlib import Path
echo.
echo # Adicionar src ao path se necessário
echo src_path = Path^(__file__^).parent.parent
echo if str^(src_path^) not in sys.path:
echo     sys.path.append^(str^(src_path^)^)
echo.
echo from config.settings import LOG_CONFIG
echo.
echo # Remover handler padrão
echo logger.remove^(^)
echo.
echo # Console handler
echo logger.add^(
echo     sys.stdout,
echo     format="^<green^>{time:YYYY-MM-DD HH:mm:ss}^</green^> ^| ^<level^>{level: ^<8}^</level^> ^| ^<cyan^>{name}^</cyan^>:^<cyan^>{function}^</cyan^> - ^<level^>{message}^</level^>",
echo     level=LOG_CONFIG['level']
echo ^)
echo.
echo # File handler
echo logger.add^(
echo     LOG_CONFIG['file'],
echo     format="{time:YYYY-MM-DD HH:mm:ss} ^| {level: ^<8} ^| {name}:{function}:{line} - {message}",
echo     level=LOG_CONFIG['level'],
echo     rotation="10 MB",
echo     retention="7 days",
echo     compression="zip"
echo ^)
) > src\utils\logger.py

echo ✓ logger.py criado!
echo.

:: ========================================
:: Criar arquivo README.md
:: ========================================
(
echo # Jurisprudência Scraper - TJPI
echo.
echo ## Estrutura do Projeto
echo.
echo ```
echo jurisprudencia_scraper/
echo ├── docker/              # Configurações Docker
echo │   ├── docker-compose.yml
echo │   └── init.sql
echo ├── src/                 # Código fonte
echo │   ├── scrapers/        # Módulos de scraping
echo │   ├── database/        # Conexões e modelos
echo │   ├── utils/           # Utilitários
echo │   └── config/          # Configurações
echo ├── data/                # Dados
echo │   ├── raw/            # HTML bruto
echo │   ├── processed/      # Dados processados
echo │   └── logs/           # Logs do sistema
echo ├── tests/              # Testes
echo ├── notebooks/          # Análises exploratórias
echo ├── requirements.txt    # Dependências Python
echo ├── .env                # Variáveis de ambiente
echo └── README.md          # Este arquivo
echo ```
echo.
echo ## Setup Inicial
echo.
echo 1. Instalar Python 3.10+
echo 2. Criar ambiente virtual: `python -m venv venv`
echo 3. Ativar ambiente: `venv\Scripts\activate`
echo 4. Instalar dependências: `pip install -r requirements.txt`
echo 5. Iniciar Docker: `cd docker ^&^& docker-compose up -d`
echo 6. Testar setup: `python test_setup.py`
echo.
echo ## Fases do Projeto
echo.
echo - **Fase 1**: Scraping e Armazenamento ^(atual^)
echo - **Fase 2**: Construção do Grafo
echo - **Fase 3**: Fine-tuning LLM
echo - **Fase 4**: Validação e Testes
) > README.md

:: ========================================
:: Criar arquivo de teste
:: ========================================
(
echo import sys
echo from pathlib import Path
echo sys.path.append^(str^(Path^(__file__^).parent^)^)
echo.
echo def test_imports^(^):
echo     """Testa imports"""
echo     print^("Testando imports..."^)
echo     try:
echo         import selenium
echo         import undetected_chromedriver
echo         import bs4
echo         import psycopg2
echo         import redis
echo         print^("✓ Imports OK"^)
echo         return True
echo     except ImportError as e:
echo         print^(f"✗ Erro: {e}"^)
echo         return False
echo.
echo if __name__ == "__main__":
echo     test_imports^(^)
) > test_setup.py

echo.
echo ====================================================
echo    SETUP CONCLUÍDO COM SUCESSO!
echo ====================================================
echo.
echo Próximos passos:
echo.
echo 1. Abrir terminal na pasta: jurisprudencia_scraper
echo 2. Criar ambiente virtual: python -m venv venv
echo 3. Ativar ambiente: venv\Scripts\activate
echo 4. Instalar dependências: pip install -r requirements.txt
echo 5. Iniciar Docker: cd docker ^& docker-compose up -d
echo.
pause