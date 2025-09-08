"""
Configurações do projeto
"""
import os
from pathlib import Path
from dotenv import load_dotenv
import sys

# Carregar variáveis de ambiente
load_dotenv()

# Diretórios do projeto
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
CACHE_DIR = DATA_DIR / 'cache'
OUTPUT_DIR = DATA_DIR / 'output'

# Criar diretórios se não existirem
for dir_path in [DATA_DIR, CACHE_DIR, OUTPUT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Configuração do banco de dados com ajustes para Windows
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'jurisprudencia_db'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'admin'),
}

# URL do banco com encoding para Windows
if sys.platform == 'win32':
    # No Windows, usar configuração específica
    DATABASE_URL = (
        f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        f"?client_encoding=utf8"
    )
else:
    DATABASE_URL = (
        f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )

# Configuração do Redis
REDIS_CONFIG = {
    'host': os.getenv('REDIS_HOST', 'localhost'),
    'port': int(os.getenv('REDIS_PORT', '6379')),
    'db': int(os.getenv('REDIS_DB', '0')),
    'decode_responses': True,
}

# Configurações do scraper
SCRAPER_CONFIG = {
    'base_url': 'https://www.tjpi.jus.br/jurisprudencia/',
    'timeout': 30,
    'max_retries': 3,
    'retry_delay': 5,
    'concurrent_requests': 2,
}

# Headers para requisições
DEFAULT_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'pt-BR,pt;q=0.9,en;q=0.8',
    'Accept-Encoding': 'gzip, deflate, br',
}

# Configurações de logging
LOGGING_CONFIG = {
    'level': os.getenv('LOG_LEVEL', 'INFO'),
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': BASE_DIR / 'logs' / 'scraper.log',
}

# Criar diretório de logs
LOGGING_CONFIG['file'].parent.mkdir(parents=True, exist_ok=True)