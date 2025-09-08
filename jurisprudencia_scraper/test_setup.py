"""
Script para testar a configuração do ambiente
"""
import sys
import os
from pathlib import Path

# Adicionar o diretório src ao path
sys.path.append(str(Path(__file__).parent / 'src'))

def test_imports():
    """Testa se todas as bibliotecas necessárias estão instaladas"""
    print("🔍 Testando imports...")
    try:
        import psycopg2
        import redis
        import selenium
        import requests
        import pandas as pd
        from bs4 import BeautifulSoup
        from sqlalchemy import create_engine
        print("✅ Todas as bibliotecas importadas com sucesso!")
        return True
    except ImportError as e:
        print(f"❌ Erro ao importar: {e}")
        return False

def test_postgresql():
    """Testa conexão com PostgreSQL"""
    print("🔍 Testando conexão com PostgreSQL...")
    try:
        from sqlalchemy import create_engine, text
        
        # Criar URL de conexão com encoding explícito
        DATABASE_URL = "postgresql://postgres:admin@localhost:5432/jurisprudencia_db"
        
        # Criar engine com configurações de encoding
        engine = create_engine(
            DATABASE_URL,
            connect_args={
                'client_encoding': 'utf8',
                'options': '-c client_encoding=utf8'
            },
            pool_pre_ping=True,
            echo=False
        )
        
        # Testar conexão
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version()"))
            version = result.fetchone()
            print("✅ PostgreSQL conectado com sucesso!")
            print(f"   Versão: {version[0][:50]}...")
            
            # Verificar tabelas
            result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name
            """))
            
            tables = result.fetchall()
            if tables:
                print(f"   Tabelas encontradas: {len(tables)}")
                for table in tables[:5]:  # Mostrar apenas as 5 primeiras
                    print(f"     - {table[0]}")
                if len(tables) > 5:
                    print(f"     ... e mais {len(tables) - 5} tabelas")
            else:
                print("   ⚠️ Nenhuma tabela encontrada")
            
        return True
        
    except Exception as e:
        print(f"❌ Erro ao conectar no PostgreSQL: {e}")
        
        # Tentar conexão alternativa com psycopg2
        try:
            import psycopg2
            print("   Tentando conexão alternativa com psycopg2...")
            
            conn = psycopg2.connect(
                host="localhost",
                port=5432,
                database="jurisprudencia_db",
                user="postgres",
                password="admin",
                options="-c client_encoding=utf8"
            )
            
            with conn.cursor() as cur:
                cur.execute("SELECT version()")
                version = cur.fetchone()
                print("✅ Conexão alternativa funcionou!")
                print(f"   Versão: {version[0][:50]}...")
            
            conn.close()
            return True
            
        except Exception as e2:
            print(f"   ❌ Conexão alternativa também falhou: {e2}")
            return False

def test_redis():
    """Testa conexão com Redis"""
    print("🔍 Testando conexão com Redis...")
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        r.ping()
        print("✅ Redis conectado e respondendo!")
        
        # Teste de escrita/leitura
        r.set('test_key', 'test_value', ex=10)
        value = r.get('test_key')
        if value == 'test_value':
            print("   Teste de escrita/leitura: OK")
        r.delete('test_key')
        return True
    except Exception as e:
        print(f"❌ Erro ao conectar no Redis: {e}")
        return False

def test_chrome_driver():
    """Testa se o Chrome driver está funcionando"""
    print("🔍 Testando Chrome driver...")
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        
        driver = webdriver.Chrome(options=options)
        driver.get("https://www.google.com")
        title = driver.title
        driver.quit()
        
        print(f"✅ Chrome driver funcionando! Título: {title}")
        return True
    except Exception as e:
        print(f"❌ Erro com Chrome driver: {e}")
        return False

def test_tjpi_access():
    """Testa acesso ao site do TJPI"""
    print("🔍 Testando acesso ao site do TJPI...")
    try:
        import requests
        response = requests.get(
            "https://www.tjpi.jus.br/portaltjpi/",
            timeout=10,
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        if response.status_code == 200:
            print(f"✅ Site acessível! Status: {response.status_code}")
            return True
        else:
            print(f"⚠️ Site respondeu com status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Erro ao acessar site: {e}")
        return False

def main():
    """Executa todos os testes"""
    print("=" * 60)
    print("TESTE DE CONFIGURAÇÃO DO AMBIENTE - JURISPRUDÊNCIA SCRAPER")
    print("=" * 60)
    
    results = []
    
    results.append(test_imports())
    results.append(test_postgresql())
    results.append(test_redis())
    results.append(test_chrome_driver())
    results.append(test_tjpi_access())
    
    passed = sum(results)
    total = len(results)
    
    print("=" * 60)
    if passed == total:
        print(f"✅ {passed}/{total} testes passaram - Ambiente configurado com sucesso!")
    else:
        print(f"⚠️ {passed}/{total} testes passaram")
        print("   Verifique os erros acima antes de continuar")
    print("=" * 60)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)