#!/usr/bin/env python3
"""
Script para diagnosticar e corrigir problemas com Chrome Driver
"""

import sys
import time
import os
from pathlib import Path

def test_chrome_configurations():
    """Testa diferentes configurações do Chrome"""
    
    print("🧪 Testando configurações do Chrome Driver...")
    
    configurations = [
        {
            'name': 'Configuração Básica',
            'options': [
                '--no-sandbox',
                '--disable-dev-shm-usage',
                '--disable-gpu',
                '--window-size=1920,1080'
            ]
        },
        {
            'name': 'Configuração Estável',
            'options': [
                '--no-sandbox',
                '--disable-dev-shm-usage',
                '--disable-gpu',
                '--disable-extensions',
                '--disable-plugins',
                '--disable-images',
                '--disable-javascript',  # Pode ajudar se houver problemas com JS
                '--window-size=1920,1080',
                '--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            ]
        },
        {
            'name': 'Configuração Compatibilidade',
            'options': [
                '--no-sandbox',
                '--disable-dev-shm-usage', 
                '--disable-gpu',
                '--disable-web-security',
                '--disable-features=VizDisplayCompositor',
                '--disable-blink-features=AutomationControlled',
                '--window-size=1920,1080',
                '--remote-debugging-port=9222'
            ]
        }
    ]
    
    for config in configurations:
        print(f"\n🔧 Testando: {config['name']}")
        
        try:
            import undetected_chromedriver as uc
            
            options = uc.ChromeOptions()
            for option in config['options']:
                options.add_argument(option)
            
            # Adicionar configurações experimentais
            options.add_experimental_option("excludeSwitches", ["enable-automation"])
            options.add_experimental_option('useAutomationExtension', False)
            
            # Tentar inicializar o driver
            driver = uc.Chrome(options=options, version_main=None)
            
            # Teste simples
            driver.get("https://www.google.com")
            title = driver.title
            
            print(f"   ✅ Sucesso! Título: {title}")
            
            # Testar o site do TJPI
            try:
                driver.get("https://jurisprudencia.tjpi.jus.br/jurisprudences/search")
                time.sleep(3)
                tjpi_title = driver.title
                print(f"   ✅ TJPI acessível! Título: {tjpi_title}")
                
            except Exception as e:
                print(f"   ⚠️ TJPI não acessível: {e}")
            
            driver.quit()
            print(f"   ✅ {config['name']} funcionou!")
            return config  # Retorna a primeira configuração que funciona
            
        except Exception as e:
            print(f"   ❌ {config['name']} falhou: {e}")
            continue
    
    print("\n❌ Nenhuma configuração funcionou")
    return None

def create_improved_scraper_config():
    """Cria arquivo de configuração melhorada"""
    
    working_config = test_chrome_configurations()
    
    if not working_config:
        print("❌ Não foi possível encontrar configuração funcional")
        return
    
    config_code = f'''# Configuração do Chrome que funcionou: {working_config['name']}

def get_chrome_options():
    """Retorna opções do Chrome que funcionam"""
    import undetected_chromedriver as uc
    
    options = uc.ChromeOptions()
    
    # Opções testadas e aprovadas
'''
    
    for option in working_config['options']:
        config_code += f"    options.add_argument('{option}')\n"
    
    config_code += '''
    # Configurações experimentais
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    
    # Desabilitar logs verbosos
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument('--disable-logging')
    options.add_argument('--log-level=3')
    
    return options

def create_stable_driver():
    """Cria driver Chrome estável"""
    import undetected_chromedriver as uc
    
    options = get_chrome_options()
    
    try:
        driver = uc.Chrome(options=options, version_main=None)
        driver.set_page_load_timeout(30)
        driver.implicitly_wait(10)
        return driver
    except Exception as e:
        print(f"Erro ao criar driver: {e}")
        raise
'''
    
    # Salvar configuração
    with open('chrome_config.py', 'w', encoding='utf-8') as f:
        f.write(config_code)
    
    print(f"✅ Configuração salva em 'chrome_config.py'")
    print(f"   Use: from chrome_config import create_stable_driver")

def check_system_requirements():
    """Verifica requisitos do sistema"""
    
    print("🔍 Verificando requisitos do sistema...")
    
    # Verificar Python
    python_version = sys.version
    print(f"   Python: {python_version}")
    
    # Verificar bibliotecas essenciais
    libraries = [
        'selenium',
        'undetected_chromedriver', 
        'beautifulsoup4',
        'psycopg2',
        'redis',
        'loguru'
    ]
    
    missing_libraries = []
    
    for lib in libraries:
        try:
            __import__(lib.replace('-', '_'))
            print(f"   ✅ {lib}")
        except ImportError:
            print(f"   ❌ {lib} - FALTANDO")
            missing_libraries.append(lib)
    
    if missing_libraries:
        print(f"\n⚠️ Instale as bibliotecas faltantes:")
        print(f"   pip install {' '.join(missing_libraries)}")
    
    # Verificar Chrome/Chromium
    chrome_paths = [
        '/usr/bin/google-chrome',
        '/usr/bin/chromium-browser', 
        'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
        'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe'
    ]
    
    chrome_found = False
    for path in chrome_paths:
        if os.path.exists(path):
            print(f"   ✅ Chrome encontrado: {path}")
            chrome_found = True
            break
    
    if not chrome_found:
        print(f"   ⚠️ Chrome não encontrado nos caminhos padrão")
        print(f"   Instale o Google Chrome se necessário")

if __name__ == "__main__":
    print("=" * 60)
    print("DIAGNÓSTICO E CORREÇÃO - CHROME DRIVER")
    print("=" * 60)
    
    check_system_requirements()
    print("\n" + "=" * 60)
    create_improved_scraper_config()