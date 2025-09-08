"""
Teste básico do scraper para verificar se conseguimos acessar o site
"""

import undetected_chromedriver as uc
from bs4 import BeautifulSoup
import time

def test_basic_access():
    """Testa acesso básico ao site de jurisprudência"""
    
    print("Iniciando navegador...")
    options = uc.ChromeOptions()
    # Comente a linha abaixo para ver o navegador funcionando
    # options.add_argument('--headless')
    
    driver = uc.Chrome(options=options)
    
    try:
        # Acessar página de busca
        url = "https://jurisprudencia.tjpi.jus.br/jurisprudences/search"
        print(f"Acessando: {url}")
        driver.get(url)
        
        # Aguardar página carregar
        time.sleep(3)
        
        # Pegar HTML
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        
        # Verificar título
        title = soup.find('title')
        print(f"Título da página: {title.text if title else 'Não encontrado'}")
        
        # Buscar por "consignado"
        search_input = driver.find_element("name", "q")
        search_input.send_keys("consignado")
        
        # Clicar em buscar
        search_button = driver.find_element("xpath", "//button[@type='submit']")
        search_button.click()
        
        # Aguardar resultados
        time.sleep(5)
        
        # Verificar se há resultados
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        results = soup.find_all('div', class_='card')
        
        print(f"Encontrados {len(results)} resultados na página")
        
        return True
        
    except Exception as e:
        print(f"Erro: {e}")
        return False
        
    finally:
        driver.quit()
        print("Navegador fechado")

if __name__ == "__main__":
    test_basic_access()