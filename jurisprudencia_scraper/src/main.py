#!/usr/bin/env python3
"""
Scraper TJPI - Versão Final para 3000 processos
"""

import sys
import time
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import re

# Adicionar src ao path
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

import undetected_chromedriver as uc
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import psycopg2
import redis
from loguru import logger

from config.settings import DB_CONFIG, REDIS_CONFIG


class TJPIScraperFinal:
    """Scraper TJPI para 3000 processos com extração completa"""
    
    def __init__(self):
        self.setup_logging()
        self.db_conn = psycopg2.connect(**DB_CONFIG)
        self.redis_conn = redis.Redis(**REDIS_CONFIG)
        self.driver = None
        
    def setup_logging(self):
        logger.remove()
        logger.add(sys.stdout, level="INFO")
        logger.add("../../data/logs/scraper.log", level="DEBUG")
        
    def create_driver(self):
        options = uc.ChromeOptions()
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        
        self.driver = uc.Chrome(options=options)
        self.driver.set_page_load_timeout(30)
        logger.info("Driver criado")
    
    def extract_process_urls_from_page(self) -> List[str]:
        """Extrai URLs dos processos individuais da página atual"""
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        urls = []
        
        # Buscar pelos divs com class "callout callout-danger"
        callout_divs = soup.find_all('div', class_='callout callout-danger')
        
        for div in callout_divs:
            # Buscar link dentro do div
            link = div.find('a', href=True)
            if link:
                href = link['href']
                
                # Verificar se é um link de processo individual
                if '/jurisprudences/' in href and '/public' in href:
                    full_url = f"https://jurisprudencia.tjpi.jus.br{href}"
                    urls.append(full_url)
        
        # Se não encontrou pelos callouts, tentar método alternativo
        if not urls:
            all_links = soup.find_all('a', href=True)
            for link in all_links:
                href = link['href']
                if '/jurisprudences/' in href and '/public' in href:
                    full_url = f"https://jurisprudencia.tjpi.jus.br{href}"
                    if full_url not in urls:
                        urls.append(full_url)
        
        return urls
    
    def scrape_process(self, url: str) -> Dict:
        """Faz scraping de um processo específico"""
        try:
            # Verificar se já foi processado
            url_hash = hashlib.md5(url.encode()).hexdigest()
            if self.redis_conn.exists(f"processed:{url_hash}"):
                return {'status': 'already_processed'}
            
            # Acessar URL do processo
            self.driver.get(url)
            time.sleep(3)
            
            # Extrair HTML
            html_content = self.driver.page_source
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extrair dados estruturados
            numero_processo = self.extract_process_number(soup.get_text())
            metadados = self.extract_metadata(soup)
            conteudo_secoes = self.extract_content_sections(soup)
            
            # Verificar se número já existe no banco
            if self.process_exists_in_db(numero_processo):
                logger.info(f"Processo {numero_processo} já existe no banco")
                self.redis_conn.setex(f"processed:{url_hash}", 86400, "1")
                return {'status': 'already_exists'}
            
            # Salvar no banco com transação individual
            process_id = self.save_to_db_complete(url, html_content, numero_processo, metadados, conteudo_secoes)
            
            # Marcar como processado
            self.redis_conn.setex(f"processed:{url_hash}", 86400, "1")
            
            return {
                'status': 'success', 
                'id': process_id, 
                'numero': numero_processo,
                'metadados_count': len(metadados),
                'secoes_count': len(conteudo_secoes)
            }
            
        except Exception as e:
            logger.error(f"Erro ao processar {url}: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def extract_process_number(self, text: str) -> str:
        """Extrai número do processo"""
        # Padrão CNJ: 1234567-12.3456.7.89.0123
        pattern = r'\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}'
        match = re.search(pattern, text)
        return match.group() if match else f"processo_{int(time.time())}"
    
    def extract_metadata(self, soup: BeautifulSoup) -> Dict:
        """Extrai metadados do processo"""
        metadados = {}
        
        try:
            text_content = soup.get_text()
            
            # Extrair relator
            relator_patterns = [
                r'relator:?\s*([^\n]+)',
                r'des\.?\s*([^\n]+)',
                r'desembargador\s+([^\n]+)'
            ]
            for pattern in relator_patterns:
                match = re.search(pattern, text_content, re.IGNORECASE)
                if match:
                    metadados['relator'] = match.group(1).strip()[:200]
                    break
            
            # Extrair órgão julgador
            orgao_patterns = [
                r'câmara[^:]*:?\s*([^\n]+)',
                r'turma[^:]*:?\s*([^\n]+)',
                r'(\d+ª\s*câmara[^,\n]*)'
            ]
            for pattern in orgao_patterns:
                match = re.search(pattern, text_content, re.IGNORECASE)
                if match:
                    metadados['orgao_julgador'] = match.group(1).strip()[:200]
                    break
            
            # Extrair classe judicial
            classe_patterns = [
                r'classe:?\s*([^\n\(]+)',
                r'apelação\s+cível',
                r'agravo\s+de\s+instrumento'
            ]
            for pattern in classe_patterns:
                match = re.search(pattern, text_content, re.IGNORECASE)
                if match:
                    metadados['classe_judicial'] = match.group(0).strip()[:200]
                    break
            
            # Extrair assunto principal
            if 'empréstimo consignado' in text_content.lower():
                metadados['assunto_principal'] = 'Empréstimo Consignado'
            elif 'consignado' in text_content.lower():
                metadados['assunto_principal'] = 'Consignado'
            
            # Extrair data de publicação
            data_patterns = [
                r'publicação:?\s*(\d{2}/\d{2}/\d{4})',
                r'data:?\s*(\d{2}/\d{2}/\d{4})'
            ]
            for pattern in data_patterns:
                match = re.search(pattern, text_content, re.IGNORECASE)
                if match:
                    metadados['data_publicacao'] = match.group(1)
                    break
            
            # Extrair partes (autor/réu)
            apelante_match = re.search(r'apelante:?\s*([^\n]+)', text_content, re.IGNORECASE)
            if apelante_match:
                metadados['autor'] = apelante_match.group(1).strip()[:200]
            
            apelado_match = re.search(r'apelado:?\s*([^\n]+)', text_content, re.IGNORECASE)
            if apelado_match:
                metadados['reu'] = apelado_match.group(1).strip()[:200]
                
        except Exception as e:
            logger.warning(f"Erro ao extrair metadados: {e}")
            
        return metadados
    
    def extract_content_sections(self, soup: BeautifulSoup) -> List[Dict]:
        """Extrai seções de conteúdo do processo"""
        sections = []
        
        try:
            text_content = soup.get_text()
            
            # Seções típicas de jurisprudência
            section_patterns = {
                'ementa': r'ementa[:\s]*([^\.]+(?:\.[^\.]+)*)',
                'relatório': r'relatório[:\s]*([^\.]+(?:\.[^\.]+)*)',
                'voto': r'voto[:\s]*([^\.]+(?:\.[^\.]+)*)', 
                'decisão': r'decisão[:\s]*([^\.]+(?:\.[^\.]+)*)',
                'fundamentos': r'(?:fundament|razões)[^:]*[:\s]*([^\.]+(?:\.[^\.]+)*)',
                'dispositivo': r'(?:diante|pelo)\s+exposto[^\.]*\.([^\.]+(?:\.[^\.]+)*)'
            }
            
            ordem = 1
            for secao_tipo, pattern in section_patterns.items():
                match = re.search(pattern, text_content, re.IGNORECASE | re.DOTALL)
                if match:
                    conteudo = match.group(1).strip()
                    if len(conteudo) > 50:  # Só incluir se tiver conteúdo substancial
                        sections.append({
                            'tipo_secao': secao_tipo,
                            'conteudo_texto': conteudo[:3000],  # Limitar tamanho
                            'ordem': ordem
                        })
                        ordem += 1
            
            # Se não encontrou seções específicas, dividir por parágrafos grandes
            if not sections:
                paragraphs = text_content.split('\n\n')
                large_paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 100]
                
                for i, paragrafo in enumerate(large_paragraphs[:10]):  # Máximo 10 parágrafos
                    sections.append({
                        'tipo_secao': f'paragrafo_{i+1}',
                        'conteudo_texto': paragrafo[:3000],
                        'ordem': i + 1
                    })
                    
        except Exception as e:
            logger.warning(f"Erro ao extrair seções: {e}")
            
        return sections
    
    def process_exists_in_db(self, numero_processo: str) -> bool:
        """Verifica se processo já existe no banco"""
        try:
            with self.db_conn.cursor() as cursor:
                cursor.execute(
                    "SELECT 1 FROM processos WHERE numero_processo = %s LIMIT 1",
                    (numero_processo,)
                )
                return cursor.fetchone() is not None
        except Exception:
            return False
    
    def save_to_db_complete(self, url: str, html: str, numero: str, metadados: Dict, conteudo: List[Dict]) -> int:
        """Salva processo completo no banco com metadados e conteúdo"""
        conn = psycopg2.connect(**DB_CONFIG)
        
        try:
            with conn:
                with conn.cursor() as cursor:
                    # Inserir processo principal
                    cursor.execute("""
                        INSERT INTO processos (numero_processo, url_original, html_completo, 
                                             hash_documento, data_coleta, status_processamento)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        RETURNING id
                    """, (
                        numero,
                        url, 
                        html,
                        hashlib.md5(html.encode()).hexdigest(),
                        datetime.now(),
                        'coletado'
                    ))
                    
                    process_id = cursor.fetchone()[0]
                    
                    # Inserir metadados se existirem
                    if metadados:
                        # Tratar data de publicação especialmente
                        data_pub = metadados.get('data_publicacao')
                        if data_pub and isinstance(data_pub, str) and '/' in data_pub:
                            # Converter DD/MM/AAAA para AAAA-MM-DD
                            try:
                                from datetime import datetime as dt
                                data_obj = dt.strptime(data_pub, '%d/%m/%Y')
                                data_pub = data_obj.strftime('%Y-%m-%d')
                            except ValueError:
                                data_pub = None  # Se não conseguir converter, salvar como NULL
                        
                        cursor.execute("""
                            INSERT INTO processos_metadados 
                            (processo_id, orgao_julgador, relator, classe_judicial, 
                             assunto_principal, data_publicacao, autor, reu)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """, (
                            process_id,
                            metadados.get('orgao_julgador'),
                            metadados.get('relator'),
                            metadados.get('classe_judicial'),
                            metadados.get('assunto_principal'),
                            data_pub,  # Data já convertida ou None
                            metadados.get('autor'),
                            metadados.get('reu')
                        ))
                    
                    # Inserir seções de conteúdo
                    for secao in conteudo:
                        cursor.execute("""
                            INSERT INTO processos_conteudo 
                            (processo_id, tipo_secao, conteudo_texto, ordem)
                            VALUES (%s, %s, %s, %s)
                        """, (
                            process_id,
                            secao['tipo_secao'],
                            secao['conteudo_texto'],
                            secao['ordem']
                        ))
                    
                    conn.commit()
                    
            conn.close()
            return process_id
            
        except psycopg2.IntegrityError as e:
            conn.rollback()
            conn.close()
            if 'duplicate key' in str(e):
                logger.warning(f"Processo {numero} já existe no banco")
                return -1
            else:
                raise e
        except psycopg2.DataError as e:
            conn.rollback()
            conn.close()
            logger.error(f"Erro de formato de dados para processo {numero}: {e}")
            raise e
        except Exception as e:
            conn.rollback()
            conn.close()
            logger.error(f"Erro inesperado ao salvar processo {numero}: {e}")
            raise e
    
    def run(self, max_pages: int = 120):
        """Executa o scraping completo limitado a 3000 processos"""
        logger.info("SCRAPER TJPI - EMPRÉSTIMOS CONSIGNADOS")
        logger.info(f"Período: 2024-01-01 a 2025-09-30")
        logger.info(f"Máximo de páginas: {max_pages} (até 3000 processos)")
        
        try:
            self.create_driver()
            
            # Processar página por página dinamicamente
            base_url = "https://jurisprudencia.tjpi.jus.br/jurisprudences/search?classe=&data_max=2025-09-30&data_min=2024-01-01&orgao=&q=consignado&relator=&tipo="
            
            total_processed = 0
            success = 0
            errors = 0
            already_exists = 0
            
            for page in range(1, max_pages + 1):
                logger.info(f"=== PÁGINA {page}/{max_pages} ===")
                
                # URL da página atual
                if page == 1:
                    page_url = base_url
                else:
                    page_url = f"{base_url}&page={page}"
                
                try:
                    # Acessar página de resultados
                    self.driver.get(page_url)
                    time.sleep(3)
                    
                    # Extrair URLs dos processos da página
                    page_urls = self.extract_process_urls_from_page()
                    
                    if not page_urls:
                        logger.info("Nenhuma URL encontrada - fim das páginas")
                        break
                    
                    logger.info(f"Encontradas {len(page_urls)} URLs na página {page}")
                    
                    # Processar cada URL da página imediatamente
                    for i, url in enumerate(page_urls, 1):
                        total_processed += 1
                        logger.info(f"[{total_processed}] Processando: {url}")
                        
                        result = self.scrape_process(url)
                        
                        if result['status'] == 'success':
                            success += 1
                            logger.info(f"   ✅ Salvo: {result['numero']} ({result['metadados_count']} metadados, {result['secoes_count']} seções)")
                        elif result['status'] == 'already_exists':
                            already_exists += 1
                            logger.info(f"   ⚠️ Já existe no banco")
                        elif result['status'] == 'error':
                            errors += 1
                            logger.error(f"   ❌ Erro: {result.get('error', 'Desconhecido')}")
                        
                        # Pausa entre processos
                        time.sleep(2)
                        
                        # Log de progresso a cada 25 processos
                        if total_processed % 25 == 0:
                            logger.info(f"PROGRESSO GERAL: {total_processed} processados | Sucessos: {success} | Já existem: {already_exists} | Erros: {errors}")
                        
                        # Parar se atingir 3000 processos
                        if total_processed >= 3000:
                            logger.info("Limite de 3000 processos atingido!")
                            break
                    
                    # Pausa entre páginas
                    time.sleep(3)
                    
                    # Parar se atingiu o limite
                    if total_processed >= 3000:
                        break
                        
                except Exception as e:
                    logger.error(f"Erro na página {page}: {e}")
                    continue
            
            logger.info("=" * 60)
            logger.info(f"SCRAPING CONCLUÍDO!")
            logger.info(f"Total processado: {total_processed}")
            logger.info(f"Sucessos: {success}")
            logger.info(f"Já existiam: {already_exists}")
            logger.info(f"Erros: {errors}")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Erro geral: {e}")
        finally:
            if self.driver:
                self.driver.quit()
            if self.db_conn:
                self.db_conn.close()


def main():
    """Função principal"""
    scraper = TJPIScraperFinal()
    
    # Processar até 120 páginas (3000 processos)
    scraper.run(max_pages=120)


if __name__ == "__main__":
    main()