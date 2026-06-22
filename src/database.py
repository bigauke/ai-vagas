import sqlite3
import csv
import re
import datetime
from pathlib import Path
from loguru import logger
import config

def init_db():
    """Inicializa o banco de dados SQLite e cria as tabelas de vagas e competências."""
    conn = sqlite3.connect(config.DATABASE_PATH)
    cursor = conn.cursor()
    
    # Tabela principal de vagas
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            title TEXT,
            company TEXT,
            location TEXT,
            description TEXT,
            url TEXT,
            date_scraped TEXT,
            work_type TEXT,
            experience_level TEXT
        )
    """)
    
    # Tabela de mapeamento de competências da vaga
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS job_skills (
            job_id TEXT PRIMARY KEY,
            skills TEXT
        )
    """)
    
    conn.commit()
    conn.close()
    logger.info("Banco de dados SQLite inicializado com tabelas 'jobs' e 'job_skills'.")

def import_job_skills_fast(csv_path: str) -> dict:
    """
    Importa o arquivo job_skills.csv para a tabela job_skills no SQLite.
    Usa leitura por stream (csv module) e inserção em lote para máxima velocidade e economia de RAM.
    """
    init_db()
    
    csv_file = Path(csv_path)
    if not csv_file.exists():
        logger.error(f"Arquivo job_skills.csv não encontrado em: {csv_path}")
        return {"status": "error", "message": "Arquivo job_skills.csv não encontrado."}
        
    conn = sqlite3.connect(config.DATABASE_PATH)
    cursor = conn.cursor()
    # Otimizações de desempenho do SQLite para inserções em lote
    cursor.execute("PRAGMA synchronous = OFF")
    cursor.execute("PRAGMA journal_mode = MEMORY")
    
    id_pattern = re.compile(r'(\d+)$')
    total_imported = 0
    total_skipped = 0
    
    logger.info(f"Iniciando importação de competências de: {csv_path}")
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            
            # Encontra os índices das colunas
            col_map = {name: idx for idx, name in enumerate(header)} if header else {}
            idx_link = col_map.get('job_link', 0)
            idx_skills = col_map.get('job_skills', 1)
            
            batch = []
            batch_size = 50000
            
            for row in reader:
                if not row or len(row) <= max(idx_link, idx_skills):
                    total_skipped += 1
                    continue
                
                link = row[idx_link]
                skills = row[idx_skills]
                
                # Extrai o ID numérico da vaga a partir da URL
                path = link.split('?')[0].rstrip('/')
                match = id_pattern.search(path)
                if match:
                    job_id = match.group(1)
                    batch.append((job_id, skills))
                else:
                    total_skipped += 1
                    
                if len(batch) >= batch_size:
                    cursor.executemany("""
                        INSERT OR REPLACE INTO job_skills (job_id, skills)
                        VALUES (?, ?)
                    """, batch)
                    conn.commit()
                    total_imported += len(batch)
                    logger.info(f"Inseridos {total_imported} registros de competências...")
                    batch = []
            
            # Insere o lote final
            if batch:
                cursor.executemany("""
                    INSERT OR REPLACE INTO job_skills (job_id, skills)
                    VALUES (?, ?)
                """, batch)
                conn.commit()
                total_imported += len(batch)
                
        return {
            "status": "success",
            "imported": total_imported,
            "skipped": total_skipped,
            "message": f"Sucesso! {total_imported} mapeamentos de competências importados."
        }
    except Exception as e:
        logger.error(f"Erro ao processar importação de competências: {e}")
        return {"status": "error", "message": f"Erro de processamento: {str(e)}"}
    finally:
        conn.close()

def import_postings_fast(csv_path: str, max_rows: int = None) -> dict:
    """
    Importa o arquivo postings.csv para a tabela jobs no SQLite.
    Usa leitura por stream (csv module) e inserção em lote para máxima velocidade e economia de RAM.
    """
    init_db()
    
    csv_file = Path(csv_path)
    if not csv_file.exists():
        logger.error(f"Arquivo postings.csv não encontrado em: {csv_path}")
        return {"status": "error", "message": "Arquivo postings.csv não encontrado."}
        
    conn = sqlite3.connect(config.DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("PRAGMA synchronous = OFF")
    cursor.execute("PRAGMA journal_mode = MEMORY")
    
    total_imported = 0
    total_skipped = 0
    date_scraped = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    logger.info(f"Iniciando importação de vagas de: {csv_path}")
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            
            # Mapeamento dinâmico baseado no cabeçalho
            col_map = {name: idx for idx, name in enumerate(header)} if header else {}
            
            idx_id = col_map.get('job_id', 0)
            idx_company = col_map.get('company_name', 1)
            idx_title = col_map.get('title', 2)
            idx_desc = col_map.get('description', 3)
            idx_location = col_map.get('location', 6)
            idx_url = col_map.get('job_posting_url', 15)
            idx_work_type = col_map.get('formatted_work_type', 11)
            idx_exp = col_map.get('formatted_experience_level', 20)
            
            batch = []
            batch_size = 10000
            max_idx = max(idx_id, idx_company, idx_title, idx_desc, idx_location, idx_url, idx_work_type, idx_exp)
            
            for row in reader:
                if not row or len(row) <= max_idx:
                    total_skipped += 1
                    continue
                
                # Coleta valores
                job_id = str(row[idx_id]).strip()
                company = str(row[idx_company]).strip()
                title = str(row[idx_title]).strip()
                description = str(row[idx_desc]).strip()
                location = str(row[idx_location]).strip()
                url = str(row[idx_url]).strip()
                work_type = str(row[idx_work_type]).strip()
                experience_level = str(row[idx_exp]).strip()
                
                # Fallbacks e limpezas simples
                if not job_id:
                    job_id = f"kaggle_{total_imported}"
                if not url:
                    url = f"https://www.linkedin.com/jobs/view/{job_id}"
                
                batch.append((job_id, title, company, location, description, url, date_scraped, work_type, experience_level))
                
                if len(batch) >= batch_size:
                    cursor.executemany("""
                        INSERT OR REPLACE INTO jobs (id, title, company, location, description, url, date_scraped, work_type, experience_level)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, batch)
                    conn.commit()
                    total_imported += len(batch)
                    logger.info(f"Inseridos {total_imported} registros de vagas...")
                    batch = []
                    
                if max_rows and total_imported >= max_rows:
                    logger.info(f"Limite máximo de {max_rows} linhas importadas alcançado.")
                    break
            
            # Insere o lote final
            if batch and (not max_rows or total_imported < max_rows):
                cursor.executemany("""
                    INSERT OR REPLACE INTO jobs (id, title, company, location, description, url, date_scraped, work_type, experience_level)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, batch)
                conn.commit()
                total_imported += len(batch)
                
        return {
            "status": "success",
            "imported": total_imported,
            "skipped": total_skipped,
            "message": f"Sucesso! {total_imported} vagas importadas para o banco de dados."
        }
    except Exception as e:
        logger.error(f"Erro ao processar importação de vagas: {e}")
        return {"status": "error", "message": f"Erro de processamento: {str(e)}"}
    finally:
        conn.close()
