import os
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

# Caminho raiz do projeto
BASE_DIR = Path(__file__).resolve().parent.parent

# Carrega arquivo .env da raiz do projeto
load_dotenv(BASE_DIR / ".env")

# Cria diretórios necessários se não existirem
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Configurações de Armazenamento
DATABASE_PATH = BASE_DIR / os.getenv("DATABASE_PATH", "data/vagas.db")
KAGGLE_CSV_PATH = BASE_DIR / os.getenv("KAGGLE_CSV_PATH", "data/linkedin_job_postings.csv")

logger.info(f"Configurações carregadas com sucesso. BD: {DATABASE_PATH}")
