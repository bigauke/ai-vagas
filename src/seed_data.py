import sqlite3
from datetime import datetime
from loguru import logger
import config
from database import init_db

def seed_database():
    """Popula a base de dados com vagas simuladas estruturadas conforme a base do Kaggle."""
    init_db()
    
    conn = sqlite3.connect(config.DATABASE_PATH)
    cursor = conn.cursor()
    
    vagas = [
        {
            "id": "mock_001",
            "title": "Engenheiro de Machine Learning",
            "company": "AeroTech IA",
            "location": "São Paulo, SP",
            "description": (
                "Estamos buscando um Engenheiro de Machine Learning para liderar o desenvolvimento "
                "de modelos preditivos. Requisitos: Forte domínio em Python, bibliotecas como Pandas, "
                "Scikit-learn, TensorFlow ou PyTorch. Experiência com Docker, controle de versão Git, "
                "e APIs (FastAPI/Flask). Diferencial: experiência com pipelines MLOps e nuvem AWS."
            ),
            "url": "https://www.linkedin.com/jobs/view/mock_001",
            "work_type": "Hybrid",
            "experience_level": "Mid-Senior level",
            "date_scraped": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "skills": "Python, Machine Learning, Pandas, Scikit-learn, TensorFlow, PyTorch, Docker, Git, FastAPI, Flask, AWS, MLOps"
        },
        {
            "id": "mock_002",
            "title": "Cientista de Dados Pleno",
            "company": "Fintech Giga",
            "location": "Rio de Janeiro, RJ",
            "description": (
                "Vaga para Cientista de Dados focado em modelagem de risco e análise preditiva. "
                "Requisitos fundamentais: Fluência em Python (Pandas, Numpy), SQL avançado para "
                "extração de bases, modelagem estatística, testes A/B, e ferramentas de visualização "
                "como Power BI ou Tableau. Habilidade para traduzir dados em insights de negócios."
            ),
            "url": "https://www.linkedin.com/jobs/view/mock_002",
            "work_type": "Remote",
            "experience_level": "Mid-Senior level",
            "date_scraped": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "skills": "Python, Pandas, Numpy, SQL, Statistics, A/B Testing, Power BI, Tableau"
        },
        {
            "id": "mock_003",
            "title": "Desenvolvedor Python Sênior",
            "company": "WebSoft Brasil",
            "location": "Belo Horizonte, MG",
            "description": (
                "Buscamos engenheiro de software com forte experiência em desenvolvimento back-end "
                "com Python. Requisitos: Frameworks Django ou FastAPI, arquitetura MVC/Microserviços, "
                "bancos de dados PostgreSQL e Redis, e testes unitários. Conhecimento em pipelines CI/CD "
                "e Docker são essenciais. Conhecimento básico em análise de dados é bem-vindo."
            ),
            "url": "https://www.linkedin.com/jobs/view/mock_003",
            "work_type": "Hybrid",
            "experience_level": "Mid-Senior level",
            "date_scraped": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "skills": "Python, Django, FastAPI, MVC, Microservices, PostgreSQL, Redis, Unit Testing, CI/CD, Docker"
        },
        {
            "id": "mock_004",
            "title": "Desenvolvedor Front-end React / TypeScript",
            "company": "PixStudio",
            "location": "Remoto",
            "description": (
                "Procuramos profissional focado em Front-end para criar interfaces incríveis e dinâmicas. "
                "Requisitos obrigatórios: JavaScript ES6+, TypeScript, React (Hooks, Context API), HTML5, "
                "CSS3/Sass ou Tailwind CSS, e integração de APIs REST. Experiência com Figma "
                "e metodologias ágeis (Scrum)."
            ),
            "url": "https://www.linkedin.com/jobs/view/mock_004",
            "work_type": "Remote",
            "experience_level": "Associate",
            "date_scraped": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "skills": "JavaScript, TypeScript, React, React Hooks, HTML5, CSS3, Sass, Tailwind CSS, REST APIs, Figma, Scrum"
        },
        {
            "id": "mock_005",
            "title": "Analista de Marketing Digital e SEO",
            "company": "Growth Hub",
            "location": "Curitiba, PR",
            "description": (
                "Vaga para Analista de Growth e Marketing Digital. Foco na criação de campanhas pagas "
                "(Google Ads, Facebook Ads), SEO técnico para indexação, e redação de artigos/blogs. "
                "Requisitos: Google Analytics, SEMRush, excelente escrita, e facilidade para interpretar "
                "métricas de conversão. Não é necessário saber programar, mas afinidade com tecnologia ajuda."
            ),
            "url": "https://www.linkedin.com/jobs/view/mock_005",
            "work_type": "On-site",
            "experience_level": "Entry level",
            "date_scraped": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "skills": "Google Ads, Facebook Ads, SEO, Google Analytics, SEMRush, Copywriting"
        }
    ]
    
    for vaga in vagas:
        # Insere a vaga
        cursor.execute("""
            INSERT OR REPLACE INTO jobs (id, title, company, location, description, url, date_scraped, work_type, experience_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            vaga['id'],
            vaga['title'],
            vaga['company'],
            vaga['location'],
            vaga['description'],
            vaga['url'],
            vaga['date_scraped'],
            vaga['work_type'],
            vaga['experience_level']
        ))
        
        # Insere as competências
        cursor.execute("""
            INSERT OR REPLACE INTO job_skills (job_id, skills)
            VALUES (?, ?)
        """, (
            vaga['id'],
            vaga['skills']
        ))
        
    conn.commit()
    conn.close()
    logger.info("Banco de dados populado com 5 vagas simuladas e suas competências!")

if __name__ == "__main__":
    seed_database()
