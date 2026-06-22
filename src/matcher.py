import sqlite3
import numpy as np
from typing import List, Dict, Any, Tuple
from pypdf import PdfReader
from loguru import logger
import config

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extrai texto bruto de um arquivo PDF."""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text_content = page.extract_text()
            if text_content:
                text += text_content + "\n"
        return text.strip()
    except Exception as e:
        logger.error(f"Erro ao ler PDF {pdf_path}: {e}")
        raise ValueError("Não foi possível ler o arquivo PDF.")

def cosine_similarity(v1, v2):
    """Calcula similaridade de cosseno entre dois vetores."""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return float(dot_product / (norm_v1 * norm_v2))

def calculate_local_similarity(resume_text: str, jobs: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], float]]:
    """Cálculo de similaridade local usando TF-IDF."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    logger.info("Utilizando TF-IDF local para triagem de similaridade.")
    corpus = [resume_text] + [job['description'] for job in jobs]
    
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    resume_vector = tfidf_matrix[0].toarray()[0]
    
    ranked_jobs = []
    for idx, job in enumerate(jobs):
        job_vector = tfidf_matrix[idx + 1].toarray()[0]
        sim = cosine_similarity(resume_vector, job_vector)
        ranked_jobs.append((job, sim))
        
    return sorted(ranked_jobs, key=lambda x: x[1], reverse=True)

def analyze_top_job(resume_text: str, job: Dict[str, Any], score_est: float) -> Dict[str, Any]:
    """Realiza análise detalhada da vaga em relação ao currículo usando heurísticas locais."""
    logger.info(f"Gerando análise local para a vaga {job['id']}.")
    
    # Heurística de Fit/No Fit (limiar calibrado localmente com o dataset)
    classification = "Fit" if score_est >= 0.08 else "No Fit"
    score_percentage = int(min(max(score_est * 500, 5), 98)) # Mapeia score local para escala 0-100
    
    # Obtém as competências associadas à vaga (se existirem no banco)
    job_skills_str = job.get('skills', '')
    if job_skills_str:
        skills_ref = [s.strip() for s in job_skills_str.split(",") if s.strip()]
    else:
        # Fallback para palavras-chave gerais se não houver mapeamento de competências
        possible_kws = ["python", "machine learning", "sql", "aws", "docker", "excel", "spark", "power bi", "scikit-learn", "tensorflow", "react", "typescript", "javascript", "django", "fastapi"]
        job_desc_lower = job['description'].lower()
        skills_ref = [kw.capitalize() for kw in possible_kws if kw in job_desc_lower]
        if not skills_ref:
            skills_ref = ["Python", "SQL", "Excel"]
            
    skills_present = []
    skills_missing = []
    resume_lower = resume_text.lower()
    
    for skill in skills_ref:
        skill_clean = skill.strip()
        if not skill_clean:
            continue
        skill_lower = skill_clean.lower()
        # Busca simples de substring
        if skill_lower in resume_lower:
            skills_present.append(skill_clean)
        else:
            skills_missing.append(skill_clean)
            
    # Define as dicas de melhoria dinamicamente com base nas skills faltantes
    if skills_missing:
        tips = [
            f"Adicione palavras-chave como {', '.join(skills_missing[:3])} ao seu currículo se tiver experiência.",
            "Revise a descrição da vaga para mapear requisitos adicionais."
        ]
    else:
        tips = [
            "Parabéns! Seu currículo cobre as principais competências técnicas identificadas nesta vaga.",
            "Revise a descrição da vaga para mapear requisitos adicionais de soft skills ou certificações."
        ]
        
    return {
        "id": job["id"],
        "title": job["title"],
        "company": job["company"],
        "location": job["location"],
        "url": job["url"],
        "work_type": job.get("work_type", ""),
        "experience_level": job.get("experience_level", ""),
        "fit_classification": classification,
        "score_percentage": score_percentage,
        "skills_present": skills_present if skills_present else ["Habilidades gerais"],
        "skills_missing": skills_missing if skills_missing else ["Nenhuma habilidade crítica ausente detectada"],
        "justification": f"Esta é uma análise gerada localmente. O candidato possui afinidade aproximada de {score_percentage}% com as palavras-chave da vaga.",
        "improvement_tips": tips
    }

def match_resume_with_database(resume_text: str, limit: int = 5, filters: dict = None) -> List[Dict[str, Any]]:
    """Carrega as vagas do banco filtradas, faz a triagem inicial e gera a análise detalhada das top N."""
    conn = sqlite3.connect(config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Constrói query SQL dinâmica com base nos filtros aplicados, integrando competências oficiais
    query = """
        SELECT j.id, j.title, j.company, j.location, j.description, j.url, j.work_type, j.experience_level, s.skills
        FROM jobs j
        LEFT JOIN job_skills s ON j.id = s.job_id
        WHERE 1=1
    """
    params = []
    
    if filters:
        if filters.get("title"):
            query += " AND j.title LIKE ?"
            params.append(f"%{filters['title']}%")
        if filters.get("location"):
            query += " AND j.location LIKE ?"
            params.append(f"%{filters['location']}%")
        if filters.get("company"):
            query += " AND j.company LIKE ?"
            params.append(f"%{filters['company']}%")
        if filters.get("work_type"):
            query += " AND j.work_type LIKE ?"
            params.append(f"%{filters['work_type']}%")
        if filters.get("experience_level"):
            query += " AND j.experience_level LIKE ?"
            params.append(f"%{filters['experience_level']}%")
            
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        logger.warning("Nenhuma vaga correspondente encontrada no banco de dados para realizar o matching.")
        return []
        
    jobs = [dict(row) for row in rows]
    logger.info(f"Iniciando cruzamento com {len(jobs)} vagas filtradas no banco de dados.")
    
    # 1. Triagem rápida (calcula score de similaridade para todas)
    ranked_jobs = calculate_local_similarity(resume_text, jobs)
    
    # 2. Pega as top N vagas
    top_matches = ranked_jobs[:limit]
    
    # 3. Realiza a análise detalhada para cada uma das Top N
    detailed_analyses = []
    for job, score_est in top_matches:
        analysis = analyze_top_job(resume_text, job, score_est)
        detailed_analyses.append(analysis)
        
    return detailed_analyses
