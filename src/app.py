import streamlit as st
import sqlite3
import pandas as pd
from pathlib import Path
from loguru import logger

# Importa as configurações do projeto e o motor de matching/banco
import config
from database import import_job_skills_fast, import_postings_fast, init_db
from matcher import match_resume_with_database, extract_text_from_pdf

# Configuração da página do Streamlit
st.set_page_config(
    page_title="AI Vagas - Job Matcher Dashboard",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização CSS Premium (Glassmorphism e Dark Mode Harmônico)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
    }
    
    .hero-title {
        background: linear-gradient(135deg, #00e5ff 0%, #7c4dff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 5px;
    }
    
    .hero-subtitle {
        color: #a4b0be;
        font-size: 1.2rem;
        margin-bottom: 30px;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 18px;
        border: 1px solid rgba(255, 255, 255, 0.06);
        padding: 25px;
        margin-bottom: 22px;
        box-shadow: 0 10px 35px 0 rgba(0, 0, 0, 0.25);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    }
    
    .glass-card:hover {
        border-color: rgba(124, 77, 255, 0.35);
        box-shadow: 0 10px 35px 0 rgba(124, 77, 255, 0.08);
        transform: translateY(-3px);
    }
    
    .badge-fit {
        background: rgba(46, 213, 115, 0.12);
        color: #2ed573;
        border: 1.5px solid rgba(46, 213, 115, 0.35);
        padding: 6px 14px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
        box-shadow: 0 0 10px rgba(46, 213, 115, 0.05);
    }
    
    .badge-nofit {
        background: rgba(255, 71, 87, 0.12);
        color: #ff4757;
        border: 1.5px solid rgba(255, 71, 87, 0.35);
        padding: 6px 14px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
        box-shadow: 0 0 10px rgba(255, 71, 87, 0.05);
    }
    
    .skill-badge-present {
        background: rgba(0, 229, 255, 0.1);
        color: #00e5ff;
        border: 1px solid rgba(0, 229, 255, 0.25);
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.8rem;
        margin-right: 6px;
        margin-bottom: 8px;
        display: inline-block;
        font-weight: 500;
    }
    
    .skill-badge-missing {
        background: rgba(255, 165, 2, 0.1);
        color: #ffa502;
        border: 1px solid rgba(255, 165, 2, 0.25);
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.8rem;
        margin-right: 6px;
        margin-bottom: 8px;
        display: inline-block;
        font-weight: 500;
    }
    
    .score-circle {
        width: 60px;
        height: 60px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 1.25rem;
        border: 3px solid;
        margin-right: 15px;
    }
    
    .footer-text {
        text-align: center;
        color: #747d8c;
        margin-top: 50px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Função auxiliar para pegar o total de vagas salvas
def get_total_jobs():
    init_db()
    conn = sqlite3.connect(config.DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM jobs")
    total = cursor.fetchone()[0]
    conn.close()
    return total

# ---- SIDEBAR ----
with st.sidebar:
    st.image("https://img.icons8.com/nolan/96/artificial-intelligence.png", width=70)
    st.title("AI Vagas - Filtros")
    
    st.markdown("---")
    
    # 🔍 Seção de Filtros para triagem
    st.subheader("🔍 Filtros de Vagas")
    
    filter_title = st.text_input("Título do Cargo", placeholder="Ex: Machine Learning")
    filter_location = st.text_input("Localização", placeholder="Ex: Brasil ou Remoto")
    filter_company = st.text_input("Empresa", placeholder="Ex: Google")
    
    filter_work_type = st.selectbox(
        "Modelo de Trabalho",
        ["Todos", "Full-time", "Part-time", "Contract", "Temporary", "Internship", "Remote", "Hybrid", "On-site"]
    )
    
    filter_experience_level = st.selectbox(
        "Nível de Experiência",
        ["Todos", "Internship", "Entry level", "Associate", "Mid-Senior level", "Director", "Executive"]
    )
    
    st.markdown("---")
    
    # 📂 Importador de base Kaggle
    st.subheader("📂 Importador de Vagas")
    postings_path_input = st.text_input("Caminho de postings.csv", value="data/postings.csv")
    skills_path_input = st.text_input("Caminho de job_skills.csv", value="data/job_skills.csv")
    max_rows_input = st.number_input("Limite de vagas a importar", min_value=1000, max_value=200000, value=50000, step=1000)
    
    btn_import = st.button("📥 Importar Vagas e Skills", use_container_width=True)
    if btn_import:
        p_path = Path(postings_path_input)
        s_path = Path(skills_path_input)
        
        if not p_path.exists():
            st.error(f"postings.csv não localizado em: {postings_path_input}")
        elif not s_path.exists():
            st.error(f"job_skills.csv não localizado em: {skills_path_input}")
        else:
            status_text = st.empty()
            progress_bar = st.progress(0.0)
            
            try:
                status_text.text("Passo 1/2: Importando Competências (job_skills.csv)...")
                progress_bar.progress(0.2)
                
                result_skills = import_job_skills_fast(str(s_path))
                
                if result_skills["status"] == "success":
                    status_text.text(f"Competências importadas: {result_skills['imported']}. Passo 2/2: Importando Vagas (postings.csv)...")
                    progress_bar.progress(0.6)
                    
                    result_postings = import_postings_fast(str(p_path), max_rows=int(max_rows_input))
                    
                    if result_postings["status"] == "success":
                        progress_bar.progress(1.0)
                        status_text.text("Importação concluída com sucesso!")
                        st.success(f"Importação Concluída!\n- {result_postings['imported']} vagas importadas.\n- {result_skills['imported']} mapeamentos de competências carregados.")
                        st.rerun()
                    else:
                        st.error(result_postings["message"])
                else:
                    st.error(result_skills["message"])
            except Exception as e:
                st.error(f"Erro ao importar: {e}")
                    
    st.markdown("---")
    
    # Exibe estatísticas da base de dados local
    total_jobs = get_total_jobs()
    st.metric("Vagas no Banco SQLite", total_jobs)
    

# ---- CORPO PRINCIPAL ----
st.markdown('<div class="hero-title">AI Vagas</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Mapeamento de Aderência e Análise de Skills contra Vagas Reais (Kaggle/HuggingFace)</div>', unsafe_allow_html=True)

# Layout do Currículo
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📄 Seu Perfil / Currículo")
    upload_method = st.radio("Como deseja enviar seu currículo?", ["Upload de PDF", "Colar texto"])
    
    resume_text = ""
    
    if upload_method == "Upload de PDF":
        uploaded_file = st.file_uploader("Selecione o arquivo PDF do currículo", type=["pdf"])
        if uploaded_file is not None:
            temp_pdf_path = Path("temp_resume.pdf")
            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            with st.spinner("Processando o PDF..."):
                try:
                    resume_text = extract_text_from_pdf(str(temp_pdf_path))
                    st.success("Currículo em PDF lido com sucesso!")
                    temp_pdf_path.unlink()
                except Exception as e:
                    st.error(f"Erro ao extrair texto do PDF: {e}")
    else:
        resume_text = st.text_area("Cole as informações do seu currículo aqui:", height=250, placeholder="Cole aqui seu resumo profissional, experiências, habilidades...")

with col2:
    st.subheader("💡 Como Funciona?")
    st.markdown("""
    1. **Filtros de Vagas**: Defina restrições na barra lateral (como cargo, localização ou tipo de trabalho) para delimitar as vagas da base estática do Kaggle a serem avaliadas.
    2. **Busca Semântica**: O currículo é processado e cruzado (embeddings/TF-IDF) com as vagas selecionadas para ranquear as Top-5 compatíveis.
    3. **Análise de Fit (IA)**: A IA calcula a porcentagem de compatibilidade, classifica como **Fit** ou **No Fit**, mapeia suas competências presentes e aponta as **skills faltantes** necessárias para o cargo.
    """)
    
    st.info("💡 **Aviso**: O sistema utiliza a base estática do Kaggle. Se você ainda não importou o CSV principal de 124 mil vagas, o banco funcionará em modo demonstração com vagas reais de exemplo.")

st.markdown("---")

# Botão de Ação Principal
if resume_text:
    btn_match = st.button("🎯 Analisar Compatibilidade e Recomendar Vagas", type="primary", use_container_width=True)
    
    if btn_match:
        if get_total_jobs() == 0:
            st.warning("O banco de dados de vagas está vazio. Popule-o primeiro importando o CSV ou rodando o seed_data.")
        else:
            with st.spinner("Analisando vagas compatíveis no banco de dados e calculando scores..."):
                # Prepara filtros
                filters = {
                    "title": filter_title,
                    "location": filter_location,
                    "company": filter_company,
                    "work_type": filter_work_type if filter_work_type != "Todos" else None,
                    "experience_level": filter_experience_level if filter_experience_level != "Todos" else None
                }
                # Remove filtros nulos/vazios
                filters = {k: v for k, v in filters.items() if v}
                
                # Executa o algoritmo de matching
                matches = match_resume_with_database(resume_text, limit=5, filters=filters)
                
                if not matches:
                    st.warning("Nenhuma vaga correspondente encontrada com os filtros aplicados. Tente afrouxar os critérios na barra lateral.")
                else:
                    st.success("Análise concluída com sucesso! Aqui estão suas Top-5 recomendações:")
                    st.markdown("### 🏆 Top-5 Vagas Recomendadas")
                    
                    # Exibe cada vaga em formato de card premium
                    for idx, match in enumerate(matches):
                        if match['fit_classification'] == "Fit":
                            badge_html = '<span class="badge-fit">FIT ✅</span>'
                            score_color = "#2ed573"
                        else:
                            badge_html = '<span class="badge-nofit">NO FIT ❌</span>'
                            score_color = "#ff4757"
                            
                        score = match['score_percentage']
                        
                        # Extrai metadados do tipo de trabalho / experiência
                        wt = match.get("work_type", "")
                        el = match.get("experience_level", "")
                        meta_details = " — ".join([x for x in [wt, el] if x])
                        meta_text = f" | {meta_details}" if meta_details else ""
                        
                        # Estrutura HTML do card premium
                        card_html = f"""
                        <div class="glass-card">
                            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap;">
                                <div style="display: flex; align-items: center;">
                                    <div class="score-circle" style="border-color: {score_color}; color: {score_color};">
                                        {score}%
                                    </div>
                                    <div>
                                        <h3 style="margin: 0; font-size: 1.4rem;">{match['title']}</h3>
                                        <p style="margin: 3px 0 0 0; color: #a4b0be;"><strong>{match['company']}</strong> — {match['location']}{meta_text}</p>
                                    </div>
                                </div>
                                <div style="margin-top: 10px;">
                                    {badge_html}
                                </div>
                            </div>
                            
                            <div style="margin-top: 20px;">
                                <strong>Justificativa da IA:</strong>
                                <p style="color: #cbd5e0; font-size: 0.95rem; margin-top: 5px; line-height: 1.5;">{match['justification']}</p>
                            </div>
                            
                            <div style="margin-top: 15px; display: flex; flex-direction: column; gap: 10px;">
                                <div>
                                    <strong style="display: block; margin-bottom: 5px;">Habilidades Correspondentes:</strong>
                                    {"".join([f'<span class="skill-badge-present">{skill}</span>' for skill in match['skills_present']])}
                                </div>
                                <div style="margin-top: 5px;">
                                    <strong style="display: block; margin-bottom: 5px; color: #ffa502;">Skills Faltantes (Gaps):</strong>
                                    {"".join([f'<span class="skill-badge-missing">{skill}</span>' for skill in match['skills_missing']])}
                                </div>
                            </div>
                            
                            <div style="margin-top: 20px; border-top: 1px solid rgba(255, 255, 255, 0.05); padding-top: 15px;">
                                <strong>Dicas de Otimização do Currículo:</strong>
                                <ul style="color: #cbd5e0; font-size: 0.9rem; margin-top: 8px; padding-left: 20px;">
                                    {"".join([f'<li>{tip}</li>' for tip in match['improvement_tips']])}
                                </ul>
                            </div>
                            
                            <div style="margin-top: 15px; text-align: right;">
                                <a href="{match['url']}" target="_blank" style="text-decoration: none;">
                                    <button style="background: linear-gradient(135deg, #7c4dff 0%, #00e5ff 100%); color: white; border: none; padding: 8px 20px; border-radius: 8px; font-weight: 700; cursor: pointer; transition: opacity 0.2s;">
                                        Visualizar Vaga 🔗
                                    </button>
                                </a>
                            </div>
                        </div>
                        """
                        # Limpa espaços no início e fim de cada linha para evitar que o markdown interprete como bloco de código
                        card_html_clean = "\n".join([line.strip() for line in card_html.split("\n")])
                        st.markdown(card_html_clean, unsafe_allow_html=True)
else:
    st.info("📄 Insira as informações do seu currículo acima ou faça upload do PDF para ver as recomendações de vagas.")

st.markdown('<div class="footer-text">Desenvolvido com 💜 para facilitar conexões profissionais reais.</div>', unsafe_allow_html=True)
