import streamlit as st
import sqlite3
import pandas as pd
from pathlib import Path
from loguru import logger

# Importa as configurações do projeto e o motor de matching
import config
from scraper import scrape_jobs, init_db
from matcher import match_resume_with_database, extract_text_from_pdf, GEMINI_AVAILABLE

# Configuração da página do Streamlit
st.set_page_config(
    page_title="AI Vagas - LinkedIn Matcher & Scraper",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização CSS Premium (Glassmorphism e Dark Mode Harmônico)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    /* Configurações básicas de fonte do aplicativo */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
    }
    
    /* Degradê de título elegante */
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
    
    /* Card Glassmorphic Premium */
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
    
    /* Badges de Compatibilidade */
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
    
    /* Badges de Habilidades */
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
    
    /* Círculo do Score de Match */
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
    
    /* Estilos do Rodapé */
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
    init_db() # Garante que o BD está inicializado
    conn = sqlite3.connect(config.DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM jobs")
    total = cursor.fetchone()[0]
    conn.close()
    return total

# ---- SIDEBAR ----
with st.sidebar:
    st.image("https://img.icons8.com/nolan/96/artificial-intelligence.png", width=70)
    st.title("Configurações & Scraper")
    
    st.markdown("---")
    
    st.subheader("Coletor de Vagas")
    kw_input = st.text_input("Termos de busca (separados por vírgula)", value=", ".join(config.DEFAULT_KEYWORDS))
    loc_input = st.text_input("Localidade da vaga", value=config.LINKEDIN_LOCATION)
    limit_input = st.number_input("Máximo de vagas para coletar", min_value=1, max_value=100, value=config.MAX_JOBS_TO_SCRAPE)
    
    btn_scrape = st.button("🚀 Buscar Novas Vagas", use_container_width=True)
    
    if btn_scrape:
        with st.spinner("Conectando ao LinkedIn e coletando vagas..."):
            try:
                # Divide palavras-chave em uma lista
                keywords_list = [kw.strip() for kw in kw_input.split(",")]
                total_collected = 0
                for kw in keywords_list:
                    # Roda o scraper para cada keyword
                    total_collected += scrape_jobs(kw, loc_input, limit_input)
                
                st.success(f"Busca concluída! {total_collected} novas vagas foram salvas/atualizadas no banco.")
            except Exception as e:
                st.error(f"Erro ao executar scraper: {e}")
                
    st.markdown("---")
    
    # Exibe estatísticas da base de dados local
    total_jobs = get_total_jobs()
    st.metric("Vagas Cadastradas no Banco", total_jobs)
    
    if not GEMINI_AVAILABLE:
        st.warning("⚠️ Chave GEMINI_API_KEY ausente. Usando processamento local simples.")
    else:
        st.success("✅ Conectado com sucesso à API do Gemini.")

# ---- CORPO PRINCIPAL ----
st.markdown('<div class="hero-title">AI Vagas</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Calcule o score de compatibilidade do seu perfil com vagas brasileiras no LinkedIn</div>', unsafe_allow_html=True)

# Layout do Currículo
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📄 Seu Perfil / Currículo")
    upload_method = st.radio("Como deseja enviar seu currículo?", ["Upload de PDF", "Colar texto"])
    
    resume_text = ""
    
    if upload_method == "Upload de PDF":
        uploaded_file = st.file_uploader("Selecione o arquivo PDF do currículo", type=["pdf"])
        if uploaded_file is not None:
            # Salva arquivo temporário para leitura
            temp_pdf_path = Path("temp_resume.pdf")
            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            with st.spinner("Processando o PDF..."):
                try:
                    resume_text = extract_text_from_pdf(str(temp_pdf_path))
                    st.success("Currículo em PDF lido com sucesso!")
                    # Deleta o arquivo temporário
                    temp_pdf_path.unlink()
                except Exception as e:
                    st.error(f"Erro ao extrair texto do PDF: {e}")
    else:
        resume_text = st.text_area("Cole as informações do seu currículo aqui:", height=250, placeholder="Cole aqui seu resumo profissional, experiências, habilidades...")

with col2:
    st.subheader("💡 Como Funciona?")
    st.markdown("""
    1. **Coleta de Vagas**: O scraper busca vagas recentes no LinkedIn (Brasil) sem expor sua conta pessoal. Configure na barra lateral e clique em **Buscar Novas Vagas**.
    2. **Busca Semântica**: O sistema compara os vetores (embeddings) do seu currículo com todas as descrições de vagas no banco local para selecionar as melhores oportunidades.
    3. **Análise de Aderência**: A inteligência artificial estuda as vagas finalistas detalhadamente, avaliando se você tem **Fit** ou **No Fit**, apontando as competências que você já tem e as **skills faltantes** para cada cargo.
    """)
    
    st.info("💡 **Dica**: Garanta que seu currículo possua detalhes sobre projetos, linguagens e frameworks para obter um score mais acurado.")

st.markdown("---")

# Botão de Ação Principal
if resume_text:
    btn_match = st.button("🎯 Analisar Compatibilidade e Recomendar Vagas", type="primary", use_container_width=True)
    
    if btn_match:
        if get_total_jobs() == 0:
            st.warning("O banco de dados está vazio. Por favor, execute a busca de vagas na barra lateral antes de cruzar dados.")
        else:
            with st.spinner("Analisando vagas compatíveis no banco de dados e calculando scores..."):
                # Executa o algoritmo de matching
                matches = match_resume_with_database(resume_text, limit=5)
                
                if not matches:
                    st.error("Nenhuma vaga encontrada para comparação. Tente rodar o coletor de vagas primeiro.")
                else:
                    st.success("Análise concluída com sucesso! Aqui estão suas Top-5 recomendações:")
                    st.markdown("### 🏆 Top-5 Vagas Recomendadas")
                    
                    # Exibe cada vaga em formato de card premium
                    for idx, match in enumerate(matches):
                        # Configurações de cores de acordo com a classificação
                        if match['fit_classification'] == "Fit":
                            badge_html = '<span class="badge-fit">FIT ✅</span>'
                            score_color = "#2ed573"
                        else:
                            badge_html = '<span class="badge-nofit">NO FIT ❌</span>'
                            score_color = "#ff4757"
                            
                        score = match['score_percentage']
                        
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
                                        <p style="margin: 3px 0 0 0; color: #a4b0be;"><strong>{match['company']}</strong> — {match['location']}</p>
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
                                        Visualizar Vaga no LinkedIn 🔗
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
