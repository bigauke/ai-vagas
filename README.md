# AI Vagas 💼🤖

Uma solução inteligente e interativa de Machine Learning ponta a ponta para raspagem de vagas brasileiras do LinkedIn, cálculo de compatibilidade de currículos (Match Score) e análise fina de requisitos do candidato (Fit / No Fit) identificando as competências faltantes (Gaps).

---

## 🌟 Recursos e Funcionalidades

- **Coleta Automatizada (Scraper):** Extrai vagas de emprego em tempo real do LinkedIn no Brasil usando **Playwright** com técnicas anti-bloqueio (User-Agent rotativo e suporte a cookie de sessão `li_at`).
- **Triagem Semântica (Embeddings):** Compara o currículo com todas as vagas no banco de dados SQLite usando similaridade de cosseno de embeddings gerados pela **API do Gemini** (com **fallback local via TF-IDF**).
- **Análise Avançada de Fit (LLM):** Utiliza o modelo **Gemini 1.5 Flash** para classificar as Top-5 vagas em *Fit* ou *No Fit*, mapear as competências presentes e apontar as **skills faltantes** com dicas personalizadas de ajuste no currículo.
- **Painel Streamlit Premium:** Interface gráfica moderna baseada em **Glassmorphism** com suporte a upload de currículos em formato PDF e colagem direta de texto.

---

## 📂 Estrutura do Projeto

```text
ai-vagas/
├── .env                  # Chaves de API e configurações locais (ignorado no Git)
├── .gitignore            # Proteção contra commit de chaves e dados locais
├── requirements.txt      # Dependências em Python
├── README.md             # Documentação principal
├── data/
│   └── vagas.db          # Banco de dados local SQLite contendo as vagas
└── src/
    ├── app.py            # Interface gráfica Streamlit
    ├── config.py         # Configuração e variáveis do .env
    ├── matcher.py        # Motor de inteligência artificial e scoring
    ├── scraper.py        # Coletor de vagas públicas no LinkedIn
    ├── seed_data.py      # Carga de mock data com vagas brasileiras de teste
    └── verify_matcher.py  # Script de teste automatizado local do pipeline
```

---

## 🚀 Como Executar o Projeto

### 1. Pré-requisitos
Certifique-se de ter o Python 3.10+ instalado em seu sistema.

### 2. Clonar e Inicializar o Ambiente
Caso tenha clonado este repositório, configure o ambiente virtual:
```bash
# Criar ambiente virtual
python -m venv .venv

# Ativar ambiente virtual
# No Windows:
.venv\Scripts\activate
# No Linux/Mac:
source .venv/bin/activate

# Instalar dependências
pip install -r requirements.txt

# Instalar navegadores necessários para o Playwright
playwright install chromium
```

### 3. Configurar Variáveis de Ambiente
Crie um arquivo `.env` na raiz do projeto seguindo o modelo abaixo:
```ini
# Chave da API do Gemini (Obtenha em: https://aistudio.google.com/)
GEMINI_API_KEY=SUA_CHAVE_DO_GEMINI

# Cookie de Sessão do LinkedIn para Evitar Bloqueios (li_at) - Opcional
LINKEDIN_LI_AT=SEU_COOKIE_LI_AT

# Configurações do Scraper de Vagas
LINKEDIN_LOCATION=Brasil
DEFAULT_KEYWORDS=Machine Learning, Cientista de Dados, Desenvolvedor Python
MAX_JOBS_TO_SCRAPE=30
DELAY_BETWEEN_REQUESTS_SEC=2

# Banco de dados
DATABASE_PATH=data/vagas.db
```

*Nota: Se você não configurar a `GEMINI_API_KEY`, o aplicativo funcionará perfeitamente utilizando o motor local de fallback por TF-IDF.*

### 4. Popular o Banco com Vagas de Teste
Para rodar a ferramenta sem precisar raspar dados do LinkedIn imediatamente, carregue as vagas simuladas em português:
```bash
python src/seed_data.py
```

### 5. Iniciar o Painel Streamlit
Execute a aplicação web:
```bash
streamlit run src/app.py
```
Acesse no navegador: `http://localhost:8501`.

---

## 🧪 Executando Testes de Verificação
Para testar a inteligência de matching no terminal, execute:
```bash
python src/verify_matcher.py
```
O script simulará um perfil e exibirá o ranqueamento das vagas com os scores e as competências identificadas.
