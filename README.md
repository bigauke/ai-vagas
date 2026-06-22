# AI Vagas 💼🤖

Uma solução inteligente e interativa de Machine Learning local e ponta a ponta para cálculo de compatibilidade de currículos (Match Score), classificação de adequação (*Fit* / *No Fit*) e análise de lacunas (*Gaps*) de competências técnicas.

O sistema opera de forma 100% off-line e local sobre bases de dados estáticas de referência em escala de mercado (datasets de 124 mil vagas e competências reais), oferecendo buscas rápidas pré-filtradas por banco e cruzamento semântico local de currículos via TF-IDF calibrado.

---

## 🌟 Recursos e Funcionalidades

- **Matching em Escala com Dataset Real**: O motor busca vagas reais pré-filtradas da base de dados **LinkedIn Job Postings 2023-2024 (Kaggle)**, cruzando o perfil do candidato com milhares de descrições estruturadas.
- **Mapeamento de Competências Reais**: Integração com o **Job Skill Set Dataset (Kaggle)**, vinculando as competências oficiais requeridas por cargo direto ao ID de cada vaga via banco de dados.
- **Calibração do Classificador**: Limiar de aderência calibrado localmente com o dataset **Resume-JD-Match (HuggingFace)** para otimização da precisão e sensibilidade da classificação local.
- **Triagem Textual e Frequência Local (TF-IDF)**: Compara o currículo com as vagas no banco de dados SQLite usando similaridade de cosseno de representações de frequência de termos (TF-IDF) locais de alta performance.
- **Identificação Dinâmica de Gaps (Skills Faltantes)**: O matcher analisa quais das competências exigidas pela vaga estão presentes ou ausentes no currículo do usuário, gerando dicas dinâmicas para otimização.
- **Interface Streamlit Premium**: Painel visual moderno com visual Glassmorphism, suporte a upload de currículo em PDF, preenchimento direto e barra de progresso interativa para importação rápida de datasets.

---

## 📂 Estrutura do Projeto

```text
ai-vagas/
├── .env                  # Configurações locais de banco (ignorado no Git)
├── .gitignore            # Proteção contra commit de chaves e dados locais
├── requirements.txt      # Dependências em Python (sem playwright e google-generativeai)
├── README.md             # Documentação principal
├── data/
│   ├── vagas.db          # Banco de dados local SQLite contendo as vagas e competências
│   ├── postings.csv      # Dataset de vagas do LinkedIn (Kaggle - 516MB, ignorado no Git)
│   ├── job_skills.csv    # Dataset de mapeamento de competências (Kaggle - 672MB, ignorado no Git)
│   ├── train-00000-of-00001.parquet  # Parquet de treino do Resume-JD-Match (HuggingFace, ignorado no Git)
│   └── test-00000-of-00001.parquet   # Parquet de teste do Resume-JD-Match (HuggingFace, ignorado no Git)
└── src/
    ├── app.py            # Interface gráfica Streamlit Premium com importador interativo
    ├── config.py         # Configuração e variáveis do .env
    ├── database.py       # Gerenciador do SQLite com importador por streaming de alto desempenho
    ├── matcher.py        # Motor de matching local por TF-IDF com join de competências
    ├── seed_data.py      # Carga de mock data com vagas estruturadas e competências associadas
    ├── calibrate.py      # Estudo de calibração de limiares offline usando os parquets locais (Treino e Teste)
    └── verify_matcher.py  # Script de teste funcional rápido do pipeline
```

---

## 🚀 Como Executar o Projeto

### 1. Pré-requisitos
Certifique-se de ter o Python 3.10+ instalado em seu sistema.

### 2. Clonar e Inicializar o Ambiente
Configure o ambiente virtual local:
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
```

> **Observação para usuários macOS:**
> Em algumas instalações do macOS, o comando `python` não está disponível por padrão. Nesse caso, utilize:
>
> ```bash
> python3 -m venv .venv
> source .venv/bin/activate
> python -m pip install --upgrade pip
> python -m pip install -r requirements.txt
> ```
>
> Para verificar se o ambiente virtual foi ativado corretamente:
>
> ```bash
> which python
> which pip
> ```
>
> Ambos os caminhos devem apontar para o diretório `.venv`.


### 3. Configurar Variáveis de Ambiente
Crie um arquivo `.env` na raiz do projeto:
```ini
# Banco de dados
DATABASE_PATH=data/vagas.db
```

### 4. Popular o Banco com Dados do Kaggle
1. Baixe os datasets do Kaggle: **LinkedIn Job Postings 2023-2024** e **Job Skill Set Dataset**.
2. Salve os arquivos `postings.csv` e `job_skills.csv` na pasta `data/`.
3. Inicie o Streamlit (passo 5) e utilize o painel lateral **"Importador de Vagas"** para carregar os registros no banco de dados SQLite local, ou rode o seed de teste rápido:
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

## 🧪 Calibração e Testes

### 1. Calibrar Limiar de Similaridade
Para recalcular a tabela comparativa de precisão/recall de thresholds de TF-IDF e redefinir o limiar ótimo do matcher offline usando os parquets de treino e teste, execute:
```bash
python src/calibrate.py
```

### 2. Testar Pipeline de Matching
Para rodar uma verificação ponta a ponta do cruzamento de currículos com vagas e competências locais no terminal, execute:
```bash
python src/verify_matcher.py
```
