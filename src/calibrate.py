import os
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from loguru import logger

# Importa similaridade
from matcher import cosine_similarity

def load_and_parse_parquet(file_path: Path, sample_size: int = 500) -> tuple:
    """Carrega o parquet local, realiza amostragem balanceada e faz o parsing de JD e Currículo."""
    import pandas as pd
    logger.info(f"Carregando dataset de {file_path}...")
    df = pd.read_parquet(file_path)
    logger.info(f"Carregado com {len(df)} registros.")
    
    # Amostragem balanceada de classes
    df_nofit = df[df['label'] == 'No Fit']
    df_fit = df[df['label'].isin(['Potential Fit', 'Good Fit'])]
    
    sample_half = sample_size // 2
    try:
        df_nofit_sample = df_nofit.sample(n=min(sample_half, len(df_nofit)), random_state=42)
        df_fit_sample = df_fit.sample(n=min(sample_half, len(df_fit)), random_state=42)
        sample_df = pd.concat([df_nofit_sample, df_fit_sample]).sample(frac=1.0, random_state=42)
    except Exception as sample_err:
        logger.warning(f"Erro na amostragem balanceada: {sample_err}. Usando amostragem sequencial.")
        sample_df = df.iloc[:sample_size]
        
    resumes = []
    job_descriptions = []
    true_labels = []
    
    for idx, row in sample_df.iterrows():
        text = str(row['text'])
        # Parser do formato: For the given job description <<[JD]>> the resume: <<[RESUME]>>. The result is, [Label]
        jd_start = text.find("For the given job description <<")
        if jd_start == -1:
            continue
        jd_start += len("For the given job description <<")
        
        jd_end = text.find(">> the resume: <<", jd_start)
        if jd_end == -1:
            continue
        jd = text[jd_start:jd_end]
        
        resume_start = jd_end + len(">> the resume: <<")
        resume_end = text.find(">>. The result is, ", resume_start)
        if resume_end == -1:
            resume_end = text.rfind(">>")
            if resume_end == -1 or resume_end <= resume_start:
                continue
        resume = text[resume_start:resume_end]
        
        label_val = str(row['label']).strip().lower()
        is_fit = label_val in ['1', 'true', 'fit', 'match', 'yes', 'sim', 'compatible', 'potential fit', 'good fit']
        
        resumes.append(resume)
        job_descriptions.append(jd)
        true_labels.append(is_fit)
        
    logger.info(f"Parseados com sucesso {len(resumes)} registros de {file_path.name}.")
    return resumes, job_descriptions, true_labels

def calculate_pair_similarities(resumes: list, jds: list) -> list:
    """Calcula similaridades TF-IDF locais para os pares."""
    similarities = []
    for i in range(len(resumes)):
        vectorizer = TfidfVectorizer(stop_words='english')
        try:
            tfidf = vectorizer.fit_transform([resumes[i], jds[i]])
            v1 = tfidf[0].toarray()[0]
            v2 = tfidf[1].toarray()[0]
            sim = cosine_similarity(v1, v2)
            similarities.append(sim)
        except Exception:
            similarities.append(0.0)
    return similarities

def inspect_and_calibrate(sample_size: int = 500):
    logger.info("Iniciando calibração e teste do Matcher com datasets locais...")
    
    train_path = Path("data/train-00000-of-00001.parquet")
    test_path = Path("data/test-00000-of-00001.parquet")
    
    if not train_path.exists() or not test_path.exists():
        logger.error("Dataset train ou test não localizado na pasta 'data/'. Certifique-se de baixá-los.")
        return
        
    # 1. Carrega e parseia os dados de treino
    logger.info("--- PASSO 1: Carregando dados de TREINO para calibração ---")
    train_resumes, train_jds, train_labels = load_and_parse_parquet(train_path, sample_size)
    train_sims = calculate_pair_similarities(train_resumes, train_jds)
    
    # 2. Varre os thresholds no TREINO para encontrar o melhor
    thresholds = np.linspace(0.05, 0.90, 86)
    best_threshold = 0.08
    best_f1 = 0.0
    best_acc = 0.0
    
    for thresh in thresholds:
        preds = [sim >= thresh for sim in train_sims]
        acc = accuracy_score(train_labels, preds)
        f1 = f1_score(train_labels, preds, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
            best_acc = acc
            
    logger.info("-" * 60)
    logger.info("Resultados de calibração em TREINO:")
    logger.info(f"{'Threshold':<10} | {'Acurácia':<10} | {'F1-Score':<10} | {'Precisão Fit':<12} | {'Recall Fit':<10}")
    logger.info("-" * 60)
    for t in [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25]:
        preds_t = [sim >= t for sim in train_sims]
        acc_t = accuracy_score(train_labels, preds_t)
        f1_t = f1_score(train_labels, preds_t, zero_division=0)
        prec_t = precision_score(train_labels, preds_t, zero_division=0)
        rec_t = recall_score(train_labels, preds_t, zero_division=0)
        logger.info(f"{t:<10.3f} | {acc_t*100:<9.1f}% | {f1_t:<10.3f} | {prec_t:<12.3f} | {rec_t:<10.3f}")
    logger.info("-" * 60)
    
    logger.info(f"Limiar Ideal Encontrado em Treino: {best_threshold:.3f}")
    logger.info(f"F1-Score correspondente em Treino: {best_f1:.3f}")
    logger.info("-" * 60)
    
    # 3. Avalia o limiar ótimo no conjunto de TESTE
    logger.info("\n--- PASSO 2: Carregando dados de TESTE para validação ---")
    test_resumes, test_jds, test_labels = load_and_parse_parquet(test_path, sample_size)
    test_sims = calculate_pair_similarities(test_resumes, test_jds)
    
    test_preds = [sim >= best_threshold for sim in test_sims]
    test_acc = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds, zero_division=0)
    
    logger.info("-" * 60)
    logger.info(f"AVALIAÇÃO FINAL NO CONJUNTO DE TESTE (Limiar: {best_threshold:.3f})")
    logger.info(f"Acurácia no Teste: {test_acc * 100:.1f}%")
    logger.info(f"F1-Score no Teste: {test_f1:.3f}")
    logger.info("-" * 60)
    
    report = classification_report(test_labels, test_preds, target_names=["No Fit (0)", "Fit (1)"], zero_division=0)
    print("\nRelatório de Classificação Final no Conjunto de Teste:")
    print(report)
    
    return float(best_threshold)

if __name__ == "__main__":
    inspect_and_calibrate(500)
