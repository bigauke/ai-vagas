import os
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, f1_score
from loguru import logger

# Importa similaridade
from matcher import cosine_similarity

def inspect_and_calibrate(sample_size: int = 500):
    logger.info("Iniciando calibração do Matcher...")
    
    # 1. Carrega o dataset de treino local se existir
    local_parquet_path = Path("data/train-00000-of-00001.parquet")
    use_local_parquet = False
    
    if local_parquet_path.exists():
        try:
            import pandas as pd
            logger.info(f"Carregando dataset local do parquet: {local_parquet_path}...")
            df = pd.read_parquet(local_parquet_path)
            logger.info(f"Parquet local carregado. Contém {len(df)} registros.")
            use_local_parquet = True
        except Exception as e:
            logger.warning(f"Falha ao carregar parquet local: {e}. Tentando via HuggingFace...")
            
    if not use_local_parquet:
        try:
            from datasets import load_dataset
            logger.info("Carregando dataset 'facehuggerapoorv/resume-jd-match' do HuggingFace...")
            dataset = load_dataset("facehuggerapoorv/resume-jd-match")
            split_name = list(dataset.keys())[0]
            data = dataset[split_name]
            logger.info(f"Dataset carregado via HuggingFace. Contém {len(data)} registros.")
        except Exception as e:
            logger.error(f"Erro ao carregar o dataset do HuggingFace: {e}")
            return
            
    resumes = []
    job_descriptions = []
    true_labels = []
    
    if use_local_parquet:
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
            
        eval_limit = len(resumes)
        logger.info(f"Parseado com sucesso {eval_limit} registros locais para calibração.")
    else:
        # Usa mapeamento do HuggingFace
        sample_row = data[0]
        resume_col = None
        jd_col = None
        label_col = None
        
        for col in sample_row.keys():
            col_lower = col.lower()
            if 'resume' in col_lower or 'curriculo' in col_lower or 'cv' in col_lower:
                resume_col = col
            elif 'jd' in col_lower or 'description' in col_lower or 'vaga' in col_lower:
                jd_col = col
            elif 'label' in col_lower or 'match' in col_lower or 'fit' in col_lower or 'class' in col_lower:
                label_col = col
                
        if not resume_col: resume_col = list(sample_row.keys())[0]
        if not jd_col: jd_col = list(sample_row.keys())[1]
        if not label_col: label_col = list(sample_row.keys())[-1]
        
        eval_limit = min(sample_size, len(data))
        eval_data = data.select(range(eval_limit))
        
        for row in eval_data:
            resumes.append(str(row[resume_col]))
            job_descriptions.append(str(row[jd_col]))
            val = str(row[label_col]).strip().lower()
            is_fit = val in ['1', 'true', 'fit', 'match', 'yes', 'sim', 'compatible', 'potential fit', 'good fit']
            true_labels.append(is_fit)
            
    if not resumes:
        logger.error("Nenhum registro carregado ou parseado com sucesso.")
        return
        
    logger.info(f"Calculando similaridade local (TF-IDF) para {eval_limit} pares de validação...")
    
    # 3. Calcula similaridade usando TF-IDF local
    similarities = []
    for i in range(eval_limit):
        vectorizer = TfidfVectorizer(stop_words='english')
        try:
            tfidf = vectorizer.fit_transform([resumes[i], job_descriptions[i]])
            v1 = tfidf[0].toarray()[0]
            v2 = tfidf[1].toarray()[0]
            sim = cosine_similarity(v1, v2)
            similarities.append(sim)
        except Exception:
            similarities.append(0.0)
            
    # 4. Varre os thresholds para encontrar o melhor
    thresholds = np.linspace(0.05, 0.90, 86)
    best_threshold = 0.20
    best_f1 = 0.0
    best_acc = 0.0
    
    for thresh in thresholds:
        preds = [sim >= thresh for sim in similarities]
        acc = accuracy_score(true_labels, preds)
        f1 = f1_score(true_labels, preds, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
            best_acc = acc
            
    logger.info("-" * 60)
    logger.info("Comparação de Limiares:")
    logger.info(f"{'Threshold':<10} | {'Acurácia':<10} | {'F1-Score':<10} | {'Precisão Fit':<12} | {'Recall Fit':<10}")
    logger.info("-" * 60)
    for t in [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25]:
        preds_t = [sim >= t for sim in similarities]
        acc_t = accuracy_score(true_labels, preds_t)
        f1_t = f1_score(true_labels, preds_t, zero_division=0)
        
        # Calcula precisão e recall para a classe Fit (True)
        from sklearn.metrics import precision_score, recall_score
        prec_t = precision_score(true_labels, preds_t, zero_division=0)
        rec_t = recall_score(true_labels, preds_t, zero_division=0)
        
        logger.info(f"{t:<10.3f} | {acc_t*100:<9.1f}% | {f1_t:<10.3f} | {prec_t:<12.3f} | {rec_t:<10.3f}")
        
    logger.info("-" * 60)
    logger.info(f"CALIBRAÇÃO COMPLETA (Tamanho da amostra: {eval_limit})")
    logger.info(f"Threshold Ideal de Similaridade: {best_threshold:.3f}")
    logger.info(f"Acurácia obtida: {best_acc * 100:.1f}%")
    logger.info(f"F1-Score obtido: {best_f1:.3f}")
    logger.info("-" * 60)
    
    # Exibe relatório final com o melhor limiar
    final_preds = [sim >= best_threshold for sim in similarities]
    report = classification_report(true_labels, final_preds, target_names=["No Fit (0)", "Fit (1)"], zero_division=0)
    print("\nRelatório de Classificação no Limiar Ideal:")
    print(report)
    
    return float(best_threshold)

if __name__ == "__main__":
    inspect_and_calibrate(500)
