"""
Master Execution Script: Cardiovascular and Metabolic Health Analytics
End-to-End Pipeline:
1. Dataset Synthesis & Validation (20 Clinical & Lifestyle Features, N=2,000)
2. Ben Fry's 7 Stages of Data Visualization & Statistical Modeling
3. Deep Multi-Layer Perceptron (MLP) Neural Network Training & Evaluation
"""

import sys
import os

from src.data_generator import generate_cardiovascular_dataset
from src.eda_analysis import run_complete_eda
from src.deep_learning_model import train_and_evaluate_model

def main():
    print("=" * 75)
    print(" CARDIOVASCULAR & METABOLIC HEALTH ANALYTICS: EDA & DEEP LEARNING ")
    print(" Course Code: U21ADP05 | Exploratory Data Analysis and Visualization ")
    print("=" * 75)
    
    # Step 1: Dataset Generation
    print("\n>>> STEP 1: DATASET GENERATION & AUDIT")
    data_path = 'data/cardiovascular_risk_data.csv'
    df = generate_cardiovascular_dataset(n_samples=2000, random_state=42)
    os.makedirs('data', exist_ok=True)
    df.to_csv(data_path, index=False)
    print(f"[SUCCESS] Dataset created at '{data_path}' with shape: {df.shape}")
    
    # Step 2 & 3: EDA & 7-Stage Visualization
    print("\n>>> STEP 2 & 3: EXPLORATORY DATA ANALYSIS & 7 STAGES OF VISUALIZATION")
    df_clean = run_complete_eda(data_path=data_path, figures_dir='figures')
    print(f"[SUCCESS] Cleaned and transformed dataset shape: {df_clean.shape}")
    
    # Step 4 & 5: Deep Learning Model & Evaluation
    print("\n>>> STEP 4 & 5: DEEP LEARNING MODEL TRAINING & PERFORMANCE EVALUATION")
    metrics = train_and_evaluate_model(figures_dir='figures')
    print(f"[SUCCESS] Deep MLP Pipeline Completed.")
    print(f"  Test Accuracy : {metrics['accuracy']:.2%}")
    print(f"  Test ROC-AUC  : {metrics['roc_auc']:.4f}")
    print(f"  Test F1-Score : {metrics['f1']:.4f}")
    
    print("\n" + "=" * 75)
    print(" PIPELINE COMPLETED SUCCESSFULLY! ALL FIGURES GENERATED IN 'figures/' ")
    print("=" * 75)

if __name__ == '__main__':
    main()
