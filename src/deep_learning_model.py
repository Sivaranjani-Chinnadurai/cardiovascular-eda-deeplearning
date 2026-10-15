"""
Deep Learning Model Architecture, Training, and Evaluation Pipeline
Implements:
1. Deep Multi-Layer Perceptron (PyTorch) with Batch Normalization and Dropout
2. Stratified Train / Validation / Test Splits (70% / 15% / 15%)
3. Basic Hyperparameter Tuning (Learning rate and architecture exploration)
4. Comprehensive Metric Evaluation (Loss, Accuracy, Confusion Matrix, ROC-AUC, PR-AUC)
5. Permutation Feature Importance Analysis
6. Publication-grade Result Visualizations (Figures 9 to 14)
"""

import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Visualization config
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

class CardiovascularMLP(nn.Module):
    """
    Deep Multi-Layer Perceptron (MLP) for Cardiovascular Event Risk Classification.
    Features:
    - 3 Hidden Dense Layers with non-linear ReLU activations
    - Batch Normalization for internal covariate shift mitigation
    - Dropout layers for regularization and overfitting prevention
    """
    def __init__(self, input_dim, hidden1=128, hidden2=64, hidden3=32, dropout_p=0.3):
        super(CardiovascularMLP, self).__init__()
        
        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            
            # Layer 2
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout_p * 0.8),
            
            # Layer 3
            nn.Linear(hidden2, hidden3),
            nn.ReLU(),
            nn.Dropout(dropout_p * 0.5),
            
            # Output Logits Layer
            nn.Linear(hidden3, 1)
        )
        
    def forward(self, x):
        return self.network(x)

def prepare_data(data_path='data/cardiovascular_processed_features.csv'):
    """
    Encodes categorical features, normalizes numerical attributes,
    and constructs stratified Train, Validation, and Test partitions.
    """
    df = pd.read_csv(data_path)
    
    # Drop non-feature columns
    cols_to_drop = ['patient_id', 'age_cohort', 'bp_stage']
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    
    # Ordinal mapping for stress
    stress_map = {'Low': 0, 'Moderate': 1, 'High': 2}
    df['chronic_stress_level'] = df['chronic_stress_level'].map(stress_map)
    
    # One-hot encode nominal variables
    categorical_cols = ['gender', 'smoking_status', 'alcohol_intake', 'physical_activity']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=float)
    
    target_col = 'cvd_risk_outcome'
    X = df.drop(columns=[target_col])
    y = df[target_col].values
    feature_names = list(X.columns)
    
    # Stratified Train (70%) and Temp (30%)
    X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    
    # Stratified Val (15%) and Test (15%)
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp_raw, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    
    # Standardize numerical features using parameters learned ONLY from train partition
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)
    
    print(f"Data Partitions Prepared:")
    print(f"  Training Set   : {X_train.shape[0]} samples, {X_train.shape[1]} features (Class balance: {np.mean(y_train):.1%})")
    print(f"  Validation Set : {X_val.shape[0]} samples, {X_val.shape[1]} features (Class balance: {np.mean(y_val):.1%})")
    print(f"  Test Set       : {X_test.shape[0]} samples, {X_test.shape[1]} features (Class balance: {np.mean(y_test):.1%})")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_names, scaler

def train_and_evaluate_model(figures_dir='figures'):
    os.makedirs(figures_dir, exist_ok=True)
    
    # 1. Prepare data
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names, scaler = prepare_data()
    
    # Convert numpy arrays to PyTorch Tensors
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).unsqueeze(1))
    val_dataset   = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val).unsqueeze(1))
    test_dataset  = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test).unsqueeze(1))
    
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    input_dim = X_train.shape[1]
    
    # -------------------------------------------------------------
    # HYPERPARAMETER TUNING EXPERIMENT (Step 4)
    # -------------------------------------------------------------
    print("\n[Step 4: Hyperparameter Tuning] Testing learning rates...")
    lr_candidates = [0.01, 0.001, 0.0002]
    tuning_curves = {}
    
    criterion = nn.BCEWithLogitsLoss()
    
    for lr in lr_candidates:
        test_model = CardiovascularMLP(input_dim=input_dim, dropout_p=0.3)
        optimizer = optim.Adam(test_model.parameters(), lr=lr, weight_decay=1e-4)
        val_losses = []
        
        for epoch in range(35):
            test_model.train()
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = test_model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
            test_model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    outputs = test_model(batch_x)
                    val_loss = criterion(outputs, batch_y)
                    total_val_loss += val_loss.item() * len(batch_y)
            val_losses.append(total_val_loss / len(val_dataset))
            
        tuning_curves[f"lr = {lr}"] = val_losses
        
    # Figure 14: Hyperparameter Tuning Convergence
    plt.figure(figsize=(9, 5))
    colors = ['#e74c3c', '#2980b9', '#27ae60']
    for idx, (label, losses) in enumerate(tuning_curves.items()):
        plt.plot(range(1, 36), losses, label=label, color=colors[idx], linewidth=2.2)
    plt.title("Hyperparameter Exploration: Validation Loss Trajectories Across Learning Rates", fontweight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("Validation BCE Loss")
    plt.legend(frameon=True)
    plt.tight_layout()
    fig14_path = os.path.join(figures_dir, 'fig14_hyperparameter_tuning.png')
    plt.savefig(fig14_path)
    plt.close()
    print(f"Saved: {fig14_path}")
    
    # -------------------------------------------------------------
    # OPTIMAL MODEL TRAINING (50 Epochs with Checkpointing)
    # -------------------------------------------------------------
    print("\n[Step 4 & 5: Deep Learning Model Training] Training optimal model with lr=0.001...")
    model = CardiovascularMLP(input_dim=input_dim, hidden1=128, hidden2=64, hidden3=32, dropout_p=0.3)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6)
    
    epochs = 50
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    best_weights = None
    
    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * len(batch_y)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct_train += (preds == batch_y).sum().item()
            total_train += len(batch_y)
            
        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train
        
        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                running_val_loss += loss.item() * len(batch_y)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                correct_val += (preds == batch_y).sum().item()
                total_val += len(batch_y)
                
        epoch_val_loss = running_val_loss / total_val
        epoch_val_acc = correct_val / total_val
        scheduler.step(epoch_val_loss)
        
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_weights = copy.deepcopy(model.state_dict())
            
        if epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs}] - Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.3f} | Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.3f}")
            
    # Load best model weights
    model.load_state_dict(best_weights)
    print("Optimal model weights restored from lowest validation loss checkpoint.")
    
    # -------------------------------------------------------------
    # MODEL EVALUATION ON UNSEEN TEST PARTITION
    # -------------------------------------------------------------
    print("\n[Step 5: Model Evaluation on Unseen Test Set]...")
    model.eval()
    test_logits = []
    test_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            logits = model(batch_x)
            test_logits.append(logits)
            test_targets.append(batch_y)
            
    test_logits = torch.cat(test_logits, dim=0).squeeze().numpy()
    test_targets = torch.cat(test_targets, dim=0).squeeze().numpy()
    test_probs = 1.0 / (1.0 + np.exp(-test_logits))
    test_preds = (test_probs >= 0.5).astype(int)
    
    # Calculate performance metrics
    acc = accuracy_score(test_targets, test_preds)
    prec = precision_score(test_targets, test_preds)
    rec = recall_score(test_targets, test_preds)
    f1 = f1_score(test_targets, test_preds)
    roc_auc = roc_auc_score(test_targets, test_probs)
    ap = average_precision_score(test_targets, test_probs)
    
    print(f"Test Set Performance Summary:")
    print(f"  Accuracy  : {acc:.4f} ({acc:.2%})")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f} (Sensitivity)")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {roc_auc:.4f}")
    print(f"  PR-AUC    : {ap:.4f}")
    print("\nClassification Report:\n", classification_report(test_targets, test_preds, target_names=['Low Risk', 'High Risk']))
    
    # -------------------------------------------------------------
    # RESULT VISUALIZATION 1: LOSS VS EPOCH (Figure 9)
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 5))
    epochs_range = range(1, epochs + 1)
    plt.plot(epochs_range, history['train_loss'], label='Training Loss', color='#2980b9', linewidth=2.4)
    plt.plot(epochs_range, history['val_loss'], label='Validation Loss', color='#e74c3c', linewidth=2.4, linestyle='--')
    plt.axvline(np.argmin(history['val_loss']) + 1, color='green', linestyle=':', label=f'Best Checkpoint (Epoch {np.argmin(history["val_loss"]) + 1})')
    plt.title("Model Convergence: Binary Cross-Entropy Loss vs. Epochs", fontsize=13, fontweight='bold')
    plt.xlabel("Training Epoch")
    plt.ylabel("Binary Cross-Entropy Loss")
    plt.legend(frameon=True)
    plt.tight_layout()
    fig9_path = os.path.join(figures_dir, 'fig09_model_loss_epoch.png')
    plt.savefig(fig9_path)
    plt.close()
    print(f"Saved: {fig9_path}")
    
    # -------------------------------------------------------------
    # RESULT VISUALIZATION 2: ACCURACY VS EPOCH (Figure 10)
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 5))
    plt.plot(epochs_range, [a * 100 for a in history['train_acc']], label='Training Accuracy', color='#27ae60', linewidth=2.4)
    plt.plot(epochs_range, [a * 100 for a in history['val_acc']], label='Validation Accuracy', color='#8e44ad', linewidth=2.4, linestyle='--')
    plt.title("Model Performance: Classification Accuracy vs. Epochs", fontsize=13, fontweight='bold')
    plt.xlabel("Training Epoch")
    plt.ylabel("Accuracy (%)")
    plt.ylim(50, 100)
    plt.legend(frameon=True, loc='lower right')
    plt.tight_layout()
    fig10_path = os.path.join(figures_dir, 'fig10_model_accuracy_epoch.png')
    plt.savefig(fig10_path)
    plt.close()
    print(f"Saved: {fig10_path}")
    
    # -------------------------------------------------------------
    # RESULT VISUALIZATION 3: CONFUSION MATRIX (Figure 11)
    # -------------------------------------------------------------
    cm = confusion_matrix(test_targets, test_preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, (ax_raw, ax_norm) = plt.subplots(1, 2, figsize=(13, 5))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_raw, cbar=False,
                annot_kws={'size': 14, 'weight': 'bold'}, linewidths=1.2)
    ax_raw.set_title("Test Confusion Matrix (Absolute Patient Counts)", fontweight='bold')
    ax_raw.set_xlabel("Predicted Diagnosis")
    ax_raw.set_ylabel("True Clinical Label")
    ax_raw.set_xticklabels(['Low Risk', 'High Risk'])
    ax_raw.set_yticklabels(['Low Risk', 'High Risk'])
    
    sns.heatmap(cm_norm * 100, annot=True, fmt='.1f', cmap='Greens', ax=ax_norm, cbar=True,
                annot_kws={'size': 14, 'weight': 'bold'}, linewidths=1.2)
    ax_norm.set_title("Normalized Confusion Matrix (% Recall / Specificity)", fontweight='bold')
    ax_norm.set_xlabel("Predicted Diagnosis")
    ax_norm.set_ylabel("True Clinical Label")
    ax_norm.set_xticklabels(['Low Risk', 'High Risk'])
    ax_norm.set_yticklabels(['Low Risk', 'High Risk'])
    
    plt.tight_layout()
    fig11_path = os.path.join(figures_dir, 'fig11_model_confusion_matrix.png')
    plt.savefig(fig11_path)
    plt.close()
    print(f"Saved: {fig11_path}")
    
    # -------------------------------------------------------------
    # RESULT VISUALIZATION 4: ROC & PRECISION-RECALL CURVES (Figure 12)
    # -------------------------------------------------------------
    fpr, tpr, _ = roc_curve(test_targets, test_probs)
    precision_vals, recall_vals, _ = precision_recall_curve(test_targets, test_probs)
    
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(13, 5.5))
    
    # ROC Curve
    ax_roc.plot(fpr, tpr, color='#c0392b', linewidth=2.8, label=f'Deep MLP (AUC = {roc_auc:.3f})')
    ax_roc.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1.5, label='Random Chance (AUC = 0.50)')
    ax_roc.set_title("Receiver Operating Characteristic (ROC) Curve", fontweight='bold')
    ax_roc.set_xlabel("False Positive Rate (1 - Specificity)")
    ax_roc.set_ylabel("True Positive Rate (Sensitivity / Recall)")
    ax_roc.legend(loc='lower right', frameon=True)
    
    # Precision-Recall Curve
    ax_pr.plot(recall_vals, precision_vals, color='#16a085', linewidth=2.8, label=f'Deep MLP (AP = {ap:.3f})')
    baseline_pr = np.mean(test_targets)
    ax_pr.axhline(baseline_pr, color='gray', linestyle='--', linewidth=1.5, label=f'Baseline Class Ratio ({baseline_pr:.2f})')
    ax_pr.set_title("Precision-Recall (PR) Curve", fontweight='bold')
    ax_pr.set_xlabel("Recall (Sensitivity)")
    ax_pr.set_ylabel("Precision (Positive Predictive Value)")
    ax_pr.legend(loc='lower left', frameon=True)
    
    plt.tight_layout()
    fig12_path = os.path.join(figures_dir, 'fig12_model_roc_pr_curve.png')
    plt.savefig(fig12_path)
    plt.close()
    print(f"Saved: {fig12_path}")
    
    # -------------------------------------------------------------
    # RESULT VISUALIZATION 5: PERMUTATION FEATURE IMPORTANCE (Figure 13)
    # -------------------------------------------------------------
    print("\n[Step 4: Permutation Feature Importance] Calculating predictor importance...")
    baseline_score = roc_auc_score(test_targets, test_probs)
    importances = []
    
    X_test_tensor = torch.FloatTensor(X_test)
    
    for f_idx in range(len(feature_names)):
        permuted_X = X_test.copy()
        np.random.shuffle(permuted_X[:, f_idx])
        with torch.no_grad():
            perm_logits = model(torch.FloatTensor(permuted_X)).squeeze().numpy()
            perm_probs = 1.0 / (1.0 + np.exp(-perm_logits))
            perm_score = roc_auc_score(test_targets, perm_probs)
            drop = baseline_score - perm_score
            importances.append((feature_names[f_idx], drop))
            
    imp_df = pd.DataFrame(importances, columns=['feature', 'importance_drop']).sort_values('importance_drop', ascending=True)
    
    plt.figure(figsize=(10, 7))
    bars = plt.barh(imp_df['feature'], imp_df['importance_drop'], color='#34495e', edgecolor='black', alpha=0.85)
    plt.title("Permutation Feature Importance (Drop in Test ROC-AUC Upon Feature Shuffling)", fontsize=13, fontweight='bold')
    plt.xlabel("Mean Drop in ROC-AUC Metric")
    plt.ylabel("Clinical / Engineered Feature")
    for bar in bars:
        width = bar.get_width()
        if width > 0:
            plt.annotate(f'{width:.3f}', xy=(width, bar.get_y() + bar.get_height()/2),
                         xytext=(5, 0), textcoords="offset points", ha='left', va='center', fontsize=9, fontweight='bold')
    plt.tight_layout()
    fig13_path = os.path.join(figures_dir, 'fig13_feature_importance.png')
    plt.savefig(fig13_path)
    plt.close()
    print(f"Saved: {fig13_path}")
    
    return {
        'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
        'roc_auc': roc_auc, 'avg_precision': ap,
        'cm': cm, 'history': history
    }

if __name__ == '__main__':
    train_and_evaluate_model()
