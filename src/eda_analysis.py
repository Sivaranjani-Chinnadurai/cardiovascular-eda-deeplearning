"""
Exploratory Data Analysis and Visualization Pipeline
Implements Ben Fry's 7 Stages of Data Visualization:
1. Acquire, 2. Parse, 3. Filter, 4. Mine, 5. Represent, 6. Refine, 7. Interact
Generates publication-quality figures for univariate, bivariate, multivariate,
and theoretical probability distribution analyses.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11

def run_complete_eda(data_path='data/cardiovascular_risk_data.csv', figures_dir='figures'):
    os.makedirs(figures_dir, exist_ok=True)
    
    # -------------------------------------------------------------
    # STAGE 1: ACQUIRE & STAGE 2: PARSE
    # -------------------------------------------------------------
    print("[Stage 1 & 2: Acquire & Parse] Ingesting dataset...")
    df_raw = pd.read_csv(data_path)
    raw_shape = df_raw.shape
    print(f"Raw Dataset Shape: {raw_shape[0]} rows, {raw_shape[1]} columns")
    
    # Audit missingness and duplicates
    initial_nulls = df_raw.isnull().sum()
    initial_dups = df_raw.duplicated().sum()
    print(f"Initial Duplicates Detected: {initial_dups}")
    print("Initial Missing Values:\n", initial_nulls[initial_nulls > 0])
    
    # Figure 1: Data Sparsity and Quality Audit (Stage 1-3)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # Missing values bar
    null_cols = initial_nulls[initial_nulls > 0]
    bars = ax1.bar(null_cols.index, null_cols.values, color=['#e74c3c', '#e67e22', '#f1c40f'], edgecolor='black', alpha=0.85)
    ax1.set_title("Pre-Filtering: Missing Values per Feature (Stage 1-3: Acquire & Filter)", fontweight='bold')
    ax1.set_ylabel("Missing Count (Rows)")
    ax1.set_xlabel("Biomarker / Feature")
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{int(height)} ({height/len(df_raw):.1%})',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 4), textcoords="offset points",
                     ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax1.set_ylim(0, max(null_cols.values) * 1.25)
    
    # Duplicate & Row Integrity Status
    status_counts = [len(df_raw) - initial_dups, initial_dups]
    ax2.pie(status_counts, labels=[f'Unique Records\n({status_counts[0]})', f'Duplicate Records\n({status_counts[1]})'],
            colors=['#2ecc71', '#e74c3c'], autopct='%1.1f%%', startangle=140, explode=(0, 0.15),
            wedgeprops={'edgecolor': 'black', 'linewidth': 1.2})
    ax2.set_title("Duplicate Row Audit (Filter Stage)", fontweight='bold')
    
    plt.tight_layout()
    fig1_path = os.path.join(figures_dir, 'fig01_missing_and_duplicates.png')
    plt.savefig(fig1_path)
    plt.close()
    print(f"Saved: {fig1_path}")
    
    # -------------------------------------------------------------
    # STAGE 3: FILTER (Data Cleaning, Imputation & Outlier Treatment)
    # -------------------------------------------------------------
    print("[Stage 3: Filter] Performing duplicate removal, imputation, and outlier capping...")
    df_clean = df_raw.drop_duplicates().copy()
    
    # Impute missing values (median for continuous skewed, mode for discrete)
    for col in ['fasting_blood_sugar', 'triglycerides', 'sleep_hours']:
        median_val = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(median_val)
        
    # Outlier Detection & Winsorization via IQR
    numeric_cols = ['systolic_bp', 'diastolic_bp', 'bmi', 'fasting_blood_sugar', 
                    'total_cholesterol', 'hdl_cholesterol', 'ldl_cholesterol', 'triglycerides']
    
    outlier_stats = {}
    for col in numeric_cols:
        q25, q75 = df_clean[col].quantile(0.25), df_clean[col].quantile(0.75)
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        outliers_detected = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
        outlier_stats[col] = outliers_detected
        # Winsorize / Cap at 1st and 99th percentiles to retain epidemiological signal without extreme skew
        p01, p99 = df_clean[col].quantile(0.01), df_clean[col].quantile(0.99)
        df_clean[col] = df_clean[col].clip(p01, p99)
        
    print("Outliers identified via IQR Tukey fences:", outlier_stats)
    
    # -------------------------------------------------------------
    # FEATURE ENGINEERING
    # -------------------------------------------------------------
    df_clean['pulse_pressure'] = df_clean['systolic_bp'] - df_clean['diastolic_bp']
    df_clean['cholesterol_ratio'] = df_clean['total_cholesterol'] / df_clean['hdl_cholesterol']
    df_clean['map_bp'] = df_clean['diastolic_bp'] + (df_clean['pulse_pressure'] / 3.0)
    
    # -------------------------------------------------------------
    # STAGE 4 & 5: MINE & REPRESENT (Univariate Continuous Analysis)
    # -------------------------------------------------------------
    print("[Stage 4 & 5: Mine & Represent] Univariate statistical analysis...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    features_to_plot = [
        ('systolic_bp', 'Systolic Blood Pressure (mmHg)', '#2980b9'),
        ('bmi', 'Body Mass Index (kg/m²)', '#27ae60'),
        ('fasting_blood_sugar', 'Fasting Blood Glucose (mg/dL)', '#8e44ad'),
        ('triglycerides', 'Serum Triglycerides (mg/dL)', '#d35400')
    ]
    
    summary_stats_dict = {}
    for idx, (col, label, color) in enumerate(features_to_plot):
        ax = axes[idx // 2, idx % 2]
        data = df_clean[col]
        mean_val = data.mean()
        median_val = data.median()
        skew_val = stats.skew(data)
        kurt_val = stats.kurtosis(data)
        summary_stats_dict[col] = {'mean': mean_val, 'median': median_val, 'skew': skew_val, 'kurt': kurt_val}
        
        sns.histplot(data, kde=True, ax=ax, color=color, stat="density", bins=30, alpha=0.55, edgecolor='black')
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
        ax.axvline(median_val, color='black', linestyle=':', linewidth=2, label=f'Median: {median_val:.1f}')
        
        ax.set_title(f"Distribution of {label}\nSkewness: {skew_val:.2f} | Kurtosis: {kurt_val:.2f}", fontweight='bold')
        ax.set_xlabel(label)
        ax.set_ylabel("Probability Density")
        ax.legend(loc='upper right', frameon=True)
        
    plt.tight_layout()
    fig2_path = os.path.join(figures_dir, 'fig02_univariate_continuous_distributions.png')
    plt.savefig(fig2_path)
    plt.close()
    print(f"Saved: {fig2_path}")
    
    # -------------------------------------------------------------
    # CO4: THEORETICAL PROBABILITY DISTRIBUTION FITTING & Q-Q PLOTS
    # -------------------------------------------------------------
    print("[CO4 Analysis] Fitting Continuous Theoretical Probability Distributions (Normal & Log-Normal)...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Systolic BP vs Theoretical Normal
    ax_hist1 = axes[0, 0]
    bp_data = df_clean['systolic_bp']
    mu, std = stats.norm.fit(bp_data)
    ks_stat_norm, ks_p_norm = stats.kstest(bp_data, 'norm', args=(mu, std))
    
    x_axis = np.linspace(bp_data.min(), bp_data.max(), 200)
    sns.histplot(bp_data, stat='density', ax=ax_hist1, color='#3498db', alpha=0.5, bins=30, edgecolor='black', label='Empirical Density')
    ax_hist1.plot(x_axis, stats.norm.pdf(x_axis, mu, std), 'r-', linewidth=2.5, label=f'Fitted Normal(μ={mu:.1f}, σ={std:.1f})')
    ax_hist1.set_title(f"Systolic BP vs Theoretical Normal Fit\nKolmogorov-Smirnov p-value: {ks_p_norm:.4f}", fontweight='bold')
    ax_hist1.set_xlabel("Systolic Blood Pressure (mmHg)")
    ax_hist1.set_ylabel("Density")
    ax_hist1.legend()
    
    # Q-Q Plot for Systolic BP
    ax_qq1 = axes[0, 1]
    stats.probplot(bp_data, dist="norm", plot=ax_qq1)
    ax_qq1.get_lines()[0].set_markerfacecolor('#2980b9')
    ax_qq1.get_lines()[0].set_markeredgecolor('none')
    ax_qq1.get_lines()[0].set_markersize(5.0)
    ax_qq1.get_lines()[1].set_color('red')
    ax_qq1.get_lines()[1].set_linewidth(2.0)
    ax_qq1.set_title("Normal Q-Q Plot: Systolic Blood Pressure", fontweight='bold')
    
    # 2. Serum Triglycerides vs Theoretical Log-Normal Fit
    ax_hist2 = axes[1, 0]
    tg_data = df_clean['triglycerides']
    shape, loc, scale = stats.lognorm.fit(tg_data, floc=0)
    ks_stat_log, ks_p_log = stats.kstest(tg_data, 'lognorm', args=(shape, loc, scale))
    
    x_axis_tg = np.linspace(tg_data.min(), tg_data.max(), 200)
    sns.histplot(tg_data, stat='density', ax=ax_hist2, color='#e67e22', alpha=0.5, bins=30, edgecolor='black', label='Empirical Density')
    ax_hist2.plot(x_axis_tg, stats.lognorm.pdf(x_axis_tg, shape, loc, scale), 'r-', linewidth=2.5,
                  label=f'Fitted Log-Normal(s={shape:.2f}, scale={scale:.1f})')
    ax_hist2.set_title(f"Serum Triglycerides vs Log-Normal Fit\nKolmogorov-Smirnov p-value: {ks_p_log:.4f}", fontweight='bold')
    ax_hist2.set_xlabel("Serum Triglycerides (mg/dL)")
    ax_hist2.set_ylabel("Density")
    ax_hist2.legend()
    
    # Q-Q Plot for Triglycerides against Log-Normal
    ax_qq2 = axes[1, 1]
    stats.probplot(np.log(tg_data), dist="norm", plot=ax_qq2)
    ax_qq2.get_lines()[0].set_markerfacecolor('#d35400')
    ax_qq2.get_lines()[0].set_markeredgecolor('none')
    ax_qq2.get_lines()[0].set_markersize(5.0)
    ax_qq2.get_lines()[1].set_color('red')
    ax_qq2.get_lines()[1].set_linewidth(2.0)
    ax_qq2.set_title("Log-Transformed Normal Q-Q Plot: Triglycerides", fontweight='bold')
    
    plt.tight_layout()
    fig3_path = os.path.join(figures_dir, 'fig03_theoretical_distribution_fitting_qq.png')
    plt.savefig(fig3_path)
    plt.close()
    print(f"Saved: {fig3_path}")
    
    # -------------------------------------------------------------
    # CO4: DISCRETE PROBABILITY DISTRIBUTIONS & PMF
    # -------------------------------------------------------------
    print("[CO4 Analysis] Discrete probability distribution exploration...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Diet Quality Score (Discrete Uniform-like / Binomial PMF)
    ax_diet = axes[0, 0]
    diet_counts = df_clean['diet_quality_score'].value_counts(normalize=True).sort_index()
    bars_diet = ax_diet.bar(diet_counts.index, diet_counts.values, color='#16a085', edgecolor='black', alpha=0.85)
    ax_diet.set_title("Probability Mass Function (PMF): Diet Quality Score (1-10)", fontweight='bold')
    ax_diet.set_xlabel("Discrete Diet Quality Rating")
    ax_diet.set_ylabel("Probability P(X = x)")
    ax_diet.set_xticks(range(1, 11))
    for bar in bars_diet:
        height = bar.get_height()
        ax_diet.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                         xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
                         
    # 2. Sleep Hours per Day (Discrete/Continuous Poisson/Gaussian-like)
    ax_sleep = axes[0, 1]
    sleep_counts = df_clean['sleep_hours'].value_counts(normalize=True).sort_index()
    ax_sleep.plot(sleep_counts.index, sleep_counts.values, marker='o', color='#2c3e50', linewidth=2, markersize=6)
    ax_sleep.bar(sleep_counts.index, sleep_counts.values, width=0.35, color='#34495e', alpha=0.4, edgecolor='black')
    ax_sleep.set_title("Discrete Probability Distribution: Daily Sleep Hours", fontweight='bold')
    ax_sleep.set_xlabel("Reported Sleep Duration (Hours)")
    ax_sleep.set_ylabel("Probability P(X = x)")
    
    # 3. Smoking Status Probability Mass
    ax_smoke = axes[1, 0]
    smoke_counts = df_clean['smoking_status'].value_counts(normalize=True)[['Never', 'Former', 'Current']]
    ax_smoke.bar(smoke_counts.index, smoke_counts.values, color=['#27ae60', '#f39c12', '#c0392b'], edgecolor='black', alpha=0.85)
    ax_smoke.set_title("Categorical PMF: Smoking Status Prevalence", fontweight='bold')
    ax_smoke.set_ylabel("Empirical Probability")
    for idx, val in enumerate(smoke_counts.values):
        ax_smoke.annotate(f'{val:.1%}', xy=(idx, val), xytext=(0, 4), textcoords="offset points", ha='center', fontweight='bold')
        
    # 4. Physical Activity Prevalence
    ax_act = axes[1, 1]
    act_counts = df_clean['physical_activity'].value_counts(normalize=True)[['Sedentary', 'Moderate', 'Active']]
    ax_act.bar(act_counts.index, act_counts.values, color=['#e74c3c', '#3498db', '#2ecc71'], edgecolor='black', alpha=0.85)
    ax_act.set_title("Categorical PMF: Physical Activity Distribution", fontweight='bold')
    ax_act.set_ylabel("Empirical Probability")
    for idx, val in enumerate(act_counts.values):
        ax_act.annotate(f'{val:.1%}', xy=(idx, val), xytext=(0, 4), textcoords="offset points", ha='center', fontweight='bold')
        
    plt.tight_layout()
    fig4_path = os.path.join(figures_dir, 'fig04_discrete_probability_distributions.png')
    plt.savefig(fig4_path)
    plt.close()
    print(f"Saved: {fig4_path}")
    
    # -------------------------------------------------------------
    # CO4 & CO5: BIVARIATE RISK ANALYSIS (Box & Violin plots)
    # -------------------------------------------------------------
    print("[CO4 Analysis] Generating bivariate risk stratification plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    bivariate_pairs = [
        ('systolic_bp', 'Systolic Blood Pressure (mmHg)', '#3498db'),
        ('fasting_blood_sugar', 'Fasting Blood Glucose (mg/dL)', '#9b59b6'),
        ('pulse_pressure', 'Calculated Pulse Pressure (mmHg)', '#e67e22'),
        ('cholesterol_ratio', 'Atherogenic Ratio (Total / HDL)', '#1abc9c')
    ]
    
    for idx, (col, label, color) in enumerate(bivariate_pairs):
        ax = axes[idx // 2, idx % 2]
        sns.boxplot(data=df_clean, x='cvd_risk_outcome', y=col, ax=ax, hue='cvd_risk_outcome',
                    palette=['#a8d8ea', '#ff8b94'], legend=False,
                    showmeans=True, meanprops={"marker":"o", "markerfacecolor":"red", "markeredgecolor":"black"})
        
        # Calculate Mann-Whitney U test p-value
        group0 = df_clean[df_clean['cvd_risk_outcome'] == 0][col]
        group1 = df_clean[df_clean['cvd_risk_outcome'] == 1][col]
        u_stat, p_val = stats.mannwhitneyu(group0, group1)
        
        ax.set_title(f"{label} by CVD Risk Outcome\nMann-Whitney U p-value: {p_val:.2e}", fontweight='bold')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Low Risk (Class 0)', 'High Risk (Class 1)'])
        ax.set_xlabel("Clinical Outcome Category")
        ax.set_ylabel(label)
        
    plt.tight_layout()
    fig5_path = os.path.join(figures_dir, 'fig05_bivariate_risk_boxplots_violin.png')
    plt.savefig(fig5_path)
    plt.close()
    print(f"Saved: {fig5_path}")
    
    # -------------------------------------------------------------
    # CO4: MULTIVARIATE CORRELATION MATRIX HEATMAP
    # -------------------------------------------------------------
    print("[CO4 Analysis] Computing multivariate correlation heatmap...")
    corr_cols = ['age', 'systolic_bp', 'diastolic_bp', 'pulse_pressure', 'map_bp',
                 'bmi', 'fasting_blood_sugar', 'total_cholesterol', 'hdl_cholesterol',
                 'ldl_cholesterol', 'triglycerides', 'cholesterol_ratio', 'sleep_hours',
                 'diet_quality_score', 'cvd_risk_outcome']
    
    corr_matrix = df_clean[corr_cols].corr()
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    plt.figure(figsize=(14, 11))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=0.8, vmin=-0.8, center=0,
                annot=True, fmt='.2f', square=True, linewidths=.8, cbar_kws={"shrink": .8})
    plt.title("Multivariate Pearson Correlation Matrix (Stage 5 & 6: Represent & Refine)", fontsize=15, fontweight='bold', pad=15)
    plt.tight_layout()
    fig6_path = os.path.join(figures_dir, 'fig06_correlation_matrix_heatmap.png')
    plt.savefig(fig6_path)
    plt.close()
    print(f"Saved: {fig6_path}")
    
    # -------------------------------------------------------------
    # CO4: MULTIVARIATE INTERACTION DYNAMICS
    # -------------------------------------------------------------
    print("[CO4 Analysis] Generating multivariate interactive relationship scatter...")
    plt.figure(figsize=(12, 7))
    sns.scatterplot(
        data=df_clean,
        x='systolic_bp',
        y='fasting_blood_sugar',
        hue='cvd_risk_outcome',
        palette={0: '#2980b9', 1: '#c0392b'},
        size='bmi',
        sizes=(30, 220),
        alpha=0.75,
        edgecolor='black',
        linewidth=0.5
    )
    # Regression trendlines
    sns.regplot(data=df_clean[df_clean['cvd_risk_outcome'] == 0], x='systolic_bp', y='fasting_blood_sugar',
                scatter=False, ax=plt.gca(), color='#2980b9', line_kws={'linestyle':'--', 'linewidth': 2})
    sns.regplot(data=df_clean[df_clean['cvd_risk_outcome'] == 1], x='systolic_bp', y='fasting_blood_sugar',
                scatter=False, ax=plt.gca(), color='#c0392b', line_kws={'linestyle':'-', 'linewidth': 2.5})
                
    # Clinical cutoff threshold annotations
    plt.axvline(140, color='darkred', linestyle=':', linewidth=1.8, label='Clinical Hypertension (Stage 2: 140 mmHg)')
    plt.axhline(126, color='purple', linestyle=':', linewidth=1.8, label='Diabetic Threshold (Fasting: 126 mg/dL)')
    
    plt.title("Multivariate Interaction: Systolic BP vs Glucose Conditioned on CVD Risk & BMI", fontsize=14, fontweight='bold')
    plt.xlabel("Systolic Blood Pressure (mmHg)")
    plt.ylabel("Fasting Blood Sugar (mg/dL)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    plt.tight_layout()
    fig7_path = os.path.join(figures_dir, 'fig07_multivariate_scatter_interactions.png')
    plt.savefig(fig7_path)
    plt.close()
    print(f"Saved: {fig7_path}")
    
    # -------------------------------------------------------------
    # CO5: CLINICAL RISK STRATIFICATION MATRIX
    # -------------------------------------------------------------
    print("[CO5 Analysis] Clinical risk heatmap generation...")
    # Bin Age into cohorts
    df_clean['age_cohort'] = pd.cut(df_clean['age'], bins=[20, 40, 50, 60, 70, 85],
                                    labels=['<40 yrs', '40-49 yrs', '50-59 yrs', '60-69 yrs', '70+ yrs'])
    
    # Bin Blood Pressure into AHA stages
    df_clean['bp_stage'] = pd.cut(df_clean['systolic_bp'], bins=[0, 120, 130, 140, 250],
                                  labels=['Normal (<120)', 'Elevated (120-129)', 'Stage 1 (130-139)', 'Stage 2 (≥140)'])
                                  
    risk_pivot = df_clean.pivot_table(index='bp_stage', columns='age_cohort', values='cvd_risk_outcome', aggfunc='mean', observed=False)
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(risk_pivot * 100, annot=True, fmt='.1f', cmap='YlOrRd', cbar_kws={'label': 'Observed High CVD Risk Probability (%)'},
                linewidths=1.5, annot_kws={'size': 12, 'weight': 'bold'})
    plt.title("Clinical Risk Stratification Matrix: Joint Risk Probability by BP Stage & Age", fontsize=13, fontweight='bold')
    plt.xlabel("Age Cohort")
    plt.ylabel("AHA Blood Pressure Category")
    plt.tight_layout()
    fig8_path = os.path.join(figures_dir, 'fig08_clinical_risk_stratification_heatmap.png')
    plt.savefig(fig8_path)
    plt.close()
    print(f"Saved: {fig8_path}")
    
    # Save processed dataset
    processed_path = 'data/cardiovascular_processed_features.csv'
    df_clean.to_csv(processed_path, index=False)
    print(f"Processed dataset saved to: {processed_path}")
    
    return df_clean

if __name__ == '__main__':
    run_complete_eda()
