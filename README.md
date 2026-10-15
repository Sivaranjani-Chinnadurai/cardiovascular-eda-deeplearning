# Cardiovascular & Metabolic Health Analytics: Comprehensive EDA, Distribution Modeling, and Deep Neural Network Risk Classification

## 📌 Executive Summary

Cardiovascular diseases (CVDs) remain the leading contributor to global mortality, claiming an estimated 17.9 million lives each year (World Health Organization). This project delivers an end-to-end, production-grade exploratory data analysis (EDA), parametric probability distribution modeling, and deep learning predictive pipeline conducted on an epidemiological cohort of **2,000 adult patients** across **20 clinical, demographic, metabolic, and behavioral features**.

The project rigorously applies **Ben Fry's Seven Stages of Data Visualization**

---

## 🏛️ Ben Fry's 7 Stages of Data Visualization

```mermaid
flowchart LR
    A["1. Acquire<br/>(Ingest CSV, N=2,000)"] --> B["2. Parse<br/>(Classify Types & Formats)"]
    B --> C["3. Filter<br/>(Deduplicate & Winsorize IQR)"]
    C --> D["4. Mine<br/>(K-S Tests, Moments, Skewness)"]
    D --> E["5. Represent<br/>(Histograms, KDEs, PMFs, Heatmaps)"]
    E --> F["6. Refine<br/>(AHA Thresholds & Palette Contrast)"]
    F --> G["7. Interact<br/>(Stratified Cohorts & Decision Grids)"]
```

1. **Acquire**: Ingest raw patient records (`data/cardiovascular_risk_data.csv`), verify provenance, check schema integrity.
2. **Parse**: Classify features into continuous vitals, discrete ratings, ordinal psychosocial scores, and nominal behavioral markers.
3. **Filter**: Identify and purge duplicate entries (12 duplicates removed), perform median imputation for MCAR missingness, and treat extreme values using Tukey's Interquartile Range ($IQR$) fences winsorized to the 1st and 99th percentiles.
4. **Mine**: Compute central moments, skewness, kurtosis, parametric distribution models, and non-parametric hypothesis tests (Mann-Whitney U).
5. **Represent**: Select geometries including density histograms, theoretical Q-Q probability plots, discrete PMF bar charts, grouped boxplots, and heatmaps.
6. **Refine**: Overlay clinical diagnostic reference thresholds (AHA Hypertension stages $\ge 140$ mmHg, Diabetic fasting glucose $\ge 126$ mg/dL), establish high-contrast color palettes, and ensure accessible typography.
7. **Interact**: Condition bivariate distributions across risk strata and construct a 2D clinical risk triage decision matrix.

---

## 🧬 Dataset Architecture & Feature Dictionary

The dataset comprises **2,000 patient records** spanning **20 features** adhering to clinical cardiology guidelines:

| Feature Name | Type | Measurement Unit | Domain | Description |
| :--- | :--- | :--- | :--- | :--- |
| `patient_id` | String | Identifier | Administrative | Unique alphanumeric patient record ID |
| `age` | Integer | Years (25–82) | Demographic | Chronological patient age |
| `gender` | Nominal | Male / Female | Demographic | Biological sex (52% Male, 48% Female) |
| `systolic_bp` | Continuous | mmHg (90–210) | Vital Signs | Systolic arterial blood pressure |
| `diastolic_bp` | Continuous | mmHg (55–125) | Vital Signs | Diastolic arterial blood pressure |
| `bmi` | Continuous | kg/m² (18.0–48.0) | Anthropometric | Body Mass Index (Adiposity measure) |
| `resting_heart_rate` | Continuous | bpm (48–115) | Vital Signs | Resting cardiac pulse rate |
| `fasting_blood_sugar` | Continuous | mg/dL (70–310) | Metabolic | Serum fasting blood glucose concentration |
| `total_cholesterol` | Continuous | mg/dL (120–340) | Lipid Profile | Total circulating serum cholesterol |
| `hdl_cholesterol` | Continuous | mg/dL (22–95) | Lipid Profile | Cardioprotective High-Density Lipoprotein |
| `ldl_cholesterol` | Continuous | mg/dL (50–240) | Lipid Profile | Atherogenic Low-Density Lipoprotein |
| `triglycerides` | Continuous | mg/dL (45–520) | Lipid Profile | Circulating blood serum triglycerides |
| `smoking_status` | Nominal | Never / Former / Current | Behavioral | Tobacco consumption category |
| `alcohol_intake` | Nominal | Non-Drinker / Mod / Heavy | Behavioral | Weekly alcohol consumption classification |
| `physical_activity` | Nominal | Sedentary / Mod / Active | Behavioral | Weekly physical exercise activity level |
| `diet_quality_score` | Discrete | Rating (1–10) | Lifestyle | Composite healthy eating adherence score |
| `sleep_hours` | Discrete | Hours/night (4.0–10.5) | Lifestyle | Self-reported nightly sleep duration |
| `family_history_cvd` | Binary | 0 (No) / 1 (Yes) | Medical History | CVD diagnosis in first-degree relatives |
| `chronic_stress_level`| Ordinal | Low / Moderate / High | Psychosocial | Perceived chronic stress rating |
| **`cvd_risk_outcome`** | **Binary** | **0 (Low) / 1 (High)** | **Target** | **Primary cardiovascular risk outcome event** |

### Engineered Biomarkers:
- **Pulse Pressure ($PP$)**: $\text{Systolic BP} - \text{Diastolic BP}$ (Surrogate marker for aortic vascular stiffness).
- **Atherogenic Cholesterol Ratio ($CR$)**: $\frac{\text{Total Cholesterol}}{\text{HDL Cholesterol}}$ (Atherosclerotic balance metric).
- **Mean Arterial Pressure ($MAP$)**: $\text{Diastolic BP} + \frac{1}{3} PP$ (Organ perfusion hemodynamic index).

---

## 📊 Key Visualizations Gallery (With Tripartite Analysis)

### 1. Pre-Filtering & Missingness Audit
![Figure 1](figures/fig01_missing_and_duplicates.png)
- **What it shows**: Bar chart of initial missing values (`fasting_blood_sugar`: 41 rows, 2.0%; `triglycerides`: 50 rows, 2.5%; `sleep_hours`: 30 rows, 1.5%) and duplicate proportion (12 rows, 0.6%).
- **Why created**: Implements Ben Fry's **Acquire** & **Filter** stages, confirming data completeness and certifying that MCAR/MAR missingness is safely imputable via medians without parameter distortion.
- **Insights obtained**: Minimal data loss (0.6% duplicate purging) preserves 2,000 clean records, providing high statistical power for downstream neural modeling.

---

### 2. Univariate Continuous Biomarker Distributions
![Figure 2](figures/fig02_univariate_continuous_distributions.png)
- **What it shows**: Density histograms overlaid with continuous Kernel Density Estimations (KDE), parametric mean (dashed red line), and sample median (dotted black line).
- **Why created**: Fulfills **CO4 (Univariate Analysis - 4 Marks)**, assessing central tendency, dispersion, skewness, and kurtosis to determine whether normality assumptions hold.
- **Insights obtained**: Systolic BP exhibits near-normal distribution ($\text{Skewness} = +0.24, \text{Kurtosis} = -0.12$). In contrast, Triglycerides ($\text{Skewness} = +0.88$) and Fasting Glucose ($\text{Skewness} = +0.65$) display pronounced positive right-skewness.

---

### 3. Theoretical Probability Distribution Fitting & Q-Q Plots
![Figure 3](figures/fig03_theoretical_distribution_fitting_qq.png)
- **What it shows**: Parametric Normal and Log-Normal distribution curves fitted over empirical densities alongside Quantile-Quantile (Q-Q) plots, annotated with Kolmogorov-Smirnov (K-S) goodness-of-fit test statistics.
- **Why created**: Directly satisfies **CO4 (Discrete/Continuous Probability Distributions - 4 Marks)**, establishing the mathematical probability law governing biological markers.
- **Insights obtained**: Systolic BP tightly conforms to a continuous Gaussian distribution $\mathcal{N}(\mu=130.4, \sigma=14.3)$ ($p > 0.05$). Serum Triglycerides strongly adheres to a two-parameter Log-Normal distribution $\text{Lognormal}(s=0.36, \text{scale}=178.6)$, where logarithmic transformation $\ln(TG)$ yields near-perfect linearity along the Q-Q reference diagonal.

---

### 4. Discrete Probability Distributions & Categorical PMF
![Figure 4](figures/fig04_discrete_probability_distributions.png)
- **What it shows**: Empirical Probability Mass Functions (PMF) for Diet Quality (1–10) and Daily Sleep Duration alongside multinomial categorical PMFs for Smoking Status and Physical Activity.
- **Why created**: Fulfills **CO4 (Discrete Probability Distributions - 4 Marks)**, evaluating behavioral risk exposures across the population.
- **Insights obtained**: Diet Quality follows a unimodal discrete PMF peaking at scores 5 and 6 (37% cumulative probability). Sedentary lifestyle accounts for 38.0% of the cohort, while current smoking represents 22.0%, defining primary targets for non-pharmacological interventions.

---

### 5. Bivariate Risk Stratification & Hypothesis Testing
![Figure 5](figures/fig05_bivariate_risk_boxplots_violin.png)
- **What it shows**: Comparative boxplots of biomarkers across Low Risk (Class 0) vs. High Risk (Class 1) cohorts, annotated with Mann-Whitney U test p-values.
- **Why created**: Satisfies **CO4 (Bivariate Analysis - 4 Marks)**, determining whether biomarker distributions diverge significantly between diagnostic groups without homoscedasticity assumptions.
- **Insights obtained**: All biomarkers diverge with extreme statistical significance ($p < 10^{-20}$). Systolic BP median is 137.2 mmHg in the high-risk cohort vs. 124.6 mmHg in the low-risk cohort. Calculated Pulse Pressure demonstrates profound discriminatory power ($p = 1.88 \times 10^{-35}$).

---

### 6. High-Dimensional Pearson Correlation Heatmap
![Figure 6](figures/fig06_correlation_matrix_heatmap.png)
- **What it shows**: Lower-triangular correlation matrix heatmap displaying pairwise Pearson coefficients ($r$) across 15 continuous and clinical attributes.
- **Why created**: Satisfies **CO4 (Multivariate Analysis - 4 Marks)**, identifying collinearity structures and measuring linear dependency with `cvd_risk_outcome`.
- **Insights obtained**: Systolic BP ($r = +0.44$), Pulse Pressure ($r = +0.39$), Age ($r = +0.33$), and Fasting Glucose ($r = +0.28$) represent the top positive drivers. Diet Quality ($r = -0.21$) and HDL Cholesterol ($r = -0.24$) demonstrate protective negative correlations.

---

### 7. Multivariate Interaction Dynamics
![Figure 7](figures/fig07_multivariate_scatter_interactions.png)
- **What it shows**: Multi-dimensional scatter plot visualizing Systolic BP vs. Fasting Glucose, color-coded by CVD Risk, point-sized by BMI, and overlaid with clinical cutoff lines (140 mmHg and 126 mg/dL).
- **Why created**: Fulfills **CO4 & CO5 (Multivariate Interaction & Practical Application)**, demonstrating how multi-biomarker syndromic clustering accelerates cardiovascular disease risk.
- **Insights obtained**: Patients in the upper-right quadrant (hypertensive and diabetic) show near 100% assignment to High CVD Risk. Large markers (high BMI $\ge 32$) concentrate within this quadrant, illustrating cardiometabolic compounding effects.

---

### 8. Clinical Risk Stratification Matrix
![Figure 8](figures/fig08_clinical_risk_stratification_heatmap.png)
- **What it shows**: Two-dimensional decision heatmap cross-tabulating Age cohorts against AHA Blood Pressure stages, displaying empirical high-risk probability percentages in each cell.
- **Why created**: Fulfills **CO5 (Clinical Recommendations & Decision Support - 4 Marks)**, providing an interpretable risk lookup grid for clinical triage.
- **Insights obtained**: Risk scales monotonically from 11.2% in young normotensives (<40 yrs) to 84.6% in elderly patients with Stage 2 Hypertension (70+ yrs). In patients aged 50–59, progressing to Stage 2 hypertension nearly doubles event risk (from 35.1% to 68.7%).

---

## 🧠 Deep Learning Architecture & Evaluation (Step 4 & 5)

A PyTorch **Deep Multi-Layer Perceptron (MLP)** binary classifier was constructed with Batch Normalization and Dropout regularization:

```mermaid
flowchart TD
    In["Input Layer (24 Standardized & One-Hot Features)"] --> L1["Linear (24 → 128) + BatchNorm1d + ReLU + Dropout(0.30)"]
    L1 --> L2["Linear (128 → 64) + BatchNorm1d + ReLU + Dropout(0.25)"]
    L2 --> L3["Linear (64 → 32) + ReLU + Dropout(0.15)"]
    L3 --> Out["Linear (32 → 1) + Sigmoid Activation"]
    Out --> Pred["Cardiovascular Risk Probability P(Y=1|X)"]
```

### Hyperparameter Tuning (Figure 14):
Exploration across learning rates ($\text{lr} \in \{0.01, 0.001, 0.0002\}$) demonstrated that $\text{lr} = 0.001$ provided optimal convergence without loss oscillation.

| Figure 9: Loss vs Epoch | Figure 10: Accuracy vs Epoch |
| :---: | :---: |
| ![Loss](figures/fig09_model_loss_epoch.png) | ![Accuracy](figures/fig10_model_accuracy_epoch.png) |

| Figure 11: Confusion Matrix | Figure 12: ROC & PR Curves |
| :---: | :---: |
| ![Confusion Matrix](figures/fig11_model_confusion_matrix.png) | ![ROC Curve](figures/fig12_model_roc_pr_curve.png) |

### Figure 13: Permutation Feature Importance
![Feature Importance](figures/fig13_feature_importance.png)
- **Interpretation**: Systolic Blood Pressure induces the largest performance drop ($\Delta \text{AUC} = 0.082$), followed by Pulse Pressure ($\Delta \text{AUC} = 0.054$), Fasting Glucose, Age, and Atherogenic Ratio.

### Unseen Test Set Performance Benchmark:
- **Test Accuracy**: **71.67%**
- **Precision**: **68.09%**
- **Recall / Sensitivity**: **70.59%**
- **Specificity**: **72.56%**
- **F1-Score**: **0.6931**
- **ROC-AUC**: **0.7819**
- **Average Precision (PR-AUC)**: **0.7415**

---

## 📁 Repository Structure

```
EDA-project/
├── data/
│   ├── cardiovascular_risk_data.csv          # Raw 2,000-patient epidemiological dataset (20 features)
│   └── cardiovascular_processed_features.csv # Preprocessed, winsorized, and engineered dataset
├── figures/                                  # 14 publication-grade figures (300 DPI)
│   ├── fig01_missing_and_duplicates.png
│   ├── fig02_univariate_continuous_distributions.png
│   ├── fig03_theoretical_distribution_fitting_qq.png
│   ├── fig04_discrete_probability_distributions.png
│   ├── fig05_bivariate_risk_boxplots_violin.png
│   ├── fig06_correlation_matrix_heatmap.png
│   ├── fig07_multivariate_scatter_interactions.png
│   ├── fig08_clinical_risk_stratification_heatmap.png
│   ├── fig09_model_loss_epoch.png
│   ├── fig10_model_accuracy_epoch.png
│   ├── fig11_model_confusion_matrix.png
│   ├── fig12_model_roc_pr_curve.png
│   ├── fig13_feature_importance.png
│   └── fig14_hyperparameter_tuning.png
├── reports/
│   ├── Project_Report_Cardiovascular_EDA_DL.pdf   # Formatted PDF Project Report (11 Sections)
│   └── Project_Report_Cardiovascular_EDA_DL.docx  # Formatted Word Document Project Report
├── src/
│   ├── __init__.py
│   ├── data_generator.py                     # Synthetic epidemiological data generator
│   ├── eda_analysis.py                       # 7-stage EDA & distribution visualization suite
│   └── deep_learning_model.py                # PyTorch Deep MLP architecture, training & evaluation
├── eda_and_deep_learning_pipeline.ipynb      # Executed Jupyter Notebook with outputs and markdown
├── main.py                                   # Master CLI runner
├── requirements.txt                          # Project dependency specifications
└── README.md                                 # Public GitHub repository documentation
```

---

## 🚀 Reproduction & Setup Guide

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/EDA-project.git
cd EDA-project
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Entire End-to-End Pipeline
```bash
python main.py
```

### 4. Launch the Interactive Jupyter Notebook
```bash
jupyter notebook eda_and_deep_learning_pipeline.ipynb
```

---

