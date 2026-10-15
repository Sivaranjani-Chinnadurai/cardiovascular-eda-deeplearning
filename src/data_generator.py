"""
Cardiovascular and Metabolic Health Dataset Generator
Synthesizes a realistic, multi-feature epidemiological dataset adhering to clinical distributions
(Framingham Heart Study, NHANES, and WHO clinical thresholds).
"""

import numpy as np
import pandas as pd

def generate_cardiovascular_dataset(n_samples=2000, random_state=42):
    """
    Generates a synthetic yet clinically grounded cardiovascular risk dataset
    with 20 features, realistic covariance, continuous & discrete distributions,
    sparse missing values, and outliers for EDA & Deep Learning.
    """
    np.random.seed(random_state)
    
    # 1. Demographics
    patient_ids = [f"PAT_{i:05d}" for i in range(1, n_samples + 1)]
    age = np.random.normal(loc=54.5, scale=11.5, size=n_samples).astype(int)
    age = np.clip(age, 25, 82)
    
    gender_prob = [0.52, 0.48] # 52% Male, 48% Female
    gender = np.random.choice(['Male', 'Female'], size=n_samples, p=gender_prob)
    gender_num = np.where(gender == 'Male', 1, 0)
    
    # 2. Physiological & Vital Sign Features (conditioned on age and gender)
    # Systolic BP: increases with age; mean ~ 125 + 0.3*(age-50)
    systolic_noise = np.random.normal(0, 14, size=n_samples)
    systolic_bp = 125.0 + 0.35 * (age - 50) + 4.0 * gender_num + systolic_noise
    systolic_bp = np.clip(systolic_bp, 90, 210)
    
    # Diastolic BP: correlates with systolic (~0.65 correlation)
    diastolic_bp = 80.0 + 0.45 * (systolic_bp - 120) + np.random.normal(0, 7.5, size=n_samples)
    diastolic_bp = np.clip(diastolic_bp, 55, 125)
    
    # Body Mass Index (BMI): Log-normal distribution, positively skewed
    # log(BMI) mean ~ 3.32 (exp ~ 27.6 kg/m^2)
    bmi = np.exp(np.random.normal(loc=3.31, scale=0.17, size=n_samples))
    bmi = np.clip(bmi, 18.0, 48.0)
    
    # Resting Heart Rate (bpm): Normal distribution
    resting_heart_rate = np.random.normal(loc=72.0, scale=9.5, size=n_samples)
    resting_heart_rate = np.clip(resting_heart_rate, 48, 115).astype(int)
    
    # 3. Metabolic & Biochemical Biomarkers
    # Fasting Blood Sugar (mg/dL): Log-normal distribution with diabetic tail
    fbs_base = np.exp(np.random.normal(loc=4.62, scale=0.22, size=n_samples)) # median ~ 101 mg/dL
    # slight correlation with BMI
    fasting_blood_sugar = fbs_base + 0.6 * (bmi - 25)
    fasting_blood_sugar = np.clip(fasting_blood_sugar, 70, 310)
    
    # Total Cholesterol (mg/dL): Normal distribution
    total_cholesterol = np.random.normal(loc=206.0, scale=36.0, size=n_samples) + 0.25 * (age - 50)
    total_cholesterol = np.clip(total_cholesterol, 120, 340)
    
    # High-Density Lipoprotein (HDL) (mg/dL): females higher than males
    hdl_base = np.random.normal(loc=49.0, scale=11.0, size=n_samples) + 6.0 * (1 - gender_num) - 0.3 * (bmi - 25)
    hdl_cholesterol = np.clip(hdl_base, 22, 95)
    
    # Low-Density Lipoprotein (LDL) (mg/dL): primary atherogenic component
    ldl_cholesterol = total_cholesterol - hdl_cholesterol - (np.random.uniform(25, 45, size=n_samples))
    ldl_cholesterol = np.clip(ldl_cholesterol, 50, 240)
    
    # Triglycerides (mg/dL): highly right-skewed Log-Normal distribution
    triglycerides = np.exp(np.random.normal(loc=4.95, scale=0.38, size=n_samples)) + 1.2 * (bmi - 25)
    triglycerides = np.clip(triglycerides, 45, 520)
    
    # 4. Lifestyle & Behavioral Factors
    smoking_prob = [0.52, 0.26, 0.22] # Never, Former, Current
    smoking_status = np.random.choice(['Never', 'Former', 'Current'], size=n_samples, p=smoking_prob)
    
    alcohol_prob = [0.46, 0.43, 0.11] # Non-Drinker, Moderate, Heavy
    alcohol_intake = np.random.choice(['Non-Drinker', 'Moderate', 'Heavy'], size=n_samples, p=alcohol_prob)
    
    activity_prob = [0.38, 0.44, 0.18] # Sedentary, Moderate, Active
    physical_activity = np.random.choice(['Sedentary', 'Moderate', 'Active'], size=n_samples, p=activity_prob)
    
    # Diet Quality Score (1 to 10 discrete integer rating)
    diet_quality_score = np.random.choice(range(1, 11), size=n_samples, 
                                          p=[0.05, 0.08, 0.12, 0.15, 0.20, 0.17, 0.11, 0.07, 0.03, 0.02])
    
    # Sleep Hours per Day (continuous/discrete, normal around 6.9 hours)
    sleep_hours = np.random.normal(loc=6.85, scale=1.15, size=n_samples)
    sleep_hours = np.clip(np.round(sleep_hours * 2) / 2, 4.0, 10.5) # rounded to half-hours
    
    # Family History of CVD (0 = No, 1 = Yes)
    family_history_cvd = np.random.choice([0, 1], size=n_samples, p=[0.68, 0.32])
    
    # Chronic Stress Level (Low, Moderate, High)
    stress_prob = [0.32, 0.48, 0.20]
    chronic_stress_level = np.random.choice(['Low', 'Moderate', 'High'], size=n_samples, p=stress_prob)
    
    # 5. Latent Risk Calculation and Target Variable (CVD Risk Outcome)
    # Framingham-like risk log-odds formula
    stress_mult = {'Low': 0.0, 'Moderate': 0.35, 'High': 0.85}
    smoking_mult = {'Never': 0.0, 'Former': 0.3, 'Current': 0.9}
    activity_mult = {'Sedentary': 0.5, 'Moderate': 0.0, 'Active': -0.6}
    
    log_odds = (
        -1.75
        + 0.048 * (age - 52)
        + 0.40 * gender_num
        + 0.035 * (systolic_bp - 125)
        + 0.028 * (bmi - 26)
        + 0.016 * (fasting_blood_sugar - 100)
        + 0.010 * (total_cholesterol - 200)
        - 0.035 * (hdl_cholesterol - 50)
        + 0.003 * (triglycerides - 150)
        + 0.70 * family_history_cvd
        + np.array([smoking_mult[s] for s in smoking_status])
        + np.array([activity_mult[a] for a in physical_activity])
        + np.array([stress_mult[st] for st in chronic_stress_level])
        - 0.10 * (diet_quality_score - 5)
        - 0.12 * (sleep_hours - 7)
    )
    
    prob_cvd = 1.0 / (1.0 + np.exp(-log_odds))
    cvd_risk_outcome = np.random.binomial(n=1, p=prob_cvd, size=n_samples)
    
    df = pd.DataFrame({
        'patient_id': patient_ids,
        'age': age,
        'gender': gender,
        'systolic_bp': np.round(systolic_bp, 1),
        'diastolic_bp': np.round(diastolic_bp, 1),
        'bmi': np.round(bmi, 2),
        'resting_heart_rate': resting_heart_rate,
        'fasting_blood_sugar': np.round(fasting_blood_sugar, 1),
        'total_cholesterol': np.round(total_cholesterol, 1),
        'hdl_cholesterol': np.round(hdl_cholesterol, 1),
        'ldl_cholesterol': np.round(ldl_cholesterol, 1),
        'triglycerides': np.round(triglycerides, 1),
        'smoking_status': smoking_status,
        'alcohol_intake': alcohol_intake,
        'physical_activity': physical_activity,
        'diet_quality_score': diet_quality_score,
        'sleep_hours': sleep_hours,
        'family_history_cvd': family_history_cvd,
        'chronic_stress_level': chronic_stress_level,
        'cvd_risk_outcome': cvd_risk_outcome
    })
    
    # Introduce ~1.5% to 2.5% Missing Values (MAR/MCAR) to test Preprocessing Imputation (Step 2)
    missing_indices_fbs = np.random.choice(df.index, size=int(0.02 * n_samples), replace=False)
    missing_indices_tg = np.random.choice(df.index, size=int(0.025 * n_samples), replace=False)
    missing_indices_sleep = np.random.choice(df.index, size=int(0.015 * n_samples), replace=False)
    
    df.loc[missing_indices_fbs, 'fasting_blood_sugar'] = np.nan
    df.loc[missing_indices_tg, 'triglycerides'] = np.nan
    df.loc[missing_indices_sleep, 'sleep_hours'] = np.nan
    
    # Introduce 12 Duplicate Rows to test Duplicate Detection and Removal (Step 2)
    duplicate_rows = df.sample(n=12, random_state=42)
    df = pd.concat([df, duplicate_rows], ignore_index=True)
    
    # Introduce a few physiological extreme values / outliers to test IQR detection
    outlier_idx = np.random.choice(df.index[:100], size=5, replace=False)
    df.loc[outlier_idx[0], 'systolic_bp'] = 245.0
    df.loc[outlier_idx[1], 'triglycerides'] = 680.0
    df.loc[outlier_idx[2], 'fasting_blood_sugar'] = 395.0
    
    return df

if __name__ == '__main__':
    df = generate_cardiovascular_dataset(n_samples=2000, random_state=42)
    output_path = 'data/cardiovascular_risk_data.csv'
    df.to_csv(output_path, index=False)
    print(f"Successfully generated {len(df)} records with {df.shape[1]} features.")
    print(f"Saved to: {output_path}")
    print("Class distribution:\n", df['cvd_risk_outcome'].value_counts(normalize=True))
    print("Null counts:\n", df.isnull().sum()[df.isnull().sum() > 0])
