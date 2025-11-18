Collecting workspace information# ADHD Detection Using Logistic Regression

## Project Overview

This project develops a machine learning model to detect ADHD (Attention-Deficit/Hyperactivity Disorder) using logistic regression. The model is trained on a combination of clinical assessments, accelerometer data, and heart rate variability (HRV) features to predict ADHD diagnosis with high accuracy (ROC-AUC: 0.9814).

## Dataset

The project uses three main data sources:

### 1. **patient_info.csv**
Contains demographic and clinical information for patients:
- Patient ID and basic demographics (sex, age)
- Clinical assessments (WURS, ASRS, MADRS, HADS-A, HADS-D)
- Medication history (antidepressants, mood stabilizers, stimulants, etc.)
- Comorbidity flags (BIPOLAR, UNIPOLAR, ANXIETY, SUBSTANCE, OTHER)
- Target variable: `ADHD` (binary: 0 or 1)
- Data quality filter: `filter_$` (1 = valid data)

### 2. **CPT_II_ConnersContinuousPerformanceTest.csv**
Continuous Performance Test (CPT-II) results measuring attention and impulse control:
- Raw scores for hits, commissions, omissions, perseverations
- Response time statistics (Hit RT, Hit SE, VarSE)
- Derived indices (D-Prime, Beta, Confidence Indices)

### 3. **features.csv**
Engineered time-series features extracted from accelerometer data:
- Statistical features (mean, variance, skewness, kurtosis, entropy)
- Frequency-domain features (FFT coefficients)
- Time-series complexity measures (Lempel-Ziv complexity, permutation entropy)
- Aggregated statistics by chunks and windows

## Model Architecture

### Pipeline
```
Data Input → Imputation (median) → Standardization → Logistic Regression
```

### Key Configuration
- **Algorithm**: Logistic Regression with L2 regularization
- **Feature Set**: Top 75 features (selected from 150+ candidates)
- **Hyperparameter Tuning**: GridSearchCV with RepeatedStratifiedKFold
- **Cross-Validation**: 5 splits × 3 repeats = 15 folds per parameter
- **Class Balancing**: Balanced class weights

### Top Performing Features
- Clinical scores: ASRS, WURS, Neuro Confidence Index
- CPT-II metrics: Raw scores for hits, commissions, omissions, perseverations
- Accelerometer FFT coefficients (real, imaginary, absolute, angle)
- Time-series complexity measures

## File Structure

```
ADHD-DETEC/
├── LR_model.py                                    # Main training script
├── Dataset/
│   ├── patient_info.csv                          # Patient demographics & assessments
│   ├── CPT_II_ConnersContinuousPerformanceTest.csv # CPT-II test results
│   ├── features.csv                              # Engineered accelerometer features
│   └── README.md                                 # Dataset documentation
└── README.md                                     # This file
```

## Usage

### Requirements
```bash
pip install pandas numpy scikit-learn
```

### Running the Model

```python
python LR_model.py
```

This will:
1. Load and merge data from all three CSV files
2. Select the top 75 features
3. Run GridSearchCV to find optimal hyperparameters
4. Train the final Logistic Regression model
5. Display performance metrics and best parameters

### Making Predictions

```python
from LR_model import load_and_merge_data, train_final_model, BEST_FEATURES

# Load data
X, y = load_and_merge_data(feature_list=BEST_FEATURES)

# Train model
model = train_final_model(X, y)

# Make predictions
predictions = model.predict(X)
probabilities = model.predict_proba(X)[:, 1]
```

## Performance

- **Best ROC-AUC**: 0.9814
- **Model**: Logistic Regression
- **Features**: Top 75
- **Training Method**: GridSearchCV with repeated stratified k-fold cross-validation

## Data Preprocessing

1. **Filtering**: Only patients with `filter_$` = 1 are included
2. **Merging**: Inner join across all three datasets on patient ID
3. **Handling Infinite Values**: Replaced with NaN
4. **Imputation**: Median imputation for missing values
5. **Scaling**: StandardScaler normalization

## Notes

- The model handles class imbalance through balanced class weights
- Missing values are common in clinical data and are handled via median imputation
- Features are standardized to ensure logistic regression performs optimally
- The feature set was selected based on extensive hyperparameter tuning

## References

- CPT-II (Conners Continuous Performance Test II): Measures sustained attention and impulse control
- WURS: Wender Utah Rating Scale for ADHD symptoms
- ASRS: Adult ADHD Self-Report Scale
- HADS: Hospital Anxiety and Depression Scale

## Author

Machine Learning Pipeline for ADHD Detection