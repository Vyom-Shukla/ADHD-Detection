"""
This Python file trains the single best-performing model identified from the
"FINAL High-Resolution Tuning (v3)" experiment.

Based on the provided output, the best model is:
- Model:          Logistic Regression
- Feature Set:    Top 75
- Best ROC-AUC:   0.9814 (from the tuning script)

This script will:
1. Load the necessary data.
2. Define the exact "Top 75" features.
3. Define the Logistic Regression pipeline and its hyperparameter grid.
4. Run GridSearchCV to find the best hyperparameters for THIS combination.
5. Print the final model's performance and best parameters.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import time
import warnings

# --- File Paths (Update if necessary) ---
PATH_PATIENT_INFO = "Dataset\\patient_info.csv"
PATH_CPT = "Dataset\\CPT_II_ConnersContinuousPerformanceTest.csv"
PATH_FEATURES = "Dataset\\features.csv"

# --- 1. Define the Top 75 Key Features ---
# This list is from your 3000-run test
KEY_FEATURES_TOP_150 = [
    'ASRS', 'Percent Perseverations', 'Raw Score Commissions', 'WURS', 'Raw Score VarSE',
    'ACC_fft_coefficientattr"real"__coeff_84', 'Raw Score HitRTIsi', 'Neuro Confidence Index',
    'Percent Commissions', 'Raw Score HitSE', 'ACC_fft_coefficientattr"abs"__coeff_22',
    'ACC_fft_coefficientattr"real"__coeff_57', 'Raw Score Perseverations',
    'ACC_fft_coefficientattr"abs"_coeff_84', 'ACCfft_coefficientattr"real"__coeff_60',
    'ACC_fft_coefficientattr"imag"_coeff_30', 'ACCfft_coefficientattr"real"__coeff_56',
    'ACC_fft_coefficientattr"imag"__coeff_52', 'Adhd Confidence Index', 'Old Overall Index',
    'ACC_fft_coefficientattr"real"__coeff_81', 'Percent Omissions',
    'ACC_fft_coefficientattr"angle"_coeff_88', 'ACCfft_coefficientattr"angle"__coeff_57',
    'ACC_fft_coefficientattr"real"_coeff_5', 'ACCfft_coefficientattr"imag"__coeff_47',
    'ACC_fft_coefficientattr"real"_coeff_51', 'ACCfft_coefficientattr"imag"__coeff_22',
    'ACC_fft_coefficientattr"real"_coeff_99', 'ACCfft_coefficientattr"real"__coeff_39',
    'ACC_fft_coefficientattr"imag"_coeff_88', 'ACCfft_coefficientattr"real"__coeff_53',
    'ACC_fft_coefficientattr"angle"_coeff_28', 'ACCfft_coefficientattr"real"__coeff_20',
    'Raw Score Omissions', 'ACC_fft_coefficientattr"real"__coeff_41',
    'ACC_fft_coefficientattr"angle"_coeff_70', 'ACCfft_coefficientattr"angle"__coeff_74',
    'ACC_fft_coefficientattr"imag"_coeff_28', 'ACCfft_coefficientattr"abs"__coeff_70',
    'ACC_fft_coefficientattr"imag"_coeff_62', 'ACCfft_coefficientattr"abs"__coeff_15',
    'ACC_fft_coefficientattr"angle"_coeff_84', 'ACCfft_coefficientattr"real"__coeff_58',
    'ACC_change_quantilesf_agg"mean"_isabs_Falseqh_0.8_ql_0.6',
    'ACC_fft_coefficientattr"imag"_coeff_36', 'ACCcwt_coefficientscoeff_3w_2widths(2, 5, 10, 20)',
    'ACC_fft_coefficientattr"imag"_coeff_74', 'ACCfft_coefficientattr"real"__coeff_28',
    'Raw Score DPrime', 'ACC_fft_coefficientattr"imag"__coeff_97',
    'ACC_fft_coefficientattr"real"_coeff_55', 'ACCfft_coefficientattr"angle"__coeff_20',
    'ACC_ratio_value_number_to_time_series_length', 'ACCfft_coefficientattr"abs"__coeff_33',
    'ACC_fft_coefficientattr"angle"_coeff_97', 'ACCfft_coefficientattr"imag"__coeff_38',
    'ACC_fft_coefficientattr"imag"__coeff_91', 'Raw Score Beta',
    'ACC_fft_coefficientattr"real"_coeff_61', 'ACCfft_coefficientattr"real"__coeff_21',
    'ACC_fft_coefficientattr"angle"_coeff_56', 'ACCfft_coefficientattr"imag"__coeff_80',
    'ACC_change_quantilesf_agg"mean"_isabs_Trueqh_1.0ql_0.8', 'ACCfft_coefficientattr"abs"__coeff_40',
    'ACC_lempel_ziv_complexitybins_100', 'ACCfft_coefficientattr"angle"__coeff_38',
    'ACC_fft_coefficientattr"imag"_coeff_20', 'ACClinear_trendattr"stderr"',
    'ACC_fft_coefficientattr"imag"_coeff_77', 'ACCfft_coefficientattr"angle"__coeff_30',
    'ACC_fft_coefficientattr"abs"_coeff_77', 'ACCfft_coefficientattr"angle"__coeff_62',
    'ACC_fft_coefficientattr"real"_coeff_49', 'ACCfft_coefficientattr"abs"__coeff_39',
    'ACC_permutation_entropydimension_4tau_1', 'ACCfft_coefficientattr"abs"__coeff_29',
    'ACC_fft_coefficientattr"angle"_coeff_75', 'ACCfft_coefficientattr"abs"__coeff_12',
    'ACC_fft_coefficientattr"real"_coeff_43', 'ACCfft_coefficientattr"real"__coeff_25',
    'ACC_fft_coefficientattr"real"__coeff_77', 'Raw Score HitRTBlock',
    'ACC_fft_coefficientattr"abs"_coeff_28', 'ACCcwt_coefficientscoeff_2w_2widths(2, 5, 10, 20)',
    'ACC_fft_coefficientattr"angle"_coeff_19', 'ACCfft_coefficientattr"angle"__coeff_5',
    'ACC_agg_linear_trendattr"stderr"_chunk_len_5f_agg"mean"', 'ACC_cwt_coefficientscoeff_3w_5widths(2, 5, 10, 20)',
    'ACC_fft_coefficientattr"abs"_coeff_93', 'ACCnumber_peaks_n_50',
    'ACC_permutation_entropydimension_5tau_1', 'ACClempel_ziv_complexity_bins_10',
    'ACC_cwt_coefficientscoeff_1w_5widths(2, 5, 10, 20)', 'ACC_fft_coefficientattr"real"__coeff_24',
    'ACC_fft_coefficientattr"angle"_coeff_21', 'ACCagg_linear_trendattr"stderr"_chunk_len_10f_agg"min"',
    'ACC_fft_coefficientattr"real"_coeff_19', 'ACCfft_coefficientattr"real"__coeff_22',
    'ACC_fft_coefficientattr"abs"_coeff_83', 'ACCcwt_coefficientscoeff_2w_5widths(2, 5, 10, 20)',
    'ACC_cwt_coefficientscoeff_6w_2widths(2, 5, 10, 20)', 'ACC_fft_coefficientattr"angle"__coeff_49',
    'ACC_agg_linear_trendattr"stderr"_chunk_len_10f_agg"max"', 'ACC_fft_coefficientattr"real"__coeff_79',
    'ACC_fft_coefficientattr"abs"_coeff_76', 'ACCfft_coefficientattr"real"__coeff_36',
    'ACC_fft_coefficientattr"imag"_coeff_60', 'ACCfft_coefficientattr"real"__coeff_63',
    'ACC_fft_coefficientattr"angle"_coeff_26', 'ACCfft_coefficientattr"angle"__coeff_81',
    'ACC_number_cwt_peaksn_1', 'ACCfft_coefficientattr"imag"__coeff_72',
    'ACC_number_cwt_peaksn_5', 'ACCfft_coefficientattr"real"__coeff_78',
    'ACC_fft_coefficientattr"abs"_coeff_97', 'ACCpartial_autocorrelation_lag_9',
    'ACC_value_countvalue_0', 'ACCfft_coefficientattr"real"__coeff_38',
    'ACC_energy_ratio_by_chunksnum_segments_10segment_focus_9', 'ACCfft_coefficientattr"imag"__coeff_24',
    'ACC_fft_coefficientattr"real"_coeff_64', 'ACCfft_coefficientattr"real"__coeff_97',
    'ACC_fft_coefficientattr"angle"_coeff_78', 'ACCfft_coefficientattr"real"__coeff_88',
    'ACC_agg_linear_trendattr"stderr"_chunk_len_50f_agg"max"', 'ACC__mean_second_derivative_central',
    'ACC_count_above_mean', 'ACCagg_linear_trendattr"stderr"_chunk_len_10f_agg"mean"',
    'ACC_fft_coefficientattr"angle"__coeff_87', 'Raw Score HitSEBlock',
    'ACC_fft_coefficientattr"abs"_coeff_35', 'ACCchange_quantilesf_agg"var"_isabs_Trueqh_1.0_ql_0.6',
    'ACC_lempel_ziv_complexitybins_5', 'ACCrange_countmax_1000000000000.0_min_0',
    'ACC_first_location_of_maximum', 'ACCchange_quantilesf_agg"mean"_isabs_Falseqh_1.0_ql_0.8',
    'ACC_fft_coefficientattr"imag"_coeff_42', 'ACCfft_coefficientattr"real"__coeff_29',
    'ACC_fft_coefficientattr"real"_coeff_13', 'ACCnumber_peaks_n_10',
    'ACC_fft_coefficientattr"real"_coeff_3', 'ACCpartial_autocorrelation_lag_2',
    'ACC_fft_coefficientattr"imag"_coeff_43', 'ACCpermutation_entropydimension_3_tau_1',
    'ACC_fourier_entropybins_100', 'ACCfft_coefficientattr"real"__coeff_96',
    'ACC_fft_coefficientattr"abs"_coeff_42', 'ACCfft_coefficientattr"angle"__coeff_41',
    'ACC_fft_coefficientattr"real"__coeff_71'
]

# Select the top 75 features
BEST_FEATURES = KEY_FEATURES_TOP_150[:75]
TARGET_COL = 'ADHD'
ID_COL = 'ID'

def load_and_merge_data(feature_list):
    """
    Loads and merges data from the specified CSV files, selecting only
    the necessary columns.
    """
    print("--- Loading and Merging Data ---")
    
    # --- 3. Load Data ---
    patient_info_df = pd.read_csv(PATH_PATIENT_INFO, delimiter=';')
    cpt_df = pd.read_csv(PATH_CPT, delimiter=';')
    features_df = pd.read_csv(PATH_FEATURES, delimiter=';')

    # --- 4. Merge Data (Full Merge) ---
    patient_filtered = patient_info_df[patient_info_df['filter_$'] == 1]

    all_needed_cols = list(set(feature_list + [ID_COL, TARGET_COL]))

    patient_cols = [col for col in all_needed_cols if col in patient_info_df.columns] + [ID_COL, TARGET_COL]
    patient_data = patient_filtered[list(set(patient_cols))]

    cpt_cols = [col for col in all_needed_cols if col in cpt_df.columns] + [ID_COL]
    cpt_data = cpt_df[list(set(cpt_cols))]

    features_cols = [col for col in all_needed_cols if col in features_df.columns] + [ID_COL]
    features_data = features_df[list(set(features_cols))]

    merged_df = patient_data.merge(cpt_data, on=ID_COL, how='inner')
    merged_df = merged_df.merge(features_data, on=ID_COL, how='inner')

    print(f"Data loaded for {merged_df.shape[0]} patients.")

    merged_df = merged_df.replace([np.inf, -np.inf], np.nan)
    
    # Prepare X and y
    X = merged_df[feature_list]
    y = merged_df[TARGET_COL]
    
    return X, y

def train_final_model(X, y):
    """
    Trains the final Logistic Regression model using GridSearchCV.
    """
    print("\n--- Training Final Model (Log. Regression, Top 75 Features) ---")
    
    # Suppress warnings
    warnings.filterwarnings(action='ignore', category=UserWarning)
    warnings.filterwarnings(action='ignore', category=FutureWarning)

    # --- 5. Define Model and Hyperparameter Grid ---

    # Robust cross-validation: 5 splits, 3 repeats = 15 runs per param
    cv_method = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

    # Logistic Regression Pipeline
    log_reg_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42, max_iter=1000))
    ])
    
    # Hyperparameters to search
    log_reg_param_grid = {
        'classifier__C': [0.01, 0.1, 0.5, 1.0, 10.0]
    }

    # --- 6. Run the GridSearchCV ---
    
    grid_search = GridSearchCV(
        estimator=log_reg_pipeline,
        param_grid=log_reg_param_grid,
        cv=cv_method,
        scoring='roc_auc',
        n_jobs=-1, # Use all available cores
        verbose=1  # Show progress
    )

    start_time = time.time()
    grid_search.fit(X, y)
    end_time = time.time()

    print(f"\n--- Training Complete ---")
    print(f"Time taken: {((end_time - start_time)):.2f} seconds")

    # --- 7. Display Results ---
    
    print("\n--- Absolute Best Model ---")
    print(f"Model:          Logistic Regression")
    print(f"Feature Set:    Top 75")
    print(f"Best CV Score:  {grid_search.best_score_:.4f} (ROC-AUC)")
    print(f"Best Params:    {grid_search.best_params_}")
    
    # You can now use this 'final_model' object for predictions
    final_model = grid_search.best_estimator_
    
    print("\n'final_model' variable now holds the trained, optimized model.")
    return final_model

def main():
    try:
        X, y = load_and_merge_data(feature_list=BEST_FEATURES)
        final_model = train_final_model(X, y)
        
        # Example of how to use the model:
        # if hasattr(final_model, "predict"):
        #     # Get probabilities for the first 5 patients
        #     example_preds = final_model.predict_proba(X.head(5))[:, 1]
        #     print("\nExample Predictions (Probabilities):")
        #     print(example_preds)
            
    except FileNotFoundError as e:
        print(f"\nERROR: Data file not found.")
        print(f"Details: {e}")
        print("Please ensure the file paths at the top of the script are correct.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()