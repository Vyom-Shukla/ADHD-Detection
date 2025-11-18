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
    'ACC__fft_coefficient__attr_"real"__coeff_84', 'Raw Score HitRTIsi', 'Neuro Confidence Index',
    'Percent Commissions', 'Raw Score HitSE', 'ACC__fft_coefficient__attr_"abs"__coeff_22',
    'ACC__fft_coefficient__attr_"real"__coeff_57', 'Raw Score Perseverations',
    'ACC__fft_coefficient__attr_"abs"__coeff_84', 'ACC__fft_coefficient__attr_"real"__coeff_60',
    'ACC__fft_coefficient__attr_"imag"__coeff_30', 'ACC__fft_coefficient__attr_"real"__coeff_56',
    'ACC__fft_coefficient__attr_"imag"__coeff_52', 'Adhd Confidence Index', 'Old Overall Index',
    'ACC__fft_coefficient__attr_"real"__coeff_81', 'Percent Omissions',
    'ACC__fft_coefficient__attr_"angle"__coeff_88', 'ACC__fft_coefficient__attr_"angle"__coeff_57',
    'ACC__fft_coefficient__attr_"real"__coeff_5', 'ACC__fft_coefficient__attr_"imag"__coeff_47',
    'ACC__fft_coefficient__attr_"real"__coeff_51', 'ACC__fft_coefficient__attr_"imag"__coeff_22',
    'ACC__fft_coefficient__attr_"real"__coeff_99', 'ACC__fft_coefficient__attr_"real"__coeff_39',
    'ACC__fft_coefficient__attr_"imag"__coeff_88', 'ACC__fft_coefficient__attr_"real"__coeff_53',
    'ACC__fft_coefficient__attr_"angle"__coeff_28', 'ACC__fft_coefficient__attr_"real"__coeff_20',
    'Raw Score Omissions', 'ACC__fft_coefficient__attr_"real"__coeff_41',
    'ACC__fft_coefficient__attr_"angle"__coeff_70', 'ACC__fft_coefficient__attr_"angle"__coeff_74',
    'ACC__fft_coefficient__attr_"imag"__coeff_28', 'ACC__fft_coefficient__attr_"abs"__coeff_70',
    'ACC__fft_coefficient__attr_"imag"__coeff_62', 'ACC__fft_coefficient__attr_"abs"__coeff_15',
    'ACC__fft_coefficient__attr_"angle"__coeff_84', 'ACC__fft_coefficient__attr_"real"__coeff_58',
    'ACC__change_quantiles__f_agg_"mean"__isabs_False__qh_0.8__ql_0.6',
    'ACC__fft_coefficient__attr_"imag"__coeff_36', 'ACC__cwt_coefficients__coeff_3__w_2__widths_(2, 5, 10, 20)',
    'ACC__fft_coefficient__attr_"imag"__coeff_74', 'ACC__fft_coefficient__attr_"real"__coeff_28',
    'Raw Score DPrime', 'ACC__fft_coefficient__attr_"imag"__coeff_97',
    'ACC__fft_coefficient__attr_"real"__coeff_55', 'ACC__fft_coefficient__attr_"angle"__coeff_20',
    'ACC__ratio_value_number_to_time_series_length', 'ACC__fft_coefficient__attr_"abs"__coeff_33',
    'ACC__fft_coefficient__attr_"angle"__coeff_97', 'ACC__fft_coefficient__attr_"imag"__coeff_38',
    'ACC__fft_coefficient__attr_"imag"__coeff_91', 'Raw Score Beta',
    'ACC__fft_coefficient__attr_"real"__coeff_61', 'ACC__fft_coefficient__attr_"real"__coeff_21',
    'ACC__fft_coefficient__attr_"angle"__coeff_56', 'ACC__fft_coefficient__attr_"imag"__coeff_80',
    'ACC__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.8', 'ACC__fft_coefficient__attr_"abs"__coeff_40',
    'ACC__lempel_ziv_complexity__bins_100', 'ACC__fft_coefficient__attr_"angle"__coeff_38',
    'ACC__fft_coefficient__attr_"imag"__coeff_20', 'ACC__linear_trend__attr_"stderr"',
    'ACC__fft_coefficient__attr_"imag"__coeff_77', 'ACC__fft_coefficient__attr_"angle"__coeff_30',
    'ACC__fft_coefficient__attr_"abs"__coeff_77', 'ACC__fft_coefficient__attr_"angle"__coeff_62',
    'ACC__fft_coefficient__attr_"real"__coeff_49', 'ACC__fft_coefficient__attr_"abs"__coeff_39',
    'ACC__permutation_entropy__dimension_4__tau_1', 'ACC__fft_coefficient__attr_"abs"__coeff_29',
    'ACC__fft_coefficient__attr_"angle"__coeff_75', 'ACC__fft_coefficient__attr_"abs"__coeff_12',
    'ACC__fft_coefficient__attr_"real"__coeff_43', 'ACC__fft_coefficient__attr_"real"__coeff_25',
    'ACC__fft_coefficient__attr_"real"__coeff_77', 'Raw Score HitRTBlock',
    'ACC__fft_coefficient__attr_"abs"__coeff_28', 'ACC__cwt_coefficients__coeff_2__w_2__widths_(2, 5, 10, 20)',
    'ACC__fft_coefficient__attr_"angle"__coeff_19', 'ACC__fft_coefficient__attr_"angle"__coeff_5',
    'ACC__agg_linear_trend__attr_"stderr"__chunk_len_5__f_agg_"mean"', 'ACC__cwt_coefficients__coeff_3__w_5__widths_(2, 5, 10, 20)',
    'ACC__fft_coefficient__attr_"abs"__coeff_93', 'ACC__number_peaks__n_50',
    'ACC__permutation_entropy__dimension_5__tau_1', 'ACC__lempel_ziv_complexity__bins_10',
    'ACC__cwt_coefficients__coeff_1__w_5__widths_(2, 5, 10, 20)', 'ACC__fft_coefficient__attr_"real"__coeff_24',
    'ACC__fft_coefficient__attr_"angle"__coeff_21', 'ACC__agg_linear_trend__attr_"stderr"__chunk_len_10__f_agg_"min"',
    'ACC__fft_coefficient__attr_"real"__coeff_19', 'ACC__fft_coefficient__attr_"real"__coeff_22',
    'ACC__fft_coefficient__attr_"abs"__coeff_83', 'ACC__cwt_coefficients__coeff_2__w_5__widths_(2, 5, 10, 20)',
    'ACC__cwt_coefficients__coeff_6__w_2__widths_(2, 5, 10, 20)', 'ACC__fft_coefficient__attr_"angle"__coeff_49',
    'ACC__agg_linear_trend__attr_"stderr"__chunk_len_10__f_agg_"max"', 'ACC__fft_coefficient__attr_"real"__coeff_79',
    'ACC__fft_coefficient__attr_"abs"__coeff_76', 'ACC__fft_coefficient__attr_"real"__coeff_36',
    'ACC__fft_coefficient__attr_"imag"__coeff_60', 'ACC__fft_coefficient__attr_"real"__coeff_63',
    'ACC__fft_coefficient__attr_"angle"__coeff_26', 'ACC__fft_coefficient__attr_"angle"__coeff_81',
    'ACC__number_cwt_peaks__n_1', 'ACC__fft_coefficient__attr_"imag"__coeff_72',
    'ACC__number_cwt_peaks__n_5', 'ACC__fft_coefficient__attr_"real"__coeff_78',
    'ACC__fft_coefficient__attr_"abs"__coeff_97', 'ACC__partial_autocorrelation__lag_9',
    'ACC__value_count__value_0', 'ACC__fft_coefficient__attr_"real"__coeff_38',
    'ACC__energy_ratio_by_chunks__num_segments_10__segment_focus_9', 'ACC__fft_coefficient__attr_"imag"__coeff_24',
    'ACC__fft_coefficient__attr_"real"__coeff_64', 'ACC__fft_coefficient__attr_"real"__coeff_97',
    'ACC__fft_coefficient__attr_"angle"__coeff_78', 'ACC__fft_coefficient__attr_"real"__coeff_88',
    'ACC__agg_linear_trend__attr_"stderr"__chunk_len_50__f_agg_"max"', 'ACC__mean_second_derivative_central',
    'ACC__count_above_mean', 'ACC__agg_linear_trend__attr_"stderr"__chunk_len_10__f_agg_"mean"',
    'ACC__fft_coefficient__attr_"angle"__coeff_87', 'Raw Score HitSEBlock',
    'ACC__fft_coefficient__attr_"abs"__coeff_35', 'ACC__change_quantiles__f_agg_"var"__isabs_True__qh_1.0__ql_0.6',
    'ACC__lempel_ziv_complexity__bins_5', 'ACC__range_count__max_1000000000000.0__min_0',
    'ACC__first_location_of_maximum', 'ACC__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.8',
    'ACC__fft_coefficient__attr_"imag"__coeff_42', 'ACC__fft_coefficient__attr_"real"__coeff_29',
    'ACC__fft_coefficient__attr_"real"__coeff_13', 'ACC__number_peaks__n_10',
    'ACC__fft_coefficient__attr_"real"__coeff_3', 'ACC__partial_autocorrelation__lag_2',
    'ACC__fft_coefficient__attr_"imag"__coeff_43', 'ACC__permutation_entropy__dimension_3__tau_1',
    'ACC__fourier_entropy__bins_100', 'ACC__fft_coefficient__attr_"real"__coeff_96',
    'ACC__fft_coefficient__attr_"abs"__coeff_42', 'ACC__fft_coefficient__attr_"angle"__coeff_41',
    'ACC__fft_coefficient__attr_"real"__coeff_71'
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