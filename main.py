import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
)
from xgboost import XGBClassifier

# Set up directory for outputs
os.makedirs('output', exist_ok=True)

print("Loading data...")
df = pd.read_csv("application_merged_final.csv")
print(f"Data loaded with shape: {df.shape}")

# Save original features for reference
original_features = df.columns.tolist()
with open('output/original_features.txt', 'w') as f:
    f.write('\n'.join(original_features))

# Drop ID column
df.drop(columns=["SK_ID_CURR"], inplace=True)

# === Data Cleaning ===
print("\n=== Data Cleaning ===")

# Check for missing values in TARGET variable
missing_target = df["TARGET"].isnull().sum()
if missing_target > 0:
    print(f"WARNING: Found {missing_target} missing values in TARGET variable. Dropping these rows.")
    df = df.dropna(subset=["TARGET"])

# Drop columns with >40% missing
missing = df.isnull().sum() / len(df)
high_missing_cols = missing[missing > 0.4].index.tolist()
print(f"Dropping {len(high_missing_cols)} columns with >40% missing values")
df.drop(columns=high_missing_cols, inplace=True)

# Handle numeric missing values
num_cols = df.select_dtypes(include=["float64", "int64"]).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Handle categorical missing values
cat_cols = df.select_dtypes(include="object").columns
for col in cat_cols:
    df[col] = df[col].fillna('Unknown')

# Clean known anomalies
df = df[df["CODE_GENDER"] != "XNA"]

# === Feature Engineering ===
print("\n=== Feature Engineering ===")
# Label Encoding for binary categorical columns
le = LabelEncoder()
binary_cats = [col for col in cat_cols if df[col].nunique() == 2]
for col in binary_cats:
    df[col] = le.fit_transform(df[col])

# One-hot Encoding for other categorical columns
multi_cats = [col for col in cat_cols if col not in binary_cats]
df = pd.get_dummies(df, columns=multi_cats, drop_first=True)

# Check for any remaining NaN values
if df.isnull().sum().sum() > 0:
    print(f"Filling {df.isnull().sum().sum()} remaining missing values")
    df = df.fillna(df.median())

# Create important financial ratios
df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan).fillna(df["AMT_INCOME_TOTAL"].median())
df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan).fillna(df["AMT_INCOME_TOTAL"].median())
df["PAYMENT_RATE"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"].replace(0, np.nan).fillna(df["AMT_CREDIT"].median())
df["AGE_YEARS"] = abs(df["DAYS_BIRTH"] / 365)
df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS'].replace(0, np.nan).fillna(1)


# Handle any infinite values that might have been created
for col in df.columns:
    if df[col].dtype == 'float64':
        if not np.isfinite(df[col]).all():
            print(f"Fixing infinite values in {col}")
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].fillna(df[col].median())

# Split features and target
print("\n=== Preparing Features and Target ===")
y = df["TARGET"]
X = df.drop(columns=["TARGET"])

# Make sure X has no constant columns
constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
if constant_cols:
    print(f"Dropping {len(constant_cols)} constant columns")
    X = X.drop(columns=constant_cols)

# Make sure X has no duplicate columns
X = X.loc[:, ~X.columns.duplicated()]
print(f"Feature set shape after cleaning: {X.shape}")

# Final check for NaN values
assert X.isnull().sum().sum() == 0, "Features still contain NaN values"
assert y.isnull().sum() == 0, "Target still contains NaN values"

# Apply standard scaling
print("Applying standardization...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save feature list for model interpretation
with open('output/engineered_features.txt', 'w') as f:
    f.write('\n'.join(X.columns))
print("✅ Engineered feature list saved to 'output/engineered_features.txt'")

# Train-test split
X_train_raw, X_test, y_train_raw, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# === Handle Class Imbalance ===
print("\n=== Handling Class Imbalance ===")
print(f"Original training distribution: {pd.Series(y_train_raw).value_counts()}")

# Apply SMOTE for class balancing
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42,)
X_train, y_train = smote.fit_resample(X_train_raw, y_train_raw)
print(f"Resampled training distribution: {pd.Series(y_train).value_counts()}")

# === Model Training and Evaluation ===
print("\n=== Training Models ===")

# Function to evaluate model performance
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    print(f"\n=== {model_name} ===")
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate AUC
    auc = roc_auc_score(y_test, y_proba)
    print(f"AUC: {auc:.4f}")
    
    # Classification report
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'output/cm_{model_name.lower()}.png')
    plt.close()
    
    # ROC Curve
    plt.figure(figsize=(8, 6))
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title(f"ROC Curve - {model_name}")
    plt.savefig(f'output/roc_{model_name.lower()}.png')
    plt.close()
    
    # Feature Importance for Random Forest
    if hasattr(model, 'feature_importances_'):
        importances = pd.Series(model.feature_importances_, index=X.columns)
        top_features = importances.sort_values(ascending=False).head(20)
        
        plt.figure(figsize=(12, 10))
        top_features.plot(kind='barh')
        plt.title(f"Top 20 Important Features - {model_name}")
        plt.tight_layout()
        plt.gca().invert_yaxis()
        plt.savefig(f'output/feature_importance_{model_name.lower()}.png')
        plt.close()
        
    # Feature Importance for Logistic Regression
    if isinstance(model, LogisticRegression):
        coefs = pd.Series(model.coef_[0], index=X.columns)
        top_pos = coefs.sort_values(ascending=False).head(10)
        top_neg = coefs.sort_values(ascending=True).head(10)

        plt.figure(figsize=(10, 5))
        top_pos.plot(kind='barh', color='green', label='Gagal Bayar ↑')
        plt.title(f"Top Positive Coefficients - {model_name}")
        plt.tight_layout()
        plt.gca().invert_yaxis()
        plt.savefig(f'output/logreg_positive_{model_name.lower()}.png')
        plt.close()

        plt.figure(figsize=(10, 5))
        top_neg.plot(kind='barh', color='red', label='Gagal Bayar ↓')
        plt.title(f"Top Negative Coefficients - {model_name}")
        plt.tight_layout()
        plt.gca().invert_yaxis()
        plt.savefig(f'output/logreg_negative_{model_name.lower()}.png')
        plt.close()

    
    return auc, model

# Train and evaluate models
models_with_smote = {
    'LogisticRegression_SMOTE': LogisticRegression(max_iter=1000, random_state=42),
    'RandomForest_SMOTE': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost_SMOTE': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}
models_no_smote = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}
results = {}
for model_name, model in models_with_smote.items():
    auc, trained_model = evaluate_model(model, X_train, y_train, X_test, y_test, model_name)
    results[model_name] = auc
for model_name, model in models_no_smote.items():
    auc, trained_model = evaluate_model(model, X_train_raw, y_train_raw, X_test, y_test, model_name)
    results[model_name] = auc

# Print final results
print("\n=== Model Comparison ===")
for model_name, auc in results.items():
    print(f"{model_name}: AUC = {auc:.4f}")

print("\n=== Credit Scoring Model Training Complete ===")
print(f"All outputs saved to the 'output' directory.")

pd.DataFrame.from_dict(results, orient='index', columns=['AUC']).to_csv('output/model_auc_scores.csv')
print("✅ AUC results saved to output/model_auc_scores.csv")

from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_model_extended(model, X_train, y_train, X_test, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "Model": model_name,
        "AUC": roc_auc_score(y_test, y_proba),
        "Recall_1": recall_score(y_test, y_pred),
        "Precision_1": precision_score(y_test, y_pred),
        "F1_1": f1_score(y_test, y_pred)
    }
comparison_results = []
for model_name, model in models_with_smote.items():
    res = evaluate_model_extended(model, X_train, y_train, X_test, y_test, model_name)
    comparison_results.append(res)

for model_name, model in models_no_smote.items():
    res = evaluate_model_extended(model, X_train_raw, y_train_raw, X_test, y_test, model_name)
    comparison_results.append(res)

# Simpan sebagai tabel perbandingan
df_compare = pd.DataFrame(comparison_results)
df_compare.to_csv("output/model_comparison_full.csv", index=False)
print("✅ Model comparison saved to output/model_comparison_full.csv")
