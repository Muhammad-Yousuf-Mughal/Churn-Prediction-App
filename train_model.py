import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def load_data_from_csv(csv_path: str = "WA_Fn-UseC_-Telco-Customer-Churn.csv") -> pd.DataFrame:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_dir, csv_path)
    if not os.path.exists(full_path):
        full_path = csv_path
        
    df = pd.read_csv(full_path)
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    return df.rename(columns={'tenure': 'tenure_months', 'contract': 'contract_type'})

def preprocess_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Financial cleaning
    if 'totalcharges' in df.columns:
        df['total_charges'] = pd.to_numeric(df['totalcharges'].astype(str).str.strip(), errors='coerce')
    elif 'total_charges' in df.columns:
        df['total_charges'] = pd.to_numeric(df['total_charges'].astype(str).str.strip(), errors='coerce')
    else:
        df['total_charges'] = np.nan

    if 'monthlycharges' in df.columns:
        df['monthly_charges'] = df['monthlycharges']

    df['total_charges'] = df['total_charges'].fillna(df['monthly_charges'] * df['tenure_months'])

    # Feature Engineering
    df['charge_per_tenure'] = (df['monthly_charges'] / np.maximum(df['tenure_months'], 1)).round(2)
    df['avg_monthly_spend'] = df['total_charges'] / np.maximum(df['tenure_months'], 1)
    df['charge_ratio'] = df['monthly_charges'] / np.maximum(df['avg_monthly_spend'], 1)
    df['is_new_customer'] = (df['tenure_months'] <= 12).astype(int)
    df['is_long_term_customer'] = (df['tenure_months'] > 48).astype(int)
    
    if 'contract_type' in df.columns:
        df['high_risk_contract'] = ((df['contract_type'] == 'Month-to-month') & (df['monthly_charges'] > 70)).astype(int)

    # Drop non-feature columns
    drop_cols = [c for c in df.columns if 'customer' in c or c in ['totalcharges', 'monthlycharges']]
    df = df.drop(columns=drop_cols, errors='ignore')

    # Encode Target
    if 'churn' in df.columns:
        df['churn'] = df['churn'].map({'Yes': 1, 'No': 0, 1: 1, 0: 0})

    # Auto-detect remaining string/categorical columns and one-hot encode them strictly to integers
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if 'churn' in cat_cols:
        cat_cols.remove('churn')
        
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)

    return df

def run_model_training_and_selection(output_model_path: str = "models/churn_best_model.pkl"):
    print("1. Loading raw dataset...")
    raw_df = load_data_from_csv()

    print("2. Preprocessing & encoding all features...")
    df = preprocess_and_engineer(raw_df)

    X = df.drop(columns=['churn'])
    y = df['churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models_config = {
        'Logistic Regression': (
            Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=1000, random_state=42))]),
            {'clf__C': [0.1, 1.0, 10.0]}
        ),
        'Decision Tree': (
            DecisionTreeClassifier(random_state=42),
            {'max_depth': [4, 6, 8], 'min_samples_split': [2, 5]}
        ),
        'KNN': (
            Pipeline([('scaler', StandardScaler()), ('clf', KNeighborsClassifier())]),
            {'clf__n_neighbors': [5, 11, 15]}
        ),
        'Random Forest': (
            RandomForestClassifier(random_state=42, n_jobs=-1),
            {'n_estimators': [100, 200], 'max_depth': [6, 10]}
        ),
        'Gradient Boosting': (
            GradientBoostingClassifier(random_state=42),
            {'n_estimators': [100, 150], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]}
        ),
        'HistGradientBoosting': (
            HistGradientBoostingClassifier(random_state=42),
            {'max_iter': [100, 150], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]}
        )
    }

    results = []
    trained_estimators = {}

    print("\n3. Executing GridSearchCV Model Tuning (5-Fold Cross Validation)...")
    for name, (model, params) in models_config.items():
        grid = GridSearchCV(model, params, cv=5, scoring='roc_auc', n_jobs=-1)
        grid.fit(X_train, y_train)

        best_m = grid.best_estimator_
        trained_estimators[name] = best_m

        y_pred = best_m.predict(X_test)
        y_proba = best_m.predict_proba(X_test)[:, 1]

        results.append({
            'Model': name,
            'Accuracy': round(accuracy_score(y_test, y_pred), 4),
            'Precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
            'Recall': round(recall_score(y_test, y_pred, zero_division=0), 4),
            'F1-Score': round(f1_score(y_test, y_pred, zero_division=0), 4),
            'ROC-AUC': round(roc_auc_score(y_test, y_proba), 4)
        })

    results_df = pd.DataFrame(results).sort_values(by='ROC-AUC', ascending=False)
    print("\n=== CLASSIFIER EVALUATION COMPARISON MATRIX ===")
    print(results_df.to_string(index=False))

    best_model_name = results_df.iloc[0]['Model']
    best_estimator = trained_estimators[best_model_name]

    abs_output_path = os.path.abspath(output_model_path)
    os.makedirs(os.path.dirname(abs_output_path), exist_ok=True)
    
    payload = {
        'model': best_estimator,
        'feature_names': list(X.columns),
        'model_name': best_model_name
    }
    joblib.dump(payload, abs_output_path)
    print(f"\n🏆 Selected Best Model: {best_model_name}")
    print(f"✅ Serialized PKL artifact successfully saved to: {abs_output_path}")

if __name__ == "__main__":
    run_model_training_and_selection()