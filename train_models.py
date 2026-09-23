import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def train_and_evaluate_models(data_path):
    df = pd.read_csv(data_path)

    # Drop raw timestamp and split target
    X = df.drop(columns=['timestamp', 'is_peak'])
    y = df['is_peak']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Define feature groups
    numeric_features = ['baseline_volume', 'travel_delay_index', 'hour_of_day', 'day_of_week', 'is_weekend', 'month']
    categorical_features = [c for c in ['corridor_id', 'weather', 'road_type', 'location_zone'] if c in X.columns]

    # Preprocessor: scale numerics, one-hot encode categoricals
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='drop'
    )

    # Calibrate tree probabilities so the UI can distinguish confidence levels
    # instead of exposing only hard 0.0/1.0 leaf probabilities.
    estimators = {
        'Decision Tree': CalibratedClassifierCV(
            DecisionTreeClassifier(max_depth=10, random_state=42),
            method='sigmoid',
            cv=5
        ),
        'Random Forest': CalibratedClassifierCV(
            RandomForestClassifier(n_estimators=100, random_state=42),
            method='sigmoid',
            cv=5
        ),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Support Vector Machine': CalibratedClassifierCV(SVC(kernel='rbf', random_state=42), ensemble=False),
        'Neural Network (MLP)': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
    }

    best_f1 = 0.0
    best_pipeline = None
    best_name = ''

    for name, estimator in estimators.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', estimator)])
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = None
        try:
            y_prob = pipeline.predict_proba(X_test)[:, 1]
        except Exception:
            # some classifiers may not support predict_proba; ignore probability metrics
            y_prob = None

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print(f"[{name}] Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1-Score: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_pipeline = pipeline
            best_name = name

    # Save the entire pipeline (preprocessing + model) as a single artifact
    joblib.dump(best_pipeline, 'best_peak_model.pkl')
    print(f"\nBest Model: '{best_name}' saved successfully as 'best_peak_model.pkl'")

    # Save a JSON summary of model metrics for UI display
    metrics = {
        'models': {},
        'best_model': best_name
    }

    # Re-evaluate each trained pipeline quickly to capture metrics
    for name, estimator in estimators.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', estimator)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        metrics['models'][name] = {
            'accuracy': round(float(acc)*100, 3),
            'precision': round(float(prec)*100, 3),
            'recall': round(float(rec)*100, 3),
            'f1_score': round(float(f1)*100, 3)
        }

    import json
    with open('metrics_summary.json', 'w') as fh:
        json.dump(metrics, fh, indent=2)
    print("Saved model metrics summary to 'metrics_summary.json'")


if __name__ == '__main__':
    train_and_evaluate_models('cleaned_traffic_data.csv')