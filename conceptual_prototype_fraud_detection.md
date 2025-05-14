# Conceptual Prototype: XGBoost for Fraud Detection

This document outlines the conceptual structure of a baseline fraud detection prototype using Python and XGBoost. This serves as the starting point before scaling.

## 1. Project Goal
To detect fraudulent financial transactions.

## 2. Technology Stack
- Python 3.x
- pandas: For data manipulation and loading.
- scikit-learn: For preprocessing (e.g., scaling, encoding), splitting data, and evaluation metrics.
- XGBoost: For the classification model.
- Matplotlib/Seaborn (Optional): For basic visualizations.

## 3. Data
- **Source:** Assumed to be a CSV file containing transaction records.
- **Features (Example):** `transaction_id`, `timestamp`, `transaction_amount`, `user_id`, `merchant_id`, `ip_address`, `device_id`, `location_country`, `is_fraud` (target variable).
- **Size:** Initially designed for a dataset of 1-10 million records.

## 4. Prototype Workflow

### 4.1. Data Loading and Initial Exploration
```python
import pandas as pd

# Load data
df = pd.read_csv("transactions_sample.csv")

# Initial exploration
print(df.head())
print(df.info())
print(df.describe())
print(df["is_fraud"].value_counts(normalize=True))
```

### 4.2. Feature Engineering and Preprocessing
- **Handle Missing Values:** Impute or remove.
- **Convert Timestamps:** Extract features like hour of day, day of week.
- **Categorical Feature Encoding:** One-hot encoding or label encoding for features like `merchant_id`, `location_country`.
- **Numerical Feature Scaling:** StandardScaler or MinMaxScaler for features like `transaction_amount`.
- **Feature Selection (Optional):** Based on correlation or feature importance from a preliminary model.

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Example: Separate features and target
X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

# Identify categorical and numerical features (example names)
numerical_features = ["transaction_amount"]
categorical_features = ["merchant_id", "location_country"]

# Create preprocessing pipelines for numerical and categorical features
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))])

# Create a preprocessor object using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

### 4.3. Model Training
- Use XGBoost (XGBClassifier).
- Handle class imbalance (common in fraud detection) using techniques like `scale_pos_weight` in XGBoost or over/undersampling techniques (e.g., SMOTE) if applied carefully.

```python
import xgboost as xgb

# Define the model pipeline
model = Pipeline(steps=[("preprocessor", preprocessor),
                        ("classifier", xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", scale_pos_weight= (y_train == 0).sum() / (y_train == 1).sum() ))]) # Example for handling imbalance

# Train the model
model.fit(X_train, y_train)
```

### 4.4. Model Evaluation
- Use appropriate metrics for imbalanced datasets: Precision, Recall, F1-score, Area Under Precision-Recall Curve (AUPRC), Confusion Matrix.
- Avoid relying solely on accuracy.

```python
from sklearn.metrics import classification_report, precision_recall_curve, auc, roc_auc_score

# Make predictions
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

# Print classification report
print(classification_report(y_test, y_pred))

# Calculate AUPRC
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
auprc = auc(recall, precision)
print(f"AUPRC: {auprc:.4f}")

# Calculate AUROC (for reference)
auroc = roc_auc_score(y_test, y_pred_proba)
print(f"AUROC: {auroc:.4f}")
```

### 4.5. (Conceptual) Saving and Loading Model
```python
import joblib

# Save model
joblib.dump(model, "fraud_detection_xgb_prototype.joblib")

# Load model (example)
# loaded_model = joblib.load("fraud_detection_xgb_prototype.joblib")
```

## 5. Potential Bottlenecks for Scaling (to be addressed in next phase)
- **Data Loading:** `pandas.read_csv` is not suitable for billions of records.
- **In-Memory Processing:** Pandas DataFrames and scikit-learn preprocessing steps typically require data to fit in memory.
- **Single-Node Training:** XGBoost training on a single machine will be too slow or impossible for very large datasets.
- **Feature Engineering:** Complex feature engineering on massive datasets can be computationally intensive.
- **Hyperparameter Tuning:** Grid search or random search can be very time-consuming on large data.

This conceptual prototype forms the basis for the scaling effort, where these bottlenecks will be addressed using distributed computing frameworks like Apache Spark.
