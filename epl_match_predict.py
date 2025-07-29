import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Loading historical dataset...")
df = pd.read_csv("final_dataset.csv")
print(f"Shape of historical dataset: {df.shape}")

# Create a clean label from pre-match info only
df['ResultLabel'] = df.apply(lambda row: 0 if row['HTP'] > row['ATP'] else (1 if row['HTP'] < row['ATP'] else 2), axis=1)

# Drop leakage and irrelevant columns
leakage_cols = ['FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR']
df.drop(columns=[col for col in leakage_cols if col in df.columns], inplace=True, errors='ignore')

# Drop team/date columns
df.drop(columns=[col for col in ['Date', 'HomeTeam', 'AwayTeam'] if col in df.columns], inplace=True, errors='ignore')

# Drop features that directly determine the label
leakage_logic_cols = ['HTP', 'ATP', 'DiffPts']
df.drop(columns=[col for col in leakage_logic_cols if col in df.columns], inplace=True, errors='ignore')

# Drop any non-numeric fields
df = df.select_dtypes(include=[np.number])

# Check label
if 'ResultLabel' not in df.columns:
    raise ValueError("Missing label!")

# Separate features and labels
X = df.drop(columns=['ResultLabel'])
y = df['ResultLabel']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest
print("\nTraining Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\nRandom Forest Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

# Top features
importances = rf.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
print("\nTop 10 Important Features:")
print(importance_df.sort_values(by='Importance', ascending=False).head(10))

# MLP
print("\nTraining MLP Classifier...")
mlp = MLPClassifier(hidden_layer_sizes=(64,), max_iter=200, random_state=42)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)

print("\nMLP Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred_mlp))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_mlp))
print("Classification Report:\n", classification_report(y_test, y_pred_mlp))