import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np

# Load data
print("Loading data...")
df = pd.read_csv("data.csv")

# Features / target
print("Preparing data...")
X = df.drop(columns=["label"])
y = df["label"]

# Encode labels (human=0, ai=1)
print("Encoding labels...")
le = LabelEncoder()
y_encoded = le.fit_transform(y)

df["label_encoded"] = y_encoded

print(df.drop(columns=["label"]).corrwith(df["label_encoded"]).sort_values(ascending=False))

# Train/test split (80/20)
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Model
print("Initializing Random Forest model...")
rf = RandomForestClassifier(
    n_estimators=100,      # number of trees
    max_depth=15,          # prevent overfitting
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,             # use all available cores
    criterion='gini'
)

# Train
print("Training model...")
rf.fit(X_train, y_train)

# Predict
print("Making predictions...")
y_pred = rf.predict(X_test)

# Evaluate
print("\nResults:")
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Feature Importance
print("\n\nTop 15 Most Important Features:")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(15))

# Plot feature importance
plt.figure(figsize=(12, 6))
top_features = feature_importance.head(15)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Importance')
plt.title('Random Forest - Top 15 Feature Importances')
plt.tight_layout()
plt.savefig('plots/random_forest_feature_importance.png', dpi=100, bbox_inches='tight')
print("\nFeature importance plot saved to: plots/random_forest_feature_importance.png")
