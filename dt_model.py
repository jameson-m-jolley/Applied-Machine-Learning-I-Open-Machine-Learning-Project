
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

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
# y = le.fit_transform(y)

df["label_encoded"] = le.fit_transform(df["label"])

print(df.drop(columns=["label"]).corrwith(df["label_encoded"]).sort_values(ascending=False))

# Train/test split (80/20)
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Model
print("Initializing model...")
clf = DecisionTreeClassifier(
    criterion="gini",      # or "entropy"
    max_depth=10,          # prevent overfitting
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

# Train
print("Training model...")
clf.fit(X_train, y_train)

# Predict
print("Making predictions...")
y_pred = clf.predict(X_test)

# Evaluate
print("\nResults:")
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

from timer import time_and_recordFN
for i in range(5):
    time_and_recordFN(clf.predict,"DT")

