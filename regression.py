import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
# Load data



#this is for the demo
def get_modle():
    df = pd.read_csv("data.csv")
    X = df.drop(columns=["label"])
    y = df["label"]
    le = LabelEncoder()
    y = le.fit_transform(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    df["label_encoded"] = le.fit_transform(df["label"])
    X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    DR = PCA(n_components=35, svd_solver='arpack')
    X_train_lda = DR.fit_transform(X_train, y_train) 
    clf = LogisticRegression(random_state=0)
    print("Training model...")
    clf.fit(X_train_lda, y_train)
    def predict_test_fn(x):
        #print(dict(zip(le.classes_, le.transform(le.classes_))))
        s = scaler.transform(x)
        c = DR.transform(s)
        y = clf.predict_proba(c)
        return y

    return predict_test_fn
#-------------------------- 
print("Loading data...")
df = pd.read_csv("data.csv")

# Features / target
print("Preparing data...")
X = df.drop(columns=["label"])
y = df["label"]

# Encode labels (human=0, ai=1)
print("Encoding labels...")
le = LabelEncoder()
y = le.fit_transform(y)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


df["label_encoded"] = le.fit_transform(df["label"])

print(df.drop(columns=["label"]).corrwith(df["label_encoded"]).sort_values(ascending=False))


# Train/test split (80/20)
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print("preforming lda")
DR = PCA(n_components=35, svd_solver='arpack')
# IMPORTANT: use fit_transform() to get 2D features back
X_train_lda = DR.fit_transform(X_train, y_train) 
X_test_lda = DR.transform(X_test)

# Model
print("Initializing model...")
clf = LogisticRegression(random_state=0)

# Train
print("Training model...")
clf.fit(X_train_lda, y_train)

# Predict
print("Making predictions...")
y_pred = clf.predict(X_test_lda)

# Evaluate
print("\nResults:")
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

def predict_test_fn(x):
    s = scaler.transform(x)
    c = DR.transform(s)
    y = clf.predict(c)

from timer import time_and_recordFN
for i in range(20):
    time_and_recordFN(predict_test_fn,"Logistic_regression")