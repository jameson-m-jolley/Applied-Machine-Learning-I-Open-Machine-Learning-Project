import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

# 1. Ensure the /plots directory exists
output_dir = "plots"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

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
y = le.fit_transform(y)

df["label_encoded"] = le.fit_transform(df["label"])

print(df.drop(columns=["label"]).corrwith(df["label_encoded"]).sort_values(ascending=False))


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("preforming pca")
DR = PCA(n_components=3)
# IMPORTANT: use fit_transform() to get 2D features back
X_pca = DR.fit_transform(X_scaled, y) 

# 4. Create the 3D Plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Map labels to colors
colors = ['#1f77b4' if label == 0 else '#d62728' for label in y]

# Scatter plot
ax.scatter(
    X_pca[:, 0], 
    X_pca[:, 1], 
    X_pca[:, 2], 
    c=colors, 
    alpha=0.6, 
    edgecolors='w', 
    s=40
)

# Titles and Axis Labels
ax.set_title("3D PCA Projection: Human vs AI", pad=20)
ax.set_xlabel(f"PC1 ({DR.explained_variance_ratio_[0]:.1%} Var)")
ax.set_ylabel(f"PC2 ({DR.explained_variance_ratio_[1]:.1%} Var)")
ax.set_zlabel(f"PC3 ({DR.explained_variance_ratio_[2]:.1%} Var)")

# Custom Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Human', markerfacecolor='#1f77b4', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='AI', markerfacecolor='#d62728', markersize=10)
]
ax.legend(handles=legend_elements)

# 5. Save and Show
save_path = os.path.join(output_dir, "pca_3d_separation.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Plot saved successfully to: {save_path}")

plt.show()