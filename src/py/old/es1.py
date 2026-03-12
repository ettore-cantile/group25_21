import json
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics import euclidean_distances

# ==========================================
# 1. USER CONFIGURATION
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(SCRIPT_DIR, '../dataset/wine_data.json')
CSV_FILENAME = os.path.join(SCRIPT_DIR, '../dataset/wine.csv')
TARGET_COLUMN = 'Producer'

# Set the number of neighbors to consider (K)
# A good default value is usually between 5 and 15 depending on the dataset
K_NEIGHBORS = 10

# ==========================================
# 2. DATA LOADING & PREPARATION
# ==========================================
print(f"Loading {CSV_FILENAME}...")
try:
    df = pd.read_csv(CSV_FILENAME)
except FileNotFoundError:
    print("ERROR: File not found.")
    exit()

df = df.dropna()

if TARGET_COLUMN and TARGET_COLUMN in df.columns:
    class_names = df[TARGET_COLUMN].astype(str).values
    class_names = ["Class " + c for c in class_names]
    data_df = df.drop(columns=[TARGET_COLUMN])
else:
    class_names = ["Item " + str(i) for i in range(len(df))]
    data_df = df

X_original = data_df.select_dtypes(include=[np.number])
X = X_original.values
print(f"Data shape: {X.shape}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# 3. DIMENSIONALITY REDUCTION
# ==========================================
print("Computing PCA...")
pca = PCA(n_components=2, random_state=0)
X_pca = pca.fit_transform(X_scaled)
X_pca[:, 0] = -X_pca[:, 0] # Axis flipping fix

print("Computing MDS...")
mds = MDS(n_components=2, dissimilarity='euclidean', random_state=42, normalized_stress='auto', n_init=4)
X_mds = mds.fit_transform(X_scaled)

# ==========================================
# 4. DISTORTION METRICS (Rank-based KNN)
# ==========================================
print("Computing advanced distortion scores (FP & FN weights)...")

def calculate_knn_distortion(D_high, D_low, K):
    N = D_high.shape[0]
    
    # Fill diagonal with infinity to avoid counting a point as its own neighbor
    np.fill_diagonal(D_high, np.inf)
    np.fill_diagonal(D_low, np.inf)

    # Calculate ranks (rankings). argsort(argsort) returns the rank of each element
    R_high = np.argsort(np.argsort(D_high, axis=1), axis=1)
    R_low = np.argsort(np.argsort(D_low, axis=1), axis=1)

    score_fp = np.zeros(N)
    score_fn = np.zeros(N)

    for i in range(N):
        # Indices of K-neighbors in high-dimensional space (Original)
        U_i = set(np.where(R_high[i] < K)[0])
        # Indices of K-neighbors in low-dimensional space (2D)
        V_i = set(np.where(R_low[i] < K)[0])

        # False Positives: Points that are close in 2D but were not originally
        fp_indices = list(V_i - U_i)
        if fp_indices:
            # Weight is how far they were originally relative to K
            score_fp[i] = np.sum(R_high[i, fp_indices] - K)

        # False Negatives: Points that were close originally but are far in 2D
        fn_indices = list(U_i - V_i)
        if fn_indices:
            # Weight is how far they were moved away in 2D relative to K
            score_fn[i] = np.sum(R_low[i, fn_indices] - K)

    # Divergent index: Positive = False Positives dominant, Negative = False Negatives dominant
    point_scores = score_fp - score_fn
    
    # Global Algorithm Score (Sum of all errors)
    global_score = np.sum(score_fp) + np.sum(score_fn)
    
    return point_scores, global_score

# Calculate Euclidean distances
D_high = euclidean_distances(X_scaled)
D_pca = euclidean_distances(X_pca)
D_mds = euclidean_distances(X_mds)

# Calculate scores
score_pca, global_pca = calculate_knn_distortion(D_high.copy(), D_pca.copy(), K=K_NEIGHBORS)
score_mds, global_mds = calculate_knn_distortion(D_high.copy(), D_mds.copy(), K=K_NEIGHBORS)

print("\n--- QUALITY ANALYSIS (INITIAL ASSESSMENT) ---")
print(f"PCA Global Error Score: {global_pca:.0f}")
print(f"MDS Global Error Score: {global_mds:.0f}")
if global_pca < global_mds:
    print("-> PCA performed better at preserving local neighborhoods.")
else:
    print("-> MDS performed better at preserving local neighborhoods.")
print("---------------------------------------------\n")

# ==========================================
# 5. EXPORT
# ==========================================
data_export = []
for i in range(len(X)):
    data_export.append({
        "id": i,
        "pca_x": float(X_pca[i, 0]),
        "pca_y": float(X_pca[i, 1]),
        "mds_x": float(X_mds[i, 0]),
        "mds_y": float(X_mds[i, 1]),
        "score_pca": float(score_pca[i]), # Use for divergent Color Scale (Blue > 0, Red < 0)
        "score_mds": float(score_mds[i]), # Use for divergent Color Scale
        "class_name": class_names[i]
    })

# Ensure the destination folder exists
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(OUTPUT_FILE, 'w') as f:
    json.dump(data_export, f)

print(f"Done! Updated JSON saved to: {OUTPUT_FILE}")