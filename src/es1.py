import json
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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

# Separate Target from Features
if TARGET_COLUMN and TARGET_COLUMN in df.columns:
    class_names = df[TARGET_COLUMN].astype(str).values
    # Prefix for better tooltip readability
    class_names = ["Class " + c for c in class_names]
    data_df = df.drop(columns=[TARGET_COLUMN])
else:
    class_names = ["Item " + str(i) for i in range(len(df))]
    data_df = df

# Select numerical features only
X_original = data_df.select_dtypes(include=[np.number])
X = X_original.values
print(f"Data shape: {X.shape}")

# StandardScaler (Important for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# 3. DIMENSIONALITY REDUCTION
# ==========================================
print("Computing PCA...")
# Using random_state=0 for consistency
pca = PCA(n_components=2, random_state=0)
X_pca = pca.fit_transform(X_scaled)

# --- AXIS FLIPPING FIX ---
# If the graph looks mirrored horizontally (left/right) compared to the professor's:
# We invert the X-axis (Component 0).
print("Applying manual axis flip to match visual orientation...")
X_pca[:, 0] = -X_pca[:, 0] 

# If it looks mirrored vertically (upside down), uncomment the line below:
# X_pca[:, 1] = -X_pca[:, 1]

print("Computing MDS...")
mds = MDS(n_components=2, dissimilarity='euclidean', random_state=42, normalized_stress='auto', n_init=4)
X_mds = mds.fit_transform(X_scaled)

# ==========================================
# 4. DISTORTION METRICS
# ==========================================
print("Computing distortion scores...")

# 1. High-Dimensional Distances
D_high = euclidean_distances(X_scaled)

# 2. Low-Dimensional Distances
D_pca = euclidean_distances(X_pca)
D_mds = euclidean_distances(X_mds)

# 3. Normalize Distances to [0, 1] for Comparison
# We use a global normalizer to keep scale proportions
min_max = MinMaxScaler()
D_high_norm = min_max.fit_transform(D_high)
D_pca_norm = min_max.fit_transform(D_pca)
D_mds_norm = min_max.fit_transform(D_mds)

# 4. Calculate Error (LowDim - HighDim)
# Negative = Compressed (Blue), Positive = Stretched (Red)
E_pca = D_pca_norm - D_high_norm
E_mds = D_mds_norm - D_high_norm

score_pca = np.mean(E_pca, axis=1)
score_mds = np.mean(E_mds, axis=1)

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
        "score_pca": float(score_pca[i]),
        "score_mds": float(score_mds[i]),
        "class_name": class_names[i]
    })

with open(OUTPUT_FILE, 'w') as f:
    json.dump(data_export, f)

print(f"Done! Updated JSON saved to: {OUTPUT_FILE}")