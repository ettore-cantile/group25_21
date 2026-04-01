import os
import json
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Modify the number of clusters (K) here
K_CLUSTERS = 3

RANDOM_STATE = 42
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, '../../dataset/wine.csv')
OUTPUT_FILE = os.path.join(SCRIPT_DIR, '../json/kmeans_results.json')

def compute_kmeans():
    # ==========================================
    # 2. DATA LOADING
    # ==========================================
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at {DATASET_PATH}")
        return

    print(f"Loading dataset from: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)

    # Assuming the first column is the label (Producer) and the rest are features
    # (Consistent with compute_relationship.py)
    original_labels = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values

    # ==========================================
    # 3. STANDARDIZATION & K-MEANS
    # ==========================================
    print("Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Running K-Means with K={K_CLUSTERS}...")
    kmeans = KMeans(n_clusters=K_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # ------------------------------------------
    # LABEL ALIGNMENT (Match KMeans clusters to Original Labels)
    # ------------------------------------------
    # Get unique true labels (e.g., 1, 2, 3) and predicted labels (e.g., 0, 1, 2)
    unique_true = np.unique(original_labels)
    unique_pred = np.unique(cluster_labels)

    # Create a custom confusion matrix matching True Labels to K-Means Clusters
    # This prevents dimension mismatch if label spaces differ (e.g., 1,2,3 vs 0,1,2)
    cm = np.zeros((len(unique_true), len(unique_pred)), dtype=int)
    for i, t_val in enumerate(unique_true):
        for j, p_val in enumerate(unique_pred):
            cm[i, j] = np.sum((original_labels == t_val) & (cluster_labels == p_val))

    # Linear Sum Assignment (Hungarian Algo) to maximize the diagonal matches
    rows, cols = linear_sum_assignment(cm.max() - cm)

    # Create mapping from K-Means cluster ID -> Original Label Space
    mapping = {unique_pred[c]: unique_true[r] for r, c in zip(rows, cols)}
    
    # Apply the mapping so K-Means clusters share the exact same names as original labels
    aligned_labels = np.array([mapping[l] for l in cluster_labels])

    # Calculate distances to centroids (High-Dimensional Anomaly Detection)
    centroids = kmeans.cluster_centers_
    distances = np.linalg.norm(X_scaled - centroids[cluster_labels], axis=1)

    # ==========================================
    # 4. JSON EXPORT
    # ==========================================
    results = []
    for i in range(len(df)):
        results.append({
            "id": i,
            "original_label": int(original_labels[i]),
            "kmeans_cluster": int(aligned_labels[i]), # Export aligned label
            "dist_to_centroid": float(distances[i]),
            "is_anomaly": bool(aligned_labels[i] != original_labels[i])
        })

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump({"k": K_CLUSTERS, "points": results}, f, indent=4)

    print(f"Analysis completed. Results saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    compute_kmeans()