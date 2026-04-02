import os
import json
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment

# --- CONFIGURATION ---
DATASETS = ["wine", "iris", "user_knowledge"]
RANDOM_STATE = 42
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

LABEL_INDEX = {
    "wine": 0,
    "iris": -1,
    "user_knowledge": -1
}

def compute_kmeans_for_dataset(dataset_name):
    print(f"\n--- Processing 13D/ND K-Means for: {dataset_name.upper()} ---")
    
    dataset_path = os.path.join(SCRIPT_DIR, '../../dataset', f'{dataset_name}.csv')
    output_file = os.path.join(SCRIPT_DIR, '../json', dataset_name, 'kmeans_results.json')

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return

    # Load dataset and extract features/labels dynamically
    df = pd.read_csv(dataset_path)
    target_idx = LABEL_INDEX.get(dataset_name, -1)
    
    original_labels = df.iloc[:, target_idx].values
    X = df.drop(df.columns[target_idx], axis=1).values

    # Determine K dynamically based on unique labels present in the dataset
    unique_true = np.unique(original_labels)
    k_clusters = len(unique_true)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=k_clusters, random_state=RANDOM_STATE, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # Hungarian Algorithm Alignment to match predicted clusters with true labels
    unique_pred = np.unique(cluster_labels)
    cm = np.zeros((len(unique_true), len(unique_pred)), dtype=int)
    for i, t_val in enumerate(unique_true):
        for j, p_val in enumerate(unique_pred):
            cm[i, j] = np.sum((original_labels == t_val) & (cluster_labels == p_val))

    rows, cols = linear_sum_assignment(cm.max() - cm)
    mapping = {unique_pred[c]: unique_true[r] for r, c in zip(rows, cols)}
    aligned_labels = np.array([mapping[l] for l in cluster_labels])

    centroids = kmeans.cluster_centers_
    distances = np.linalg.norm(X_scaled - centroids[cluster_labels], axis=1)

    results = []
    for i in range(len(df)):
        results.append({
            "id": i,
            "original_label": str(original_labels[i]), 
            "kmeans_cluster": str(aligned_labels[i]), 
            "dist_to_centroid": float(distances[i]),
            "is_anomaly": bool(str(aligned_labels[i]) != str(original_labels[i]))
        })

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump({"k": k_clusters, "points": results}, f, indent=4)

    print(f"Saved: {output_file}")

if __name__ == "__main__":
    for ds in DATASETS:
        compute_kmeans_for_dataset(ds)