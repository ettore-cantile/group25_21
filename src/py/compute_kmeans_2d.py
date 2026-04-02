import json
import os
import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

# ==========================================
# 1. CONFIGURATION
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.abspath(os.path.join(SCRIPT_DIR, '../json/step2_final_data.json'))
OUTPUT_FILE = os.path.abspath(os.path.join(SCRIPT_DIR, '../json/kmeans_2d_results.json'))

def compute_kmeans_and_align(X, original_labels, k=3):
    """
    Computes K-Means and aligns cluster IDs to the original labels
    to minimize mismatches using the Hungarian algorithm.
    """
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_
    
    # Create cost matrix based on intersection size (negative for max assignment)
    cost_matrix = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            intersection = np.sum((original_labels == i) & (clusters == j))
            cost_matrix[i, j] = -intersection
            
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    mapping = {kmeans_cluster: orig_label for orig_label, kmeans_cluster in zip(row_ind, col_ind)}
    
    aligned_clusters = np.array([mapping[c] for c in clusters])
    
    # Reorder centroids according to the aligned clusters
    aligned_centroids = np.zeros_like(centroids)
    for kmeans_cluster, orig_label in mapping.items():
        aligned_centroids[orig_label] = centroids[kmeans_cluster]
        
    distances = np.linalg.norm(X - aligned_centroids[aligned_clusters], axis=1)
    
    return aligned_clusters, distances, aligned_centroids

def main():
    print(f"Loading data from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)
        
    points = data['points']
    
    pca_coords = np.array([[p['pca_x'], p['pca_y']] for p in points])
    mds_coords = np.array([[p['mds_x'], p['mds_y']] for p in points])
    original_labels = np.array([int(p['label']) for p in points])
    ids = [p['id'] for p in points]
    
    k = 3
    print("Computing K-Means for PCA...")
    pca_clusters, pca_distances, pca_centroids = compute_kmeans_and_align(pca_coords, original_labels, k=k)
    
    print("Computing K-Means for MDS...")
    mds_clusters, mds_distances, mds_centroids = compute_kmeans_and_align(mds_coords, original_labels, k=k)
    
    results = {
        "k": k,
        "pca_centroids": [{"cluster": i, "x": float(pca_centroids[i][0]), "y": float(pca_centroids[i][1])} for i in range(k)],
        "mds_centroids": [{"cluster": i, "x": float(mds_centroids[i][0]), "y": float(mds_centroids[i][1])} for i in range(k)],
        "points": []
    }
    
    for i in range(len(ids)):
        results["points"].append({
            "id": ids[i],
            "original_label": int(original_labels[i]),
            "pca_kmeans_cluster": int(pca_clusters[i]),
            "pca_dist_to_centroid": float(pca_distances[i]),
            "pca_is_anomaly": bool(original_labels[i] != pca_clusters[i]),
            "mds_kmeans_cluster": int(mds_clusters[i]),
            "mds_dist_to_centroid": float(mds_distances[i]),
            "mds_is_anomaly": bool(original_labels[i] != mds_clusters[i]),
            "pca_x": float(pca_coords[i][0]),
            "pca_y": float(pca_coords[i][1]),
            "mds_x": float(mds_coords[i][0]),
            "mds_y": float(mds_coords[i][1])
        })
        
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"Export completed in {OUTPUT_FILE}")

if __name__ == '__main__':
    main()