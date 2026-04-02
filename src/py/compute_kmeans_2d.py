import json
import os
import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

# ==========================================
# 1. CONFIGURATION
# ==========================================
DATASETS = ["wine", "iris", "user_knowledge"]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def compute_kmeans_and_align(X, original_labels, k_clusters, unique_true_labels):
    """
    Computes K-Means and aligns cluster IDs to the original labels
    to minimize mismatches using the Hungarian algorithm.
    Handles arbitrary label types (strings or integers).
    """
    kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_
    
    # Create cost matrix based on intersection size (negative for max assignment)
    unique_pred = np.unique(clusters)
    cost_matrix = np.zeros((k_clusters, len(unique_pred)))
    
    for i, t_val in enumerate(unique_true_labels):
        for j, p_val in enumerate(unique_pred):
            intersection = np.sum((original_labels == t_val) & (clusters == p_val))
            cost_matrix[i, j] = -intersection
            
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Create mapping from K-Means cluster ID -> Original Label Space
    mapping = {unique_pred[c]: unique_true_labels[r] for r, c in zip(row_ind, col_ind)}
    
    # Apply the mapping so K-Means clusters share the exact same names as original labels
    aligned_clusters = np.array([mapping[c] for c in clusters])
    
    # Reorder centroids according to the aligned clusters using a dictionary
    aligned_centroids = {}
    for kmeans_cluster, orig_label in mapping.items():
        aligned_centroids[orig_label] = centroids[kmeans_cluster]
        
    # Calculate distances from points to their assigned cluster centroid
    distances = []
    for i in range(len(X)):
        c_coord = aligned_centroids[aligned_clusters[i]]
        distances.append(np.linalg.norm(X[i] - c_coord))
        
    return aligned_clusters, distances, aligned_centroids

def process_dataset(dataset_name):
    print(f"\n--- Processing 2D K-Means for: {dataset_name.upper()} ---")
    
    # Setup dynamic I/O paths
    input_file = os.path.abspath(os.path.join(SCRIPT_DIR, f'../json/{dataset_name}/step2_final_data.json'))
    output_file = os.path.abspath(os.path.join(SCRIPT_DIR, f'../json/{dataset_name}/kmeans_2d_results.json'))
    
    if not os.path.exists(input_file):
        print(f"Skipping {dataset_name}: Input not found at {input_file}")
        return

    print(f"Loading data from {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)
        
    points = data['points']
    
    pca_coords = np.array([[p['pca_x'], p['pca_y']] for p in points])
    mds_coords = np.array([[p['mds_x'], p['mds_y']] for p in points])
    
    # Convert labels to strings universally to support categorical classes (e.g. "setosa")
    original_labels = np.array([str(p['label']) for p in points])
    ids = [p['id'] for p in points]
    
    # Determine K dynamically
    unique_true_labels = np.unique(original_labels)
    k_clusters = len(unique_true_labels)
    
    print(f"Computing K-Means for PCA (K={k_clusters})...")
    pca_clusters, pca_distances, pca_centroids = compute_kmeans_and_align(pca_coords, original_labels, k_clusters, unique_true_labels)
    
    print(f"Computing K-Means for MDS (K={k_clusters})...")
    mds_clusters, mds_distances, mds_centroids = compute_kmeans_and_align(mds_coords, original_labels, k_clusters, unique_true_labels)
    
    # Structure output
    results = {
        "k": k_clusters,
        "pca_centroids": [{"cluster": lbl, "x": float(pca_centroids[lbl][0]), "y": float(pca_centroids[lbl][1])} for lbl in unique_true_labels if lbl in pca_centroids],
        "mds_centroids": [{"cluster": lbl, "x": float(mds_centroids[lbl][0]), "y": float(mds_centroids[lbl][1])} for lbl in unique_true_labels if lbl in mds_centroids],
        "points": []
    }
    
    for i in range(len(ids)):
        results["points"].append({
            "id": ids[i],
            "original_label": str(original_labels[i]),
            "pca_kmeans_cluster": str(pca_clusters[i]),
            "pca_dist_to_centroid": float(pca_distances[i]),
            "pca_is_anomaly": bool(original_labels[i] != pca_clusters[i]),
            "mds_kmeans_cluster": str(mds_clusters[i]),
            "mds_dist_to_centroid": float(mds_distances[i]),
            "mds_is_anomaly": bool(original_labels[i] != mds_clusters[i]),
            "pca_x": float(pca_coords[i][0]),
            "pca_y": float(pca_coords[i][1]),
            "mds_x": float(mds_coords[i][0]),
            "mds_y": float(mds_coords[i][1])
        })
        
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"Export completed in {output_file}")

def main():
    for ds in DATASETS:
        process_dataset(ds)

if __name__ == '__main__':
    main()