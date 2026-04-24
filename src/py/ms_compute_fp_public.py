import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = '/home/matteotwentywings/dataset'

# Column index for labels in each dataset
LABEL_INDEX = {
    "wine": 0,
    "iris": -1,
    "user_knowledge": -1
}

app = Flask(__name__)
CORS(app)

# --- HELPER FUNCTIONS ---

def load_and_project(dataset_name, k_neighbors):
    """Loads the dataset, extracts labels, scales data, and computes projections (PCA, MDS) and distances."""
    dataset_path = os.path.join(DATASET_DIR, f'{dataset_name}.csv')
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset '{dataset_name}' not found at {dataset_path}.")

    df = pd.read_csv(dataset_path)
    target_idx = LABEL_INDEX.get(dataset_name, -1)
    
    y = df.iloc[:, target_idx].values # Extract ground truth labels
    X = df.drop(df.columns[target_idx], axis=1).values
    n_samples = X.shape[0]

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Baseline (HD - High Dimensional space) distances and neighbors
    D_orig = squareform(pdist(X_scaled, metric='euclidean'))
    effective_k = min(k_neighbors, n_samples - 2)
    nn_orig = np.argsort(D_orig, axis=1)[:, 1:effective_k+1]

    # PCA Projection computation
    X_pca = PCA(n_components=2).fit_transform(X_scaled)
    D_pca = squareform(pdist(X_pca, metric='euclidean'))
    nn_pca = np.argsort(D_pca, axis=1)[:, 1:effective_k+1]

    # MDS Projection computation
    X_mds = MDS(n_components=2, normalized_stress='auto', random_state=42).fit_transform(X_scaled)
    D_mds = squareform(pdist(X_mds, metric='euclidean'))
    nn_mds = np.argsort(D_mds, axis=1)[:, 1:effective_k+1]

    return {
        "n_samples": n_samples, "effective_k": effective_k,
        "X_hd": X_scaled, "labels": y, "D_orig": D_orig, "nn_orig": nn_orig,
        "X_pca": X_pca, "D_pca": D_pca, "nn_pca": nn_pca,
        "X_mds": X_mds, "D_mds": D_mds, "nn_mds": nn_mds
    }

def calc_filtered_stress(D_orig, D_proj, fp_points):
    """Calculates the global Kruskal stress ignoring pairs that involve the identified False Positives."""
    i_upper, j_upper = np.triu_indices_from(D_orig, k=1)
    
    if fp_points:
        # Filter out pairs where at least one point is a False Positive
        mask_i = ~np.isin(i_upper, fp_points)
        mask_j = ~np.isin(j_upper, fp_points)
        valid_mask = mask_i & mask_j
        d = D_orig[i_upper[valid_mask], j_upper[valid_mask]]
        delta = D_proj[i_upper[valid_mask], j_upper[valid_mask]]
    else:
        # If no False Positives, use all pairs
        d = D_orig[i_upper, j_upper]
        delta = D_proj[i_upper, j_upper]
        
    # Prevent division by zero errors on edge cases
    if len(d) == 0 or np.sum(d**2) == 0 or np.sum(delta**2) == 0:
        return 0.0
        
    # Compute rescaling factor alpha
    alpha = np.sum(d * delta) / np.sum(delta**2)
    # Compute standard MDS stress formula
    stress = np.sqrt(np.sum((d - alpha * delta)**2) / np.sum(d**2))
    
    return float(stress)

# --- ALGORITHMS PER ENDPOINT ---

def calc_fp_weighted(D_orig, D_proj, nn_orig, nn_proj, k_effective, threshold):
    """Original logic: score based on the magnitude of the distance error for false neighbors."""
    n_samples = D_proj.shape[0]
    fp_scores = np.zeros(n_samples)

    for i in range(n_samples):
        false_neighbors = set(nn_proj[i]) - set(nn_orig[i])
        for j in false_neighbors:
            weight = max(0, D_orig[i, j] - D_proj[i, j])
            fp_scores[i] += weight
            fp_scores[j] += weight 

    max_score = np.max(fp_scores)
    if max_score > 0:
        fp_scores = fp_scores / max_score

    return np.where(fp_scores >= threshold)[0].tolist()

def calc_fp_mismatch(nn_orig, nn_proj, k, threshold):
    """False positives calculated strictly using K-neighbourhood mismatch rate."""
    n_samples = nn_orig.shape[0]
    fp_points = []
    for i in range(n_samples):
        set_orig = set(nn_orig[i])
        set_proj = set(nn_proj[i])
        mismatch_rate = len(set_proj - set_orig) / k
        if mismatch_rate > threshold:
            fp_points.append(i)
    return fp_points

def calc_fp_stress(D_orig, D_proj, threshold):
    """False positives based on the alpha factor and point-to-point relative error."""
    i_upper, j_upper = np.triu_indices_from(D_orig, k=1)
    d = D_orig[i_upper, j_upper]
    delta = D_proj[i_upper, j_upper]
    
    alpha = np.sum(d * delta) / np.sum(delta**2)
    error = np.zeros_like(d)
    valid_mask = d > 0 
    
    # Calculate the relative error: (d - alpha * delta) / d
    error[valid_mask] = (d[valid_mask] - alpha * delta[valid_mask]) / d[valid_mask]
    
    fp_pairs_idx = np.where(error > threshold)[0]
    fp_points = set()
    for idx in fp_pairs_idx:
        fp_points.add(int(i_upper[idx]))
        fp_points.add(int(j_upper[idx]))
    return list(fp_points)

def calc_fp_iterative_stress(D_orig, D_proj, target_stress, max_iters=5000):
    """
    Iteratively identifies the worst false positive pairs and removes 
    ALL edges of the corresponding nodes, updating the global stress 
    until it falls below the target_stress threshold.
    """
    i_upper, j_upper = np.triu_indices_from(D_orig, k=1)
    valid_mask = np.ones(len(i_upper), dtype=bool)
    fp_points = set()
    
    d_all = D_orig[i_upper, j_upper]
    delta_all = D_proj[i_upper, j_upper]
    
    for _ in range(max_iters):
        d = d_all[valid_mask]
        delta = delta_all[valid_mask]
        
        # Guard against zero-division in remaining pairs
        if len(d) == 0 or np.sum(d**2) == 0 or np.sum(delta**2) == 0:
            break
            
        # Compute rescaling factor alpha
        alpha = np.sum(d * delta) / np.sum(delta**2)
        
        # Compute standard MDS stress based on current active pairs
        errors = (d - alpha * delta)**2
        stress = np.sqrt(np.sum(errors) / np.sum(d**2))
        
        # Terminate iteration if global stress has reached the desired threshold
        if stress <= target_stress:
            break
            
        # Filter to only consider pairs where original distance is unexpectedly larger
        # than projected distance (d > alpha * delta) as False Positives candidates.
        fp_cond = d > alpha * delta
        
        if not np.any(fp_cond):
            break
            
        # Assign actual errors where condition holds, -1 otherwise to ignore them
        fp_errors = np.where(fp_cond, errors, -1)
        max_err_idx = np.argmax(fp_errors)
        
        # Map the index from the masked array back to the original index
        actual_idx = np.where(valid_mask)[0][max_err_idx]
        
        # Identify the points forming the worst edge
        p1 = int(i_upper[actual_idx])
        p2 = int(j_upper[actual_idx])
        
        # Add them to the false positive set
        fp_points.add(p1)
        fp_points.add(p2)
        
        # When a pair is identified as false, we consider the nodes corrupted.
        # Therefore, we must invalidate ALL pairs connected to these nodes.
        invalid_new = (i_upper == p1) | (i_upper == p2) | (j_upper == p1) | (j_upper == p2)
        valid_mask[invalid_new] = False
        
    return list(fp_points)

def calc_fp_centroids(X_hd, X_proj, labels, threshold):
    """False positives based on false distances to pseudo-centroids."""
    unique_labels = np.unique(labels)
    
    # Compute the average coordinate (centroid) for each cluster
    centroids_hd = {l: np.mean(X_hd[labels == l], axis=0) for l in unique_labels}
    centroids_2d = {l: np.mean(X_proj[labels == l], axis=0) for l in unique_labels}
    
    n_samples = X_hd.shape[0]
    d_ic = np.zeros(n_samples)
    delta_ic = np.zeros(n_samples)
    
    # Compute point-to-centroid distances
    for i in range(n_samples):
        l = labels[i]
        d_ic[i] = np.linalg.norm(X_hd[i] - centroids_hd[l])
        delta_ic[i] = np.linalg.norm(X_proj[i] - centroids_2d[l])
        
    alpha = np.sum(d_ic * delta_ic) / np.sum(delta_ic**2)
    error = np.zeros_like(d_ic)
    valid_mask = d_ic > 0
    error[valid_mask] = (d_ic[valid_mask] - alpha * delta_ic[valid_mask]) / d_ic[valid_mask]
    
    fp_points = np.where(error > threshold)[0].tolist()
    pseudo_centroids = {str(k): v.tolist() for k, v in centroids_2d.items()}
    
    return fp_points, pseudo_centroids

# --- ENDPOINTS ---

@app.route('/api/compute_fp_weighted', methods=['POST'])
def endpoint_fp_weighted():
    data = request.get_json()
    dataset_name = data.get('dataset')
    k = data.get('k', 15)
    threshold = data.get('threshold', 0.5)

    try:
        ctx = load_and_project(dataset_name, k)
        pca_fp = calc_fp_weighted(ctx['D_orig'], ctx['D_pca'], ctx['nn_orig'], ctx['nn_pca'], ctx['effective_k'], threshold)
        mds_fp = calc_fp_weighted(ctx['D_orig'], ctx['D_mds'], ctx['nn_orig'], ctx['nn_mds'], ctx['effective_k'], threshold)

        return jsonify({
            "metadata": {"dataset": dataset_name, "method": "weighted", "k": ctx['effective_k'], "threshold": threshold},
            "false_positive_points_pca": pca_fp,
            "false_positive_points_mds": mds_fp,
            "stress_pca": calc_filtered_stress(ctx['D_orig'], ctx['D_pca'], pca_fp),
            "stress_mds": calc_filtered_stress(ctx['D_orig'], ctx['D_mds'], mds_fp)
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/compute_fp_mismatch', methods=['POST'])
def endpoint_fp_mismatch():
    data = request.get_json()
    dataset_name = data.get('dataset')
    k = data.get('k', 15)
    threshold = data.get('threshold', 0.5)

    try:
        ctx = load_and_project(dataset_name, k)
        pca_fp = calc_fp_mismatch(ctx['nn_orig'], ctx['nn_pca'], ctx['effective_k'], threshold)
        mds_fp = calc_fp_mismatch(ctx['nn_orig'], ctx['nn_mds'], ctx['effective_k'], threshold)

        return jsonify({
            "metadata": {"dataset": dataset_name, "method": "knn_mismatch", "k": ctx['effective_k'], "threshold": threshold},
            "false_positive_points_pca": pca_fp,
            "false_positive_points_mds": mds_fp,
            "stress_pca": calc_filtered_stress(ctx['D_orig'], ctx['D_pca'], pca_fp),
            "stress_mds": calc_filtered_stress(ctx['D_orig'], ctx['D_mds'], mds_fp)
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/compute_fp_stress', methods=['POST'])
def endpoint_fp_stress():
    data = request.get_json()
    dataset_name = data.get('dataset')
    threshold = data.get('threshold', 0.1)

    try:
        ctx = load_and_project(dataset_name, 15) 
        pca_fp = calc_fp_stress(ctx['D_orig'], ctx['D_pca'], threshold)
        mds_fp = calc_fp_stress(ctx['D_orig'], ctx['D_mds'], threshold)

        return jsonify({
            "metadata": {"dataset": dataset_name, "method": "stress_alpha", "threshold": threshold},
            "false_positive_points_pca": pca_fp,
            "false_positive_points_mds": mds_fp,
            "stress_pca": calc_filtered_stress(ctx['D_orig'], ctx['D_pca'], pca_fp),
            "stress_mds": calc_filtered_stress(ctx['D_orig'], ctx['D_mds'], mds_fp)
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/compute_fp_iterative_stress', methods=['POST'])
def endpoint_fp_iterative_stress():
    """Endpoint for processing iterative stress reduction with independent targets for PCA and MDS."""
    data = request.get_json()
    dataset_name = data.get('dataset')
    
    # Extract independent target stress values from the payload
    target_stress_pca = data.get('target_stress_pca', 0.1)
    target_stress_mds = data.get('target_stress_mds', 0.1)

    try:
        ctx = load_and_project(dataset_name, 15) 
        
        # Calculate iterative stress independently for each projection using its respective target
        pca_fp = calc_fp_iterative_stress(ctx['D_orig'], ctx['D_pca'], target_stress_pca)
        mds_fp = calc_fp_iterative_stress(ctx['D_orig'], ctx['D_mds'], target_stress_mds)

        return jsonify({
            "metadata": {
                "dataset": dataset_name, 
                "method": "iterative_stress", 
                "target_stress_pca": target_stress_pca,
                "target_stress_mds": target_stress_mds
            },
            "false_positive_points_pca": pca_fp,
            "false_positive_points_mds": mds_fp,
            "stress_pca": calc_filtered_stress(ctx['D_orig'], ctx['D_pca'], pca_fp),
            "stress_mds": calc_filtered_stress(ctx['D_orig'], ctx['D_mds'], mds_fp)
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/compute_fp_centroids', methods=['POST'])
def endpoint_fp_centroids():
    data = request.get_json()
    dataset_name = data.get('dataset')
    threshold = data.get('threshold', 0.1)

    try:
        ctx = load_and_project(dataset_name, 15)
        pca_fp, pca_centroids = calc_fp_centroids(ctx['X_hd'], ctx['X_pca'], ctx['labels'], threshold)
        mds_fp, mds_centroids = calc_fp_centroids(ctx['X_hd'], ctx['X_mds'], ctx['labels'], threshold)

        return jsonify({
            "metadata": {"dataset": dataset_name, "method": "pseudo_centroids", "threshold": threshold},
            "false_positive_points_pca": pca_fp,
            "false_positive_points_mds": mds_fp,
            "pseudo_centroids_2d_pca": pca_centroids,
            "pseudo_centroids_2d_mds": mds_centroids,
            "stress_pca": calc_filtered_stress(ctx['D_orig'], ctx['D_pca'], pca_fp),
            "stress_mds": calc_filtered_stress(ctx['D_orig'], ctx['D_mds'], mds_fp)
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)