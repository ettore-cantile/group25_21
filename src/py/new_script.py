import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from flask_cors import CORS

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, '..', '..', 'dataset')

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
    
    y = df.iloc[:, target_idx].values # Labels
    X = df.drop(df.columns[target_idx], axis=1).values
    n_samples = X.shape[0]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Baseline (HD)
    D_orig = squareform(pdist(X_scaled, metric='euclidean'))
    effective_k = min(k_neighbors, n_samples - 2)
    nn_orig = np.argsort(D_orig, axis=1)[:, 1:effective_k+1]

    # PCA
    X_pca = PCA(n_components=2).fit_transform(X_scaled)
    D_pca = squareform(pdist(X_pca, metric='euclidean'))
    nn_pca = np.argsort(D_pca, axis=1)[:, 1:effective_k+1]

    # MDS
    X_mds = MDS(n_components=2, normalized_stress='auto', random_state=42).fit_transform(X_scaled)
    D_mds = squareform(pdist(X_mds, metric='euclidean'))
    nn_mds = np.argsort(D_mds, axis=1)[:, 1:effective_k+1]

    return {
        "n_samples": n_samples, "effective_k": effective_k,
        "X_hd": X_scaled, "labels": y, "D_orig": D_orig, "nn_orig": nn_orig,
        "X_pca": X_pca, "D_pca": D_pca, "nn_pca": nn_pca,
        "X_mds": X_mds, "D_mds": D_mds, "nn_mds": nn_mds
    }

# --- ALGORITHMS PER ENDPOINT ---

def calc_fp_weighted(D_orig, D_proj, nn_orig, nn_proj, k_effective, threshold):
    """(Endpoint 1) Original logic: score based on the magnitude of the distance error."""
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
    """(Endpoint 2) False positives calculated using K-neighbourhood mismatch rate."""
    n_samples = nn_orig.shape[0]
    fp_points = []
    
    for i in range(n_samples):
        set_orig = set(nn_orig[i])
        set_proj = set(nn_proj[i])
        
        # How many neighbors in 2D were NOT neighbors in HD?
        false_positives_count = len(set_proj - set_orig)
        mismatch_rate = false_positives_count / k
        
        if mismatch_rate > threshold:
            fp_points.append(i)
            
    return fp_points


def calc_fp_stress(D_orig, D_proj, threshold):
    """(Endpoint 3) False positives based on alpha and the stress function."""
    # Consider only the upper triangle to avoid evaluating pairs twice and exclude the diagonal
    i_upper, j_upper = np.triu_indices_from(D_orig, k=1)
    
    d = D_orig[i_upper, j_upper]     # HD Distances
    delta = D_proj[i_upper, j_upper] # 2D Distances
    
    # Calculate alpha as per PDF: sum(d * delta) / sum(delta^2)
    alpha = np.sum(d * delta) / np.sum(delta**2)
    
    # Calculate the error: (d - alpha * delta) / d
    error = np.zeros_like(d)
    valid_mask = d > 0 # Prevent division by zero
    error[valid_mask] = (d[valid_mask] - alpha * delta[valid_mask]) / d[valid_mask]
    
    # Find the indices of the pairs that exceed the threshold
    fp_pairs_idx = np.where(error > threshold)[0]
    
    # Extract the individual points involved (a point is "false" if involved in at least one false relationship)
    fp_points = set()
    for idx in fp_pairs_idx:
        fp_points.add(int(i_upper[idx]))
        fp_points.add(int(j_upper[idx]))
        
    return list(fp_points)


def calc_fp_centroids(X_hd, X_proj, labels, threshold):
    """(Endpoint 4) False positives based on distances from pseudo-centroids."""
    unique_labels = np.unique(labels)
    
    # 1. Calculate pseudo-centroids by aggregating by label
    centroids_hd = {l: np.mean(X_hd[labels == l], axis=0) for l in unique_labels}
    centroids_2d = {l: np.mean(X_proj[labels == l], axis=0) for l in unique_labels}
    
    n_samples = X_hd.shape[0]
    d_ic = np.zeros(n_samples)      # Point-Centroid Distance (HD)
    delta_ic = np.zeros(n_samples)  # Point-Centroid Distance (2D)
    
    # 2. Calculate distances from the assigned centroid
    for i in range(n_samples):
        l = labels[i]
        d_ic[i] = np.linalg.norm(X_hd[i] - centroids_hd[l])
        delta_ic[i] = np.linalg.norm(X_proj[i] - centroids_2d[l])
        
    # 3. Calculate the specific alpha for centroids and the error
    alpha = np.sum(d_ic * delta_ic) / np.sum(delta_ic**2)
    
    error = np.zeros_like(d_ic)
    valid_mask = d_ic > 0
    error[valid_mask] = (d_ic[valid_mask] - alpha * delta_ic[valid_mask]) / d_ic[valid_mask]
    
    fp_points = np.where(error > threshold)[0].tolist()
    
    # Convert centroids to serializable formats for the JSON response
    pseudo_centroids = {str(k): v.tolist() for k, v in centroids_2d.items()}
    
    return fp_points, pseudo_centroids


# --- ENDPOINTS ---

@app.route('/api/compute_fp_weighted', methods=['POST'])
def endpoint_fp_weighted():
    """Endpoint 1: Original (score weighted by distance magnitude)"""
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
            "false_positive_points_mds": mds_fp
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/compute_fp_mismatch', methods=['POST'])
def endpoint_fp_mismatch():
    """Endpoint 2: Based strictly on the K-Neighbourhood mismatch rate"""
    data = request.get_json()
    dataset_name = data.get('dataset')
    k = data.get('k', 15)
    threshold = data.get('threshold', 0.5) # E.g., if > 0.5 (50% of neighbors are false), mark as FP

    try:
        ctx = load_and_project(dataset_name, k)
        pca_fp = calc_fp_mismatch(ctx['nn_orig'], ctx['nn_pca'], ctx['effective_k'], threshold)
        mds_fp = calc_fp_mismatch(ctx['nn_orig'], ctx['nn_mds'], ctx['effective_k'], threshold)

        return jsonify({
            "metadata": {"dataset": dataset_name, "method": "knn_mismatch", "k": ctx['effective_k'], "threshold": threshold},
            "false_positive_points_pca": pca_fp,
            "false_positive_points_mds": mds_fp
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/compute_fp_stress', methods=['POST'])
def endpoint_fp_stress():
    """Endpoint 3: Based on Alpha, the PDF Stress formula, and relative error"""
    data = request.get_json()
    dataset_name = data.get('dataset')
    threshold = data.get('threshold', 0.1) # Threshold for (d - alpha*delta)/d >> 0

    try:
        ctx = load_and_project(dataset_name, 15) # k is not used in the stress formula, but the helper requires it
        pca_fp = calc_fp_stress(ctx['D_orig'], ctx['D_pca'], threshold)
        mds_fp = calc_fp_stress(ctx['D_orig'], ctx['D_mds'], threshold)

        return jsonify({
            "metadata": {"dataset": dataset_name, "method": "stress_alpha", "threshold": threshold},
            "false_positive_points_pca": pca_fp,
            "false_positive_points_mds": mds_fp
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/compute_fp_centroids', methods=['POST'])
def endpoint_fp_centroids():
    """Endpoint 4: False distances towards pseudo-centroids"""
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
            "pseudo_centroids_2d_mds": mds_centroids
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)