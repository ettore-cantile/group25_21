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

LABEL_INDEX = {
    "wine": 0,
    "iris": -1,
    "user_knowledge": -1
}

app = Flask(__name__)
CORS(app)

def calculate_fp_metrics(nn_orig, X_proj, k_effective, min_mismatch_pct):
    """
    Given the original k-NN indices and a 2D projection,
    calculates and returns the list of False Positive points.
    """
    n_samples = X_proj.shape[0]

    D_proj = squareform(pdist(X_proj, metric='euclidean'))
    nn_proj = np.argsort(D_proj, axis=1)[:, 1:k_effective+1]

    FP_rate = np.zeros(n_samples)

    for i in range(n_samples):
        set_orig = set(nn_orig[i])
        set_proj = set(nn_proj[i])

        false_neighbors_of_i = set_proj - set_orig
        FP_rate[i] = len(false_neighbors_of_i) / k_effective

    fp_points = np.where(FP_rate >= min_mismatch_pct)[0]
    return fp_points.tolist()


@app.route('/api/compute_fp', methods=['POST'])
def compute_fp():
    # 1. Parsing the JSON request
    data = request.get_json()

    if not data or 'dataset' not in data:
        return jsonify({"error": "Missing 'dataset' parameter in JSON body."}), 400

    dataset_name = data['dataset']
    k_neighbors = data.get('k', 15)  # Default to 15 if not provided
    min_mismatch_pct = data.get('threshold', 0.8) # Default to 0.8 if not provided

    dataset_path = os.path.join(DATASET_DIR, f'{dataset_name}.csv')

    if not os.path.exists(dataset_path):
        return jsonify({"error": f"Dataset '{dataset_name}' not found at {dataset_path}."}), 404

    try:
        # 2. Data loading and preparation
        df = pd.read_csv(dataset_path)
        target_idx = LABEL_INDEX.get(dataset_name, -1)
        X = df.drop(df.columns[target_idx], axis=1).values
        n_samples = X.shape[0]

        # 3. Baseline Calculation (HD)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        D_orig = squareform(pdist(X_scaled, metric='euclidean'))
        effective_k = min(k_neighbors, n_samples - 2)
        nn_orig = np.argsort(D_orig, axis=1)[:, 1:effective_k+1]

        # 4. Projections and FP Calculation
        X_pca = PCA(n_components=2).fit_transform(X_scaled)
        pca_fp_pts = calculate_fp_metrics(nn_orig, X_pca, effective_k, min_mismatch_pct)

        X_mds = MDS(n_components=2, normalized_stress='auto', random_state=42).fit_transform(X_scaled)
        mds_fp_pts = calculate_fp_metrics(nn_orig, X_mds, effective_k, min_mismatch_pct)

        # 5. Building the response
        response_data = {
            "metadata": {
                "dataset": f"{dataset_name}.csv",
                "k_neighbors": effective_k,
                "min_mismatch_pct": min_mismatch_pct,
                "total_samples": n_samples,
                "num_fp_points_pca": len(pca_fp_pts),
                "num_fp_points_mds": len(mds_fp_pts)
            },
            "false_positive_points_pca": pca_fp_pts,
            "false_positive_points_mds": mds_fp_pts
        }

        return jsonify(response_data), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Start the Flask development server on port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)