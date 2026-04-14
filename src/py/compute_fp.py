import os
import json
import argparse
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import MDS

# --- CONFIGURATION ---
DATASETS = ["wine", "iris", "user_knowledge"]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LABEL_INDEX = {
    "wine": 0,
    "iris": -1,
    "user_knowledge": -1
}

def calculate_fp_metrics(nn_orig, X_proj, k_effective, min_mismatch_pct):
    """
    Given the original k-NN indices and a 2D projection, 
    calculates the False Positive points, rates, and 2D neighbors.
    """
    n_samples = X_proj.shape[0]

    # Distance matrix for the projection
    D_proj = squareform(pdist(X_proj, metric='euclidean'))
    
    # k-NN indices in 2D
    nn_proj = np.argsort(D_proj, axis=1)[:, 1:k_effective+1]

    FP_rate = np.zeros(n_samples)

    for i in range(n_samples):
        set_orig = set(nn_orig[i])
        set_proj = set(nn_proj[i])

        # False neighbors: points in 2D neighborhood that are NOT in HD neighborhood
        false_neighbors_of_i = set_proj - set_orig
        FP_rate[i] = len(false_neighbors_of_i) / k_effective

    # Thresholding
    fp_points = np.where(FP_rate >= min_mismatch_pct)[0]

    return fp_points.tolist()


def process_dataset(dataset_name, k_neighbors, min_mismatch_pct):
    print(f"\n--- Processing FP for: {dataset_name.upper()} ---")

    dataset_path = os.path.join(BASE_DIR, '..', '..', 'dataset', f'{dataset_name}.csv')
    output_dir = os.path.join(BASE_DIR, '..', 'json', dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, 'step_fp_results.json')

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset {dataset_path} not found. Skipping.")
        return

    # Load dataset
    df = pd.read_csv(dataset_path)
    target_idx = LABEL_INDEX.get(dataset_name, -1)
    X = df.drop(df.columns[target_idx], axis=1).values
    n_samples = X.shape[0]

    # 1. STANDARDIZATION & HD BASELINE (Computed only once)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    D_orig = squareform(pdist(X_scaled, metric='euclidean'))
    effective_k = min(k_neighbors, n_samples - 2)
    nn_orig = np.argsort(D_orig, axis=1)[:, 1:effective_k+1]

    # 2. PCA PROJECTION
    print("  -> Computing PCA...")
    X_pca = PCA(n_components=2).fit_transform(X_scaled)
    pca_fp_pts = calculate_fp_metrics(nn_orig, X_pca, effective_k, min_mismatch_pct)

    # 3. MDS PROJECTION (random_state set for reproducibility)
    print("  -> Computing MDS...")
    X_mds = MDS(n_components=2, normalized_stress='auto', random_state=42).fit_transform(X_scaled)
    mds_fp_pts = calculate_fp_metrics(nn_orig, X_mds, effective_k, min_mismatch_pct)

    # 4. BUILD SIMPLIFIED JSON STRUCTURE
    output_data = {
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

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)

    print(f"Saved: {output_file}")
    print(f"  - FP Found (PCA): {len(pca_fp_pts)}")
    print(f"  - FP Found (MDS): {len(mds_fp_pts)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate the False Positives introduced by PCA and MDS.")
    parser.add_argument(
        "--k", 
        type=int, 
        default=15, 
        help="Number of neighbors (k) to consider. Default: 15"
    )
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.8, 
        help="Minimum mismatch threshold (0.0 to 1.0) to declare a point as a False Positive. Example: 0.6 = 60%. Default: 0.8"
    )
    
    args = parser.parse_args()

    print(f"Starting script with K={args.k} and Mismatch Threshold={args.threshold * 100}%")

    for ds in DATASETS:
        process_dataset(ds, args.k, args.threshold)