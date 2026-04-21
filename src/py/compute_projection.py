import os
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, trustworthiness
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

# --- CONFIGURATION ---
DATASETS = ["wine", "iris", "user_knowledge"]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LABEL_INDEX = {
    "wine": 0,
    "iris": -1,
    "user_knowledge": -1
}

def calculate_normalized_stress(X_high, X_low):
    """
    Calculates the normalized stress using the scale factor alpha,
    respecting the condition i < j (only the upper triangle of the matrix).
    """
    # Calculate the full distance matrices
    d_high_matrix = pairwise_distances(X_high, metric='euclidean')
    d_low_matrix = pairwise_distances(X_low, metric='euclidean')
    
    # Extract only the upper triangle (excluding the diagonal with k=1) to ensure i < j
    idx = np.triu_indices_from(d_high_matrix, k=1)
    d_high = d_high_matrix[idx]
    d_low = d_low_matrix[idx]
    
    # Calculate alpha: sum(d_ij * delta_ij) / sum(delta_ij^2)
    # Adding a small epsilon to the denominator to prevent division by zero
    alpha = np.sum(d_high * d_low) / (np.sum(d_low ** 2) + 1e-10)
    
    # Calculate stress
    stress = np.sqrt(np.sum((d_high - alpha * d_low) ** 2) / np.sum(d_high ** 2))
    
    return float(stress)


def run_projections_for_dataset(dataset_name):
    print(f"\n--- Processing Projections for: {dataset_name.upper()} ---")
    
    dataset_path = os.path.join(BASE_DIR, '..', '..', 'dataset', f'{dataset_name}.csv')
    input_json = os.path.join(BASE_DIR, '..', 'json', dataset_name, 'step1_results.json')
    output_json = os.path.join(BASE_DIR, '..', 'json', dataset_name, 'step2_final_data.json')

    if not os.path.exists(dataset_path) or not os.path.exists(input_json):
        print(f"Missing files for {dataset_name}. Run compute_relationship.py first.")
        return

    # Load Dataset and drop the label column dynamically
    df = pd.read_csv(dataset_path)
    target_idx = LABEL_INDEX.get(dataset_name, -1)
    X = df.drop(df.columns[target_idx], axis=1).values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_scaled)

    mds = MDS(n_components=2, dissimilarity='euclidean', random_state=42, normalized_stress='auto')
    mds_coords = mds.fit_transform(X_scaled)
        
    with open(input_json, 'r') as f:
        data = json.load(f)

    k_val = int(data["metadata"]["k_neighbors_used"]) if str(data["metadata"]["k_neighbors_used"]).isdigit() else 15

    # Calculate Trustworthiness & Continuity
    pca_trust = trustworthiness(X_scaled, pca_coords, n_neighbors=k_val)
    mds_trust = trustworthiness(X_scaled, mds_coords, n_neighbors=k_val)
    pca_cont = trustworthiness(pca_coords, X_scaled, n_neighbors=k_val)
    mds_cont = trustworthiness(mds_coords, X_scaled, n_neighbors=k_val)
    
    # Calculate Stress (Added)
    pca_stress = calculate_normalized_stress(X_scaled, pca_coords)
    mds_stress = calculate_normalized_stress(X_scaled, mds_coords)

    data["metadata"]["global_assessment"]["pca"] = {
        "trustworthiness": round(float(pca_trust), 4),
        "continuity": round(float(pca_cont), 4),
        "stress": round(pca_stress, 4)
    }
    data["metadata"]["global_assessment"]["mds"] = {
        "trustworthiness": round(float(mds_trust), 4),
        "continuity": round(float(mds_cont), 4),
        "stress": round(mds_stress, 4)
    }

    # Inject 2D coordinates into the JSON
    for i, point in enumerate(data["points"]):
        point["pca_x"] = round(float(pca_coords[i, 0]), 4)
        point["pca_y"] = round(float(pca_coords[i, 1]), 4)
        point["mds_x"] = round(float(mds_coords[i, 0]), 4)
        point["mds_y"] = round(float(mds_coords[i, 1]), 4)

    data["metadata"]["projections_included"] = ["PCA", "Euclidean MDS"]

    with open(output_json, 'w') as f:
        json.dump(data, f, indent=4)
        
    print(f"Saved: {output_json}")

if __name__ == "__main__":
    for ds in DATASETS:
        run_projections_for_dataset(ds)