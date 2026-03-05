import os
import json
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler

# --- PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, '..', '..', 'dataset', 'wine.csv')
INPUT_JSON = os.path.join(BASE_DIR, '..', 'json', 'step1_results.json')
OUTPUT_JSON = os.path.join(BASE_DIR, '..', 'json', 'step2_final_data.json')

def run_projections():
    # 1. Load the original dataset
    print(f"Loading dataset from: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    
    # The label is the 'producer' (the first column).
    # We only need the features (all columns except the first) to perform the projections.
    X = df.iloc[:, 1:].values
    
    # Standardize the features (Zero mean, Unit variance)
    # Crucial for PCA and distance-based MDS to avoid scale dominance by features with large numbers (e.g., Proline)
    print("Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. Perform PCA (Principal Component Analysis)
    print("Running PCA projection (2 components)...")
    # Using random_state for reproducibility
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_scaled)

    # 3. Perform Euclidean MDS (Multidimensional Scaling)
    print("Running Euclidean MDS projection (2 components)...")
    # 'euclidean' is the default dissimilarity for sklearn's MDS
    mds = MDS(n_components=2, dissimilarity='euclidean', random_state=42, normalized_stress='auto')
    mds_coords = mds.fit_transform(X_scaled)

    # 4. Load Step 1 JSON data (Relationship metrics)
    print(f"Loading relationship metrics from: {INPUT_JSON}")
    if not os.path.exists(INPUT_JSON):
        raise FileNotFoundError(f"Run step1_relationships.py first. File not found: {INPUT_JSON}")
        
    with open(INPUT_JSON, 'r') as f:
        data = json.load(f)

    # 5. Inject 2D coordinates into the JSON structure
    print("Injecting 2D coordinates into the JSON data...")
    for i, point in enumerate(data["points"]):
        # Add PCA coordinates
        point["pca_x"] = round(float(pca_coords[i, 0]), 4)
        point["pca_y"] = round(float(pca_coords[i, 1]), 4)
        
        # Add MDS coordinates
        point["mds_x"] = round(float(mds_coords[i, 0]), 4)
        point["mds_y"] = round(float(mds_coords[i, 1]), 4)

    # Update metadata to reflect that mapping phase is complete
    data["metadata"]["projections_included"] = ["PCA", "Euclidean MDS"]

    # 6. Save the final integrated JSON
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(data, f, indent=4)
        
    print(f"Process completed successfully! Final payload saved to: {OUTPUT_JSON}")

if __name__ == "__main__":
    run_projections()