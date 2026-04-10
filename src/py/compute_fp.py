import os
import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- CONFIGURATION ---
DATASETS = ["wine", "iris", "user_knowledge"]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LABEL_INDEX = {
    "wine": 0,
    "iris": -1,
    "user_knowledge": -1
}

K_NEIGHBORS = 15


def compute_false_positive_points(X):
    """
    Compute local false positive points (Victims) introduced by PCA.
    Returns the FP points, the FP rates, and the lists of neighbors in HD and 2D.
    """
    n_samples = X.shape[0]

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA projection (2D)
    X_pca = PCA(n_components=2).fit_transform(X_scaled)

    # Distance matrices
    D_orig = squareform(pdist(X_scaled, metric='euclidean'))
    D_pca = squareform(pdist(X_pca, metric='euclidean'))

    # Sicurezza per dataset piccoli
    k_effettivo = min(K_NEIGHBORS, n_samples - 2)

    # k-NN indices (indici dei k-vicini)
    nn_orig = np.argsort(D_orig, axis=1)[:, 1:k_effettivo+1]
    nn_pca = np.argsort(D_pca, axis=1)[:, 1:k_effettivo+1]

    # Vettore per il rate delle Vittime
    FP_rate = np.zeros(n_samples)

    for i in range(n_samples):
        set_orig = set(nn_orig[i])
        set_pca = set(nn_pca[i])

        # Quali punti circondano il punto i nel 2D ma NON nell'originale?
        falsi_vicini_di_i = set_pca - set_orig
        
        # Punteggio da 0.0 (tutti amici veri) a 1.0 (tutti estranei)
        FP_rate[i] = len(falsi_vicini_di_i) / k_effettivo

    # Calcolo soglia dinamica per evidenziare solo i punti visibilmente peggiori
    media_errore = np.mean(FP_rate)
    soglia_dinamica = np.percentile(FP_rate, 95) # Prendi il top 10% peggiore
    
    fp_points = np.where((FP_rate >= soglia_dinamica) & (FP_rate > media_errore))[0]

    # Restituiamo anche le matrici dei vicini convertite in liste Python (.tolist()) 
    # per la serializzazione JSON
    return fp_points.tolist(), FP_rate.tolist(), nn_orig.tolist(), nn_pca.tolist()


def process_dataset(dataset_name):
    print(f"\n--- Processing FP for: {dataset_name.upper()} ---")

    dataset_path = os.path.join(BASE_DIR, '..', '..', 'dataset', f'{dataset_name}.csv')
    output_dir = os.path.join(BASE_DIR, '..', 'json', dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, 'step_fp_pca.json')

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset {dataset_path} not found. Skipping.")
        return

    # Load dataset
    df = pd.read_csv(dataset_path)

    # Extract labels and features
    target_idx = LABEL_INDEX.get(dataset_name, -1)
    labels = df.iloc[:, target_idx].values
    X = df.drop(df.columns[target_idx], axis=1).values

    # Compute FP points and neighbors
    fp_points, fp_rates, nn_orig_list, nn_pca_list = compute_false_positive_points(X)

    # Build per-point structure
    points_data = []
    for i in range(len(X)):
        points_data.append({
            "id": i,
            "label": str(labels[i]),
            "fp_rate": round(fp_rates[i], 6),
            "is_false_positive": i in fp_points,
            "neighbors_hd": nn_orig_list[i],  # Aggiunta: lista id vicini HD
            "neighbors_2d": nn_pca_list[i]    # Aggiunta: lista id vicini 2D
        })

    # Output JSON
    output_data = {
        "metadata": {
            "dataset": f"{dataset_name}.csv",
            "k_neighbors": min(K_NEIGHBORS, len(X) - 2),
            "num_fp_points": len(fp_points)
        },
        "false_positive_points": fp_points,
        "points": points_data
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)

    print(f"Saved: {output_file}")


if __name__ == "__main__":
    for ds in DATASETS:
        process_dataset(ds)