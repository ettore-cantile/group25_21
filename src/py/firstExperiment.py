import json
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics import euclidean_distances
from sklearn.datasets import load_wine

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.abspath(os.path.join(SCRIPT_DIR, '../../dataset/firstExperiment.json'))
CSV_FILENAME = os.path.abspath(os.path.join(SCRIPT_DIR, '../../dataset/wine.csv'))
TARGET_COLUMN = 'Producer'

print(f"Caricamento dataset...")
try:
    df = pd.read_csv(CSV_FILENAME)
    df = df.dropna()
    if TARGET_COLUMN in df.columns:
        class_names = ["Class " + c for c in df[TARGET_COLUMN].astype(str).values]
        data_df = df.drop(columns=[TARGET_COLUMN])
    else:
        class_names = ["Item " + str(i) for i in range(len(df))]
        data_df = df
except FileNotFoundError:
    wine = load_wine()
    data_df = pd.DataFrame(wine.data, columns=wine.feature_names)
    class_names = ["Class " + str(c) for c in wine.target]

X = data_df.select_dtypes(include=[np.number]).values
N = X.shape[0]
X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
X_pca[:, 0] = -X_pca[:, 0] 

mds = MDS(n_components=2, dissimilarity='euclidean', random_state=42, normalized_stress='auto', n_init=4, max_iter=300)
X_mds = mds.fit_transform(X_scaled)

def calculate_distance_distortion(D_high, D_low):
    D_high_norm = D_high / np.max(D_high)
    D_low_norm = D_low / np.max(D_low)
    Delta = D_high_norm - D_low_norm
    np.fill_diagonal(Delta, 0)
    point_scores = np.sum(Delta, axis=1)
    global_score = np.sum(np.abs(Delta)) / 2
    return point_scores, global_score

D_high = euclidean_distances(X_scaled)
score_pca, global_pca = calculate_distance_distortion(D_high, euclidean_distances(X_pca))
score_mds, global_mds = calculate_distance_distortion(D_high, euclidean_distances(X_mds))

TOLERANCE = 0.10 * N 
pca_matches = np.sum(np.abs(score_pca) <= TOLERANCE) / N * 100
mds_matches = np.sum(np.abs(score_mds) <= TOLERANCE) / N * 100

# --- NOVITÀ: Trova i 5 vicini più prossimi in R13 ---
K = 5
# argsort ordina le distanze crescenti. [:, 1:K+1] prende dal 1° al 5° (escludendo lo 0, che è il punto stesso)
neighbors_R13 = np.argsort(D_high, axis=1)[:, 1:K+1].tolist()

# Struttura JSON aggiornata con le Statistiche e i Vicini
export_data = {
    "stats": {
        "pca_global": round(global_pca, 2),
        "mds_global": round(global_mds, 2),
        "pca_matches": round(pca_matches, 1),
        "mds_matches": round(mds_matches, 1)
    },
    "points": []
}

for i in range(N):
    export_data["points"].append({
        "id": i,
        "pca_x": float(X_pca[i, 0]),
        "pca_y": float(X_pca[i, 1]),
        "mds_x": float(X_mds[i, 0]),
        "mds_y": float(X_mds[i, 1]),
        "score_pca": float(score_pca[i]),
        "score_mds": float(score_mds[i]),
        "class_name": class_names[i],
        "neighbors": neighbors_R13[i] # Lista degli ID dei vicini reali
    })

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, 'w') as f:
    json.dump(export_data, f, indent=4)
print(f"Esportazione completata in {OUTPUT_FILE}")