import json
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics import euclidean_distances

# ==========================================
# 1. USER CONFIGURATION
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(SCRIPT_DIR, '../dataset/wine_data.json')
CSV_FILENAME = os.path.join(SCRIPT_DIR, '../dataset/wine.csv')
TARGET_COLUMN = 'Producer'

# Imposta il numero di vicini da considerare (K)
# Un buon valore di default è solitamente tra 5 e 15 a seconda del dataset
K_NEIGHBORS = 10 

# ==========================================
# 2. DATA LOADING & PREPARATION
# ==========================================
print(f"Loading {CSV_FILENAME}...")
try:
    df = pd.read_csv(CSV_FILENAME)
except FileNotFoundError:
    print("ERROR: File not found.")
    exit()

df = df.dropna()

if TARGET_COLUMN and TARGET_COLUMN in df.columns:
    class_names = df[TARGET_COLUMN].astype(str).values
    class_names = ["Class " + c for c in class_names]
    data_df = df.drop(columns=[TARGET_COLUMN])
else:
    class_names = ["Item " + str(i) for i in range(len(df))]
    data_df = df

X_original = data_df.select_dtypes(include=[np.number])
X = X_original.values
print(f"Data shape: {X.shape}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# 3. DIMENSIONALITY REDUCTION
# ==========================================
print("Computing PCA...")
pca = PCA(n_components=2, random_state=0)
X_pca = pca.fit_transform(X_scaled)
X_pca[:, 0] = -X_pca[:, 0] # Axis flipping fix

print("Computing MDS...")
mds = MDS(n_components=2, dissimilarity='euclidean', random_state=42, normalized_stress='auto', n_init=4)
X_mds = mds.fit_transform(X_scaled)

# ==========================================
# 4. DISTORTION METRICS (Rank-based KNN)
# ==========================================
print("Computing advanced distortion scores (FP & FN weights)...")

def calculate_knn_distortion(D_high, D_low, K):
    N = D_high.shape[0]
    
    # Riempiamo la diagonale con infinito per non contare un punto come vicino di se stesso
    np.fill_diagonal(D_high, np.inf)
    np.fill_diagonal(D_low, np.inf)

    # Calcoliamo i ranghi (le "classifiche"). argsort(argsort) restituisce il rango di ogni elemento
    R_high = np.argsort(np.argsort(D_high, axis=1), axis=1)
    R_low = np.argsort(np.argsort(D_low, axis=1), axis=1)

    score_fp = np.zeros(N)
    score_fn = np.zeros(N)

    for i in range(N):
        # Indici dei K-vicini nello spazio ad alta dimensionalità (Originale)
        U_i = set(np.where(R_high[i] < K)[0])
        # Indici dei K-vicini nello spazio a bassa dimensionalità (2D)
        V_i = set(np.where(R_low[i] < K)[0])

        # Falsi Positivi: Punti che sono vicini nel 2D ma non lo erano in origine
        fp_indices = list(V_i - U_i)
        if fp_indices:
            # Il peso è quanto erano lontani in origine rispetto a K
            score_fp[i] = np.sum(R_high[i, fp_indices] - K)

        # Falsi Negativi: Punti che erano vicini in origine ma sono lontani nel 2D
        fn_indices = list(U_i - V_i)
        if fn_indices:
            # Il peso è quanto sono stati allontanati nel 2D rispetto a K
            score_fn[i] = np.sum(R_low[i, fn_indices] - K)

    # Indice divergente: Positivo = dominante Falsi Positivi, Negativo = dominante Falsi Negativi
    point_scores = score_fp - score_fn
    
    # Punteggio Globale dell'algoritmo (Somma di tutti gli errori)
    global_score = np.sum(score_fp) + np.sum(score_fn)
    
    return point_scores, global_score

# Calcolo distanze euclidee
D_high = euclidean_distances(X_scaled)
D_pca = euclidean_distances(X_pca)
D_mds = euclidean_distances(X_mds)

# Calcolo scores
score_pca, global_pca = calculate_knn_distortion(D_high.copy(), D_pca.copy(), K=K_NEIGHBORS)
score_mds, global_mds = calculate_knn_distortion(D_high.copy(), D_mds.copy(), K=K_NEIGHBORS)

print("\n--- QUALITY ANALYSIS (INITIAL ASSESSMENT) ---")
print(f"PCA Global Error Score: {global_pca:.0f}")
print(f"MDS Global Error Score: {global_mds:.0f}")
if global_pca < global_mds:
    print("-> PCA performed better at preserving local neighborhoods.")
else:
    print("-> MDS performed better at preserving local neighborhoods.")
print("---------------------------------------------\n")

# ==========================================
# 5. EXPORT
# ==========================================
data_export = []
for i in range(len(X)):
    data_export.append({
        "id": i,
        "pca_x": float(X_pca[i, 0]),
        "pca_y": float(X_pca[i, 1]),
        "mds_x": float(X_mds[i, 0]),
        "mds_y": float(X_mds[i, 1]),
        "score_pca": float(score_pca[i]), # Usare per la Color Scale divergente (Blu > 0, Rosso < 0)
        "score_mds": float(score_mds[i]), # Usare per la Color Scale divergente
        "class_name": class_names[i]
    })

# Assicurati che la cartella di destinazione esista
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(OUTPUT_FILE, 'w') as f:
    json.dump(data_export, f)

print(f"Done! Updated JSON saved to: {OUTPUT_FILE}")