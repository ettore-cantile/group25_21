import json
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics import euclidean_distances

# ==========================================
# 1. CONFIGURAZIONE UTENTE
# ==========================================
# Percorsi relativi alla posizione di questo file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(SCRIPT_DIR, '../dataset/wine_data.json')
CSV_FILENAME = os.path.join(SCRIPT_DIR, '../dataset/wine.csv')  # <--- Inserisci qui il nome del tuo file
TARGET_COLUMN = 'Producer'       # <--- Inserisci il nome della colonna che contiene le etichette/classi
                               #      (Se non ce l'hai, lascia una stringa vuota "")

# ==========================================
# 2. CARICAMENTO E PREPARAZIONE DATI
# ==========================================
print(f"Caricamento di {CSV_FILENAME}...")
try:
    df = pd.read_csv(CSV_FILENAME)
except FileNotFoundError:
    print("ERRORE: File non trovato. Controlla il nome o il percorso.")
    exit()

# Rimuoviamo righe con valori mancanti (NaN) per evitare crash
df = df.dropna()

# Gestione della colonna Target (Etichette)
if TARGET_COLUMN and TARGET_COLUMN in df.columns:
    # Salviamo le etichette
    class_names = df[TARGET_COLUMN].astype(str).values
    # Creiamo il dataset numerico rimuovendo la colonna target
    data_df = df.drop(columns=[TARGET_COLUMN])
else:
    # Se non c'è target, usiamo un placeholder
    class_names = ["Item"] * len(df)
    data_df = df

# SELEZIONE SOLO COLONNE NUMERICHE
# PCA e MDS funzionano solo con numeri. Se hai colonne ID o Nomi, verranno scartate qui.
X_original = data_df.select_dtypes(include=[np.number])
X = X_original.values

print(f"Dati caricati: {X.shape[0]} righe, {X.shape[1]} colonne numeriche.")

# Standardizzazione (Media=0, Std=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# 3. CALCOLO PCA E MDS
# ==========================================
print("Calcolo PCA...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Calcolo MDS (potrebbe richiedere tempo)...")
mds = MDS(n_components=2, dissimilarity='euclidean', random_state=42, normalized_stress='auto', n_init=4)
X_mds = mds.fit_transform(X_scaled)

# ==========================================
# 4. CALCOLO METRICHE DI ERRORE
# ==========================================
print("Calcolo distanze e score di errore...")

# Distanze originali (High-Dim)
D_high = euclidean_distances(X_scaled)
# Distanze proiettate (Low-Dim)
D_pca = euclidean_distances(X_pca)
D_mds = euclidean_distances(X_mds)

# Normalizzazione [0, 1]
min_max = MinMaxScaler()
D_high_norm = min_max.fit_transform(D_high)
D_pca_norm = min_max.fit_transform(D_pca)
D_mds_norm = min_max.fit_transform(D_mds)

# Calcolo Score: (Distanza 2D - Distanza Reale)
E_pca = D_pca_norm - D_high_norm
E_mds = D_mds_norm - D_high_norm

score_pca = np.mean(E_pca, axis=1)
score_mds = np.mean(E_mds, axis=1)

# ==========================================
# 5. ESPORTAZIONE JSON
# ==========================================
data_export = []
for i in range(len(X)):
    data_export.append({
        "id": i,
        "pca_x": float(X_pca[i, 0]),
        "pca_y": float(X_pca[i, 1]),
        "mds_x": float(X_mds[i, 0]),
        "mds_y": float(X_mds[i, 1]),
        "score_pca": float(score_pca[i]),
        "score_mds": float(score_mds[i]),
        "class_name": class_names[i] # Usa le etichette dal CSV
    })

 # Mantengo lo stesso nome così l'HTML funziona senza modifiche
with open(OUTPUT_FILE, 'w') as f:
    json.dump(data_export, f)

print(f"Fatto! File '{OUTPUT_FILE}' aggiornato con i dati del CSV.")