import os
import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler

# --- PATH CONFIGURATION ---
# The script is located in src/python, so we go up to reach dataset and json folders
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, '..', '..', 'dataset', 'wine.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'json')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'step1_results.json')

# --- ALGORITHM PARAMETERS ---
EPS = 1e-5
USE_KNN = True
K_NEIGHBORS = 15  # Number of neighbors to define relationships (default for UMAP/t-SNE)

def calculate_metrics():
    # 1. Create output folder if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Dataset Loading
    print(f"Loading dataset from: {DATASET_PATH}")
    # pandas read_csv automatically uses the first row as column headers by default
    df = pd.read_csv(DATASET_PATH)
    
    # The label is the 'producer' (the first column)
    labels = df.iloc[:, 0].values
    # The features are all columns except the first one
    X = df.iloc[:, 1:].values
    
    unique_labels = np.unique(labels)
    n_samples = len(X)

    # Standardize the features (Crucial for Euclidean distance)
    print("Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Euclidean Distance Matrix Calculation
    print("Calculating Euclidean distances and weights...")
    # Use the scaled features for distance calculation
    dist_matrix = squareform(pdist(X_scaled, metric='euclidean'))

    # 4. Weight Calculation: W = 1 / (dist + eps)
    weights = 1.0 / (dist_matrix + EPS)
    np.fill_diagonal(weights, 0) # Remove self-loops (distance to oneself)

    # 5. Graph Construction (Optional: K-NN filtering)
    if USE_KNN:
        # Keep only the top K weights for each row (the K nearest neighbors)
        for i in range(n_samples):
            # Indices of the points that are NOT among the top K
            idx_to_zero = np.argsort(weights[i])[:-K_NEIGHBORS]
            weights[i, idx_to_zero] = 0.0

    # 6. Calculate Precision, Recall, and F-score for each point
    print("Calculating Precision and Recall...")
    points_data = []
    
    for i in range(n_samples):
        current_label = labels[i]
        
        # Masks to identify the same class (TP) and different classes (FP)
        same_class_mask = (labels == current_label)
        diff_class_mask = (labels != current_label)
        
        # Find who is actually connected in the graph (weight > 0)
        connected_mask = (weights[i] > 0)
        
        # --- PRECISION ---
        # True Positives: connected neighbors with the SAME label
        tp_mask = connected_mask & same_class_mask
        tp_weight = np.sum(weights[i, tp_mask])
        
        # False Positives: connected neighbors with a DIFFERENT label
        fp_mask = connected_mask & diff_class_mask
        fp_weight = np.sum(weights[i, fp_mask])
        
        total_attraction = tp_weight + fp_weight
        precision = (tp_weight / total_attraction) if total_attraction > 0 else 0.0
        
        # --- RECALL ---
        # Classical definition: how many of my class did I "capture" in my neighborhood?
        # False Negatives: elements of my class to which I am NOT connected
        fn_mask = same_class_mask & ~connected_mask
        fn_mask[i] = False # Exclude ourselves
        
        # We use cardinality for Recall (as per the paper)
        tp_count = np.sum(tp_mask)
        fn_count = np.sum(fn_mask)
        
        total_relevant = tp_count + fn_count
        recall = (tp_count / total_relevant) if total_relevant > 0 else 0.0
        
        # --- F-SCORE ---
        if precision + recall > 0:
            f_score = 2 * (precision * recall) / (precision + recall)
        else:
            f_score = 0.0
            
        points_data.append({
            "id": i,
            "label": str(current_label),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f_score": round(f_score, 4)
        })

    # 7. Initial Assessment: Global F-score (Average of the averages per class)
    class_fscores = []
    for lbl in unique_labels:
        fscores_in_class = [p["f_score"] for p in points_data if p["label"] == str(lbl)]
        if fscores_in_class:
            class_fscores.append(np.mean(fscores_in_class))
            
    global_f_score = np.mean(class_fscores) if class_fscores else 0.0

    output_data = {
        "metadata": {
            "dataset": "wine.csv",
            "k_neighbors_used": K_NEIGHBORS if USE_KNN else "All (Dense)",
            "global_assessment": {
                "global_f_score": round(global_f_score, 4),
                "message": f"{round(global_f_score*100, 1)}% of the expected cluster structure is supported by the original relationships."
            }
        },
        "points": points_data
    }

    # 8. Save to JSON
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output_data, f, indent=4)
        
    print(f"Processing completed! Results saved in: {OUTPUT_FILE}")
    print(output_data["metadata"]["global_assessment"]["message"])

if __name__ == "__main__":
    calculate_metrics()