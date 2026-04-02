import os
import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION ---
DATASETS = ["wine", "iris", "user_knowledge"]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define where the label column is located for each dataset (0 = first, -1 = last)
LABEL_INDEX = {
    "wine": 0,
    "iris": -1,
    "user_knowledge": -1
}

EPS = 1e-5
USE_KNN = True
K_NEIGHBORS = 15 

def calculate_metrics_for_dataset(dataset_name):
    print(f"\n--- Processing Relationships for: {dataset_name.upper()} ---")
    
    dataset_path = os.path.join(BASE_DIR, '..', '..', 'dataset', f'{dataset_name}.csv')
    output_dir = os.path.join(BASE_DIR, '..', 'json', dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'step1_results.json')

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset {dataset_path} not found. Skipping.")
        return

    # Load Dataset
    df = pd.read_csv(dataset_path)
    
    # Extract labels and features dynamically based on known index
    target_idx = LABEL_INDEX.get(dataset_name, -1)
    labels = df.iloc[:, target_idx].values
    X = df.drop(df.columns[target_idx], axis=1).values
    
    unique_labels = np.unique(labels)
    n_samples = len(X)

    # Standardize Features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Compute Euclidean Distance Matrix and Weights
    dist_matrix = squareform(pdist(X_scaled, metric='euclidean'))
    weights = 1.0 / (dist_matrix + EPS)
    np.fill_diagonal(weights, 0) # Remove self-loops

    # Build the K-NN Graph
    if USE_KNN:
        for i in range(n_samples):
            idx_to_zero = np.argsort(weights[i])[:-K_NEIGHBORS]
            weights[i, idx_to_zero] = 0.0

    points_data = []
    for i in range(n_samples):
        current_label = labels[i]
        same_class_mask = (labels == current_label)
        diff_class_mask = (labels != current_label)
        
        connected_mask = (weights[i] > 0)
        connected_indices = np.where(connected_mask)[0].tolist()
        if i in connected_indices:
            connected_indices.remove(i)
        
        # Precision calculation
        tp_mask = connected_mask & same_class_mask
        tp_weight = np.sum(weights[i, tp_mask])
        fp_mask = connected_mask & diff_class_mask
        fp_weight = np.sum(weights[i, fp_mask])
        
        total_attraction = tp_weight + fp_weight
        precision = (tp_weight / total_attraction) if total_attraction > 0 else 0.0
        
        # Recall calculation
        fn_mask = same_class_mask & ~connected_mask
        fn_mask[i] = False 
        
        tp_count = np.sum(tp_mask)
        fn_count = np.sum(fn_mask)
        total_relevant = tp_count + fn_count
        recall = (tp_count / total_relevant) if total_relevant > 0 else 0.0
        
        # F-Score calculation
        if precision + recall > 0:
            f_score = 2 * (precision * recall) / (precision + recall)
        else:
            f_score = 0.0
            
        points_data.append({
            "id": i,
            "label": str(current_label),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f_score": round(f_score, 4),
            "neighbors": connected_indices
        })

    # Global F-score Assessment
    class_fscores = []
    for lbl in unique_labels:
        fscores_in_class = [p["f_score"] for p in points_data if p["label"] == str(lbl)]
        if fscores_in_class:
            class_fscores.append(np.mean(fscores_in_class))
            
    global_f_score = np.mean(class_fscores) if class_fscores else 0.0

    output_data = {
        "metadata": {
            "dataset": f"{dataset_name}.csv",
            "k_neighbors_used": K_NEIGHBORS if USE_KNN else "All (Dense)",
            "global_assessment": {
                "global_f_score": round(global_f_score, 4)
            }
        },
        "points": points_data
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)
        
    print(f"Saved: {output_file}")

if __name__ == "__main__":
    for ds in DATASETS:
        calculate_metrics_for_dataset(ds)