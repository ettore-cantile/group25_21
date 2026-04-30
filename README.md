# Visual Analytics Project: Comparative Analysis of PCA and Euclidean MDS

## 🎯 Project Goal
The exploration of high-dimensional datasets relies fundamentally on Dimensionality Reduction (DR) techniques to embed complex mathematical structures into visualizable 2D spaces. However, these representations inherently suffer from topological distortions.

The main goal of this interactive Visual Analytics dashboard is to rigorously evaluate and compare the spatial fidelity of **Principal Component Analysis (PCA)** against **Euclidean Multidimensional Scaling (MDS)**. Ultimately, the aim is to empower domain experts to visually track structural discrepancies and demonstrate how foundational algorithmic choices dictate the validity of natural data groupings.

---

## 📊 Datasets
The project uses the following benchmark datasets to observe algorithmic behaviors across varying degrees of feature overlap:
- **Wine Quality:** 13 features, 3 classes (moderate class entanglement).
- **Iris Flowers:** 4 features, 3 classes (highly separable topological structure).
- **User Knowledge Modeling:** 5 features, 4 classes (highly entangled, non-linear topology).

---

## 🏗️ System Architecture
The platform is built on a computationally decoupled client-server paradigm:
* **Data & Logic Layer (Python):** Handles offline pre-computation (standardization, DR, K-Means) and provides dynamic Flask microservices for real-time False Positive computations.
* **Presentation Layer (D3.js/HTML/CSS):** Provides a reactive, Coordinated Multiple Views (CMV) frontend for complex interactive exploration.

---

## 🛠️ Analytical Pipeline & Core Features

### 1. Quality Analysis & False Positives/Negatives Identification
The system moves beyond simple binary inclusion by calculating continuous, distance-weighted metrics:
* **Local Metrics:** Computes Distance-Weighted Precision (penalizing spatial False Positives), Recall (Lack of False Negatives), and an F-Score using high-dimensional K-Nearest Neighbor (K-NN) graphs.
* **Global Metrics:** Evaluates overall distance preservation using Trustworthiness, Continuity, and Normalized Stress.
* **Explicit False Positive Filtering:** Instead of just observing errors, the backend dynamically isolates artifact points using strategies like *Point-to-Point Relative Error Analysis*, *Iterative Stress Minimization*, and *Pseudo-Centroid Distance Evaluation*.

### 2. Labels, K-Means Clustering, and Discrepancies
To evaluate the discrepancies between abstract ground-truth labels and data-inherent clusters, the dashboard incorporates:
* **Cluster Agreement Flow:** A dedicated **Sankey Diagram** maps the migration of instances between ground truth, PCA-derived clusters, and MDS-derived clusters.
* **Centroid Contextualization:** "Show Centroids" mode renders mathematical cluster centers and links points to centroids to visualize intra-cluster variance.
* **Multivariate Filtering:** A **Parallel Coordinates Plot (PCP)** allows for axis-constrained selections that propagate back to the projection space.

### 3. Dimensional Fidelity Evaluation & Deep Diagnostics
The interface relies on a robust Focus + Context strategy:
* **Cross-View Brushing:** Geometric selections in scatterplots update Quality Gauges, the Sankey diagram, and the PCP simultaneously.
* **Single-Point Inspection:** Narrowing focus to a single instance provides:
    * A **Force-directed Neighbor Graph:** Reconstructs the high-dimensional neighborhood to identify "intruder" points.
    * A **Multidimensional Radar Chart:** Compares the instance profile against reference classes.

---

## 🐳 Local installation

### 1. Change target microservice
Set variable ```USE_LOCAL_API = true``` inside ```src/js/globals.js```


### 2. Build the Docker Image
From the root directory of the project (where the `Dockerfile` is located), run:

```bash
docker build -t visual-analytics-app .
```


### 3. Run the Docker Container
From the root directory of the project (where the `Dockerfile` is located), run:

```bash
docker run -p 8000:8000 -p 5000:5000 visual-analytics-app
```

### 4. Use the dashboard
Access through a browser at the following url:
```
http://localhost:8000
```

---

## Global site
The dashboard is also available at:
```
https://matteoventali.github.io/group25_21/
```

---

## 👥 Authors
- Matteo Ventali (1985026)
- Ettore Cantile (2026562)
- Leonardo Chiarparin (2016363)