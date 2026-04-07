# Visual Analytics Project: PCA vs. Euclidean MDS

## 🎯 Project Goal
The main goal is to evaluate the **fidelity of PCA** (assessment and judgment) with respect to Euclidean distance and demonstrate how **Euclidean MDS** is better. Ultimately, the aim is to improve the quality of what is being observed and evaluate the usefulness of the visualization.

**Main Focus:** Euclidean distance (PCA vs. Euclidean MDS).

---

## 📊 Dataset
The project uses the following datasets:
- **Wine**
- **Iris**
- **User Knowledge Modelling**

---

## 🛠️ Development Pipeline (Three Main Steps)

### 1. Quality Analysis & False Positives/Negatives
Report the quality of PCA (what goes wrong; what the false positives are) and Euclidean MDS (both false positives and negatives) with respect to the dataset.
* **Visualization:** Provide a double (and synchronized) visualization of the two scatterplots.
* **Color Scale:** Implement a color scale useful for representing the positive or negative falsity index of the points.
* **Quality Metrics:** Include metrics such as a global score representing the sum of the involved points.
* **Initial Assessment:** Provide a preliminary judgment on the dataset (e.g., "80% of the points match"). 
* **Defining Weights:** False positives and negatives (blue/red) must be given an appropriate **definition**. You must use metrics that are not simply "on/off" but instead consider their weight (i.e., *how much* of a false positive or negative they are).

### 2. Labels, K-Means Clustering, and Discrepancies
Introduce labels (abstract ones, like wine producers, and data-dependent ones). 
* **Cluster Analysis:** The objective is to evaluate the discrepancies between abstract clusters (which do not depend on the Euclidean distance calculation and are therefore independent of the dataset values) and data-inherent clusters, using techniques like **K-Means**.
* **Visualizing Anomalies:** The visualization must offer the possibility to show the portions of the clusters that do not seem to belong to their assigned cluster via centroids (Note: some tools/references should have already been published by Santucci himself).
* **Additional Tools:** Quality metrics and **filters** (for the removal of false positives or negatives) must be present, utilizing both Euclidean MDS and PCA.

### 3. Dimensional Fidelity Evaluation
Highlight how faithful the obtained two-dimensional projection is to the original multidimensional clusters/spaces.
* **Switch Functionality:** The requirement is to create a switch that, through the use of K-Means, allows the user to visualize the possible intersections/areas of interest among the mentioned elements (the 3 original clusters, the "producers" clusters, etc.).

---

## Authors
- Matteo Ventali (1985026)
- Ettore Cantile (2026562)
- Leonardo Chiarparin (2016363)