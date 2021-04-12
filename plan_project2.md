# QUESTION 1: Report the dimensions of the TF-IDF matrix you get.

# QUESTION 2: Report the contingency table of your clustering result. You may use
the provided plotmat.py to visualize the matrix.

# QUESTION 3: Report the 5 measures above for the K-means clustering results you
get


# QUESTION 4: Report the plot of the percent of variance the top r principle components can retain v.s. r, for r = 1 to 1000.

# QUESTION 5:
Let r be the dimension that we want to reduce the data to (i.e. n_components).
Try r = 1, 2, 3, 5, 10, 20, 50, 100, 300, and plot the 5 measure scores v.s. r for both
SVD and NMF.
Report a good choice of r for SVD and NMF respectively.
Note: In the choice of r, there is a trade-off between the information preservation, and better performance of
k-means in lower dimensions.
# QUESTION 6: How do you explain the non-monotonic behavior of the measures as r
increases?

# QUESTION 7: Visualize the clustering results for:
• SVD with your choice of r
• NMF with your choice of r
# QUESTION 8: What do you observe in the visualization? How are the data points of the
two classes distributed? Is the data distribution ideal for K-Means clustering?

# QUESTION 9: Load documents with the same configuration as in Question 1, but for ALL
20 categories. Construct the TF-IDF matrix, reduce its dimensionality properly using either
NMF or SVD, and perform K-Means clustering with n_components=20 . Visualize the
contingency matrix and report the five clustering metrics.
There is a mismatch between cluster labels and class labels. For example, the cluster #3 may
correspond to the class #8. As a result, the high-value entries of the 20 × 20 contingency
matrix can be scattered around, making it messy to inspect, even if the clustering result is not
bad.
One can use scipy.optimize.linear_sum_assignment to identify the best-matching
cluster-class pairs, and permute the columns of the contingency matrix accordingly. See below
for an example:

```python
import numpy as np
from plotmat import plot_mat # using the provided plotmat.py
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels, clustering_labels)
rows, cols = linear_sum_assignment(cm, maximize=True)
plot_mat(cm[rows[:, np.newaxis], cols], xticklabels=cols, yticklabels=rows, size=(15,15))
```

# QUESTION 10: Kullback-Leibler Divergence for NMF
By default sklearn.decomposition.NMF uses Frobenius norm as its cost function. Another
choice is Kullback-Leibler divergence. It is shown that with this cost function, NMF is equivalent to Probabilistic Latent Semantic Analysis (PLSA)[4]. Try Kullback-Leibler divergence
for NMF and see whether it helps with the clustering of our text data. Report the five
evaluation metrics.


# QUESTION 11: Use UMAP to reduce the dimensionality of the 20 categories TF-IDF
matrix, and apply K-Means clustering with n_components=20 .
Find a good n_components choice for UMAP, and compare the performance of two metrics by setting metric="euclidean" and metric="cosine" respectively.

Report the permuted contingency matrix and the five clustering evaluation metrics
for "euclidean" and "cosine".


# QUESTION 12: Analyze the contingency matrix.
From the contingency matrices, identify the categories that are prone to be confused with each other. Does the result make sense?


# QUESTION 13: Use UMAP to reduce the dimensionality properly, and perform Agglomerative clustering with n_clusters=20 . Compare the performance of “ward” and “single”
linkage criteria.
Report the five clustering evaluation metrics for each case.


# QUESTION 14: Apply DBSCAN and HDBSCAN on UMAP-transformed 20-category data.
Use min_cluster_size=100 .
Experiment on the hyperparameters and report your findings in terms of the five
clustering evaluation metrics.


# QUESTION 15: Contingency matrix
Plot the permuted contingency matrix for the best clustering model from Question 14.
How many clusters are given by the model? What does “-1” mean for the clustering labels?
Interpret the contingency matrix considering the answer to these questions.



# QUESTION 16: Report your process:
• data acquiring,
• feature engineering (doesn’t need to be the same as those in part 1),
• clustering,
• performance evaluation






