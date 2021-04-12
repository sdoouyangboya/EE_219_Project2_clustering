import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from pprint import pprint
from itertools import product
import json

from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
from plotmat import plot_mat
from helpers import get_tf_idf,get_contingency,get_all_data,get_KMeans,get_cluster_metrics, LSI_builtin, get_NMF, close_factors,get_binary_dataset,get_LSI_matrix,get_UMAP_matrix,get_AgglomerativeClustering, HDBSCAN,get_KL_NMF,DBSCAN_builtin
import umap


np.random.seed(0)
random.seed(0)



def q1():
	data = get_binary_dataset()
	X_tfidf = get_tf_idf(data.data)
	print(f"TF-IDF Matrix Shape (all data): {X_tfidf.shape}")


def q2():
	data = get_binary_dataset()
	X_tfidf = get_tf_idf(data.data)
	n_clusters = 2
	predictions,kmeans = get_KMeans(X_tfidf,n_clusters=n_clusters)
	A = get_contingency(data.target,predictions)
	plot_mat(A,title=f"Contingency Table, K-Means",pic_fname="q2.pdf",sizemult=2.5,ylabel="True Class",xlabel="Cluster Index")
	
def q3():
	data = get_binary_dataset()
	X_tfidf = get_tf_idf(data.data)
	n_clusters = 2
	predictions,kmeans = get_KMeans(X_tfidf,n_clusters=n_clusters)
	pprint(get_cluster_metrics(data.target,predictions))

def q4():
	data = get_binary_dataset()
	X_tfidf = get_tf_idf(data.data)
	rmax=1000
	X_lsi,svd = LSI_builtin(X_tfidf,rmax)
	expl_variances = { r : 100*sum(svd.explained_variance_ratio_[0:r]) for r in range(1,rmax+1)}
	fig,ax = plt.subplots()
	ax.plot(tuple(expl_variances.keys()), tuple(expl_variances.values()))
	ax.set_xlabel("Number of Principal Components $r$")
	ax.set_ylabel("Percent of Explained Variance in First $r$ Principal Components")
	ax.set_title("Percent Variance Explained by Top $r$ Principal Components vs. $r$")
	plt.tight_layout()
	plt.savefig("figures/q4_variance_explained_vs_pca_dimension.pdf")
	plt.close()

def q5():
	data = get_binary_dataset()
	X_tfidf = get_tf_idf(data.data)
	rmax = 1000
	n_clusters = 2
	X_lsi = get_LSI_matrix(X_tfidf,rmax)
	rvals =  (1, 2, 3, 5, 10, 20, 50, 100, 300)
	metrics_lsi, metrics_nmf = [], []
	for r in rvals:
		predictions_lsi,kmeans_lsi = get_KMeans(X_lsi[:,:(r+1)],n_clusters=n_clusters)
		metrics_lsi.append(get_cluster_metrics(data.target,predictions_lsi))

		predictions_NMF,kmeans_nmf = get_KMeans(get_NMF(X_tfidf,r), n_clusters=n_clusters)
		metrics_nmf.append(get_cluster_metrics(data.target,predictions_NMF))

	df_lsi = pd.DataFrame(metrics_lsi)
	df_nmf = pd.DataFrame(metrics_nmf)
	columns = df_nmf.columns
	# for df in (df_lsi,df_nmf):
	# df['Dimension'] = rvals
	nrows,ncols = close_factors(len(columns))
	fig,axs = plt.subplots(nrows=nrows,ncols=ncols,sharex=True)
	for (col,ax) in zip(columns,fig.axes):
		for method,df in {"LSI": df_lsi,"NMF" : df_nmf}.items():
			ax.plot(rvals,df[col],label=method)
			ax.set_title(f"{col} vs.\nReduced Dimension $r$",fontsize=8)
			ax.set_ylabel(f"{col}", fontsize=6)
			ax.legend()
			plt.xticks(fontsize=8)
			plt.yticks(fontsize=8)
	# plt.xlabel("Reduced Dimension $r$")
	fig.add_subplot(111, frameon=False)
	plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
	plt.xlabel("Reduced Dimension $r$")
	plt.tight_layout() #plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
	plt.savefig("figures/q5_purity_vs_dimension.pdf")
	plt.close()


def q7():
	data= get_binary_dataset()
	n_classes = len(set(data.target))
	n_samples = len(data.target)
	filename_base=f"figures/q7_kmeans_labels_predictions_2d_{n_classes}_classes"

	target = data.target #ground truth
	X_tfidf = get_tf_idf(data.data)
	# dimensionality reduction
	r=20
	n_clusters=2
	fig,axs = plt.subplots(nrows=2,ncols=2,sharex=True,sharey=True)
	performance_outfile = open(f"{filename_base}_performance.txt", "w")
	performance_outfile.write(f"{n_samples} Samples, {n_classes} Classes, {n_clusters} Clusters\n")
	for (method,transformer),axlist in zip({"LSI": get_LSI_matrix, "NMF" : get_NMF}.items(),axs):
		X_reduced = transformer(X_tfidf,r)
		predictions, _ = get_KMeans(X_reduced,n_clusters=n_clusters)
		X_2d = get_LSI_matrix(X_reduced,2)
		labels = {" Predictions" : predictions, "Ground Truth": target}
		performance_outfile.write(f"\nClustering Performance After {method} Dimensionality Reduction:\n")
		pprint(get_cluster_metrics(target,predictions),performance_outfile)
		for ax,(label_type,label) in zip(axlist,labels.items()):
			ax.scatter(
				X_2d[:,0],X_2d[:,1],
				c=label,
				s=1,
				alpha = 0.5,
				linewidths=0,
				edgecolors=None
			)
			ax.set_title(f"K-Means {label_type} Using {method}\n{n_classes} Classes, $\\in R^2$ via SVD",fontsize=8)
			ax.set_xticks([])
			ax.set_yticks([])
	plt.tight_layout() #plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
	plt.savefig(f"{filename_base}.png", dpi=350, transparent=False)
	plt.savefig(f"{filename_base}.pdf", transparent=False)
	plt.close()
	performance_outfile.close()

def q9():
	data = get_all_data()
	X_tfidf = get_tf_idf(data.data)
	n_classes = len(set(data.target))
	n_samples = len(data.target)
	r_components = 20
	n_clusters = 20
	target = data.target 

	filename_base=f"figures/q9_kmeans_labels_predictions_2d_{n_classes}_classes_{r_components}_components"
	performance_outfile = open(f"{filename_base}_performance.txt", "w")
	performance_outfile.write(f"{n_samples} Samples, {n_classes} Classes, {n_clusters} Clusters, {r_components}Components\n")
	for (method,transformer) in {"LSI": get_LSI_matrix, "NMF" : get_NMF}.items():
		X_reduced = transformer(X_tfidf,r_components)
		predictions, _ = get_KMeans(X_reduced,n_clusters=n_clusters)
		performance_outfile.write(f"\nClustering Performance After {method} Dimensionality Reduction:\n")
		pprint(get_cluster_metrics(target,predictions),performance_outfile)
		cm = confusion_matrix(data.target, predictions)
		rows, cols = linear_sum_assignment(cm, maximize=True)
		plot_mat(cm[rows[:, np.newaxis], cols], xticklabels=cols,yticklabels=rows, size=(15,15), pic_fname = f"figures/q9_{method}")
		pass

	performance_outfile.close()
	# # Do K-Means clustering.
	# predictions,kmeans = get_KMeans(X_tfidf,n_clusters=n_clusters)
	# # Print out the five clustering metrics.
	# pprint(get_cluster_metrics(data.target,predictions))
	# # Generate confusion matrix.
	# cm = confusion_matrix(data.target, predictions)
	# rows, cols = linear_sum_assignment(cm, maximize=True)
	# plot_mat(cm[rows[:, np.newaxis], cols], xticklabels=cols,yticklabels=rows, size=(15,15), pic_fname = "figures/q9")





def q10():
	data = get_all_data(categories = None)
	X_tfidf = get_tf_idf(data.data)
	n_classes = len(set(data.target))
	n_samples = len(data.target)
	r_components = 20
	n_clusters = 20
	target = data.target #ground truth
	# X_tfidf = get_tf_idf(data.data)

	filename_base=f"figures/q10_kmeans_labels_predictions_2d_{n_classes}_classes_{r_components}_components"
	performance_outfile = open(f"{filename_base}_performance.txt", "w")
	performance_outfile.write(f"{n_samples} Samples, {n_classes} Classes, {n_clusters} Clusters, {r_components}Components\n")
	for (method,transformer) in {"LSI": get_LSI_matrix,"KL_NMF": get_KL_NMF, "FB_NMF" : get_NMF}.items():

		X_reduced = transformer(X_tfidf,r_components)
		predictions, _ = get_KMeans(X_reduced,n_clusters=n_clusters)

		performance_outfile.write(f"\nClustering Performance After {method} Dimensionality Reduction:\n")
		pprint(get_cluster_metrics(target,predictions),performance_outfile)

	performance_outfile.close()


def q11():
	data = get_all_data(categories = None)
	X_tfidf = get_tf_idf(data.data)
	n_classes = len(set(data.target))
	n_samples = len(data.target)
	r_components = 20
	n_clusters = 20
	target = data.target #ground truth
	X_tfidf = get_tf_idf(data.data)
	filename_base=f"figures/q11_kmeans_labels_predictions_2d_{n_classes}_classes"

	X_eu_umap = get_UMAP_matrix(X_tfidf,r_components,'euclidean')
	predictions,kmeans = get_KMeans(X_eu_umap,n_clusters=n_clusters)
	A = get_contingency(data.target,predictions)
	rows, cols = linear_sum_assignment(A, maximize=True)
	plot_mat(A[rows[:, np.newaxis], cols],title=f"Contingency Table, K-Means",pic_fname="q11_euclidean.pdf",
			xticklabels=cols, yticklabels=rows, size=(15,15))

	X_co_umap = get_UMAP_matrix(X_tfidf,r_components,'cosine')
	predictions,kmeans = get_KMeans(X_co_umap,n_clusters=n_clusters)
	B = get_contingency(data.target,predictions)
	rows, cols = linear_sum_assignment(B, maximize=True)
	plot_mat(B[rows[:, np.newaxis], cols],title=f"Contingency Table, K-Means",pic_fname="q11_cosine.pdf",
			 xticklabels=cols, yticklabels=rows, size=(15,15))

	metric = ['euclidean','cosine']
	performance_outfile = open(f"{filename_base}_performance.txt", "w")
	performance_outfile.write(f"{n_samples} Samples, {n_classes} Classes, {n_clusters} Clusters, {r_components}Components\n")
	# for (method,transformer),axlist,metric in zip({"UMAP_euclidean": get_UMAP_matrix,"UMAP_cosine": get_UMAP_matrix}.items(),axs,metric):
	for (method,transformer),metric in zip({"UMAP_euclidean": get_UMAP_matrix,"UMAP_cosine": get_UMAP_matrix}.items(),metric):
		X_reduced = transformer(X_tfidf,r_components,metric)
		predictions, _ = get_KMeans(X_reduced,n_clusters=n_clusters)

		performance_outfile.write(f"\nClustering Performance After {method} Dimensionality Reduction:\n")
		pprint(get_cluster_metrics(target,predictions),performance_outfile)

	performance_outfile.close()

def q11_2():

	for r_components in [2,5,10,20,30,40,60,80,120,160,320]:
		data = get_all_data(categories = None)
		X_tfidf = get_tf_idf(data.data)
		n_classes = len(set(data.target))
		n_samples = len(data.target)
		n_clusters = 20
		target = data.target #ground truth
		X_tfidf = get_tf_idf(data.data)
		filename_base=f"figures/q11_2_kmeans_labels_predictions_2d_{n_classes}_classes_{r_components}components"

		metric = ['euclidean','cosine']
		performance_outfile = open(f"{filename_base}_performance.txt", "w")
		performance_outfile.write(f"{r_components} Components, {n_samples} Samples, {n_classes} Classes, {n_clusters} Clusters\n")
		for (method,transformer),metric in zip({"UMAP_euclidean": get_UMAP_matrix,"UMAP_cosine": get_UMAP_matrix}.items(),metric):
			X_reduced = transformer(X_tfidf,r_components,metric)
			predictions, _ = get_KMeans(X_reduced,n_clusters=n_clusters)

			# labels = {" Predictions" : predictions, "Ground Truth": target}
			performance_outfile.write(f"\nClustering Performance After {method} Dimensionality Reduction:\n")
			pprint(get_cluster_metrics(target,predictions),performance_outfile)

		performance_outfile.close()

def q13():

	data = get_all_data(categories = None)
	n_classes = len(set(data.target))
	n_samples = len(data.target)
	r_components = 120 # Should this be n_components??
	n_clusters = 20
	target = data.target #ground truth
	X_tfidf = get_tf_idf(data.data)
	filename_base=f"figures/q13_kmeans_labels_predictions_2d_{n_classes}_classes_{r_components}components"

	metric = ['ward','single']
	performance_outfile = open(f"{filename_base}_performance.txt", "w")
	performance_outfile.write(f"{n_samples} Samples, {n_classes} Classes, {n_clusters} Clusters\n")
	for (method,transformer),metric in zip({"Ward": get_AgglomerativeClustering,"Single": get_AgglomerativeClustering}.items(),metric):
		X_reduced = get_UMAP_matrix(X_tfidf,r_components,'cosine')
		predictions = transformer(X_reduced,n_clusters,metric)

		#labels = {" Predictions" : predictions, "Ground Truth": target}
		performance_outfile.write(f"\nClustering Performance After {method} Dimensionality Reduction:\n")
		pprint(get_cluster_metrics(target,predictions),performance_outfile)

	performance_outfile.close()

def q14():
	data = get_all_data(categories = None)
	n_classes = len(set(data.target))
	n_samples = len(data.target)
	n_components = 120
	n_clusters = 20
	target = data.target
	X_tfidf = get_tf_idf(data.data)
	filename_base=f"figures/q14_hdbscan_labels_predictions_2d_{n_classes}_classes"
	performance_outfile = open(f"{filename_base}_performance.txt", "w")
	performance_outfile.write(f"{n_samples} Samples, {n_classes} Classes, {n_clusters} Clusters\n")
	X_reduced = get_UMAP_matrix(X_tfidf,n_components,'cosine')

	for min_samples in [None, 1, 2, 3, 5, 10, 20, 40, 60]:
		for alpha in [.2,.4,.6,.8,1.0,1.2,1.4,1.6,1.8,2.0]:
			for metric in ['euclidean', 'l1', 'l2']:
				predictions, clusterer = HDBSCAN(X_reduced,min_samples,alpha,metric)
				# print(clusterer.labels_.max())
				performance_outfile.write(f"\nClustering Performance After UMAP Dimensionality Reduction with Parameters:\n")
				performance_outfile.write(f"\nmin_samples: {min_samples}; alpha: {alpha}; metric: {metric}\n")
				pprint(get_cluster_metrics(target,predictions), performance_outfile)
	performance_outfile.close()

	filename_base=f"figures/q14_dbscan_labels_predictions_2d_{n_classes}_classes"
	performance_outfile = open(f"{filename_base}_performance.txt", "w")
	performance_outfile.write(f"{n_samples} Samples, {n_classes} Classes, {n_clusters} Clusters\n")
	for eps in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.8, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
		for metric in ['cosine', 'euclidean', 'l1', 'l2']:
			predictions, clusterer = DBSCAN_builtin(X_reduced,eps,metric)
			performance_outfile.write(f"\nClustering Performance After UMAP Dimensionality Reduction with Parameters:\n")
			performance_outfile.write(f"\neps: {eps}; metric: {metric}\n")
			pprint(get_cluster_metrics(target,predictions), performance_outfile)
	performance_outfile.close()

def q15():
    # Can Peter please do this after looking over the data for q14()? This question is simply getting the contingency matrix for the best result.
    data = get_all_data(categories = None)
    n_classes = len(set(data.target))
    n_samples = len(data.target)
    n_components = 120
    n_clusters = 20
    target = data.target
    X_tfidf = get_tf_idf(data.data)
    filename_base=f"figures/q15_dbscan_labels_predictions_2d_{n_classes}_classes_best" # CHANGE TO DB/HDB ACCORDINGLY
    performance_outfile = open(f"{filename_base}_performance.txt", "w")
    performance_outfile.write(f"{n_samples} Samples, {n_classes} Classes, {n_clusters} Clusters\n")
    X_reduced = get_UMAP_matrix(X_tfidf,n_components,'cosine')

    # If HDBSCAN performed better.
    # min_samples =
    # alpha =
    # metric =
    # predictions, clusterer = HDBSCAN(X_reduced,min_samples,alpha,metric)
    # performance_outfile.write(f"\nClustering Performance After UMAP Dimensionality Reduction with Parameters:\n")
    # performance_outfile.write(f"\nmin_samples: {min_samples}; alpha: {alpha}; metric: {metric}\n")
    # pprint(get_cluster_metrics(target,predictions), performance_outfile)
    # performance_outfile.close()

    # IF DBSCAN performed better.
    eps = 0.6
    metric = 'euclidean'
    predictions, clusterer = DBSCAN_builtin(X_reduced,eps,metric)
    performance_outfile.write(f"\nClustering Performance After UMAP Dimensionality Reduction with Parameters:\n")
    performance_outfile.write(f"\neps: {eps}; metric: {metric}\n")
    pprint(get_cluster_metrics(target,predictions), performance_outfile)
    performance_outfile.close()
    
    labels = predictions
    shape = predictions.shape
    for x in range(0, shape[0]):
        if predictions[x] == -1:
            predictions[x] = 19
            
    
    
    # For both
    cm = confusion_matrix(data.target, predictions)
    #rows, cols = linear_sum_assignment(cm, maximize=True)
    #plot_mat(cm[rows[:, np.newaxis], cols], xticklabels=cols,yticklabels=rows, size=(15,15), pic_fname = f"figures/q15")
    plot_mat(cm, size=(15,15), pic_fname = f"figures/q15")

    pass


def q16():
	# Finding the effective dimension of the data through inspection of the top singular values of TF-IDF matrix.

	#Load the data.
	train = pd.read_csv("input/BBC News Train.csv",header=0)
	#Add an additional category that changes the string category to an integer.
	train['category_id'] = train['Category'].factorize()[0]
	# Store the data and labels.
	data = train.Text
	labels = train.category_id
	# TF-IDF with min_df=3 
	X_tfidf = get_tf_idf(data)
	print(X_tfidf.shape)

	rmax=1000
	X_lsi,svd = LSI_builtin(X_tfidf,rmax)
	expl_variances = { r : 100*sum(svd.explained_variance_ratio_[0:r]) for r in range(1,rmax+1)}
	fig,ax = plt.subplots()
	ax.plot(tuple(expl_variances.keys()), tuple(expl_variances.values()))
	ax.set_xlabel("Number of Principal Components $r$")
	ax.set_ylabel("Percent of Explained Variance in First $r$ Principal Components")
	ax.set_title("Percent Variance Explained by Top $r$ Principal Components vs. $r$")
	plt.tight_layout()
	plt.savefig("figures/q16_variance_explained_vs_pca_dimension.pdf")
	plt.close()


def q16_1():

	#Determining best r value and whether to use LSI or NMF. 

	#Load the data.
	train = pd.read_csv("input/BBC News Train.csv",header=0)
	test = pd.read_csv("input/BBC News Test.csv")
	
	#Add an additional category that changes the string category to an integer.
	train['category_id'] = train['Category'].factorize()[0]
	category_id_df = train[['Category', 'category_id']].drop_duplicates().sort_values('category_id')
	#Plot the count of data for each category to make sure the data is balanced.
	plot = train.groupby('Category').category_id.count().plot.barh(fontsize=8, figsize=(10,10), title='Distribution of Data for Each Category')
	fig = plot.get_figure()
	# Save the figure.
	fig.savefig("figures/q16_category_bar_graph")
	# Store the data and labels.
	data = train.Text
	labels = train.category_id
	# TF-IDF with min_df=3 
	X_tfidf = get_tf_idf(data)
	print(X_tfidf.shape)
	# Determine best r value and whether to use LSI or NMF. 
	rmax = 1000
	n_clusters = 5
	X_lsi = get_LSI_matrix(X_tfidf,rmax)
	rvals =  (1, 2, 3, 5, 10, 20, 50, 100, 300)
	metrics_lsi, metrics_nmf = [], []
	for r in rvals:
		predictions_lsi,kmeans_lsi = get_KMeans(X_lsi[:,:(r+1)],n_clusters=n_clusters)
		metrics_lsi.append(get_cluster_metrics(labels,predictions_lsi))

		predictions_NMF,kmeans_nmf = get_KMeans(get_NMF(X_tfidf,r), n_clusters=n_clusters)
		metrics_nmf.append(get_cluster_metrics(labels,predictions_NMF))

	df_lsi = pd.DataFrame(metrics_lsi)
	df_nmf = pd.DataFrame(metrics_nmf)
	columns = df_nmf.columns

	nrows,ncols = close_factors(len(columns))
	fig,axs = plt.subplots(nrows=nrows,ncols=ncols,sharex=True)
	for (col,ax) in zip(columns,fig.axes):
		for method,df in {"LSI": df_lsi,"NMF" : df_nmf}.items():
			ax.plot(rvals,df[col],label=method)
			ax.set_title(f"{col} vs.\nReduced Dimension $r$",fontsize=8)
			ax.set_ylabel(f"{col}", fontsize=6)
			ax.legend()
			plt.xticks(fontsize=8)
			plt.yticks(fontsize=8)
	# plt.xlabel("Reduced Dimension $r$")
	fig.add_subplot(111, frameon=False)
	plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
	plt.xlabel("Reduced Dimension $r$")
	plt.tight_layout() #plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
	plt.savefig("figures/q16_purity_vs_dimension_larger_range.pdf")
	plt.close()

def q16_2():
	# Visualizing the clustering results for SVD and NMF.
	# Note that SVD seems to work better.

	#Load the data.
	train = pd.read_csv("input/BBC News Train.csv",header=0)
	test = pd.read_csv("input/BBC News Test.csv")
	
	#Add an additional category that changes the string category to an integer.
	train['category_id'] = train['Category'].factorize()[0]

	# Store the data and labels.
	data = train.Text
	target = train.category_id
	# TF-IDF with min_df=3 
	X_tfidf = get_tf_idf(data)
	print(X_tfidf.shape)
	# dimensionality reduction
	r=5
	n_clusters=5
	n_classes = len(set(target))
	n_samples = len(data)

	filename_base=f"figures/q16_2_kmeans_labels_predictions_2d_{n_classes}_classes"
	fig,axs = plt.subplots(nrows=2,ncols=2,sharex=True,sharey=True)
	performance_outfile = open(f"{filename_base}_performance.txt", "w")
	performance_outfile.write(f"{n_samples} Samples, {n_classes} Classes, {n_clusters} Clusters\n")
	for (method,transformer),axlist in zip({"LSI": get_LSI_matrix, "NMF" : get_NMF}.items(),axs):
		X_reduced = transformer(X_tfidf,r)
		predictions, _ = get_KMeans(X_reduced,n_clusters=n_clusters)
		X_2d = get_LSI_matrix(X_reduced,2)
		labels = {" Predictions" : predictions, "Ground Truth": target}
		performance_outfile.write(f"\nClustering Performance After {method} Dimensionality Reduction:\n")
		pprint(get_cluster_metrics(target,predictions),performance_outfile)
		for ax,(label_type,label) in zip(axlist,labels.items()):
			ax.scatter(
				X_2d[:,0],X_2d[:,1],
				c=label,
				s=1,
				alpha = 0.5,
				linewidths=0,
				edgecolors=None
			)
			ax.set_title(f"K-Means {label_type} Using {method}\n{n_classes} Classes, $\\in R^2$ via SVD",fontsize=8)
			ax.set_xticks([])
			ax.set_yticks([])
		# Generating Confusion Matrix
		cm = confusion_matrix(target, predictions)
		rows, cols = linear_sum_assignment(cm, maximize=True)
		# Generating Contingency Matrix
		plot_mat(cm[rows[:, np.newaxis], cols], xticklabels=cols,yticklabels=rows, size=(15,15), pic_fname = f"figures/q16_{method}")
	plt.tight_layout() #plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
	plt.savefig(f"{filename_base}.png", dpi=350, transparent=False)
	plt.savefig(f"{filename_base}.pdf", transparent=False)
	plt.close()
	performance_outfile.close()

def q16_3():
	#Trying Kullback-Leibler Divergence for NMF

	#Load the data.
	train = pd.read_csv("input/BBC News Train.csv",header=0)
	test = pd.read_csv("input/BBC News Test.csv")
	
	#Add an additional category that changes the string category to an integer.
	train['category_id'] = train['Category'].factorize()[0]

	# Store the data and labels.
	data = train.Text
	target = train.category_id
	# TF-IDF with min_df=3 
	X_tfidf = get_tf_idf(data)
	print(X_tfidf.shape)
	# dimensionality reduction
	r_components=5
	n_clusters=5
	n_classes = len(set(target))
	n_samples = len(data)

	filename_base=f"figures/q16_3_kmeans_labels_predictions_2d_{n_classes}_classes_{r_components}_components"
	performance_outfile = open(f"{filename_base}_performance.txt", "w")
	performance_outfile.write(f"{n_samples} Samples, {n_classes} Classes, {n_clusters} Clusters, {r_components}Components\n")
	for (method,transformer) in {"LSI": get_LSI_matrix,"KL_NMF": get_KL_NMF, "FB_NMF" : get_NMF}.items():

		X_reduced = transformer(X_tfidf,r_components)
		predictions, _ = get_KMeans(X_reduced,n_clusters=n_clusters)

		performance_outfile.write(f"\nClustering Performance After {method} Dimensionality Reduction:\n")
		pprint(get_cluster_metrics(target,predictions),performance_outfile)

		if method == "KL_NMF":
			# Generating Confusion Matrix
			cm = confusion_matrix(target, predictions)
			rows, cols = linear_sum_assignment(cm, maximize=True)
			# Generating Contingency Matrix
			plot_mat(cm[rows[:, np.newaxis], cols], xticklabels=cols,yticklabels=rows, size=(15,15), pic_fname = f"figures/q16_{method}")

	performance_outfile.close()

def q16_4():
	# Using UMAP and finding a good n_components choice for UMAP
	#Load the data.
	train = pd.read_csv("input/BBC News Train.csv",header=0)
	test = pd.read_csv("input/BBC News Test.csv")
	
	#Add an additional category that changes the string category to an integer.
	train['category_id'] = train['Category'].factorize()[0]

	# Store the data and labels.
	data = train.Text
	target = train.category_id
	# TF-IDF with min_df=3 
	X_tfidf = get_tf_idf(data)
	print(X_tfidf.shape)
	n_clusters=5
	n_classes = len(set(target))
	n_samples = len(data)


	filename_base=f"figures/q16_4_kmeans_labels_predictions_2d_{n_classes}_classes_varying_r_components"
	performance_outfile = open(f"{filename_base}_performance.txt", "w")
	
	for r_components in [2,5,10,20,30,40,60,80,120,160,320]:
		performance_outfile.write(f"{r_components} Components, {n_samples} Samples, {n_classes} Classes, {n_clusters} Clusters\n")
		metric = ['euclidean','cosine']
		for (method,transformer),metric in zip({"UMAP_euclidean": get_UMAP_matrix,"UMAP_cosine": get_UMAP_matrix}.items(),metric):
			X_reduced = transformer(X_tfidf,r_components,metric)
			predictions, _ = get_KMeans(X_reduced,n_clusters=n_clusters)
			performance_outfile.write(f"\nClustering Performance After {method} Dimensionality Reduction:\n")
			pprint(get_cluster_metrics(target,predictions),performance_outfile)

	performance_outfile.close()
	pass

def q16_5():
	#Analyze the data from q16_4() and determine what is a good components choice. Then get the contingency matrix.

	#Load the data.
	train = pd.read_csv("input/BBC News Train.csv",header=0)
	test = pd.read_csv("input/BBC News Test.csv")
	
	#Add an additional category that changes the string category to an integer.
	train['category_id'] = train['Category'].factorize()[0]

	# Store the data and labels.
	data = train.Text
	target = train.category_id
	# TF-IDF with min_df=3 
	X_tfidf = get_tf_idf(data)
	print(X_tfidf.shape)
	# FILL THIS OUT AFTER ANALYZING q16_4()!!!!!!!!!
	r_components=40    # Fill this out after analyzing q16_4()!!!!!!!!!! I DID THIS BUT PLEASE CHECK
	metric = 'cosine'   # Fill this out after analyzing q16_4()!!!!!!!!!! 
	n_clusters=5
	n_classes = len(set(target))
	n_samples = len(data)

	filename_base=f"figures/q16_5_kmeans_labels_predictions_2d_{n_classes}_classes_{r_components}components"
	performance_outfile = open(f"{filename_base}_performance.txt", "w")
	performance_outfile.write(f"{r_components} Components, {n_samples} Samples, {n_classes} Classes, {n_clusters} Clusters\n")
	X_reduced = get_UMAP_matrix(X_tfidf,r_components,metric)
	predictions, _ = get_KMeans(X_reduced,n_clusters=n_clusters)
	performance_outfile.write(f"\nClustering Performance After UMAP {metric} Dimensionality Reduction:\n")
	pprint(get_cluster_metrics(target,predictions),performance_outfile)

	cm = confusion_matrix(target, predictions)
	rows, cols = linear_sum_assignment(cm, maximize=True)
	plot_mat(cm[rows[:, np.newaxis], cols], xticklabels=cols,yticklabels=rows, size=(15,15), pic_fname = f"figures/q16_5_UMAP_reduction_cosine")
	pass

def q16_6():
	#Trying Agglomerative Clustering.

	#Load the data.
	train = pd.read_csv("input/BBC News Train.csv",header=0)
	test = pd.read_csv("input/BBC News Test.csv")
	
	#Add an additional category that changes the string category to an integer.
	train['category_id'] = train['Category'].factorize()[0]

	# Store the data and labels.
	data = train.Text
	target = train.category_id
	# TF-IDF with min_df=3 
	X_tfidf = get_tf_idf(data)
	print(X_tfidf.shape)
	n_classes = len(set(target))
	n_samples = len(data)


	n_components = 40 # Should this be n_components??
	n_clusters = 20

	filename_base=f"figures/q16_6_kmeans_labels_predictions_2d_{n_classes}_classes_{n_components}components"

	metric = ['ward','single']
	performance_outfile = open(f"{filename_base}_performance.txt", "w")
	performance_outfile.write(f"{n_samples} Samples, {n_classes} Classes, {n_clusters} Clusters\n")
	for (method,transformer),metric in zip({"Ward": get_AgglomerativeClustering,"Single": get_AgglomerativeClustering}.items(),metric):
		X_reduced = get_UMAP_matrix(X_tfidf,n_components,'cosine')
		predictions = transformer(X_reduced,n_clusters,metric)

		performance_outfile.write(f"\nClustering Performance After {method} Dimensionality Reduction:\n")
		pprint(get_cluster_metrics(target,predictions),performance_outfile)

	performance_outfile.close()
	

def q16_7():
	# Trying DBSCAN and HDBSCAN on UMAP-transformed data.

	#Load the data.
	train = pd.read_csv("input/BBC News Train.csv",header=0)
	test = pd.read_csv("input/BBC News Test.csv")

	#Add an additional category that changes the string category to an integer.
	train['category_id'] = train['Category'].factorize()[0]

	# Store the data and labels.
	data = train.Text
	target = train.category_id
	# TF-IDF with min_df=3 
	X_tfidf = get_tf_idf(data)
	print(X_tfidf.shape)
	n_classes = len(set(target))
	n_samples = len(data)

	n_components = 40
	n_clusters = 5

	filename_base=f"figures/q16_hdbscan_labels_predictions_2d_{n_classes}_classes"
	performance_outfile = open(f"{filename_base}_performance.txt", "w")
	performance_outfile.write(f"{n_samples} Samples, {n_classes} Classes, {n_clusters} Clusters\n")
	X_reduced = get_UMAP_matrix(X_tfidf,n_components,'cosine')

	for min_samples in [None, 1, 2, 3, 5, 10, 20, 40, 60]:
		for alpha in [.2,.4,.6,.8,1.0,1.2,1.4,1.6,1.8,2.0]:
			for metric in ['euclidean', 'l1', 'l2']:
				predictions, clusterer = HDBSCAN(X_reduced,min_samples,alpha,metric)
				# print(clusterer.labels_.max())
				performance_outfile.write(f"\nClustering Performance After UMAP Dimensionality Reduction with Parameters:\n")
				performance_outfile.write(f"\nmin_samples: {min_samples}; alpha: {alpha}; metric: {metric}\n")
				pprint(get_cluster_metrics(target,predictions), performance_outfile)
	performance_outfile.close()

	filename_base=f"figures/q16_dbscan_labels_predictions_2d_{n_classes}_classes"
	performance_outfile = open(f"{filename_base}_performance.txt", "w")
	performance_outfile.write(f"{n_samples} Samples, {n_classes} Classes, {n_clusters} Clusters\n")
	for eps in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.8, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
		for metric in ['cosine', 'euclidean', 'l1', 'l2']:
			predictions, clusterer = DBSCAN_builtin(X_reduced,eps,metric)
			performance_outfile.write(f"\nClustering Performance After UMAP Dimensionality Reduction with Parameters:\n")
			performance_outfile.write(f"\neps: {eps}; metric: {metric}\n")
			pprint(get_cluster_metrics(target,predictions), performance_outfile)
	performance_outfile.close()

	pass

def q16_8():
	# Plot the contingency matrix.

	#Load the data.
	train = pd.read_csv("input/BBC News Train.csv",header=0)
	test = pd.read_csv("input/BBC News Test.csv")

	#Add an additional category that changes the string category to an integer.
	train['category_id'] = train['Category'].factorize()[0]

	# Store the data and labels.
	data = train.Text
	target = train.category_id
	# TF-IDF with min_df=3 
	X_tfidf = get_tf_idf(data)
	print(X_tfidf.shape)
	n_classes = len(set(target))
	n_samples = len(data)

	n_components = 40
	n_clusters = 5


	X_reduced = get_UMAP_matrix(X_tfidf,n_components,'cosine')


	# If HDBSCAN performed better.
	min_samples = 40
	alpha = 2.0 
	metric = 'euclidean'
	predictions, clusterer = HDBSCAN(X_reduced,min_samples,alpha,metric)
	pprint(get_cluster_metrics(target,predictions))


	# For both
	cm = confusion_matrix(target, predictions)
	rows, cols = linear_sum_assignment(cm, maximize=True)
	plot_mat(cm[rows[:, np.newaxis], cols], xticklabels=cols,yticklabels=rows, size=(15,15), pic_fname = f"figures/q16_contingency_hdb") # CHANGE TO db/hdb ACCORDINGLY

	# IF DBSCAN performed better.
	eps = 1.0
	metric = 'l2'
	predictions, clusterer = DBSCAN_builtin(X_reduced,eps,metric)
	pprint(get_cluster_metrics(target,predictions))

	# For both
	cm = confusion_matrix(target, predictions)
	rows, cols = linear_sum_assignment(cm, maximize=True)
	plot_mat(cm[rows[:, np.newaxis], cols], xticklabels=cols,yticklabels=rows, size=(15,15), pic_fname = f"figures/q16_contingency_db") # CHANGE TO db/hdb ACCORDINGLY



	pass





if __name__=="__main__":
	for fn in (q1,q2,q3,q4,q5,q7,q9,q10,q11,q11_2,q13,q14,q15,q16,q16_1,q16_2,q16_3,q16_4,q16_5,q16_6,q16_7,q16_8):
		print(fn.__name__)
		fn()

