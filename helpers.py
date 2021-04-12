import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn import datasets, cluster
from sklearn import metrics
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
import umap
from caching import cached
import hdbscan




def get_all_data(categories = None):
	return fetch_20newsgroups(categories=categories,shuffle=False,random_state=None,remove=('headers','footers'))



def get_binary_dataset(cats1=["comp.graphics", "comp.os.ms-windows.misc", "comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware"],cats0=["rec.autos", "rec.motorcycles", "rec.sport.baseball", "rec.sport.hockey"]):
	data = fetch_20newsgroups(categories = cats1 + cats0,shuffle=False,random_state=None,remove=('headers','footers'))
	cat1_inds = [data.target_names.index(cat) for cat in cats1]
	data.target = [1 if target_ind in cat1_inds else 0 for target_ind in data.target]
	return data


def get_vectorizer(basic=False):
	return CountVectorizer(
		strip_accents='unicode',
		stop_words='english',
		ngram_range=(1,1), #default (1,1)
		token_pattern=r"(?u)\b(\d*[a-zA-Z]+\d*)+\b", #default r"(?u)\b\w\w+\b"
		min_df=3
	)

@cached
def get_tf_idf(train_data):
	vectorizer = get_vectorizer(basic=True) #no lemmatization
	X_train_counts = vectorizer.fit_transform(train_data)
	tfidf_transformer = TfidfTransformer()
	X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
	return X_train_tfidf


def get_contingency(ground_truth_target,predictions):
	A = np.zeros((len(set(ground_truth_target)),len(set(predictions))))
	for (i,j) in zip(ground_truth_target,predictions):
		A[i,j] += 1
	return A

@cached
def get_KMeans(X,n_clusters=8):
	kmeans = KMeans(
		n_clusters=n_clusters,
		random_state=0,
		max_iter = 2000,
		n_init = 50
	)
	predictions = kmeans.fit_predict(X)
	return predictions,kmeans


def get_cluster_metrics(labels,predictions):
	metric_fns = (
		metrics.homogeneity_score,
		metrics.completeness_score,
		metrics.v_measure_score,
		metrics.adjusted_rand_score,
		metrics.mutual_info_score,
		metrics.balanced_accuracy_score
	)
	return { fn.__name__ : fn(labels,predictions) for fn in metric_fns}


@cached
def LSI_builtin(X,n_components):
	svd = TruncatedSVD(n_components=n_components)
	X_reduced= svd.fit_transform(X)
	return X_reduced,svd

def get_LSI_matrix(*args,**kwargs):
	X_reduced,svd = LSI_builtin(*args,**kwargs)
	return X_reduced

def ss_error(mat1,mat2):
	return np.linalg.norm(mat1-mat2,ord='fro')**2

@cached
def get_NMF(X,n_components):
	model = NMF(n_components=n_components)
	W = model.fit_transform(X)
	return W


def close_factors(n):
	return min([(n//i,i,abs((n//i) - i)) for i in range(1, int(n**0.5)+1) if n % i == 0],key=lambda tup: tup[2])[0:2]

def get_KL_NMF(X,n_components,beta_loss = 'kullback-leibler', solver = 'mu'):
	model = NMF(n_components=n_components,beta_loss = beta_loss, solver=solver)
	W = model.fit_transform(X)
	return W

def UMAP_builtin(X,n_components,metric):
	reducer =umap.UMAP(metric=metric,n_components=n_components)
	X_reduced= reducer.fit_transform(X)
	return X_reduced,reducer

def get_UMAP_matrix(*args,**kwargs):
	X_reduced,reducer = UMAP_builtin(*args,**kwargs)
	return X_reduced

def AgglomerativeClustering_builtin(X,n_clusters,linkage):
	reducer = AgglomerativeClustering(n_clusters=n_clusters,linkage=linkage)
	X_reduced= reducer.fit_predict(X)
	return X_reduced,reducer

def get_AgglomerativeClustering(*args,**kwargs):
	X_reduced,reducer = AgglomerativeClustering_builtin(*args,**kwargs)
	return X_reduced

def HDBSCAN(X,min_samples,alpha,metric):
	clusterer = hdbscan.HDBSCAN(min_cluster_size=100,min_samples=min_samples,alpha=alpha,metric=metric)
	clusterer.fit(X)
	
	return clusterer.labels_, clusterer

def DBSCAN_builtin(X,eps,metric):
	clusterer = DBSCAN(eps=eps,min_samples=100,metric=metric)
	clusterer.fit(X)
	return clusterer.labels_, clusterer
