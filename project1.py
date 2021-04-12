import os
import matplotlib.pyplot as plt
import numpy as np
import random
from pprint import pprint

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk import pos_tag, wordnet
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_curve, auc
from sklearn.svm import LinearSVC
# from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from scipy.sparse.linalg import svds
from sklearn.decomposition import NMF
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import get_data_home
from sklearn.preprocessing import StandardScaler
import umap

np.random.seed(42)
random.seed(42)





## Q1
def cat_hist():
	data = fetch_20newsgroups()
	cat_nfiles = [len(fetch_20newsgroups(subset='train', categories=[cat]).filenames) for cat in data.target_names]
	n,bins,patches = plt.hist(cat_nfiles,bins=50)
	plt.xlabel("Number of Samples in Category")
	plt.ylabel("Number of Categories")
	plt.title("Number of Samples in the Categories")
	plt.savefig(f"figures/category_histogram.pdf")


def q1():
	cat_hist()

## Q2

def pos_tagger(nltk_tag):
	if nltk_tag.startswith('J'):
		return wordnet.wordnet.ADJ
	elif nltk_tag.startswith('V'):
		return wordnet.wordnet.VERB
	elif nltk_tag.startswith('N'):
		return wordnet.wordnet.NOUN
	elif nltk_tag.startswith('R'):
		return wordnet.wordnet.ADV
	else:
		return wordnet.wordnet.NOUN

def build_lemmatizing_tokenizer(tokenizer):
	lemmatizer = nltk.wordnet.WordNetLemmatizer()
	# https://www.geeksforgeeks.org/python-lemmatization-approaches-with-examples/
	def better_tokenizer(*args,**kwargs):
		tokens = tokenizer(*args,**kwargs)
		return [lemmatizer.lemmatize(word,pos=pos_tagger(pos)) for (word,pos) in nltk.pos_tag(tokens)]
	return better_tokenizer

# vectorizer_type can be 'vanilla','lemmatizing', or 'lemmatizing2'
# (?u)\b(\S+)?[a-zA-Z]+(\S+)?\b works on regexr but has issue here
# r"(?u)\b(\d*[a-zA-Z]+\d*)+\b"
# r"(?u)\b(\S*[a-zA-Z]+\S*)+\b"
def get_vectorizer():
	vectorizer = CountVectorizer(
		strip_accents='unicode',
		stop_words='english',
		ngram_range=(1,1), #default (1,1)
		token_pattern=r"(?u)\b(\d*[a-zA-Z]+\d*)+\b", #default r"(?u)\b\w\w+\b"
		min_df=3
	)
	better_tokenizer = build_lemmatizing_tokenizer(vectorizer.build_tokenizer())
	better_vectorizer = CountVectorizer(
		tokenizer=better_tokenizer,
		strip_accents='unicode',
		stop_words='english',
		ngram_range=(1,1), #default (1,1)
		# token_pattern=r"(?u)\b[A-Za-z_][A-Za-z_]+\b", #tokenizer builtin
		min_df=3
	)
	return better_vectorizer

def get_train_test(categories = ['comp.graphics', 'comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'rec.autos', 'rec.motorcycles','rec.sport.baseball', 'rec.sport.hockey'
	]):
	twenty_train = fetch_20newsgroups(subset = 'train', categories = categories, shuffle = True, random_state = None)
	twenty_test = fetch_20newsgroups(subset = 'test', categories = categories, shuffle = True, random_state = None)
	return twenty_train,twenty_test

def get_tf_idf(train_data,test_data):
	vectorizer = get_vectorizer()
	X_train_counts = vectorizer.fit_transform(train_data)
	X_test_counts = vectorizer.transform(test_data)
	tfidf_transformer = TfidfTransformer()
	X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
	X_test_tfidf = tfidf_transformer.transform(X_test_counts)
	return X_train_tfidf, X_test_tfidf

def q2():
	twenty_train,twenty_test = get_train_test()
	X_train_tfidf, X_test_tfidf = get_tf_idf(twenty_train.data,twenty_test.data)
	print(f"TF-IDF Train Matrix Shape: {X_train_tfidf.shape}")
	print(f"TF-IDF Test Matrix Shape: {X_test_tfidf.shape}")




## Q3


def ss_error(mat1,mat2):
	return np.linalg.norm(mat1-mat2,ord='fro')**2

# Latent Sematic Indexing
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.svds.html
# Used to test whether LSI implementation is correct. By visual inspection it is!
# Note scipy.sparse.linalg.svds orders signular values from least to greatest, so I flipped all matrices.
def LSI_builtin(X,X_test):
	svd = TruncatedSVD(n_components=50,n_iter=25)
	X_reduced= svd.fit_transform(X)
	X_test_reduced = svd.transform(X_test)
	return X_reduced, X_test_reduced


def get_LSI(X,X_test):
	U, Sigma, Vh = svds(X,k=50)
	U = np.flip(U,axis=1)
	Sigma = np.flip(Sigma)
	Vh = np.flip(Vh,axis=0)
	X_reduced = U.dot(np.diag(Sigma))
	X_test_reduced =  X_test.dot(Vh.T)
	train_error = ss_error(X, X_reduced.dot(Vh))
	test_error =  ss_error(X_test, X_test_reduced.dot(Vh))
	return X_reduced, X_test_reduced, train_error, test_error

# Non-negative Matrix Factorization
# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html
def get_NMF(X,X_test):
	model = NMF(n_components=50)
	W = model.fit_transform(X)
	W_test = model.transform(X_test)
	H = model.components_
	train_error = ss_error(X, W.dot(H))
	test_error = ss_error(X_test,W_test.dot(H))
	return W, W_test, train_error,test_error


def q3(mse=True):
	twenty_train,twenty_test = get_train_test()
	X_train_tfidf, X_test_tfidf = get_tf_idf(twenty_train.data,twenty_test.data)
	X_train_LSI,X_test_LSI,LSI_train_error,LSI_test_error = get_LSI(X_train_tfidf,X_test_tfidf)
	X_train_NMF,X_test_NMF,NMF_train_error,NMF_test_error = get_NMF(X_train_tfidf,X_test_tfidf)
	print("LSI:")
	print(f"Train Error: {LSI_train_error}, Test Error: {LSI_test_error}")
	print("NMF:")
	print(f"Train Error: {NMF_train_error}, Test Error: {NMF_test_error}")
	if mse:
		n_train = len(twenty_train.data)
		n_test = len(twenty_test.data)
		LSI_train_MSE = LSI_train_error/n_train
		LSI_test_MSE = LSI_test_error/n_test
		NMF_train_MSE = NMF_train_error/n_train
		NMF_test_MSE = NMF_test_error/n_test
		print("LSI:")
		print(f"Train MSE: {LSI_train_MSE}, Test MSE: {LSI_test_MSE}")
		print("NMF:")
		print(f"Train MSE: {NMF_train_MSE}, Test MSE: {NMF_test_MSE}")



## Q4

def get_base_filename(filepath):
	return filepath.rsplit(".",1)[0].rsplit("/",1)[-1]
def get_binary_dataset(cats1,cats0):
	data_train,data_test = get_train_test(categories = cats1 + cats0)

	cat1_inds = [data_train.target_names.index(cat) for cat in cats1]
	# rec_ac_inds = [binary_train.target_names.index(cat) for cat in rec_ac_cats]

	for dataset in data_train,data_test:
		dataset.target = [1 if target_ind in cat1_inds else 0 for target_ind in dataset.target]
	return data_train,data_test

def plot_roc(fprs, tprs):
	fig, ax = plt.subplots()
	roc_auc = auc(fprs,tprs)
	ax.plot(fprs, tprs, lw=2, label= f'area under curve = {roc_auc : 0.04f}')
	ax.grid(color='0.7', linestyle='--', linewidth=1)
	ax.set_xlim([-0.1, 1.1])
	ax.set_ylim([0.0, 1.05])
	ax.set_xlabel('False Positive Rate',fontsize=15)
	ax.set_ylabel('True Positive Rate',fontsize=15)
	ax.set_title("ROC Curve")
	ax.legend(loc="lower right")
	for label in ax.get_xticklabels()+ax.get_yticklabels():
		label.set_fontsize(15)
	return fig, ax

def get_error_metrics(targets,predictions,scores=None,roc_filename=None):
	cm = confusion_matrix(targets,predictions) #cm[i,j] = # observations in group i predicted as group j
	tp = cm[1,1]
	tn = cm[0,0]
	fp = cm[1,0]
	fn = cm[0,1]
	precision = tp/(tp+fp)
	recall = tp/(tp+ fn)
	accuracy = (tp+tn)/(tp+tn+fp+fn)
	f1 = 2*tp/(2*tp+fp + fn)
	if roc_filename is not None:
		fprs,tprs, _ = roc_curve(targets,scores)
		fig, ax = plot_roc(fprs,tprs)
		ax.set_title("ROC Curve " + get_base_filename(roc_filename))
		plt.savefig(roc_filename)
	return {'recall': recall, 'precision': precision,'accuracy': accuracy,'f1': f1, 'cm': cm, 'roc': roc_filename}


def get_folds(X,target,nfolds=5):
	n = len(target)
	n_per = int(np.ceil(n/nfolds)) #last one may be slightly smaller
	inds = list(range(n))
	random.shuffle(inds)
	folds = [sorted(inds[i:(i + n_per)]) for i in range(0, len(inds), n_per)]
	return [X[fold,:] for fold in folds], [[target[i] for i in fold] for fold in folds]

def get_test_train_folds(X_folds,target_folds):
	nfolds = len(X_folds)
	for i in range(nfolds):
		X_test, target_test = X_folds[i], target_folds[i]
		X_train = np.concatenate([X_folds[j] for j in range(nfolds) if j != i])
		target_train = [sample_target for j in range(nfolds) if j != i for sample_target in target_folds[j] ]
		yield X_train, X_test, target_train, target_test


def get_linear_svc(gamma,max_iter=10000,penalty='l2',dual=False,**kwargs):
	classifier = LinearSVC(C=1/gamma,max_iter=max_iter,penalty=penalty,dual=dual,**kwargs)
	return classifier

def get_classifier_metrics(classifier,X_train,X_test,target_train,target_test,roc_filename=None):
	classifier.fit(X_train,target_train)
	predictions_test = classifier.predict(X_test)
	scores_test = classifier.decision_function(X_test)
	return get_error_metrics(target_test,predictions_test,scores_test,roc_filename=roc_filename)

def get_linear_svc_metrics(gamma,X_train,X_test,target_train,target_test,roc_filename=None,max_iter=10000,penalty='l2',**kwargs):
	print("Training linear SVC")
	classifier = get_linear_svc(gamma,max_iter=10000,penalty='l2',**kwargs)
	return get_classifier_metrics(classifier,X_train,X_test,target_train,target_test,roc_filename=roc_filename)


def q4():
	binary_train,binary_test = get_binary_dataset(["comp.graphics", "comp.os.ms-windows.misc", "comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware"],["rec.autos", "rec.motorcycles", "rec.sport.baseball", "rec.sport.hockey"])
	X_train_tfidf, X_test_tfidf = get_tf_idf(binary_train.data,binary_test.data)
	X_train_LSI,X_test_LSI,_,_ = get_LSI(X_train_tfidf,X_test_tfidf)
	for gamma in (0.0001, 1000):
		error_metrics = get_linear_svc_metrics(gamma,X_train_LSI,X_test_LSI,binary_train.target,binary_test.target,roc_filename=f"figures/roc_LSI_SVC_gamma{gamma:.04f}.pdf")
		print(f"Gamma={gamma}")
		pprint(error_metrics)
	nfolds = 5
	X_folds, target_folds = get_folds(X_train_LSI,binary_train.target,nfolds=nfolds)
	gammas = [10**k for k in range(-3,4)]
	avg_accuracies = []
	for gamma in gammas:
		accuracies = []
		for X_train, X_test, target_train, target_test in get_test_train_folds(X_folds,target_folds):
			error_metrics = get_linear_svc_metrics(gamma,X_train,X_test,target_train,target_test)
			accuracies.append(error_metrics['accuracy'])
		avg_accuracies.append(np.mean(accuracies))
	gamma = gammas[np.argmax(avg_accuracies)]
	error_metrics = get_linear_svc_metrics(gamma,X_train_LSI,X_test_LSI,binary_train.target,binary_test.target,roc_filename=f"figures/roc_LSI_SVC_BEST{nfolds}-fold_gamma{gamma:.04f}.pdf")
	print(f"BEST {nfolds}-fold-validated Gamma={gamma}")
	pprint(error_metrics)







## Q5

def get_linear_logistic(penalty,gamma,max_iter=100,solver='liblinear',**kwargs):
	classifier = LogisticRegression(penalty=penalty, C=1/gamma, solver=solver,**kwargs)
	return classifier

def get_linear_logistic_metrics(penalty,gamma,X_train,X_test,target_train,target_test,roc_filename=None,max_iter=100,solver='liblinear'):
	print("Training linear logistic")
	classifier = get_linear_logistic(penalty,gamma,max_iter=max_iter,solver=solver)
	return get_classifier_metrics(classifier,X_train,X_test,target_train,target_test,roc_filename=roc_filename)

def q5():
	binary_train,binary_test = get_binary_dataset(["comp.graphics", "comp.os.ms-windows.misc", "comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware"],["rec.autos", "rec.motorcycles", "rec.sport.baseball", "rec.sport.hockey"])
	X_train_tfidf, X_test_tfidf = get_tf_idf(binary_train.data,binary_test.data)
	X_train_LSI,X_test_LSI,_,_ = get_LSI(X_train_tfidf,X_test_tfidf)
	# No regularization.
	error_metrics_no_reg = get_linear_logistic_metrics('none',0,X_train_LSI,X_test_LSI,binary_train.target,binary_test.target,roc_filename="figures/roc_LogReg_NoReg.png")
	print("Gamma=0")
	pprint(error_metrics_no_reg)

	# Regularization. For high regularization for l1, the true positives become 0.
	nfolds = 5
	X_folds, target_folds = get_folds(X_train_LSI,binary_train.target,nfolds=nfolds)
	gammas = [10**k for k in range(-3,4)]

	#L1 Regularization
	avg_accuracies = []
	for gamma in gammas:
		accuracies = []
		for X_train, X_test, target_train, target_test in get_test_train_folds(X_folds,target_folds):
			error_metrics = get_linear_logistic_metrics('l1',gamma,X_train,X_test,target_train,target_test)
			accuracies.append(error_metrics['accuracy'])
		avg_accuracies.append(np.mean(accuracies))
	gamma = gammas[np.argmax(avg_accuracies)]
	error_metrics = get_linear_logistic_metrics('l1',gamma,X_train,X_test,target_train,target_test, roc_filename=f"figures/roc_LogReg_l1_gamma_{gamma:.04f}.png")
	print(f"BEST L1, {nfolds}-fold-validated Gamma={gamma}")
	pprint(error_metrics)

	#L2 Regularization
	avg_accuracies = []
	for gamma in gammas:
		accuracies = []
		for X_train, X_test, target_train, target_test in get_test_train_folds(X_folds,target_folds):
			error_metrics = get_linear_logistic_metrics('l2',gamma,X_train,X_test,target_train,target_test)
			accuracies.append(error_metrics['accuracy'])
		avg_accuracies.append(np.mean(accuracies))
	gamma = gammas[np.argmax(avg_accuracies)]
	error_metrics = get_linear_logistic_metrics('l2',gamma,X_train,X_test,target_train,target_test, roc_filename=f"figures/roc_LogReg_l2_gamma_{gamma:.04f}.png")
	print(f"BEST L2, {nfolds}-fold-validated Gamma={gamma}")
	pprint(error_metrics)

def get_GaussianNB_metrics(X_train,X_test,target_train,target_test,roc_filename=None):
	classifier = GaussianNB()
	classifier.fit(X_train, target_train)
	predictions_test = classifier.predict(X_test)
	scores_test = classifier.predict_proba(X_test)[:, 1]

	return get_error_metrics(target_test,predictions_test, scores_test, roc_filename=roc_filename)


def q6():
	binary_train,binary_test = get_binary_dataset(["comp.graphics", "comp.os.ms-windows.misc", "comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware"],["rec.autos", "rec.motorcycles", "rec.sport.baseball", "rec.sport.hockey"])
	X_train_tfidf, X_test_tfidf = get_tf_idf(binary_train.data,binary_test.data)
	X_train_LSI,X_test_LSI,_,_ = get_LSI(X_train_tfidf,X_test_tfidf)

	error_metrics = get_GaussianNB_metrics(X_train_LSI,X_test_LSI,binary_train.target,binary_test.target,roc_filename="figures/roc_GaussianNB.png")
	print("GaussianNB")
	pprint(error_metrics)



## Q9
def get_glove_embeddings(dimension_of_glove=300):
	embeddings_dict = {}
	with open(f"{get_data_home()}/glove.6B/glove.6B.{dimension_of_glove}d.txt", 'r') as f:
		for line in f:
			values = line.split()
			word = values[0]
			vector = np.asarray(values[1:], "float32")
			embeddings_dict[word] = vector
	return embeddings_dict


def get_glove_article_matrix(X_count,vocabulary,embeddings_dict,dimension_of_glove=300):
	n = X_count.shape[0]
	X_glove = np.zeros((n,dimension_of_glove))
	for i,row in enumerate(X_count):
		X_glove[i,:] = sum(row[0,ind]*embeddings_dict.get(vocabulary[ind],np.zeros(dimension_of_glove)) for ind in row.indices)/np.sum(row) # divide by number of words in article to normalize. Is there a better
	return X_glove

def get_classifiers_q9():
	gammas = [10**k for k in range(-3,4)]
	penalties = ('l1','l2')
	logistic_solvers = ('lbfgs', 'liblinear')
	classifiers = []
	for gamma in gammas:
		for penalty in penalties:
			classifiers.append(get_linear_svc(gamma,max_iter=10000,penalty=penalty))
			for solver in logistic_solvers:
				if penalty == 'l2':
					classifiers.append(get_linear_logistic(penalty,gamma,max_iter=1000,solver=solver))
	return classifiers

# LinearSVC(C=0.1, dual=False, max_iter=10000) with parameters {'C': 0.1, 'class_weight': None, 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1, 'loss': 'squared_hinge', 'max_iter': 10000, 'multi_class': 'ovr', 'penalty': 'l2', 'random_state': None, 'tol': 0.0001, 'verbose': 0}
def get_one_good_classifier_q9():
	return LinearSVC(**{'C': 0.1, 'class_weight': None, 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1, 'loss': 'squared_hinge', 'max_iter': 10000, 'multi_class': 'ovr', 'penalty': 'l2', 'random_state': None, 'tol': 0.0001, 'verbose': 0})

def get_cross_validated_classifier(classifiers,X_train_glove,X_test_glove,binary_train,binary_test,nfolds=5):
	classifiers = get_classifiers_q9()
	nfolds = 5
	X_folds, target_folds = get_folds(X_train_glove,binary_train.target,nfolds=nfolds)
	avg_accuracies = []
	for classifier in classifiers:
		accuracies = []
		for X_train, X_test, target_train, target_test in get_test_train_folds(X_folds,target_folds):
			error_metrics = get_classifier_metrics(classifier,X_train,X_test,target_train,target_test)
			accuracies.append(error_metrics['accuracy'])
		avg_accuracies.append(np.mean(accuracies))
	classifier = classifiers[np.argmax(avg_accuracies)]
	return classifier


def q9(dimension_of_glove=300):
	binary_train,binary_test = get_binary_dataset(["comp.graphics", "comp.os.ms-windows.misc", "comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware"],["rec.autos", "rec.motorcycles", "rec.sport.baseball", "rec.sport.hockey"])
	vectorizer = CountVectorizer(strip_accents='unicode',stop_words='english',ngram_range=(1,1),token_pattern=r"(?u)\b(\d*[a-zA-Z]+\d*)+\b",min_df=3)
	X_train_counts = vectorizer.fit_transform(binary_train.data)
	X_test_counts = vectorizer.transform(binary_test.data)
	vocabulary = vectorizer.get_feature_names()
	embeddings_dict = get_glove_embeddings(dimension_of_glove=dimension_of_glove)
	X_train_glove = get_glove_article_matrix(X_train_counts,vocabulary,embeddings_dict,dimension_of_glove)
	X_test_glove = get_glove_article_matrix(X_test_counts,vocabulary,embeddings_dict,dimension_of_glove)
	classifiers = get_classifiers_q9()
	classifier = get_cross_validated_classifier(classifiers,X_train_glove,X_test_glove,binary_train,binary_test,nfolds=5)
	classifier_name = str(classifier)
	error_metrics =  get_classifier_metrics(classifier,X_train_glove,X_test_glove,binary_train.target,binary_test.target,roc_filename=f"figures/roc_GloVe_cross-validated_{classifier_name}.pdf")
	print(f"{classifier} with parameters {classifier.get_params()}")
	pprint(error_metrics)


## Q10
def q10():
	binary_train,binary_test = get_binary_dataset(["comp.graphics", "comp.os.ms-windows.misc", "comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware"],["rec.autos", "rec.motorcycles", "rec.sport.baseball", "rec.sport.hockey"])
	vectorizer = CountVectorizer(strip_accents='unicode',stop_words='english',ngram_range=(1,1),token_pattern=r"(?u)\b(\d*[a-zA-Z]+\d*)+\b",min_df=3)
	X_train_counts = vectorizer.fit_transform(binary_train.data)
	X_test_counts = vectorizer.transform(binary_test.data)
	vocabulary = vectorizer.get_feature_names()
	dimensions = (50,100,200,300)
	accuracies = {}
	for dimension_of_glove in dimensions:
		embeddings_dict = get_glove_embeddings(dimension_of_glove=dimension_of_glove)
		X_train_glove = get_glove_article_matrix(X_train_counts,vocabulary,embeddings_dict,dimension_of_glove)
		X_test_glove = get_glove_article_matrix(X_test_counts,vocabulary,embeddings_dict,dimension_of_glove)
		classifier = get_one_good_classifier_q9()
		error_metrics =  get_classifier_metrics(classifier,X_train_glove,X_test_glove,binary_train.target,binary_test.target)
		accuracies[dimension_of_glove] = error_metrics['accuracy']
	classifier_name = str(classifier)
	# plt.plot(tuple(accuracies.keys()), tuple(accuracies.values()))
	fig, ax = plt.subplots()
	ax.plot(tuple(accuracies.keys()), tuple(accuracies.values()))
	ax.set_xlabel('GloVe Embedding Dimension',fontsize=15)
	ax.set_ylabel('Accuracy',fontsize=15)
	ax.set_title(f"GloVe Accuracy vs. Embedding Dimension\nwith {classifier_name}")
	plt.savefig(f"figures/GloVe_accuracy_vs_dimension_{classifier_name}.pdf")

def get_random_matrix_and_labels(nsamples,dim):
	return np.random.rand(nsamples,dim), np.full(nsamples,-1).tolist()

def q11():
	binary_train,binary_test = get_binary_dataset(["comp.graphics", "comp.os.ms-windows.misc", "comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware"],["rec.autos", "rec.motorcycles", "rec.sport.baseball", "rec.sport.hockey"])
	target_train = binary_train.target
	target_names = [ "Recreational Activity","Computer Technology"]
	# target_test = binary_test.target
	vectorizer = CountVectorizer(strip_accents='unicode',stop_words='english',ngram_range=(1,1),token_pattern=r"(?u)\b(\d*[a-zA-Z]+\d*)+\b",min_df=3)
	X_train_counts = vectorizer.fit_transform(binary_train.data)
	# X_test_counts = vectorizer.transform(binary_test.data)
	vocabulary = vectorizer.get_feature_names()
	dimension_of_glove = 300
	embeddings_dict = get_glove_embeddings(dimension_of_glove=dimension_of_glove)
	X_train_glove = get_glove_article_matrix(X_train_counts,vocabulary,embeddings_dict,dimension_of_glove)
	# X_test_glove = get_glove_article_matrix(X_test_counts,vocabulary,embeddings_dict,dimension_of_glove)
	n_articles = X_train_glove.shape[0]
	X_random, target_random = get_random_matrix_and_labels(nsamples=n_articles,dim=dimension_of_glove)
	reducer = umap.UMAP()
	X_random_umap = reducer.fit_transform(X_random)
	X_random_umap_normalized = StandardScaler().fit_transform(X_random_umap)
	reducer = umap.UMAP()
	X_umap = reducer.fit_transform(X_train_glove)
	X_umap_normalized = StandardScaler().fit_transform(X_umap)
	all_colors = ['red','green','blue','orange']
	cmap = {target : all_colors.pop() for target in set(target_train + target_random)}
	fig, ax = plt.subplots()
	ax.scatter(
		X_random_umap_normalized[:,0],
		X_random_umap_normalized[:,1],
		c=[cmap[-1] for i in target_random],
		s=4,
		alpha = 0.5,
		label="Random Vectors"
	)
	for target_label in set(target_train):
		inds = [i for i,val in enumerate(target_train) if val == target_label]
		ax.scatter(
			X_umap_normalized[inds,0],
			X_umap_normalized[inds,1],
			c=[cmap[target_train[i]] for i in inds],
			s=4,
			alpha = 0.5,
			label=f"Class {target_label} ({target_names[target_label]})",
		)
	ax.legend()
	plt.title('UMAP Projection of the GloVe Article Embeddings')
	plt.savefig(f"figures/umap_glove_vs_random.pdf")
	# plt.show(block=False)

if __name__=="__main__":
	# q9()
	pass


# q1()
# q2()
# q3()
# q4()
# q5()
# q6()
# pass
# q9()
# q11()
