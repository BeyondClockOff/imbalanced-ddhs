from sklearn.mixture import GaussianMixture
from sklearn import datasets


iris = datasets.load_iris()
gmm = GaussianMixture(n_components=3, random_state=0)
gmm.fit(iris.data)
gmm_cluster_labels = gmm.predict(iris.data)
