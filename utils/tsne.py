from sklearn import datasets

iris = datasets.load_iris()
x, y = iris["data"], iris["target"]

from openTSNE import TSNE

embedding = TSNE().fit(x)
