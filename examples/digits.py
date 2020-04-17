import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits import mplot3d  # noqa: F401
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from endochrone import naive_knn as knn
from endochrone import metrics
from endochrone import pca

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


""" This example uses the MNIST data set to try and classify hand-written
digits. It does so by first transforming using our PCA algortihm to reduce the
dimensionality then naive-KNN to classify our test examples. It has an
accuracy of around 99%"""

digits = load_digits()

# This just shows the first 100 digits in the dataset, change show_digits to
# True if you want this
show_digits = False
if show_digits:
    fig, axes = plt.subplots(10, 10, figsize=(8, 8),
                             subplot_kw={'xticks': [], 'yticks': []},
                             gridspec_kw=dict(hspace=0.1, wspace=0.1))

    for i, ax in enumerate(axes.flat):
        ax.imshow(digits.images[i], cmap='binary', interpolation='nearest')
        ax.text(0.05, 0.05, str(digits.target[i]), transform=ax.transAxes,
                color='g')
    plt.show()

Xtrain, Ytrain, Xtest, Ytest = train_test_split(digits.data, digits.target)

# TODO refactor KNN so this isn't necessary
Xtest = Xtest[:, np.newaxis]
Ytest = Ytest[:, np.newaxis]

# Test run to figure out how many components we should keep
# TODO should be able to do this with a single model
pcam_test = pca.PCA()
pcam_test.fit(Xtrain)
cutoff = 0.97  # i.e. we want to retain this % of variance
n_comp = np.argmax(np.cumsum(pcam_test.explained_variance_ratio_) > cutoff)

# Now reduce our training set with this PCA model
pcam = pca.PCA(n_components=n_comp)
pcam.fit(Xtrain)
pca_Xtrain = pcam.transform(Xtrain)
pca_Ytrain = pcam.transform(Ytrain)

# Try KNN to see if we're any good at classifying
ypred = knn.classify(pca_Xtrain, Xtest, pca_Ytrain, k=3)
acc = accuracy_score(Ytest, ypred)*100
print("cut-off: %s \n n_comp: %s \n accuracy: %0.4f%%" % (cutoff, n_comp, acc))

metrics.print_confusion_matrix(ypred, Ytest)
print("precision:", metrics.multiclass_precision(Ytest, ypred))
print("recall:", metrics.multiclass_recall(Ytest, ypred))
print("f1_score:", metrics.multiclass_f1_score(Ytest, ypred))
# plt.show()
