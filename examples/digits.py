import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits import mplot3d  # noqa: F401
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from endochrone.classification import KNearest
from endochrone.stats import metrics
from endochrone.decomposition import PCA

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

Xtrain, Xtest, Ytrain, Ytest = train_test_split(digits.data, digits.target)

# Test run to figure out how many components we should keep
# TODO should be able to do this with a single model
pcam_test = PCA()
pcam_test.fit(features=Xtrain)
cutoff = 0.97  # i.e. we want to retain this % of variance
n_comp = np.argmax(np.cumsum(pcam_test.explained_variance_ratio_) > cutoff)

# Now reduce our features with this PCA model
pcam = PCA(n_components=n_comp)
pca_Xtrain = pcam.fit_and_transform(features=Xtrain)
pca_Xtest = pcam.transform(features=Xtest)

# Try KNN to see if we're any good at classifying
knn_model = KNearest()
knn_model.fit(features=pca_Xtrain, targets=Ytrain)
ypred = knn_model.predict(features=pca_Xtest)
acc = accuracy_score(Ytest, ypred)*100
print("cut-off: %s \nn_comp: %s \naccuracy: %0.4f%%" % (cutoff, n_comp, acc))

perf = metrics.MulticlassMetrics(Ytest, ypred)
perf.print_confusion_matrix()
print("macro precision:", perf.macro_precision)
print("macro recall:", perf.macro_recall)
print("macro f1_score:", perf.macro_f1_score)
