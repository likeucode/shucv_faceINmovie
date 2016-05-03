import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from scipy import io as spio
from sklearn.neighbors import KNeighborsClassifier 

# Import some data to play with
# iris = datasets.load_iris()
# X = iris.data
# y = iris.target

data=spio.loadmat("openface_ytb500_20.mat")
feats = data["feature"]
ids =data["id"].astype(int)
ids=ids[0]-1
ids=label_binarize(ids,classes=list(set(ids)))
X_train,X_test,y_train,y_test=train_test_split(feats,ids,test_size=0.4,random_state=0)

# Binarize the output
# y = label_binarize(y, classes=[0, 1, 2])
n_classes = y_test.shape[1]
# n_classes=len(set(y_test))

# Add noisy features to make the problem harder
random_state = np.random.RandomState(0)
# n_samples, n_features = X.shape
# X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

# shuffle and split training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
#                                                     random_state=0)

# Learn to predict each class against the other
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', C=10,probability=True,random_state=random_state))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

##############################################################################
# Plot ROC curves for the multiclass problem

# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
# plt.plot(fpr["micro"], tpr["micro"],
#          label='SVM ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc["micro"]),
#          linewidth=2)

plt.plot(fpr["macro"], tpr["macro"],
         label='SVM ROC curve (area = {0:0.5f})'
               ''.format(roc_auc["macro"]),
         linewidth=2)
for i in range(2):
    plt.plot(fpr[i], tpr[i], label='SVM ROC curve of class {0} (area = {1:0.2f})'
                                   ''.format(i, roc_auc[i]))


############################# KNN ################################################
knn=KNeighborsClassifier(n_neighbors=5,weights='distance',p=2)
knn_score = knn.fit(X_train,y_train).predict(X_test)
knn_fpr = dict()
knn_tpr = dict()
knn_roc_auc = dict()

for i in range(n_classes):
    knn_fpr[i], knn_tpr[i], _ = roc_curve(y_test[:, i], knn_score[:, i])
    knn_roc_auc[i] = auc(knn_fpr[i], knn_tpr[i])

knn_fpr["micro"], knn_tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
knn_roc_auc["micro"] = auc(knn_fpr["micro"], knn_tpr["micro"])

knn_all_fpr = np.unique(np.concatenate([knn_fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
knn_mean_tpr = np.zeros_like(knn_all_fpr)
for i in range(n_classes):
    knn_mean_tpr += interp(knn_all_fpr, knn_fpr[i], knn_tpr[i])

# Finally average it and compute AUC
knn_mean_tpr /= n_classes

knn_fpr["macro"] = knn_all_fpr
knn_tpr["macro"] = knn_mean_tpr
knn_roc_auc["macro"] = auc(knn_fpr["macro"], knn_tpr["macro"])

##############plot knn##########################################
# plt.plot(knn_fpr["micro"], knn_tpr["micro"],'r*',
#          label='KNN ROC curve (area = {0:0.2f})'
#                ''.format(knn_roc_auc["micro"]),
#          linewidth=1)

plt.plot(knn_fpr["macro"], knn_tpr["macro"],
         label='KNN ROC curve (area = {0:0.5f})'
               ''.format(knn_roc_auc["macro"]),
         linewidth=1)
for i in range(2):
    plt.plot(knn_fpr[i], knn_tpr[i], label='KNN ROC curve of class {0} (area = {1:0.2f})'
                                   ''.format(i, knn_roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC of SVM and KNN')
plt.legend(loc="lower right")

plt.show()