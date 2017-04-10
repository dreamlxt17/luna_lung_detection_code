# -*- coding: utf-8 -*-
#!/usr/bin/env python

import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import brentq

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold

iris = datasets.load_iris()
X = iris.data
y = iris.target
X, y = X[y != 2], y[y != 2]
n_samples, n_features = X.shape

# Add noisy features
random_state = np.random.RandomState(0)
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]
print len(X)  # 100

cv = StratifiedKFold(y, n_folds=6)
classifier = svm.SVC(kernel='linear', probability=True,
                     random_state=random_state)  # 注意这里，probability=True,需要，不然预测的时候会出现异常。另外rbf核效果更好些。

mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []
all_eer = []
for i, (train, test) in enumerate(cv):
    # 通过训练数据，使用svm线性核建立模型，并对测试集进行测试，求出预测得分
    probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    mean_tpr += interp(mean_fpr, fpr, tpr)  # 对mean_tpr在mean_fpr处进行插值，通过scipy包调用interp()函数
    mean_tpr[0] = 0.0  # 初始处为0
    roc_auc = auc(fpr, tpr)


# 画对角线
# plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.plot([1,0], [0,1], '--', color=(0.6, 0.6, 0.6), label='EER')

mean_tpr /= len(cv)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
eer = brentq(lambda x : 1. - x - interp1d(mean_fpr, mean_tpr)(x), 0., 1.)

print mean_auc, eer

# 画平均ROC曲线
plt.plot(mean_fpr, mean_tpr, 'k--', label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


def plot_roc(y_true, y_score):

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    print roc_auc, eer

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    plt.plot([1, 0], [0, 1], '--', color=(0.6, 0.6, 0.6), label='EER')
    plt.plot(fpr, tpr, 'k--', label='Mean ROC (area = %0.2f, eer = %0.2f)' % (auc, eer), lw=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()




def plot_roc(y_true, y_score):

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    print roc_auc, eer

    dif = abs(fpr - eer)
    ind = np.argwhere(dif==np.min(dif))[0]

    plt.plot([1, 0], [0, 1], '--', color=(0.6, 0.6, 0.6), label='EER')
    plt.plot([0, 1], [0, 1], '--', color=(0.8, 0.2, 0.2), label='Guess')
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f, EER = %0.2f)' % (roc_auc, eer))

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-Curve')
    plt.legend(loc="lower right")
    # plt.show()

    return fpr[ind], tpr[ind], thresholds[ind]


# TP: 0.000038
# FN: 0.000385
# FP: 0.022549
# TN: 0.977029
