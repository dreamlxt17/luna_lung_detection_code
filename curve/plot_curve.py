# -*- coding:utf -*-

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import numpy as np
import glob

label_list = glob.glob('/home/didia/didia/luna16/result/theano/*labels.npy')
preds_list = glob.glob('/home/didia/didia/luna16/result/theano/*preds.npy')

label_list.sort()
preds_list.sort()

def read_file(label_fl, preds_fl ):
    ''' read y_true and y_score'''
    labels = np.load(label_fl)
    preds = np.load(preds_fl)
    # print len(labels), len(preds)
    return labels, preds


def get_prob(labels, preds, tpr):

    p = float(np.sum(labels[:,1]))/len(labels)
    n = 1 - p
    tp = tpr * p
    fn = p - tp
    preds[preds>0.5] = 1
    preds[preds<0.5] = 0

    p_hat = float(np.sum(preds[:,1])/len(labels))
    fp = p_hat - tp
    tn = n - fp
    return tp, fn, fp, tn


def plot_avg():

    m=1000
    m_tp = m_fn = m_tn = m_fp = 0
    mean_tpr = np.zeros(m)
    mean_fpr = np.zeros(m)
    mean_th = np.zeros(m)
    num=40
    tmp=[]

    for i, (f1, f2) in enumerate(zip(label_list[30:70], preds_list[30:70])):
        labels, preds = read_file(f1, f2)

        # (0,1) is positive
        fpr, tpr, thresholds = roc_curve(labels[:,1], preds[:,1])
        # print fpr[-1], tpr[-1]

        in_tpr = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        tp, fn, fp, tn = get_prob(labels, preds, in_tpr)
        # print tp, fn, fp, tn

        dif = abs(fpr - in_tpr)
        ind = np.argwhere(dif == np.min(dif))[0]
        tmp.append(thresholds[ind])

        m_tp += tp
        m_fn += fn
        m_fp += fp
        m_tn += tn

        dist = int(len(fpr) / m)
        fpr = [fpr[i * dist] for i in range(m)]
        tpr = [tpr[i * dist] for i in range(m)]
        thresholds = [thresholds[i*dist] for i in range(m)]
        # print 500 * dist , len(thresholds)
        mean_fpr += fpr
        mean_tpr += tpr
        mean_th += thresholds

        mean_tpr[0]=0.0

    print ('TP: %f\nFN: %f\nFP: %f\nTN: %f') % (m_tp/num, m_fn/num,m_fp/num, m_tn/num)
    # print m_tp/num + m_fn/num + m_fp/num+ m_tn/num

    print np.mean(np.array(tmp))

    tpr = mean_tpr/num
    fpr = mean_fpr/num
    tpr[0]=0
    tpr[-1]=1
    fpr[-1]=1
    mean_th = mean_th/num
    m_auc = auc(fpr, tpr)

    eer=0
    for i in range(m):
        if fpr[i] + tpr[i]  < 1.01 and fpr[i] + tpr[i]  > 0.99:
            eer = fpr[i]
            break

    dif = abs(fpr - eer)
    ind = np.argwhere(dif == np.min(dif))[0]


    plt.plot([1, 0], [0, 1], '--', color=(0.6, 0.6, 0.6), label='EER')
    plt.plot([0, 1], [0, 1], '--', color=(0.8, 0.2, 0.2), label='Guess')
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f, eer=%0.2f, th=%.2f)' % (m_auc, eer, np.mean(np.array(tmp))))

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-Theano')
    plt.legend(loc="lower right")
    plt.show()


plot_avg()
