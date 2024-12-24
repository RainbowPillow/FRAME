

import sklearn
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_recall_curve, \
    average_precision_score, matthews_corrcoef, roc_auc_score
from imblearn.metrics import geometric_mean_score


def auc_prc(label, y_pred):
    '''Compute AUCPRC score.'''
    return average_precision_score(label, y_pred)


def f1_optim(label, y_pred):
    '''Compute optimal F1 score.'''
    y_pred = y_pred.copy()
    prec, reca, _ = precision_recall_curve(label, y_pred)
    f1s = 2 * (prec * reca) / (prec + reca)
    return max(f1s)


def gm_optim(label, y_pred):
    '''Compute optimal G-mean score.'''
    y_pred = y_pred.copy()
    prec, reca, _ = precision_recall_curve(label, y_pred)
    gms = np.power((prec*reca), 0.5)
    return max(gms)


def mcc_optim(label, y_pred):
    '''Compute optimal MCC score.'''
    mccs = []
    for t in range(100):
        y_pred_b = y_pred.copy()
        y_pred_b[y_pred_b < 0+t*0.01] = 0
        y_pred_b[y_pred_b >= 0+t*0.01] = 1
        mcc = matthews_corrcoef(label, y_pred_b)
        mccs.append(mcc)
    return max(mccs)


def accuracy(output, labels):

    preds = output.max(1)[1].cpu().numpy()
    labels = labels.cpu().numpy()

    mcc = matthews_corrcoef(labels, preds)
    # mcc_op = mcc_optim(labels, preds)
    aucprc = auc_prc(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    gmean = gm_optim(labels, preds)

    return aucprc, f1, gmean, mcc


def accuracy_affinity(output, labels):

    output = output[:, :-1]
    output = output.argmax(dim=1, keepdim=True)
    labels = labels.argmax(dim=1, keepdim=True)

    # preds = output.max(1)[1].cpu().numpy()
    preds = output.cpu().numpy()
    labels = labels.cpu().numpy()
    # accuracy_score = (sklearn.metrics.accuracy_score(labels, preds))

    mcc = matthews_corrcoef(labels, preds)
    # Gmean = geometric_mean_score(np.squeeze(labels), np.squeeze(preds))

    return mcc


def accuracy_baseline(output, labels, model_name):

    if model_name == 'SPE':
        preds = output
        f1 = f1_score(labels, preds, average='macro')
        mcc = matthews_corrcoef(labels, preds)
        Gmean = gm_optim(labels, preds)
        auc = auc_prc(labels, preds)
    else:

        preds = np.argmax(output, axis=1)

        f1 = f1_score(labels, preds, average='macro')
        mcc = matthews_corrcoef(labels, preds)
        Gmean = gm_optim(labels, preds)
        auc = auc_prc(labels, preds)

    return f1, auc, mcc, Gmean
