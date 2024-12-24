

import numpy as np


def dataloader(path, dataset):
    path = path + '/' + dataset

    if dataset == 'aps':
        X_train = np.load(path + '/' + 'X_train.npy')
        X_test = np.load(path + '/' + 'X_test.npy')
        y_train = np.load(path + '/' + 'y_train.npy')
        y_test = np.load(path + '/' + 'y_test.npy')

        X = np.concatenate((X_train, X_test), axis=0)
        Y = np.concatenate((y_train, y_test), axis=0)

    else:
        data_zip = np.load(path)
        X = data_zip['data']
        Y = data_zip['label']

        nb_classes = int(np.amax(Y)) + 1
        Y = (Y - Y.min()) / (Y.max() - Y.min()) * (nb_classes - 1)

        Y = np.array(Y).astype(dtype=int)

    return X, Y


