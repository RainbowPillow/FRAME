import os.path
import sys
import time
import argparse
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from utils.data_loader import dataloader
from utils.metrics import accuracy
from models.frame import FRAME
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold


# training function
def train(model, optimizer, model_name):

    criterion = F.cross_entropy
    # criterion1 = Balanced_softmax_loss(cls_num_list)
    test_best_possible, best_so_far = 0.0, sys.maxsize
    test_best_auc = 0.0
    test_best_f1 = 0.0
    test_best_mcc = 0.0
    test_best_gmean = 0.0
    lamda = 0.1

    for epoch in range(args.epochs):

        t = time.time()
        model.train()
        optimizer.zero_grad()

        output = model(input)

        loss_train = criterion(output[idx_train], torch.squeeze(labels[idx_train]))

        aucprc_train, f1_train, gmean_train, mcc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        aucprc_val, f1_val, gmean_val, mcc_val = accuracy(output[idx_val], labels[idx_val])

        if aucprc_val > test_best_auc:
            test_best_auc = aucprc_val

        if f1_val > test_best_f1:
            test_best_f1 = f1_val

        if gmean_val > test_best_gmean:
            test_best_gmean = gmean_val

        if mcc_val > test_best_mcc:
            test_best_mcc = mcc_val

        print('Epoch[{:4d}/{:4d}]'.format(epoch + 1, args.epochs),
              'loss_train: {:.4f}'.format(loss_train),
              # 'mcc_train: {:.4f}'.format(mcc_train),
              # 'mcc_val: {:.4f}'.format(mcc_val),
              'best_val_auc: {:.4f}'.format(test_best_auc),
              'best_val_f1: {:.4f}'.format(test_best_f1),
              'best_val_gmean: {:.4f}'.format(test_best_gmean),
              'best_val_mcc: {:.4f}'.format(test_best_mcc),
              'time: {:.2f}s'.format(time.time() - t))

    return test_best_auc, test_best_f1, test_best_gmean, test_best_mcc


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # dataset settings
    parser.add_argument('--data_path', type=str, default="./datasets/",
                        help='the path of data.')

    # cuda settings
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')

    # Training parameter settings
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Initial learning rate. default:[0.00001]')
    parser.add_argument('--wd', type=float, default=1e-5,
                        help='Weight decay (L2 loss on parameters). default: 5e-3')

    # Model parameters
    parser.add_argument('--filters', type=int, default=128,
                        help='number of filters, Default:128')
    parser.add_argument('--centers', type=int, default=1,
                        help='number of centers, Default:1')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = 'cpu'

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        device = 'cuda'

    AUC_results = list()
    F1_results = list()
    Gmean_results = list()
    MCC_results = list()

    datasets = [
                'yeast1.npz',
                'ecoli1.npz',
                'SatImage.npz',
                'US_Crime.npz',
                'Scene.npz',
                'ozone_eighthr.npz',
                'Ozone_Level.npz',
                'poker-8-9_vs_6.npz',
                'Protein_homo.npz',
                'creditcard.npz',
                ]

    model_name = 'FRAME'

    log_dir = './logs/log_frame/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    for dataset in datasets:

        print(dataset.strip('.npz'))
        print("Loading dataset", dataset, "...")
        X, Y = dataloader(args.data_path, dataset)
        file_name = 'results_' + model_name + '_' + str(args.centers) + '_' + dataset.strip('.npz') + '.txt'

        result_record = open(log_dir + file_name, 'w')
        Gmean_results.clear()
        MCC_results.clear()
        F1_results.clear()
        AUC_results.clear()

        skf = StratifiedKFold(n_splits=5)
        start = time.time()
        for trn_index, test_index in skf.split(X, Y):

            x_train = X[trn_index]
            x_test = X[test_index]
            y_train = Y[trn_index]
            y_test = Y[test_index]

            nclass = int(np.amax(y_train)) + 1

            ts = np.concatenate((x_train, x_test), axis=0)
            labels = np.concatenate((y_train, y_test), axis=0)
            labels = np.squeeze(labels)

            train_size = y_train.shape[0]
            total_size = labels.shape[0]
            idx_train = range(train_size)
            idx_val = range(train_size, total_size)

            features = torch.FloatTensor(np.array(ts))
            labels = torch.LongTensor(labels)

            idx_train = torch.LongTensor(idx_train)
            idx_val = torch.LongTensor(idx_val)

            y_t = labels[idx_train].detach().cpu().numpy()
            classes = np.unique(y_t)
            le = LabelEncoder()
            y_ind = le.fit_transform(y_t.ravel())
            recip_freq = len(y_t) / (len(le.classes_) *
                                     np.bincount(y_ind).astype(np.float64))
            class_weight = recip_freq[le.transform(classes)]

            model = FRAME(features.shape[1], nclass, args.filters, args.centers)

            if args.cuda:
                model.cuda()
                features, labels, idx_train = features.cuda(), labels.cuda(), idx_train.cuda()

            input = (features, labels, idx_train, idx_val)

            # init the optimizer
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

            auc, f1, gmm, mcc = train(model, optimizer, model_name)

            AUC_results.append(auc)
            F1_results.append(f1)
            Gmean_results.append(gmm)
            MCC_results.append(mcc)

        end = time.time()
        print('running time:', end-start)
        avg_AUC = round(np.mean(AUC_results), 4)
        std_AUC = round(np.std(AUC_results), 4)
        avg_F1 = round(np.mean(F1_results), 4)
        std_F1 = round(np.std(F1_results), 4)
        avg_Gmean = round(np.mean(Gmean_results), 4)
        std_Gmean = round(np.std(Gmean_results), 4)
        avg_MCC = round(np.mean(MCC_results), 4)
        std_MCC = round(np.std(MCC_results), 4)

        print('dataset name: {}'.format(dataset),
              'running time: {:.4f}'.format(end - start),
              'AUC: {:.4f}_{:.4f} '.format(np.mean(AUC_results), np.std(AUC_results)),
              'F1: {:.4f}_{:.4f} '.format(np.mean(F1_results), np.std(F1_results)),
              'Gmean: {:.4f}_{:.4f} '.format(np.mean(Gmean_results), np.std(Gmean_results)),
              'MCC: {:.4f}_{:.4f}'.format(np.mean(MCC_results), np.std(MCC_results)))

        write_str = 'Dataset:' + dataset + ': \n' \
                    + '\ttime: ' + str(end - start) + '\n' \
                    + '\tAUC: ' + str(avg_AUC) + '_' + str(std_AUC) + '\n' \
                    + '\tF1: ' + str(avg_F1) + '_' + str(std_F1) + '\n' \
                    + '\tGmean: ' + str(avg_Gmean) + '_' + str(std_Gmean) + '\n' \
                    + '\tMCC: ' + str(avg_MCC) + '_' + str(std_MCC) + '\n'
        result_record.write(write_str)

    result_record.close()








