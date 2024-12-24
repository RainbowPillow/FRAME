
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attn import SetTransformer


def euclidean_dist(x, y):
    # x: N x D
    # y: M x C x D

    f_expand = x.unsqueeze(1).unsqueeze(1)
    w_expand = y.unsqueeze(0)
    dists = torch.sum((torch.pow(f_expand - w_expand, 2)).sum(2), -1)

    return dists


class FRAME(nn.Module):
    def __init__(self, c_in, c_out, filters, centers):
        super().__init__()

        self.nclass = c_out
        self.feat_dim = filters
        self.clas_repr = None
        self.normal = nn.BatchNorm1d(c_in)

        self.mlp = nn.Sequential(
            nn.Linear(c_in, filters),
            nn.LeakyReLU(),
            nn.BatchNorm1d(filters),
            nn.Dropout(0.4),
        )

        self.att_models = nn.ModuleList()
        for _ in range(c_out):
            att_model = SetTransformer(filters, 1, centers)
            self.att_models.append(att_model)

    def forward(self, x_input):
        x, labels, idx_train, idx_val = x_input

        x = self.normal(x)

        x = self.mlp(x)

        # generate the class protocal with dimension C * D (nclass * centers * dim)
        proto_list = []
        for i in range(self.nclass):
            idx = (labels[idx_train].squeeze() == i).nonzero().squeeze(1)
            x_select = x[idx_train][idx]

            A = self.att_models[i](x_select)  # N_k * 1

            A = torch.transpose(A, 2, 1)  # 1 * N_k
            A = torch.transpose(A, 1, 0)  # 1 * N_k
            A = F.softmax(A, dim=-1)  # softmax over N_k
            class_repr = torch.matmul(A, x_select)  # 1 * L
            class_repr = torch.transpose(class_repr, 1, 0)  # 1 * N_k

            proto_list.append(class_repr)
        x_proto = torch.cat(proto_list, dim=0)

        dists = -euclidean_dist(x, x_proto)

        return dists




