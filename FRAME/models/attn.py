# Referencing https://arxiv.org/pdf/1810.00825.pdf
# and the original PyTorch implementation https://github.com/TropComplique/set-transformer/blob/master/blocks.py
import torch
import torch.nn as nn
from models.memory_efficient_attention import Attention


class SetTransformer(nn.Module):

    def __init__(self, in_dimension, k, out_dimension, d=4):
        """
        Arguments:
            in_dimension: an integer.
            k: an integer, number of seed vectors.
            out_dimension: an integer.
        """
        super().__init__()

        self.embed = nn.Sequential(
            nn.Linear(in_dimension, d * 2),
            nn.ReLU(inplace=True),
            nn.Linear(d * 2, d),
            nn.ReLU(inplace=True),
        )

        self.attention = Attention(
                dim=d,
                dim_head=8,  # dimension per head
                heads=k,  # number of attention heads
                causal=True,  # autoregressive or not
                memory_efficient=True,
                # whether to use memory efficient attention (can be turned off to test against normal attention)
                q_bucket_size=2048,  # bucket size along queries dimension
                k_bucket_size=4096  # bucket size along key / values dimension
            )

        self.predictor = nn.Sequential(
            nn.Linear(d, out_dimension),
        )

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, n, in_dimension].
        Returns:
            a float tensor with shape [b, out_dimension].
        """
        
        x = self.embed(x)  # shape [n, d]
        x = x.unsqueeze(0)  # shape [b, n, d]

        x = self.attention(x)  # shape [b, n, d]

        x = self.predictor(x)
        
        return x





