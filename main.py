import os
import torch

os.environ["TORCH"] = torch.__version__
from torch import Tensor
import torch.nn as nn
from torch.nn import (
    Linear,
    MultiheadAttention,
    Dropout,
    Sigmoid,
    ReLU,
    Parameter,
    BCELoss,
    LayerNorm,
    AdaptiveAvgPool2d,
    Conv2d,
    MaxPool2d,
    Dropout2d,
    Embedding,
    Flatten,
)
import torch.optim as optim
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import copy
from copy import deepcopy
import sklearn.metrics
from sklearn.metrics import (
    roc_curve,
    auc,
    roc_auc_score,
    precision_recall_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import matplotlib.pyplot as plt
import random
import time


ds_num_embeddings = 35
mir_num_embeddings = 5
emb_dim = 512
num_heads = 64
dro = 0.2
lin_dim = 2048
ds_cha1 = 1
ds_cha2 = 8
ds_cha3 = 16
ds_k_size = 3
ds_reduction = 4
ds_conv_dro = 0.2


class DsEmb(nn.Module):
    def __init__(self, ds_emb_dim, ds_num_heads, ds_dro, ds_lin_dim):
        super(DsEmb, self).__init__()
        self.ds_mha_attention = nn.MultiheadAttention(
            ds_emb_dim, ds_num_heads, dropout=ds_dro
        )
        self.ds_linear1 = nn.Linear(ds_emb_dim, ds_lin_dim)
        self.ds_linear2 = nn.Linear(ds_lin_dim, ds_emb_dim)
        self.ds_relu = nn.ReLU()
        self.ds_dropout1 = nn.Dropout(p=ds_dro)
        self.ds_dropout2 = nn.Dropout(p=ds_dro)
        self.ds_dropout3 = nn.Dropout(p=ds_dro)
        self.ds_layer_norm1 = nn.LayerNorm(ds_emb_dim)
        self.ds_layer_norm2 = nn.LayerNorm(ds_emb_dim)

    def forward(self, x):
        mha_output, _ = self.ds_mha_attention(x, x, x)
        x = self.ds_layer_norm1(x + self.ds_dropout1(mha_output))
        lin_output = self.ds_linear1(x)
        lin_output = self.ds_relu(lin_output)
        lin_output = self.ds_dropout2(lin_output)
        lin_output = self.ds_linear2(lin_output)
        x = self.ds_layer_norm2(x + self.ds_dropout3(lin_output))
        return x


class MirEmb(nn.Module):
    def __init__(self, mir_emb_dim, mir_num_heads, mir_dro, mir_lin_dim):
        super(MirEmb, self).__init__()
        self.mir_mha_attention = nn.MultiheadAttention(
            mir_emb_dim, mir_num_heads, dropout=mir_dro
        )
        self.mir_linear1 = nn.Linear(mir_emb_dim, mir_lin_dim)
        self.mir_linear2 = nn.Linear(mir_lin_dim, mir_emb_dim)
        self.mir_relu = nn.ReLU()
        self.mir_dropout1 = nn.Dropout(p=mir_dro)
        self.mir_dropout2 = nn.Dropout(p=mir_dro)
        self.mir_dropout3 = nn.Dropout(p=mir_dro)
        self.mir_layer_norm1 = nn.LayerNorm(mir_emb_dim)
        self.mir_layer_norm2 = nn.LayerNorm(mir_emb_dim)

    def forward(self, x):
        mha_output, _ = self.mir_mha_attention(x, x, x)
        x = self.mir_layer_norm1(x + self.mir_dropout1(mha_output))
        lin_output = self.mir_linear1(x)
        lin_output = self.mir_relu(lin_output)
        lin_output = self.mir_dropout2(lin_output)
        lin_output = self.mir_linear2(lin_output)
        x = self.mir_layer_norm2(x + self.mir_dropout3(lin_output))
        return x


class DsSqE(nn.Module):
    def __init__(self, ds_ch, ds_reduction):
        super(DsSqE, self).__init__()
        self.ds_sqe_fc1 = nn.Linear(ds_ch, ds_ch // ds_reduction, bias=False)
        self.ds_sqe_fc2 = nn.Linear(ds_ch // ds_reduction, ds_ch, bias=False)
        self.ds_sqe_adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.ds_sqe_relu = nn.ReLU()
        self.ds_sqe_sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        y = self.ds_sqe_adaptive_pool(x)
        y = y.view(batch_size, channels)
        y = self.ds_sqe_fc1(y)
        y = self.ds_sqe_relu(y)
        y = self.ds_sqe_fc2(y)
        y = self.ds_sqe_sigmoid(y)
        y = y.view(batch_size, channels, 1, 1)
        x = x * y.expand_as(x)
        return x


class DsConv(nn.Module):
    def __init__(self, ds_cha1, ds_cha2, ds_cha3, ds_k_size, ds_reduction, ds_conv_dro):
        super(DsConv, self).__init__()
        self.dsc_conv1 = nn.Conv2d(ds_cha1, ds_cha2, ds_k_size, stride=1, padding=1)
        self.dsc_conv2 = nn.Conv2d(ds_cha2, ds_cha3, ds_k_size, stride=1, padding=1)
        self.dsc_sqe = DsSqE(ds_cha3, ds_reduction)
        self.dsc_pool = nn.MaxPool2d(2, stride=2)
        self.dsc_conv_dropout1 = nn.Dropout2d(p=ds_conv_dro)
        self.dsc_conv_dropout2 = nn.Dropout2d(p=ds_conv_dro)
        self.dsc_relu = nn.ReLU()

    def forward(self, x):
        x = self.dsc_conv1(x)
        x = self.dsc_relu(x)
        x = self.dsc_pool(x)
        x = self.dsc_conv_dropout1(x)
        x = self.dsc_conv2(x)
        x = self.dsc_relu(x)
        x = self.dsc_sqe(x)
        x = self.dsc_pool(x)
        x = self.dsc_conv_dropout2(x)
        return x


class PredictorModel(nn.Module):
    def __init__(
        self,
        ds_num_embeddings,
        mir_num_embeddings,
        emb_dim,
        num_heads,
        dro,
        lin_dim,
        ds_cha1,
        ds_cha2,
        ds_cha3,
        ds_k_size,
        ds_reduction,
        ds_conv_dro,
    ):
        super(PredictorModel, self).__init__()
        self.ds_embedding = nn.Embedding(ds_num_embeddings, emb_dim)
        self.ds_positional_emb = nn.Parameter(torch.zeros(1, 6, emb_dim))
        self.mir_embedding = nn.Embedding(mir_num_embeddings, emb_dim)
        self.mir_positional_emb = nn.Parameter(torch.zeros(1, 30, emb_dim))
        self.ds_emb = DsEmb(emb_dim, num_heads, dro, lin_dim)
        self.mir_emb = MirEmb(emb_dim, num_heads, dro, lin_dim)
        self.ds_dropout = nn.Dropout(p=dro)
        self.mir_dropout = nn.Dropout(p=dro)
        self.ds_conv = DsConv(
            ds_cha1, ds_cha2, ds_cha3, ds_k_size, ds_reduction, ds_conv_dro
        )
        self.flatten = nn.Flatten()
        self.co_dropout = nn.Dropout(p=dro)
        self.linear1a = nn.Linear(emb_dim * 2, emb_dim)
        self.linear2a = nn.Linear(emb_dim, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, A, B):
        xd = self.ds_embedding(A)
        xd = xd + self.ds_positional_emb[:, : xd.size(1), :]
        xd = xd.permute(1, 0, 2)
        xd = self.ds_dropout(xd)
        xd = self.ds_emb(xd)
        xd = xd.permute(1, 0, 2)
        xd = xd[:, 0, :]
        xd = xd.view(xd.size(0), 1, 32, 16)
        xd = self.ds_conv(xd)
        xd = self.flatten(xd)

        xm = self.mir_embedding(B)
        xm = xm + self.mir_positional_emb[:, : xm.size(1), :]
        xm = xm.permute(1, 0, 2)
        xm = self.mir_dropout(xm)
        xm = self.mir_emb(xm)
        xm = xm.permute(1, 0, 2)
        xm = xm[:, 0, :]

        x = torch.cat((xd, xm), dim=1)
        x = self.linear1a(x)
        x = self.relu(x)
        x = self.co_dropout(x)
        x = self.linear2a(x)
        x = self.sigmoid(x)
        return x
