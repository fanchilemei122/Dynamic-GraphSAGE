import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class graphSAGELayer(nn.Module):
    def __init__(self, args, features_dim, out_feat_dim):
        super(graphSAGELayer, self).__init__()
        self.args = args
        self.features_dim = features_dim
        self.out_feat_dim = out_feat_dim
        self.W = nn.Parameter(torch.empty(size=(self.features_dim * 2, self.out_feat_dim)))
        nn.init.xavier_uniform_(self.W.data)
        self.W_gcn = nn.Parameter(torch.empty(size=(self.features_dim, self.out_feat_dim)))
        nn.init.xavier_uniform_(self.W_gcn.data)
        self.leakyrelu = nn.LeakyReLU(self.args.alpha)

    def forward(self, h, adj, aggregate_num):
        aggregate_neighbors = self.aggregate_neighbors_sample(adj, aggregate_num)
        neigh_feats = self.aggregate_neighbors_feats_func_gcn(h, aggregate_neighbors)  # mean聚合器
        feat = self.leakyrelu(neigh_feats)
        feat = F.normalize(feat, p=2., dim=1)
        return feat

    # 若邻居数足够则采样，否则加入所有邻居即可（该邻居包括源节点）
    def aggregate_neighbors_sample(self, adj, aggregate_num):
        # np.random.seed(self.args.seed)
        aggregate_neighbors = {}
        for i in range(len(adj)):
            if aggregate_num <= len(adj[i]):
                aggregate_neighbors[i] = list(np.random.choice(adj[i], size=aggregate_num, replace=False))
            else:
                aggregate_neighbors[i] = adj[i]
        return aggregate_neighbors

    # 按照mean的形式聚合邻居信息，并且拼接源节点
    def aggregate_neighbors_feats_func_mean(self, h, aggregate_neighbors):
        neighbors_feats = []
        for i in aggregate_neighbors:
            feat = 0
            for neighbor in aggregate_neighbors[i]:
                feat += F.dropout(h[neighbor], 0.5, training=self.training)
            feat = feat / len(aggregate_neighbors[i])
            feat = torch.cat([feat, h[i]], dim=0)
            neighbors_feats.append(feat.tolist())
        neighbors_feats = torch.tensor(neighbors_feats).cuda()
        return torch.mm(neighbors_feats, self.W)

    def aggregate_neighbors_feats_func_maxpool(self, h, aggregate_neighbors):
        neighbors_feats = []
        for i in aggregate_neighbors:
            feat = -9e15 * torch.ones_like(h[0])
            for neighbor in aggregate_neighbors[i]:
                feat = torch.max(feat, h[neighbor])
            feat = torch.cat([feat, h[i]], dim=0)
            neighbors_feats.append(feat.tolist())
        neighbors_feats = torch.tensor(neighbors_feats).cuda()
        return torch.mm(neighbors_feats, self.W)

    def aggregate_neighbors_feats_func_lstm(self, h, aggregate_neighbors):
        lstm_layer = torch.nn.LSTM(input_size=self.features_dim, hidden_size=self.features_dim, bias=True, dropout=0.5)
        neighbors_feats = []
        for i in aggregate_neighbors:
            feat = []
            feat.append(h[i].tolist())
            for neighbor in aggregate_neighbors[i]:
                feat.append(h[neighbor].tolist())
            feat = torch.tensor([feat])
            output, (_, _) = lstm_layer(feat)
            neighbors_feats.append(torch.cat([output[-1][0].cuda(), h[i]], dim=0).tolist())
        neighbors_feats = torch.tensor(neighbors_feats).cuda()
        return torch.mm(neighbors_feats, self.W)

    def aggregate_neighbors_feats_func_gcn(self, h, aggregate_neighbors):
        neighbors_feats = []
        for i in aggregate_neighbors:
            feat = 0
            feat += h[i]
            for neighbor in aggregate_neighbors[i]:
                feat += F.dropout(h[neighbor], 0.5, training=self.training)
            feat = feat / (len(aggregate_neighbors[i]) + 1)
            neighbors_feats.append(feat.tolist())
        neighbors_feats = torch.tensor(neighbors_feats).cuda()
        return torch.mm(neighbors_feats, self.W_gcn)
