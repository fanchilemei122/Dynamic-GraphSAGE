from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import graphSAGELayer
import warnings

warnings.simplefilter("ignore")


class dygraphSAGE(nn.Module):
    def __init__(self, args, features_dim):
        super(dygraphSAGE, self).__init__()
        self.args = args
        self.features_dim = features_dim
        self.sage_1_order = graphSAGELayer(self.args, self.features_dim, self.args.hidden_dim)
        self.sage_2_order = graphSAGELayer(self.args, self.args.hidden_dim, self.args.out_dim)
        self.W = nn.ParameterDict({})
        for i in range(self.args.group_num):
            self.W[str(i)] = nn.Parameter(torch.empty(size=(self.args.out_dim, self.args.out_dim)))
            nn.init.xavier_uniform_(self.W[str(i)].data)
        self.W_T = nn.Parameter(torch.empty(size=(self.args.out_dim * 2, self.args.out_dim)))
        nn.init.xavier_uniform_(self.W_T.data)
        self.W_T_gcn = nn.Parameter(torch.empty(size=(self.args.out_dim, self.args.out_dim)))
        nn.init.xavier_uniform_(self.W_T_gcn.data)
        self.leakyrelu = nn.LeakyReLU(self.args.alpha)

    def forward(self, x, adj, past):
        num = self.args.number[self.args.now - (self.args.group_num - 1)]
        feat = self.sage_1_order(x, adj, self.args.aggregate_1_num)
        feat = self.sage_2_order(feat, adj, self.args.aggregate_2_num)
        if len(past) < self.args.group_num - 1:
            return feat
        features = self.aggregate_neighbors_time_feats_func_mean(num, feat, past)
        features = self.leakyrelu(features)
        features = F.normalize(features, p=2., dim=1)
        feat = torch.cat((features, feat[num:]), dim=0)
        return feat

    def aggregate_neighbors_time_feats_func_mean(self, num, feat, past):
        time_feat = (torch.zeros(size=(num, self.args.out_dim))).cuda()
        for i in range(num):
            time_feat[i] += F.dropout((torch.matmul(feat[i], self.W[str(0)].cuda())) / self.args.group_num, 0.5,
                                      training=self.training)
        for i in range(num):
            for j in range(self.args.group_num - 1):
                time_feat[i] += F.dropout(
                    (torch.matmul(torch.tensor(past[j][i]).cuda(), self.W[str(j + 1)].cuda())) / self.args.group_num,
                    0.5, training=self.training)
        return torch.mm(torch.cat([feat[:num], time_feat], dim=1), self.W_T)

    def aggregate_neighbors_time_feats_func_maxpool(self, num, feat, past):
        time_feat = (-9e15 * torch.ones(size=(num, self.args.out_dim))).cuda()
        for i in range(self.args.group_num - 1):
            time_feat = torch.max(feat[:num], torch.tensor(past[i][:num]).cuda())
        return torch.mm(torch.cat([feat[:num], time_feat], dim=1), self.W_T)

    def aggregate_neighbors_time_feats_func_lstm(self, num, feat, past):
        lstm_layer = torch.nn.LSTM(input_size=self.args.out_dim, hidden_size=self.args.out_dim, bias=True, dropout=0.5)
        neighbors_time_feats = []
        for i in range(num):
            time_feat = []
            time_feat.append(torch.matmul(torch.tensor(feat[i]).cuda(), self.W[str(0)].cuda()).tolist())
            for j in range(self.args.group_num - 1):
                time_feat.append((torch.matmul(torch.tensor(past[j][i]).cuda(), self.W[str(j + 1)].cuda())).tolist())
            time_feat = torch.tensor([time_feat])
            output, (_, _) = lstm_layer(time_feat)
            neighbors_time_feats.append(output[-1][0].tolist())
        neighbors_feats = torch.tensor(neighbors_time_feats).cuda()
        return torch.mm(torch.cat([feat[:num], neighbors_feats], dim=1), self.W_T)

    def aggregate_neighbors_time_feats_func_gcn(self, num, feat, past):
        time_feat = (torch.zeros(size=(num, self.args.out_dim))).cuda()
        for i in range(num):
            time_feat[i] += F.dropout((torch.matmul(feat[i], self.W[str(0)].cuda())) / self.args.group_num, 0.5,
                                      training=self.training)
        for i in range(num):
            for j in range(self.args.group_num - 1):
                time_feat[i] += F.dropout(
                    (torch.matmul(torch.tensor(past[j][i]).cuda(), self.W[str(j + 1)].cuda())) / self.args.group_num,
                    0.5, training=self.training)
        return torch.mm(time_feat, self.W_T_gcn)

