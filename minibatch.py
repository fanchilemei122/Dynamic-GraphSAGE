import numpy as np
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, args, adj_list):
        super(MyDataset, self).__init__()
        self.args = args
        self.adj_list = adj_list
        self.train_nodes = list(range(int(self.args.N)))
        self.__createitems__()

    def __len__(self):
        return self.args.N

    def __getitem__(self, index):
        node = self.train_nodes[index]  # 涉及到到所有节点
        return self.data_items[node]  # 该节点对应到的所有信息

    def __createitems__(self):
        self.data_items = {}
        for i in range(self.args.N):
            feed_dict = {}
            node_1 = []
            node_pos_1 = []
            node_neg_1 = []
            node_1.append([i])
            pos_sample, neg_sample = self.sampling(i)
            node_pos_1.append(pos_sample)
            node_neg_1.append(neg_sample)
            node_1_list = [torch.LongTensor(node) for node in node_1]
            node_pos_list = [torch.LongTensor(node) for node in node_pos_1]
            node_neg_list = [torch.LongTensor(node) for node in node_neg_1]
            feed_dict['node_1'] = node_1_list  # 源节点
            feed_dict['node_2'] = node_pos_list  # 源节点正采样节点
            feed_dict['node_2_neg'] = node_neg_list  # 源节点负采样节点
            self.data_items[i] = feed_dict

    def sampling(self, node):
        np.random.seed(self.args.seed)
        neighbors_pos = list(set(self.adj_list[node]))
        neighbors_neg = list(set(range(self.args.N)) - set(neighbors_pos))
        if len(neighbors_pos) >= self.args.pos_num:
            pos_sample = list(np.random.choice(neighbors_pos, size=self.args.pos_num, replace=False))
        else:
            pos_sample = list(np.random.choice(self.adj_list[node], size=self.args.pos_num, replace=True))

        if len(neighbors_neg) >= self.args.neg_num:
            neg_sample = list(np.random.choice(neighbors_neg, size=self.args.neg_num, replace=False))
        else:
            neg_sample = list(np.random.choice(neighbors_neg, size=self.args.neg_num, replace=True))
        return pos_sample, neg_sample
