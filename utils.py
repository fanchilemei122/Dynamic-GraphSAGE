import numpy as np
import torch
from sklearn.model_selection import train_test_split


def load_data(args):
    features = np.eye(143)
    features = features[0:args.N]
    adj_list = []
    for _ in range(args.N):
        adj_list.append([])
    with open("/data/enron/adj_%s.txt" % args.T) as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            for _ in range(int(info[2])):
                adj_list[int(info[0])].append(int(info[1]))
    for node in range(args.N):
        if node not in adj_list[node]:
            adj_list[node].append(node)
    features = torch.FloatTensor(features)
    # 读取前一张时间快照中的信息，新出现节点的历史嵌入为同维度的零向量
    past = {}
    if args.T >= args.group_num - 1:
        for j in range(args.group_num - 1):
            last = []
            with open("/embedding_%s.txt" % (args.T - (j + 1)), "r", encoding='utf-8') as f:
                for i, line in enumerate(f):
                    info = line.strip().split()
                    last.append(list(map(float, info)))
                past[j] = last
    return features, adj_list, past


# 20%的训练集，20%的验证集，60%的测试集
def get_evaluation_data(args):
    # np.random.seed(args.seed)
    edges_next = []
    with open("/data/enron/adj_%s.txt" % (args.T + 1)) as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            edges_next.append([])
            edges_next[i].append(int(info[0]))
            edges_next[i].append(int(info[1]))
    edges_pos = []  # 正样本
    for e in edges_next:
        if e[0] < args.N and e[1] < args.N and [e[1], e[0]] not in edges_pos:  # 该边连接的两点若在上一时刻就出现则可以作为正样本
            edges_pos.append(e)
    edges_neg = []  # 负样本
    while len(edges_neg) < len(edges_pos):  # 正负采样数量相同
        idx_i = np.random.randint(0, args.N)  # 随机选择i,j节点
        idx_j = np.random.randint(0, args.N)
        if idx_i == idx_j:  # 忽略自连接
            continue
        if [idx_i, idx_j] in edges_pos or [idx_j, idx_i] in edges_pos:  # 忽略正采样的边
            continue
        if idx_i >= args.N or idx_j >= args.N:
            continue
        if edges_neg:
            if [idx_i, idx_j] in edges_neg or [idx_j, idx_i] in edges_neg:  # 该边已经存在于edges_neg中
                continue
        edges_neg.append([idx_i, idx_j])
        # 划分训练集，测试集，验证集
    train_edges_pos, test_pos, train_edges_neg, test_neg = \
        train_test_split(edges_pos, edges_neg, test_size=0.2 + 0.6)
    val_edges_pos, test_edges_pos, val_edges_neg, test_edges_neg = \
        train_test_split(test_pos, test_neg, test_size=0.6 / (0.6 + 0.2))
    return train_edges_pos, train_edges_neg, val_edges_pos, val_edges_neg, test_edges_pos, test_edges_neg
