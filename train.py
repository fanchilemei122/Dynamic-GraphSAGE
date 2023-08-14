from __future__ import division
from __future__ import print_function
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from minibatch import MyDataset
from models import dygraphSAGE
from utils import load_data, get_evaluation_data
from link_prediction import evaluate_classifier

parser = argparse.ArgumentParser()
parser.add_argument('--T', type=int, default=0, help='now time')
parser.add_argument('--N', type=int, default=18, help='nodes number')
parser.add_argument('--time', default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
parser.add_argument('--now', type=int, default=0.)
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--batch_size', type=int, default=512, help='number of batch size.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--num_sample', type=int, default=10, help='sample negative node number')
parser.add_argument('--aggregate_1_num', type=int, default=10)
parser.add_argument('--aggregate_2_num', type=int, default=20)
parser.add_argument('--pos_num', type=int, default=10)
parser.add_argument('--neg_num', type=int, default=20)
parser.add_argument('--group_num', type=int, default=3)  # 3张图一组（包括当前图）
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.cuda.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
for ii in range(15):
    args.now = ii
    args.T = args.time[args.now]
    args.N = args.number[args.now]
    features, adj_list, past = load_data(args)
    dataset = MyDataset(args, adj_list)
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=True)
    model = dygraphSAGE(
        args=args,
        features_dim=features.shape[1]
    )
    optimizer_dygraphSAGE = optim.Adam(model.parameters(),
                                       lr=args.lr,
                                       weight_decay=args.weight_decay)
    if int(args.T) >= args.group_num - 1:
        train_edges_pos, train_edges_neg, val_edges_pos, val_edges_neg, test_edges_pos, test_edges_neg \
            = get_evaluation_data(args)
    # print('args.cuda', args.cuda)
    if args.cuda:
        model.cuda()
        features = features.cuda()
    features = Variable(features)
    best_epoch_val = 0


    def train():
        model.train()
        loss_all = 0
        torch.manual_seed(args.seed)
        for idx, feed_dict in enumerate(dataloader):
            optimizer_dygraphSAGE.zero_grad()
            loss_train, embed = loss(feed_dict)
            loss_all += loss_train
            loss_train.backward()
            optimizer_dygraphSAGE.step()
        global best_loss
        global patient
        if loss_all < best_loss:
            best_loss = loss_all
            patient = 0
            torch.save(model.state_dict(), '/models/dygraphSAGE_%s.pth' % args.T)
        else:
            patient += 1
        print('patient', patient)
        return patient


    def loss(feed_dict):
        node_source, node_pos, node_neg = feed_dict.values()
        node_source = node_source[0].reshape((1, node_source[0].size(0)))[0]
        node_pos = node_pos[0]
        node_neg = node_neg[0]
        embes = model(features, adj_list, past)
        neg_loss = 0
        for node in range(len(node_source)):
            for neg in node_neg[node]:
                neg_loss += -F.logsigmoid(-(embes[int(node_source[node])] * embes[int(neg)]).sum(-1)) / args.neg_num
        pos_loss = 0  # 正采样损失
        for node in range(len(node_source)):
            for pos in node_pos[node]:
                pos_loss += -F.logsigmoid((embes[int(node_source[node])] * embes[int(pos)]).sum(-1)) / args.pos_num
        losses = (pos_loss + neg_loss) / len(node_source)
        # 时间维度与近邻产生的损失
        if args.T >= (args.group_num - 1):
            time_loss1 = 0
            for node in range(args.number[args.now - (args.group_num - 1)]):
                # 鼓励同一节点相邻时刻拥有相似的表示
                near_time_loss = -F.logsigmoid((embes[node] * torch.tensor(past[0][node]).cuda()).sum(-1))
                losses += (near_time_loss / (args.number[args.now - (args.group_num - 1)]))
                time_loss1 += losses
        print('损失为', losses)
        return losses, embes


    best_loss = 9e15
    patient = 0
    for epoch in range(args.epochs):
        print(epoch)
        if train() > 20:
            break
    model.load_state_dict(
        torch.load('/models/dygraphSAGE_%s.pth' % (args.T - 1),
                   map_location='cuda:0'))
    model.eval()
    embedding = model(features, adj_list, past).data.cpu().numpy()
    if int(args.T) >= args.group_num - 1:
        val_result, test_result, _, _ = evaluate_classifier(
            train_edges_pos,
            train_edges_neg,
            val_edges_pos,
            val_edges_neg,
            test_edges_pos,
            test_edges_neg,
            embedding,
            embedding,
            args)
        auc_val = val_result["HAD"][1]
        auc_test = test_result["HAD"][1]
        print("{:.5f}".format(auc_test))
    for i in range(args.N):
        f = open("/embedding_%s.txt" % args.T, "a", encoding="UTF-8")
        s = " ".join(map(str, embedding[i]))
        f.write(s)
        f.write("\n")
        f.flush()
