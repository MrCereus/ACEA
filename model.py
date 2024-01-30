# coding: UTF-8
import os
from posixpath import join
import torch
import torch.nn as nn
from torch.nn import *
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from settings import *
import torch.utils.data as Data
import random
import faiss
import pandas as pd
import argparse
import logging
from datetime import datetime
# using labse
# from transformers import *
import torch
from load import *

# Labse embedding dim
MAX_LEN = 88
PROJ_PATH = '/home/mrcactus/Thesis/ACEA/util'

def adjust_learning_rate(optimizer, epoch, lr):
    if (epoch+1) % 10 == 0:
        lr *= 0.5
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class NCESoftmaxLoss(nn.Module):

    def __init__(self, device):
        super(NCESoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.device = device

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.squeeze()
        label = torch.zeros([batch_size]).to(self.device).long()
        loss = self.criterion(x, label)
        return loss
class MyEmbedder(nn.Module):
    def __init__(self, args, vocab_size, padding=ord(' ')):
        super(MyEmbedder, self).__init__()

        self.args = args

        self.device = torch.device(self.args['device'])

        self.attn = BatchMultiHeadGraphAttention(self.device, self.args)
        
        self.attn_mlp = nn.Sequential(
            nn.Linear(LaBSE_DIM * 2, LaBSE_DIM),
        )

        # loss
        self.criterion = NCESoftmaxLoss(self.device)

        # batch queue
        self.batch_queue = []

    def contrastive_loss(self, pos_1, pos_2, neg_value):
        # print("pos_1.shape: ", pos_1.shape)
        # print("pos_2.shape: ", pos_2.shape)
        # print("neg_value.shape: ", neg_value.shape)
        bsz = pos_1.shape[0]
        l_pos = torch.bmm(pos_1.view(bsz, 1, -1), pos_2.view(bsz, -1, 1))
        l_pos = l_pos.view(bsz, 1)
        # print("l_pos.shape: ", l_pos.shape)
        l_neg = torch.mm(pos_1.view(bsz, -1), neg_value.t())
        # l_neg = torch.bmm(pos_1.view(bsz, 1, -1), neg_value.view(bsz, -1, 1))
        # l_neg = l_neg.view(bsz, 1)
        # print("l_neg.shape: ", l_neg.shape)

        # l_neg_rep = l_neg.t().repeat(bsz, 1)
        logits = torch.cat((l_pos, l_neg), dim=1)
        # logits = torch.cartesian_prod(l_pos.squeeze(), l_neg.squeeze())
        # print("logits.shape: ", logits.shape)
        # print(logits)
        logits = logits.squeeze().contiguous()
        return self.criterion(logits / self.args['t'])

    def update(self, network: nn.Module):
        for key_param, query_param in zip(self.parameters(), network.parameters()):
            key_param.data *= self.args['momentum']
            key_param.data += (1 - self.args['momentum']) * query_param.data
        self.eval()

    def forward(self, batch):
        batch = batch.to(self.device)
        batch_in = batch[:, :, :LaBSE_DIM]
        adj = batch[:, :, LaBSE_DIM:]

        center = batch_in[:, 0].to(self.device)
        center_neigh = batch_in.to(self.device)

        for i in range(0, self.args['gat_num']):
            center_neigh = self.attn(center_neigh, adj.bool()).squeeze(1)
        
        center_neigh = center_neigh[:, 0]

        if self.args['center_norm']:
            center = F.normalize(center, p=2, dim=1)
        if self.args['neighbor_norm']:
            center_neigh = F.normalize(center_neigh, p=2, dim=1)
        if self.args['combine']:
            out_hat = torch.cat((center, center_neigh), dim=1)
            out_hat = self.attn_mlp(out_hat)
            if self.args['emb_norm']:
                out_hat = F.normalize(out_hat, p=2, dim=1)
        else:
            out_hat = center_neigh

        return out_hat


class BatchMultiHeadGraphAttention(nn.Module):
    def __init__(self, device, args, n_head=MULTI_HEAD_DIM, f_in=LaBSE_DIM, f_out=LaBSE_DIM, bias=True):
        super(BatchMultiHeadGraphAttention, self).__init__()
        self.device = device
        self.n_head = n_head
        self.w = Parameter(torch.Tensor(n_head, f_in, f_out))
        self.a_src = Parameter(torch.Tensor(n_head, f_out, 1))
        self.a_dst = Parameter(torch.Tensor(n_head, f_out, 1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(args['dropout'])
        if bias:
            self.bias = Parameter(torch.Tensor(f_out))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)

        nn.init.xavier_uniform_(self.w)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, h, adj):
        bs, n = h.size()[:2]  # h is of size bs x n x f_in
        h_prime = torch.matmul(h.unsqueeze(1), self.w)  # bs x n_head x n x f_out
        attn_src = torch.matmul(torch.tanh(h_prime), self.a_src)  # bs x n_head x n x 1
        attn_dst = torch.matmul(torch.tanh(h_prime), self.a_dst)  # bs x n_head x n x 1
        attn = attn_src.expand(-1, -1, -1, n) + attn_dst.expand(-1, -1, -1, n).permute(0, 1, 3, 2)  # bs x n_head x n x n

        attn = self.leaky_relu(attn)
        mask = ~(adj.unsqueeze(1) | torch.eye(adj.shape[-1]).bool().to(self.device))  # bs x 1 x n x n
        attn.data.masked_fill_(mask, float("-inf"))
        attn = self.softmax(attn)  # bs x n_head x n x n
        attn = self.dropout(attn)
        # logging.info("attn: ", attn)
        # logging.info("attn.shape: ", attn.shape)
        output = torch.matmul(attn, h_prime)  # bs x n_head x n x f_out
        if self.bias is not None:
            return output + self.bias
        else:
            return output
        
class MyKG:
    def __init__(self, args, device, path, loader):
        self.args = args
        self.device = device
        self.path = path + '/' + self.args['language']
        self.loader = loader
        self.lr = self.args['lr']
        self.model = MyEmbedder(self.args, VOCAB_SIZE).to(self.device)
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.lr)
    def cal_sim(self, v1, v2, link, ids_1, inverse_ids_2):
        source = [_id for _id in ids_1 if _id in link]
        target = np.array(
            [inverse_ids_2[link[_id]] if link[_id] in inverse_ids_2 else 99999 for _id in source])
        src_idx = [idx for idx in range(len(ids_1)) if ids_1[idx] in link]
        v1 = np.concatenate(tuple(v1), axis=0)[src_idx, :]
        v2 = np.concatenate(tuple(v2), axis=0)
        index = faiss.IndexFlatIP(v2.shape[1])
        index.add(np.ascontiguousarray(v2))
        D, I = index.search(np.ascontiguousarray(v1), 10)
        return source, target, D, I # D是相似性矩阵， I是ID矩阵
    def evaluate(self, model, eval_loader1, eval_loader2, link, step, predict = False):
        # print("Evaluate at epoch {}...".format(step))

        ids_1, ids_2, vector_1, vector_2 = list(), list(), list(), list()
        inverse_ids_2 = dict()
        ids_2_index = dict()
        with torch.no_grad():
            model.eval()
            for sample_id_1, (token_data_1, id_data_1) in enumerate(eval_loader1):
                entity_vector_1 = model(token_data_1).squeeze().detach().cpu().numpy()
                ids_1.extend(id_data_1.squeeze().tolist())
                vector_1.append(entity_vector_1)

            for sample_id_2, (token_data_2, id_data_2) in enumerate(eval_loader2):
                entity_vector_2 = model(token_data_2).squeeze().detach().cpu().numpy()
                ids_2.extend(id_data_2.squeeze().tolist())
                vector_2.append(entity_vector_2)

        for idx, _id in enumerate(ids_2):
            inverse_ids_2[_id] = idx 
            ids_2_index[idx] = _id
        source, target, D, I = self.cal_sim(vector_1, vector_2, link, ids_1, inverse_ids_2)
        def cal_hit(source, target, D, I):
            # print(len(I))
            hit1 = (I[:, 0] == target).astype(np.int32).sum() / len(source)
            hit10 = (I == target[:, np.newaxis]).astype(np.int32).sum() / len(source)
            # print("#Entity: {}".format(len(source)))
            # print("Hit@1: {}".format(round(hit1, 3)))
            # print("Hit@10:{}".format(round(hit10, 3)))
            return round(hit1, 3), round(hit10, 3)
        if predict:
            I = np.array(list(map(lambda x: [ids_2_index[y] for y in x], I)))
            return source, target, D, I, vector_1
        # print('===========Test===========')
        hit1_test, hit10_test = cal_hit(source, target, D, I)
        return hit1_test, hit10_test

    def train(self):
        start_time = datetime.now()
        self.evaluate(self.model, self.loader.eval_loader1, self.loader.eval_loader2, self.loader.link, 0)
        best_hit1_valid_epoch = 0
        best_hit10_valid_epoch = 0
        best_hit1_test_epoch = 0
        best_hit10_test_epoch = 0
        best_hit1_valid = 0
        best_hit10_valid = 0
        best_hit1_valid_hit10 = 0
        best_hit10_valid_hit1 = 0
        best_hit1_test = 0
        best_hit10_test = 0
        best_hit1_test_hit10 = 0
        best_hit10_test_hit1 = 0
        record_hit1 = 0
        record_hit10 = 0
        record_epoch = 0
        record_batch_id = 0
        last_loss = 99999
        for epoch in range(self.args['epoch']):
            adjust_learning_rate(self.optimizer, epoch, self.lr)
            for batch_id, (x_ids, y_ids, ny_ids) in enumerate(self.loader.all_data_batches):
                kg1_train_ent_idx = list(map(lambda x: int(x), list(x_ids)))
                kg1_train_ent_emb = None 
                kg2_train_ent_idx = list(map(lambda x: int(x), list(y_ids)))
                kg2_train_ent_emb = None 
                neg_ent_idx = list(map(lambda x: int(x), list(ny_ids)))
                neg_ent_emb = None
                with torch.no_grad():
                    for idx in kg1_train_ent_idx:
                        if kg1_train_ent_emb==None:
                            kg1_train_ent_emb = self.loader.myset1.id_emb[idx].unsqueeze(0)
                        else:
                            kg1_train_ent_emb = torch.cat((kg1_train_ent_emb,\
                                                        self.loader.myset1.id_emb[idx].unsqueeze(0)),0)
                    for idx in kg2_train_ent_idx:
                        if kg2_train_ent_emb==None:
                            kg2_train_ent_emb = self.loader.myset2.id_emb[idx].unsqueeze(0)
                        else:
                            kg2_train_ent_emb = torch.cat((kg2_train_ent_emb,\
                                                        self.loader.myset2.id_emb[idx].unsqueeze(0)),0)
                    # kg1_train_ent_emb.append(myset1.id_emb[idx])
                    for idx in neg_ent_idx:
                        if neg_ent_emb==None:
                            neg_ent_emb = self.loader.myset2.id_emb[idx].unsqueeze(0)
                        else:
                            neg_ent_emb = torch.cat((neg_ent_emb,\
                                                        self.loader.myset2.id_emb[idx].unsqueeze(0)),0)
                    
                
                self.optimizer.zero_grad()
                pos_1 = self.model(kg1_train_ent_emb)
                pos_2 = self.model(kg2_train_ent_emb)
                neg = self.model(neg_ent_emb)
                contrastive_loss = self.model.contrastive_loss(pos_1, pos_2, neg)

                contrastive_loss.backward(retain_graph=True)
                self.optimizer.step()

                if batch_id == len(self.loader.all_data_batches) - 1:
                # if batch_id % 200 == 0 or batch_id == len(all_data_batches) - 1:
                    loss = contrastive_loss.detach().cpu().data / 64
                    # print('epoch: {} batch: {} loss: {}'.format(epoch, batch_id, loss))
                    hit1_test, hit10_test = self.evaluate(self.model, self.loader.eval_loader1, self.loader.eval_loader2, self.loader.link, str(epoch)+": batch "+str(batch_id))

                    if hit1_test > best_hit1_test:
                        best_hit1_test = hit1_test
                        best_hit1_test_hit10 = hit10_test
                        best_hit1_test_epoch = epoch
                    if hit10_test  > best_hit10_test:
                        best_hit10_test = hit10_test
                        best_hit10_test_hit1 = hit1_test
                        best_hit10_test_epoch = epoch
                    
            #         # print('Test Hit@1(10)    = {}({}) at epoch {} batch {}'.format(hit1_test, hit10_test, epoch, batch_id))
            #         # print('Best Valid Hit@1  = {}({}) at epoch {}'.format(best_hit1_valid, best_hit1_valid_hit10, best_hit1_valid_epoch))
            #         # print('Best Valid Hit@10 = {}({}) at epoch {}'.format(best_hit10_valid,best_hit10_valid_hit1, best_hit10_valid_epoch))
            #         # print('Test @ Best Valid = {}({}) at epoch {} batch {}'.format(record_hit1, record_hit10, record_epoch, record_batch_id))

            #         print('Best Test  Hit@1  = {}({}) at epoch {}'.format(best_hit1_test, best_hit1_test_hit10, best_hit1_test_epoch))
            #         print('Best Test  Hit@10 = {}({}) at epoch {}'.format(best_hit10_test,best_hit10_test_hit1, best_hit10_test_epoch))
            #         print("====================================")
            # print(last_loss, loss)
            if abs(last_loss - loss) < 0.000001:
                print("Should early stop at epoch" , epoch)
                break
            last_loss = loss
        print('Best Test  Hit@1  = {}({}) at epoch {}'.format(best_hit1_test, best_hit1_test_hit10, best_hit1_test_epoch))
        print('Best Test  Hit@10 = {}({}) at epoch {}'.format(best_hit10_test,best_hit10_test_hit1, best_hit10_test_epoch))
        
        end_time = datetime.now()
        print("start: "+start_time.strftime("%Y-%m-%d %H:%M:%S"))
        print("end: "+end_time.strftime("%Y-%m-%d %H:%M:%S"))
        print("used_time: "+ str(end_time - start_time))
        print("====================================")
        return self.evaluate(self.model, self.loader.eval_loader1, self.loader.eval_loader2, self.loader.link, str(epoch)+": batch "+str(batch_id), predict = True)