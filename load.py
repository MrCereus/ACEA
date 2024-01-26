import torch
from torch.utils.data import Dataset
from settings import *
import pickle
import pandas as pd
import torch.utils.data as Data
from datetime import datetime

def adjust_learning_rate(optimizer, epoch, lr):
    if (epoch+1) % 10 == 0:
        lr *= 0.5
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def load_triples(data_dir, file_num=2):
    if file_num == 2:
        file_names = [data_dir + str(i) for i in range(1, 3)]
    else:
        file_names = [data_dir]
    triple = []
    for file_name in file_names:
        with open(file_name, "r", encoding="utf-8") as f:
            data = f.read().strip().split("\n")
            data = [tuple(map(int, i.split("\t"))) for i in data]
            triple += data
    np.random.shuffle(triple)
    return triple

class DBP15KRawNeighbors():
    def __init__(self, path, language, doc_id):
        self.language = language
        self.doc_id = doc_id
        self.path = path# join(path, self.language)
        self.id_entity = {}
        # self.id_neighbor_loader = {}
        self.id_adj_tensor_dict = {}
        self.id_neighbors_dict = {}
        self.load()
        self.id_neighbors_loader()
        self.get_center_adj()

    def load(self):
        with open(join(self.path, "raw_LaBSE_emb_" + self.doc_id + '.pkl'), 'rb') as f:
            self.id_entity = pickle.load(f)


    def id_neighbors_loader(self):
        data = pd.read_csv(join(self.path, 'triples_' + self.doc_id), header=None, sep='\t')
        data.columns = ['head', 'relation', 'tail']
        # self.id_neighbor_loader = {head: {relation: [neighbor1, neighbor2, ...]}}

        for index, row in data.iterrows():
            # head-rel-tail, tail is a neighbor of head
            # print("int(row['head']): ", int(row['head']))
            head_str = self.id_entity[int(row['head'])][0]
            tail_str = self.id_entity[int(row['tail'])][0]

            if not int(row['head']) in self.id_neighbors_dict.keys():
                self.id_neighbors_dict[int(row['head'])] = [head_str]
            if not tail_str in self.id_neighbors_dict[int(row['head'])]:
                self.id_neighbors_dict[int(row['head'])].append(tail_str)
            
            if not int(row['tail']) in self.id_neighbors_dict.keys():
                self.id_neighbors_dict[int(row['tail'])] = [tail_str]
            if not head_str in self.id_neighbors_dict[int(row['tail'])]:
                self.id_neighbors_dict[int(row['tail'])].append(head_str)
    
    def get_adj(self, valid_len):
        adj = torch.zeros(NEIGHBOR_SIZE, NEIGHBOR_SIZE).bool()
        for i in range(0, valid_len):
            adj[i, i] = 1
            adj[0, i] = 1
            adj[i, 0] = 1
        return adj

    def get_center_adj(self):
        for k, v in self.id_neighbors_dict.items():
            if len(v) < NEIGHBOR_SIZE:
                self.id_adj_tensor_dict[k] = self.get_adj(len(v))
                self.id_neighbors_dict[k] = v + [[0]*LaBSE_DIM] * (NEIGHBOR_SIZE - len(v))
            else:
                self.id_adj_tensor_dict[k] = self.get_adj(NEIGHBOR_SIZE)
                self.id_neighbors_dict[k] = v[:NEIGHBOR_SIZE]

class SeedDataset(Dataset):
    def __init__(self, seed):
        super(SeedDataset, self).__init__()
        self.x_train = [[k] for k in seed[0]]
        self.y_train = [[k] for k in seed[1]]
        self.num = len(self.x_train)
        self.x_train = torch.Tensor(self.x_train).long()
        self.y_train = torch.Tensor(self.y_train).long()

    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index]

    def __len__(self):
        return self.num
    
class MyRawdataset(Dataset):
    def __init__(self, id_features_dict, adj_tensor_dict,is_neighbor=True):
        super(MyRawdataset, self).__init__()
        self.num = len(id_features_dict)  # number of samples

        self.x_train = []
        self.x_train_adj = None
        self.y_train = []
        self.id_emb = {}

        for k in id_features_dict:
            if is_neighbor:
                if self.x_train_adj==None:
                    self.x_train_adj = adj_tensor_dict[k].unsqueeze(0)
                else:
                    self.x_train_adj = torch.cat((self.x_train_adj, adj_tensor_dict[k].unsqueeze(0)), dim=0)
            self.x_train.append(id_features_dict[k])
            self.y_train.append([k])

        # transfer to tensor
        # if type(self.x_train[0]) is list:
        self.x_train = torch.Tensor(self.x_train)
        if is_neighbor:
            self.x_train = torch.cat((self.x_train, self.x_train_adj), dim=2) 
            ###
            # 拼接邻居嵌入
            # 可以在这里对邻居嵌入进行操作，比如换成平均关系嵌入
            ###
        self.y_train = torch.Tensor(self.y_train).long()
        for idx in range(self.num):
            self.id_emb[int(self.y_train[idx][0])] = self.x_train[idx]

    # indexing
    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index]

    def __len__(self):
        return self.num
