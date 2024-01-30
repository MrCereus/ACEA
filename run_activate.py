from model import MyKG
import torch
from datetime import datetime
from load import MyDataserLoader
from strategy_util import *
args = {
    'device':'cuda:0',
    'time':datetime.now().strftime("%Y%m%d%H%M%S"),
    'language':'zh_en',
    'epoch':150,
    'batch_size':64,
    'queue_length':64,
    'center_norm':False,
    'neighbor_norm':True,
    'emb_norm':True,
    'combine':True,
    'gat_num':1,
    't': 0.08,
    'momentum':0.9999,
    'lr':1e-6,
    'dropout':0.3
}
device = torch.device('cuda')
path = "/home/mrcactus/Thesis/ACEA/data/DBP15K"
# args['language'] = 'zh_en'
print("Language: " + args['language'])
g = construct_graph(path+'/'+args['language'], 'origin')
loader = MyDataserLoader(path, args)
tot = 750
while tot <= 4500:
    print("Train data size: " + str(tot))
    model = MyKG(args, device, path, loader)
    source, target, D, I = model.train()
    selected_ent = measure_struct_uncertainty(g, source, D, 0.1)
    # selected_ent = measure_random(source)
    source_I = {source[i]:I[:,0:2][i] for i in range(len(source))}
    loader.update_data(selected_ent[:750], source_I, rand=True)
    tot += 750