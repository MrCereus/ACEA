from openea.reader import *
import networkx as nx
import pandas as pd
import numpy as np 
from networkx.algorithms.link_analysis.pagerank_alg import pagerank
import random
from sklearn.cluster import KMeans

def construct_graph(data_dir, edge_mode):
    # construct graph
    kg1, kg2, _ = read_kgs_n_links(data_dir)
    g = nx.Graph()
    g.add_nodes_from(kg1.entities_list)
    self_ent_pair = [(ent, ent) for ent in kg1.entities_list]
    triples_list = kg1.relation_triples_list

    if edge_mode == "add_inverse" or edge_mode == "add_inverse_func":
        inv_triple_list = [(tri[2], "inv_" + tri[1], tri[0]) for tri in triples_list]
        triples_list += inv_triple_list

    if edge_mode == "origin":
        ent_pairs = [(tri[0], tri[2]) for tri in triples_list]
    else:
        ent_pairs = [(tri[2], tri[0]) for tri in triples_list]

    if edge_mode == "basic_func" or edge_mode == "add_inverse_func":
        triple_df = pd.DataFrame(triples_list, columns=["head", "relation", "tail"])
        relation_types = triple_df["relation"].unique()
        rel2func_map = dict()
        for rel in relation_types:
            triples_of_rel = triple_df[triple_df["relation"] == rel]
            func = len(triples_of_rel["head"].unique()) / len(triples_of_rel)
            rel2func_map[rel] = func
        all_pairs = ent_pairs + self_ent_pair
        edge_weights = [rel2func_map[tri[1]] for tri in triples_list] + [1.0] * len(self_ent_pair)
        g.add_edges_from(all_pairs, weight=edge_weights)
    else:
        g.add_edges_from(ent_pairs + self_ent_pair)
    return g



def measure_uncertainty(simi_mtx, topK=5, measure="entropy"):  # measure options: entropy, margin, variation_ratio, similarity
    sorted_simi_mtx = np.sort(simi_mtx, axis=-1)
    if measure == "entropy":
        topk_simi_mtx = sorted_simi_mtx[:, -topK:]
        prob_mtx = topk_simi_mtx / topk_simi_mtx.sum(axis=1, keepdims=True)
        uncertainty = - np.sum(prob_mtx*np.log2(prob_mtx), axis=1)
    elif measure == "margin":
        margin = sorted_simi_mtx[:, -1] - sorted_simi_mtx[:, -2]
        uncertainty = - margin  # larger margin means small uncertainty
        uncertainty = uncertainty - uncertainty.min()
    elif measure == "variation_ratio":
        topk_simi_mtx = sorted_simi_mtx[:, -topK:]
        prob_mtx = topk_simi_mtx / topk_simi_mtx.sum(axis=1, keepdims=True)
        uncertainty = 1.0 - prob_mtx[:, -1]
    elif measure == "similarity":
        uncertainty = - sorted_simi_mtx[:, -1]
    else:
        raise Exception("unknown uncertainty measure")
    return uncertainty

def measure_kmeans(source, kg1_vector, k, loader):
    src_idx = [idx for idx in range(len(source)) if source[idx] in loader.link]
    v1 = np.concatenate(tuple(kg1_vector), axis=0)[src_idx,:]
    kmeans = KMeans(n_clusters=k, init='k-means++').fit(v1)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    distances = np.linalg.norm(v1 - centroids[labels], axis=1)
    score = (1 - distances) / (1 + distances)
    # 2/1+d -1 2-1-d=1-d
    return score

def measure_struct_uncertainty(graph, unlabeled_ent1_list, simi_mtx, pr_alpha, kg1_vector = None, k=0, loader=None, beta = 0.2):
    uncertainty = measure_uncertainty(simi_mtx, topK=10, measure='margin')
    if k != 0 and loader is not None and kg1_vector is not None:
        km = measure_kmeans(unlabeled_ent1_list, kg1_vector, k, loader) * beta
        ent1_uncertainty_map = {unlabeled_ent1_list[i]: uncertainty[i]+km[i] for i in range(len(unlabeled_ent1_list))}
    else:
        ent1_uncertainty_map = {unlabeled_ent1_list[i]: uncertainty[i] for i in range(len(unlabeled_ent1_list))}
    nodes = graph.nodes()
    node2weight_map = {n: ent1_uncertainty_map.get(n, 0.0) for n in nodes}
    # print(node2weight_map)
    new_weights = pagerank(graph, alpha=pr_alpha, personalization=node2weight_map, nstart=node2weight_map, dangling=None)  # todo: I dont need to use dangling. how to disable it? it will use personalization if I set is as None
    ent1_influence_map = {n: new_weights[n] for n in unlabeled_ent1_list}
    sorted_unlabeled_ent_list = sorted(unlabeled_ent1_list, key=lambda item: -ent1_influence_map.get(item))
    return sorted_unlabeled_ent_list

def measure_random(unlabeled_ent1_list):
    random.shuffle(unlabeled_ent1_list)
    return unlabeled_ent1_list