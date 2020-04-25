import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms import bipartite
import random
import csv
from scipy.special import comb
import matplotlib.pyplot as plt
random.seed(0)

class ModelData:
    """The class reads 5 files as specified in the init function, it creates basic containers for the data.
    See the get functions for the possible options, it also creates and stores a unique index for each user and movie
    """

    def __init__(self, dataset):
        """Expects data set file with index column (train and test) """
        self.init_graph = pd.read_csv(dataset)
        self.nodes = self.init_graph[['source', 'target']]

def new_w(w):

    if w>0.5:
        return 0.5
    if w>0.4:
        return 0.4
    if w>0.35:
        return 0.35
    if w>0.3:
        return 0.3
    if w>0.2:
        return 0.25
    if w>0.15:
        return 0.075
    else:
        return 0.05


# def competitive_part(links_dataset, number):
#     G = nx.from_pandas_edgelist(links_dataset.init_graph, 'target', 'source', edge_attr=True, create_using=nx.Graph())
#     iter = 0
#     list_of_removed = list()
#     ITERS=50
#     hist={0.5:0,0.4:0,0.3:0,0.35:0,0.2:0,0.15:0,0.1:0, 0.075:0, 0.05:0}

#     for i in range(ITERS):
#         cc = {}
#         all_nodes = list(G.nodes)
#         while len(all_nodes)>0:
#             n = all_nodes[0]
#             the_cc = nx.node_connected_component(G, n)
#             for n2 in the_cc:
#                 cc[n2] = len(the_cc)
#                 all_nodes.remove(n2)
#         weight_dict = dict()
#         for edge in G.edges():
#             node1 = edge[0]
#             node2 = edge[1]
#             ne1 = set(list(G.neighbors(node1)))
#             ne2 = set(list(G.neighbors(node2)))
#             if (cc[node1]>20):
#                 nw = new_w(G[edge[0]][edge[1]]['weight'])
#                 weight_dict[node1, node2] = np.log(len(ne1)*len(ne2))*(nw)
#             else:
#                 weight_dict[node1, node2] = 0 

#         weight_orderd_dict = dict(sorted(weight_dict.items(), key=lambda item: item[1], reverse=True))
#         for key1,key2 in weight_orderd_dict.keys():
#             if iter >= ((i+1)/ITERS)*number:
#                 break
#             iter = iter + 1
#             nw = new_w(G[key1][key2]['weight'])
#             hist[nw]+=1
#             G.remove_edge(key1,key2)
#             list_of_removed.append([key1,key2])

#     return list_of_removed

def competitive_part(links_dataset , number):
    # Edge Filtering
    k = 1
    G = nx.from_pandas_edgelist(links_dataset.init_graph, 'target', 'source', edge_attr=True, create_using=nx.Graph())
    removed_edges = list()
    for i in range(number // k):
        print("iter = ", i)

        bridges = list(nx.bridges(G))
        # ranking = nx.pagerank(G)
        weight_dict = dict()

        # reweight edges
        for edge in G.edges():
            node1 = edge[0]
            node2 = edge[1]
            deg1 = len(list(G.neighbors(node1)))
            deg2 = len(list(G.neighbors(node2)))
            weight = G[node1][node2]['weight']
            weight_dict[node1, node2] = np.log(deg1 * deg2) * new_w(weight) * (edge not in bridges)

                
        # sort descending
        weight_dict = sorted(weight_dict.items(), key=lambda x:x[1], reverse=True)
        
        i = 0
        # remove bridges with maximal weighted span
        for edge in dict(weight_dict).keys():
            if i == k:
                break
            removed_edges.append(edge)
            G.remove_edge(edge[0], edge[1])
            i += 1

    while len(removed_edges) < number:
        removed_edges.append(local_bridges[-1])
        local_bridges = local_bridges[:-1]

    return removed_edges

def write_file_competition (list_of_removed):
    filename = "hw1_competition.csv"
    with open(filename, 'w') as csv_file:
        writer = csv.writer(csv_file, lineterminator='\n')
        fieldnames2 = ["source", "target"]
        writer.writerow(fieldnames2)
        for edge in list_of_removed:
            writer.writerow([edge[0], edge[1]])

