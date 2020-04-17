import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
import random
from scipy.special import comb
from scipy.stats import linregress
random.seed(0)

""" The return values in all functions are only for demonstrating the format, and should be overwritten """

def graph_hist(G):
    x = {}
    for k,v in G.degree:
        if v in x.keys():
            x[v] += 1
        else:
            x[v] = 1
    x = sorted(x.items())
    return zip(*x)

def query_infected_nodes(G):
    infected = []
    for node in G.nodes.keys():
        if G.nodes[node]['infected']:
            infected.append(int(node))
    return infected

class ModelData:
    def __init__(self, dataset):
        self.init_graph = pd.read_csv(dataset)
        self.nodes = self.init_graph[['source', 'target']]

def graphA(links_data):
    graph = nx.Graph()
    edges = [tuple(x) for x in links_data.nodes.values]
    graph.add_edges_from(edges)
    return graph

def calc_best_power_law(G):
    x,y = graph_hist(G)
    slope,intercept,_,_,_ = linregress(np.log(x), np.log(y))
    return -slope, math.exp(intercept)

def plot_histogram(G, alpha, beta):
    x,y = graph_hist(G)
    plt.title('Q1S3 - Node degree histogram log-log scale')
    plt.loglog(x, y, marker="s", linewidth=0)
    plt.loglog(x,beta*(x**(-alpha)))
    # plt.show()
    plt.close()
    # -- another option for plotting --
    # plt.plot(np.log(x), np.log(y))
    # plt.plot(np.log(x),np.log(x)*(-alpha)+np.log(beta))
    return

def G_features(G):
    closeness = nx.closeness_centrality(G)
    betweeness = nx.betweenness_centrality(G)
    return {'Closeness': closeness,'Betweeness':betweeness}

def create_undirected_weighted_graph(links_data , users_data, question):
    graph = nx.Graph()
    edges = [tuple(x) for x in links_data.init_graph.values]
    graph.add_weighted_edges_from(edges)
    for key, value in users_data.iterrows():
        if str(value[question]).lower() == 'no':
            graph.nodes[value['node']]['infected'] = False
        else:
            graph.nodes[value['node']]['infected'] = True
    return graph

def run_k_iterations(WG, k , Threshold):
    S = dict()
    S[0] = query_infected_nodes(WG)
    for i in range(1,k+1):
        S[i] = []
        for node in WG.nodes:
            if not WG.nodes[node]['infected']:
                sum = 0
                for neighbor in WG[node]:
                    if WG.nodes[neighbor]['infected']:
                        sum += WG[node][neighbor]['weight']
                if sum >= Threshold:
                    S[i].append(int(node))
        for node in S[i]:
            WG.nodes[node]['infected'] = True
    return S

def calculate_branching_factor(S,k):
    branching_fac = {}
    sum = 0
    for i in range(1,k+1):
        if len(S[i-1])==0:
            break
        branching_fac[i] = float(len(S[i])) / len(S[i-1])
        sum += branching_fac[i]
    return branching_fac

def find_maximal_h_deg(WG,h):
    degrees = WG.degree
    degrees = sorted(degrees, key=lambda x: x[1], reverse=True)    
    return dict(degrees[:h])


def calculate_clustering_coefficient(WG,nodes_dict):
    nodes_dict_clustering = {}
    for node in nodes_dict:
        neighbors = list(WG.neighbors(node))
        if len(list(neighbors)) == 0:
            nodes_dict_clustering[node] = 0
            continue
        connected_neighbors = 0
        for y,z in [(y,z) for y in neighbors for z in neighbors if y>z]:
            if WG.has_edge(y,z):
                connected_neighbors += 1
        n = len(neighbors)
        nodes_dict_clustering[node] = connected_neighbors / ((n*(n-1))/2)
    return nodes_dict_clustering

def infected_nodes_after_k_iterations(WG, k, Threshold):
    infection_iterations = run_k_iterations(WG, k, Threshold)
    infected = set()
    for nodes in infection_iterations.values():
        infected.update(nodes)
    return len(infected)

def slice_dict(dict_nodes,number):
    # nodes_list = sorted(dict_nodes.items(), key=lambda x: x[1], reverse=True)[:number]
    return list(dict_nodes.keys())[:number]

# remove all nodes in [nodes_list] from the graph WG, with their edges, and return the new graph
def nodes_removal(WG,nodes_list):
    _WG = WG.copy()
    _WG.remove_nodes_from(nodes_list)
    return _WG

# plot the graph according to Q4 , add the histogram to the pdf and run the program without it
def graphB(number_nodes_1,number_nodes_2,number_nodes_3):
    x_1, y_1 = zip(*sorted(number_nodes_1.items(), key=lambda x: x[0]))
    x_2, y_2 = zip(*sorted(number_nodes_2.items(), key=lambda x: x[0]))
    x_3, y_3 = zip(*sorted(number_nodes_3.items(), key=lambda x: x[0]))
    plt.clf()
    plt.title('Q4 - GraphB')
    line1 = plt.plot(x_1, y_1, c='g', marker='^', label="Random")
    line2 = plt.plot(x_2, y_2, c='r', marker='_', label="Sorted ASC")
    line3 = plt.plot(x_3, y_3, c='b', marker='s', label="Sorted DSC")
    plt.legend()
    # plt.show()
    plt.close()
    return