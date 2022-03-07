# ------ Import necessary packages ----
import math
import random
from numpy import isin
import pandas as pd
import networkx as nx
from collections import defaultdict
from itertools import combinations
import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt

import dimod
import dwave.inspector
from minorminer import find_embedding
from dwave.system import LeapHybridSampler
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite, LazyFixedEmbeddingComposite, FixedEmbeddingComposite

import json
import pickle

def define_dirs(n, k, dim, ord, g=10, custom="", type=0):
    # n-size, k-k_nn, dim-dimensions, ord-max_degree, g-gamma, custom-for one's needs, type-type of graph
    type_names = ["_", "_trimmed_", "_negedges_", "_trimmed_negedges_"]
    dirs = {
        "name"      : ''.join([str(n), "_graph_snn", "_k", str(k), "_dim", str(dim), "_g", str(g), type_names[type], str(ord), custom, ".gexf"]),
        "graph_in"  : ''.join(["./DatasetsIn/"  , str(n), "_graph_snn", "_k", str(k), "_dim", str(dim),               type_names[type], str(ord), custom, ".gexf"]),
        "graph_out" : ''.join(["./DatasetsOut/" , str(n), "_graph_snn", "_k", str(k), "_dim", str(dim), "_g", str(g), type_names[type], str(ord), custom, "_out.gexf"]),
        "img_in"    : ''.join(["./OutputIn/"    , str(n), "_graph_snn", "_k", str(k), "_dim", str(dim), "_g", str(g), type_names[type], str(ord), custom, ".png"]),
        "img_out"   : ''.join(["./OutputOut/"   , str(n), "_graph_snn", "_k", str(k), "_dim", str(dim), "_g", str(g), type_names[type], str(ord), custom, "_out.png"]),
        "embedding" : ''.join(["./Embedding/"   , str(n), "_graph_snn", "_k", str(k), "_dim", str(dim), "_g", str(g), type_names[type], str(ord), custom, ".pkl"])
    }
    return dirs

def create_graph(dirs):
    G = nx.read_gexf(dirs["graph_in"])
    pos = nx.spring_layout(G)
    return G, pos

def plot_and_save_graph_in(G, pos, dirs):
    plt.cla()
    nx.draw_networkx_nodes(G, pos, node_size=10, nodelist=G.nodes)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges, style='solid', alpha=0.5, width=1)
    plt.savefig(dirs["img_in"], bbox_inches='tight')
    print("graph saved as: ", dirs["img_in"])

def plot_and_save_graph_out(G, pos, dirs):
    cut_edges = [(u, v) for u, v in G.edges if list(G.nodes[u].values())[-1]!=list(G.nodes[v].values())[-1]]
    uncut_edges = [(u, v) for u, v in G.edges if list(G.nodes[u].values())[-1]==list(G.nodes[v].values())[-1]]

    len(cut_edges)
    len(uncut_edges)

    # colors = [sum(list(y.values())) for x,y in G.nodes(data=True)]
    colors = [list(y.values())[-1] for x,y in G.nodes(data=True)]

    # ------- plot and & output graph -------
    plt.cla()

    nx.draw_networkx_nodes(G, pos, node_size=10, nodelist=G.nodes,  node_color=colors)
    nx.draw_networkx_edges(G, pos, edgelist=cut_edges, style='dashdot', alpha=0.5, width=1)
    nx.draw_networkx_edges(G, pos, edgelist=uncut_edges, style='solid', width=1)

    plt.savefig(dirs["img_out"], bbox_inches='tight')

    nx.write_gexf(G, dirs["graph_out"])

def clustering(G, iteration, dirs, name, solver, gamma_factor, gamma_type, color=0, ising=True):

    label = ''.join([name, solver]) 
    
    if ising:
        h = defaultdict(int)
        J = defaultdict(int)
        # Fill in h
        for i in G.nodes:
            h[i] += gamma
        # Fill in J
        for u, v in G.edges:
            J[(u,v)] += -G.get_edge_data(u, v)["weight"]
    else:
        # Initialize our Q matrix
        Q = defaultdict(int)
        
        if gamma_type == "by_nodes":
            gamma = 10 / len(G.nodes)
        elif gamma_type == "by_edges":
            edges_weights = G.size(weight="weight")
            nodes_weights = len(G.edges)
            ratio = edges_weights/nodes_weights 
            gamma = gamma_factor * ratio

        # Fill in Q matrix
        for u, v in G.edges:
            
            Q[(u,u)] += G.get_edge_data(u, v)["weight"]
            Q[(v,v)] += G.get_edge_data(u, v)["weight"]
            Q[(u,v)] += -2*G.get_edge_data(u, v)["weight"]

        for i in G.nodes:
            Q[(i,i)] += gamma*(1-len(G.nodes))

        for i, j in combinations(G.nodes, 2): # do you need this term ???
            Q[(i,j)] += 2*gamma

    # --------------
    print("Running on QPU...")
    
    num_reads = 1000
    chain_strength = 4

    if solver == "hybrid":
        sampler = LeapHybridSampler()
        if ising:
            response = sampler.sample_ising(h, J, label=name)
        else:
            response = sampler.sample_qubo(Q, label=name)
    elif solver == "fixed_embedding":
        try:
            a_file = open(dirs["embedding"], "rb")
            embedding = pickle.load(a_file)
            a_file.close()
            sampler = FixedEmbeddingComposite(DWaveSampler(), embedding)
        except IOError:
            embedding = find_embedding(Q, DWaveSampler().edgelist)         
            a_file = open(dirs["embedding"], "wb")
            pickle.dump(embedding, a_file)
            a_file.close()
            sampler = FixedEmbeddingComposite(DWaveSampler(), embedding)
        if ising:
            response = sampler.sample_isng(h, J, label=name, chain_strength=chain_strength, num_reads=num_reads)
        else:
            response = sampler.sample_qubo(Q, label=name, chain_strength=chain_strength, num_reads=num_reads)
    elif solver == "embedding_composite":
        sampler = EmbeddingComposite(DWaveSampler())
        if ising:
            response = sampler.sample_isng(h, J, label=name, chain_strength=chain_strength, num_reads=num_reads)
        else:
            response = sampler.sample_qubo(Q, label=name, chain_strength=chain_strength, num_reads=num_reads)

    # ------- Print results to user -------
    print('-' * 60)
    print('{:>15s}{:>15s}{:^15s}{:^15s}'.format('Set 0','Set 1','Energy','Cut Size'))
    print('-' * 60)

    i=0
    for sample, E in response.data(fields=['sample','energy']):
        # select clusters
        S0 = [k for k,v in sample.items() if v == 0]
        S1 = [k for k,v in sample.items() if v == 1]

        print('{:>15s}{:>15s}{:^15s}{:^15s}'.format(str(S0),str(S1),str(E),str(int(-1*E))))
        
        if (i > 3):
            break
        i = i + 1

    
    label = "label" + str(iteration)
    lut = response.first.sample

    # Interpret best result in terms of nodes and edges
    S0 = [node for node in G.nodes if not lut[node]]
    S1 = [node for node in G.nodes if lut[node]]

    print("S0 length: ", len(S0))
    print("S1 length: ", len(S1))
    if(len(S0)>10 and len(S1)>10):
        # Assign nodes' labels
        col = random.randint(0, 100)
        for i in S0:
            # G.nodes(data=True)[i][label] = 100 - color
            G.nodes(data=True)[i][label] = col
        
        col = random.randint(120, 220)    
        for i in S1:
            # G.nodes(data=True)[i][label] = color - 100
            G.nodes(data=True)[i][label] = col
        # write to the graph file
        # file_name = "clustring_" + str(iteration) + ".gexf"
        # nx.write_gexf(G, file_name)

        clustering(G.subgraph(S0), iteration+1, name, color+20)
        clustering(G.subgraph(S1), iteration+1, name, color+20)

solvers = {
    "h"     : "hybrid",
    "fe"    : "fixed_embedding",
    "ec"    : "embedding_composite"
}
n = 128     # size of the graph
k = 5       # k_nn used for SNN
ord = 15    # maximum order of node degree when "trimmed" mode is enabled
dim = 15    # number of dimensions used in SNN
solver = solvers["fe"]
type = 0    #["_", "_trimmed_", "_negedges_", "_trimmed_negedges_"], where "_" -> unaltered SNN output
g = 1       # gamma
custom = "" # additional metadata for file names

dirs = define_dirs(n, k, dim, ord, g, custom, type)

G, pos = create_graph(dirs)

plot_and_save_graph_in(G, pos, dirs)