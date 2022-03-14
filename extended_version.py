# ------ Import necessary packages ----
from cProfile import label
from email import iterators
import math
import random
import types
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
from itertools import combinations
import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt

import dimod
import dwavebinarycsp
import dwave.inspector
from minorminer import find_embedding
from dwave.embedding import embed_ising
from dwave.system import LeapHybridSampler
from dwave.system import LeapHybridDQMSampler
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite, LazyFixedEmbeddingComposite, FixedEmbeddingComposite

import json
import pickle

def define_dirs(n, k, dim, ord, g, gf, custom,type):
    # n-size, k-k_nn, dim-dimensions, ord-max_degree, g-gamma, custom-for one's needs, type-type of graph
    type_names = ["_", "_trimmed_", "_negedges_", "_trimmed_negedges_"]
    g = str(g).replace( ".", "")

    dirs = {
        "name"              : ''.join([                   str(n), "_graph_snn"    , "_k", str(k), "_dim", str(dim),                 type_names[type], str(ord)]                       ),
        "graph_in"          : ''.join(["./DatasetsIn/"  , str(n), "_graph_snn"    , "_k", str(k), "_dim", str(dim),                 type_names[type], str(ord),         ".gexf"       ]),
        "graph_in_csv"      : ''.join(["./DatasetsIn/"  , str(n), "_graph_snn"    , "_k", str(k), "_dim", str(dim),                 type_names[type], str(ord),         ".csv"       ]),
        "graph_out_bqm"     : ''.join(["./DatasetsOut/" , str(n), "_graph_snn"    , "_k", str(k), "_dim", str(dim), "_gf", str(gf), type_names[type], str(ord), custom, "_out.gexf"   ]),
        "graph_out_dqm"     : ''.join(["./DatasetsOut/" , str(n), "_dqm_graph_snn", "_k", str(k), "_dim", str(dim), "_g", str(g)  , type_names[type], str(ord), custom, ".gexf"       ]),
        "img_in"            : ''.join(["./PlotsIn/"     , str(n), "_graph_snn"    , "_k", str(k), "_dim", str(dim),                 type_names[type], str(ord), custom, ".png"        ]),
        "img_out_bqm"       : ''.join(["./PlotsOut/"    , str(n), "_graph_snn"    , "_k", str(k), "_dim", str(dim), "_gf", str(gf), type_names[type], str(ord), custom, "_out.png"    ]),
        "img_out_dqm"       : ''.join(["./PlotsOut/"    , str(n), "_dqm_graph_snn", "_k", str(k), "_dim", str(dim), "_g", str(g)  , type_names[type], str(ord), custom, "_out.png"    ]),
        "embedding"         : ''.join(["./Embedding/"   , str(n), "_graph_snn"    , "_k", str(k), "_dim", str(dim),                 type_names[type], str(ord),       , ".json"       ])
    }
    return dirs

def create_graph(dirs):
    G = nx.read_gexf(dirs["graph_in"])
    pos = nx.spring_layout(G)
    return G, pos

def create_graph_csv(dirs):
    input_data = pd.read_csv(dirs["graph_in_csv"], header=0, usecols={1,2,3})
    records = input_data.to_records(index=False)
    result = list(records)
    len(result)
    G = nx.Graph()
    G.add_weighted_edges_from(result)
    pos = nx.spring_layout(G)
    return G, pos

def plot_and_save_graph_in(G, pos, dirs):
    plt.cla()
    nx.draw_networkx_nodes(G, pos, node_size=10, nodelist=G.nodes)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges, style='solid', alpha=0.5, width=1)
    plt.savefig(dirs["img_in"], bbox_inches='tight')
    print("graph saved as: ", dirs["img_in"])

def plot_and_save_graph_out_recur2(G, pos, dirs):
    cut_edges = [(u, v) for u, v in G.edges if list(G.nodes[u].values())[-1]!=list(G.nodes[v].values())[-1]]
    uncut_edges = [(u, v) for u, v in G.edges if list(G.nodes[u].values())[-1]==list(G.nodes[v].values())[-1]]

    len(cut_edges)
    len(uncut_edges)

    # colors = [sum(list(y.values())) for x,y in G.nodes(data=True)]
    colors = [int(list(y.values())[-1]) for x,y in G.nodes(data=True)]

    # ------- plot and & output graph -------
    plt.cla()

    nx.draw_networkx_nodes(G, pos, node_size=10, nodelist=G.nodes,  node_color=colors)
    nx.draw_networkx_edges(G, pos, edgelist=cut_edges, style='dashdot', alpha=0.5, width=1)
    nx.draw_networkx_edges(G, pos, edgelist=uncut_edges, style='solid', width=1)

    plt.savefig(dirs["img_out_bqm"], bbox_inches='tight')

    nx.write_gexf(G, dirs["graph_out_bqm"])

def plot_and_save_graph_out_dqm(G, pos, dirs, sampleset):
    plt.cla()
    nx.draw(G, pos=pos, with_labels=False, node_color=list(sampleset.first.sample.values()), node_size=10, cmap=plt.cm.rainbow)                 
    plt.savefig(dirs["img_out_dqm"], bbox_inches='tight')

    lut = sampleset.first.sample
    nx.set_node_attributes(G, lut, name="label1")

    nx.write_gexf(G, dirs["graph_out_dqm"])

def check_embedding_inspector(G, gamma_factor):
    print("starting")
    name = "for_inspection"

    edges_weights = G.size(weight="weight")
    nodes_len = len(G.nodes)
    gamma = gamma_factor * edges_weights/nodes_len
    print("gama: ", gamma)
    k = 8

    # Initialize our Q matrix
    Q = defaultdict(int)
    # Fill in Q matrix
    for u, v in G.edges:
        Q[(u,u)] += k*G.get_edge_data(u, v)["weight"]
        Q[(v,v)] += k*G.get_edge_data(u, v)["weight"]
        Q[(u,v)] += k *-2*G.get_edge_data(u, v)["weight"]

    for i in G.nodes:
        Q[(i,i)] += gamma*(1-len(G.nodes))

    for i, j in combinations(G.nodes, 2): # do you need this term ???
        Q[(i,j)] += 2*gamma
    
    num_reads = 5
    chain_strength = 4
    print("Looking for a file")
    try:
        a_file = open(dirs["embedding"])
        embedding = json.load(a_file)
        a_file.close()
        print("found saved embedding and started embedding")
        sampler = FixedEmbeddingComposite(DWaveSampler(), embedding)
        print("endded embedding")
    except IOError:
        print("embedding not found")
        return

    print("Sampling")
    response = sampler.sample_qubo(Q, label=name, chain_strength=chain_strength, num_reads=num_reads)    
    dwave.inspector.show(response) # , block='never'

def clustering_recur2(G, iteration, dirs, solver, gamma_factor, color, terminate_on, size_limit):

    name_spec = ''.join([dirs["name"], "_", solver]) 
    
    edges_weights = G.size(weight="weight")
    nodes_len = len(G.nodes)
    gamma = gamma_factor * edges_weights/nodes_len
    print("gama: ", gamma)
    k = 8

    # Initialize our Q matrix
    Q = defaultdict(int)
    # Fill in Q matrix
    for u, v in G.edges:
        Q[(u,u)] += k*G.get_edge_data(u, v)["weight"]
        Q[(v,v)] += k*G.get_edge_data(u, v)["weight"]
        Q[(u,v)] += k *-2*G.get_edge_data(u, v)["weight"]

    for i in G.nodes:
        Q[(i,i)] += gamma*(1-len(G.nodes))

    for i, j in combinations(G.nodes, 2):
        Q[(i,j)] += 2*gamma

    # --------------
    print("... Running on QPU ...")
    
    num_reads = 1000
    chain_strength = 4

    if solver == "hybrid":
        sampler = LeapHybridSampler()
        response = sampler.sample_qubo(Q, label=name_spec)
    elif solver == "fixed_embedding":
        save = False
        try:
            a_file = open(dirs["embedding"])
            embedding = json.load(a_file)
            a_file.close()

            sub_embedding = dict((k, embedding[k]) for k in G.nodes if k in embedding)
            
            sampler = FixedEmbeddingComposite(DWaveSampler(), sub_embedding)
            print("found save embedding")
        except IOError:
            save = True
            print("generate new embedding")
            sampler = LazyFixedEmbeddingComposite(DWaveSampler())

        response = sampler.sample_qubo(Q, label=name_spec, chain_strength=chain_strength, num_reads=num_reads)    
        
        if save:
            embedding = sampler.properties['embedding']
            a_file = open(dirs["embedding"], "w")
            json.dump(embedding, a_file)
            a_file.close()   
    elif solver == "embedding_composite":
        sampler = EmbeddingComposite(DWaveSampler())
        response = sampler.sample_qubo(Q, label=name_spec, chain_strength=chain_strength, num_reads=num_reads)

    # ------- Print results to user -------
    print('-' * 60)
    print('{:>15s}{:>15s}{:^15s}{:^15s}'.format('Set 0','Set 1','Energy','Num. of occurrences'))
    print('-' * 60)

    i=0
    for sample, E, occur in response.data(fields=['sample','energy', "num_occurrences"]):
        # select clusters
        S0 = [k for k,v in sample.items() if v == 0]
        S1 = [k for k,v in sample.items() if v == 1]

        print('{:>15s}{:>15s}{:^15s}{:^15s}'.format(str(S0),str(S1),str(E),str(occur)))
        
        if terminate_on == "conf":
            if i==0:
                size1 = occur
            elif i==1:
                size2 = occur
            elif i==2:
                size3 = occur 
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
    if terminate_on == "min_size":
        if(len(S0)>size_limit and len(S1)>size_limit):
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

            clustering_recur2(G.subgraph(S0), iteration+1, dirs, name, solver, gamma_factor, color+20, terminate_on, size_limit)
            clustering_recur2(G.subgraph(S1), iteration+1, dirs, name, solver, gamma_factor, color+20, terminate_on, size_limit)
    elif terminate_on == "conf":
        print("size1", size1)
        print("size2", size2)
        print("size3", size3)
        confidence = size1/size3
        if confidence > 2 and min(len(S0), len(S1)) > 5:
            clustering_recur2(G.subgraph(S0), iteration+1, dirs, name, solver, gamma_factor, color+20, terminate_on, size_limit)
            clustering_recur2(G.subgraph(S1), iteration+1, dirs, name, solver, gamma_factor, color+20, terminate_on, size_limit)
    elif terminate_on == "once":
        col = random.randint(0, 100)
        for i in S0:
            G.nodes(data=True)[i][label] = col
        
        col = random.randint(120, 220)    
        for i in S1:
            G.nodes(data=True)[i][label] = col

    return

def clustering_discrete(G, num_of_clusters, gamma):
    nodes = G.nodes
    edges = G.edges
    clusters = [i for i  in range(0, num_of_clusters)]

    dqm = dimod.DiscreteQuadraticModel()
    for node in nodes:
        dqm.add_variable(num_of_clusters, label=node)
    
    for node in nodes:
        dqm.set_linear(node, [gamma*(1-len(G.nodes)/num_of_clusters) for cluster in clusters])

    for i, j in combinations(nodes, 2):
        dqm.set_quadratic(i, j, {(cluster, cluster) : 2*gamma for cluster in clusters})

    for u, v in edges:
        dqm.set_quadratic(u, v, {(cluster, cluster) : -2*G.get_edge_data(u, v)["weight"] for cluster in clusters})
        dqm.set_linear(u, [G.get_edge_data(u, v)["weight"] for cluster in clusters])
        dqm.set_linear(v, [G.get_edge_data(u, v)["weight"] for cluster in clusters])

    sampleset = LeapHybridDQMSampler().sample_dqm(dqm, label='DQM - scRAN-seq') 
    print("Energy: {}\nSolution: {}".format(sampleset.first.energy, sampleset.first.sample)) 
    return sampleset

solvers = {
    "h"     : "hybrid",
    "fe"    : "fixed_embedding",
    "ec"    : "embedding_composite"
}
solver = solvers["fe"] # type of used solver

n = 128     # size of the graph
k = 5       # k_nn used for SNN
ord = 15    # maximum order of node degree when "trimmed" mode is enabled
dim = 15    # number of dimensions used for SNN
g_type = 1    #["_", "_trimmed_", "_negedges_", "_trimmed_negedges_"], where "_" -> unaltered SNN output
color = 0   # initial value of clusters coloring fro bqm
gamma_factor = 0.05         # to be used with dqm, weights the clusters' sizes constraint
gamma = 0.005               # to be used with bqm
custom = "_14_03_v5"        # additional metadata for file names
terminate_on = "min_size"   # other options: "conf", "min_size"
size_limit = 10             # may be used in both bqm and dqm // to finish
num_of_clusters = 5         # may be used in both bqm and dqm // to finish

# definition of files locations
dirs = define_dirs(n, k, dim, ord, gamma, gamma_factor, custom, g_type)

G, pos = create_graph(dirs) # create input graph from .gexf file
# G, pos = create_graph_csv(dirs) # create input graph from .csv file

plot_and_save_graph_in(G, pos, dirs)

#  --------- clustering with discrete variables -----------
sampleset = clustering_discrete(G, num_of_clusters, gamma)       
plot_and_save_graph_out_dqm(G, pos, dirs, sampleset)

#  --------- clustering recursively with binary variables -----------
# iteration = 1
# clustering_recur2(G, iteration, dirs, solver, gamma_factor, color, terminate_on, size_limit)
# plot_and_save_graph_out_recur2(G, pos, dirs)

#  --------- check graph embedding in the inspector -----------
# check_embedding_inspector(G, gamma_factor)
