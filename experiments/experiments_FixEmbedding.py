# ------ Import necessary packages ----
import math
import random
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

size = 128
k = 5
dim = 15
# graph_in_name = ''.join(["./Datasets/", str(size), "_check_point_graph_snn.gexf"])
graph_in_name = ''.join(["./Datasets/", str(size), "_check_point_graph_snn", "_k", str(k), "_dim", str(dim), ".gexf"])
# graph_in_name = ''.join(["./Datasets/","8trimed_128_check_point_graph_snn_k5_dim15.gexf"])

graph_in_name_csv = ''.join(["./Datasets/", str(size), "graph_snn", "_k", str(k), "_dim", str(dim), ".csv"])
# graph_out_name = ''.join(["./Datasets/", str(size), "_check_point_graph_snn_out.gexf"])
graph_out_name = ''.join(["./Datasets/", str(size), "_check_point_graph_snn_fixed", "_k", str(k), "_dim", str(dim), "_out.gexf"])
# img_in_name = ''.join(["./Output/", str(size), "_check_point_graph_snn_in_fix.png"])
img_in_name = ''.join(["./Output/", str(size), "_check_point_graph_snn_in_fixed", "_k", str(k), "_dim", str(dim),".png"])
# img_out_name = ''.join(["./Output/", str(size), "_check_point_graph_snn_out_fix.png"])
img_out_name = ''.join(["./Output/", str(size), "_check_point_graph_snn_out_fixed", "_k", str(k), "_dim", str(dim),".png"])
# embed_name = ''.join(["./Embedding/", str(size), "_fix_embedding.pkl"])
embed_name = ''.join(["./Embedding/", str(size), "_fix_embedding", "_k", str(k), "_dim", str(dim), ".pkl"])
embed_name = ''.join(["./Embedding/", str(size), "_fix_embedding_PC", "_k", str(k), "_dim", str(dim), ".pkl"])

a_file = open(embed_name, "rb")
embedding = pickle.load(a_file)
print(embedding)


# ------- import from csv edgelist -------
# input_data = pd.read_csv(graph_in_name_csv, header=0, usecols={1,2,3})

# records = input_data.to_records(index=False)
# result = list(records)
# len(result)
# G = nx.Graph()
# G.add_weighted_edges_from(result)
# pos = nx.spring_layout(G)

# ------- import from .gexf adjacency matrix -------
G = nx.read_gexf(graph_in_name)
pos = nx.spring_layout(G)

# ------- Check Graph features -------
len(G.nodes())
len(G.edges())
print(sum([val for (node, val) in G.degree()]))
print([val for (node, val) in G.degree()])
print(len(G.edges())/len(G.nodes()))

plt.cla()

nx.draw_networkx_nodes(G, pos, node_size=10, nodelist=G.nodes)
nx.draw_networkx_edges(G, pos, edgelist=G.edges, style='solid', alpha=0.5, width=1)

plt.savefig(img_in_name, bbox_inches='tight')

Q = defaultdict(int)
gamma = 1 / len(G.edges)

# Fill in Q matrix
for u, v in G.edges:
    Q[(u,u)] += 8*G.get_edge_data(u, v)["weight"]
    Q[(v,v)] += 8*G.get_edge_data(u, v)["weight"]
    Q[(u,v)] += -16*G.get_edge_data(u, v)["weight"]

for i in G.nodes:
    Q[(i,i)] += gamma*(1-len(G.nodes))

for i, j in combinations(G.nodes, 2):
    Q[(i,j)] += 2*gamma


# ------- Run our QUBO on the QPU -------
print("Running QUBO...")
chain_strength = 4

# dwave_sampler = DWaveSampler()
# embedding = find_embedding(Q, dwave_sampler.edgelist)
# sampler = FixedEmbeddingComposite(DWaveSampler(), embedding)

sampler = LazyFixedEmbeddingComposite(DWaveSampler())

response = sampler.sample_qubo(Q, chain_strength=chain_strength, num_reads=1000)

# embedding = sampler.properties['embedding']

# a_file = open(embed_name, "wb")
# pickle.dump(embedding, a_file)
# a_file.close()

# sampler = EmbeddingComposite(DWaveSampler())


# ------- Print results to user -------
print('-' * 60)
print('{:>15s}{:>15s}{:^15s}{:^15s}'.format('Set 0','Set 1','Energy','Cut Size'))
print('-' * 60)

i=0
for sample, E in response.data(fields=['sample','energy']):
    i += 1
    S0 = [k for k,v in sample.items() if v == 0]
    S1 = [k for k,v in sample.items() if v == 1]
    print('{:>15s}{:>15s}{:^15s}{:^15s}'.format(str(S0),str(S1),str(E),str(int(-1*E))))
    
    S0 = [node for node in G.nodes if sample[node]==0]
    S1 = [node for node in G.nodes if sample[node]==1]
    cut_edges = [(u, v) for u, v in G.edges if sample[u]!=sample[v]]
    uncut_edges = [(u, v) for u, v in G.edges if sample[u]==sample[v]]

    plt.cla()
    nx.draw_networkx_nodes(G, pos, node_size=10, nodelist=S0, node_color='r')
    nx.draw_networkx_nodes(G, pos, node_size=10, nodelist=S1, node_color='c')
    nx.draw_networkx_edges(G, pos, edgelist=cut_edges, style='dashdot', alpha=0.5, width=3)
    nx.draw_networkx_edges(G, pos, edgelist=uncut_edges, style='solid', width=1)

    filename = "./Output/2fix_" + str(size) + "_" + str(i) + ".png"
    plt.savefig(filename, bbox_inches='tight')
    print("\nYour plot is saved to {}".format(filename))
    
    if i == 1:
        nx.write_gexf(G, graph_out_name)

    if i > 3:
        break

dwave.inspector.show(response, block='never')