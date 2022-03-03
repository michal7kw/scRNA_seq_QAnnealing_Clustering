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

size = str(128)
graph_in_name = ''.join(["./Datasets/", size, "_check_point_graph_snn.gexf"])
graph_out_name = ''.join(["./Datasets/", size, "_check_point_graph_snn_out.gexf"])
img_in_name = ''.join(["./Output/", size, "_check_point_graph_snn_in_fix.png"])
img_out_name = ''.join(["./Output/", size, "_check_point_graph_snn_out_fix.png"])
embed_name = ''.join(["./Embedding/", size, "_fix_embedding.pkl"])

a_file = open(embed_name, "rb")
embedding = pickle.load(a_file)
print(embedding)

G = nx.read_gexf(graph_in_name)

pos = nx.spring_layout(G)

plt.cla()

nx.draw_networkx_nodes(G, pos, node_size=10, nodelist=G.nodes)
nx.draw_networkx_edges(G, pos, edgelist=G.edges, style='solid', alpha=0.5, width=1)

plt.savefig(img_in_name, bbox_inches='tight')

Q = defaultdict(int)
gamma = 0.01

# Fill in Q matrix
for u, v in G.edges:
    Q[(u,u)] += 1
    Q[(v,v)] += 1
    Q[(u,v)] += -2

for i in G.nodes:
    Q[(i,i)] += gamma*(1-len(G.nodes))

for i, j in combinations(G.nodes, 2):
	Q[(i,j)] += 2*gamma


# ------- Run our QUBO on the QPU -------
print("Running QUBO...")
chain_strength = 4

dwave_sampler = DWaveSampler()
# embedding = find_embedding(Q, dwave_sampler.edgelist)

# a_file = open(embed_name, "wb")
# pickle.dump(embedding, a_file)
# a_file.close()

sampler = FixedEmbeddingComposite(DWaveSampler(), embedding)
response = sampler.sample_qubo(Q, chain_strength=chain_strength, num_reads=100)

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

    filename = "./Output/fix_" + size + "_" + str(i) + ".png"
    plt.savefig(filename, bbox_inches='tight')
    print("\nYour plot is saved to {}".format(filename))
    
    if i == 1:
        nx.write_gexf(G, graph_out_name)

    if i > 3:
        break

dwave.inspector.show(response, block='never')