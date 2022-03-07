import math
import networkx as nx
from collections import defaultdict
from itertools import combinations

import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt

from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite

G = nx.read_gexf("./Datasets/check_point_graph_snn.gexf")

pos = nx.spring_layout(G)

plt.cla()

nx.draw_networkx_nodes(G, pos, node_size=10, nodelist=G.nodes)
nx.draw_networkx_edges(G, pos, edgelist=G.edges, style='solid', alpha=0.5, width=1)

filename = "./Output/check_point_graph_snn_in.png"
plt.savefig(filename, bbox_inches='tight')

Q = defaultdict(int)
gamma = 0.01

for u, v in G.edges:
    Q[(u,u)] += 1
    Q[(v,v)] += 1
    Q[(u,v)] += -2

for i in G.nodes:
    Q[(i,i)] += gamma*(1-len(G.nodes))

for i, j in combinations(G.nodes, 2):
	Q[(i,j)] += 2*gamma


print("Running QUBO...")

chain_strength = 4

sampler = EmbeddingComposite(DWaveSampler())

response = sampler.sample_qubo(Q, chain_strength=chain_strength, num_reads=1000)

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

    filename = "./Output/check_point_graph_snn_out_" + str(i) + ".png"
    plt.savefig(filename, bbox_inches='tight')
    print("\nYour plot is saved to {}".format(filename))
    
    if (i == 1):
        nx.write_gexf(G, "check_point_graph_snn_out.gexf")
    
    if (i > 3):
        break

