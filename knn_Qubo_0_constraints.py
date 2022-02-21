# ------ Import necessary packages ----
import networkx as nx
from collections import defaultdict
from itertools import combinations
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
import math
import pandas as pd
import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt

# ------- Set up our graph -------
# Create empty graph
G = nx.Graph()

input_data = pd.read_csv('edge_list2.csv', header=0, usecols={1,2})

records = input_data.to_records(index=False)
result = list(records)

G.add_edges_from(result)
pos = nx.spring_layout(G)

# ------- Set up our QUBO dictionary -------
# Initialize our Q matrix
Q = defaultdict(int)

# Update Q matrix for every edge in the graph
for i, j in G.edges:
    Q[(i,i)]+= 1
    Q[(j,j)]+= 1
    Q[(i,j)]+= -2

# ------- Run our QUBO on the QPU -------
# Set up QPU parameters
chainstrength = 8
numruns = 50

# Run the QUBO on the solver from your config file
sampler = EmbeddingComposite(DWaveSampler())
response = sampler.sample_qubo(Q,
                               chain_strength=chainstrength,
                               num_reads=numruns,
                               label='Example - Maximum Cut')

# ------- Print results to user -------
print('-' * 60)
print('{:>15s}{:>15s}{:^15s}{:^15s}'.format('Set 0','Set 1','Energy','Cut Size'))
print('-' * 60)

i=0
for sample, E in response.data(fields=['sample','energy']):
    i = i + 1
    S0 = [k for k,v in sample.items() if v == 0]
    S1 = [k for k,v in sample.items() if v == 1]
    print('{:>15s}{:>15s}{:^15s}{:^15s}'.format(str(S0),str(S1),str(E),str(int(-1*E))))
    
    S0 = [node for node in G.nodes if sample[node]==0]
    S1 = [node for node in G.nodes if sample[node]==1]
    cut_edges = [(u, v) for u, v in G.edges if sample[u]!=sample[v]]
    uncut_edges = [(u, v) for u, v in G.edges if sample[u]==sample[v]]

    plt.cla()
    # pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=30, nodelist=S0, node_color='r')
    nx.draw_networkx_nodes(G, pos, node_size=30, nodelist=S1, node_color='c')
    nx.draw_networkx_edges(G, pos, edgelist=cut_edges, style='dashdot', alpha=0.5, width=3)
    nx.draw_networkx_edges(G, pos, edgelist=uncut_edges, style='solid', width=3)
    nx.draw_networkx_labels(G, pos)

    filename = "clusters_" + str(i) + ".png"
    plt.savefig(filename, bbox_inches='tight')
    print("\nYour plot is saved to {}".format(filename))
    if (i > 3):
        break

