# ------ Import necessary packages ----
import networkx as nx
from collections import defaultdict
from itertools import combinations
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite, LazyFixedEmbeddingComposite, FixedEmbeddingComposite
import math
import pandas as pd
import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt
import random
from dwave.system import LeapHybridSampler

def clustering(G, iteration, color):

    # ------- Set up our QUBO dictionary -------
    # Initialize our Q matrix
    Q = defaultdict(int)
    gamma = 10 / len(G.nodes)

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

    # ------ Hybrid Sampler ------
    sampler = LeapHybridSampler(time_limit=3)
    response = sampler.sample_qubo(Q)

    # ------- Print results to user -------
    print('-' * 60)
    print('{:>15s}{:>15s}{:^15s}{:^15s}'.format('Set 0','Set 1','Energy','Cut Size'))
    print('-' * 60)

    i=0
    for sample, E in response.data(fields=['sample','energy']):
        i = i + 1
        # select clusters
        S0 = [k for k,v in sample.items() if v == 0]
        S1 = [k for k,v in sample.items() if v == 1]
        
        print('{:>15s}{:>15s}{:^15s}{:^15s}'.format(str(S0),str(S1),str(E),str(int(-1*E))))
    
        if (i > 5):
            break

    
    label = "label" + str(iteration)
    lut = response.first.sample

    # Interpret best result in terms of nodes and edges
    S0 = [node for node in G.nodes if not lut[node]]
    S1 = [node for node in G.nodes if lut[node]]

    print("S0 length: ", len(S0))
    print("S1 length: ", len(S1))
    if(len(S0)>40 and len(S1)>40):
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

        clustering(G.subgraph(S0), iteration+1, color+20)
        clustering(G.subgraph(S1), iteration+1, color+20)
        
    return


input_data = pd.read_csv('./Datasets/edge_list2v500knn.csv', header=0, usecols={1,2})

records = input_data.to_records(index=False)
result = list(records)

# ------- Set up our graph -------
# Create empty graph
G = nx.Graph()

G.add_edges_from(result)
pos = nx.spring_layout(G)
len(G.nodes)

plt.cla()

nodes = ()
nx.draw_networkx_nodes(G, pos, node_size=10, nodelist=G.nodes)
nx.draw_networkx_edges(G, pos, edgelist=G.edges, style='solid', alpha=0.5, width=1)

filename = "./Output/G_knn_recur_in.png"
plt.savefig(filename, bbox_inches='tight')

iteration = 1
clustering(G, iteration, color=0)

# sum(list(G.nodes(data=True)[0].values()))

cut_edges = [(u, v) for u, v in G.edges if list(G.nodes[u].values())[-1]!=list(G.nodes[v].values())[-1]]
uncut_edges = [(u, v) for u, v in G.edges if list(G.nodes[u].values())[-1]==list(G.nodes[v].values())[-1]]

len(cut_edges)
len(uncut_edges)

# colors = [sum(list(y.values())) for x,y in G.nodes(data=True)]
colors = [list(y.values())[-1] for x,y in G.nodes(data=True)]

plt.cla()

nodes = G.nodes()
nx.draw_networkx_nodes(G, pos, node_size=10, nodelist=nodes,  node_color=colors)
nx.draw_networkx_edges(G, pos, edgelist=cut_edges, style='dashdot', alpha=0.5, width=1)
nx.draw_networkx_edges(G, pos, edgelist=uncut_edges, style='solid', width=1)

filename = "./Output/G_knn_recur_out.png"
plt.savefig(filename, bbox_inches='tight')

nx.write_gexf(G, "final_graph_knn.gexf")

