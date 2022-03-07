# ------ Import necessary packages ----
import networkx as nx
from collections import defaultdict
from itertools import combinations
import math
import pandas as pd
import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt
import random
from dwave.system import LeapHybridSampler

def clustering(G, iteration, name, color):

    # ------- Set up our QUBO dictionary -------
    # Initialize our Q matrix
    # Q = defaultdict(int)
    # gamma = 10 / len(G.nodes)

    # Fill in Q matrix
    # for u, v in G.edges:
    #     Q[(u,u)] += 2*G.get_edge_data(u, v)["weight"]
    #     Q[(v,v)] += 2*G.get_edge_data(u, v)["weight"]
    #     Q[(u,v)] += -4*G.get_edge_data(u, v)["weight"]

    # for i in G.nodes:
    #     Q[(i,i)] += gamma*(1-len(G.nodes))

    # for i, j in combinations(G.nodes, 2):
    #     Q[(i,j)] += 2*gamma

    Q = defaultdict(int)
    gamma = 100 / len(G.edges)

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
    
    # ------ Hybrid Sampler ------
    sampler = LeapHybridSampler()
    response = sampler.sample_qubo(Q, label=name)

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

    return

# G = nx.read_gexf("./Datasets/128_graph_snn_k5_dim15_trimmed15.gexf")
# name = "v2_128_graph_snn_k5_dim15_trimmed15"
# name_g = "./Output/v2_128_graph_snn_k5_dim15_trimmed15.gexf"
# name_p_in = "v2_128_graph_snn_k5_dim15_trimmed15.png"
# name_p_out = "v2_128_graph_snn_k5_dim15_trimmed15_out.png"

G = nx.read_gexf("./Datasets/128_graph_snn_k5_dim15_onlySNN.gexf")
name = "v2_128_graph_snn_k5_dim15_onlySNN"
name_g = "./Output/v2_128_graph_snn_k5_dim15_onlySNN.gexf"
name_p_in = "v2_128_graph_snn_k5_dim15_onlySNN.png"
name_p_out = "v2_128_graph_snn_k5_dim15_onlySNN_out.png"
pos = nx.spring_layout(G)

plt.cla()

nx.draw_networkx_nodes(G, pos, node_size=10, nodelist=G.nodes)
nx.draw_networkx_edges(G, pos, edgelist=G.edges, style='solid', alpha=0.5, width=1)

filename = "./Output/G_experiment_in.png"
plt.savefig(name_p_in, bbox_inches='tight')

color=0
iteration = 1
clustering(G, iteration, name , color)

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

filename = "G_experiment_out.png"
plt.savefig(name_p_out, bbox_inches='tight')

nx.write_gexf(G, name_g)

