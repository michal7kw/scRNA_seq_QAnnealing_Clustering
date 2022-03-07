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

def clustering(G, iteration, color):

    # ------- Set up our QUBO dictionary -------
    # Initialize our Q matrix
    Q = defaultdict(int)
    gamma = 10 / len(G.nodes)

    # Fill in Q matrix
    for u, v in G.edges:
        Q[(u,u)] += 2*G.get_edge_data(u, v)["weight"]
        Q[(v,v)] += 2*G.get_edge_data(u, v)["weight"]
        Q[(u,v)] += -4*G.get_edge_data(u, v)["weight"]

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

# ------- import from csv edgelist -------
# input_data = pd.read_csv('./Datasets/edge_list2v500snn_k10_max.csv', header=0, usecols={1,2,3})

# records = input_data.to_records(index=False)
# result = list(records)
# len(result)
# G = nx.Graph()
# G.add_weighted_edges_from(result)

# ------- import from .gexf adjacency matrix -------

size = 128
k = 5
dim = 15

# id_type, type = int(r.id_type)-1, r.type
# n = int(r.n)
# k = int(r.k)
# ord = int(r.ord)
# dim = int(r.dim)

# if id_type==1 or id_type==3:
#   file_name = ''.join(["./graphs_samples/", str(n), "_graph_snn", "_k", str(k), "_dim", str(dim), "_", type[id_type], str(ord), ".gexf"])
# else:
#   file_name = ''.join(["./graphs_samples/", str(n), "_graph_snn", "_k", str(k), "_dim", str(dim), "_", type[id_type], ".gexf"])

# graph_in_name = ''.join(["./Datasets/", str(size), "_graph_snn", "_k", str(k), "_dim", str(dim), ".gexf"])
graph_in_name = ''.join(["./Datasets/","128_graph_snn_k5_dim15_trimmed15.gexf"])

graph_in_name_csv = ''.join(["./Datasets/", str(size), "graph_snn", "_k", str(k), "_dim", str(dim), ".csv"])
graph_out_name = ''.join(["./Datasets/", str(size), "_check_point_graph_snn_fixed", "_k", str(k), "_dim", str(dim), "_out.gexf"])
# img_in_name = ''.join(["./Output/", str(size), "_check_point_graph_snn_in_fixed", "_k", str(k), "_dim", str(dim),".png"])
img_in_name = ''.join(["./Output/","128_graph_snn_k5_dim15_trimmed15_in.png"])
# img_out_name = ''.join(["./Output/", str(size), "_check_point_graph_snn_out_fixed", "_k", str(k), "_dim", str(dim),".png"])
img_out_name = ''.join(["./Output/","128_graph_snn_k5_dim15_trimmed15_out.png"])

G = nx.read_gexf(graph_in_name)
pos = nx.spring_layout(G)

plt.cla()
nx.draw_networkx_nodes(G, pos, node_size=10, nodelist=G.nodes)
nx.draw_networkx_edges(G, pos, edgelist=G.edges, style='solid', alpha=0.5, width=1)

plt.savefig(img_in_name, bbox_inches='tight')


# G = nx.Graph()
# G = nx.read_gexf("./Datasets/final_graph_snn_5.gexf")
# pos = nx.spring_layout(G)

# ------- plot input graph -------
# plt.cla()

# nx.draw_networkx_nodes(G, pos, node_size=10, nodelist=G.nodes)
# nx.draw_networkx_edges(G, pos, edgelist=G.edges, style='solid', alpha=0.5, width=1)

# filename = "./Output/G_snn_in.png"
# plt.savefig(filename, bbox_inches='tight')

# ------- run clustering -------
iteration = 1
clustering(G, iteration, color=0)

cut_edges = [(u, v) for u, v in G.edges if list(G.nodes[u].values())[-1]!=list(G.nodes[v].values())[-1]]
uncut_edges = [(u, v) for u, v in G.edges if list(G.nodes[u].values())[-1]==list(G.nodes[v].values())[-1]]

len(cut_edges)
len(uncut_edges)

# colors = [sum(list(y.values())) for x,y in G.nodes(data=True)]
colors = [list(y.values())[-1] for x,y in G.nodes(data=True)]

# ------- plot and & output graph -------
plt.cla()

nodes = G.nodes()
nx.draw_networkx_nodes(G, pos, node_size=10, nodelist=nodes,  node_color=colors)
nx.draw_networkx_edges(G, pos, edgelist=cut_edges, style='dashdot', alpha=0.5, width=1)
nx.draw_networkx_edges(G, pos, edgelist=uncut_edges, style='solid', width=1)

filename = "G_snn_out.png"
plt.savefig(filename, bbox_inches='tight')

nx.write_gexf(G, "final_graph_snn.gexf")

