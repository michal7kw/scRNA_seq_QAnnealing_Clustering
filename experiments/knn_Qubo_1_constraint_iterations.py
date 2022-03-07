# ------ Import necessary packages ----
import networkx as nx
from collections import defaultdict
from itertools import combinations
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite, LazyFixedEmbeddingComposite
import math
import pandas as pd
import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt

def clustering(prev_result, iteration):

    # ------- Set up our graph -------
    # Create empty graph
    G = nx.Graph()

    G.add_edges_from(prev_result)
    pos = nx.spring_layout(G)

    # ------- Set up our QUBO dictionary -------
    # Initialize our Q matrix
    Q = defaultdict(int)
    gamma = 0.05

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
    # Set chain strength
    # chain_strength = gamma*len(G.nodes)
    chain_strength = 4

    # Run the QUBO on the solver from your config file
    sampler = EmbeddingComposite(DWaveSampler())
    # sampler = LazyFixedEmbeddingComposite(DWaveSampler())

    response = sampler.sample_qubo(Q,
                                chain_strength=chain_strength,
                                num_reads=50,
                                label='Example - Graph Partitioning')

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
        # check the class object
        if i==1:
            print("sample object format: ", sample)
            print("sample.items() format: ", sample.items())
        # print clusters data
        print('{:>15s}{:>15s}{:^15s}{:^15s}'.format(str(S0),str(S1),str(E),str(int(-1*E))))
        
        S0v2 = [node for node in G.nodes if sample[node]==0]
        S1v2 = [node for node in G.nodes if sample[node]==1]
        if i==1:
            print("### start: check nodes ###")
            print('{:>15s}{:>15s}'.format(str(S0),str(S1)))
            print('{:>15s}{:>15s}'.format(str(S0v2),str(S1v2)))
            print("### end: check nodes ###")
        cut_edges = [(u, v) for u, v in G.edges if sample[u]!=sample[v]]
        uncut_edges = [(u, v) for u, v in G.edges if sample[u]==sample[v]]

        plt.cla()
        # pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G, pos, node_size=30, nodelist=S0, node_color='r')
        nx.draw_networkx_nodes(G, pos, node_size=30, nodelist=S1, node_color='c')
        nx.draw_networkx_edges(G, pos, edgelist=cut_edges, style='dashdot', alpha=0.5, width=3)
        nx.draw_networkx_edges(G, pos, edgelist=uncut_edges, style='solid', width=3)
        nx.draw_networkx_labels(G, pos)

        filename = "clusters_constrained_1_recursion" + str(i) + str(iteration) + ".png"
        plt.savefig(filename, bbox_inches='tight')
        print("\nYour plot is saved to {}".format(filename))
        if (i > 3):
            break

    

    label = "label" + str(iteration)
    lut = response.first.sample

    # Interpret best result in terms of nodes and edges
    S0 = [node for node in G.nodes if not lut[node]]
    S1 = [node for node in G.nodes if lut[node]]
    cut_edges = [(u, v) for u, v in G.edges if lut[u]!=lut[v]]
    uncut_edges = [(u, v) for u, v in G.edges if lut[u]==lut[v]]

    # Assign nodes' labels
    for i in S0:
        G.nodes(data=True)[i][label] = iteration
    for i in S1:
        G.nodes(data=True)[i][label] = iteration+1

    # write to the graph file
    file_name = "clustring_" + str(iteration) + ".gexf"
    nx.write_gexf(G, file_name)

    clust1 = [(u, v) for u, v in uncut_edges if lut[u]==0]
    clust2 = [(u, v) for u, v in uncut_edges if lut[u]==1]
    print("cluster_1 length: ", len(clust1))
    print("cluster_2 length: ", len(clust2))
    return clust1, clust2, S0, S1


input_data = pd.read_csv('edge_list2.csv', header=0, usecols={1,2})

records = input_data.to_records(index=False)
result = list(records)

iteration = 1
clusters = clustering(result, iteration)

cluster1, cluster2, S0, S1 = clusters

plt.cla()
iteration = iteration + 1
clusters = clustering(cluster2, iteration)

iteration = iteration + 1
clusters = clustering(cluster1, iteration)

G = nx.Graph()
G.add_edges_from(result)

G_in = nx.read_gexf("clustring_1.gexf")
list(G_in)
list(G_in.nodes(data=True))
G_in.remove_nodes_from([n for n,y in G_in.nodes(data=True) if y['label1'] != 1])

len(G_in.edges())
len(G.edges())
len(G_in.nodes())
len(G.nodes())

plt.cla()
pos = nx.spring_layout(G_in)
nx.draw_networkx_nodes(G_in, pos, node_size=30, node_color='r')
nx.draw_networkx_edges(G_in, pos, style='solid', width=3)
nx.draw_networkx_labels(G_in, pos)

filename = "G_in.png"
plt.savefig(filename, bbox_inches='tight')