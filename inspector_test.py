import dimod
import dwave.inspector
from dwave.system import DWaveSampler, EmbeddingComposite
from dwave.system.composites import LazyFixedEmbeddingComposite, FixedEmbeddingComposite


# ------ Import necessary packages ----
import networkx as nx
from collections import defaultdict
from itertools import combinations
import math
import pandas as pd

import json
import pickle

# a_file = open("data.pkl", "rb")
# output = pickle.load(a_file)
# print(output)
# embedding = output


# ------- Set up graph -------
G = nx.Graph()

input_data = pd.read_csv('./Datasets/edge_list2.csv', header=0, usecols={1,2})

records = input_data.to_records(index=False)
result = list(records)

G.add_edges_from(result)
pos = nx.spring_layout(G)

len(G.nodes)
# print("Graph on {} nodes created with {} out of {} possible edges.".format(len(G.nodes), len(G.edges), len(G.nodes) * (len(G.nodes)-1) / 2))

# ------- Set up our QUBO dictionary -------
# Initialize our Q matrix
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
# Set chain strength
chain_strength = 4

# Run the QUBO on the solver from your config file
# sampler = EmbeddingComposite(DWaveSampler())
# sampler = LazyFixedEmbeddingComposite(DWaveSampler(solver=dict(qpu=True)))

model = dimod.BinaryQuadraticModel.from_qubo(Q)

# s = json.dumps(model.to_serializable())

from minorminer import find_embedding
# dwave_sampler = DWaveSampler()
embedding = find_embedding(Q, dwave_sampler.edgelist)

# sampler = LazyFixedEmbeddingComposite(DWaveSampler())
sampler = FixedEmbeddingComposite(DWaveSampler(), embedding)
response = sampler.sample_qubo(Q, chain_strength=chain_strength, num_reads=50)
# response = sampler.sample(model, chain_strength=chain_strength, num_reads=50)

# response = sampler.sample_qubo(Q,
#                                chain_strength=chain_strength,
#                                num_reads=50)

# print(sampler.properties['embedding'])

for smpl, energy in response.data(['sample', 'energy']):
    print(smpl, energy)

# # Define problem
# bqm = dimod.BQM.from_ising({}, {'ab': 1, 'bc': 1, 'ca': 1})

# # Get sampler
# sampler = EmbeddingComposite(DWaveSampler(solver=dict(qpu=True)))

# # Sample with low chain strength
# sampleset = sampler.sample(bqm, num_reads=1000, chain_strength=0.1)

dwave.inspector.show(response) #, block='never'
# dwave.inspector.show(response)

# dictionary_data = embedding
# a_file = open("data.pkl", "wb")
# pickle.dump(dictionary_data, a_file)
# a_file.close()
