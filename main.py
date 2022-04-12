# ------ Import necessary packages ----
from cProfile import label
from email import iterators
import math
import random
from tkinter.tix import Tree
import types
import itertools

from matplotlib.colors import same_color
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
from itertools import combinations, count
import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt

import dimod
import hybrid 
# import dwavebinarycsp
import dwave.inspector
import dwave_networkx as dnx
# from minorminer import find_embedding
# from dwave.embedding import embed_ising
from dimod import BinaryQuadraticModel
from dwave.system.samplers import DWaveSampler
from dwave.system import LeapHybridSampler, LeapHybridDQMSampler, LeapHybridCQMSampler
from dwave.system.composites import EmbeddingComposite, LazyFixedEmbeddingComposite, FixedEmbeddingComposite
from dwave.cloud.client import Client

import json
import pickle

from Python_Functions.create_graphs import *
from Python_Functions.plot_and_save import *
from Python_Functions.other_tools import *
from Python_Functions.QA_subsampling import *
from Python_Functions.BQM_clustering import *
from Python_Functions.DQM_clustering import *
from Python_Functions.CQM_clustering import *

def define_dirs(n, k, dim, ord, g, gf, custom,type):
    # n-size, k-k_nn, dim-dimensions, ord-max_degree, g-gamma, custom-for one's needs, type-type of graph
    type_names = ["_", "_trimmed_", "_negedges_", "_trimmed_negedges_"]
    g = str(g).replace( ".", "")
    gf = str(gf).replace( ".", "")

    dirs = {
        "name"              : ''.join([                   str(n), "_graph_snn"     , "_k", str(k), "_dim", str(dim),                 type_names[type], str(ord)]                       ),
        
        "graph_in"          : ''.join(["./DatasetsIn/"  , str(n), "_graph_snn"     , "_k", str(k), "_dim", str(dim),                 type_names[type], str(ord),         ".gexf"       ]),
        "graph_in_csv"      : ''.join(["./DatasetsIn/"  , str(n), "_graph_snn"     , "_k", str(k), "_dim", str(dim),                 type_names[type], str(ord),         ".csv"        ]),
        "graph_in_pru"      : ''.join(["./DatasetsIn/"  , str(n), "_pru_graph_snn" , "_k", str(k), "_dim", str(dim),                 type_names[type], str(ord), custom, ".gexf"       ]),
        
        "graph_out_bqm"     : ''.join(["./DatasetsOut/" , str(n), "_graph_snn"     , "_k", str(k), "_dim", str(dim), "_gf", str(gf), type_names[type], str(ord), custom, "_out.gexf"   ]),
        "graph_out_dqm"     : ''.join(["./DatasetsOut/" , str(n), "_dqm_graph_snn" , "_k", str(k), "_dim", str(dim), "_g", str(g)  , type_names[type], str(ord), custom, ".gexf"       ]),
        "graph_out_cqm"     : ''.join(["./DatasetsOut/" , str(n), "_cqm_graph_snn" , "_k", str(k), "_dim", str(dim), "_g", str(g)  , type_names[type], str(ord), custom, ".gexf"       ]),
        "graph_out_pru1"    : ''.join(["./DatasetsOut/" , str(n), "_pru_graph_snn" , "_k", str(k), "_dim", str(dim),                 type_names[type], str(ord), custom, ".gexf"       ]),
        "graph_out_pru2"    : ''.join(["./DatasetsOut/" , str(n), "_pru_graph_snn" , "_k", str(k), "_dim", str(dim),                 type_names[type], str(ord), custom, "2.gexf"      ]),

        "img_in"            : ''.join(["./PlotsIn/"     , str(n), "_graph_snn"     , "_k", str(k), "_dim", str(dim),                 type_names[type], str(ord), custom, ".png"        ]),
        "img_out_bqm"       : ''.join(["./PlotsOut/"    , str(n), "_bqm_graph_snn" , "_k", str(k), "_dim", str(dim), "_gf", str(gf), type_names[type], str(ord), custom, "_out.png"    ]),
        "img_out_dqm"       : ''.join(["./PlotsOut/"    , str(n), "_dqm_graph_snn" , "_k", str(k), "_dim", str(dim), "_g", str(g)  , type_names[type], str(ord), custom, "_out.png"    ]),
        "img_out_cqm"       : ''.join(["./PlotsOut/"    , str(n), "_cqm_graph_snn" , "_k", str(k), "_dim", str(dim), "_g", str(g)  , type_names[type], str(ord), custom, "_out.png"    ]),
        "img_out_p1"        : ''.join(["./PlotsOut/"    , str(n), "_pru_graph_snn" , "_k", str(k), "_dim", str(dim),                 type_names[type], str(ord), custom, "_out1.png"   ]),
        "img_out_p2"        : ''.join(["./PlotsOut/"    , str(n), "_pru_graph_snn" , "_k", str(k), "_dim", str(dim),                 type_names[type], str(ord), custom, "_out2.png"   ]),
        "img_out_p3"        : ''.join(["./PlotsOut/"    , str(n), "_pru_graph_snn" , "_k", str(k), "_dim", str(dim),                 type_names[type], str(ord), custom, "_out3.png"   ]),
        
        "embedding"         : ''.join(["./Embedding/"   , str(n), "_graph_snn"     , "_k", str(k), "_dim", str(dim),                 type_names[type], str(ord),         ".json"       ]),
        "embedding_pru"     : ''.join(["./Embedding/"   , str(n), "_pru_graph_snn" , "_k", str(k), "_dim", str(dim),                 type_names[type], str(ord),         ".json"       ])
    }
    return dirs
  
solvers = {
    "h"     : "hybrid",
    "fe"    : "fixed_embedding",
    "ec"    : "embedding_composite"
}
solver = solvers["h"] # type of used solver

n = 942     # size of the graph
k = 10       # k_nn used for SNN
ord = 15    # maximum order of node degree when "trimmed" mode is enabled
dim = 30    # number of dimensions used for SNN
g_type = 1  # ["_", "_trimmed_", "_negedges_", "_trimmed_negedges_"], where "_" -> unaltered SNN output
color = 0   # initial value of clusters coloring fro bqm
gamma_factor = 0.05         # to be used with dqm, weights the clusters' sizes constraint
gamma = 0.005               # to be used with bqm
custom = ""                 # additional metadata for file names
terminate_on = "min_size"   # other options: "conf", "min_size"
size_limit = 20             # may be used in both bqm and dqm // to finish
num_of_clusters = 5         # may be used in both bqm and dqm // to finish
iter_limit = 3              # limit of iteration 

# define local directories
dirs = define_dirs(n, k, dim, ord, gamma, gamma_factor, custom, g_type)

# --------- import graph custom name ---------
graph_name="kidney/942pru_graph_snn_k10_dim30_trimmed_15.gexf"
graph_name="kidney/942pru_graph_snn_k10_dim30_trimmed_15enh.gexf"
graph_name="kidney/1000_graph_snn_k10_dim30_trimmed_15_selected_a_2_snn_enh2.gexf"
graph_name="kidney/1000_graph_snn_k10_dim30_trimmed_15_selected_a_2_snn_enh.gexf"
G, pos = create_graph("./DatasetsIn/" + graph_name)

# --------- import graph automatic name --------
G, pos = create_graph(dirs["graph_in"])

# --------- save input graph ---------
plot_and_save_graph_in(G, pos, dirs)

# --------- look for disconnected components ---------
G, S, lengths = disconnected_components(G)
print(lengths)


#  --------- Graph Subsampling ---------
response = graph_subsampling(G, 7, solver)
# S = graph_subsampling_2(G, 10)
# dwave.inspector.show(response)
plot_and_save_graph_out_mvc(G, pos, dirs)
H = prune_graph(G, pos, dirs)


# --------- DQM -----------
sampleset_dqm = clustering_dqm(G, num_of_clusters, gamma)       
plot_and_save_graph_out_dqm(G, pos, dirs, sampleset_dqm)

# --------- CQM ---------
sampleset_cqm = clustering_cqm(G, num_of_clusters)
plot_and_save_graph_out_cqm(G, pos, dirs, sampleset_cqm, num_of_clusters)

# --------- CQM 2 ---------
sampleset_cqm = clustering_cqm_2(G, num_of_clusters)
pos = nx.spring_layout(G)
plot_and_save_graph_out_cqm_2(G, pos, dirs, sampleset_cqm, num_of_clusters)

#  --------- BQM recursive -----------
iteration = 1
clustering_bqm(G, iteration, dirs, solver, gamma_factor, color, terminate_on, size_limit, iter_limit)
plot_and_save_graph_out_bqm(G, pos, dirs)

#  --------- BQM_2 recursive, lessened constraints  -----------
iteration = 1
response = clustering_bqm_2(G, iteration, dirs, solver, 0.0001, color, terminate_on, size_limit, 0, 100)
plot_and_save_graph_out_bqm(G, pos, dirs)

#  --------- BQM_3 recursive, lessened constraints-----------
iteration = 1
clustering_bqm_3(G, iteration, dirs, solver, gamma_factor, color, terminate_on, size_limit)
plot_and_save_graph_out_bqm(G, pos, dirs)


#  --------- Check graph embedding in the inspector -----------
check_embedding_inspector(G, gamma_factor)


#  --------- Retrive response -----------
sampleset = retrive_response(problem_id="555e72bf-f014-463d-977c-8bf071380f30", token="DEV-96c7ac68f866387f382beade0d34ca0640a19935")
dwave.inspector.show(sampleset)
print(sampleset)
plot_and_save_graph_out_cqm_multi(G, pos, dirs, sampleset, num_of_clusters, 16)




