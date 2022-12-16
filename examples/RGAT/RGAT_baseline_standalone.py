# This is a non-graphiler standalone script to evaluate the performance of RGAT PyG implementation since the default python packages version demanded by graphiler does not allow installation of the latest PyG which provides the RGATConv module.
# the RGATConv is from torch_geometric/nn/conv/rgat_conv.py and can be copied from latest PyG in a standalone manner but it seems the message passing API is different and therefore not compatible.
from bench_softlink import (
    # load_data,
    # setup,
    check_equal,
    bench,
    # hetero_dataset,
    init_log,
    empty_cache,
)
from torch_sparse import SparseTensor
from RGAT_PyG import RGAT_PyG
from RGAT_DGL import DGL_RGAT_Hetero

import dgl
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import nvtx


from setup_lite_softlink import load_data_as_dgl_graph, hetero_dataset, setup

device = setup()


def profile(dataset, feat_dim, out_dim, repeat=1000):
    log = init_log(
        ["0-DGL-UDF", "1-DGL-primitives", "2-PyG-primitives", "3-Graphiler"],
        ["time", "mem"],
    )
    print("benchmarking on: " + dataset)
    # g, features = load_data(dataset, feat_dim, prepare=False)
    g_hetero = load_data_as_dgl_graph(dataset)
    g = dgl.to_homogeneous(g_hetero)
    features = torch.rand(
        [sum([g.number_of_nodes(ntype) for ntype in g.ntypes]), feat_dim]
    )

    features = features.to(device)

    @empty_cache
    def run_pyg_slice(g, features):
        with nvtx.annotate("pyg", color="blue"):

            g = g.to(device)
            edge_type = g.edata["_TYPE"].to(device)
            u, v = g.edges()
            adj = SparseTensor(
                row=u, col=v, sparse_sizes=(g.num_src_nodes(), g.num_dst_nodes())
            ).to(device)
            # print(len(g.canonical_etypes))
            net_pyg = RGAT_PyG(
                in_dim=feat_dim, out_dim=out_dim, num_rels=int(edge_type.max() + 1)
            ).to(device)
            net_pyg.eval()
            # print type of g.edata['_ID']
            # print(g.edata['_ID'])
            # g.edata['_ID']
            with torch.no_grad():
                bench(
                    net=net_pyg,
                    net_params=(adj, features, edge_type),
                    tag="2-PyG-primitives",
                    nvprof=False,
                    repeat=repeat,
                    memory=True,
                    log=log,
                )
            del edge_type, u, v, adj, net_pyg

    @empty_cache
    def run_dgl_hetero(g, features):
        with nvtx.annotate("dgl", color="purple"):
            g = g.to(device)
            net_dgl = DGL_RGAT_Hetero(
                g=g,
                h_dim=feat_dim,
                out_dim=out_dim,
                n_heads=1,
                num_hidden_layers=0,  # , num_rels = g.num_rels
            ).to(device)
            net_dgl.eval()
            with torch.no_grad():
                bench(
                    net=net_dgl,
                    net_params=(g, g.ndata["h"]),
                    tag="1-DGL-primitives",
                    nvprof=False,
                    repeat=repeat,
                    memory=True,
                    log=log,
                )
            del g, net_dgl

    run_dgl_hetero(g_hetero, features)
    run_pyg_slice(g, features)


if __name__ == "__main__":
    repeat = int(os.environ.get("REPEAT", 50))
    if len(sys.argv) != 4:
        print("usage: python RGATSingleLayer.py [dataset] [feat_dim] [out_dim]")
        exit()
    if sys.argv[1] == "all":
        log = {}
        for d in hetero_dataset:
            log[d] = profile(d, int(sys.argv[2]), int(sys.argv[3]), repeat)
        pd.DataFrame(log).to_pickle("output/RGAT.pkl")
    elif sys.argv[1] == "breakdown":
        log = {}
        for d in hetero_dataset:
            log[d] = breakdown(d, int(sys.argv[2]), int(sys.argv[3]), repeat)
        pd.DataFrame(log).to_pickle("output/RGAT_breakdown.pkl")
    else:
        profile(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), repeat)
