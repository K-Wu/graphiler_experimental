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
    g = load_data_as_dgl_graph(dataset)
    g = dgl.to_homogeneous(g)
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
            net_pyg = RGAT_PyG(
                in_dim=feat_dim, out_dim=out_dim, num_rels=len(g.canonical_etypes)
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
