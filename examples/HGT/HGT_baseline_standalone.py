import sys
import os
import math

import torch
import torch.nn as nn
import pandas as pd
import nvtx


import dgl

from bench_softlink import (
    # load_data,
    # setup,
    check_equal,
    bench,
    # hetero_dataset,
    init_log,
    empty_cache,
)

from setup_lite_softlink import load_data_as_dgl_graph, hetero_dataset, setup


import dgl.function as fn
from dgl.nn.functional import edge_softmax

from HGT_DGL import HGT_DGLHetero, HGT_DGL_SegmentMM
from HGT_PyG import HGT_PyG, get_ntype

device = setup()


def profile(dataset, feat_dim, out_dim, repeat=1000):
    log = init_log(
        [
            "0-DGL-UDF",
            "1-DGL-slice",
            "2-PyG-slice",
            "3-DGL-bmm",
            "4-PyG-bmm",
            "5-Graphiler",
        ],
        ["time", "mem"],
    )
    print("benchmarking on: " + dataset)
    g_hetero = load_data_as_dgl_graph(dataset)
    g = dgl.to_homogeneous(g_hetero)
    features = torch.rand(
        [sum([g.number_of_nodes(ntype) for ntype in g.ntypes]), feat_dim]
    )
    features = features.to(device)

    @empty_cache
    def run_dgl_slice(g_hetero, features):
        with nvtx.annotate("dgl-slice", color="purple"):
            g_hetero = g_hetero.to(device)
            node_dict = {}
            edge_dict = {}
            for ntype in g_hetero.ntypes:
                node_dict[ntype] = len(node_dict)
            for etype in g_hetero.canonical_etypes:
                edge_dict[etype] = len(edge_dict)
            net_hetero = HGT_DGLHetero(node_dict, edge_dict, feat_dim, out_dim).to(
                device
            )
            net_hetero.eval()
            with torch.no_grad():
                bench(
                    net=net_hetero,
                    net_params=(g_hetero, g_hetero.ndata["h"]),
                    tag="1-DGL-slice",
                    nvprof=False,
                    repeat=repeat,
                    memory=True,
                    log=log,
                )
            del g_hetero, node_dict, edge_dict, net_hetero

    @empty_cache
    def run_dgl_segmentmm(g, features):
        with nvtx.annotate("dgl-segmentmm", color="purple"):
            g = g.to(device)
            # norm = torch.rand(g.num_edges(), 1).to(device)
            net_dgl = HGT_DGL_SegmentMM(
                feat_dim,
                out_dim,
                1,
                int(g.ndata["_TYPE"].max()) + 1,
                int(g.edata["_TYPE"].max()) + 1,
            ).to(device)
            net_dgl.eval()
            with torch.no_grad():
                bench(
                    net=net_dgl,
                    net_params=(g, features, g.ndata["_TYPE"], g.edata["_TYPE"], True),
                    tag="3MK1-DGL-segmentmm",
                    nvprof=False,
                    repeat=repeat,
                    memory=True,
                    log=log,
                )
            del g, net_dgl  # , norm

    @empty_cache
    def run_pyg_slice(g, features):

        with nvtx.annotate("pyg-slice", color="blue"):
            u, v = g.edges()
            adj = torch.stack([u, v]).to(device)
            src_type, dst_type = get_ntype(
                adj, g.edata["_TYPE"], g.ndata["_TYPE"], g.num_rels
            )
            net_pyg_slice = HGT_PyG(
                feat_dim, out_dim, g.num_ntypes, g.num_rels, mode="slice"
            ).to(device)
            net_pyg_slice.eval()
            with torch.no_grad():
                bench(
                    net=net_pyg_slice,
                    net_params=(
                        adj,
                        features,
                        g.edata["_TYPE"],
                        g.ndata["_TYPE"],
                        src_type,
                        dst_type,
                    ),
                    tag="2-PyG-slice",
                    nvprof=False,
                    repeat=repeat,
                    memory=True,
                    log=log,
                )
            del u, v, adj, src_type, dst_type, net_pyg_slice

    @empty_cache
    def run_pyg_bmm(g, features):
        with nvtx.annotate("pyg-bmm", color="red"):
            u, v = g.edges()
            adj = torch.stack([u, v]).to(device)
            src_type, dst_type = get_ntype(
                adj, g.edata["_TYPE"], g.ndata["_TYPE"], g.num_rels
            )
            net_pyg_bmm = HGT_PyG(
                feat_dim, out_dim, g.num_ntypes, g.num_rels, mode="bmm"
            ).to(device)
            net_pyg_bmm.eval()
            with torch.no_grad():
                bench(
                    net=net_pyg_bmm,
                    net_params=(
                        adj,
                        features,
                        g.edata["_TYPE"].to(device),
                        g.ndata["_TYPE"].to(device),
                        src_type,
                        dst_type,
                    ),
                    tag="4-PyG-bmm",
                    nvprof=False,
                    repeat=repeat,
                    memory=True,
                    log=log,
                )
            del u, v, adj, src_type, dst_type, net_pyg_bmm

    # run_dgl_slice(g_hetero, features)
    run_dgl_segmentmm(g, features)
    # run_pyg_bmm(g, features)
    # run_pyg_slice(g, features)

    return log


if __name__ == "__main__":
    # repeat = int(os.environ.get('REPEAT', 50))
    # torch.backends.cudnn.allow_tf32 = False
    # torch.backends.cudnn.enabled = False
    # torch.backends.cuda.matmul.allow_tf32 = False
    repeat = 1
    if len(sys.argv) != 4:
        print("usage: python HGT.py [dataset] [feat_dim] [out_dim]")
        exit()
    if sys.argv[1] == "all":
        log = {}
        for d in hetero_dataset:
            log[d] = profile(d, int(sys.argv[2]), int(sys.argv[3]), repeat)
        pd.DataFrame(log).to_pickle("output/HGT.pkl")
    elif sys.argv[1] == "breakdown":
        raise NotImplementedError("implemented only for graphiler-related routines")
    else:
        profile(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), repeat)
