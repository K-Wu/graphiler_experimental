import sys
import os
from timeit import repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import nvtx

# from graphiler import EdgeBatchDummy, NodeBatchDummy, mpdfg_builder, update_all
# from graphiler.utils import (
#     load_data,
#     setup,
#     check_equal,
#     bench,
#     hetero_dataset,
#     DEFAULT_DIM,
#     init_log,
#     empty_cache,
# )

import contextlib

import dgl

from bench_softlink import (
    # load_data,
    # setup,
    check_equal,
    bench,
    bench_with_bck_prop,
    # hetero_dataset,
    init_log,
    empty_cache,
)

from setup_lite_softlink import (
    load_data_as_dgl_graph,
    hetero_dataset,
    setup,
    prepare_hetero_graph_simplified,
)

from RGCN_DGL import RGCN_DGL, RGCN_DGL_hetero
from RGCN_PyG import RGCN_PyG

device = setup()

# to successfully benchmark this model using Seastar
# a smaller feature dimension is used in the paper
RGCN_FEAT_DIM = 16


def profile(dataset, feat_dim, out_dim, repeat=1000, bench_item="abcd-+"):
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
    # g, features = load_data(dataset, feat_dim, prepare=False)
    # g_hetero, _ = load_data(dataset, feat_dim, to_homo=False)

    g_hetero = load_data_as_dgl_graph(dataset)
    g = dgl.to_homogeneous(g_hetero)
    features = torch.nn.Parameter(
        torch.rand([sum([g.number_of_nodes(ntype) for ntype in g.ntypes]), feat_dim])
    )
    g_hetero, _ = prepare_hetero_graph_simplified(g_hetero, features)
    # print(features.shape)

    features = features.to(device)

    @empty_cache
    def run_dgl_hetero(g_hetero, features):
        with nvtx.annotate("dgl-hetero"):
            g_hetero = g_hetero.to(device)
            rel_names = list(set(g_hetero.etypes))
            net_dgl_hetero = RGCN_DGL_hetero(
                feat_dim, out_dim, rel_names, len(rel_names)
            ).to(device)

            hetero_features = g_hetero.ndata["h"]
            if len(g.ntypes) == 1:
                hetero_features = {g.ntypes[0]: hetero_features}

            switch_bench_cm_list = []
            if "-" in bench_item:
                switch_bench_cm_list.append((net_dgl_hetero.eval, bench, torch.no_grad))
            if "+" in bench_item:
                switch_bench_cm_list.append(
                    (net_dgl_hetero.train, bench_with_bck_prop, contextlib.nullcontext)
                )
            for (switch_func, bench_func, cm) in switch_bench_cm_list:
                switch_func()
                with cm():
                    bench_func(
                        net=net_dgl_hetero,
                        net_params=(g_hetero, hetero_features),
                        tag="1-DGL-slice",
                        nvprof=False,
                        repeat=repeat,
                        memory=True,
                        log=log,
                    )
            del g_hetero, rel_names, net_dgl_hetero

    @empty_cache
    def run_pyg_bmm(g, features):
        with nvtx.annotate("pyg-bmm", color="red"):
            edge_type = g.edata["_TYPE"]
            u, v = g.edges()
            adj = torch.stack([u, v]).to(device)
            net_pyg_bmm = RGCN_PyG(
                feat_dim, out_dim, int(edge_type.max() + 1), mode="bmm"
            ).to(device)

            switch_bench_cm_list = []
            if "-" in bench_item:
                switch_bench_cm_list.append((net_pyg_bmm.eval, bench, torch.no_grad))
            if "+" in bench_item:
                switch_bench_cm_list.append(
                    (net_pyg_bmm.train, bench_with_bck_prop, contextlib.nullcontext)
                )
            for (switch_func, bench_func, cm) in switch_bench_cm_list:
                switch_func()
                with cm():
                    bench_func(
                        net=net_pyg_bmm,
                        net_params=(adj, features, edge_type),
                        tag="4-PyG-bmm",
                        nvprof=False,
                        repeat=repeat,
                        memory=True,
                        log=log,
                    )
            del edge_type, u, v, adj, net_pyg_bmm

    @empty_cache
    def run_dgl_bmm(g, features):
        with nvtx.annotate("dgl-bmm", color="purple"):
            edge_type = g.edata["_TYPE"]
            g = g.to(device)
            norm = torch.rand(g.num_edges(), 1).to(device)
            net_dgl = RGCN_DGL(feat_dim, out_dim, int(edge_type.max() + 1)).to(device)

            switch_bench_cm_list = []
            if "-" in bench_item:
                switch_bench_cm_list.append((net_dgl.eval, bench, torch.no_grad))
            if "+" in bench_item:
                switch_bench_cm_list.append(
                    (net_dgl.train, bench_with_bck_prop, contextlib.nullcontext)
                )
            for (switch_func, bench_func, cm) in switch_bench_cm_list:
                switch_func()
                with cm():
                    bench_func(
                        net=net_dgl,
                        net_params=(g, features, g.edata["_TYPE"], norm),
                        tag="3-DGL-bmm (segmentmm in latest dgl)",
                        nvprof=False,
                        repeat=repeat,
                        memory=True,
                        log=log,
                    )
            del g, norm, net_dgl

    @empty_cache
    def run_pyg_slice(g, features):
        with nvtx.annotate("pyg-slice", color="blue"):
            edge_type = g.edata["_TYPE"]
            u, v = g.edges()
            adj = torch.stack([u, v]).to(device)

            net_pyg_slice = RGCN_PyG(
                feat_dim, out_dim, int(edge_type.max() + 1), mode="slice"
            ).to(device)

            switch_bench_cm_list = []
            if "-" in bench_item:
                switch_bench_cm_list.append((net_pyg_slice.eval, bench, torch.no_grad))
            if "+" in bench_item:
                switch_bench_cm_list.append(
                    (net_pyg_slice.train, bench_with_bck_prop, contextlib.nullcontext)
                )
            for (switch_func, bench_func, cm) in switch_bench_cm_list:
                switch_func()
                with cm():
                    bench_func(
                        net=net_pyg_slice,
                        net_params=(adj, features, edge_type),
                        tag="2-PyG-slice",
                        nvprof=False,
                        repeat=repeat,
                        memory=True,
                        log=log,
                    )
            del edge_type, u, v, adj, net_pyg_slice

    if "a" in bench_item:
        run_dgl_hetero(g_hetero, features)  # "1-DGL-slice"
    if "b" in bench_item:
        run_pyg_slice(g, features)  # "2-PyG-slice"
    if "c" in bench_item:
        run_dgl_bmm(g, features)  # "3-DGL-bmm (segmentmm in latest dgl)"
    if "d" in bench_item:
        run_pyg_bmm(g, features)  # "4-PyG-bmm"

    return log


if __name__ == "__main__":
    # repeat = int(os.environ.get('REPEAT', 50))
    repeat = 1
    if len(sys.argv) != 4 and len(sys.argv) != 5:
        print(
            "usage: python GCN.py [dataset] [feat_dim] [out_dim] [bench idx abcd-+ (optional)]"
        )
        exit()
    if sys.argv[1] == "all":
        log = {}
        for d in hetero_dataset:
            print(d)
            if len(sys.argv) == 5:
                log[d] = profile(
                    d, int(sys.argv[2]), int(sys.argv[3]), repeat, sys.argv[4].strip()
                )
            else:
                log[d] = profile(d, int(sys.argv[2]), int(sys.argv[3]), repeat)
        pd.DataFrame(log).to_pickle("RGCN_baseline_standalone.pkl")
    else:
        if len(sys.argv) == 5:
            log = profile(
                sys.argv[1],
                int(sys.argv[2]),
                int(sys.argv[3]),
                repeat,
                sys.argv[4].strip(),
            )
        else:
            profile(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), repeat)
