import sys
import os
import math

import torch
import torch.nn as nn
import pandas as pd
import nvtx

import contextlib

import dgl

from bench_softlink import (
    # load_data,
    # setup,
    check_equal,
    bench,
    # hetero_dataset,
    init_log,
    empty_cache,
    bench_with_bck_prop,
)

from setup_lite_softlink import (
    load_data_as_dgl_graph,
    hetero_dataset,
    setup,
    prepare_hetero_graph_simplified,
)


import dgl.function as fn
from dgl.nn.functional import edge_softmax

from HGT_DGL import HGT_DGLHetero, HGT_DGL_SegmentMM
from HGT_PyG import HGT_PyG, get_ntype

device = setup()


def profile(dataset, feat_dim, out_dim, repeat=1000, bench_item="abcd-+"):
    nvtx_enable_flag = "*" in bench_item
    # '-' means enabling profiling inference
    # '+' means enabling profiling training

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
    features = torch.nn.Parameter(
        torch.rand([sum([g.number_of_nodes(ntype) for ntype in g.ntypes]), feat_dim])
    )
    g_hetero, _ = prepare_hetero_graph_simplified(g_hetero, features)
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

            switch_bench_cm_list = []
            if "-" in bench_item:
                switch_bench_cm_list.append((net_hetero.eval, bench, torch.no_grad))
            if "+" in bench_item:
                switch_bench_cm_list.append(
                    (net_hetero.train, bench_with_bck_prop, contextlib.nullcontext)
                )
            for (switch_func, bench_func, cm) in switch_bench_cm_list:
                switch_func()
                with cm():
                    # print(g_hetero.ndata["h"])
                    bench_func(
                        net=net_hetero,
                        net_params=(g_hetero, g_hetero.ndata["h"]),
                        tag="1-DGL-slice",
                        nvprof=False,
                        repeat=repeat,
                        memory=True,
                        log=log,
                        nvtx_flag=nvtx_enable_flag,
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
                        net_params=(
                            g,
                            features,
                            g.ndata["_TYPE"],
                            g.edata["_TYPE"],
                            True,
                        ),
                        tag="3MK1-DGL-segmentmm",
                        nvprof=False,
                        repeat=repeat,
                        memory=True,
                        log=log,
                        nvtx_flag=nvtx_enable_flag,
                    )

            del g, net_dgl  # , norm

    @empty_cache
    def run_pyg_slice(g, features):

        with nvtx.annotate("pyg-slice", color="blue"):
            u, v = g.edges()
            adj = torch.stack([u, v]).to(device)
            src_type, dst_type = get_ntype(
                adj, g.edata["_TYPE"], g.ndata["_TYPE"], int(g.edata["_TYPE"].max()) + 1
            )
            net_pyg_slice = HGT_PyG(
                feat_dim,
                out_dim,
                int(g.ndata["_TYPE"].max()) + 1,
                int(g.edata["_TYPE"].max()) + 1,
                mode="slice",
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
                        nvtx_flag=nvtx_enable_flag,
                    )

            del u, v, adj, src_type, dst_type, net_pyg_slice

    @empty_cache
    def run_dgl_bmm(g, features):
        g, _ = load_data(dataset, feat_dim, prepare=True)
        g = g.to(device)
        net = HGT(feat_dim, out_dim, g.num_ntypes, g.num_rels).to(
            device
        )  # DEFAULT_DIM, len(g._ntypes), len(g._etypes)).to(device)
        net.eval()
        with torch.no_grad():
            # with nvtx.annotate("DGL-bmm", color="green"):
            bench(
                net=net,
                net_params=(g, features, "batch"),
                tag="3-DGL-bmm",
                nvprof=False,
                repeat=repeat,
                memory=True,
                log=log,
            )
        del g, net  # , compile_res#, res

    @empty_cache
    def run_dgl_udf(g, features):
        g, _ = load_data(dataset, feat_dim, prepare=True)
        g = g.to(device)
        net = HGT(feat_dim, out_dim, g.num_ntypes, g.num_rels).to(
            device
        )  # DEFAULT_DIM, len(g._ntypes), len(g._etypes)).to(device)
        net.eval()
        with torch.no_grad():
            # with nvtx.annotate("baseline", color="yellow"):
            bench(
                net=net,
                net_params=(g, features, "naive"),
                tag="0-DGL-UDF",
                nvprof=False,
                repeat=repeat,
                memory=True,
                log=log,
            )
        del g, net  # , compile_res#, res

    @empty_cache
    def run_pyg_bmm(g, features):
        with nvtx.annotate("pyg-bmm", color="red"):
            u, v = g.edges()
            adj = torch.stack([u, v]).to(device)
            src_type, dst_type = get_ntype(
                adj, g.edata["_TYPE"], g.ndata["_TYPE"], int(g.edata["_TYPE"].max()) + 1
            )
            net_pyg_bmm = HGT_PyG(
                feat_dim,
                out_dim,
                int(g.ndata["_TYPE"].max()) + 1,
                int(g.edata["_TYPE"].max()) + 1,
                mode="bmm",
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
                        nvtx_flag=nvtx_enable_flag,
                    )
            del u, v, adj, src_type, dst_type, net_pyg_bmm

    if "a" in bench_item:
        run_dgl_slice(g_hetero, features)  # "1-DGL-slice"
    if "b" in bench_item:
        run_pyg_slice(g, features)  # "2-PyG-slice"
    if "c" in bench_item:
        run_dgl_segmentmm(g, features)  # "3MK1-DGL-segmentmm"
    if "d" in bench_item:
        run_pyg_bmm(g, features)  # "4-PyG-bmm"

    return log


if __name__ == "__main__":
    # repeat = int(os.environ.get('REPEAT', 50))
    # torch.backends.cudnn.allow_tf32 = False
    # torch.backends.cudnn.enabled = False
    # torch.backends.cuda.matmul.allow_tf32 = False
    repeat = 1
    if len(sys.argv) != 4 and len(sys.argv) != 5:
        print(
            "usage: python HGT.py [dataset] [feat_dim] [out_dim] [bench_item abcd-+* (optional)]"
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
        pd.DataFrame(log).to_pickle("HGT_baseline_standalone.pkl")
    elif sys.argv[1] == "breakdown":
        raise NotImplementedError("implemented only for graphiler-related routines")
    else:
        if len(sys.argv) == 5:
            profile(
                sys.argv[1],
                int(sys.argv[2]),
                int(sys.argv[3]),
                repeat,
                sys.argv[4].strip(),
            )
        else:
            profile(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), repeat)
