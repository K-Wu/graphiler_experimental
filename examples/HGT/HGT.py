import sys
import os
import math

import torch
import torch.nn as nn
import pandas as pd
import nvtx

from graphiler import EdgeBatchDummy, NodeBatchDummy, mpdfg_builder, update_all
from graphiler.utils import (
    load_data,
    setup,
    check_equal,
    bench,
    hetero_dataset,
    init_log,
    empty_cache,
    # bench_with_bck_prop,
)
import contextlib

import dgl.function as fn
from dgl.nn.functional import edge_softmax

from HGT_DGL import HGT_DGLHetero, HGT_DGL_SegmentMM
from HGT_PyG import HGT_PyG, get_ntype

device = setup()

BREAK_FLAG = 2


# note: for heterogenous GNNs, instead of remaining compatible with DGL interface
# we introduced a simplified interface which might be adopted in DGL in the future:
# nodes.type['weight'], edges.srctype, edges.dsttype and edges.type
# which are consistent to nodes.data['h'], edges.src, edges.dst and edge.data
# v.s. weight[nodes.data['_TYPE']], edges.src['_TYPE'], edges.dst['_TYPE'] and edges.data['_TYPE']
def message_func(edges: EdgeBatchDummy, sqrt_dk: float):

    k_weight = edges.srctype["k_weight"]
    v_weight = edges.srctype["v_weight"]
    q_weight = edges.dsttype["q_weight"]

    k = torch.bmm(edges.src["h"].unsqueeze(1), k_weight).squeeze()
    v = torch.bmm(edges.src["h"].unsqueeze(1), v_weight).squeeze()
    q = torch.bmm(edges.dst["h"].unsqueeze(1), q_weight).squeeze()

    relation_att = edges.type["relation_att"]
    relation_msg = edges.type["relation_msg"]
    relation_pri = edges.type["relation_pri"]

    k = torch.bmm(k.unsqueeze(1), relation_att).squeeze()
    v = torch.bmm(v.unsqueeze(1), relation_msg).squeeze()
    t = k * q
    attn_score = torch.sum(t, dim=1, keepdim=True) * relation_pri / sqrt_dk
    return {"attn": attn_score, "v": v}


def reduce_func(nodes: NodeBatchDummy):
    t = torch.softmax(nodes.mailbox["attn"], dim=1)
    m = t * nodes.mailbox["v"]
    return {"t": torch.sum(m, dim=1)}


def update_func(nodes: NodeBatchDummy):
    skip = nodes.type["skip"]
    a_weight = nodes.type["a_weight"]
    alpha = torch.sigmoid(skip)
    trans_out = torch.bmm(nodes.data["t"].unsqueeze(1), a_weight).squeeze()
    return {"h": trans_out * alpha.unsqueeze(1)}


mpdfg = mpdfg_builder(message_func, reduce_func, update_func)
mpdfg_compile = mpdfg_builder(message_func, reduce_func, update_func, opt_level=0)
mpdfg_plus_reorder = mpdfg_builder(message_func, reduce_func, update_func, opt_level=1)


class HGTLayer_simplified(nn.Module):
    def __init__(self, in_feat_dim, out_feat_dim, num_ntypes, num_rels):
        super(HGTLayer_simplified, self).__init__()
        # set the num_head to be 1
        self.in_feat_dim = in_feat_dim
        self.out_feat_dim = out_feat_dim
        self.sqrt_dk = math.sqrt(self.out_feat_dim)
        self.num_ntypes = num_ntypes
        self.num_rels = num_rels

        self.k_weight = torch.rand(
            self.num_ntypes, self.in_feat_dim, self.out_feat_dim
        ).to(device)
        self.q_weight = torch.rand(
            self.num_ntypes, self.in_feat_dim, self.out_feat_dim
        ).to(device)
        self.v_weight = torch.rand(
            self.num_ntypes, self.in_feat_dim, self.out_feat_dim
        ).to(device)
        self.a_weight = torch.rand(
            self.num_ntypes, self.out_feat_dim, self.out_feat_dim
        ).to(device)

        self.relation_pri = torch.ones(self.num_rels, 1).to(device)
        self.relation_att = torch.rand(
            self.num_rels, self.out_feat_dim, self.out_feat_dim
        ).to(device)
        self.relation_msg = torch.rand(
            self.num_rels, self.out_feat_dim, self.out_feat_dim
        ).to(device)

        self.skip = torch.ones(self.num_ntypes).to(device)

    def message_func(self, edges):
        # (E, in_dim, out_dim)
        k_weight = self.k_weight[edges.src["_TYPE"]]
        v_weight = self.v_weight[edges.src["_TYPE"]]
        q_weight = self.q_weight[edges.dst["_TYPE"]]
        # (E, 1)
        relation_pri = self.relation_pri[edges.data["_TYPE"]]
        # (E, out_dim, out_dim)
        relation_att = self.relation_att[edges.data["_TYPE"]]
        # (E, out_dim, out_dim)
        relation_msg = self.relation_msg[edges.data["_TYPE"]]

        # (E, out_dim) <- (E, 1, in_dim) * (E, in_dim, out_dim)
        k = torch.bmm(edges.src["h"].unsqueeze(1), k_weight).squeeze()
        v = torch.bmm(edges.src["h"].unsqueeze(1), v_weight).squeeze()
        q = torch.bmm(edges.dst["h"].unsqueeze(1), q_weight).squeeze()

        # (E, out_dim) <- (E, 1, out_dim) * (E, out_dim, out_dim)
        k = torch.bmm(k.unsqueeze(1), relation_att).squeeze()
        v = torch.bmm(v.unsqueeze(1), relation_msg).squeeze()
        # (E, out_dim)
        t = k * q
        # (E, 1)
        attn_score = torch.sum(t, dim=1, keepdims=True) * relation_pri / self.sqrt_dk
        return {"attn": attn_score, "v": v}

    def update_func(self, nodes):
        # (N, 1)
        skip = self.skip[nodes.data["_TYPE"]]
        # (N, out_dim, out_dim)
        a_weight = self.a_weight[nodes.data["_TYPE"]]
        # (N, 1)
        alpha = torch.sigmoid(skip)
        # (N, out_dim)
        trans_out = torch.bmm(nodes.data["t"].unsqueeze(1), a_weight).squeeze()
        return {"h": trans_out * alpha.unsqueeze(1)}

    def forward(self, g, h, flag=False):
        g.ndata["h"] = h
        g.ntype_data["k_weight"] = self.k_weight
        g.ntype_data["v_weight"] = self.v_weight
        g.ntype_data["q_weight"] = self.q_weight
        g.etype_data["relation_pri"] = self.relation_pri
        g.etype_data["relation_att"] = self.relation_att
        g.etype_data["relation_msg"] = self.relation_msg
        g.ntype_data["skip"] = self.skip
        g.ntype_data["a_weight"] = self.a_weight

        if flag == "compile":
            range_id = nvtx.start_range("my_code_range")
            if BREAK_FLAG == 0:
                update_all(g, mpdfg_compile, msg_params=(self.sqrt_dk,))
            elif BREAK_FLAG == 1:
                update_all(g, mpdfg_plus_reorder, msg_params=(self.sqrt_dk,))
            else:
                update_all(g, mpdfg, msg_params=(self.sqrt_dk,))
            nvtx.end_range(range_id)
        elif flag == "batch":
            # use dgl built-in functions as dgl-batch baseline
            g.apply_edges(self.message_func)
            g.edata["m"] = edge_softmax(g, g.edata["attn"]) * g.edata["v"]
            g.update_all(fn.copy_e("m", "m"), fn.sum("m", "t"), self.update_func)
        elif flag == "naive":
            g.update_all(self.message_func, reduce_func, self.update_func)
        else:
            assert False & "unsupported flag"


class HGT(nn.Module):
    def __init__(self, in_dim, out_dim, num_ntypes, num_rels):  # , h_dim
        super(HGT, self).__init__()
        self.in_dim = in_dim
        # self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_ntypes = num_ntypes
        self.num_rels = num_rels
        # print("num_rels", num_rels)

        self.layer0 = HGTLayer_simplified(
            self.in_dim, self.out_dim, self.num_ntypes, self.num_rels
        )
        # self.layer1 = HGTLayer_simplified(
        #     self.h_dim, self.out_dim, self.num_ntypes, self.num_rels)

    def forward(self, g, h, flag="naive"):
        self.layer0(g, h, flag=flag)
        # self.layer1(g, g.ndata['h'], flag=flag)
        return g.ndata.pop("h")


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
    g, features = load_data(dataset, feat_dim, prepare=False)
    features = features.to(device)

    @empty_cache
    def run_baseline_graphiler(g, features):
        g, _ = load_data(dataset, feat_dim, prepare=True)
        g = g.to(device)
        net = HGT(feat_dim, out_dim, g.num_ntypes, g.num_rels).to(
            device
        )  # DEFAULT_DIM, len(g._ntypes), len(g._etypes)).to(device)
        net.eval()
        with torch.no_grad():
            with nvtx.annotate("graphiler", color="orange"):
                # range_id = nvtx.start_range("my_code_range")
                compile_res = bench(
                    net=net,
                    net_params=(g, features, "compile"),
                    tag="5-Graphiler",
                    nvprof=False,
                    repeat=repeat,
                    memory=True,
                    log=log,
                )
                # nvtx.end_range(range_id)
            # with nvtx.annotate("DGL-bmm", color="green"):
            #     res = bench(
            #         net=net,
            #         net_params=(g, features, "batch"),
            #         tag="3-DGL-bmm",
            #         nvprof=False,
            #         repeat=repeat,
            #         memory=True,
            #         log=log,
            #     )
            # check_equal(compile_res, res)
            # with nvtx.annotate("baseline", color="yellow"):
            #     bench(
            #         net=net,
            #         net_params=(g, features, "naive"),
            #         tag="0-DGL-UDF",
            #         nvprof=False,
            #         repeat=repeat,
            #         memory=True,
            #         log=log,
            #     )
        del g, net, compile_res  # , res

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
            for (switch_func, bench_func, cm) in [
                (net_hetero.eval, bench, torch.no_grad),
                (net_hetero.train, bench_with_bck_prop, contextlib.nullcontext),
            ]:
                switch_func()
                with cm():
                    bench_func(
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
                feat_dim, out_dim, 1, g.num_ntypes, g.num_rels
            ).to(device)
            for (switch_func, bench_func, cm) in [
                (net_dgl.eval, bench, torch.no_grad),
                (net_dgl.train, bench_with_bck_prop, contextlib.nullcontext),
            ]:
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

    run_baseline_graphiler(g, features)
    print(
        "Warning: baselines are disabled in this script to make sure we are using the latest version of dgl and pyg"
    )
    if False:
        g_hetero, _ = load_data(dataset, feat_dim, to_homo=False)
        run_dgl_slice(g_hetero, features)
        run_dgl_segmentmm(g, features)
        run_pyg_bmm(g, features)
        run_pyg_slice(g, features)

    return log


def breakdown(dataset, feat_dim, out_dim, repeat=1000):
    log = init_log(["0-DGL-UDF", "1+compile", "2+reorder", "3+fusion"], ["time", "mem"])
    print("benchmarking on: " + dataset)
    g, features = load_data(dataset, feat_dim)
    g, features = g.to(device), features.to(device)
    net = HGT(feat_dim, out_dim, g.num_ntypes, g.num_rels).to(device)
    net.eval()
    with torch.no_grad():
        bench(
            net=net,
            net_params=(g, features, "naive"),
            tag="0-DGL-UDF",
            nvprof=False,
            repeat=repeat,
            memory=True,
            log=log,
        )
        global BREAK_FLAG
        BREAK_FLAG = 0
        bench(
            net=net,
            net_params=(g, features, "compile"),
            tag="1+compile",
            nvprof=False,
            repeat=repeat,
            memory=True,
            log=log,
        )
        BREAK_FLAG = 1
        bench(
            net=net,
            net_params=(g, features, "compile"),
            tag="2+reorder",
            nvprof=False,
            repeat=repeat,
            memory=True,
            log=log,
        )
        BREAK_FLAG = 2
        bench(
            net=net,
            net_params=(g, features, "compile"),
            tag="3+fusion",
            nvprof=False,
            repeat=repeat,
            memory=True,
            log=log,
        )

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
        pd.DataFrame(log).to_pickle("HGT.pkl")
    elif sys.argv[1] == "breakdown":
        log = {}
        for d in hetero_dataset:
            log[d] = breakdown(d, int(sys.argv[2]), int(sys.argv[3]), repeat)
        pd.DataFrame(log).to_pickle("HGT_breakdown.pkl")
    else:
        profile(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), repeat)
