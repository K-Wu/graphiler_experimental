#!/usr/bin/env python3
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import nvtx

from torch_sparse import SparseTensor
from RGAT_DGL import DGL_RGAT_Hetero
from RGAT_PyG import RGAT_PyG

from graphiler import EdgeBatchDummy, NodeBatchDummy, mpdfg_builder, update_all
from graphiler.utils import (
    load_data,
    setup,
    check_equal,
    bench,
    hetero_dataset,
    #    DEFAULT_DIM,
    init_log,
    empty_cache,
)

# from GAT_DGL import GAT_DGL
# from GAT_PyG import GAT_PyG


device = setup()

BREAK_FLAG = 2

# copying what RGCN used
# RGAT_FEAT_DIM = 16


# Currently Graphiler do not support full module compilation
# therefore, we pass extra parameters as a workaround for class member
# e.g., self.fc_weight, compare with GATLayer.message_func for the difference
def message_func(edges: EdgeBatchDummy):
    # (E, 1, in_dim) * (E, in_dim, out_dim) -> (E, 1, out_dim)
    fc_weight = edges.type["fc_weight"]
    # fc_weight_dst = edges.type["fc_weight"]
    z_s = torch.bmm(edges.src["h"].unsqueeze(1), fc_weight).squeeze()
    z_d = torch.bmm(edges.dst["h"].unsqueeze(1), fc_weight).squeeze()

    # (E, 1, 2*out_dim)
    z_2 = torch.cat([z_s, z_d], dim=1)

    # (E, 1, 2*out_dim) * (E, 2*out_dim, 1) -> (E, 1, 1) > (E, 1)
    a = torch.bmm(z_2.unsqueeze(1), edges.type["attn_weight"]).squeeze(1)

    return {"z": z_s, "e": F.leaky_relu_(a)}


def reduce_func(nodes: NodeBatchDummy):
    alpha = torch.softmax(nodes.mailbox["e"], dim=1)
    h = torch.sum(alpha * nodes.mailbox["z"], dim=1)
    return {"h": h}


mpdfg = mpdfg_builder(message_func, reduce_func)
mpdfg_compile = mpdfg_builder(message_func, reduce_func, opt_level=0)
mpdfg_plus_reorder = mpdfg_builder(message_func, reduce_func, opt_level=1)


class RGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_rels):
        super(RGATLayer, self).__init__()
        self.fc_weight = torch.rand(num_rels, in_dim, out_dim).to(device)
        self.attn_weight = torch.rand(num_rels, 2 * out_dim, 1).to(device)

    def reduce_func(self, nodes):
        alpha = torch.softmax(nodes.mailbox["e"], dim=1)
        h = torch.sum(alpha * nodes.mailbox["z"], dim=1)
        return {"h": h}

    # def update_func(self, nodes):
    #     return {"h": nodes.data["h"]}

    def message_func(self, edges):
        # TODO: update according to message_func in mpdfg
        fc_weight = self.fc_weight[edges.data["_TYPE"]]
        attn_weight = self.attn_weight[edges.data["_TYPE"]]
        # (Em)
        z_s = torch.bmm(edges.src["h"].unsqueeze(1), fc_weight).squeeze()
        z_d = torch.bmm(edges.dst["h"].unsqueeze(1), fc_weight).squeeze()
        z2 = torch.cat([z_s, z_d], dim=1)
        a = torch.bmm(z2.unsqueeze(1), attn_weight).squeeze(1)
        return {"z": z_s, "e": torch.relu(a)}

    def forward(self, g, feature, compile=False):
        g.ndata["h"] = feature
        g.etype_data["fc_weight"] = self.fc_weight
        g.etype_data["attn_weight"] = self.attn_weight
        if compile:
            if BREAK_FLAG == 0:
                update_all(g, mpdfg_compile, msg_params=())
            elif BREAK_FLAG == 1:
                update_all(g, mpdfg_plus_reorder, msg_params=())
            else:
                update_all(g, mpdfg, msg_params=())
        else:
            g.update_all(self.message_func, self.reduce_func)
        return g.ndata.pop("h")


# class RGAT(nn.Module):
#     def __init__(self, in_dim, hidden_dim, out_dim, num_rels):
#         super(RGAT, self).__init__()
#         self.layer1 = RGATLayer(in_dim, hidden_dim, num_rels)
#         self.layer2 = RGATLayer(hidden_dim, out_dim, num_rels)

#     def forward(self, g, features, compile=False):
#         h = self.layer1(g, features, compile)
#         h = F.elu(h)
#         h = self.layer2(g, h, compile)
#         return h


class RGATSingleLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_rels):
        super(RGATSingleLayer, self).__init__()
        self.layer1 = RGATLayer(in_dim, out_dim, num_rels)

    def forward(self, g, features, compile=False):
        h = self.layer1(g, features, compile)
        h = F.elu(h)
        return h


def profile(dataset, feat_dim, out_dim, repeat=1000):
    log = init_log(
        ["0-DGL-UDF", "1-DGL-primitives", "2-PyG-primitives", "3-Graphiler"],
        ["time", "mem"],
    )
    print("benchmarking on: " + dataset)
    g, features = load_data(dataset, feat_dim, prepare=False)
    g_hetero, _ = load_data(dataset, feat_dim, to_homo=False)
    features = features.to(device)

    @empty_cache
    def run_baseline_graphiler(g, features):
        g, _ = load_data(dataset, feat_dim, prepare=True)
        g = g.to(device)
        print(g.num_ntypes, g.num_rels, len(g.canonical_etypes))
        net = RGATSingleLayer(in_dim=feat_dim, out_dim=out_dim, num_rels=g.num_rels).to(
            device
        )
        net.eval()
        with torch.no_grad():
            with nvtx.annotate("graphiler", color="orange"):
                compile_res = bench(
                    net=net,
                    net_params=(g, features, True),
                    tag="3-Graphiler",
                    nvprof=False,
                    repeat=repeat,
                    memory=True,
                    log=log,
                )
            with nvtx.annotate("baseline", color="yellow"):
                res = bench(
                    net=net,
                    net_params=(g, features, False),
                    tag="0-DGL-UDF",
                    nvprof=False,
                    repeat=repeat,
                    memory=True,
                    log=log,
                )
            # check_equal(compile_res, res)
        del g, net, compile_res, res

    @empty_cache
    def run_pyg_slice(g, features):
        with nvtx.annotate("pyg", color="blue"):
            edge_type = g.edata["_TYPE"]
            u, v = g.edges()
            adj = SparseTensor(
                row=u, col=v, sparse_sizes=(g.num_src_nodes(), g.num_dst_nodes())
            ).to(device)
            net_pyg = RGAT_PyG(
                in_dim=feat_dim, out_dim=out_dim, num_rels=g.num_rels
            ).to(device)
            net_pyg.eval()
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

    run_baseline_graphiler(g, features)
    print(
        "Warning: baselines are disabled in this script to make sure we are using the latest version of dgl and pyg"
    )
    if False:
        run_dgl_hetero(g_hetero, features)
        run_pyg_slice(g, features)

    return log


def breakdown(dataset, feat_dim, out_dim, repeat=1000):
    log = init_log(["0-DGL-UDF", "1+compile", "2+reorder", "3+fusion"], ["time", "mem"])
    print("benchmarking on: " + dataset)
    g, features = load_data(dataset, feat_dim)
    g, features = g.to(device), features.to(device)

    net = RGATSingleLayer(
        in_dim=feat_dim, out_dim=out_dim, num_rels=len(g.canonical_etypes)
    ).to(device)
    net.eval()
    with torch.no_grad():
        bench(
            net=net,
            net_params=(g, features, False),
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
            net_params=(g, features, True),
            tag="1+compile",
            nvprof=False,
            repeat=repeat,
            memory=True,
            log=log,
        )
        BREAK_FLAG = 1
        bench(
            net=net,
            net_params=(g, features, True),
            tag="2+reorder",
            nvprof=False,
            repeat=repeat,
            memory=True,
            log=log,
        )
        BREAK_FLAG = 2
        bench(
            net=net,
            net_params=(g, features, True),
            tag="3+fusion",
            nvprof=False,
            repeat=repeat,
            memory=True,
            log=log,
        )

    return log


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
