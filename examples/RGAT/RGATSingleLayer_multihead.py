import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import nvtx

from torch_sparse import SparseTensor

from graphiler import EdgeBatchDummy, NodeBatchDummy, mpdfg_builder, update_all
from graphiler.utils import (
    load_data,
    setup,
    check_equal,
    bench,
    hetero_dataset,
    DEFAULT_DIM,
    init_log,
    empty_cache,
)

# from GAT_DGL import GAT_DGL
# from GAT_PyG import GAT_PyG


device = setup()

BREAK_FLAG = 2

# copying what RGCN used
RGAT_FEAT_DIM = 16


# Currently Graphiler do not support full module compilation
# therefore, we pass extra parameters as a workaround for class member
# e.g., self.fc_weight, compare with GATLayer.message_func for the difference


def message_func(edges: EdgeBatchDummy, num_heads: int, out_dim: int):
    # (E, 1, in_dim) * (E, in_dim, out_dim) -> (E, 1, out_dim)
    fc_weight_src = edges.srctype["fc_weight"]
    fc_weight_dst = edges.dsttype["fc_weight"]
    z_s = torch.bmm(edges.src["h"].unsqueeze(1), fc_weight_src).view(
        -1, num_heads, out_dim // num_heads
    )
    z_d = torch.bmm(edges.dst["h"].unsqueeze(1), fc_weight_dst).view(
        -1, num_heads, out_dim // num_heads
    )

    # (E, num_heads, 2 * out_dim//num_heads)
    z2 = torch.cat([z_s, z_d], dim=2)
    # (E, num_heads, 2 * out_dim//num_heads) broadcast * (E, num_heads, 2 * out_dim//num_heads) -> (E, num_heads, 2 * out_dim//num_heads) -> (sum) (E, num_heads, 1) > (E, num_heads)
    a = torch.sum(z2 * edges.type["attn_weight"], dim=2).squeeze()
    # edges.type["attn_weight"].view(-1,num_heads, 2*out_dim//num_heads).shape
    return {"z": z_s, "e": F.leaky_relu_(a)}
    # return {"z": z_s, "e": z_s}


def reduce_func(nodes: NodeBatchDummy):

    alpha = torch.softmax(nodes.mailbox["e"], dim=1)
    h = torch.sum(alpha * nodes.mailbox["z"], dim=1)
    return {"h": h}


mpdfg = mpdfg_builder(message_func, reduce_func)
mpdfg_compile = mpdfg_builder(message_func, reduce_func, opt_level=0)
mpdfg_plus_reorder = mpdfg_builder(message_func, reduce_func, opt_level=1)


class Multihead_RGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_rels, num_heads):
        super(Multihead_RGATLayer, self).__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.fc_weight = torch.rand(num_rels, in_dim, out_dim).to(device)
        self.attn_weight = torch.rand(num_rels, num_heads, 2 * out_dim // num_heads).to(
            device
        )

    def message_func(self, edges):
        # TODO: update according to message_func in mpdfg
        fc_weight = self.fc_weight[edges.data["_TYPE"]]
        attn_weight = self.attn_weight[edges.data["_TYPE"]]
        # (E, in_dim) * (in_dim, out_dim) -> (E, out_dim) > (E, num_heads, out_dim // num_heads)
        z_s = torch.mm(edges.src["h"], fc_weight).view(-1, self.num_heads, self.out_dim)
        z_d = torch.mm(edges.dst["h"], fc_weight).view(-1, self.num_heads, self.out_dim)
        # (E, num_heads, 2 * out_dim // num_heads)
        z2 = torch.cat([z_s, z_d], dim=1)
        # (E, num_heads, 2 * out_dim // num_heads) * (num_heads, 2 * out_dim // num_heads) -> (E, num_heads)
        a = torch.sum(z2 * attn_weight, dim=2).squeeze()
        return {"z": z_s, "e": torch.relu(a)}

    def forward(self, g, feature, compile=False):
        g.ndata["h"] = feature
        g.ntype_data["fc_weight"] = self.fc_weight
        g.etype_data["attn_weight"] = self.attn_weight
        if compile:
            if BREAK_FLAG == 0:
                update_all(
                    g,
                    mpdfg_compile,
                    msg_params=(
                        self.num_heads,
                        self.out_dim,
                    ),
                )
            elif BREAK_FLAG == 1:
                update_all(
                    g,
                    mpdfg_plus_reorder,
                    msg_params=(
                        self.num_heads,
                        self.out_dim,
                    ),
                )
            else:
                update_all(
                    g,
                    mpdfg,
                    msg_params=(
                        self.num_heads,
                        self.out_dim,
                    ),
                )
        else:
            g.update_all(self.message_func, reduce_func)
        return g.ndata.pop("h")


class Multihead_RGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_rels, num_heads):
        super(Multihead_RGAT, self).__init__()
        self.layer1 = Multihead_RGATLayer(in_dim, out_dim, num_rels, num_heads)
        # self.layer2 = Multihead_RGATLayer(hidden_dim, out_dim, num_rels, num_heads)

    def forward(self, g, features, compile=False):
        h = self.layer1(g, features, compile)
        h = F.elu(h)
        # h = self.layer2(g, h, compile)
        return h


class Multihead_RGATSingleLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_rels, num_heads):
        super(Multihead_RGATSingleLayer, self).__init__()
        self.layer1 = Multihead_RGATLayer(in_dim, out_dim, num_rels, num_heads)

    def forward(self, g, features, compile=False):
        h = self.layer1(g, features, compile)
        h = F.elu(h)
        return h


def profile(dataset, feat_dim, out_dim, num_heads, repeat=1000):
    log = init_log(
        ["0-DGL-UDF", "1-DGL-primitives", "2-PyG-primitives", "3-Graphiler"],
        ["time", "mem"],
    )
    print("benchmarking on: " + dataset)
    g, features = load_data(dataset, feat_dim, prepare=False)
    features = features.to(device)

    @empty_cache
    def run_baseline_graphiler(g, features):
        g, _ = load_data(dataset, feat_dim, prepare=True)
        g = g.to(device)
        net = Multihead_RGATSingleLayer(
            in_dim=feat_dim,
            out_dim=out_dim,
            num_rels=g.num_rels,
            num_heads=num_heads,
        ).to(device)
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
            check_equal(compile_res, res)
        del g, net, compile_res, res

    @empty_cache
    def run_pyg(g, features):
        with nvtx.annotate("pyg", color="blue"):
            u, v = g.edges()
            adj = SparseTensor(
                row=u, col=v, sparse_sizes=(g.num_src_nodes(), g.num_dst_nodes())
            ).to(device)
            net_pyg = RGAT_PyG(
                in_dim=feat_dim, hidden_dim=DEFAULT_DIM, out_dim=DEFAULT_DIM
            ).to(device)
            net_pyg.eval()
            with torch.no_grad():
                bench(
                    net=net_pyg,
                    net_params=(features, adj),
                    tag="2-PyG-primitives",
                    nvprof=False,
                    repeat=repeat,
                    memory=True,
                    log=log,
                )
            del u, v, adj, net_pyg

    @empty_cache
    def run_dgl(g, features):
        with nvtx.annotate("dgl", color="purple"):
            g = g.to(device)
            net_dgl = RGAT_DGL(
                in_dim=feat_dim, hidden_dim=DEFAULT_DIM, out_dim=DEFAULT_DIM
            ).to(device)
            net_dgl.eval()
            with torch.no_grad():
                bench(
                    net=net_dgl,
                    net_params=(g, features),
                    tag="1-DGL-primitives",
                    nvprof=False,
                    repeat=repeat,
                    memory=True,
                    log=log,
                )
            del g, net_dgl

    run_baseline_graphiler(g, features)
    run_pyg(g, features)
    run_dgl(g, features)

    return log


def breakdown(dataset, feat_dim, out_dim, num_heads, repeat=1000):
    log = init_log(["0-DGL-UDF", "1+compile", "2+reorder", "3+fusion"], ["time", "mem"])
    print("benchmarking on: " + dataset)
    g, features = load_data(dataset, feat_dim)
    g, features = g.to(device), features.to(device)

    net = Multihead_RGATSingleLayer(
        in_dim=feat_dim,
        out_dim=out_dim,
        num_rels=len(g.canonical_etypes),
        num_heads=num_heads,
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
    if len(sys.argv) != 5:
        print(
            "usage: python RGATSingleLayer.py [dataset] [feat_dim] [out_dim] [num_heads]"
        )
        exit()
    if sys.argv[1] == "all":
        log = {}
        for d in hetero_dataset:
            log[d] = profile(
                d, int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), repeat
            )
        pd.DataFrame(log).to_pickle("output/Multihead_RGAT.pkl")
    elif sys.argv[1] == "breakdown":
        log = {}
        for d in hetero_dataset:
            log[d] = breakdown(
                d, int(sys.argv[2]), int(sys.argv[3]), int(argv[4]), repeat
            )
        pd.DataFrame(log).to_pickle("output/RGAT_breakdown.pkl")
    else:
        profile(
            sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), repeat
        )
