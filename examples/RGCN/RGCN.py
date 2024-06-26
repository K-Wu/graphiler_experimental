import sys
import os
from timeit import repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import nvtx

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

from RGCN_DGL import RGCN_DGL, RGCN_DGL_hetero
from RGCN_PyG import RGCN_PyG

device = setup()

# to successfully benchmark this model using Seastar
# a smaller feature dimension is used in the paper
RGCN_FEAT_DIM = 16


# note: for heterogenous GNNs, instead of remaining compatible with DGL interface
# we introduce a simplified interface: edges.type['weight'] v.s. weight[edges.data['_TYPE']]
# which is consistent to designs for ndata and edata and might be adopted in DGL
def message_func(edges: EdgeBatchDummy):
    relation_weight = edges.type["weight"]
    msg = torch.bmm(edges.src["h"].unsqueeze(1), relation_weight).squeeze()
    msg = msg * edges.data["norm"]
    return {"m": msg}


def reduce_func(nodes: NodeBatchDummy):
    return {"h": torch.sum(nodes.mailbox["m"], dim=1)}


mpdfg = mpdfg_builder(message_func, reduce_func)


class RGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_rels):
        super(RGCNLayer, self).__init__()
        self.weight = torch.rand(num_rels, in_dim, out_dim).to(device)

    def message_func(self, edges):
        relation_weight = self.weight[edges.data["_TYPE"]]
        msg = torch.bmm(edges.src["h"].unsqueeze(1), relation_weight).squeeze()
        msg = msg * edges.data["norm"]
        return {"m": msg}

    def forward(self, g, feature, norm, compile=False):
        g.ndata["h"] = feature
        g.edata["norm"] = norm
        g.etype_data["weight"] = self.weight
        if compile:
            range_id = nvtx.start_range("my_code_range")
            update_all(g, mpdfg)
            nvtx.end_range(range_id)
        else:
            g.update_all(self.message_func, reduce_func)
            # use built-in as the DGL-bmm baseline
            # g.update_all(self.message_func, fn.sum('m', 'h'))
        return g.ndata.pop("h")


class RGCN(nn.Module):
    def __init__(self, in_dim, out_dim, num_rels):
        super(RGCN, self).__init__()

        self.layer1 = RGCNLayer(in_dim, out_dim, num_rels)
        # self.layer1 = RGCNLayer(in_dim, hidden_dim, num_rels)
        # self.layer2 = RGCNLayer(hidden_dim, out_dim, num_rels)

    def forward(self, g, features, norm, compile=False):
        x = self.layer1(g, features, norm, compile)
        x = F.relu(x)
        # x = self.layer2(g, x, norm, compile)
        return x


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
        norm = torch.rand(g.num_edges(), 1).to(device)
        net = RGCN(feat_dim, out_dim, g.num_rels).to(device)
        net.eval()
        with torch.no_grad():
            with nvtx.annotate("GRAPHILER", color="orange"):
                # range_id = nvtx.start_range("my_code_range")
                compile_res = bench(
                    net=net,
                    net_params=(g, features, norm, True),
                    tag="5-Graphiler",
                    nvprof=False,
                    repeat=repeat,
                    memory=True,
                    log=log,
                )
                # nvtx.end_range(range_id)
            with nvtx.annotate("baseline", color="yellow"):
                res = bench(
                    net=net,
                    net_params=(g, features, norm, False),
                    tag="0-DGL-UDF",
                    nvprof=False,
                    repeat=repeat,
                    memory=True,
                    log=log,
                )
            check_equal(compile_res, res)
        del g, norm, net, compile_res, res

    @empty_cache
    def run_dgl_hetero(g_hetero, features):
        with nvtx.annotate("dgl-hetero"):
            g_hetero = g_hetero.to(device)
            rel_names = list(set(g_hetero.etypes))
            net_dgl_hetero = RGCN_DGL_hetero(
                feat_dim, out_dim, rel_names, len(rel_names)
            ).to(device)
            net_dgl_hetero.eval()
            with torch.no_grad():
                bench(
                    net=net_dgl_hetero,
                    net_params=(g_hetero, g_hetero.ndata["h"]),
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
            net_pyg_bmm = RGCN_PyG(feat_dim, out_dim, g.num_rels, mode="bmm").to(device)
            net_pyg_bmm.eval()
            with torch.no_grad():
                bench(
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
            g = g.to(device)
            norm = torch.rand(g.num_edges(), 1).to(device)
            net_dgl = RGCN_DGL(feat_dim, out_dim, g.num_rels).to(device)
            net_dgl.eval()
            with torch.no_grad():
                bench(
                    net=net_dgl,
                    net_params=(g, features, g.edata["_TYPE"], norm),
                    tag="3-DGL-bmm",
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
            net_pyg_slice = RGCN_PyG(feat_dim, out_dim, g.num_rels, mode="slice").to(
                device
            )
            net_pyg_slice.eval()
            with torch.no_grad():
                bench(
                    net=net_pyg_slice,
                    net_params=(adj, features, edge_type),
                    tag="2-PyG-slice",
                    nvprof=False,
                    repeat=repeat,
                    memory=True,
                    log=log,
                )
            del edge_type, u, v, adj, net_pyg_slice

    run_baseline_graphiler(g, features)
    print(
        "Warning: baselines are disabled in this script to make sure we are using the latest version of dgl and pyg"
    )
    if False:
        g_hetero, _ = load_data(dataset, feat_dim, to_homo=False)
        run_dgl_bmm(g, features)
        run_dgl_hetero(g_hetero, features)
        run_pyg_bmm(g, features)
        run_pyg_slice(g, features)

    return log


if __name__ == "__main__":
    # repeat = int(os.environ.get('REPEAT', 50))
    repeat = 1
    if len(sys.argv) != 4:
        print("usage: python GCN.py [dataset] [feat_dim] [out_dim]")
        exit()
    if sys.argv[1] == "all":
        log = {}
        for d in hetero_dataset:
            log[d] = profile(d, RGCN_FEAT_DIM, repeat)
        pd.DataFrame(log).to_pickle("RGCN.pkl")
    else:
        profile(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), repeat)
