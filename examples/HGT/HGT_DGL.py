import math

import torch
import torch.nn as nn

import dgl.function as fn
from dgl.nn.functional import edge_softmax


class HGTLayerHetero(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        node_dict,
        edge_dict,
        n_heads=1,
        dropout=0.2,
        use_norm=False,
    ):
        super(HGTLayerHetero, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.num_types = len(node_dict)
        self.num_relations = len(edge_dict)
        self.total_rel = self.num_types * self.num_relations * self.num_types
        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.att = None

        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.use_norm = use_norm

        for t in range(self.num_types):
            self.k_linears.append(nn.Linear(in_dim, out_dim))
            self.q_linears.append(nn.Linear(in_dim, out_dim))
            self.v_linears.append(nn.Linear(in_dim, out_dim))
            self.a_linears.append(nn.Linear(out_dim, out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))

        self.relation_pri = nn.Parameter(torch.ones(self.num_relations, self.n_heads))
        self.relation_att = nn.Parameter(
            torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k)
        )
        self.relation_msg = nn.Parameter(
            torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k)
        )
        self.skip = nn.Parameter(torch.ones(self.num_types))
        self.drop = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def forward(self, G, h):
        with G.local_scope():
            node_dict, edge_dict = self.node_dict, self.edge_dict
            for srctype, etype, dsttype in G.canonical_etypes:
                sub_graph = G[srctype, etype, dsttype]

                k_linear = self.k_linears[node_dict[srctype]]
                v_linear = self.v_linears[node_dict[srctype]]
                q_linear = self.q_linears[node_dict[dsttype]]
                if len(node_dict) == 1:
                    k = k_linear(h).view(-1, self.n_heads, self.d_k)
                    v = v_linear(h).view(-1, self.n_heads, self.d_k)
                    q = q_linear(h).view(-1, self.n_heads, self.d_k)
                else:
                    k = k_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                    v = v_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                    q = q_linear(h[dsttype]).view(-1, self.n_heads, self.d_k)

                e_id = self.edge_dict[(srctype, etype, dsttype)]

                relation_att = self.relation_att[e_id]
                relation_pri = self.relation_pri[e_id]
                relation_msg = self.relation_msg[e_id]

                k = torch.einsum("bij,ijk->bik", k, relation_att)
                v = torch.einsum("bij,ijk->bik", v, relation_msg)

                sub_graph.srcdata["k"] = k
                sub_graph.dstdata["q"] = q
                sub_graph.srcdata["v_%d" % e_id] = v

                sub_graph.apply_edges(fn.v_dot_u("q", "k", "t"))
                attn_score = (
                    sub_graph.edata.pop("t").sum(-1) * relation_pri / self.sqrt_dk
                )
                attn_score = edge_softmax(sub_graph, attn_score, norm_by="dst")

                sub_graph.edata["t"] = attn_score.unsqueeze(-1)

            G.multi_update_all(
                {
                    etype: (fn.u_mul_e("v_%d" % e_id, "t", "m"), fn.sum("m", "t"))
                    for etype, e_id in edge_dict.items()
                },
                cross_reducer="mean",
            )

            new_h = {}
            for ntype in G.ntypes:
                """
                Step 3: Target-specific Aggregation
                x = norm( W[node_type] * gelu( Agg(x) ) + x )
                """
                n_id = node_dict[ntype]
                alpha = torch.sigmoid(self.skip[n_id])
                if "t" not in G.nodes[ntype].data:
                    continue
                t = G.nodes[ntype].data["t"].view(-1, self.out_dim)
                trans_out = self.drop(self.a_linears[n_id](t))
                trans_out = trans_out * alpha  # + h[ntype] * (1-alpha) ?
                if self.use_norm:
                    new_h[ntype] = self.norms[n_id](trans_out)
                else:
                    new_h[ntype] = trans_out
            return new_h


class HGT_DGLHetero(nn.Module):
    def __init__(self, node_dict, edge_dict, in_dim, out_dim):  # h_dim,
        super(HGT_DGLHetero, self).__init__()
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.gcs = nn.ModuleList()
        self.in_dim = in_dim
        # self.h_dim = h_dim
        self.out_dim = out_dim
        self.layer0 = HGTLayerHetero(in_dim, out_dim, node_dict, edge_dict)
        # self.layer1 = HGTLayerHetero(h_dim, out_dim, node_dict, edge_dict)

    def forward(self, G, h):
        h = self.layer0(G, h)
        # h = self.layer1(G, h)
        return h


# TODO: implement HGT SegmentMM with hgtconv in lib/python3.9/site-packages/dgl/nn/pytorch/conv/hgtconv.py
class HGT_DGL_SegmentMM(nn.Module):
    def __init__(
        self,
        in_size,
        head_size,
        num_heads,
        num_ntypes,
        num_etypes,
        dropout=0.2,
        use_norm=False,
    ):
        from dgl.nn.pytorch.conv import HGTConv

        super(HGT_DGL_SegmentMM, self).__init__()
        self.layer1 = HGTConv(
            in_size,
            head_size,
            num_heads,
            num_ntypes,
            num_etypes,
            dropout=dropout,
            use_norm=use_norm,
        )

    def forward(self, g, x, ntype, etype, presorted=False):
        """
        Parameters
        ----------
        g : DGLGraph
            The input graph.
        x : torch.Tensor
            A 2D tensor of node features. Shape: :math:`(|V|, D_{in})`.
        ntype : torch.Tensor
            An 1D integer tensor of node types. Shape: :math:`(|V|,)`.
        etype : torch.Tensor
            An 1D integer tensor of edge types. Shape: :math:`(|E|,)`.
        presorted : bool, optional
            Whether *both* the nodes and the edges of the input graph have been sorted by
            their types. Forward on pre-sorted graph may be faster. Graphs created by
            :func:`~dgl.to_homogeneous` automatically satisfy the condition.
            Also see :func:`~dgl.reorder_graph` for manually reordering the nodes and edges.

        Returns
        -------
        torch.Tensor
            New node features. Shape: :math:`(|V|, D_{head} * N_{head})`.
        """
        return self.layer1(g, x, ntype, etype, presorted=presorted)
