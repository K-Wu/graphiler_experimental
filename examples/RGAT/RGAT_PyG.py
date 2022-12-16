#!/usr/bin/env python3
import torch.nn as nn
import torch.nn.functional as F

# from torch_geometric.nn import RGATConv


class RGAT_PyG(nn.Module):
    def __init__(self, in_dim, out_dim, num_rels, mode="slice"):
        from torch_geometric.nn import RGATConv

        # from rgat_conv_copy import RGATConv

        super(RGAT_PyG, self).__init__()
        if mode == "bmm":
            raise NotImplementedError("PyG does not provide FastRGATConv")
        RGATLayer = RGATConv
        self.layer1 = RGATLayer(in_dim, out_dim, num_rels, aggr="add")
        # self.layer2 = RGCNLayer(hidden_dim, out_dim, num_rels, aggr='add')

    def forward(self, adj, features, edge_type):
        x = self.layer1(features, adj, edge_type=edge_type)
        x = F.relu(x)
        # x = self.layer2(x, adj, edge_type)
        return x
