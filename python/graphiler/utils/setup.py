import numpy as np
from pathlib import Path

import torch

from ..utils_lite.setup_lite import *

import dgl


DEFAULT_DIM = 64
DGL_PATH = str(Path.home()) + "/.dgl/"
torch.classes.load_library(DGL_PATH + "libgraphiler.so")


def load_data(name, feat_dim=DEFAULT_DIM, prepare=True, to_homo=True):

    g = load_data_as_dgl_graph(name)

    node_feats = torch.rand(
        [sum(g.number_of_nodes(ntype) for ntype in g.ntypes), feat_dim]
    )

    if name in hetero_dataset:
        g, type_pointers = prepare_hetero_graph_simplified(g, node_feats)
        if to_homo:
            g = dgl.to_homogeneous(g)
            if prepare:
                g = prepare_graph(g)
                g.DGLGraph.SetNtypePointer(type_pointers["ntype_node_pointer"].cuda())
                g.DGLGraph.SetEtypePointer(type_pointers["etype_edge_pointer"].cuda())
                g.DGLGraph.SetNtypeCOO(g.ndata["_TYPE"].type(torch.LongTensor).cuda())
                g.DGLGraph.SetEtypeCOO(g.edata["_TYPE"].type(torch.LongTensor).cuda())

            g.num_ntypes = len(type_pointers["ntype_node_pointer"]) - 1
            # note #rels is different to #etypes in some cases
            # for simplicity we use these two terms interchangeably
            # and refer an edge type as (src_type, etype, dst_type)
            # see DGL document for more information
            g.num_rels = num_etypes = len(type_pointers["etype_edge_pointer"]) - 1
    elif prepare:
        g = prepare_graph(g)
    g.ntype_data = {}
    g.etype_data = {}
    return g, node_feats


def prepare_graph(g, ntype=None):
    # Todo: integrate with dgl.graph, long int, multiple devices

    reduce_node_index = g.in_edges(g.nodes(ntype))[0]
    reduce_node_index = reduce_node_index.type(torch.IntTensor).cuda()
    reduce_edge_index = (
        g.in_edges(g.nodes(ntype), form="eid").type(torch.IntTensor).cuda()
    )
    message_node_index = (g.out_edges(g.nodes(ntype))[1]).type(torch.IntTensor).cuda()
    message_edge_index = (
        g.out_edges(g.nodes(ntype), form="eid").type(torch.IntTensor).cuda()
    )
    assert (
        len(reduce_node_index)
        == len(reduce_edge_index)
        == len(message_node_index)
        == len(message_edge_index)
        == g.num_edges()
    )

    src, dst = g.edges()
    Coosrc, Coodst = (
        src.type(torch.LongTensor).cuda(),
        dst.type(torch.LongTensor).cuda(),
    )

    reduce_node_pointer = [0] + g.in_degrees(g.nodes(ntype)).tolist()
    message_node_pointer = [0] + g.out_degrees(g.nodes(ntype)).tolist()

    for i in range(1, len(reduce_node_pointer)):
        reduce_node_pointer[i] += reduce_node_pointer[i - 1]
        message_node_pointer[i] += message_node_pointer[i - 1]
    reduce_node_pointer = torch.IntTensor(reduce_node_pointer).cuda()
    message_node_pointer = torch.IntTensor(message_node_pointer).cuda()

    g.DGLGraph = torch.classes.my_classes.DGLGraph(
        reduce_node_pointer,
        reduce_node_index,
        reduce_edge_index,
        message_node_pointer,
        message_node_index,
        message_edge_index,
        Coosrc,
        Coodst,
        None,
        None,
    )

    return g


if __name__ == "__main__":
    # a place for testing data loading
    for dataset in homo_dataset:
        load_data(dataset)
    for dataset in hetero_dataset:
        load_data(dataset)
