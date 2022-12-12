# import torch
def update_all(g, mpdfg, msg_params=(), reduce_params=(), update_params=()):
    # with torch.jit.optimized_execution(False):
    updated = mpdfg.forward(
        g.DGLGraph,
        g.ndata,
        g.edata,
        g.ntype_data,
        g.etype_data,
        *msg_params,
        *reduce_params,
        *update_params
    )
    for key in updated:
        g.ndata[key] = updated[key]
