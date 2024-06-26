from graphiler.mpdfg import split, reorder, fusion#, backpropemitter


def optimizer(mpdfg, opt_level):
    #backpropemitter(mpdfg)
    if opt_level == 0:
        return
    if opt_level > 0:
        split(mpdfg)
        reorder(mpdfg)
        reorder(mpdfg)
    if opt_level > 1:
        # convergence check?
        for _ in range(3):
            reorder(mpdfg)
            fusion(mpdfg)
    
    return
