
from graphiler.utils import load_data, hetero_dataset

if __name__ == "__main__":
    for dataset in hetero_dataset:
        g, _ = load_data(dataset, feat_dim=1, prepare=False, to_homo=False)
        print(dataset, len(g.canonical_etypes))

# aifb 104
# mutag 50
# bgs 122
# biokg 51
# am 108
# wikikg2 535
