pip install dgl --find-links https://data.dgl.ai/wheels/cu118/repo.html
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install ogb nvtx rdflib pandas
conda install -c conda-forge torch-scatter 
pip install torch_geometric
# setting up LD_LIBRARY_PATH to let pytorch find cuda 11.8 libraries
conda env config vars set LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
#conda install torch-scatter torch-sparse torch-cluster torch-spline-conv -c pyg -c nvidia
#conda install -c pyg torch-sparse torch-cluster torch-spline-conv
#pip install torch==1.8.2+cu111 --find-links --find-links https://download.pytorch.org/whl/lts/1.8/torch_lts.html
# --find-links https://data.pyg.org/whl/torch-1.8.0+cu111.html
# torch-scatter==2.0.8
# torch-sparse==0.6.12
# torch-cluster==1.5.9
# torch-spline-conv==1.2.1