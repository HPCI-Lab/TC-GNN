# %%
import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import Dataset

# %%
print(f"Torch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Torch geometric version: {torch_geometric.__version__}")

# %% Creating the dataset
dataset = Dataset.PilotDataset(root='./data/cmcc_structured')
dataset.get(0).edge_index

# %% The Dataloader allows to feed data by batch into the model.
# The "batch" information is used by the GNN model to tell which nodes belong to which graph
# "batch_size" specifies how many grids will be saved in a single batch
loader = DataLoader(dataset, batch_size=1, shuffle=True)
print(loader.num_workers)

# % Print the batches
for batch in loader:
    print(batch)

# %%
