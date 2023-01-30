# %%
import torch
import torch_geometric
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import Dataset
import Model

# %%
print(f"Torch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Torch geometric version: {torch_geometric.__version__}")

# %% Creating the dataset
dataset = Dataset.PilotDataset(root='./data/cmcc_structured')
dataset[0]

# %% The Dataloader allows to feed data by batch into the model to exploit parallelism
# The "batch" information is used by the GNN model to tell which nodes belong to which graph
# "batch_size" specifies how many grids will be saved in a single batch
# If "shuffle=True", the data will be reshuffled at every epoch
#dataset = dataset.shuffle()
train_loader = DataLoader(dataset[:8], batch_size=1, shuffle=True)
test_loader = DataLoader(dataset[8:], batch_size=1, shuffle=False)

# Print the batches
for batch in train_loader:
    print(batch)

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

# %% cora.py example on PyG repository
model = Model.GCN(
    in_channels = dataset.num_features,
    hidden_channels = 16,
    out_channels = dataset.num_classes
).to(device)
model.cpu

data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model(data), data.y).backward()
    optimizer.step()

train()

# %% Example at ppi.py in PyG repository
loss_op = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = loss_op(model(data.x, data.edge_index), data.y)
        total_loss += loss.item() * data.num_graphs
        loss.backward()
        optimizer.step()
    return total_loss / len(train_loader.dataset)

train()

