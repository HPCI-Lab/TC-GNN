# %%
from sklearn.metrics import f1_score
import torch
import torch_geometric
from torch_geometric.loader import DataLoader

import Dataset
import Models
import summary  # should be torch_geometric.nn.summary, but it doesn't work

print(f"Torch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Torch geometric version: {torch_geometric.__version__}")

# Global variables
_num_features = None
_num_classes = Dataset.num_classes
_model_name = "GCNet"
_train_batch_size = 1
_test_batch_size = 5
_valid_batch_size = 5

# %% Creating the dataset
dataset = Dataset.PilotDataset(root='./data/bsc')#'./data/cmcc_structured')

# Note that cyclone files start from 1, not form 0
dataset.get(year=1983, cyclone=1).is_directed()
dataset.get(1983, 1)

# %% The Dataloader allows to feed data by batch into the model to exploit parallelism
# The "batch" information is used by the GNN model to tell which nodes belong to which graph
# "batch_size" specifies how many grids will be saved in a single batch
# If "shuffle=True", the data will be reshuffled at every epoch
#dataset = dataset.shuffle()

dataset_size = dataset.len() -2     # remove pre_filter.pt and pre_transform.pt
train_set = []
test_set = []
valid_set = []

for c in range(20):#dataset_size):  # just 20 patches to see if it works
    train_set.append(dataset.get(1983, c+1))

for c in range(20, 25):
    test_set.append(dataset.get(1983, c+1))

for c in range(25, 30):
    valid_set.append(dataset.get(1983, c+1))

train_loader = DataLoader(train_set, batch_size=_train_batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=_test_batch_size, shuffle=False)
valid_loader = DataLoader(valid_set, batch_size=_valid_batch_size, shuffle=False)

# Print the batches
print("\tTrain batches:")
for batch in train_loader:
    print(batch)

print("\tTest batches:")
for batch in test_loader:
    print(batch)

print("\tValidation batches:")
for batch in valid_loader:
    print(batch)

device = torch.device('cpu')#'cuda' if torch.cuda.is_available() else 'cpu')

_num_features = train_loader.dataset[0].num_features

# %% Example at ppi.py in PyG repository
if _model_name == "GCNet":
    Model = Models.GCNet
elif _model_name == "GUNet":
    Model = Models.GUNet

model = Model(
    in_channels = _num_features,
    hidden_channels = 32,   # 16 for GCNet, 32 for GUNet
    out_channels = _num_classes,
    data = train_loader.dataset[0]
).to(device)

print("\nSummary of NN:")
print(f"\tdummy input: {train_loader.dataset[0]}")
print(summary.summary(model, train_loader.dataset[0]))

# %%
loss_op = torch.nn.functional.nll_loss#torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = loss_op(model(data), data.y)
        total_loss += loss.item() * data.num_graphs
        loss.backward()
        optimizer.step()
    return total_loss / len(train_loader.dataset)

loss = train()
print(loss)
# During the creation of train_loader:
#   batch_size=1 leads to a loss of 64.45. Then 20626.77. Then 1203.
#   batch_size=10 leads to a loss of 31.14. Then 228.81. It doesn't seem to depend by the shuffle option

# %% Let's define the testing
@torch.no_grad()
def test(loader):
    model.eval()

    ys, preds = [], []
    for data in loader:
        ys.append(data.y)
        out = model(data.to(device))
        preds.append((out > 0).float().cpu())

    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    return f1_score(y, pred, average='micro') if pred.sum() > 0 else 0

f1_res = test(test_loader)
print(f1_res)

# %% Everything together
for epoch in range(100):
    loss = train()
    valid_f1 = test(valid_loader)
    test_f1 = test(test_loader)
    print(f'Epoch: {epoch+1:03d}, Loss: {loss:.4f}, Val: {valid_f1:.4f}, '
          f'Test: {test_f1:.4f}')

# %%
