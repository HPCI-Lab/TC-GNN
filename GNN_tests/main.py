# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
import torch
import torch_geometric
from torch_geometric.loader import DataLoader

import Dataset
import Models
from utils import time_func, summary  # should be torch_geometric.nn.summary, but it doesn't work

print(f"Torch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Torch geometric version: {torch_geometric.__version__}")

### GLOBAL SETTINGS ###
_model_name = "GUNet"
_num_features = None
_hidden_channels = 32   # 16 for GCNet, 32 for GUNet(from the examples)
_num_classes = 1
_train_batch_size = 5   # TODO see how the results change if you change this
_test_batch_size = 5
_valid_batch_size = 5

# %% Creating the dataset
timestamp = time_func.start_time()
dataset = Dataset.PilotDataset(root='./data/bsc')#'./data/cmcc_structured')
time_func.stop_time(timestamp, "Dataset creation")

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
    hidden_channels = _hidden_channels,
    out_channels = _num_classes,
    data = train_loader.dataset[0]
).to(device)

print("\nSummary of NN:")
print(f"\tdummy input: {train_loader.dataset[0]}")
print(summary.summary(model, train_loader.dataset[0]))

# %%
loss_op = torch.nn.CrossEntropyLoss()#torch.nn.functional.nll_loss#torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

def train():
    model.train()
    total_loss = 0

    for batch in train_loader:
        batch = batch.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + loss
        pred = model(batch)
        pred = pred.squeeze()
        #print(np.shape(pred))
        #print(np.shape(batch.y))
        loss = loss_op(pred, batch.y)

        # backward + optimize
        # loss * _train_batch_size(5)
        total_loss += loss.item() * batch.num_graphs
        loss.backward()
        optimizer.step()

    # average loss = total_loss / training graps(20)
    total_loss = total_loss / len(train_loader.dataset)
    return total_loss

loss = train()
print(loss)

# %% 
# torch.no_grad() as decorator is useful because I'm testing, so not calling 
# the backward(). Mouse on it and read the doc
@torch.no_grad()
def test(loader):
    model.eval()

    ys, preds = [], []
    for batch in loader:
        ys.append(batch.y)
        batch = batch.to(device)

        outputs = model(batch)  # shape: [8000, 1]
        outputs = outputs.squeeze()
        #print(np.shape(outputs))
        preds.append((outputs > 0).float().cpu())   # "> 0" mette a bool

    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    #print(np.shape(ys[1]))
    #print(pred)
    #print(outputs.sum())
    #print(pred.sum())
    return f1_score(y, pred, average='micro') if pred.sum() > 0 else 0

f1_res = test(test_loader)
print(f1_res)

# %% Everything together
timestamp = time_func.start_time()

for epoch in range(50):
    loss = train()
    valid_f1 = test(valid_loader)
    print(f'Epoch: {epoch+1:03d}, Loss: {loss:.4f}, Val: {valid_f1:.4f}')

time_func.stop_time(timestamp, "Training Complete!")

test_f1 = test(test_loader)
print(f'Test F1: {test_f1:.4f}')









# %% 
@torch.no_grad()
def visual():
    model.eval()

    ys, preds = [], []
    for batch in test_loader:
        ys.append(batch.y)
        batch = batch.to(device)

        outputs = model(batch)  # shape: [8000, 1]
        outputs = outputs.squeeze()
        preds.append((outputs > 0).float().cpu())   # "> 0" mette a bool
        
        mat_pred = np.zeros(shape=(40, 40))
        mat_target = np.zeros(shape=(40, 40))

        patch_id = 0
        index = 0+1600*patch_id
        for lon in range(40):
            for lat in range(40):
                mat_pred[lat, lon] = outputs[index].item()
                mat_target[lat, lon] = batch.y[index].item()
                index += 1

        fig, axs = plt.subplots(2, 1, figsize=(6, 6))
        ax0 = axs[0].matshow(mat_pred)
        ax1 = axs[1].matshow(mat_target)
        fig.colorbar(ax0)
        fig.colorbar(ax1)
        plt.show()
        #print(test_set[0].y, '\t', np.shape(test_set[0].y))
        print("Pred:\t", outputs[:], '\n\t', np.shape(outputs[:]))
        print("Target:\t", batch.y[:], '\n\t', np.shape(batch.y[:]))

        #y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()

visual()










# %%
# Test from here: https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html#full-implementation
@torch.no_grad()
def my_test(loader):
    num_samples = len(loader.dataset)
    num_batches = len(loader)
    test_loss, correct = 0, 0

    for batch in loader:
        batch.to(device)

        pred = model(batch)
        test_loss += loss_op(pred, batch.y).item()
        correct += (pred.argmax(1) == batch.y).type(torch.float).sum().item()
    
    test_loss /= num_batches
    correct /= num_samples
    print(correct)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

my_test(test_loader)        
