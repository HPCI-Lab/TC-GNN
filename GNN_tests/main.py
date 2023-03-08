# %%
# Imports + Global settings
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
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
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ON_CLUSTER = False
if ON_CLUSTER:
    DATA_PATH = '../../data/gnn_records/'
else:
    DATA_PATH = './data/bsc_records/'

_TRAIN_SET = [1980, 1982]
_VALID_SET = [1983]
_TEST_SET = [1981]
LABEL_TYPE = "binary"    # binary(classification) or distance(regression)
_model_name = "GUNet"
_num_features = None
_hidden_channels = 32   # 16 for GCNet, 32 for GUNet(from the examples)
_num_classes = 1
_train_batch_size = 512   # TODO see how the results change if you change this
_test_batch_size = 128
_valid_batch_size = 128

TIMESTAMP = time_func.start_time()

# %%
# Dataset creation
timestamp = time_func.start_time()
dataset = Dataset.PilotDataset(root=DATA_PATH + LABEL_TYPE, label_type=LABEL_TYPE)
time_func.stop_time(timestamp, "Dataset creation")

# Note that cyclone files start from 1, not form 0
print("Year 1983, cyclone 1: ", dataset.get(1983, 1))
#dataset.get(year=1983, cyclone=1).is_directed()

# %%
# Extract number of cyclones per year
graph_names = os.listdir(dataset.processed_dir)
if 'pre_filter.pt' in graph_names: graph_names.remove('pre_filter.pt')
if 'pre_transform.pt' in graph_names: graph_names.remove('pre_transform.pt')

years = []
graph_dict =  {}

for gn in graph_names:
	y = gn.split('_')[1]
	years.append(y)

years = np.unique(years)

for y in years:
	graph_dict[y] = 0

for gn in graph_names:
	y = gn.split('_')[1]
	c = gn.split('_')[3].split('.')[0]
	if graph_dict[y] < int(c):
		graph_dict[y] = int(c)

print("Dictionary of graph years and number of cyclones:\n", graph_dict)

# %%
# Data loader set up + Feature normalization
# The Dataloader allows to feed data by batch into the model to exploit parallelism
# The "batch" information is used by the GNN model to tell which nodes belong to which graph
# "batch_size" specifies how many grids will be saved in a single batch
# TODO: If "shuffle=True", the data will be reshuffled at every epoch. understand
# if this means that you may potentially train over the same patch/batch over and over
#dataset = dataset.shuffle()

train_set = []
test_set = []
valid_set = []

# Feature normalization. Fit the training and tranform training, test and validation
scaler = MinMaxScaler()     # default range: [0, 1]

# Fit the scaler to the training set, one piece at a time
for year in _TRAIN_SET:
    for cyclone in range(graph_dict[str(year)]):
        patch = dataset.get(year, cyclone+1)        # Cyclones files start from 1
        scaler.partial_fit(patch.x)
        train_set.append(patch)

# Normalize training, test and validation sets with the obtained values
for year in _TRAIN_SET:
    for cyclone in range(graph_dict[str(year)]):
        patch = train_set[cyclone]
        patch.x = torch.tensor(scaler.transform(patch.x), dtype=torch.float)
        train_set[cyclone] = patch

for year in _VALID_SET:
    for cyclone in range(500):#graph_dict[str(year)]):
        patch = dataset.get(year, cyclone+1)
        patch.x = torch.tensor(scaler.transform(patch.x), dtype=torch.float)
        valid_set.append(patch)

for year in _TEST_SET:
    for cyclone in range(500):#graph_dict[str(year)]):
        patch = dataset.get(year, cyclone+1)
        patch.x = torch.tensor(scaler.transform(patch.x), dtype=torch.float)
        test_set.append(patch)

train_loader = DataLoader(train_set, batch_size=_train_batch_size, shuffle=True)#, batch_sampler=None)
test_loader = DataLoader(test_set, batch_size=_test_batch_size, shuffle=False)
valid_loader = DataLoader(valid_set, batch_size=_valid_batch_size, shuffle=False)

''' # Cannot print in this way if I use _train_batch_size=None
# Print the batches and create an element with all its features
print("\tTrain batches:")
for batch in train_loader:
    print(batch)
'''

print("Train set size: ", len(train_set))
print("Valid set size: ", len(valid_set))
print("Test set size: ", len(test_set))

_num_features = train_loader.dataset[0].num_features

#%%
# Tests with channels
patch = train_loader.dataset[0]
print("Channels last: ", patch.x.is_contiguous(memory_format=torch.channels_last))
print("Contiguous(channels first): ", patch.x.is_contiguous(memory_format=torch.contiguous_format))
tmp = patch.x.unsqueeze(0)
print(tmp.size())
batch = next(iter(train_loader))
print(batch)

# %%
# Model instantiation and summary
if _model_name == "GCNet":
    Model = Models.GCNet
elif _model_name == "GUNet":
    Model = Models.GUNet

model = Model(
    in_channels = _num_features,
    hidden_channels = _hidden_channels,
    out_channels = _num_classes,
    data = train_loader.dataset[0]      # Just to initialize the model
).to(DEVICE)

#print("\nSummary of NN:")
#print(f"\tdummy input: {train_loader.dataset[0]}")
#print(summary.summary(model, train_loader.dataset[0]))

# %%
# Loss + Optimizer + train()

# Classification losses
loss_op = torch.nn.BCELoss()            # works best with _train_size=100, higher in general
#loss_op = torch.nn.CrossEntropyLoss()  # works best with _train_size=20, lower in general
#loss_op = torch.nn.BCEWithLogitsLoss()
#loss_op = torch.nn.functional.nll_loss
#from dice import dice_score
#loss_op = dice_score

# Regression losses
#loss_op = torch.nn.L1Loss()
#loss_op = torch.nn.MSELoss()

print(f"Loss created!\n\tIn use: {loss_op}")
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

def train():
    model.train()
    total_loss = 0

    for batch in train_loader:
        batch = batch.to(DEVICE)

        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + loss
        pred = model(batch)
        pred = pred.squeeze()
        #print(pred)
        #print(batch.y)
        loss = loss_op(pred, batch.y)   # both [8000] in size
        #loss /= 10  # TODO be careful with this
        
        # If you try the Soft Dice Score, use this(even if the loss stays constant)
        #loss.requires_grad = True
        #loss = torch.tensor(loss.item(), requires_grad=True)

        # backward + optimize
        # loss * _train_batch_size(5)
        total_loss += loss.item() * batch.num_graphs
        loss.backward()
        optimizer.step()

    # average loss = total_loss / training graps(20)
    total_loss = total_loss / len(train_loader.dataset)
    return total_loss

loss = train()
print("Train loss, debug: ", loss)

# %%
# Evaluate()
# torch.no_grad() as decorator is useful because I'm validating/testing, so not calling 
# the backward(). Mouse on it and read the doc
@torch.no_grad()
def evaluate(loader):
    model.eval()
    total_loss = 0

    for batch in loader:
        batch = batch.to(DEVICE)

        outputs = model(batch)  # shape: [8000, 1]
        outputs = outputs.squeeze()

        loss = loss_op(outputs, batch.y)
        # loss /= 10    # TODO be careful with this
        total_loss += loss.item() * batch.num_graphs

    #print(outputs.sum())
    #print(pred.sum())
    
    total_loss = total_loss / len(loader.dataset)
    return total_loss

loss = evaluate(test_loader)
print("Test loss, debug: ", loss)

time_func.stop_time(TIMESTAMP, "Computation before training finished!")

# %%
# Epoch training, validation and testing
timestamp = time_func.start_time()

train_loss = []
valid_loss = []

for epoch in range(100):
    t_loss = train()
    v_loss = evaluate(valid_loader)
    print(f'Epoch: {epoch+1:03d}, Train loss: {t_loss:.4f}, Val loss: {v_loss:.4f}')
    train_loss.append(t_loss)
    valid_loss.append(v_loss)

time_func.stop_time(timestamp, "Training Complete!")

metric = evaluate(test_loader)
print(f'Metric for test: {metric:.4f}')

# %%
# Plot train and validation losses
plt.figure(figsize=(8, 4))
plt.plot(train_loss, label='Train loss')
plt.plot(valid_loss, label='Validation loss')
plt.legend(title="Loss type: " + str(loss_op))
if ON_CLUSTER:
    plt.savefig('./images/train_valid_losses.png')
    plt.close()
else:
    plt.show()

# %% 
# Visualization of test batch truth against predictions
@torch.no_grad()
def visual():
    model.eval()

    batch = next(iter(test_loader))
    batch = batch.to(DEVICE)

    outputs = model(batch)  # shape: [1600*_test_batch_size, 1]
    outputs = outputs.squeeze()
    
    # Preparing the plot
    fig, axs = plt.subplots(ncols=4, nrows=2, figsize=(16, 6))
    #fig.suptitle('Predictions / Ground Truths', fontsize=14, y=0.95)
    col = 0

    # Take 4 patches from the batch
    shift = 14
    for patch_id in range(0+shift, 4+shift):#len(test_loader.dataset)):

        # Allocate the empty prediction and target matrices
        mat_pred = np.zeros(shape=(40, 40))
        mat_target = np.zeros(shape=(40, 40))

        # Put the data inside
        index = 0+1600*patch_id
        for lon in range(40):
            for lat in range(40):
                mat_pred[lat, lon] = outputs[index].item()
                mat_target[lat, lon] = batch.y[index].item()
                index += 1

        ax_pred = axs[0, col].matshow(mat_pred)
        ax_target = axs[1, col].matshow(mat_target)
        col += 1
        fig.colorbar(ax_pred, orientation='vertical', format='%.0e')#location='top'
        fig.colorbar(ax_target, orientation='vertical')#location='top'

    #plt.subplots_adjust(hspace=0.3, wspace=0.3)    # 5x2 setup
    plt.subplots_adjust(hspace=0.1, wspace=0.3)    # 2x5 setup
    if ON_CLUSTER:
        plt.savefig('./images/testbatch4.png')
        plt.close(fig)
    else:
        plt.show()
    
    #print(test_set[0].y, '\t', np.shape(test_set[0].y))
    print("Pred:\t", outputs[:], '\n\t', np.shape(outputs[:]))
    print("Target:\t", batch.y[:], '\n\t', np.shape(batch.y[:]))

visual()

#%%
# Test from here: https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html#full-implementation
'''
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
'''
# %%
