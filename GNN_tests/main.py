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
if torch.cuda.is_available():
    print(f"Cuda device: {torch.cuda.get_device_name()}")
print(f"Torch geometric version: {torch_geometric.__version__}")

### GLOBAL SETTINGS ###
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ON_CLUSTER = False
if ON_CLUSTER:
    DATA_PATH = '../../data/gnn_records/'
else:
    DATA_PATH = './data/bsc_records/'

TRAIN_SET = [1982, 1983]
VALID_SET = [1980]
TEST_SET = [1981]
LABEL_TYPE = "distanceInverse"    # binary(classification), distance(regression) distanceInverse(classification)
_model_name = "GUNet"
_num_features = None
_hidden_channels = 32   # 16 for GCNet, 32 for GUNet(from the examples)
_num_classes = 1
_train_batch_size = 512   # TODO see how the results change if you change this
_test_batch_size = 512
_valid_batch_size = 512
FINAL_ACTIVATION = torch.sigmoid#torch.nn.Linear(1, 1)#
GRAPH_DICT = {}

TIMESTAMP = time_func.start_time()

# %%
# Dataset creation
timestamp = time_func.start_time()
dataset = Dataset.PilotDataset(root=DATA_PATH + LABEL_TYPE, label_type=LABEL_TYPE)
time_func.stop_time(timestamp, "Dataset creation")

# Note that cyclone files start from 1, not from 0
print("Year 1983, cyclone 1: ", dataset.get(1983, 1))
#dataset.get(year=1983, cyclone=1).is_directed()

# %%
# Extract number of cyclones per year
graph_names = os.listdir(dataset.processed_dir)
if 'pre_filter.pt' in graph_names: graph_names.remove('pre_filter.pt')
if 'pre_transform.pt' in graph_names: graph_names.remove('pre_transform.pt')

years = []

for gn in graph_names:
	y = gn.split('_')[1]
	years.append(y)

years = np.unique(years)

for y in years:
	GRAPH_DICT[y] = 0

for gn in graph_names:
	y = gn.split('_')[1]
	c = gn.split('_')[3].split('.')[0]
	if GRAPH_DICT[y] < int(c):
		GRAPH_DICT[y] = int(c)

print("Dictionary of graph years and number of cyclones:\n", GRAPH_DICT)

# %%
# [Temporary] calculation 80-20-20 set for train-valid-test
# TODO: it assigns a fraction of the average per each year. It SHOULD be safe for now, but look up for array oveflows
train_average, valid_total, test_total = 0, 0, 0
for year in TRAIN_SET:
    train_average += GRAPH_DICT[str(year)]

train_average /= len(TRAIN_SET)

valid_total = int(train_average*20/80)
test_total = int(train_average*20/80)

# %%
# Data loader set up + Feature normalization
# The Dataloader allows to feed data by batch into the model to exploit parallelism
# The "batch" information is used by the GNN model to tell which nodes belong to which graph
# "batch_size" specifies how many grids will be saved in a single batch
# TODO: If "shuffle=True", the data will be reshuffled at every epoch. understand
# if this means that you may potentially train over the same patch/batch over and over
#dataset = dataset.shuffle()

# Feature normalization. Fit the training and tranform training, test and validation
scaler = MinMaxScaler()     # default range: [0, 1]

# Fit the scaler to the training set, one piece at a time
for year in TRAIN_SET:
    for cyclone in range(GRAPH_DICT[str(year)]):
        patch = dataset.get(year, cyclone+1)        # Cyclones files start from 1
        scaler.partial_fit(patch.x)

train_set = []
valid_set = []
test_set = []

# Normalize training, test and validation sets with the obtained values
for year in TRAIN_SET:
    for cyclone in range(GRAPH_DICT[str(year)]):
        patch = dataset.get(year, cyclone+1)
        patch.x = torch.tensor(scaler.transform(patch.x), dtype=torch.float)
        train_set.append(patch)

for year in VALID_SET:
    for cyclone in range(valid_total):#GRAPH_DICT[str(year)]):
        patch = dataset.get(year, cyclone+1)
        patch.x = torch.tensor(scaler.transform(patch.x), dtype=torch.float)
        valid_set.append(patch)

for year in TEST_SET:
    for cyclone in range(test_total):#GRAPH_DICT[str(year)]):
        patch = dataset.get(year, cyclone+1)
        patch.x = torch.tensor(scaler.transform(patch.x), dtype=torch.float)
        test_set.append(patch)

train_loader = DataLoader(train_set, batch_size=_train_batch_size, shuffle=True)#, batch_sampler=None)
valid_loader = DataLoader(valid_set, batch_size=_valid_batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=_test_batch_size, shuffle=False)

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
    num_nodes = train_loader.dataset[0].num_nodes,  # just initialization
    final_act = FINAL_ACTIVATION
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

        loss = loss_op(pred, batch.y)

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

#loss = train()
#print("Train loss, debug: ", loss)

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

        pred = model(batch)
        pred = pred.squeeze()

        loss = loss_op(pred, batch.y)

        total_loss += loss.item() * batch.num_graphs
    
    total_loss = total_loss / len(loader.dataset)
    return total_loss

'''
loss = evaluate(train_loader)
print("Train loss, debug: ", loss)
loss = evaluate(valid_loader)
print("Valid loss, debug: ", loss)
loss = evaluate(test_loader)
print("Test loss, debug: ", loss)
'''

time_func.stop_time(TIMESTAMP, "Computation before training finished!")

# %%
# Settings recap
print(f"Type of target: {LABEL_TYPE}")
print(f"Final act: {FINAL_ACTIVATION}")
print(f"Loss function: {loss_op}")

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
#ax = plt.gca()
#ax.yaxis.set_major_formatter(lambda x, pos: f'{abs(x):g}')
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

    pred = model(batch)  # shape: [1600*_test_batch_size, 1]
    pred = pred.squeeze()
    
    # Preparing the plot
    fig, axs = plt.subplots(ncols=2, nrows=5, figsize=(6, 14))
    row = 0

    # Take 4 patches from the batch
    shift = 20
    for patch_id in range(0+shift, 5+shift):#len(test_loader.dataset)):

        # Allocate the empty prediction and target matrices
        mat_pred = np.zeros(shape=(40, 40))
        mat_target = np.zeros(shape=(40, 40))

        # Put the data inside
        index = 0+1600*patch_id
        for lon in range(40):
            for lat in range(40):
                mat_pred[lat, lon] = pred[index].item()
                mat_target[lat, lon] = batch.y[index].item()
                index += 1

        ax_target = axs[row, 0].matshow(mat_target)
        ax_pred = axs[row, 1].matshow(mat_pred)
        row += 1
        fig.colorbar(ax_pred, fraction=0.046, pad=0.04)#, format='%.0e')
        fig.colorbar(ax_target, fraction=0.046, pad=0.04)
    
    plt.subplots_adjust(hspace=0.05, wspace=0.4)    # 5x2 setup
    #plt.subplots_adjust(hspace=0.1, wspace=0.3)    # 2x5 setup
    if ON_CLUSTER:
        plt.savefig('./images/testbatch.png')
        plt.close(fig)
    else:
        plt.show()
    
    #print(test_set[0].y, '\t', np.shape(test_set[0].y))
    print("Pred:\t", pred[:], '\n\t', np.shape(pred[:]))
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
