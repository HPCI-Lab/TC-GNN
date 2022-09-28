import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import Dataset

### Example code for the creation of a Graph Neural Network

## Data - The graph is composed by 4 nodes, and each one is associated with a 2-dimensional feature vector, and a label y indicating its class

# These are the feature vectors
x = torch.tensor([[2,1], [5,6], [3,7], [12,0]], dtype=torch.float)

# These are the labels, one for each of the 4 nodes
y = torch.tensor([0, 1, 0, 1], dtype=torch.float)

# Graph connectivity, the first list contains the index of the source nodes while the second contains the index of target nodes
# Since this info is only for computing the adjacency matrix, the order is irrelevant
edge_index = torch.tensor([[0, 1, 2, 0, 3],                             # source nodes
                           [1, 0, 1, 3, 2]], dtype=torch.long)          # target nodes

# Creation of the Data element
data = Data(x=x, y=y, edge_index=edge_index)

print(data)
dataset = Dataset.PilotDataset('../data/pilot/')    # It will automatically search for processed/ and raw/ in here, creating the folders if not found

## DataLoader - allows to feed data by batch into the model. The "batch" information is used by the GNN model to tell which nodes belong to which graph
loader = DataLoader(dataset, batch_size=512, shuffle=True)      # IT BREAKS HERE

for batch in loader:
    batch
