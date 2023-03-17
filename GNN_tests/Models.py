import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GraphUNet
from torch_geometric.utils import dropout_edge     # for torch-geometric 2.2.0
from torch_geometric.utils import dropout_adj       # for torch-geometric 2.0.3

# A simple 2-layer GCN model using GCNConv
class GCNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, data):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    # This is called 20 times for a train() call.
    # Called once for a test() if _test_batch_size=5, called 5 times if _test_batch_size=1
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return torch.sigmoid(x)#F.log_softmax(x, dim=1)
        

class GUNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_nodes, final_act):
        super().__init__()

        self.act_middle = torch.nn.functional.relu
        # activation = torch.nn.Linear(1, 1); activation(x)
        # torch.sigmoid
        # torch.nn.Sigmoid
        # F.log_softmax(x, dim=1)     # original version
        # torch.nn.Linear(1, 1)
        self.act_final = final_act


        pool_ratios = [2000 / num_nodes, 0.5]
        self.unet = GraphUNet(in_channels, hidden_channels, out_channels,
                              depth=3, pool_ratios=pool_ratios, act=self.act_middle)
        
        self._log_network()
        
    def forward(self, data):
        edge_index, _ = dropout_edge(data.edge_index, p=0.2,
                                     force_undirected=True,
                                     training=self.training)
        
        x = F.dropout(data.x, p=0.92, training=self.training)
        x = self.unet(x, edge_index)

        return self.act_final(x)

    def _log_network(self):
        middle = self.act_middle.__name__
        final = self.act_final.__module__
        print(f"GUNet instantiated!\n\tMiddle act: {middle}\n\tFinal act: {final}")
