import numpy as np
import os
import torch
from torch_geometric.data import InMemoryDataset            # For data fitting in my RAM
import xarray as xr

class PilotDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored and divided into processed/ and raw/
        """
        super(PilotDataset, self).__init__(root, transform, pre_transform)
        #self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    # Return a list of raw, unprocessed file names
    def raw_file_names(self):
        return ['./pilot_mesh.nc', './pilot_ssh.nc', './pilot_unod.nc', './pilot_vnod.nc']

    @property
    # Return a list of files in the processed/ folder. These files need to be found in order to skip the processing.
    # After process(), the returned list should have the only processed data file name
    def processed_file_names(self):
        return ['./not_implemented.pt']

    # Download the raw data into raw/, or the folder specified in self.raw_dir
    def download(self):
        pass

    # Process raw data and save it into the processed/
    # This function is triggered as soon as the PilotDataset is instantiated
    def process(self):
        print("Processing...")

        # Get node features
        node_feats = self._get_node_features()
        print("...node features collected...")

        # Get edge features
        edge_feats = self._get_edge_features()
        print("...edge features collected...")

        # Get adjacency info 
        edge_index = self._get_adjacency_info()
        print("...adjacency info collected...")

        # Get labels info
        label = self._get_labels()
        print("...labels collected.")

        # Create the Data object
#        data = Data(x=node_feats,               # node feature matrix
#                    edge_index=edge_index,      # graph connectivity in COO format
#                    edge_attr=edge_feats,       # edge feature matrix
#                    y=label,                    # graph or node targets
#                    smiles=
#                    )

        # The collated data object has concatenated all examples into one big data object and returns a "slices" dictionary
        # to reconstruct single examples from this object
        #data, slices = self.collate(data_list)

        # Load data and slices in the constructor into the properties "self.data" and "self.slices"
        #torch.save((data, slices), self.processed_paths[0])       
        #torch.save(data, os.path.join(self.processed_dir, f'data_{0}.pt'))

    def _get_node_features(self):
        """
        This will return a matrix, nodes per features
        """
        self.data_mesh = xr.open_dataset(self.raw_paths[0])
        self.data_unod = xr.open_dataarray(self.raw_paths[2])

        _node_feats = []
        # TODO iterate over the nodes and append stuff into the list. the guy in the video has 

        _node_feats = np.asarray(_node_feats)
        return torch.tensor(_node_feats, dtype=torch.float)

    def _get_edge_features(self):
        # TODO same thing as before, but iterate over the edges. Not sure if I'll need this
        _edge_feats = []
        _edge_feats = np.asarray(_edge_feats)
        return _edge_feats

    def _get_adjacency_info(self):
        # TODO convert the mesh edges to an adjacency matrix
        N_nodes = self.data_mesh.spatial_subset.size
        adj_matrix = np.zeros((N_nodes, N_nodes), dtype=bool)   # TODO: too big in RAM, an adjacency list should be much more efficient
        
        # Let's put some random connections in here
        adj_matrix[0][0] = True
        adj_matrix[10][10] = True
        adj_matrix[100][100] = True

        row, col = np.where(adj_matrix)
        coo = np.array(list(zip(row, col)))                     # conversion to COO format
        coo = np.reshape(coo, (2, -1))                          # no idea why these numbers
        return torch.tensor(coo, dtype=torch.long)              # it works even here

    def _get_labels(self):
        pass