import os
import torch
from torch_geometric.data import InMemoryDataset            # For data fitting in my RAM
import xarray as xr

class PilotDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        '''
        root = Where the dataset should be stored and divided into processed/ and raw/
        '''
        super(PilotDataset, self).__init__(root, transform, pre_transform)
        #self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    # Return a list of raw, unprocessed file names
    def raw_file_names(self):
        return ['./pilot_data.nc']

    @property
    # Return a list of files in the processed/ folder. These files need to be found in order to skip the processing.
    # After process(), the returned list should have the only processed data file name
    def processed_file_names(self):
        return ['./not_implemented.pt']

    # Download the raw data into raw/, or the folder specified in self.raw_dir
    def download(self):
        pass

    # Process raw data and save it into the processed/
    def process(self):
        self.data = xr.open_dataset(self.raw_paths[0])

        # Get node features
        node_feats = self._get_node_features()

        # Get edge features
        edge_feats = self._get_edge_features()

        # Get adjacency info 
        edge_index = self._get_adjacency_info()

        # Get labels info
        label = self._get_labels()

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
        print(self.data)

    def _get_edge_features(self):
        pass

    def _get_adjacency_info(self):
        pass

    def _get_labels(self):
        pass