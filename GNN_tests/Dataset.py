import numpy as np
import os
import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset
import xarray as xr

class PilotDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        """
        root = Where the dataset should be stored and divided into processed/ and raw/
        """
        #super(PilotDataset, self).__init__(root, transform, pre_transform)     # from the "GNN hands on" guide, and youtube video
        super().__init__(root, transform, pre_transform, pre_filter)            # from PyG documentation


    @property
    # Return a list of raw, unprocessed file names
    def raw_file_names(self):
        return ['./ERA5_test_0.nc', './ERA5_test_1.nc']


    @property
    # Return a list of files in the processed/ folder.
    # If these files don't exist, process() will start and create them. If these files exist, process() will be skipped.
    # After process(), the returned list should have the only processed data file name
    def processed_file_names(self):
        return ['data_0.pt', 'data_1.pt']


    # Download the raw data into raw/, or the folder specified in self.raw_dir
    def download(self):
        pass


    # Returns the number of examples in the dataset
    def len(self):
        return len(self.processed_file_names)
    

    # Implements the logic to load a single graph
    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data


    # Process raw data and save it into the processed/
    # This function is triggered as soon as the PilotDataset is instantiated
    def process(self):
        idx = 0
        for raw_path in self.raw_paths:

            print(f'Chunk number {idx}...')
            raw_data = xr.open_dataset(raw_path)    # TODO verify if passing the data is sub-optimal

            # Get node features
            node_feats = self._get_node_features(raw_data)

            # Get edge features - for our task we don't need this
            edge_feats = self._get_edge_features(raw_data)

            # Get adjacency info 
            edge_index = self._get_adjacency_info(raw_data)

            # Get labels info
            labels = self._get_labels(raw_data)

            # Create the Data object
            data = Data(
                x=node_feats,                       # node features
                edge_index=edge_index,              # edge connectivity
                y=labels,                           # labels for classification
            )

            # The collated data object has concatenated all examples into one big data object and returns a "slices" dictionary
            # to reconstruct single examples from this object
            #data, slices = self.collate(data_list)

            # Load data and slices in the constructor into the properties "self.data" and "self.slices"
            #torch.save((data, slices), self.processed_paths[0])       
            torch.save(data, os.path.join(self.processed_dir, f'data_{idx}.pt'))
            idx += 1


    # This will return a matrix with shape=[num_nodes, num_node_features]
    #   nodes: the geographic locations
    #   features: the ERA5 variables
    def _get_node_features(self, data):

        all_nodes_feats =[]

        # Extract the list of ERA5 variables
        ERA5_vars = []
        for key in data.data_vars:
            if key!='ibtracs':      # TODO: check the variable name in the final dataset
                ERA5_vars.append(data.data_vars[key].values)   # TODO: talk with cmcc guys to understand if they treat this in some way 

        # The order of nodes is implicit in how I perform these lon/lat loops
        for lon in range(data.lon.size):
            for lat in range(data.lat.size):
                node_feats = []
                for variable in ERA5_vars:
                    node_feats.append(variable[0, lat, lon])     # 0 is the timestamp
                    # too slow alternative: .append(float(data.msl.isel(time=0, lat=lat, lon=lon).values))
                
                all_nodes_feats.append(node_feats)

        print("   Shape of node feature matrix:", np.shape(all_nodes_feats))
        all_nodes_feats = np.asarray(all_nodes_feats)
        return torch.tensor(all_nodes_feats, dtype=torch.float)

    
    # We won't need this, since the edges are only useful to connect the spatial locations
    def _get_edge_features(self, data):
        all_edge_feats = []
        all_edge_feats = np.asarray(all_edge_feats)
        return all_edge_feats


    # This has to return the graph connetivity in COO with shape=[2, num_edges]
    # We're gonna skip the creation of an adjacency matrix to save computation time and memory
    def _get_adjacency_info(self, data):
        
        lon_size = data.lon.size
        lat_size = data.lat.size
        coo_links = [[], []]
        this_node = 0

        # The order of nodes is implicit in how I perform these lon/lat loops and in the variable "this_node"
        for lon in range(lon_size):
            for lat in range(lat_size):
                # Check whether a cell below does exist. If so, add the link
                if (lat-1)>=0:
                    coo_links[0].append(this_node)
                    coo_links[1].append(this_node-1)
                # Check whether a cell on the right does exist. If so, add the link
                if (lon+1)<lon_size:
                    coo_links[0].append(this_node)
                    coo_links[1].append(this_node+lat_size)
                # Check whether a cell above does exist. If so, add the link
                if (lat+1)<lat_size:
                    coo_links[0].append(this_node)
                    coo_links[1].append(this_node+1)
                # Check whether a cell on the left does exist. If so, add the link
                if (lon-1)>=0:
                    coo_links[0].append(this_node)
                    coo_links[1].append(this_node-lat_size)
                
                this_node += 1
        '''
        N_nodes = data_mesh.spatial_subset.size
        adj_matrix = np.zeros((N_nodes, N_nodes), dtype=bool)   # TODO: too big in RAM, an adjacency list should be much more efficient
        '''
        print("   Shape of graph connectivity in COO format:", np.shape(coo_links))
        return torch.tensor(coo_links, dtype=torch.long)


    # Here we're gonna put the ibtracs data to classify the nodes
    def _get_labels(self, data):
        
        labels = []
        tmp_ibtracs = data.ibtracs.values
        
        for lon in range(data.lon.size):
            for lat in range(data.lat.size):
                labels.append(tmp_ibtracs[lat, lon])
        
        print("   Shape of labels:", np.shape(labels))
        return torch.tensor(labels, dtype=torch.float)


#%% Creating the dataset 
dataset = Dataset.PilotDataset(root='./data/cmcc_structured')
dataset.get(0).edge_index

# %%
from torch_geometric.loader import DataLoader
loader = DataLoader(dataset, batch_size=512, shuffle=True)
print(loader.dataset)

# %%
