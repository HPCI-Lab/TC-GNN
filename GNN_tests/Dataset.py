import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset
import xarray as xr

from utils import time_func

class PilotDataset(Dataset):

    # root: Where the dataset should be stored and divided into processed/ and raw/
    def __init__(self, root, label_type, transform=None, pre_transform=None, pre_filter=None):
        self.label_type = label_type
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    # If you directly return the names of the files, the '.' will be in /data/bsc/raw
    # If you return os.listdir, the '.' will be where "Dataset.py" is
    def raw_file_names(self):
        return os.listdir(self.root + '/raw')
        #['./ERA5_test.nc']

    @property
    # Return a list of files in the processed/ folder.
    # If these files don't exist, process() will start and create them.
    # If these files exist, process() will be skipped.
    # After process(), the returned list should have the only processed data file name
    def processed_file_names(self):
        return os.listdir(self.root + '/processed')
        #['ERA5_test_ibtracs_int_0.pt']
    
    # Process raw data and save it into the processed/
    # This function is triggered as soon as the PilotDataset is instantiated
    def process(self):
        
        # Conserving adjacency info to avoid computing it every time
        edge_index = None

        for raw_path in self.raw_paths:

            year = raw_path.split('_')[2]
            cyclone = raw_path.split('_')[4].split('.')[0]
            #print(f'    Year {year}, Patch number {cyclone}...')
            raw_data = xr.open_dataset(raw_path)

            # Get node features
            node_feats = self._get_node_features(raw_data)

            # Get edge features - for our task we don't need this
            #edge_feats = self._get_edge_features(raw_data)

            # Get adjacency info
            if edge_index==None:
                edge_index = self._get_adjacency_info(raw_data)

            # Get labels info
            if self.label_type == "binary":
                labels = self._get_labels_binary(raw_data)
            elif self.label_type == "distance":
                labels = self._get_labels_distance(raw_data)
            elif self.label_type == "distanceInverse":
                labels = self._get_labels_distance(raw_data, dist_inverse_01=True)
            else:
                print("LABEL TYPE NOT RECOGNIZED!! Available labels: [distance/binary/distanceInverse]")
                exit(0)

            # Create the Data object
            data = Data(
                x=node_feats,                       # node features
                edge_index=edge_index,              # edge connectivity
                y=labels,                           # labels for classification
            )

            #print(os.path.join(self.processed_dir, f'year_{year}_cyclone_{cyclone}.pt'))
            torch.save(data, os.path.join(self.processed_dir, f'year_{year}_cyclone_{cyclone}.pt'))

        print("    Shape of node feature matrix:", np.shape(node_feats))
        print("    Shape of graph connectivity in COO format:", np.shape(edge_index))
        print("    Shape of labels:", np.shape(labels))

    # This will return a matrix with shape=[num_nodes, num_node_features]
    #   nodes: the geographic locations
    #   features: the ERA5 variables for each cell
    def _get_node_features(self, data):

        all_nodes_feats =[]

        # Remove dimension with length=1, which is "time"
        data = data.squeeze()

        # Extract the list of ERA5 variables
        ERA5_vars = []
        for key in data.data_vars:
            if key!='Ymsk':   # TODO: check the variable name in the final dataset
                ERA5_vars.append(data.data_vars[key].values)   # TODO: talk with cmcc guys to understand if they treat this in some way 

        # Calculate for each cell the distance from the cyclone cell
        #mat_dist = self._get_dist_matrix(data)

        # The order of nodes is implicit in how I perform these lon/lat loops
        for lon in range(data.lon.size):
            for lat in range(data.lat.size):
                node_feats = []
                for variable in ERA5_vars:
                    node_feats.append(variable[lat, lon])
                    # too slow alternative: .append(float(data.msl.isel(time=0, lat=lat, lon=lon).values))
                
                #node_feats.append(mat_dist[lat, lon])
                all_nodes_feats.append(node_feats)

        all_nodes_feats = np.asarray(all_nodes_feats)
        return torch.tensor(all_nodes_feats, dtype=torch.float)
    
    # Return the cyclone distance matrix
    def _get_dist_matrix(self, data, dist_inverse_01=None):
        tmp_ibtracs = data.Ymsk.values
        mat_dist = np.zeros(shape=(40, 40))

        # Find the cyclone cell
        row, col = None, None
        for lon in range(data.lon.size):
            for lat in range(data.lat.size):
                if tmp_ibtracs[lat, lon]==1:
                    row, col = lat, lon

        # Assign the Euclidean distances
        for lon in range(data.lon.size):
            for lat in range(data.lat.size):
                mat_dist[lat, lon] = np.sqrt((lat-row)**2 + (lon-col)**2)

        if dist_inverse_01:
            max_dist = np.sqrt(2)*40
            for lon in range(data.lon.size):
                for lat in range(data.lat.size):
                    mat_dist[lat, lon] = (max_dist - mat_dist[lat, lon]) / max_dist

        return mat_dist


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

        return torch.tensor(coo_links, dtype=torch.long)

    # Here we're gonna put the ibtracs data to classify the nodes
    def _get_labels_binary(self, data):
        labels = []
        tmp_ibtracs = data.Ymsk.values
        
        for lon in range(data.lon.size):
            for lat in range(data.lat.size):
                labels.append(int(tmp_ibtracs[lat, lon]))
        
        return torch.tensor(labels, dtype=torch.float)
    
    # Test setup for regression on distances instead of classification on presence
    def _get_labels_distance(self, data, dist_inverse_01=None):
        labels = []
        mat_dist = self._get_dist_matrix(data, dist_inverse_01)

        for lon in range(data.lon.size):
            for lat in range(data.lat.size):
                labels.append(mat_dist[lat, lon])

        return torch.tensor(labels, dtype=torch.float)

    # Download the raw data into raw/, or the folder specified in self.raw_dir
    def download(self):
        pass

    # Returns the number of examples in the dataset
    def len(self):
        return len(self.processed_file_names)
    
    # Implements the logic to load a single graph
    def get(self, year, cyclone):
        data = torch.load(os.path.join(self.processed_dir, f'year_{year}_cyclone_{cyclone}.pt'))
        return data
