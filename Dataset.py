import torch
from torch_geometric.data import InMemoryDataset            # For data fitting in my RAM

class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(MyOwnDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    # Returns a list of raw, unprocessed file names
    def raw_file_names(self):
        raw_files = ['./raw_file.nc']
        return raw_files

    @property
    # Returns a list of processed file names. After process(), the returned list should have the only processed data file name
    def processed_file_names(self):
        processed_files = ['data.pt']
        return processed_files

    # Download the data to the working directory specified in self.raw_dir
    def download(self):
        pass

    # Gather your data into a list of Data objects
    def process(self):
        # Read data into huge `Data` list.
        data_list = [...]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])       
