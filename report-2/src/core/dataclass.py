import torch
import numpy as np
from .dataLoader import DataLoader
from torch.utils.data.dataset import Dataset
import torchvision as tv

class Dataclass(Dataset):

    def __init__(self, data_dir, cwd, numParticles, loadProcessed=False, saveProcessed=False,):
        super().__init__()

        # Load in dataset
        dataLoader = DataLoader(data_dir, cwd, numParticles, loadProcessed=loadProcessed, saveProcessed=saveProcessed)
        dataset = dataLoader.getDataset(asNumpy=True)

        # Apply transformations to dataloader
        dataset = self.scale_dataset(dataset)
        self.dataset = torch.from_numpy(dataset).to(torch.float32)

        print(self.dataset.shape)

    def scale_dataset(self, dataset, scaled_min=-1, scaled_max=1,):
        """
        Function which will scale input numpy array of tf tensor within the defined range
        """
        # Check scale range is valid
        if not scaled_min < scaled_max:
            raise Exception(f"Scale range is not valid. Minimum value is larger than maximum value in range.")

        # Find the current minimim and maximum value of dataset
        if isinstance(dataset, np.ndarray):
            current_min, current_max = dataset.min(), dataset.max()
        else:
            raise Exception("Error")
        # Scale the data
        scaled_dataset = (scaled_max - scaled_min) * (dataset - current_min)/(current_max - current_min) + scaled_min
        
        # # Print check
        # print(f"Expected dataset scaling is [{int(scaled_min)}, {(scaled_max)}]")
        # print(f"Actual dataset scaling is [{int(scaled_dataset.min())}, {(scaled_dataset.max())}]\n")
        
        # Return a tensorflow tensor or numpy array
        return scaled_dataset

    def __getitem__(self, idx):
        """
        Return a new data point in dataset
        """  

        return self.dataset[idx]

    def __len__(self):
        return self.dataset.shape[0]
