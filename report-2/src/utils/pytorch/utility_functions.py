import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

def compute_anomaly_scores(data, network, anomaly_score_function, config):
    """
    Use a trained network to compute the anomaly score of a given dataset
    """ 

    # Define device to send
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instansiate dataloader
    if not isinstance(data, Dataset):
        raise Exception("Dataset passed to function is not a torch Dataset class")
    loader = DataLoader(data, batch_size=1, shuffle=False)

    # Set network to eval mode and send to device
    network = network.eval().to(device)

    # Define empty list to contain anomaly scores
    anomaly_socres = []

    # Iterate through dataset and compute anomaly score
    with torch.no_grad():
        for input in iter(loader):
            # Send input to device
            input = input.to(device)

            if config["trainer"]["model_name"] == "denseVAE":
                # Compute model output and compute anomaly score
                output, mean, logvar = network(input)
                anomaly_socre = anomaly_score_function(input, output, mean, logvar).mean().item()
            else:
                # Compute model output and compute anomaly score
                output = network(input)
                anomaly_socre = anomaly_score_function(input, output).mean().item()
            
            # Append anomaly score to list
            anomaly_socres.append(anomaly_socre)
    
    # Return anomaly scores
    return anomaly_socres

