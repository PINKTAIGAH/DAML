import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

def compute_anomaly_scores(data, network, anomaly_score_function, config, batch_size=128):
    """
    Use a trained network to compute the anomaly score of a given dataset
    """ 

    # Define device to send
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instansiate dataloader
    if not isinstance(data, Dataset):
        raise Exception("Dataset passed to function is not a torch Dataset class")
    loader = DataLoader(data, batch_size=batch_size, shuffle=False)

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
                anomaly_socre = anomaly_score_function(input, output, mean, logvar).cpu().tolist()
            else:
                # Compute model output and compute anomaly score
                output = network(input)
                anomaly_socre = anomaly_score_function(input, output).cpu().tolist()
                # print(anomaly_socre)
            # Append all anomaly score to list
            anomaly_socres.extend(anomaly_socre)
    
    # Return anomaly scores as a numpy array
    anomaly_socres = np.array(anomaly_socres)
    return anomaly_socres

def scale_dataset(dataset, scaled_min=-1, scaled_max=1,):
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
        current_min, current_max = dataset.min(), dataset.max()
    # Scale the data
    scaled_dataset = (scaled_max - scaled_min) * (dataset - current_min)/(current_max - current_min) + scaled_min
    
    # # Print check
    # print(f"Expected dataset scaling is [{int(scaled_min)}, {(scaled_max)}]")
    # print(f"Actual dataset scaling is [{int(scaled_dataset.min())}, {(scaled_dataset.max())}]\n")
    
    # Return a tensorflow tensor or numpy array
    return scaled_dataset


def anomaly_function(input, output, mean=None, logvar=None):
    """
    Return the Anomaly function for the output of a Autoencoder
    """
    # Compute each anomaly score for each event in batch
    mseLoss = nn.functional.mse_loss(input, output, reduction="none").mean(axis=1)
    return mseLoss 

def false_positive_rate(sm_scores, bsm_scores,):
    """
    Return the false positive rate of for incorrectly classified anomalies (i.e: sm events classified as bsm).
    Takes anomaly scores for a sm and bsm dataset
    """

    # Find minimum value in bsm anomaly socres. This is used for out classification cutoff 
    # While not the optimum, this is done so that we can compare the performance of different models
    bsm_minimum = bsm_scores.min()
    # Compute number of sm events incorrectly classified as bsm
    count = (sm_scores>bsm_minimum).sum()
    # Return percentage of sm events incorrectly classified
    return (count/sm_scores.size), bsm_minimum