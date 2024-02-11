import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

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
        current_min, current_max = tf.reduce_min(dataset), tf.reduce_max(dataset)
    # Scale the data
    scaled_dataset = (scaled_max - scaled_min) * (dataset - current_min)/(current_max - current_min) + scaled_min
    
    # # Print check
    # print(f"Expected dataset scaling is [{int(scaled_min)}, {(scaled_max)}]")
    # print(f"Actual dataset scaling is [{int(scaled_dataset.min())}, {(scaled_dataset.max())}]\n")
    
    # Return a tensorflow tensor or numpy array
    return scaled_dataset

def apply_data_transformations(dataset, test_size=0.2, scale_range=(0, 1), shuffle=True, batch_size=1):
    """
    Takes a raw numpy input, applies relevant data transformatios and returns val and train dataset tensors
    """
    # Scale dataset
    scaled_dataset = scale_dataset(dataset, scaled_min=scale_range[0], scaled_max=scale_range[1])
    # Shuffle and split dataset
    train_dataset, val_dataset = train_test_split(scaled_dataset, test_size=test_size, shuffle=shuffle)
    # Convert datasets to tensorflow dataset
    train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices(val_dataset).batch(batch_size)
    
    return train_dataset, val_dataset 