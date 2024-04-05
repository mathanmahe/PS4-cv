import glob
import os
from typing import Tuple

import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler


def compute_mean_and_std(dir_name: str) -> Tuple[np.ndarray, np.array]:
    """
    Compute the mean and the standard deviation of the pixel values in the dataset.

    Note: convert the image in grayscale and then scale to [0,1] before computing
    mean and standard deviation

    Hints: use StandardScalar (check import statement)

    Args:
    -   dir_name: the path of the root dir
    Returns:
    -   mean: mean value of the dataset (np.array containing a scalar value)
    -   std: standard deviation of th dataset (np.array containing a scalar value)
    """

    mean = None
    std = None

    pixel_values = []
    labels = []

    ############################################################################
    # Student code begin
    ############################################################################
    train_directory = os.path.join(dir_name, "train")
    test_directory = os.path.join(dir_name, "test")
    directories = [train_directory, test_directory]
    print(dir_name)
    # print(os.path.listdir(dir_name))
    for directory in directories:
        if os.path.exists(directory) and os.path.isdir(directory):
            for folder_name in os.listdir(directory):
                folder_path = os.path.join(directory, folder_name)
                if os.path.isdir(folder_path):
                    for image_name in os.listdir(folder_path):
                        if image_name.endswith('.jpg') or image_name.endswith('.png'):
                            image_path = os.path.join(folder_path, image_name)
                            image = Image.open(image_path).convert("L")
                            image_arr = np.array(image) / 255.0

                            # image_arr = np.array(image.convert("L")) / 255.0
                            flattened_arr = image_arr.flatten()
                            pixel_values.extend(flattened_arr)
                            labels.append(folder_name)

    # this reshape is required because StandardScaler expects 2d data
    pixel_values = np.array(pixel_values).reshape(-1,1)
    labels = np.array(labels)
    
    scaler = StandardScaler()
    scaler.fit(pixel_values)
    mean = scaler.mean_
    std = scaler.scale_

    ############################################################################
    # Student code end
    ############################################################################
    return mean, std
