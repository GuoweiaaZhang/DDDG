# -*- coding: utf-8 -*-
"""
This script provides a data loading and preprocessing function for fault diagnosis tasks using 1D signals stored in MATLAB `.mat` format.

Main functionalities:
1. Load data from a specified `.mat` file and key.
2. Optionally remove samples of specified classes (e.g., for class exclusion or label imbalance handling).
3. Split data into training and testing sets based on a fixed number of samples per class.
4. Apply Fast Fourier Transform (FFT) to convert time-domain signals into frequency-domain features.
5. Normalize features to the [0, 1] range using training set statistics.
6. Convert all data into PyTorch FloatTensors for further processing.

Function:
- `load_data(path, temp, com_class, class_sample, mis_class)`
    - `path`: Path to the `.mat` file.
    - `temp`: Key name used to extract the dataset from the file.
    - `com_class`: Total number of categories.
    - `class_sample`: Number of samples per class for training.
    - `mis_class`: A list of class labels to exclude from the dataset.

Returns:
    - `train_x`, `train_y`: Training features and labels.
    - `test_x`, `test_y`: Testing features and labels.
"""

import torch
import scipy.io as scio
import numpy as np
from scipy.fftpack import fft

def load_data(path, temp, com_class, class_sample, mis_class):
    # 1. Load the .mat data file and get the data for the specified key
    mat_data = scio.loadmat(path)
    data = mat_data.get(temp)
    if data is None:
        raise KeyError(f"Key '{temp}' not found in the MATLAB file.")

    n_class = com_class - len(mis_class)  # Number of valid categories (total categories minus excluded categories)

    # 2. Delete rows of category data to be excluded
    if len(mis_class) > 0:
        labels = data[:, 0, -1] if data.ndim == 3 else data[..., -1]
        mask = ~np.isin(labels, mis_class)
        data = data[mask] 

    # 3. Divide the training and test set data
    train_count = class_sample * n_class
    train_data = data[:train_count]
    test_data = data[train_count:]
    # Separate datas and labels
    train_x = train_data[..., :-1]  # All feature columns
    train_y = train_data[..., -1]  # labels
    test_x = test_data[..., :-1]
    test_y = test_data[..., -1]

    if train_y.ndim > 1:
        train_y = train_y[:, 0]
    if test_y.ndim > 1:
        test_y = test_y[:, 0]

    # 4. Define data preprocessing function: FFT transform and intercept half of the spectrum
    def transform_signal(x):       
        X_fft = 2 * np.abs(fft(x).real) / x.shape[-1]
        X_fft = X_fft[..., :x.shape[-1] // 2]
        return X_fft

    # 5. FFT transformation of training and test set features
    train_x = transform_signal(train_x)
    test_x = transform_signal(test_x)

    # 6. Feature normalization: scaling features to [0, 1] using the minimum and maximum values of the training set
    min_val = train_x.min()
    max_val = train_x.max()
    train_x = (train_x - min_val) / (max_val - min_val)
    test_x = (test_x - min_val) / (max_val - min_val)

    # 7. Convert to PyTorch's FloatTensor type
    train_x = torch.FloatTensor(train_x)
    train_y = torch.FloatTensor(train_y)
    test_x = torch.FloatTensor(test_x)
    test_y = torch.FloatTensor(test_y)

    return train_x, train_y, test_x, test_y



