import numpy as np
from numba import njit,prange
import timeit
import torch
# Function without JIT
# The cumulative charge calculates the magnitude of charge passed at a time step
# We are calculating magnitude of charge using integration ∫|i|dt
# The numerical formula used for integration is for simple trapezium formed using the magnitude of charges
def cumulative_integrate_no_jit(current, time):
    charge_passed = np.zeros_like(current)
    charge_passed[0] = np.abs(current[0])*time[0] # Rectangular approximation for the first time step
    for i in range(1, len(current)):
        charge_passed[i] = 0.5 * (np.abs(current[i]) + np.abs(current[i-1])) * (time[i] - time[i-1]) #numerical formula used is that for area of trapezium
    return charge_passed

# Function with JIT (numba), The function with jit is faster
# The cumulative charge calculates the magnitude of charge passed at a time step
# We are calculating magnitude of charge using integration ∫|i|dt
# The numerical formula used for integration is for simple trapezium formed using the magnitude of charges
@njit(parallel=True)
def cumulative_integrate_with_jit(current, time):
    charge_passed = np.zeros_like(current)
    charge_passed[0] = np.abs(current[0])*time[0] # Rectangular approximation for the first time step
    for i in prange(1, len(current)):
        charge_passed[i] = 0.5 * (np.abs(current[i]) + np.abs(current[i-1])) * (time[i] - time[i-1]) #numerical formula used is that for area of trapezium
    return charge_passed

def generate_indices(total_size):
    # Generate all indices
    all_indices = torch.arange(total_size)

    # Select odd indices for training
    train_indices = all_indices[all_indices % 2 == 1]

    # Select even indices for validation and testing
    even_indices = all_indices[all_indices % 2 == 0]

    # Select half of the even indices for validation and testing
    valid_indices = even_indices[even_indices % 4 == 0]  # Half of the even indices (0, 4, 8, ...)
    test_indices = even_indices[even_indices % 4 == 2]   # Other half of the even indices (2, 6, 10, ...)

    return train_indices, valid_indices, test_indices


## Calculates the MAPE per dimension
def calculate_dimensional_mape(target, prediction):
    """
    Calculate the Mean Absolute Percentage Error (MAPE) for each of the three dimensions.

    Parameters:
    target (torch.Tensor): The target tensor with shape (x, y, 3)
    prediction (torch.Tensor): The prediction tensor with shape (x, y, 3)

    Returns:
    tuple: A tuple containing the MAPE for each of the three dimensions
    """
    assert target.shape == prediction.shape, "Target and prediction must have the same shape"

    # Avoid division by zero
    target, prediction = target + 1e-8, prediction + 1e-8

    # Calculate MAPE for each dimension
    mape1 = torch.mean(torch.abs((target[:, :, 0] - prediction[:, :, 0]) / target[:, :, 0])) * 100
    mape2 = torch.mean(torch.abs((target[:, :, 1] - prediction[:, :, 1]) / target[:, :, 1])) * 100
    return mape1.item(), mape2.item()
