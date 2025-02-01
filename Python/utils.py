import numpy as np
from numba import cuda
import math

def is_power_of_two(n):
    """Check if a number is power of 2"""
    return n > 0 and (n & (n - 1)) == 0

def initialize_cuda_array(host_array):
    """Initialize CUDA array from host array"""
    device_array = cuda.to_device(host_array)
    return device_array

def verify_sort(arr):
    """Verify if array is sorted"""
    return np.all(arr[:-1] <= arr[1:])

def pad_to_power_of_two(arr):
    """Pad array to next power of 2 with infinity"""
    original_size = len(arr)
    next_power_of_two = 1 << (original_size - 1).bit_length()
    if next_power_of_two == original_size:
        return arr, original_size
    
    padded_arr = np.full(next_power_of_two, np.inf, dtype=arr.dtype)
    padded_arr[:original_size] = arr
    return padded_arr, original_size

def unpad_array(arr, original_size):
    """Remove padding and return original size array"""
    return arr[:original_size]

def initialize_cuda_array(host_array):
    """Initialize CUDA array from host array"""
    device_array = cuda.to_device(host_array)
    return device_array

def verify_sort(arr):
    """Verify if array is sorted"""
    return np.all(arr[:-1] <= arr[1:])

def get_grid_dim(size, block_size):
    """Calculate grid dimensions based on data size and block size"""
    return (size + block_size - 1) // block_size