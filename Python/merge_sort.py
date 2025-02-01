import numpy as np
from numba import cuda
import math
from utils import initialize_cuda_array, get_grid_dim

@cuda.jit
def merge_kernel(input_arr, output_arr, width, n):
    """CUDA kernel for merging sorted sequences"""
    tid = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
    
    # Calculate the starting indices for this thread
    left = tid * 2 * width
    if left >= n:
        return
        
    right = min(left + 2 * width, n)
    mid = min(left + width, right)
    
    # Merge the two sorted sequences
    i = left
    j = mid
    k = left
    
    while i < mid and j < right:
        if input_arr[i] <= input_arr[j]:
            output_arr[k] = input_arr[i]
            i += 1
        else:
            output_arr[k] = input_arr[j]
            j += 1
        k += 1
    
    while i < mid:
        output_arr[k] = input_arr[i]
        i += 1
        k += 1
        
    while j < right:
        output_arr[k] = input_arr[j]
        j += 1
        k += 1

class CudaMergeSort:
    def __init__(self, block_size=256):
        self.block_size = block_size

    def sort(self, arr):
        """Sort array using CUDA merge sort"""
        n = len(arr)
        
        # Initialize device arrays
        d_arr = initialize_cuda_array(arr)
        d_temp = cuda.device_array_like(d_arr)
        
        # Perform merge sort
        width = 1
        while width < n:
            num_blocks = get_grid_dim(n, self.block_size * 2 * width)
            
            merge_kernel[num_blocks, self.block_size](
                d_arr, d_temp, width, n
            )
            
            # Swap input and temp arrays
            d_arr, d_temp = d_temp, d_arr
            width *= 2
        
        # Copy result back to host
        result = d_arr.copy_to_host()
        return result

if __name__ == "__main__":
    # Test merge sort
    size = 1024  # Must be power of 2
    data = np.random.randint(0, 1000, size=size, dtype=np.int32)
    
    sorter = CudaMergeSort()
    result = sorter.sort(data)
    
    # Verify result
    expected = np.sort(data)
    assert np.array_equal(result, expected), "Merge sort failed!"
    print("Merge sort successful!")