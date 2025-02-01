import numpy as np
from numba import cuda
import math
from utils import initialize_cuda_array, get_grid_dim, is_power_of_two

@cuda.jit
def bitonic_sort_kernel(data, j, k):
    """CUDA kernel for bitonic sort"""
    i = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
    ij = i ^ j
    
    if ij > i:
        ascending = (i & k) == 0
        ai = data[i]
        bi = data[ij]
        
        if (ai > bi) == ascending:
            data[i] = bi
            data[ij] = ai

class CudaBitonicSort:
    def __init__(self, block_size=256):
        self.block_size = block_size

    def sort(self, arr):
        """Sort array using CUDA bitonic sort"""
        n = len(arr)
        if not is_power_of_two(n):
            raise ValueError("Array size must be a power of 2")
        
        # Initialize device array
        d_arr = initialize_cuda_array(arr)
        
        # Perform bitonic sort
        k = 2
        while k <= n:
            j = k // 2
            while j > 0:
                num_blocks = get_grid_dim(n, self.block_size)
                bitonic_sort_kernel[num_blocks, self.block_size](d_arr, j, k)
                j //= 2
            k *= 2
        
        # Copy result back to host
        result = d_arr.copy_to_host()
        return result

if __name__ == "__main__":
    # Test bitonic sort
    size = 1024  # Must be power of 2
    data = np.random.randint(0, 1000, size=size, dtype=np.int32)
    
    sorter = CudaBitonicSort()
    result = sorter.sort(data)
    
    # Verify result
    expected = np.sort(data)
    assert np.array_equal(result, expected), "Bitonic sort failed!"
    print("Bitonic sort successful!")