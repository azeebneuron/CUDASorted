import numpy as np
import time
from merge_sort import CudaMergeSort
from bitonic_sort import CudaBitonicSort
from utils import pad_to_power_of_two, unpad_array, is_power_of_two
from visualizer import SortVisualizer, LiveVisualizer
import matplotlib.pyplot as plt

class SortingBenchmark:
    def __init__(self):
        self.sorters = {
            'Merge Sort': CudaMergeSort(),
            'Bitonic Sort': CudaBitonicSort(),
        }
        
    def run_single_test(self, data, algorithm, visualize=False):
        """Run a single sorting test with optional visualization"""
        original_size = len(data)
        
        # Pad array if necessary for bitonic sort
        if algorithm == 'Bitonic Sort':
            data, original_size = pad_to_power_of_two(data)
            
        # Create visualizer if requested
        if visualize:
            visualizer = SortVisualizer(data, f"{algorithm} Visualization")
        
        # Sort the array
        sorter = self.sorters[algorithm]
        start_time = time.time()
        
        if visualize:
            # For visualization, we'll need to copy intermediate states
            # This is a simplified version - you might want to modify the sorter
            # classes to yield intermediate states for more accurate visualization
            result = sorter.sort(data.copy())
            visualizer.add_frame(result)
            visualizer.save_animation(f"{algorithm.lower().replace(' ', '_')}_animation.gif")
        else:
            result = sorter.sort(data.copy())
            
        elapsed_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Unpad if necessary
        if algorithm == 'Bitonic Sort':
            result = unpad_array(result, original_size)
            
        return result, elapsed_time
        
    def run_benchmark(self, sizes=[1000, 5000, 10000, 50000]):
        """Run benchmark for all algorithms with different sizes"""
        results = {size: {} for size in sizes}
        
        for size in sizes:
            print(f"\nTesting with array size: {size}")
            data = np.random.randint(-1000, 1000, size=size, dtype=np.int32)
            
            for algorithm in self.sorters.keys():
                try:
                    _, time_taken = self.run_single_test(data.copy(), algorithm)
                    results[size][algorithm] = time_taken
                    print(f"{algorithm}: {time_taken:.2f}ms")
                except Exception as e:
                    print(f"Error in {algorithm}: {str(e)}")
                    results[size][algorithm] = None
                    
        return results
        
    def plot_results(self, results):
        """Plot benchmark results"""
        sizes = list(results.keys())
        algorithms = list(self.sorters.keys())
        
        plt.figure(figsize=(12, 6))
        for algorithm in algorithms:
            times = [results[size][algorithm] for size in sizes]
            plt.plot(sizes, times, marker='o', label=algorithm)
            
        plt.xlabel('Array Size')
        plt.ylabel('Time (ms)')
        plt.title('Sorting Algorithm Performance Comparison')
        plt.legend()
        plt.grid(True)
        plt.savefig('sorting_benchmark.png')
        plt.close()

def main():
    benchmark = SortingBenchmark()
    
    # Run visualization example with small array
    print("Running visualization example...")
    small_data = np.random.randint(-100, 100, size=50, dtype=np.int32)
    benchmark.run_single_test(small_data, 'Merge Sort', visualize=True)
    
    # Run performance benchmark
    print("\nRunning performance benchmark...")
    results = benchmark.run_benchmark()
    
    # Plot results
    benchmark.plot_results(results)
    print("\nBenchmark complete! Check sorting_benchmark.png for results.")

if __name__ == "__main__":
    main()