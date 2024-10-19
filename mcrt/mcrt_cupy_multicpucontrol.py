import cupy as cp
import numpy as np
import threading
import time

# Define the number 
num_gpus = 11  # Change this to match your hardware configuration

for q in range(21):

    start_time = time.time()

    for w in range(5):

        N = 2000000000

        mempool = cp.get_default_memory_pool()

        # Create a lock to synchronize access to the shared variables
        lock = threading.Lock()

        # Function to be executed by each thread
        def generateRandomNumbersOnGPU(gpu_index, host_array, N):
            with cp.cuda.Device(gpu_index):
                gen = cp.random.Generator(cp.random.XORWOW(1234 + gpu_index))
                p = -cp.log(gen.random(N, dtype=cp.float32)) * (1.0 / 0.5)
                absorbed_flux_per_gpu = cp.sum(p < 1.0)
                with lock:
                    host_array[gpu_index] = absorbed_flux_per_gpu

        # Create a list to hold the random number arrays for 
        host_array = np.zeros(num_gpus, dtype=np.float32)

        for q in range(10):
            # Create a list to hold the threads
            threads = []

            # Initialize random generators on each GPU using threads
            for i in range(num_gpus):
                thread = threading.Thread(target=generateRandomNumbersOnGPU, args=(i, host_array, N))
                threads.append(thread)
                thread.start()

            # Wait for all threads to finish
            for thread in threads:
                thread.join()

            # Sum up the transmitted flux on the host
            transmitted_flux = int(host_array.sum())

            # Update N for the next iteration
            N = int(transmitted_flux / num_gpus)

            #print(transmitted_flux)

        mempool.free_all_blocks()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(elapsed_time)

