import cupy as cp
import numpy as np
import time

for q in range(21):
    start_time = time.time()

    for w in range(5):

        # Define the number of GPUs to use
        num_gpus = 13  # Change this to match your hardware configuration

        # Define the size of the random number arrays
        N = 2000000000

        mempool = cp.get_default_memory_pool()

        # Create a list to hold the random number arrays for 
        host_array = np.zeros(num_gpus, dtype=np.float32)

        for q in range(10):
            #absorbed_flux_per_gpu = cp.zeros(num_gpus, dtype=cp.float32)

         # Initialize random generators on each GPU
            for i in range(num_gpus):
                with cp.cuda.Device(i):
                    gen = cp.random.Generator(cp.random.XORWOW(1234+i))
                    p = -cp.log(gen.random(N, dtype=cp.float32)) * (1.0 / 0.5)
                    #p = -cp.log(cp.random.Generator(cp.random.XORWOW(1234+i)).random(int(1666666666), dtype=cp.float32)) * (1.0 / 0.5)
                    absorbed_flux_per_gpu = cp.sum(p < 1.0)
                   # print(absorbed_flux_per_gpu)
                    host_array[i] = absorbed_flux_per_gpu



            # Sum up the transmitted flux on the host
            transmitted_flux = int(host_array.sum())

            # Update N for the next iteration
            N = int(transmitted_flux / num_gpus)

        cp.cuda.Stream.null.synchronize()

        mempool.free_all_blocks()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(elapsed_time)

#    print(f"Iteration {q + 1}: Host Flux Array = {host_array}, Transmitted Flux = {transmitted_flux}")

# No need to free GPU memory explicitly as it will be handled automatically by CuPy

