import cupy as cp
import numpy as np
import time
import threading

# Define the CUDA kernel with atomic add
kernel_code = '''
#include <curand_kernel.h>

extern "C" __global__
void generate(unsigned long long *output, unsigned long long N, int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    curandState state;
    curand_init(seed, tid, 0, &state);

    unsigned long long count = 0;
    for (unsigned long long i = tid; i < N; i += stride) {
        float q = curand_uniform(&state);
        float distance_to_collision = -__logf(q) * 2.0f;  // Using fast math for log
        if (distance_to_collision < 1.0f) {
            count++;
        }
    }
    atomicAdd(output, count);
}
'''

# Compile the kernel with fast math options
module = cp.RawModule(code=kernel_code, options=('--use_fast_math',))
#module = cp.RawModule(code=kernel_code)
generate_kernel = module.get_function('generate')

#num_gpus = 2
#N = 5000000000

def run_on_gpu(gpu_id, N, seed, result_array):
    with cp.cuda.Device(gpu_id):
        d_p = cp.zeros(1, dtype=cp.uint64)
        block_size = 256
        grid_size = 320

        generate_kernel((grid_size,), (block_size,), (d_p, N, seed))
        cp.cuda.Stream.null.synchronize()

        result_array[gpu_id] = d_p.get()

for q in range(21):
    start_time = time.time()

    for w in range(1):

        num_gpus = 10
        N = 10000000000

        mempool = cp.get_default_memory_pool()
        host_array = np.zeros(num_gpus, dtype=np.uint64)

        for q in range(10):
            threads = []
            for i in range(num_gpus):
                seed = 1234 + i
                thread = threading.Thread(target=run_on_gpu, args=(i, N, seed, host_array))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            transmitted_flux = int(host_array.sum())
            #print(f"Iteration {q}: Transmitted Flux = {transmitted_flux}")  # Debugging statement

            if transmitted_flux == 0:
                break

            N = int(transmitted_flux / num_gpus)

        cp.cuda.Stream.null.synchronize()
        mempool.free_all_blocks()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(elapsed_time)

