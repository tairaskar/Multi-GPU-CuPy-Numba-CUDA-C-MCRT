import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import threading
import time

@cuda.jit
def gpu_random_numbers(rng_states, out, N):
    tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    stride = cuda.blockDim.x * cuda.gridDim.x
    for i in range(tid, N, stride):
        out[i] = cuda.random.xoroshiro128p_uniform_float32(rng_states, tid)

def generate_random_numbers_on_gpus(gpu_id, num_gpus, num_numbers_per_gpu, result_per_gpu):
    block_size = 256
    grid_size = 128

    cuda.select_device(gpu_id)

    rng_states = create_xoroshiro128p_states(grid_size * block_size, seed=1234+gpu_id)
    d_out = cuda.device_array(num_numbers_per_gpu, dtype=np.float32)
    gpu_random_numbers[grid_size, block_size](rng_states, d_out, num_numbers_per_gpu)
    result_per_gpu[gpu_id] = d_out

def run_on_gpu(num_gpus, num_numbers_per_gpu, result_per_gpu):
    num_threads = num_gpus
    threads = []

    for i in range(num_threads):
        thread = threading.Thread(target=generate_random_numbers_on_gpus,
                                  args=(i, num_gpus, num_numbers_per_gpu, result_per_gpu))
        threads.append(thread)

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

num_gpus = 11
num_numbers_per_gpu = 8000000000

for q in range(21):
    start_time = time.time()
    for w in range(1):
        result_per_gpu = [None] * num_gpus
        run_on_gpu(num_gpus, num_numbers_per_gpu, result_per_gpu)
        cuda.synchronize()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(elapsed_time)

