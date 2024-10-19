import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import time

@cuda.jit
def gpu_random_numbers(rng_states, out, N):
    tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    stride = cuda.blockDim.x * cuda.gridDim.x
    for i in range(tid, N, stride):
        out[i] = cuda.random.xoroshiro128p_uniform_float32(rng_states, tid)

def generate_random_numbers_on_gpus(num_gpus, num_numbers_per_gpu):
    block_size = 256
    grid_size = 128

    random_numbers_per_gpu = []

    for gpu_id in range(num_gpus):
        with cuda.gpus[gpu_id]:
            rng_states = create_xoroshiro128p_states(grid_size * block_size, seed=1234+gpu_id)
            d_out = cuda.device_array(num_numbers_per_gpu, dtype=np.float32)
            gpu_random_numbers[grid_size, block_size](rng_states, d_out, num_numbers_per_gpu)
            random_numbers_per_gpu.append(d_out)

  #  return random_numbers_per_gpu

# Example: Generate 10 million random numbers on 4 GPUs
num_gpus = 11
num_numbers_per_gpu = 8000000000


for q in range(21):
    start_time = time.time()

    for w in range(1):
        result_per_gpu = generate_random_numbers_on_gpus(num_gpus, num_numbers_per_gpu)
        cuda.synchronize()

    end_time = time.time()

    wall_time = end_time - start_time
    print(wall_time)

