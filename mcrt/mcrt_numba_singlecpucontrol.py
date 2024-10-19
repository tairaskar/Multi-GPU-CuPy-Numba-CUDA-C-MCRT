import numpy as np
from numba import cuda, float32
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
from numba import jit
from math import log
import numba as nb
import time

@cuda.jit
def gpu_random_numbers(rng_states, N, arr):
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    count_res = 0
    for i in range(tid, N, stride):
        mean_free_path = nb.float32(1.0/0.5)
        q = cuda.random.xoroshiro128p_uniform_float32(rng_states, tid)
        distance_to_collision = nb.float32(-log(q) * mean_free_path)
        distance_to_surface = nb.float32(1.0)
        if distance_to_collision < distance_to_surface:
            count_res += 1
    cuda.atomic.add(arr, 0, count_res)

def multi_gpu_random_numbers(num_gpus, num_numbers_per_gpu, block_size=256, grid_size=128):
    num_numbers_total = num_gpus * num_numbers_per_gpu
   # result = 0

    for q in range(1):
        for i in range(10):
            num_numbers_per_gpu = int(num_numbers_total / num_gpus)
            result = 0
        #    arr = np.zeros(1, dtype=np.int32)

            # Use cuda.device_array to allocate device memory for each GPU
            device_arrays = [np.zeros(1, dtype=np.int64) for _ in range(num_gpus)]

            # Launch kernels on each GPU
            for gpu_id in range(num_gpus):
                with cuda.gpus[gpu_id]:
                    rng_states = create_xoroshiro128p_states(grid_size * block_size, seed=1234)
                    gpu_random_numbers[grid_size, block_size](rng_states, num_numbers_per_gpu, device_arrays[gpu_id])
                   # cuda.synchronize()
                    result += device_arrays[gpu_id][0]
            # Synchronize GPUs and sum up the results
           # for gpu_id in range(num_gpus):
            #    cuda.synchronize()
             #   result += device_arrays[gpu_id][0]

  #              print(result)
            num_numbers_total = result
#            print(num_numbers_total)
    #        print(result)

for q in range(21):
    start_time = time.time()

    multi_gpu_random_numbers(num_gpus=11, num_numbers_per_gpu=10000000000)
    cuda.synchronize()

    end_time = time.time()

    wall_time = end_time - start_time
    print(wall_time)

