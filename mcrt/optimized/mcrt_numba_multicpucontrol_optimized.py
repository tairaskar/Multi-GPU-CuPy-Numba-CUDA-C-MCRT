import numpy as np
from numba import cuda, float32
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import math
import threading
import time

@cuda.jit(fastmath=True)
def gpu_random_numbers(rng_states, N, arr):
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    count_res = 0
    for i in range(tid, N, stride):
        mean_free_path = 1.0 / 0.5
        q = xoroshiro128p_uniform_float32(rng_states, tid)
        distance_to_collision = -math.log(q) * mean_free_path  # Using fast approximate log
        if distance_to_collision < 1.0:
            count_res += 1
    cuda.atomic.add(arr, 0, count_res)

def generate_random_numbers_on_gpu(gpu_id, num_numbers_per_gpu, result_array, grid_size, block_size):
    cuda.select_device(gpu_id)  # Select the GPU for the thread
    rng_states = create_xoroshiro128p_states(grid_size * block_size, seed=1234)
    num_numbers_per_gpu = int(num_numbers_per_gpu)

    device_array = np.zeros(1, dtype=np.int64)

    gpu_random_numbers[grid_size, block_size](rng_states, num_numbers_per_gpu, device_array)

    cuda.synchronize()

    result_array[gpu_id] = device_array[0]

def multi_gpu_random_numbers(num_gpus, num_numbers_per_gpu, block_size=256, grid_size=128):

    num_numbers_total = num_gpus * num_numbers_per_gpu

    for q in range(1):

        for w in range(10):

            num_numbers_per_gpu = int(num_numbers_total / num_gpus)
            result_array = [0] * num_gpus
            threads = []

            for gpu_id in range(num_gpus):
                thread = threading.Thread(target=generate_random_numbers_on_gpu,
                                          args=(gpu_id, num_numbers_per_gpu, result_array, grid_size, block_size))
                threads.append(thread)

            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join()

            final_result = sum(result_array)
            num_numbers_total = final_result
           # print(num_numbers_total)

for q in range(21):

    start_time = time.time()

    multi_gpu_random_numbers(num_gpus=10, num_numbers_per_gpu=10000000000)

    end_time = time.time()

    wall_time = end_time - start_time
    print(wall_time)

