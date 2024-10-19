import cupy as cp
import concurrent.futures
import time

num_gpus = 9
array_size = 2000000000

def generate_random_array(seed):
    with cp.cuda.Device(seed):
        mempool = cp.get_default_memory_pool()
       # gen = cp.random.Generator(cp.random.XORWOW(1234 + seed))
       # random_array = gen.random(array_size, dtype=cp.float32)
        random_array = cp.random.Generator(cp.random.XORWOW(1234+seed)).random(int(array_size), dtype=cp.float32)

        cp.cuda.Stream.null.synchronize()
        mempool.free_all_blocks()

    return random_array

def main():
    start_time = time.time()
    for w in range(4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as executor:

            # Submit jobs for generating random arrays in parallel
            futures = [executor.submit(generate_random_array, i) for i in range(num_gpus)]
            concurrent.futures.wait(futures)

            # Retrieve results
            random_arrays = [future.result() for future in futures]

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(elapsed_time)

if __name__ == "__main__":
    for q in range(21):
        main()

