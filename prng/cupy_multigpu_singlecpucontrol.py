import cupy as cp
import time

#cp.cuda.set_allocator(None)

# Define the number of GPUs to use
#num_gpus = 5  # Change this to match your hardware configuration

# Define the size of the random number arrays
#array_size = cp.int64(2000000000)

for q in range(21):
    start_time = time.time()

    for w in range(4):

        num_gpus = 8

       # mempool = cp.get_default_memory_pool()

        # Initialize random generators on each GPU
        for i in range(num_gpus):
            with cp.cuda.Device(i):
               # cp.cuda.set_allocator(None)
                mempool = cp.get_default_memory_pool()
               # gen = cp.random.Generator(cp.random.XORWOW(1234+i))
               # random_array = gen.random(array_size, dtype=cp.float32)
                random_array = cp.random.Generator(cp.random.XORWOW(1234+i)).random(int(2000000000), dtype=cp.float32)
                cp.cuda.Stream.null.synchronize()
                #del random_array
                mempool.free_all_blocks()


    # Ensure that the GPU operations are completed before continuing
        #cp.cuda.Stream.null.synchronize()

        #mempool.free_all_blocks()

    #cp.cuda.Stream.null.synchronize()            
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(elapsed_time)


