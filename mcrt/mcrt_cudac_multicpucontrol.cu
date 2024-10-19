#include <curand_kernel.h>
#include <iostream>
#include <stdio.h>
#include <chrono>
#include <pthread.h>

#define NUM_GPUS 10
#define NUM_ITERATIONS 10
#define PRINT_FRACTION 0.1 // Print 10% of the generated random numbers

// Struct to pass parameters to pthread
struct ThreadArgs {
    unsigned long long* d_randomNumbers;
    unsigned long long seed;
    unsigned long long numRandomNumbers;
};

// Kernel function to generate random numbers on each GPU
__global__ void generateRandomNumbers(unsigned long long* randomNumbers, unsigned long long seed, unsigned long long numRandomNumbers) {
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long numThreads = blockDim.x * gridDim.x;

    curandState state;
    curand_init(seed, tid, 0, &state);
    unsigned long long count_res = 0;
    for (unsigned long long i = tid; i < numRandomNumbers; i += numThreads) {
        float mean_free_path = 1.0 / 0.5;
        float q = curand_uniform(&state);
        float distance_to_collision = -log(q) * mean_free_path;  // Generate a random number (0.0 to 1.0)
        float distance_to_surface = 1;
        if (distance_to_collision < distance_to_surface) {
            count_res++;
        }
    }
    atomicAdd(randomNumbers, count_res);
}

// Function executed by each pthread
void* gpuWorker(void* arg) {
    ThreadArgs* threadArgs = (ThreadArgs*)arg;

    cudaSetDevice(threadArgs->seed);
    unsigned long long seed = time(NULL) + threadArgs->seed;

    // Allocate memory on the GPU
    unsigned long long* d_randomNumbers;
    cudaMalloc((void**)&d_randomNumbers, sizeof(unsigned long long));

    // Launch kernel to generate random numbers
    const int blockSize = 256;  // Adjust as needed for your GPU
    const int gridSize = 128;
    generateRandomNumbers<<<gridSize, blockSize>>>(d_randomNumbers, seed, threadArgs->numRandomNumbers);

    // Synchronize
    cudaDeviceSynchronize();

    // Copy result from GPU to CPU
    unsigned long long h_randomNumbers;
    cudaMemcpy(&h_randomNumbers, d_randomNumbers, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

//    std::cout << "GPU " << threadArgs->seed << ": " << h_randomNumbers << std::endl;

    // Free GPU memory
    cudaFree(d_randomNumbers);

    delete threadArgs;

//    free(h_randomNumbers);

    pthread_exit((void*)h_randomNumbers);
}

int main() {
//    unsigned long long totalRandomNumbers = 0;

        for (int q = 0; q < 21; ++q) {
        auto start = std::chrono::high_resolution_clock::now();
        for (int w = 0; w < 1; ++w) {
    unsigned long long num = 10000000000;

    for (int i = 0; i < NUM_ITERATIONS; ++i) {
//        auto start = std::chrono::high_resolution_clock::now();
//      unsigned long long num = 2000000000;
        unsigned long long totalRandomNumbers = 0;
        pthread_t threads[NUM_GPUS];
        for (int j = 0; j < NUM_GPUS; ++j) {
            ThreadArgs* args = new ThreadArgs();
            args->seed = j;
            args->numRandomNumbers = num;
            pthread_create(&threads[j], NULL, gpuWorker, (void*)args);
        }

        for (int j = 0; j < NUM_GPUS; ++j) {
            void* result;
            pthread_join(threads[j], &result);
            unsigned long long h_randomNumbers = (unsigned long long)(intptr_t)result;
            totalRandomNumbers += h_randomNumbers;
       }

 //       std::cout << "Total Random Numbers: " << totalRandomNumbers << std::endl;

//        auto stop = std::chrono::high_resolution_clock::now();
//        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

        // Print the total execution time for each iteration
  //      std::cout << "Time for iteration " << i << ": " << duration.count() << " ms" << std::endl;

        // Update numRandomValues
        num = totalRandomNumbers / NUM_GPUS;
    }}
    auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

        // Print the total execution time for each iteration
        std::cout << "Time for iteration " << duration.count() << " ms" << std::endl;

        }
    return 0;
}

