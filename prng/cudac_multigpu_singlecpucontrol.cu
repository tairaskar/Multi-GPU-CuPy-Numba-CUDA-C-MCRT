#include <curand_kernel.h>
#include <iostream>
#include <chrono>

// Define the number of random numbers to generate on each GPU
#define NUM_RANDOM_NUMBERS_PER_GPU 8000000000ULL // 1 billion numbers per GPU
#define NUM_GPUS 10

// Kernel function to generate random numbers on each GPU
__global__ void generateRandomNumbers(float* randomNumbers, unsigned long long seed, unsigned long long numRandomNumbers) {
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long numThreads = blockDim.x * gridDim.x;

    curandState state;
    curand_init(seed, tid, 0, &state);

    for (unsigned long long i = tid; i < numRandomNumbers; i += numThreads) {
        randomNumbers[i] = curand_uniform(&state);  // Generate a random number (0.0 to 1.0)
//      printf("%f\n", randomNumbers[i]);
    }
}

int main() {
//    const int numRandomNumbersTotal = NUM_RANDOM_NUMBERS_PER_GPU * NUM_GPUS;

        const int blockSize = 256;  // Adjust as needed for your GPU
        const int gridSize = 128;

for (int i = 0; i < 21; ++i) {

        auto start = std::chrono::high_resolution_clock::now();

        for (int q = 0; q < 1; ++q) {

    float* d_randomNumbers[NUM_GPUS];
    unsigned long long seeds[NUM_GPUS];

    // Initialize CUDA devices and allocate memory
    for (int i = 0; i < NUM_GPUS; ++i) {
        cudaSetDevice(i);

        cudaMalloc((void**)&d_randomNumbers[i], sizeof(float) * NUM_RANDOM_NUMBERS_PER_GPU);

        // Use different seeds for each GPU
       seeds[i] = time(NULL) + i;

        // Launch kernel to generate random numbers on each GPU
        generateRandomNumbers<<<gridSize, blockSize>>>(d_randomNumbers[i], seeds[i], NUM_RANDOM_NUMBERS_PER_GPU);
    }

    // Synchronize and retrieve results from each GPU
    for (int i = 0; i < NUM_GPUS; ++i) {
        cudaSetDevice(i);
        cudaDeviceSynchronize();
    }

    // Process or save the generated random numbers as needed

    // Clean up and free GPU memory
    for (int i = 0; i < NUM_GPUS; ++i) {
       cudaSetDevice(i);
        cudaFree(d_randomNumbers[i]);
    }
        }
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

        // Print the total execution time for each iteration
        std::cout << duration.count() << "\n";

}
    return 0;
}

