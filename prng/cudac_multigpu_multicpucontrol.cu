#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <curand_kernel.h>

#define NUM_RANDOM_NUMBERS_PER_GPU 8000000000ULL // 1 billion numbers per GPU
#define NUM_GPUS 10

__global__ void generateRandomNumbers(float* randomNumbers, unsigned long long seed, unsigned long long numRandomNumbers) {
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long numThreads = blockDim.x * gridDim.x;

    curandState state;
    curand_init(seed, tid, 0, &state);

    for (unsigned long long i = tid; i < numRandomNumbers; i += numThreads) {
        randomNumbers[i] = curand_uniform(&state);
    }
}

void workerThread(int gpuID) {
    const int blockSize = 256;
    const int gridSize = 160;

  //  auto start = std::chrono::high_resolution_clock::now();

    for (int q = 0; q < 1; ++q) {
        cudaSetDevice(gpuID);

        float* d_randomNumbers;
        unsigned long long seed = time(NULL) + gpuID;

        cudaMalloc((void**)&d_randomNumbers, sizeof(float) * NUM_RANDOM_NUMBERS_PER_GPU);

        generateRandomNumbers<<<gridSize, blockSize>>>(d_randomNumbers, seed, NUM_RANDOM_NUMBERS_PER_GPU);

        cudaDeviceSynchronize();

        cudaFree(d_randomNumbers);
    }

   // auto stop = std::chrono::high_resolution_clock::now();
  //  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    //std::cout << "Thread " << gpuID << ": " << duration.count() << " ms\n";
}

int main() {

    for (int w = 0; w < 21; ++w){
    auto start = std::chrono::high_resolution_clock::now();
   std::vector<std::thread> threads;

    for (int i = 0; i < NUM_GPUS; ++i) {
        threads.emplace_back(workerThread, i);
    }

    for (auto& thread : threads) {
        thread.join();
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    // Print the total execution time for each iteration
    std::cout << duration.count() << "\n";
    }
    return 0;
}

