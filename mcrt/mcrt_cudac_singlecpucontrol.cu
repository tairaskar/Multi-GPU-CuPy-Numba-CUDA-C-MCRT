#include <curand_kernel.h>
#include <iostream>
#include <stdio.h>
#include <chrono>

#define NUM_GPUS 1
#define PRINT_FRACTION 0.1 // Print 10% of the generated random numbers

// Kernel function to generate random numbers on each GPU
__global__ void generateRandomNumbers(unsigned long long* randomNumbers, unsigned long long seed, unsigned long long numRandomNumbers) {
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long numThreads = blockDim.x * gridDim.x;

    curandState state;
    curand_init(seed, tid, 0, &state);
    int count_res = 0;
    for (unsigned long long i = tid; i < numRandomNumbers; i += numThreads) {
        float mean_free_path = 1.0 / 0.5;
        float q = curand_uniform(&state);
        float distance_to_collision = -log(q) * mean_free_path;  // Generate a random number (0.0 to 1.0)
        float distance_to_surface = 1;
        if (distance_to_collision < distance_to_surface)
        {
                count_res++;}
}
        atomicAdd(randomNumbers, count_res);
//      printf("%f\n", randomNumbers[i]);
    }



int main() {

        for (int i = 0; i < 21; ++i) {

        auto start = std::chrono::high_resolution_clock::now();

        for (int q = 0; q < 1; ++q) {

    const int blockSize = 256;  // Adjust as needed for your GPU
    const int gridsize = 128;

    unsigned long long* d_randomNumbers[NUM_GPUS];
    unsigned long long seeds[NUM_GPUS];
    unsigned long long NUM_RANDOM_NUMBERS_PER_GPU = 10000000000;
    for (int q=0; q<10; ++q){
    unsigned long long sum = 0;
    // Initialize CUDA devices and allocate memory
    for (int i = 0; i < NUM_GPUS; ++i) {
        cudaSetDevice(i);

        cudaMalloc((void**)&d_randomNumbers[i], sizeof(unsigned long long));
                // Use different seeds for each GPU
        seeds[i] = time(NULL) + i;

        // Launch kernel to generate random numbers on each GPU
        generateRandomNumbers<<<gridsize, blockSize>>>(d_randomNumbers[i], seeds[i], NUM_RANDOM_NUMBERS_PER_GPU);
   }
   // Synchronize and retrieve results from each GPU
    for (int i = 0; i < NUM_GPUS; ++i) {
        cudaSetDevice(i);
        cudaDeviceSynchronize();
    }

    // Copy and print a fraction of the generated random numbers from each GPU
    for (int i = 0; i < NUM_GPUS; ++i) {
        cudaSetDevice(i);
        unsigned long long* h_randomNumbers = new unsigned long long[1];
        cudaMemcpy(h_randomNumbers, d_randomNumbers[i], sizeof(unsigned long long), cudaMemcpyDeviceToHost);

        // Print a fraction of the results
//        int numToPrint = static_cast<int>(NUM_RANDOM_NUMBERS_PER_GPU * PRINT_FRACTION);
//        for (int j = 0; j < numToPrint; ++j) {
//            std::cout << "GPU " << i << " - Random Number " << j << ": " << h_randomNumbers[j] << std::endl;
//        }

//    int sum = h_randomNumbers;
    // Print the result
  //  printf("Sum of all values in the array: %llu\n", h_randomNumbers[0]);
    sum = sum + h_randomNumbers[0];
   // printf("%llu\n", sum);
        delete[] h_randomNumbers;
    }

    // Clean up and free GPU memory
    for (int i = 0; i < NUM_GPUS; ++i) {
        cudaSetDevice(i);
        cudaFree(d_randomNumbers[i]);
    }
    NUM_RANDOM_NUMBERS_PER_GPU = sum/NUM_GPUS;
    }

}
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        // Print the total execution time for each iteration
        std::cout << duration.count() << "\n";

}

    return 0;
}

