**Content Overview**

The repository consists of two folders: PRNG and MCRT.

**Dependencies**

To run the codes in this repository, you will need the following dependencies:

    CUDA Toolkit: Version 10.0 or later.
    CuPy: Version 7.0 or later.
    Numba: Version 0.49 or later.
    Python: Version 3.6 or later.
    NVIDIA GPU: Supporting CUDA with appropriate drivers installed.

**Build and Run Instructions**

Build and Run CUDA C Code: compile using nvcc

    nvcc mcrt_cudac_multicpucontrol.cu
    ./a.out

To run optimized code, navigate to the optimized folder and run the following:
    
    nvcc mcrt_cudac_multicpucontrol_optimized.cu -use_fast_math
    ./a.out

Run CuPy and Numba Codes:

    python3 mcrt_cupy_multicpucontrol.py
    python3 mcrt_numba_multicpucontrol.py

To run optimized code, navigate to the optimized folder and run the following:

    python3 mcrt_cupy_multicpucontrol_optimized.py
    python3 mcrt_numba_multicpucontrol_optimized.py

**Additional Notes**

- Ensure that the GPU drivers are up-to-date and the CUDA environment is properly configured.
- For optimal performance, use the configuration tuning scripts to determine the best grid and block sizes for your specific hardware. Recommended values: For grid size, use 4 times the total number of Streaming Multiprocessors
of the utilized GPU card; for block size: 128-256 threads.
- For double precision calculations, change the data type to the appropriate value. 
