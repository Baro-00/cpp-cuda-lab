# CUDA Progamming

## Introduction

**CUDA** (*Compute Unified Device Architecture*) is a parallel computing platform and API model created by NVIDIA. It enables developers to utilize the power of NVIDIA GPUs for general-purpose processing (GPGPU). CUDA allows programmers to leverage parallelism in GPU cores to dramatically speed up computations, especially useful in scientific calculations, image processing, machine learning, and data-intensive tasks.

### Prerequisites

Before getting started with CUDA programming, ensure you have:

- NVIDIA GPU compatible with CUDA

- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

- [Visual Studio IDE](https://visualstudio.microsoft.com/pl/downloads/) (just for binaries)

### Supported Hardware

Make sure your NVIDIA GPU supports CUDA. You can check compatibility [here](https://developer.nvidia.com/cuda-gpus).

### Documentation

[CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)

> **Hint**: Source code with CUDA integration has `.cu` extension.

---

## Getting started

### Simple CUDA Example (*Vector Addition*)

File: `vector_add/vector_add.cu`

``` cpp
#include <iostream>
#include <cuda_runtime.h>

// Kernel function executed on GPU
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int N = 1024;
    size_t size = N * sizeof(float);

    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];

    for (int i = 0; i < N; ++i) {
        h_A[i] = i * 1.0f;
        h_B[i] = i * 2.0f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    std::cout << "First 5 results:\n";
    for (int i = 0; i < 5; ++i) {
        std::cout << h_A[i] << " + " << h_B[i] << " = " << h_C[i] << "\n";
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
```

### Build and run

For compiling CUDA applications, use the provided NVIDIA compiler (`nvcc`).

#### Using *x64 Native Tools Command Prompt for VS 2022*

Open `x64 Native Tools Command Prompt for VS 2022`

**Build**

``` console
nvcc -o vector_add vector_add.cu
```

**Run**

``` console
vector_add.exe
```

#### Using *PowerShell*

To compile CUDA code using PowerShell, you must first load Visual Studio environment variables.

Run the following command in PowerShell to initialize the Visual Studio environment:

``` console
cmd /c "`"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat`" && powershell"
```

After that, you can build and run your CUDA application:

``` console
nvcc -o vector_add.exe vector_add.cu
.\vector_add.exe
```
