#include <iostream>
#include <cuda_runtime.h>

// Kernel CUDA - wykonywany na GPU
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main(void) {
    int N = 1024;
    size_t size = N * sizeof(float);

    // Alokacja pamięci na CPU
    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];

    // Inicjalizacja danych wejściowych
    for (int i = 0; i < N; ++i) {
        h_A[i] = i * 1.0f;
        h_B[i] = i * 2.0f;
    }

    // Alokacja pamięci na GPU
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Kopiowanie danych z CPU na GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Ustawienia uruchamiania kernela
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Uruchomienie kernela CUDA
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Kopiowanie wyników z GPU na CPU
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Wypisanie kilku wyników
    std::cout << "Pierwsze 5 wynikow:\n";
    for (int i = 0; i < 5; ++i) {
        std::cout << h_A[i] << " + " << h_B[i] << " = " << h_C[i] << "\n";
    }

    // Zwolnienie pamięci
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
