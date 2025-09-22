#include <iostream>
#include <random>

#include <cblas.h>

#include "helpers.hpp"
#include "pset1.hpp"

int main(int argc, char *argv[])
{
    int M = 512;
    int N = 512;
    int K = 256;

    float *A = new float[M * K];
    float *B = new float[K * N];
    float *C = new float[M * N];
    float *C_ref = new float[M * N];

    std::cout << "Initializing matrices" << std::endl;

    std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<float> dist;

    for (int i = 0; i < M * K; ++i)
    {
        A[i] = dist(e2);
    }

    for (int i = 0; i < M * K; ++i)
    {
        B[i] = dist(e2);
    }

    std::fill(C, C + M * N, 0.0f);
    std::fill(C_ref, C_ref + M * N, 0.0f);

    // Reference BLAS test
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1, A, K, B, N, 0, C_ref, N);

    inner_product_mmm(C, A, B, M, N, K);
    validate_matrix<float>(C, C_ref, M * N);
    std::fill(C, C + M * N, 0.0f);

    outer_product_mmm(C, A, B, M, N, K);
    validate_matrix<float>(C, C_ref, M * N);
    std::fill(C, C + M * N, 0.0f);

    // Cleanup
    delete[] A;
    delete[] B;
    delete[] C;
    delete[] C_ref;

    return 0;
}
