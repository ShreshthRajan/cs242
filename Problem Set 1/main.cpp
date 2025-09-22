#include <chrono>
#include <iostream>

#include "helpers.hpp"
#include "pset1.hpp"

int main(int argc, char *argv[])
{
    int N = 512;
    int iters = 5;

    if (argc < 2)
    {
        std::cout << "Defaulting to N = " << N << std::endl;
    }
    else
    {
        if (sscanf(argv[1], "%i", &N) != 1)
        {
            std::cout << "Warning, non-integer value for N." << std::endl;
        }
    }

    int N2 = N * N;

    float *A = new float[N2];
    float *B = new float[N2];
    float *C = new float[N2];
    float *C_ref = new float[N2];

    std::cout << "Initializing matrices" << std::endl;
    std::fill(A, A + N2, 1.f);
    std::fill(B, B + N2, 1.f);
    std::fill(C, C + N2, 0.f);
    std::fill(C_ref, C_ref + N2, N * iters);

    std::cout << "Initialization done" << std::endl;
    std::cout << "Performing MMs for " << N << "x" << N << " matrices (" << iters << " iters)" << std::endl;

    using std::chrono::duration;
    using std::chrono::duration_cast;
    using std::chrono::high_resolution_clock;
    using std::chrono::milliseconds;
    using std::chrono::nanoseconds;

    {
        auto start = high_resolution_clock::now();
        for (int i = 0; i < iters; ++i)
        {
            inner_product_mmm(C, A, B, N, N, N);
        }
        auto end = high_resolution_clock::now();
        auto time = duration_cast<nanoseconds>(end - start);
        std::cout << "Inner product MMM avg. run time (ns): " << time.count() / iters << std::endl;
        validate_matrix<float>(C, C_ref, N2);
        std::fill(C, C + N2, 0.f);
    }
    {
        auto start = high_resolution_clock::now();
        for (int i = 0; i < iters; ++i)
        {
            outer_product_mmm(C, A, B, N, N, N);
        }
        auto end = high_resolution_clock::now();
        auto time = duration_cast<nanoseconds>(end - start);
        std::cout << "Outer product MMM avg. run time (ns): " << time.count() / iters << std::endl;
        validate_matrix<float>(C, C_ref, N2);
        std::fill(C, C + N2, 0.f);
    }

    // Cleanup
    delete[] A;
    delete[] B;
    delete[] C;
    delete[] C_ref;

    return 0;
}
