#pragma once

#include <cmath>
#include <limits>

template <typename T>
bool validate_matrix(T const *test, T const *ref, size_t size)
{
    int errors = 0;
    double total_error = 0.0;
    for (int i = 0; i < size; ++i)
    {
        const auto diff = std::fabs(test[i] - ref[i]);
        // accounting for floating point differences,
        // values should still be within 0.0001
        if (diff > 0.0001)
        {
            total_error += diff;
            ++errors;
        }
    }

    if (errors > 0)
    {
        std::cout << "[FAIL] Incorrect values: " << errors << " / " << size << std::endl;
        std::cout << "            Total Error: " << total_error << std::endl;
        return false;
    }
    std::cout << "[PASS] All values correct!" << std::endl;
    return true;
}
