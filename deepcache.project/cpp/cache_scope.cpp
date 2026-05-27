#include "cache_scope.hpp"
#include <vector>
#include <chrono>

volatile int sink;

double CacheScope::measure_latency(size_t buffer_size) {
    std::vector<int> data(buffer_size / sizeof(int), 1);

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < data.size(); i += 16) {
        sink += data[i];
    }

    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::nano>(end - start).count();
}
