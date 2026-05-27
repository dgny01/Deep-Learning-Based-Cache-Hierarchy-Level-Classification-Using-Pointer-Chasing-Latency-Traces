#include "cache_scope.hpp"
#include <vector>
#include <iostream>

void export_csv(double latency, size_t buffer, int label);

int main() {
    CacheScope cs;

    std::vector<size_t> L1 = {32 * 1024};
    std::vector<size_t> L2 = {256 * 1024};
    std::vector<size_t> L3 = {8 * 1024 * 1024};
    std::vector<size_t> RAM = {256 * 1024 * 1024};

    for (int i = 0; i < 200; i++) {
        for (auto b : L1) export_csv(cs.measure_latency(b), b, 0);
        for (auto b : L2) export_csv(cs.measure_latency(b), b, 1);
        for (auto b : L3) export_csv(cs.measure_latency(b), b, 2);
        for (auto b : RAM) export_csv(cs.measure_latency(b), b, 3);
    }

    std::cout << "Dataset generated!\n";
}
