#include <fstream>

void export_csv(double latency, size_t buffer, int label) {
    std::ofstream file("dataset/cache_dataset.csv", std::ios::app);
    file << latency << "," << buffer << "," << label << "\n";
}
