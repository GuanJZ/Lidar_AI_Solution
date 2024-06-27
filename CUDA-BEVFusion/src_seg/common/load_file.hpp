#include <iostream>
#include <fstream>
#include <vector>

std::vector<float> load_data_from_file(const std::string &filename) {
    std::vector<float> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return data;
    }

    float value;
    while (file >> value) {
        data.push_back(value);
    }
    file.close();
    return data;
}