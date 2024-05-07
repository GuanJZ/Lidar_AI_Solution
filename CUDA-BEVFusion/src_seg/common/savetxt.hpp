#include <cuda_fp16.h>
#include <fstream>
#include <iostream>
#include <string>

void copyFromDeviceToHost(const nvtype::half* device_data, std::vector<nvtype::half>& host_data, size_t n) {
    host_data.resize(n);
    cudaMemcpy(host_data.data(), device_data, n * sizeof(nvtype::half), cudaMemcpyDeviceToHost);
}

void copyFromHostToDevice(const std::vector<nvtype::half>& host_data, nvtype::half* device_data) {
    cudaMemcpy(device_data, host_data.data(), host_data.size() * sizeof(nvtype::half), cudaMemcpyHostToDevice);
}

float halfToFloat(nvtype::half h) {
    __half raw_half;
    memcpy(&raw_half, &h, sizeof(__half));  // 将数据复制到 __half 结构体
    return __half2float(raw_half);          // 使用CUDA函数转换为float
}

nvtype::half floatToHalf(float f) {
    __half h = __float2half(f);
    nvtype::half result;
    memcpy(&result, &h, sizeof(__half));  // 将 __half 数据复制到 nvtype::half 结构体
    return result;
}

std::vector<nvtype::half> loadFromTxt(const std::string& file_path) {
    std::vector<nvtype::half> data;
    std::ifstream in_file(file_path);
    float value;

    if (in_file.is_open()) {
        while (in_file >> value) {
            data.push_back(floatToHalf(value));
        }
        in_file.close();
    } else {
        std::cerr << "Unable to open file" << std::endl;
    }

    return data;
}


void saveToTxt(std::string save_path, const nvtype::half* feature, int num_elements) {
    // 在CPU上分配内存
    std::vector<nvtype::half> cpu_feature;
    
    // 从GPU拷贝到CPU
    copyFromDeviceToHost(feature, cpu_feature, num_elements);
    
    // 写入文件
    std::ofstream out_file(save_path);
    if (out_file.is_open()) {
        for (int i = 0; i < num_elements; ++i) {
            float value = halfToFloat(cpu_feature[i]);  // 转换为float
            out_file << value << std::endl;
        }
        out_file.close();
    } else {
        std::cerr << "Unable to open file";
    }
}

void loadTxtToGpuMemory(const std::string& file_path, nvtype::half* device_data) {
    // 从文本文件加载数据
    std::vector<nvtype::half> host_data = loadFromTxt(file_path);

    // 将数据从主机复制到显存
    copyFromHostToDevice(host_data, device_data);
}

bool load_data_from_file(const std::string &filename, float* data, int size) {
    std::vector<float> host_input_data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return false;
    }

    float value;
    while (file >> value) {
        host_input_data.push_back(value);
    }
    file.close();
    cudaMemcpy(data, host_input_data.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    return true;
}

bool save_data_to_file(const std::string &filename, const float* data, int size) {
    std::vector<float> host_output_data(size);
    cudaMemcpy(host_output_data.data(), data, size * sizeof(float), cudaMemcpyDeviceToHost);

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return false;
    }

    for (float value : host_output_data) {
        file << value << std::endl;
    }
    file.close();
    return true;
}
