#include <cuda_fp16.h>
#include <fstream>
#include <iostream>
#include <string>

void copyFromDeviceToHost(const nvtype::half* device_data, std::vector<nvtype::half>& host_data, size_t n) {
    host_data.resize(n);
    cudaMemcpy(host_data.data(), device_data, n * sizeof(nvtype::half), cudaMemcpyDeviceToHost);
}

float halfToFloat(nvtype::half h) {
    __half raw_half;
    memcpy(&raw_half, &h, sizeof(__half));  // 将数据复制到 __half 结构体
    return __half2float(raw_half);          // 使用CUDA函数转换为float
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