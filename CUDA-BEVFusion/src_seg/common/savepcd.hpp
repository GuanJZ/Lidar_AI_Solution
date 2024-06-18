#include <iostream>
#include <string>
#include <vector>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <cuda_runtime.h>
#include "common/check.hpp"

void save_pcd(const float* points, int num_points, std::string save_path){
    std::vector<float> h_pcl_points(num_points * 3);
    checkRuntime(
        cudaMemcpy(h_pcl_points.data(), points, num_points * 3 * sizeof(float), cudaMemcpyDeviceToHost)
    );
    pcl::PointCloud<pcl::PointXYZ> cloud;
    cloud.points.resize(num_points);
    for (size_t i = 0; i < num_points; i++){
        pcl::PointXYZ point;
        // std::cout << "333333333333333" << std::endl;
        point.x = h_pcl_points[i * 3 + 0];
        point.y = h_pcl_points[i * 3 + 1];
        point.z = h_pcl_points[i * 3 + 2];
        cloud.points[i] = point;
    }

    cloud.width = num_points;
    cloud.height = 1;
    cloud.is_dense = false;
    
    pcl::io::savePCDFileASCII(save_path, cloud);

    std::cout << "points saved" << std::endl;

}