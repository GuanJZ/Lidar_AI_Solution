/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <cuda_fp16.h>
#include <numeric>

#include "sample-grid.hpp"
#include "common/check.hpp"
#include "common/launch.cuh"
#include "common/tensorrt.hpp"

namespace bevfusion {
namespace camera {

static __device__ half bilinear_interpolate(const half* input, int height, int width, float y, float x) {
    int x0 = floor(x);
    int x1 = x0 + 1;
    int y0 = floor(y);
    int y1 = y0 + 1;

    float xa = x - x0, ya = y - y0;
    float wa = (1 - xa) * (1 - ya);
    float wb = xa * (1 - ya);
    float wc = (1 - xa) * ya;
    float wd = xa * ya;

    float val00 = (x0 >= 0 && x0 < width && y0 >= 0 && y0 < height) ? __half2float(input[y0 * width + x0]) : 0;
    float val01 = (x1 >= 0 && x1 < width && y0 >= 0 && y0 < height) ? __half2float(input[y0 * width + x1]) : 0;
    float val10 = (x0 >= 0 && x0 < width && y1 >= 0 && y1 < height) ? __half2float(input[y1 * width + x0]) : 0;
    float val11 = (x1 >= 0 && x1 < width && y1 >= 0 && y1 < height) ? __half2float(input[y1 * width + x1]) : 0;

    return __float2half(wa * val00 + wb * val01 + wc * val10 + wd * val11);
}

static __global__ void grid_sample_kernel(const half* input, half* output, int channels, int input_height, int input_width, int output_height, int output_width) {
    // blockSize(16, 16, 1), 表示每一个线程块有16x16个线程；
    // gridSize(13, 13, 256), 表示每一个网格有 13x13x256个线程块;
    // 接下来会有总计 16x16x13x13x256次索引计算和双线性插值计算。
    int x = blockIdx.x * blockDim.x + threadIdx.x; // blockIdx.x[0, 12], blockDim.x: 16, threadIdx.x[0, 15]
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z * blockDim.z + threadIdx.z; // blockIdx.z [0, 255], blockDim.z: 1, threadIdx.z: 0

    if (x < output_width && y < output_height && c < channels) {
        int input_index = c * input_height * input_width;
        int output_index = c * output_height * output_width;
        float input_x = ((float)x / output_width) * input_width;
        float input_y = ((float)y / output_height) * input_height;
        output[output_index + y * output_width + x] = bilinear_interpolate(input + input_index, input_height, input_width, input_y, input_x);
    }
}

class SampleGridImplement : public SampleGrid {
 public:
  virtual ~SampleGridImplement() {
    if (output_feature_) checkRuntime(cudaFree(output_feature_));
  }

  bool init(SampleGridParameter param) {
    this->input_width_ = param.input_width;
    this->input_height_ = param.input_height;
    this->input_channel_ = param.input_channel;
    this->output_width_ = param.output_width;
    this->output_height_ = param.output_height;
    this->output_channel_ = param.output_channel;

    volumn_output_ = param.output_channel * param.output_width * param.output_height;
    output_dims_ = {1, (int)param.output_channel, (int)param.output_height, (int)param.output_width};
    checkRuntime(cudaMalloc(&output_feature_, volumn_output_ * sizeof(half)));
    return true;
  }

  virtual std::vector<int> shape() override { return output_dims_; }

  virtual nvtype::half* forward(const nvtype::half* input_feature, void* stream = nullptr) override {
    dim3 blockSize(16, 16, 1);
    dim3 gridSize((output_width_ + blockSize.x - 1) / blockSize.x, (output_height_ + blockSize.y - 1) / blockSize.y, output_channel_);

    cudaStream_t _stream = static_cast<cudaStream_t>(stream);
    checkRuntime(cudaMemsetAsync(output_feature_, 0x00, volumn_output_ * sizeof(half), _stream));
    checkKernel(grid_sample_kernel<<<gridSize, blockSize, 0, _stream>>>(
                          reinterpret_cast<const half*>(input_feature), output_feature_, 
                          output_channel_, input_height_, input_width_, output_height_, output_width_));
    return reinterpret_cast<nvtype::half*>(output_feature_);
  }

 private:
  unsigned int output_width_ = 0;
  unsigned int output_height_ = 0;
  unsigned int output_channel_ = 0;
  unsigned int input_width_ = 0;
  unsigned int input_height_ = 0;
  unsigned int input_channel_ = 0;
  half* output_feature_ = nullptr;
  std::vector<int> output_dims_;
  unsigned int volumn_output_ = 0;
};

std::shared_ptr<SampleGrid> create_samplegrid(SampleGridParameter param) {
  std::shared_ptr<SampleGridImplement> instance(new SampleGridImplement());
  if (!instance->init(param)) {
    instance.reset();
  }
  return instance;
}

};  // namespace camera
};  // namespace bevfusion