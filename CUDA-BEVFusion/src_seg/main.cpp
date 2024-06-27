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

#include <cuda_runtime.h>
#include <string.h>

#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include "bevfusion/bevfusion.hpp"
#include "common/check.hpp"
#include "common/tensor.hpp"
#include "common/timer.hpp"

#include "common/savepcd.hpp"
#include "common/load_file.hpp"

static std::vector<unsigned char*> load_images(const std::string& root) {
  const char* file_names[] = {"1-FRONT_LEFT.jpg", "2-FRONT_RIGHT.jpg", "3-LEFT.jpg",
                              "4-RIGHT.jpg",  "5-BACK_LEFT.jpg",   "6-BACK_RIGHT.jpg"};

  std::vector<unsigned char*> images;
  for (int i = 0; i < 6; ++i) {
    char path[200];
    sprintf(path, "%s/%s", root.c_str(), file_names[i]);

    int width, height, channels;
    images.push_back(stbi_load(path, &width, &height, &channels, 0));
    // printf("Image info[%d]: %d x %d : %d\n", i, width, height, channels);
  }
  return images;
}

static void free_images(std::vector<unsigned char*>& images) {
  for (size_t i = 0; i < images.size(); ++i) stbi_image_free(images[i]);

  images.clear();
}

std::shared_ptr<bevfusion::Core> create_core(const std::string& model, const std::string& precision) {

  printf("Create by %s, %s\n", model.c_str(), precision.c_str());
  bevfusion::camera::NormalizationParameter normalization;
  normalization.image_width = 1920;
  normalization.image_height = 1080;
  normalization.output_width = 704;
  normalization.output_height = 256;
  normalization.num_camera = 6;
  normalization.resize_lim = 0.48f;
  normalization.interpolation = bevfusion::camera::Interpolation::Bilinear;

  float mean[3] = {0.485, 0.456, 0.406};
  float std[3] = {0.229, 0.224, 0.225};
  normalization.method = bevfusion::camera::NormMethod::mean_std(mean, std, 1 / 255.0f, 0.0f);


  // 这里和训练框架是不一一样的
  // xbound: [-51.2, 51.2, 0.4]
  // ybound: [-51.2, 51.2, 0.4]
  bevfusion::camera::GeometryParameter geometry;
  geometry.xbound = nvtype::Float3(-51.2f, 51.2f, 0.4f);
  geometry.ybound = nvtype::Float3(-51.2f, 51.2f, 0.4f);

  geometry.zbound = nvtype::Float3(-10.0f, 10.0f, 20.0f);
  geometry.dbound = nvtype::Float3(1.0, 60.0f, 0.5f);

  // 训练框架里面是 [height, weidth]
  // [256, 704]
  // [32, 88]
  geometry.image_width = 704;
  geometry.image_height = 256;
  geometry.feat_width = 88;
  geometry.feat_height = 32;
  geometry.num_camera = 6;
  // bevpool输出feature尺寸
  // (360, 360, 80) -> (256, 256, 80)
  geometry.geometry_dim = nvtype::Int3(256, 256, 80);

  bevfusion::camera::SampleGridParameter sample_grid;
  sample_grid.input_channel = 256;
  sample_grid.input_width = 128;
  sample_grid.input_height = 128;

  sample_grid.output_channel = 256;
  sample_grid.output_width = 200;
  sample_grid.output_height = 200;

  bevfusion::camera::PostProcParameter post_proc;
  post_proc.height = sample_grid.output_height;
  post_proc.width = sample_grid.output_width;
  post_proc.num_classes = 1;
  post_proc.resolusion = 0.5f;
  post_proc.x_start = -50.0f;
  post_proc.y_start = -50.0f;
  post_proc.threshold = 0.5;

  // 统计模块参数
  bevfusion::CoreParameter param;
  param.camera_model = nv::format("model/%s/build_seg/camera.backbone.plan", model.c_str());
  param.normalize = normalization;
  param.geometry = geometry;
  param.transfusion = nv::format("model/%s/build_seg/fuser.plan", model.c_str());
  param.camera_vtransform = nv::format("model/%s/build_seg/camera.vtransform.plan", model.c_str());
  param.sample_grid = sample_grid;
  param.headmap = nv::format("model/%s/build_seg/head.map.plan", model.c_str());
  param.post_proc = post_proc;

  return bevfusion::create_core(param);
}

int main(int argc, char** argv) {

  const char* data      = "deploy_data";
  // const char* model     = "seg_camera_only_resnet50";
  const char* model     = "seg_camera_only_resnet50_ge_bev_output_scope_0.5";
  const char* precision = "fp16";

  if (argc > 1) data      = argv[1];
  if (argc > 2) model     = argv[2];
  if (argc > 3) precision = argv[3];

  auto core = create_core(model, precision);
  if (core == nullptr) {
    printf("Core has been failed.\n");
    return -1;
  }

  cudaStream_t stream;
  cudaStreamCreate(&stream);
 
  core->print();
  core->set_timer(true);

  // Load matrix to host
  auto camera2lidar = load_data_from_file(nv::format("model/%s/%s/camera2lidar.txt", model, data));
  auto camera_intrinsics = load_data_from_file(nv::format("model/%s/%s/camera_intrinsics.txt", model, data));
  auto img_aug_matrix = load_data_from_file(nv::format("model/%s/%s/img_aug_matrix.txt", model, data));
  core->update(camera2lidar.data(), camera_intrinsics.data(), img_aug_matrix.data(), stream);

  // Load image and lidar to host
  auto images = load_images(nv::format("model/%s/%s/", model, data));
  
  // warmup
  const float* bev_points =
      core->forward((const unsigned char**)images.data(), stream);

  std::string save_path = nv::format(
  "/media/gpal/8e78e258-6a68-4733-8ec2-b837743b11e6/workspace/github/Lidar_AI_Solution/CUDA-BEVFusion/model/%s/assets/seg_points.pcd", model);
  int num_points = 200*200;
  save_pcd(bev_points, num_points, save_path);

  // evaluate inference time
  for (int i = 0; i < 5; ++i) {
    core->forward((const unsigned char**)images.data(), stream);
  }

  // destroy memory
  free_images(images);
  checkRuntime(cudaStreamDestroy(stream));
  return 0;
}