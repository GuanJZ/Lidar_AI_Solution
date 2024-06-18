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

#include "bevfusion.hpp"

#include <numeric>

#include "common/check.hpp"
#include "common/timer.hpp"
#include "common/savetxt.hpp"

namespace bevfusion {

class CoreImplement : public Core {
 public:
  virtual ~CoreImplement() {
  }

  bool init(const CoreParameter& param) {
    camera_backbone_ = camera::create_backbone(param.camera_model);
    if (camera_backbone_ == nullptr) {
      printf("Failed to create camera backbone.\n");
      return false;
    }

    camera_bevpool_ =
        camera::create_bevpool(camera_backbone_->camera_shape(), param.geometry.geometry_dim.x, param.geometry.geometry_dim.y);
    if (camera_bevpool_ == nullptr) {
      printf("Failed to create camera bevpool.\n");
      return false;
    }

    camera_vtransform_ = camera::create_vtransform(param.camera_vtransform);
    if (camera_vtransform_ == nullptr) {
      printf("Failed to create camera vtransform.\n");
      return false;
    }

    transfusion_ = fuser::create_transfusion(param.transfusion);
    if (transfusion_ == nullptr) {
      printf("Failed to create transfusion.\n");
      return false;
    }

    normalizer_ = camera::create_normalization(param.normalize);
    if (normalizer_ == nullptr) {
      printf("Failed to create normalizer.\n");
      return false;
    }

    camera_geometry_ = camera::create_geometry(param.geometry);
    if (camera_geometry_ == nullptr) {
      printf("Failed to create geometry.\n");
      return false;
    }

    sample_grid_ = camera::create_samplegrid(param.sample_grid);
    if (sample_grid_ == nullptr) {
      printf("Failed to create sample_grid.\n");
      return false;
    }

    head_map_ = camera::create_headmap(param.headmap);
    if (head_map_ == nullptr) {
      printf("Failed to create head map.\n");
      return false;
    }

    post_proc_ = camera::create_postproc(param.post_proc);
    if (post_proc_ == nullptr) {
      printf("Failed to create post proc.\n");
      return false;
    }

    param_ = param;
    return true;
  }

  const float* forward_only(const void* camera_images, void* stream, bool do_normalization) {

    cudaStream_t _stream = static_cast<cudaStream_t>(stream);
    nvtype::half* normed_images = (nvtype::half*)camera_images;
    if (do_normalization) {
      normed_images = (nvtype::half*)this->normalizer_->forward((const unsigned char**)(camera_images), stream);
    }

    this->camera_backbone_->forward(normed_images, stream);
    const nvtype::half* camera_bev = this->camera_bevpool_->forward(
        this->camera_backbone_->feature(), this->camera_backbone_->depth(), this->camera_geometry_->indices(),
        this->camera_geometry_->intervals(), this->camera_geometry_->num_intervals(), stream);

    const nvtype::half* camera_bevfeat = camera_vtransform_->forward(camera_bev, stream);
    const nvtype::half* fusion_feature = this->transfusion_->forward(camera_bevfeat, stream);
    const nvtype::half* middle = this->sample_grid_->forward(fusion_feature, stream);
    const nvtype::half* bev_seg = head_map_->forward(middle, _stream);
    const float* bev_points = post_proc_->forward(bev_seg, _stream);
    return bev_points;
  }

  const float* forward_timer(const void* camera_images, void* stream, bool do_normalization) {

    printf("==================BEVFusion===================\n");
    std::vector<float> times;
    cudaStream_t _stream = static_cast<cudaStream_t>(stream);

    nvtype::half* normed_images = (nvtype::half*)camera_images;
    if (do_normalization) {
      timer_.start(_stream);
      normed_images = (nvtype::half*)this->normalizer_->forward((const unsigned char**)(camera_images), _stream);
      times.emplace_back(timer_.stop("[NoSt] ImageNrom"));
    }

    // std::string save_dir = 
    //   "/media/gpal/8e78e258-6a68-4733-8ec2-b837743b11e6/workspace/github/Lidar_AI_Solution/CUDA-BEVFusion/model/seg_camera_only_resnet50/assets/";

    timer_.start(_stream);
    this->camera_backbone_->forward(normed_images, stream);
    times.emplace_back(timer_.stop("Camera Backbone"));

    timer_.start(_stream);
    const nvtype::half* camera_bev = this->camera_bevpool_->forward(
        this->camera_backbone_->feature(), this->camera_backbone_->depth(), this->camera_geometry_->indices(),
        this->camera_geometry_->intervals(), this->camera_geometry_->num_intervals(), stream);
    times.emplace_back(timer_.stop("Camera Bevpool"));

    timer_.start(_stream);
    const nvtype::half* camera_bevfeat = camera_vtransform_->forward(camera_bev, stream);
    times.emplace_back(timer_.stop("VTransform.Downsample"));

    timer_.start(_stream);
    const nvtype::half* fusion_feature = this->transfusion_->forward(camera_bevfeat, stream);
    times.emplace_back(timer_.stop("Transfusion.Decoder"));

    // 添加 grid_sample
    // nvtype::half* input_feature = nullptr;
    // checkRuntime(cudaMalloc(&input_feature, 256*128*128 * sizeof(nvtype::half)));
    // std::string file_path = save_dir + "decoder.output.cpp.txt";
    // loadTxtToGpuMemory(file_path, input_feature);
    timer_.start(_stream);
    const nvtype::half* middle = this->sample_grid_->forward(fusion_feature, stream);
    times.emplace_back(timer_.stop("Samplegrid"));
    // std::string save_path = save_dir + "sample_grid.output.cpp.half.txt";
    // int num_elements = 256 * 200 * 200;
    // saveToTxt(save_path, middle, num_elements);


    // std::string middle_file_path = "/media/gpal/8e78e258-6a68-4733-8ec2-b837743b11e6/workspace/github/Lidar_AI_Solution/CUDA-BEVFusion/model/seg_camera_only_resnet50/assets/bev_grid_sample_output.txt";
    // loadTxtToGpuMemory(middle_file_path, middle);
    timer_.start(_stream);
    const nvtype::half* bev_seg = head_map_->forward(middle, _stream);
    times.emplace_back(timer_.stop("Headmap"));

    // int num_elements = 6 * 200 * 200;
    // std::string save_path = save_dir + "head.map.classifier.output.cpp.total3.txt";
    // saveToTxt(save_path, bev_seg, num_elements);

    timer_.start(_stream);
    const float* bev_points = post_proc_->forward(bev_seg, _stream);
    times.emplace_back(timer_.stop("Postprocess"));

    float total_time = std::accumulate(times.begin(), times.end(), 0.0f, std::plus<float>{});
    printf("Total: %.3f ms\n", total_time);
    printf("=============================================\n");
    return bev_points;
  }

  virtual const float* forward(const unsigned char** camera_images, void* stream) override {
    if (enable_timer_) {
      return this->forward_timer(camera_images, stream, true);
    } else {
      return this->forward_only(camera_images, stream, true);
    }
  }

  virtual const float* forward_no_normalize(const nvtype::half* camera_normed_images_device, void* stream) override {
    if (enable_timer_) {
      return this->forward_timer(camera_normed_images_device, stream, false);
    } else {
      return this->forward_only(camera_normed_images_device, stream, false);
    }
  }

  virtual void set_timer(bool enable) override { enable_timer_ = enable; printf("set_timer\n");}

  virtual void print() override {
    camera_backbone_->print();
    camera_vtransform_->print();
    transfusion_->print();
    head_map_->print();
  }

  virtual void update(const float* camera2lidar, const float* camera_intrinsics,
                      const float* img_aug_matrix, void* stream) override {
    camera_geometry_->update(camera2lidar, camera_intrinsics, img_aug_matrix, stream);
  }

  virtual void free_excess_memory() override { camera_geometry_->free_excess_memory(); }

 private:
  CoreParameter param_;
  nv::EventTimer timer_;

  std::shared_ptr<camera::Normalization> normalizer_;
  std::shared_ptr<camera::Backbone> camera_backbone_;
  std::shared_ptr<camera::BEVPool> camera_bevpool_;
  std::shared_ptr<camera::VTransform> camera_vtransform_;
  std::shared_ptr<camera::Geometry> camera_geometry_;
  std::shared_ptr<fuser::Transfusion> transfusion_;
  std::shared_ptr<camera::SampleGrid> sample_grid_;
  std::shared_ptr<camera::HeadMap> head_map_;
  std::shared_ptr<camera::PostProc> post_proc_;
  float confidence_threshold_ = 0;
  bool enable_timer_ = false;
};

std::shared_ptr<Core> create_core(const CoreParameter& param) {
  std::shared_ptr<CoreImplement> instance(new CoreImplement());
  if (!instance->init(param)) {
    instance.reset();
  }
  return instance;
}

};  // namespace bevfusion