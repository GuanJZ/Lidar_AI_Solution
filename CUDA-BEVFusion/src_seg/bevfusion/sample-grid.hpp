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

#ifndef __SAMPLE_GRID_HPP__
#define __SAMPLE_GRID_HPP__

#include <memory>
#include <string>
#include <vector>

#include "common/dtype.hpp"

namespace bevfusion {
namespace camera {

struct SampleGridParameter{
  unsigned int input_width;
  unsigned int input_height;
  unsigned int input_channel;
  
  unsigned int output_width;
  unsigned int output_height;
  unsigned int output_channel;
};

class SampleGrid {
 public:
  virtual nvtype::half* forward(const nvtype::half* input_feature, void* stream = nullptr) = 0;
  virtual std::vector<int> shape() = 0;
};

std::shared_ptr<SampleGrid> create_samplegrid(SampleGridParameter param);

};  // namespace camera
};  // namespace bevfusion

#endif  // __SAMPLE_GRID_HPP__