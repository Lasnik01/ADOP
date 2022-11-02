/**
 * Copyright (c) 2021 Darius R�ckert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

//TODO: SHORTEN INCLUDES 
#pragma once

#include "saiga/cuda/imageProcessing/image.h"
#include "saiga/cuda/imgui_cuda.h"

#include "saiga/normal_packing.h"
#include "saiga/vision/torch/TorchHelper.h"
#include <torch/torch.h>


class MyConvUnet2d
{
    //ALL IMAGES IN ONE BATCH MUST HAVE THE SAME SIZE!!!
    //stride = 1, dilation = 1, padding = 1, kernel_size = 3

   public:
    MyConvUnet2d(){};
    MyConvUnet2d(int input_channels, int output_channels, std::string activation_str);
    void PushNewWeights(const torch::Tensor feature_weight, const torch::Tensor feature_bias,
                        const torch::Tensor mask_weight, const torch::Tensor mask_bias);

    void ForwardMulti1(const torch::Tensor data_in);
    void ForwardMulti2(const torch::Tensor data_in);
    void ForwardMulti3(const torch::Tensor data_in);
    void ForwardMulti4(const torch::Tensor data_in);
    torch::Tensor Forward(const torch::Tensor data_in);


    int input_channels;
    int output_channels;
    bool hasNewWeights = false;
    bool use_elu       = false;

   private:
    int image_width;
    int image_height;
    int num_batches;
    torch::Tensor data_out;
    torch::Tensor data_tmp;

    torch::Tensor weight1;
    torch::Tensor bias1;

    torch::Tensor weight2;
    torch::Tensor bias2;

    torch::Tensor weight3;
    torch::Tensor bias3;

    torch::Tensor weight4;
    torch::Tensor bias4;
};