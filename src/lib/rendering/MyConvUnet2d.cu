/**
 * Copyright (c) 2021 Darius R�ckert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// TODO: SHORTEN INCLUDES
#pragma once
#include "saiga/cuda/reduce.h"
#include "saiga/vision/torch/CudaHelper.h"

#include "MyConvUnet2d.h"

#include "cooperative_groups.h"

#ifdef CUDA_DEBUG
#    define CUDA_DEBUG_ASSERT(_x) CUDA_KERNEL_ASSERT(_x)
#else
#    define CUDA_DEBUG_ASSERT(_x)
#endif

#include "saiga/cuda/shfl_helper.h"
#include "saiga/cuda/reduce.h"

/*
const int stride        = 1;
const int dilation      = 1;
const int padding       = 1;
*/


const int kernel_size   = 3;//odd
const int kernel_radius = (kernel_size-1)/2;
const int kernel_size_sqared = kernel_size * kernel_size;
const dim3 block_dim = dim3(16,16,1);

const int work_size_x = 14;
const int work_size_x_2 = work_size_x + 2;
const int work_size_y = 14;
const int work_size_y_2 = work_size_y + 2;
const int work_size_squared = work_size_x*work_size_y;

struct ForwardRenderParams
{
    StaticDeviceTensor<float, 4> weight;
    StaticDeviceTensor<float, 1> bias;
};

static __device__ __constant__ ForwardRenderParams d_forward_params;
static __device__ __constant__ __half2 weights[100];


__global__ void PreprocessParameters1(StaticDeviceTensor<__half, 4> out_weight, StaticDeviceTensor<__half, 1> out_bias,
                                     StaticDeviceTensor<__half, 4> forward_weight,StaticDeviceTensor<__half, 4> mask_weight,
                                     StaticDeviceTensor<__half, 1> forward_bias, StaticDeviceTensor<__half, 1> mask_bias)
{
    int out_layer = blockIdx.x;
    int in_layer = blockIdx.y;
    int dx = threadIdx.x;
    int dy = threadIdx.y;
    __half* dest1 = &(out_weight(out_layer, in_layer, dy, dx * 2));
    dest1[0]      = forward_weight(out_layer, in_layer, dy, dx);
    __half* dest2 = &(out_weight(out_layer, in_layer, dy, dx * 2 + 1));
    dest2[0]      = mask_weight(out_layer, in_layer, dy, dx);
    if(in_layer == 0 && dx == 0 && dy == 0){
        __half* dest3 = &(out_bias(out_layer * 2));
        dest3[0]      = forward_bias(out_layer);
        __half* dest4 = &(out_bias(out_layer * 2 + 1));
        dest4[0]      = mask_bias(out_layer);
    }
}

template <int in_channels, int out_channels, bool use_elu>
__global__ void Forward1(StaticDeviceTensor<__half, 4> in, StaticDeviceTensor<__half, 4> out, int num_batches,
                         int width, int height , StaticDeviceTensor<__half, 4> in_weight,
                         StaticDeviceTensor<__half, 1> in_bias)
{
    int gx = blockIdx.x * blockDim.x + threadIdx.x - 1;
    int gy = blockIdx.y * blockDim.y + threadIdx.y - 1;
    ushort out_layer = blockIdx.z;

    __shared__ __half2 shared_weight[in_channels][kernel_size_sqared];
    __shared__ __half2 shared_bias;
    ushort idx = threadIdx.x + threadIdx.y * blockDim.x;
    if(idx < in_channels){
        for (ushort k = 0; k < kernel_size; k++)
        {
            reinterpret_cast<int3*>(shared_weight[idx])[k] = reinterpret_cast<int3*>(&in_weight(out_layer, idx, k))[0];
        }
    }else if(idx == in_channels){
        shared_bias = reinterpret_cast<__half2*>(&in_bias(out_layer*2))[0];
    }
    __syncthreads();

    if (gx + 1 >= width || gy + 1 >= height)
    {
        return;
    }
    for (int batch = 0; batch < num_batches; batch++)
    {
        __half2 sum;
        sum.x = 0;
        sum.y = 0;
        if (gx == -1 || gy == -1 || gx == width - 2 || gy == height - 2)
        {
#pragma unroll 4
            for (int in_layer = 0; in_layer < in_channels; ++in_layer)
            {
                for (ushort dy = 0; dy < kernel_size; dy++)
                {
                    __half2 local_weight[kernel_size];
                    reinterpret_cast<int3*>(local_weight)[0] = reinterpret_cast<int3*>(shared_weight[in_layer])[dy];
                    int ly                                   = gy + dy;
                    for (ushort dx = 0; dx < kernel_size; dx++)
                    {
                        int lx = gx + dx;
                        if (lx >= 0 && ly >= 0 && lx < width && ly < height)
                        {
                            __half2 value;
                            value.x = in(batch, in_layer, ly, lx);
                            value.y = in(batch, in_layer, ly, lx);
                            // Weight
                            sum += (local_weight[dx] * value);
                        }
                    }
                }
            }
        }
        else
        {
            __half value[3];
#pragma unroll 4
            for (int in_layer = 0; in_layer < in_channels; ++in_layer)
            {
                for (ushort dy = 0; dy < kernel_size; dy++)
                {
                    __half2 local_weight[kernel_size];
                    reinterpret_cast<int3*>(local_weight)[0] = reinterpret_cast<int3*>(shared_weight[in_layer])[dy];
                    int ly                                   = gy + dy;

                    value[0] = in(batch, in_layer, ly, gx);
                    value[1] = in(batch, in_layer, ly, gx + 1);
                    value[2] = in(batch, in_layer, ly, gx + 2);

                    // Weight
                    sum.x += (local_weight[0].x * value[0]);
                    sum.y += (local_weight[0].y * value[0]);
                    sum.x += (local_weight[1].x * value[1]);
                    sum.y += (local_weight[1].y * value[1]);
                    sum.x += (local_weight[2].x * value[2]);
                    sum.y += (local_weight[2].y * value[2]);
                }
            }
        }
        // Bias
        sum.x += shared_bias.x;
        sum.y += shared_bias.y;

        // Feature Elu
        if (use_elu)
        {
            if (sum.x <= (__half)0)
            {
                sum.x = hexp(sum.x) - (__half)1;
            }
        }

        // mask Sigmoid
        sum.y = (__half)1 / (hexp(-sum.y) + (__half)1);


        // Output
        __half* dest = &(out(batch, out_layer, gy + 1, gx + 1));
        dest[0]      = sum.x * sum.y;
    }
}


__global__ void PreprocessParameters2(StaticDeviceTensor<float, 4> out_weight, StaticDeviceTensor<float, 1> out_bias,
                                      StaticDeviceTensor<__half, 4> forward_weight,StaticDeviceTensor<__half, 4> mask_weight,
                                      StaticDeviceTensor<__half, 1> forward_bias, StaticDeviceTensor<__half, 1> mask_bias)
{
    int out_layer = blockIdx.x;
    int in_layer = blockIdx.y;
    int dx = threadIdx.x;
    int dy = threadIdx.y;
    __half2 weight_in;
    weight_in.x = forward_weight(out_layer, in_layer, dy, dx);
    weight_in.y = mask_weight(out_layer, in_layer, dy, dx);
    __half2* weight_dest = reinterpret_cast<__half2*>(&(out_weight(out_layer, dx, dy, in_layer)));
    weight_dest[0]      = weight_in;
    if(in_layer == 0 && dx == 0 && dy == 0){
        __half2 bias_in;
        bias_in.x = forward_bias(out_layer);
        bias_in.y = mask_bias(out_layer);
        __half2* bias_dest = reinterpret_cast<__half2*>(&(out_bias(out_layer)));
        bias_dest[0]      = bias_in;
    }
}

template <int in_channels, int out_channels>
__global__ void Forward2(StaticDeviceTensor<__half, 4> in, StaticDeviceTensor<float, 4> out, int num_batches, int width, int height)
{

    int out_x = threadIdx.x%work_size_x_2;//[0-15]
    int out_y = threadIdx.x/work_size_x_2 + 2*threadIdx.y;//[0-1]
    int out_gx = (int)blockIdx.x * work_size_x - 1 + out_x;
    int out_gy = (int)blockIdx.y * work_size_y - 1 + out_y;
    int out_layer = blockIdx.z;

    __half2 weights[9];
    for (int k = 0; k < 9; k++)
    {
        weights[k] = reinterpret_cast<__half2*>(&d_forward_params.weight(out_layer, k/3, k%3, (int)threadIdx.x))[0];
    }

    for (int batch = 0; batch < num_batches; batch++)
    {
        __shared__ __half2 shared_out[work_size_y_2][work_size_x_2];
        shared_out[out_y][out_x] = __half2(0, 0);

        __syncthreads();

        for (int offset = threadIdx.y; offset < work_size_squared; offset +=blockDim.y)
        {
            int gx = (int)blockIdx.x * work_size_x + offset/work_size_y;
            if(gx>=0 && gx<width)
            {
                int gy = (int)blockIdx.y * work_size_y + offset%work_size_y;
                if (gy>=0 && gy < height)
                {
                    __half data_in =in(batch, gx, gy, (int)threadIdx.x);

                    __half2 data_out[9];
                    for (int k = 0; k < 9; k++)
                    {
                        data_out[k].x = data_in * weights[k].x;
                        data_out[k].y = data_in * weights[k].y;
                        data_out[k]   = Saiga::CUDA::warpReduceSum<__half2, 32, true, int>(data_out[k]);
                    }
                    if (threadIdx.x < 9)
                    {
                        //TODO ggf atomic
                        atomicAdd(&shared_out[offset%work_size_y + 2 - threadIdx.x %3][offset/work_size_y + 2 - threadIdx.x /3], data_out[threadIdx.x]);
                        //shared_out[offset%work_size_y + 2 - dy][offset/work_size_y + 2 - dx] += weights;
                    }
                }

            }
        }
        __syncthreads();
        if (out_gx >= 0 && out_gx < width && out_gy < height && out_gy >= 0)
        {
            atomicAdd(reinterpret_cast<__half2*>(&out(batch, out_layer, out_gy, out_gx)),
                      shared_out[out_y][out_x]);
        }

    }
}

template <bool use_elu>
__global__ void Postprocess2(StaticDeviceTensor<float, 4> in, StaticDeviceTensor<__half, 4> out, int num_batches, int width, int height)
{

    const int gx = blockIdx.x * blockDim.x + threadIdx.x;
    const int gy = blockIdx.y * blockDim.y + threadIdx.y;
    if (gx >= width || gy >= height)
    {
        return;
    }

    int out_layer = blockIdx.z;
    half2 bias    = reinterpret_cast<__half2*>(&d_forward_params.bias(out_layer))[0];

    for (int batch = 0; batch < num_batches; batch++)
    {
        half2 data_in = reinterpret_cast<__half2*>(&in(batch, out_layer, gy, gx))[0];
        // Bias
        data_in += bias;

        // Feature Elu
        if (use_elu)
        {
            if (data_in.x <= (__half)0)
            {
                data_in.x = hexp(data_in.x) - (__half)1;
            }
        }

        // mask Sigmoid
        data_in.y = (__half)1 / (hexp(-data_in.y) + (__half)1);

        // Output
        __half* dest = &(out(batch, out_layer, gy, gx));
        dest[0]      = data_in.x * data_in.y;
    }
}



__global__ void PreprocessParameters3(StaticDeviceTensor<float, 4> out_weight, StaticDeviceTensor<float, 1> out_bias,
                                      StaticDeviceTensor<__half, 4> forward_weight,StaticDeviceTensor<__half, 4> mask_weight,
                                      StaticDeviceTensor<__half, 1> forward_bias, StaticDeviceTensor<__half, 1> mask_bias)
{
    int out_layer = blockIdx.x;
    int in_layer = blockIdx.y;
    int dx = threadIdx.x;
    int dy = threadIdx.y;
    __half2 weight_in;
    weight_in.x = forward_weight(out_layer, in_layer, dy, dx);
    weight_in.y = mask_weight(out_layer, in_layer, dy, dx);
    __half2* weight_dest = reinterpret_cast<__half2*>(&(out_weight(dx, dy, out_layer, in_layer)));
    weight_dest[0]      = weight_in;
    if(in_layer == 0 && dx == 0 && dy == 0){
        __half2 bias_in;
        bias_in.x = forward_bias(out_layer);
        bias_in.y = mask_bias(out_layer);
        __half2* bias_dest = reinterpret_cast<__half2*>(&(out_bias(out_layer)));
        bias_dest[0]      = bias_in;
    }
}

template <int output_channels, int calc_size, int work_size, int calc_width>
__global__ void Forward3(StaticDeviceTensor<__half, 4> in, StaticDeviceTensor<float, 4> out, int width, int height , StaticDeviceTensor<float, 4> in_weight)
{
    const int gx = blockIdx.x * 8;
    const int gy = blockIdx.y * 4;
    const int in_layer = threadIdx.x;
    if (gx > width || gy > height)
    {
        return;
    }
    __shared__ __half shared_in[12][6][32];
    if(gx==0 || gy == 0 || gx >width - 12 || gy > height - 6){
        for (int dx = 0; dx < 12; dx++)
        {
            __half value = 0;
            int lx       = gx - 1 + dx;
            int ly       = gy - 1 + threadIdx.y;
            if (lx >= 0 && ly >= 0 && lx < width && ly < height)
            {
                value = in((int)blockIdx.z, lx, ly, in_layer);
            }
            shared_in[dx][threadIdx.y][in_layer] = value;
        }
        __syncthreads();
        for (int out_layer = threadIdx.y; out_layer < output_channels; out_layer += calc_size)
        {
            __half2 weights[kernel_size][kernel_size];
            for (int dx = 0; dx < kernel_size; dx++)
            {
                for (int dy = 0; dy < kernel_size; dy++)
                {
                    weights[dx][dy] = reinterpret_cast<__half2*>(&in_weight(dx, dy, out_layer, in_layer))[0];
                }
            }
            half2 sum[32];
            for (int k = 0; k < 32; k++)
            {
                sum[k].x = 0;
                sum[k].y = 0;
            }
            for (int dx = 0; dx < kernel_size; dx++)
            {
                for (int dy = 0; dy < kernel_size; dy++)
                {
                    for (int k = 0; k < 32; k++)
                    {
                        sum[k].x += (weights[dx][dy].x * shared_in[dx + k / work_size][dy + k % work_size][in_layer]);
                        sum[k].y += (weights[dx][dy].y * shared_in[dx + k / work_size][dy + k % work_size][in_layer]);
                    }
                }
            }
            for (int k = 0; k < 32; k++)
            {
                sum[k] = Saiga::CUDA::warpReduceSum<__half2, 32, true, int>(sum[k]);
            }
            if (threadIdx.x == 0)
            {
                for (int k = 0; k < 32; k++)
                {
                    reinterpret_cast<__half2*>(
                        &out((int)blockIdx.z, out_layer, gy + k % work_size, gx + k / work_size))[0] = sum[k];
                }
            }
        }
    }else
    {
        for (int dx = 0; dx < 12; dx++)
        {
            int lx       = gx - 1 + dx;
            int ly       = gy - 1 + threadIdx.y;
            shared_in[dx][threadIdx.y][in_layer] = in((int)blockIdx.z, lx, ly, in_layer);
        }
        __syncthreads();
        for (int out_layer = threadIdx.y; out_layer < output_channels; out_layer += calc_size)
        {
            __half2 weights[kernel_size][kernel_size];
            for (int dx = 0; dx < kernel_size; dx++)
            {
                for (int dy = 0; dy < kernel_size; dy++)
                {
                    weights[dx][dy] = reinterpret_cast<__half2*>(&in_weight(dx, dy, out_layer, in_layer))[0];
                }
            }
            half2 sum[32];
            for (int k = 0; k < 32; k++)
            {
                sum[k].x = 0;
                sum[k].y = 0;
            }
            for (int dx = 0; dx < kernel_size; dx++)
            {
                for (int dy = 0; dy < kernel_size; dy++)
                {
                    for (int k = 0; k < 8; k++)
                    {
                        int tmp = dx + k;
                        for (int l = 0; l < 4; l++)
                        {
                            int tmp2 = dy + l;
                            sum[k*4+l].x += (weights[dx][dy].x * shared_in[tmp][tmp2][in_layer]);
                            sum[k*4+l].y +=(weights[dx][dy].y * shared_in[tmp][tmp2][in_layer]);
                        }
                    }
                }
            }
            for (int k = 0; k < 32; k++)
            {
                sum[k] = Saiga::CUDA::warpReduceSum<__half2, 32, false, int>(sum[k]);
            }
            if (threadIdx.x == 0)
            {
                for (int k = 0; k < 8; k++)
                {
                    int tmp = gx + k;
                    for (int l = 0; l < 4; l++)
                    {
                        int tmp2 = gy + l;
                        reinterpret_cast<__half2*>(
                            &out((int)blockIdx.z, out_layer, tmp2, tmp))[0] = sum[k*4+l];
                    }
                }
            }
        }
    }

}

template <bool use_elu>
__global__ void Postprocess3(StaticDeviceTensor<float, 4> in, StaticDeviceTensor<__half, 4> out, int num_batches, int width, int height, StaticDeviceTensor<float, 1> in_bias)
{
    const int gx = blockIdx.x * blockDim.x + threadIdx.x;
    const int gy = blockIdx.y * blockDim.y + threadIdx.y;
    if (gx >= width || gy >= height)
    {
        return;
    }

    int out_layer = blockIdx.z;
    half2 bias    = reinterpret_cast<__half2*>(&in_bias(out_layer))[0];

    for (int batch = 0; batch < num_batches; batch++)
    {
        half2 data_in = reinterpret_cast<__half2*>(&in(batch, out_layer, gy, gx))[0];
        // Bias
        data_in += bias;

        // Feature Elu
        if (use_elu)
        {
            if (data_in.x <= (__half)0)
            {
                data_in.x = hexp(data_in.x) - (__half)1;
            }
        }

        // mask Sigmoid
        data_in.y = (__half)1 / (hexp(-data_in.y) + (__half)1);

        // Output
        out(batch, out_layer, gy, gx) = data_in.x * data_in.y;
    }
}


void MyConvUnet2d::ForwardMulti1(const torch::Tensor  data_in)
{
    int bx = iDivUp(image_width, block_dim.x);
    int by = iDivUp(image_height, block_dim.y);
    SAIGA_ASSERT(bx > 0 && by > 0);

    if (use_elu)
    {
        if (input_channels == 4 && output_channels == 16)
        {
            ::Forward1<4, 16, true><<<dim3(bx, by, 16), block_dim>>>(data_in, data_out, num_batches, image_width, image_height, weight1,  bias1);
        }
        else if (input_channels == 16 && output_channels == 28)
        {
            ::Forward1<16, 28, true>
                <<<dim3(bx, by, 28), block_dim>>>(data_in, data_out, num_batches, image_width, image_height, weight1, bias1);
        }
        else if (input_channels == 32 && output_channels == 60)
        {
            ::Forward1<32, 60, true>
                <<<dim3(bx, by, 60), block_dim>>>(data_in, data_out, num_batches, image_width, image_height, weight1, bias1);
        }
        else if (input_channels == 64 && output_channels == 124)
        {
            ::Forward1<64, 124, true>
                <<<dim3(bx, by, 124), block_dim>>>(data_in, data_out, num_batches, image_width, image_height, weight1,  bias1);
        }
        else if (input_channels == 128 && output_channels == 64)
        {
            ::Forward1<128, 64, true>
                <<<dim3(bx, by, 64), block_dim>>>(data_in, data_out, num_batches, image_width, image_height, weight1,  bias1);
        }
        else if (input_channels == 64 && output_channels == 32)
        {
            ::Forward1<64, 32, true>
                <<<dim3(bx, by, 32), block_dim>>>(data_in, data_out, num_batches, image_width, image_height, weight1, bias1);
        }
        else if (input_channels == 32 && output_channels == 16)
        {
            ::Forward1<32, 16, true>
                <<<dim3(bx, by, 16), block_dim>>>(data_in, data_out, num_batches, image_width, image_height, weight1,  bias1);
        }
        else
        {
            std::cout << "IN : " << input_channels << "; OUT: " << output_channels << std::endl;
            SAIGA_EXIT_ERROR("invalid number of channels! ");
        }
    }
    else
    {
        if (input_channels == 4 && output_channels == 16)
        {
            ::Forward1<4, 16, false>
                <<<dim3(bx, by, 16), block_dim>>>(data_in, data_out, num_batches, image_width, image_height, weight1,  bias1);
        }
        else if (input_channels == 16 && output_channels == 28)
        {
            ::Forward1<16, 28, false>
                <<<dim3(bx, by, 28), block_dim>>>(data_in, data_out, num_batches, image_width, image_height, weight1,  bias1);
        }
        else if (input_channels == 32 && output_channels == 60)
        {
            ::Forward1<32, 60, false>
                <<<dim3(bx, by, 60), block_dim>>>(data_in, data_out, num_batches, image_width, image_height, weight1,  bias1);
        }
        else if (input_channels == 64 && output_channels == 124)
        {
            ::Forward1<64, 124, false>
                <<<dim3(bx, by, 124), block_dim>>>(data_in, data_out, num_batches, image_width, image_height, weight1,  bias1);
        }
        else if (input_channels == 128 && output_channels == 64)
        {
            ::Forward1<128, 64, false>
                <<<dim3(bx, by, 64), block_dim>>>(data_in, data_out, num_batches, image_width, image_height, weight1,  bias1);
        }
        else if (input_channels == 64 && output_channels == 32)
        {
            ::Forward1<64, 32, false>
                <<<dim3(bx, by, 32), block_dim>>>(data_in, data_out, num_batches, image_width, image_height, weight1,  bias1);
        }
        else if (input_channels == 32 && output_channels == 16)
        {
            ::Forward1<32, 16, false>
                <<<dim3(bx, by, 16), block_dim>>>(data_in, data_out, num_batches, image_width, image_height, weight1, bias1);
        }
        else
        {
            std::cout << "IN : " << input_channels << "; OUT: " << output_channels << std::endl;
            SAIGA_EXIT_ERROR("invalid number of channels! ");
        }
    }

    CUDA_SYNC_CHECK_ERROR();
}

void MyConvUnet2d::ForwardMulti2(const torch::Tensor  data_in)
{
    int bx = iDivUp(image_width, block_dim.x);
    int by = iDivUp(image_height, block_dim.y);
    SAIGA_ASSERT(bx > 0 && by > 0);

    if (use_elu)
    {
        if (input_channels == 4 && output_channels == 16)
        {
            ::Forward1<4, 16, true><<<dim3(bx, by, 16), block_dim>>>(data_in, data_out, num_batches, image_width, image_height, weight1,  bias1);
        }
        else if (input_channels == 16 && output_channels == 28)
        {
            ::Forward1<16, 28, true>
                <<<dim3(bx, by, 28), block_dim>>>(data_in, data_out, num_batches, image_width, image_height, weight1, bias1);
        }
        else if (input_channels == 32 && output_channels == 60)
        {
            torch::Tensor permuted_data_in = data_in.permute({0, 3, 2, 1}).contiguous();
            //PUSH PARAMS
            ForwardRenderParams params;
            params.weight = weight2;
            params.bias   = bias2;
            CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_forward_params, &params, sizeof(params)));
            CUDA_SYNC_CHECK_ERROR();
            ::Forward2<32, 60><<<dim3(iDivUp(image_width, work_size_x), iDivUp(image_height, work_size_y), output_channels), dim3(32,8,1)>>>(permuted_data_in, data_tmp, num_batches, image_width, image_height);
            ::Postprocess2<true><<<dim3(iDivUp(image_width, 32), iDivUp(image_height, 8), 60), dim3(32, 8)>>>(data_tmp, data_out, num_batches, image_width, image_height);
        }
        else if (input_channels == 64 && output_channels == 124)
        {
            ::Forward1<64, 124, true>
                <<<dim3(bx, by, 124), block_dim>>>(data_in, data_out, num_batches, image_width, image_height, weight1,  bias1);
        }
        else if (input_channels == 128 && output_channels == 64)
        {
            ::Forward1<128, 64, true>
                <<<dim3(bx, by, 64), block_dim>>>(data_in, data_out, num_batches, image_width, image_height, weight1,  bias1);
        }
        else if (input_channels == 64 && output_channels == 32)
        {
            ::Forward1<64, 32, true>
                <<<dim3(bx, by, 32), block_dim>>>(data_in, data_out, num_batches, image_width, image_height, weight1, bias1);
        }
        else if (input_channels == 32 && output_channels == 16)
        {
            ::Forward1<32, 16, true>
                <<<dim3(bx, by, 16), block_dim>>>(data_in, data_out, num_batches, image_width, image_height, weight1,  bias1);
        }
        else
        {
            std::cout << "IN : " << input_channels << "; OUT: " << output_channels << std::endl;
            SAIGA_EXIT_ERROR("invalid number of channels! ");
        }
    }
    else
    {
        if (input_channels == 4 && output_channels == 16)
        {
            ::Forward1<4, 16, false>
                <<<dim3(bx, by, 16), block_dim>>>(data_in, data_out, num_batches, image_width, image_height, weight1,  bias1);
        }
        else if (input_channels == 16 && output_channels == 28)
        {
            ::Forward1<16, 28, false>
                <<<dim3(bx, by, 28), block_dim>>>(data_in, data_out, num_batches, image_width, image_height, weight1,  bias1);
        }
        else if (input_channels == 32 && output_channels == 60)
        {
            ::Forward1<32, 60, false>
                <<<dim3(bx, by, 60), block_dim>>>(data_in, data_out, num_batches, image_width, image_height, weight1,  bias1);
        }
        else if (input_channels == 64 && output_channels == 124)
        {
            ::Forward1<64, 124, false>
                <<<dim3(bx, by, 124), block_dim>>>(data_in, data_out, num_batches, image_width, image_height, weight1,  bias1);
        }
        else if (input_channels == 128 && output_channels == 64)
        {
            ::Forward1<128, 64, false>
                <<<dim3(bx, by, 64), block_dim>>>(data_in, data_out, num_batches, image_width, image_height, weight1,  bias1);
        }
        else if (input_channels == 64 && output_channels == 32)
        {
            ::Forward1<64, 32, false>
                <<<dim3(bx, by, 32), block_dim>>>(data_in, data_out, num_batches, image_width, image_height, weight1,  bias1);
        }
        else if (input_channels == 32 && output_channels == 16)
        {
            ::Forward1<32, 16, false>
                <<<dim3(bx, by, 16), block_dim>>>(data_in, data_out, num_batches, image_width, image_height, weight1, bias1);
        }
        else
        {
            std::cout << "IN : " << input_channels << "; OUT: " << output_channels << std::endl;
            SAIGA_EXIT_ERROR("invalid number of channels! ");
        }
    }

    CUDA_SYNC_CHECK_ERROR();
}

void MyConvUnet2d::ForwardMulti3(const torch::Tensor  data_in)
{
    int bx = iDivUp(image_width, block_dim.x);
    int by = iDivUp(image_height, block_dim.y);
    SAIGA_ASSERT(bx > 0 && by > 0);

    if (use_elu)
    {
        if (input_channels == 4 && output_channels == 16)
        {
            ::Forward1<4, 16, true><<<dim3(bx, by, 16), block_dim>>>(data_in, data_out, num_batches, image_width, image_height, weight1,  bias1);
        }
        else if (input_channels == 16 && output_channels == 28)
        {
            ::Forward1<16, 28, true>
                <<<dim3(bx, by, 28), block_dim>>>(data_in, data_out, num_batches, image_width, image_height, weight1, bias1);
        }
        else if (input_channels == 32 && output_channels == 60)
        {
            torch::Tensor permuted_data_in = data_in.permute({0, 3, 2, 1}).contiguous();
            const int calc_size = 6;    //How large is the bounding rectangle
            const int work_size = calc_size-2; // How large is the cumpute rectangle
            const int calc_width = 2;   //Scale the width of the compute/bounding rectangle
            ::Forward3<60, calc_size, work_size, calc_width><<<dim3(iDivUp(image_width, (work_size*calc_width)), iDivUp(image_height, work_size), num_batches), dim3(32,calc_size,1)>>>(permuted_data_in, data_tmp, image_width, image_height, weight3);
            ::Postprocess3<true><<<dim3(iDivUp(image_width, 32), iDivUp(image_height, 8), output_channels), dim3(32, 8)>>>(data_tmp, data_out, num_batches, image_width, image_height, bias3);
        }
        else if (input_channels == 64 && output_channels == 124)
        {
            ::Forward1<64, 124, true>
                <<<dim3(bx, by, 124), block_dim>>>(data_in, data_out, num_batches, image_width, image_height, weight1,  bias1);
        }
        else if (input_channels == 128 && output_channels == 64)
        {
            ::Forward1<128, 64, true>
                <<<dim3(bx, by, 64), block_dim>>>(data_in, data_out, num_batches, image_width, image_height, weight1,  bias1);
        }
        else if (input_channels == 64 && output_channels == 32)
        {
            ::Forward1<64, 32, true>
                <<<dim3(bx, by, 32), block_dim>>>(data_in, data_out, num_batches, image_width, image_height, weight1, bias1);
        }
        else if (input_channels == 32 && output_channels == 16)
        {
            ::Forward1<32, 16, true>
                <<<dim3(bx, by, 16), block_dim>>>(data_in, data_out, num_batches, image_width, image_height, weight1,  bias1);
        }
        else
        {
            std::cout << "IN : " << input_channels << "; OUT: " << output_channels << std::endl;
            SAIGA_EXIT_ERROR("invalid number of channels! ");
        }
    }
    else
    {
        if (input_channels == 4 && output_channels == 16)
        {
            ::Forward1<4, 16, false>
                <<<dim3(bx, by, 16), block_dim>>>(data_in, data_out, num_batches, image_width, image_height, weight1,  bias1);
        }
        else if (input_channels == 16 && output_channels == 28)
        {
            ::Forward1<16, 28, false>
                <<<dim3(bx, by, 28), block_dim>>>(data_in, data_out, num_batches, image_width, image_height, weight1,  bias1);
        }
        else if (input_channels == 32 && output_channels == 60)
        {
            ::Forward1<32, 60, false>
                <<<dim3(bx, by, 60), block_dim>>>(data_in, data_out, num_batches, image_width, image_height, weight1,  bias1);
        }
        else if (input_channels == 64 && output_channels == 124)
        {
            ::Forward1<64, 124, false>
                <<<dim3(bx, by, 124), block_dim>>>(data_in, data_out, num_batches, image_width, image_height, weight1,  bias1);
        }
        else if (input_channels == 128 && output_channels == 64)
        {
            ::Forward1<128, 64, false>
                <<<dim3(bx, by, 64), block_dim>>>(data_in, data_out, num_batches, image_width, image_height, weight1,  bias1);
        }
        else if (input_channels == 64 && output_channels == 32)
        {
            ::Forward1<64, 32, false>
                <<<dim3(bx, by, 32), block_dim>>>(data_in, data_out, num_batches, image_width, image_height, weight1,  bias1);
        }
        else if (input_channels == 32 && output_channels == 16)
        {
            ::Forward1<32, 16, false>
                <<<dim3(bx, by, 16), block_dim>>>(data_in, data_out, num_batches, image_width, image_height, weight1, bias1);
        }
        else
        {
            std::cout << "IN : " << input_channels << "; OUT: " << output_channels << std::endl;
            SAIGA_EXIT_ERROR("invalid number of channels! ");
        }
    }

    CUDA_SYNC_CHECK_ERROR();
}


MyConvUnet2d::MyConvUnet2d(int input_channels, int output_channels, std::string activation_str)
{
    this->input_channels = input_channels;
    this->output_channels = output_channels;
    if (activation_str == "elu")
    {
        use_elu = true;
    }
    else if (activation_str != "id")
    {
        SAIGA_EXIT_ERROR("Unnknown activation_str!");
    }
    weight1 = torch::zeros({output_channels, input_channels, kernel_size, kernel_size*2},
                           torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat16));
    bias1 = torch::zeros({output_channels * 2},
                           torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat16));

    weight2 = torch::zeros({output_channels, kernel_size, kernel_size, input_channels},
                           torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
    bias2 = torch::zeros({output_channels},
                           torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));

    weight3 = torch::zeros({kernel_size, kernel_size,output_channels, input_channels},
                           torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
    bias3 = torch::zeros({output_channels},
                           torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));


}

void MyConvUnet2d::PushNewWeights(const torch::Tensor feature_weight, const torch::Tensor feature_bias,
                                  const torch::Tensor mask_weight, const torch::Tensor mask_bias)
{
    hasNewWeights = true;

    SAIGA_ASSERT(feature_weight.size(0) == output_channels);
    SAIGA_ASSERT(feature_weight.size(1) == input_channels);
    SAIGA_ASSERT(feature_weight.size(2) == kernel_size);
    SAIGA_ASSERT(feature_weight.size(3) == kernel_size);
    SAIGA_ASSERT(feature_bias.size(0) == output_channels);

    SAIGA_ASSERT(mask_weight.size(0) == output_channels);
    SAIGA_ASSERT(mask_weight.size(1) == input_channels);
    SAIGA_ASSERT(mask_weight.size(2) == kernel_size);
    SAIGA_ASSERT(mask_weight.size(3) == kernel_size);
    SAIGA_ASSERT(mask_bias.size(0) == output_channels);

    ::PreprocessParameters1<<<dim3(output_channels, input_channels, 1), dim3(3,3,1)>>>(weight1, bias1, feature_weight.cuda().to(torch::kFloat16), mask_weight.cuda().to(torch::kFloat16), feature_bias.cuda().to(torch::kFloat16), mask_bias.cuda().to(torch::kFloat16));
    ::PreprocessParameters2<<<dim3(output_channels, input_channels, 1), dim3(3,3,1)>>>(weight2, bias2, feature_weight.cuda().to(torch::kFloat16), mask_weight.cuda().to(torch::kFloat16), feature_bias.cuda().to(torch::kFloat16), mask_bias.cuda().to(torch::kFloat16));
    ::PreprocessParameters3<<<dim3(output_channels, input_channels, 1), dim3(3,3,1)>>>(weight3, bias3, feature_weight.cuda().to(torch::kFloat16), mask_weight.cuda().to(torch::kFloat16), feature_bias.cuda().to(torch::kFloat16), mask_bias.cuda().to(torch::kFloat16));

    CUDA_SYNC_CHECK_ERROR();
}

torch::Tensor MyConvUnet2d::Forward(const torch::Tensor data_in)
{
    //READ SIZES
    num_batches = data_in.size(0);
    SAIGA_ASSERT(data_in.size(1) == input_channels);
    image_height = data_in.size(2);
    image_width  = data_in.size(3);

    //INIT OUTPUT
    data_out = torch::zeros({num_batches, output_channels, image_height, image_width},
        torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat16));

    //TODO ggf nur einmal
    data_tmp = torch::zeros({num_batches, output_channels, image_height, image_width},
                            torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));

    SAIGA_ASSERT(data_in.is_cuda() == true);
    SAIGA_ASSERT(data_in.dtype() == torch::kFloat16);
    //CALCULATE OUTPUT
    ForwardMulti3(data_in);

    //RETURN OUTPUT
    return std::move(data_out);
};
