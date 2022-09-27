#pragma once

#include "models/Pipeline.h"

class UNetTest
{
   public:
    UNetTest();
    ~UNetTest() {}

    void runTest();


   private:
     MyConvUnet2d myConv;
     torch::nn::Sequential feature_transform;
     torch::nn::Sequential mask_transform;
     CUDA::CudaTimerSystem timer_system;

     std::string GetCurrentTimeForFileName()
     {
         auto time = std::time(nullptr);
         std::stringstream ss;
         ss << std::put_time(std::localtime(&time), "%F_%T");
         auto s = ss.str();
         std::replace(s.begin(), s.end(), ':', '-');
         return s;
     }
};