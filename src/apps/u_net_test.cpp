#include "u_net_test.h"

const int in_channels = 32;
const int out_channels = 60;
const int kernel_size = 3;
const int stride = 1;
const int dilation = 1;
const int n_pad_pxl = 1;
const int width = 480;
const int height = 270;

const int num_cycles = 100;


UNetTest::UNetTest(){

    //GateBlock
    auto feature_conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                                              .stride(stride)
                                              .dilation(dilation)
                                              .padding(n_pad_pxl));
    feature_transform->push_back(feature_conv);
    feature_transform->push_back(ActivationFromString("elu"));
    feature_transform->to(torch::kFloat16);
    feature_transform->to(torch::kCUDA);

    auto mask_conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                                           .stride(stride)
                                           .dilation(dilation)
                                           .padding(n_pad_pxl));
    mask_transform->push_back(mask_conv);
    mask_transform->push_back(torch::nn::Sigmoid());
    mask_transform->to(torch::kFloat16);
    mask_transform->to(torch::kCUDA);


    //MyBlock
    myConv    = MyConvUnet2d(in_channels, out_channels, "elu");
    if (!myConv.hasNewWeights)
    {
        myConv.PushNewWeights(feature_conv->weight, feature_conv->bias, mask_conv->weight, mask_conv->bias);
    }
}

void UNetTest::runTest()
{
    timer_system.Reset();

    float max_div                  = 0;
    float avg_diff                 = 0;
    int count_diff_over_threshhold = 0;
    int count_diff_zero            = 0;

    for (int cycle = 0; cycle < num_cycles; ++cycle)
    {
        timer_system.BeginFrame();
        at::Tensor x;
        {
            SAIGA_OPTIONAL_TIME_MEASURE("init", (&timer_system));
            x = torch::rand({1, in_channels, width, height},torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat16));
        }
        at::Tensor res_torch;
        {
            SAIGA_OPTIONAL_TIME_MEASURE("torch_conv", (&timer_system));
            auto x_t       = feature_transform->forward(x);
            auto m_t       = mask_transform->forward(x);
            res_torch = x_t * m_t;
        }

        at::Tensor res_my;
        {
            SAIGA_OPTIONAL_TIME_MEASURE("my_conv", (&timer_system));
            res_my = myConv.Forward(x);
        }

        {
            SAIGA_OPTIONAL_TIME_MEASURE("transform", (&timer_system));
            res_torch = res_torch.to(torch::kFloat32);
            res_torch = res_torch.to(torch::kCPU);
            res_my    = res_my.to(torch::kFloat32);
            res_my    = res_my.to(torch::kCPU);
        }

        {
            SAIGA_OPTIONAL_TIME_MEASURE("evaluate", (&timer_system));
            for (int i = 0; i < out_channels * width * height; ++i)
            {
                float diff = res_my.data_ptr<float>()[i] - res_torch.data_ptr<float>()[i];
                if (diff < 0)
                {
                    diff *= -1;
                }
                avg_diff += diff;
                if (diff > 0.001f)
                {
                    count_diff_over_threshhold++;
                }
                if (diff == 0)
                {
                    count_diff_zero++;
                }
                if (diff > max_div)
                {
                    max_div = diff;
                }
            }
        }
        timer_system.EndFrame();
        std::cout << "\rProgress:" <<((int)((cycle + 1)*100.0f/num_cycles)) <<"%";
    }
    avg_diff = avg_diff/(out_channels*width*height*num_cycles);
    count_diff_over_threshhold = count_diff_over_threshhold/num_cycles;
    count_diff_zero = count_diff_zero/num_cycles;

    std::cout<<"\nEvaluation Done!\n" << std::endl;
    std::cout<<"Max Diff: "<< max_div <<"!" << std::endl;
    std::cout<<"Avg Diff: " << avg_diff <<"!" << std::endl;
    std::cout<<"Diff over Threshhold: "<< count_diff_over_threshhold <<"!" << std::endl;
    std::cout<<"Diff equal 0: "<< count_diff_zero <<"!" << std::endl;
    std::cout<<"Total Values: "<< out_channels*width*height <<"!" << std::endl;

    timer_system.PrintTable(std::cout);

    //Store Result
    {
        std::filesystem::create_directories("u_net_test/");
        std::string date = GetCurrentTimeForFileName() + "/";
        std::filesystem::create_directories("u_net_test/" + date);
        std::string result_directory =  "u_net_test/" + date;

        std::ofstream file(result_directory + "result.txt");
        if (file.is_open())
        {
            file << "U-Net Test"
                 << "\n"
                 << "\n"
                 << "Date: " << GetCurrentTimeForFileName() << "\n"
                 << "Max Diff: "<< max_div <<"!" << "\n"
                 << "Avg Diff: " << avg_diff <<"!" << "\n"
                 << "Diff over Threshhold: "<< count_diff_over_threshhold <<"!" << "\n"
                 << "Diff equal 0: "<< count_diff_zero <<"!" << "\n"
                 << "Total Values: "<< out_channels*width*height <<"!" << "\n"
                 << "\n"
                 << "\n"
                 << "Result:\n"
                 << "\n";
            timer_system.PrintTable(file);
            file << "\n";
            file.close();
        }
    }
}




int main(int argc, char* argv[])
{
    UNetTest test;
    test.runTest();
}