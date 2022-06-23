#pragma once

#include "data/Dataset.h"
#include "models/Pipeline.h"
#include "opengl/RealTimeRenderer.h"

class SpeedMeasurement
{
    public:
    SpeedMeasurement();
     ~SpeedMeasurement() {}

    void runTest();

    //Default Test Parameters
    std::string scene_dir = "scenes/tt_playground";
    float render_scale    = 0.5f;
    int num_cycles        = 100;
    int skip_cycles       = 5;
    int selected_cam      = 0;
    int num_cams          = 16;
    bool all_cams         = false;
    int batch_size        = 4;

    private:
    std::shared_ptr<SceneData> scene;
    CUDA::CudaTimerSystem timer_system;
    std::shared_ptr<NeuralPipeline> pipeline;
    std::shared_ptr<NeuralScene> ns;
    std::string result_directory;

    void initTest();

    void checkTestParameters();

    void initPipelineAndNeuralScene();

    std::string initResultDirectory();

    void test();

    void storeResult(double time_render, int batch, int curr_batch_size);

    void storeOverallResult(double time_ges);

    // Returns  current systemtime for filenames
    std::string GetCurrentTimeForFileName()
    {
        auto time = std::time(nullptr);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time), "%F_%T");
        auto s = ss.str();
        std::replace(s.begin(), s.end(), ':', '-');
        return s;
    }

    // Devides string with given delimiter
    const std::vector<std::string> explode(const std::string& s, const char& c)
    {
        std::string buff{""};
        std::vector<std::string> v;

        for (auto n : s)
        {
            if (n != c)
                buff += n;
            else if (n == c && buff != "")
            {
                v.push_back(buff);
                buff = "";
            }
        }
        if (buff != "") v.push_back(buff);

        return v;
    }
};