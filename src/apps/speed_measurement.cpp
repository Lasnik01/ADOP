
#include "speed_measurement.h"

SpeedMeasurement::SpeedMeasurement(){};

void SpeedMeasurement::runTest()
{
    std::cout << "\n*ADOP Speed Measurement for Scenes*\n" << std::endl;

    std::cout << "\n*Initialisizing Test*\n" << std::endl;
    initTest();

    std::cout << "\n*Starting Test For Scene " << scene_dir << "*" << std::endl;
    Timer timer_ges;
    timer_ges.start();
    test();
    timer_ges.stop();

    if (num_cams > 1)
    {
        std::cout << "\n*Saving overall result*\n" << std::endl;
        double time_ges = timer_ges.getTimeMS();
        storeOverallResult(time_ges);
    }

    std::cout << "\n\n*Stored result to " << result_directory << "*\n" << std::endl;

    std::cout << "===============" << std::endl;
    std::cout << "*Test Finished*" << std::endl;
    std::cout << "===============" << std::endl;
}

void SpeedMeasurement::initTest(){
    //Create Scene
    try
    {
        scene = std::make_shared<SceneData>(scene_dir);
    }
    catch (std::filesystem::filesystem_error e)
    {
        std::cout << "\n*Error: Can't load scene " << scene_dir << "!*\n" << std::endl;
        exit(-1);
    }

    //Check and correct test parameters
    checkTestParameters();

    //Create pipeline and neural scene
    initPipelineAndNeuralScene();

    //Create result directory
    result_directory = initResultDirectory();
};

void SpeedMeasurement::checkTestParameters(){
    if (selected_cam >= scene->frames.size())
    {
        selected_cam = scene->frames.size() - 1;
    }
    if (selected_cam < 0)
    {
        selected_cam = 0;
    }
    if (num_cams > scene->frames.size() - selected_cam)
    {
        num_cams = scene->frames.size() - selected_cam;
    }
    if (num_cams < 0)
    {
        num_cams = 0;
    }
    if (batch_size < 1)
    {
        batch_size = 1;
    }
    if (all_cams)
    {
        selected_cam = 0;
        num_cams     = scene->frames.size();
    }
};

void SpeedMeasurement::initPipelineAndNeuralScene() {
    //Read existing experiments
    std::vector<RealTimeRenderer::Experiment> experiments;
    {
        std::string experiments_base = "experiments/";
        Directory dir(experiments_base);
        auto ex_names = dir.getDirectories();
        std::sort(ex_names.begin(), ex_names.end(), std::greater<std::string>());

        for (auto n : ex_names)
        {
            RealTimeRenderer::Experiment e(experiments_base + "/" + n + "/", n, scene->scene_name);
            if (!e.eps.empty())
            {
                experiments.push_back(e);
            }
        }
    }
    auto ex = experiments[0];
    auto ep = ex.eps[(experiments.empty()) ? 0 : experiments[0].eps.size() - 1];

    //Create params with latest experiment
    std::shared_ptr<CombinedParams> params;
    {
        params = std::make_shared<CombinedParams>(ex.dir + "/params.ini");

        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        if (deviceProp.major > 6)
        {
            std::cout << "Using half_float inference" << std::endl;
            params->net_params.half_float = true;
        }

        params->pipeline_params.train             = false;
        params->render_params.render_outliers     = false;
        params->train_params.checkpoint_directory = ep.dir;

        params->train_params.loss_vgg = 0;
        params->train_params.loss_l1  = 0;
        params->train_params.loss_mse = 0;
    }

    //Create pipeline
    pipeline = std::make_shared<NeuralPipeline>(params);
    pipeline->Train(false);
    pipeline->timer_system = &timer_system;

    //Create neural scene
    ns = std::make_shared<NeuralScene>(scene, params);
    ns->to(torch::kCUDA);
    ns->Train(0, false);
}

std::string SpeedMeasurement::initResultDirectory() {
    std::filesystem::create_directories("speedMeasurements/");
    std::string directory = "speedMeasurements/" + scene->scene_name + "/";
    std::filesystem::create_directories(directory);
    std::string date = GetCurrentTimeForFileName() + "/";
    std::filesystem::create_directories(directory + date);
    return directory + date;
}

void SpeedMeasurement::test() {
    std::string progressBar = "Progress: 0%";
    std::cout << progressBar;

    for (int i = 0; i < ceil(float(num_cams) / batch_size); i++)
    {
        //Shorten last batchsize
        int curr_batch_size = ((i == ceil(float(num_cams) / batch_size) - 1) && (num_cams % batch_size != 0))
                                  ? (num_cams % batch_size)
                                  : batch_size;
        //Create imageInfo 
        ImageInfo* fd = new ImageInfo[curr_batch_size];
        for (int j = 0; j < curr_batch_size; j++)
        {
            fd[j].w              = scene->scene_cameras[0].w * scene->dataset_params.render_scale;
            fd[j].h              = scene->scene_cameras[0].h * scene->dataset_params.render_scale;
            fd[j].K              = scene->scene_cameras[0].K;
            fd[j].distortion     = scene->scene_cameras[0].distortion;
            fd[j].ocam           = scene->scene_cameras[0].ocam.cast<float>();
            fd[j].crop_transform = fd[j].crop_transform.scale(scene->dataset_params.render_scale);
            auto& f              = scene->frames[selected_cam + batch_size * i + j];
            fd[j].pose           = Sophus::SE3f::fitToSE3(f.OpenglModel() * GL2CVView()).cast<double>();

            fd[j].w              = fd[j].w * render_scale;
            fd[j].h              = fd[j].h * render_scale;
            fd[j].K              = fd[j].K.scale(render_scale);
            fd[j].exposure_value = f.exposure_value;
            ns->poses->SetPose(j, fd[j].pose);
            ns->intrinsics->SetPinholeIntrinsics(j, fd[j].K, fd[j].distortion);
        }
        auto neural_exposure_value = fd[0].exposure_value - scene->dataset_params.scene_exposure_value;

        //Reset timers
        timer_system.Reset();

        // Rendern
        torch::Tensor frames;
        Timer timer_render;
        timer_render.start();
        for (int cycle = 0; cycle < skip_cycles + num_cycles; cycle++)
        {
            //Create Batch
            std::vector<NeuralTrainData> batch(curr_batch_size);
            for (int k = 0; k < curr_batch_size; k++)
            {
                batch[k]                        = std::make_shared<TorchFrameData>();
                batch[k]->img                   = fd[k];
                batch[k]->img.camera_index      = 0;
                batch[k]->img.image_index       = k;
                batch[k]->img.camera_model_type = CameraModel::PINHOLE_DISTORTION;
                auto uv_image                   = InitialUVImage(fd[k].h, fd[k].w);
                torch::Tensor uv_tensor;
                uv_tensor    = ImageViewToTensor(uv_image.getImageView()).to(torch::kCUDA);
                batch[k]->uv = uv_tensor;
            }

            if (cycle < skip_cycles)
            {
                //Render without measuring time
                pipeline->Forward(*ns, batch, {}, false, false, neural_exposure_value, fd[0].white_balance);
            }
            else if (cycle != skip_cycles + num_cycles - 1)
            {
                // Render with measuring time
                timer_system.BeginFrame();
                pipeline->Forward(*ns, batch, {}, false, false, neural_exposure_value, fd[0].white_balance);
                timer_system.EndFrame();
            }
            else
            {
                // Render with measuring time and store rendered frames
                timer_system.BeginFrame();
                auto f_result = pipeline->Forward(*ns, batch, {}, false, false, neural_exposure_value, fd[0].white_balance);
                timer_system.EndFrame();
                frames = f_result.x;
            }
        }
        timer_render.stop();
        double time_render = timer_render.getTimeMS();

        // Store Frames
        auto frame = frames.split(1, 0);
        for (int j = 0; j < curr_batch_size; j++)
        {
            std::string path = result_directory + "example_frame_camera" + std::to_string(selected_cam + batch_size * i + j) + ".png";
            Saiga::TensorToImage<ucvec3>(frame[j]).save(path);
        }

        // Store Result
        storeResult(time_render, i, curr_batch_size);

        // Update Progressbar
        {
            for (size_t i = 0; i < progressBar.size(); i++)
            {
                std::cout << "\b";
            }
            int eta = (time_render * (ceil(float(num_cams) / batch_size) - i - 1)) / 1000;
            progressBar =
                "Progress: " + std::to_string(int((i + 1) * 100.0f / ceil(float(num_cams) / batch_size))) +
                "% (ETA: " + std::to_string(eta / 3600) + "h " + std::to_string((eta % 3600) / 60) + "min " +
                std::to_string((eta % 60)) + "sec)  ";
            std::cout << progressBar;
        }
    }

    std::cout << "\n";
}

void SpeedMeasurement::storeResult( double time_render, int batch, int curr_batch_size)
{
    std::ofstream file(result_directory + "result_batch" + std::to_string(batch) + ".txt");
    if (file.is_open())
    {
        file << "Speed Measurement"
             << "\n"
             << "\n"
             << "Date: " << GetCurrentTimeForFileName() << "\n"
             << "Scene: " << scene->scene_name << "\n"
             << "Points: " << scene->point_cloud.NumVertices() << "\n"
             << "Num Images: " << scene->frames.size() << "\n"
             << "\n"
             << "RenderSize: "
                << scene->scene_cameras[0].w * scene->dataset_params.render_scale * render_scale << "x"
                << scene->scene_cameras[0].h * scene->dataset_params.render_scale * render_scale << "\n"
             << "Cycles: " << num_cycles << "\n"
             << "Rendered Camera: " << std::to_string(selected_cam + batch * batch_size)
                << " - " << std::to_string(selected_cam + batch * batch_size + curr_batch_size - 1) << "\n"
             << "Batch Size: " << batch_size << "\n"
             << "Current Batch Size: " << curr_batch_size << "\n"
             << "Test Time: " << time_render << "\n"
             << "\n"
             << "\n"
             << "Result:\n"
             << "\n";
        timer_system.PrintTable(file);
        file << "\n";
        file.close();

    }
}

void SpeedMeasurement::storeOverallResult(double time_ges)
{
    //Read all test values
    std::vector<std::string> names;
    std::vector<std::vector<float>> values;
    {
        for (const auto& file : std::filesystem::directory_iterator(result_directory))
        {
            if (file.is_regular_file() && file.path().extension() == ".txt")
            {
                std::ifstream res;
                res.open(file.path());
                std::string read;
                for (size_t i = 0; i < 12; i++)
                {
                    getline(res, read);
                }
                std::vector<std::string> vals = explode(read, '\ ');
                int curr_batch_size           = std::stoi(vals[3]);
                for (size_t i = 0; i < 6; i++)
                {
                    getline(res, read);
                }
                while (!res.eof())
                {
                    getline(res, read);
                    std::vector<std::string> vals = explode(read, '\0');
                    if (vals.size() == 6)
                    {
                        std::vector<std::string>::iterator it = std::find(names.begin(), names.end(), vals[0]);
                        auto index = std::distance(names.begin(), it);
                        if (index == names.size())
                        {
                            names.push_back(vals[0]);
                            values.push_back({});
                        }
                        values[index].push_back(std::stof(vals[3]) / curr_batch_size); 
                    }

                }
                res.close();
            }
        }
    }

    std::ofstream file(result_directory + "overall_result.txt");

    //Write header
    {
        file << "Speed Measurement"
             << "\n"
             << "\n"
             << "Date: " << GetCurrentTimeForFileName() << "\n"
             << "Scene: " << scene->scene_name << "\n"
             << "Points: " << scene->point_cloud.NumVertices() << "\n"
             << "Num Images: " << scene->frames.size() << "\n"
             << "\n"
             << "RenderSize: " << scene->scene_cameras[0].w * scene->dataset_params.render_scale * render_scale << "x"
             << scene->scene_cameras[0].h * scene->dataset_params.render_scale * render_scale << "\n"
             << "Cycles: " << num_cycles << "\n"
             << "Rendered Camera: " << std::to_string(selected_cam) << " - " << std::to_string(selected_cam + num_cams)
             << "\n"
             << "Batch Size: " << batch_size << "\n"
             << "Test Time: " << time_ges << "\n"
             << "\n"
             << "\n"
             << "Result:\n"
             << "\n";
    }
    
    //Calculate and Write overall result
    {
        Table tab({30, 5, 10, 10, 10, 10}, file);
        tab.setFloatPrecision(5);
        tab << "Name"
            << "N"
            << "Mean"
            << "Median"
            << "Min"
            << "Max";
        for (size_t i = 0; i < values.size(); i++)
        {
            Statistics st(values[i]);
            tab << names[i] << num_cams << st.mean << st.median << st.min << st.max;
        }
    }

    file << "\n";
    file.close();
}

int main(int argc, char* argv[])
{
    SpeedMeasurement measurement;

    // Handle Arguments
    CLI::App app{"ADOP Speed Measurement for Scenes", "speed_measurement"};
    app.add_option("--scene_dir", measurement.scene_dir, "Directory of the sceen");
    //->required();
    app.add_option("--render_scale", measurement.render_scale, "Resulution scale");
    app.add_option("--num_cycles", measurement.num_cycles, "Number of cycles to measure");
    app.add_option("--skip_cycles", measurement.skip_cycles, "Number of cycles to run bevore measurement");
    app.add_option("--selected_cam", measurement.selected_cam, "Camera from witch to render From (0 - NumSceneImages)");
    app.add_option("--num_cams", measurement.num_cams, "How many cameras to measure(1 - (NumSceneImages-selected_cam))");
    app.add_option("--all_cams", measurement.all_cams, "Measure all available cameras");
    app.add_option("--batch_size", measurement.batch_size, "Number of Images to render Paralell");
    CLI11_PARSE(app, argc, argv);

    measurement.runTest();
}