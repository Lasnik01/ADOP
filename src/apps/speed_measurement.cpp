#pragma once

#include "saiga/opengl/rendering/deferredRendering/deferred_renderer.h"
#include "saiga/opengl/window/WindowTemplate.h"
#include "opengl/RealTimeRenderer.h"
#include "saiga/opengl/opengl_helper.h"

const int max_batch_size = 16;
std::string scene_dir = "C:/ADOP/MYADOP/ADOP/scenes/tt_playground";
float render_scale = 1.0f;
int num_cycles     = 10;
int skip_cycles       = 5;
int selected_cam      = 0;
int num_cams          = 7;
bool all_cams         = false;
int batch_size         = 2;

std::string GetCurrentTimeForFileName()
{
    auto time = std::time(nullptr);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%F_%T");
    auto s = ss.str();
    std::replace(s.begin(), s.end(), ':', '-');
    return s;
}

void storeResult(std::string path, double time_render, std::shared_ptr<SceneData>* scene, CUDA::CudaTimerSystem* timer_system, int batch, int curr_batch_size)
{
    std::ofstream file(path + "result_batch" + std::to_string(batch) + ".txt");
    if (file.is_open())
    {
        file << "Speed Measurement"
             << "\n"
             << "\n"
             << "Date: " << GetCurrentTimeForFileName() << "\n"
             << "Cycles: " << num_cycles << "\n"
             << "Scene: " << scene->get()->scene_name << "\n"
             << "Points: " << scene->get()->point_cloud.NumVertices() << "\n"
             << "Num Images: " << scene->get()->frames.size() << "\n"
             << "RenderSize: "
             << scene->get()->scene_cameras[0].w * scene->get()->dataset_params.render_scale * render_scale << "x"
             << scene->get()->scene_cameras[0].h * scene->get()->dataset_params.render_scale * render_scale << "\n"
             << "Test Time: " << time_render << "\n"
             << "Selected Camera: " << std::to_string(selected_cam + batch * batch_size);
        if (batch_size > 1)
        {
            file << " - " << std::to_string(selected_cam + batch * batch_size + curr_batch_size - 1) << "\n"
                 << "Batch Size: " << batch_size;
        }
        else
        {
            file << "\n";
        }

        file << "\n";
        file << "\n";
        file << "\n";
        file << "Result:\n";
        file << "\n";
        timer_system->PrintTable(file);
        file << "\n";
        file.close();
    }
}

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

void storeResultGes(std::string path, double time_ges, std::shared_ptr<SceneData>* scene)
{
    std::vector<std::string> names;
    std::vector<std::vector<float>> values;
    {
        for (const auto& file : std::filesystem::directory_iterator(path))
        {
            if (file.is_regular_file() && file.path().extension() == ".txt")
            {
                std::ifstream res;
                res.open(file.path());
                std::string read;
                for (size_t i = 0; i < 16; i++)
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
                        values[index].push_back(std::stof(vals[2]));
                    }

                }
                res.close();
            }
        }
    }

    std::ofstream file(path + "result.txt");
    file << "Speed Measurement"
         << "\n"
         << "\n"
         << "Date: " << GetCurrentTimeForFileName() << "\n"
         << "Cycles: " << num_cycles << "\n"
         << "Scene: " << scene->get()->scene_name << "\n"
         << "Points: " << scene->get()->point_cloud.NumVertices() << "\n"
         << "Num Images: " << scene->get()->frames.size() << "\n"
         << "RenderSize: "
         << scene->get()->scene_cameras[0].w * scene->get()->dataset_params.render_scale * render_scale << "x"
         << scene->get()->scene_cameras[0].h * scene->get()->dataset_params.render_scale * render_scale << "\n"
         << "Test Time: " << time_ges << "\n"
         << "Num Cameras: " << num_cams << "\n"
         << "Batch Size: " << batch_size << "\n";
    

    file << "\n";
    file << "\n";
    file << "Result:\n";
    file << "\n";

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
        tab << names[i] << values[i].size() << st.mean << st.median << st.min << st.max;
    }

    file << "\n";
    file.close();
}

int main(int argc, char* argv[])
{
    std::cout << "*ADOP Speed Measurement for Scenes*" << std::endl;
    
    //---
    //Handle Arguments
    //---
    CLI::App app{"ADOP Speed Measurement for Scenes", "speed_measurement"};
    app.add_option("--scene_dir", scene_dir, "Directory of the sceen");
    //->required();
    app.add_option("--render_scale", render_scale, "Resulution scale");
    app.add_option("--num_cycles", num_cycles, "Number of cycles to measure");
    app.add_option("--skip_cycles", skip_cycles, "Number of cycles to run bevore measurement");
    app.add_option("--selected_cam", selected_cam, "Camera from witch to render From (0 - NumSceneImages)");
    app.add_option("--num_cams", num_cams, "How many cameras to measure(1 - (NumSceneImages-selected_cam))");
    app.add_option("--all_cams", all_cams, "Measure all available cameras");
    app.add_option("--batch_size", batch_size, "Number of Images to render Paralell");
    CLI11_PARSE(app, argc, argv);

    //---
    // Load Scene and Create Renderer
    //---

    // Scene erstellen & Argguments checken
    std::shared_ptr<SceneData> scene;
    {
        scene = std::make_shared<SceneData>(scene_dir);
        if (selected_cam >= scene->frames.size())
        {
            selected_cam = scene->frames.size() - 1;
        }
        if (num_cams > scene->frames.size() - selected_cam)
        {
            num_cams = scene->frames.size() - selected_cam;
        }
        if (all_cams)
        {
            selected_cam = 0;
            num_cams     = scene->frames.size();
        }
        if (batch_size > max_batch_size)
        {
            batch_size = max_batch_size;
        }
    }

    //Experiments erstellen
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

    //Params erstellen
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

    //TimerSystem erstellen
    CUDA::CudaTimerSystem timer_system;

    //Pipeline erstellen
    std::shared_ptr<NeuralPipeline> pipeline;
    pipeline = std::make_shared<NeuralPipeline>(params);
    pipeline->Train(false);
    pipeline->timer_system = &timer_system;

    //NeuralScene erstellen
    std::shared_ptr<NeuralScene> ns;
    ns = std::make_shared<NeuralScene>(scene, params);
    ns->to(torch::kCUDA);
    ns->Train(0, false);

    //Testdirectory erstellen
    std::filesystem::create_directories("speedMeasurements/");
    std::string directory = "speedMeasurements/" + scene->scene_name + "/";
    std::filesystem::create_directories(directory);
    std::string date = GetCurrentTimeForFileName() + "/";
    std::filesystem::create_directories(directory + date);

    //---
    // Redner and Measure
    //---
    std::cout << "*Render Scene " << scene_dir << "*" << std::endl;
    std::string progressBar = "Progress: 0%";
    std::cout << "\n" << progressBar;
    Timer timer_ges;
    timer_ges.start();
    for (int i = 0; i < ceil(float(num_cams)/batch_size); i++)
    {
        int curr_batch_size = ((i == ceil(float(num_cams) / batch_size) - 1) && (num_cams % batch_size != 0))
                                  ? (num_cams % batch_size)
                                  : batch_size;
        // fd erstellen
        ImageInfo* fd = new ImageInfo[curr_batch_size];
        {
            for (int j = 0; j < curr_batch_size; j++)
            {
                fd[j].w              = scene->scene_cameras[0].w * scene->dataset_params.render_scale;
                fd[j].h              = scene->scene_cameras[0].h * scene->dataset_params.render_scale;
                fd[j].K              = scene->scene_cameras[0].K;
                fd[j].distortion     = scene->scene_cameras[0].distortion;
                fd[j].ocam           = scene->scene_cameras[0].ocam.cast<float>();
                fd[j].crop_transform = fd[j].crop_transform.scale(scene->dataset_params.render_scale);
                auto& f           = scene->frames[selected_cam + batch_size*i + j];
                fd[j].pose           = Sophus::SE3f::fitToSE3(f.OpenglModel() * GL2CVView()).cast<double>();

                fd[j].w              = fd[j].w * render_scale;
                fd[j].h              = fd[j].h * render_scale;
                fd[j].K              = fd[j].K.scale(render_scale);
                fd[j].exposure_value = f.exposure_value;
                ns->poses->SetPose(j, fd[j].pose);
                ns->intrinsics->SetPinholeIntrinsics(j, fd[j].K, fd[j].distortion);
            }
        }
       
        auto neural_exposure_value = fd[0].exposure_value - scene->dataset_params.scene_exposure_value;

        timer_system.Reset();
        Timer timer_render;
        timer_render.start();
        // Rendern
        torch::Tensor x;
        {
            for (int j = 0; j < skip_cycles; j++)
            {
                // batch erstellen
                std::vector<NeuralTrainData> batch(curr_batch_size);
                {
                    for (int k = 0; k < curr_batch_size; k++)
                    {
                        batch[k]                        = std::make_shared<TorchFrameData>();
                        batch[k]->img                        = fd[k];
                        batch[k]->img.camera_index           = 0;
                        batch[k]->img.image_index            = 0;
                        batch[k]->img.camera_model_type      = CameraModel::PINHOLE_DISTORTION;
                        auto uv_image                        = InitialUVImage(fd[k].h, fd[k].w);
                        torch::Tensor uv_tensor;
                        uv_tensor         = ImageViewToTensor(uv_image.getImageView()).to(torch::kCUDA);
                        batch[k]->uv = uv_tensor;
                    }
                }
                pipeline->Forward(*ns, batch, {}, false, false, neural_exposure_value, fd[0].white_balance);
            }
            for (int j = 0; j < num_cycles; j++)
            {
                // batch erstellen
                std::vector<NeuralTrainData> batch(curr_batch_size);
                {
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
                }
                if (j != num_cycles - 1)
                {
                    timer_system.BeginFrame();
                    pipeline->Forward(*ns, batch, {}, false, false, neural_exposure_value, fd[0].white_balance);
                    timer_system.EndFrame();
                }
                else
                {
                    timer_system.BeginFrame();
                    auto f_result = pipeline->Forward(*ns, batch, {}, false, false, neural_exposure_value, fd[0].white_balance);
                    x = f_result.x;
                    timer_system.EndFrame();
                }
            }
        }
        timer_render.stop();
        double time_render = timer_render.getTimeMS();
        
        // Safe Result
        {
            // Store renderImage
            {
                auto y = x.split(1, 0);
                for (int j = 0; j < curr_batch_size; j++)
                {
                    std::string path = directory + date + "scene_example_camera" +
                                       std::to_string(selected_cam + batch_size * i + j) + ".png";
                    Saiga::TensorToImage<ucvec3>(y[j]).save(path);
                }
               
            }

            //Store Result
            storeResult(directory + date, time_render, &scene, &timer_system, i, curr_batch_size);
            
            //Update Progressbar
            {
                for (size_t i = 0; i < progressBar.size(); i++)
                {
                    std::cout << "\b";
                }
                int eta     = (time_render * (ceil(float(num_cams) / batch_size) - i - 1)) / 1000;
                progressBar =
                    "Progress: " + std::to_string(int((i + 1) * 100.0f / ceil(float(num_cams) / batch_size))) +
                              "% (ETA: " + std::to_string(eta / 3600) + "h " + std::to_string((eta % 3600) / 60) +
                              "min " + std::to_string((eta % 60)) +
                              "sec)";
                std::cout << progressBar;
            }
            
        }
    }
    std::cout << "\n";
    timer_ges.stop();
    if (ceil(float(num_cams) / batch_size) > 1)
    {
        double time_ges = timer_ges.getTimeMS();
        storeResultGes(directory + date, time_ges, &scene);
    }
    else
    {
        timer_system.PrintTable(std::cout);
    }
    std::cout << "\n\n*Stored result to " << directory + date << "*" << std::endl;

    std::cout << std::endl;
    std::cout << "===============" << std::endl;
    std::cout << "*Test Finished*" << std::endl;
    std::cout << "===============" << std::endl;
}