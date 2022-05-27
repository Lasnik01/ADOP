#pragma once

#include "saiga/opengl/rendering/deferredRendering/deferred_renderer.h"
#include "saiga/opengl/window/WindowTemplate.h"
#include "opengl/RealTimeRenderer.h"
#include "saiga/opengl/opengl_helper.h"

const int num_timers = 1;
const std::string timers_names[num_timers] = {"Frame Time"};


std::string scene_dir = "C:/ADOP/ADOPWIN/ADOP/scenes/tt_playground";
float render_scale = 1.0f;
int num_cycles     = 100;
int skip_cycles       = 5;
int selected_cam      = 0;
int num_cams          = 1;
bool all_cams         = false;

void initSaiga() {
    //---
    // Init Window(only to set SAIGA Paths)
    // TODO: ggf. inizialisize SAIGA Paths without creating hidden Window
    //---
    WindowParameters windowParameters;
    OpenGLParameters openglParameters;
    windowParameters.fromConfigFile("config.ini");
    windowParameters.hidden = true;
    auto window             = std::make_unique<glfw_Window>(windowParameters, openglParameters);
    window.release();
}

std::string GetCurrentTimeForFileName()
{
    auto time = std::time(nullptr);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%F_%T");
    auto s = ss.str();
    std::replace(s.begin(), s.end(), ':', '-');
    return s;
}

void storeResult(std::string path, double time_render, std::shared_ptr<SceneData>* scene, CUDA::CudaTimerSystem* timer_system)
{
    std::ofstream file(path + "result_camera" + std::to_string(selected_cam) + ".txt");
    file << "Speed Measurement"
         << "\n"
         << "\n"
         << "Date: " << GetCurrentTimeForFileName() << "\n"
         << "Cycles: " << num_cycles << "\n"
         << "Scene: " << scene->get()->scene_name << "\n"
         << "Points: " << scene->get()->point_cloud.NumVertices() << "\n"
         << "Num Images: " << scene->get()->frames.size() << "\n"
         << "Renderer: " << glGetString(GL_RENDERER) << "\n"
         << "RenderSize: "
         << scene->get()->scene_cameras[0].w * scene->get()->dataset_params.render_scale * render_scale << "x"
         << scene->get()->scene_cameras[0].h * scene->get()->dataset_params.render_scale * render_scale << "\n"
         << "Test Time: " << time_render << "\n"
         << "Selected Camera: " << selected_cam << "\n";
         
    file << "\n";
    file << "\n";
    file << "Result:\n";
    file << "\n";
    timer_system->PrintTable(file);
    file << "\n";
    file.close();
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
         << "Renderer: " << glGetString(GL_RENDERER) << "\n"
         << "RenderSize: "
         << scene->get()->scene_cameras[0].w * scene->get()->dataset_params.render_scale * render_scale << "x"
         << scene->get()->scene_cameras[0].h * scene->get()->dataset_params.render_scale * render_scale << "\n"
         << "Test Time: " << time_ges << "\n"
         << "Num Camera: " << num_cams << "\n";

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
    CLI11_PARSE(app, argc, argv);

    initSaiga();

    //---
    // Load Scene and Create Renderer
    //---

    // Scene erstellen
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

    //Memory for OutputImage
    TemplatedImage<vec4> output_image;
    std::shared_ptr<Saiga::CUDA::Interop> texure_interop;
    std::shared_ptr<Texture> output_texture;

    //---
    // Redner and Measure
    //---
    std::cout << "*Render Scene " << scene_dir << "*" << std::endl;
    Timer timer_ges;
    timer_ges.start();
    for (int i = 0; i < num_cams; i++)
    {
        // fd erstellen
        ImageInfo fd;
        {
            fd.w              = scene->scene_cameras[0].w * scene->dataset_params.render_scale;
            fd.h              = scene->scene_cameras[0].h * scene->dataset_params.render_scale;
            fd.K              = scene->scene_cameras[0].K;
            fd.distortion     = scene->scene_cameras[0].distortion;
            fd.ocam           = scene->scene_cameras[0].ocam.cast<float>();
            fd.crop_transform = fd.crop_transform.scale(scene->dataset_params.render_scale);
            auto& f           = scene->frames[selected_cam];
            fd.pose           = Sophus::SE3f::fitToSE3(f.OpenglModel() * GL2CVView()).cast<double>();

            fd.w              = fd.w * render_scale;
            fd.h              = fd.h * render_scale;
            fd.K              = fd.K.scale(render_scale);
            fd.exposure_value = f.exposure_value;

            ns->poses->SetPose(0, fd.pose);
            ns->intrinsics->SetPinholeIntrinsics(0, fd.K, fd.distortion);
        }

        // OutputImage erstellen
        {
            output_image.create(fd.h, fd.w);
            output_image.getImageView().set(vec4(1, 1, 1, 1));
            output_texture = std::make_shared<Texture>(output_image);
            texure_interop = std::make_shared<Saiga::CUDA::Interop>();
            texure_interop->initImage(output_texture->getId(), output_texture->getTarget());
        }

        timer_system.Reset();
        Timer timer_render;
        timer_render.start();
        // Rendern
        torch::Tensor x;
        {
            for (int i = 0; i < skip_cycles; i++)
            {
                // batch erstellen
                std::vector<NeuralTrainData> batch(1);
                {
                    batch.front()                        = std::make_shared<TorchFrameData>();
                    batch.front()->img                   = fd;
                    batch.front()->img.camera_index      = 0;
                    batch.front()->img.image_index       = 0;
                    batch.front()->img.camera_model_type = CameraModel::PINHOLE_DISTORTION;
                    auto uv_image                        = InitialUVImage(fd.h, fd.w);
                    torch::Tensor uv_tensor;
                    uv_tensor         = ImageViewToTensor(uv_image.getImageView()).to(torch::kCUDA);
                    batch.front()->uv = uv_tensor;
                }

                auto neural_exposure_value = fd.exposure_value - scene->dataset_params.scene_exposure_value;
                pipeline->Forward(*ns, batch, {}, false, false, neural_exposure_value, fd.white_balance);
            }
            for (int i = 0; i < num_cycles; i++)
            {
                // batch erstellen
                std::vector<NeuralTrainData> batch(1);
                {
                    batch.front()                        = std::make_shared<TorchFrameData>();
                    batch.front()->img                   = fd;
                    batch.front()->img.camera_index      = 0;
                    batch.front()->img.image_index       = 0;
                    batch.front()->img.camera_model_type = CameraModel::PINHOLE_DISTORTION;
                    auto uv_image                        = InitialUVImage(fd.h, fd.w);
                    torch::Tensor uv_tensor;
                    uv_tensor         = ImageViewToTensor(uv_image.getImageView()).to(torch::kCUDA);
                    batch.front()->uv = uv_tensor;
                }

                auto neural_exposure_value = fd.exposure_value - scene->dataset_params.scene_exposure_value;
                if (i != num_cycles - 1)
                {
                    timer_system.BeginFrame();
                    pipeline->Forward(*ns, batch, {}, false, false, neural_exposure_value, fd.white_balance);
                    timer_system.EndFrame();
                }
                else
                {
                    timer_system.BeginFrame();
                    auto f_result = pipeline->Forward(*ns, batch, {}, false, false, neural_exposure_value, fd.white_balance);
                    x = f_result.x;
                    timer_system.EndFrame();
                }
            }
        }
        timer_render.stop();
        double time_render = timer_render.getTimeMS();
        
        // Postprocess for visuialisation
        {
            x                           = x.squeeze();
            torch::Tensor alpha_channel = torch::ones({1, x.size(1), x.size(2)},
                                                      torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat));
            x                           = torch::cat({x, alpha_channel}, 0);
            x                           = x.permute({1, 2, 0});
            x                           = x.contiguous();
        }

        // Safe Result
        {
            // Download renderImage
            {
               
                std::string path = directory + date + "scene_example_camera" + std::to_string(selected_cam) + ".png";
                texure_interop->mapImage();
                cudaMemcpy2DToArray(texure_interop->array, 0, 0, x.data_ptr(), x.stride(0) * sizeof(float),
                                    x.size(1) * x.size(2) * sizeof(float), x.size(0), cudaMemcpyDeviceToDevice);
                texure_interop->unmap();

                TemplatedImage<ucvec4> tmp(output_texture->getHeight(), output_texture->getWidth());
                output_texture->bind();
                glGetTexImage(output_texture->getTarget(), 0, GL_RGBA, GL_UNSIGNED_BYTE, tmp.data());
                assert_no_glerror();
                output_texture->unbind();
                tmp.save(path);
                
            }

            //Store Result
            storeResult(directory + date, time_render, &scene, &timer_system);

            //Update Progressbar
            {
                if (i == 0)
                {
                    std::cout << "\nProgress: 00%";
                }
                if (int((i + 1) * 100.0f / num_cams) < 10)
                {
                    std::cout << "\b\b" << int((i + 1) * 100.0f / num_cams) << "%";
                }
                else
                {
                    std::cout << "\b\b\b" << int((i + 1) * 100.0f / num_cams) << "%";
                }
            }
        }

        selected_cam++;
    }
    std::cout << "\n";
    timer_ges.stop();
    if (num_cams > 1)
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