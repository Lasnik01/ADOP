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

void renderScene(std::unique_ptr<RealTimeRenderer> *neural_renderer, std::shared_ptr<SceneData> *scene, double* data)
{
    ImageInfo fd;
    fd.w              = scene->get()->scene_cameras[0].w * scene->get()->dataset_params.render_scale;
    fd.h              = scene->get()->scene_cameras[0].h * scene->get()->dataset_params.render_scale;
    fd.K              = scene->get()->scene_cameras[0].K;
    fd.distortion     = scene->get()->scene_cameras[0].distortion;
    fd.ocam           = scene->get()->scene_cameras[0].ocam.cast<float>();
    fd.crop_transform = fd.crop_transform.scale(scene->get()->dataset_params.render_scale);
    // fd.pose           = Sophus::SE3f::fitToSE3(scene_camera.model * GL2CVView()).cast<double>();
    fd.w = fd.w * render_scale;
    fd.h = fd.h * render_scale;
    fd.K = fd.K.scale(render_scale);
    // fd.exposure_value = renderer->tone_mapper.params.exposure_value;
    // fd.white_balance  = renderer->tone_mapper.params.white_point;
    // neural_renderer->tone_mapper.params      = renderer->tone_mapper.params;
    // neural_renderer->tone_mapper.tm_operator = renderer->tone_mapper.tm_operator;
    neural_renderer->get()->tone_mapper.params.exposure_value -= scene->get()->dataset_params.scene_exposure_value;
    neural_renderer->get()->tone_mapper.params_dirty = true;

    auto& f = scene->get()->frames[selected_cam];
    fd.pose = Sophus::SE3f::fitToSE3(f.OpenglModel() * GL2CVView()).cast<double>();

    for (int i = 0; i < skip_cycles; i++)
    {
        neural_renderer->get()->Render(fd);
    }
    for (int i = 0; i < num_cycles; i++)
    {
        Timer timer_frame;
        timer_frame.start();
        neural_renderer->get()->Render(fd);
        timer_frame.stop();
        data[i * num_timers + 0] = timer_frame.getTimeMS();
    }
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

void storeResult(std::string path, double time_render, double* data, std::shared_ptr<SceneData>* scene, double* data_ges)
{
    double *time_sum = new double[num_timers];
    double *time_avg = new double[num_timers];
    double *time_sd = new double[num_timers];
    for (int i = 0; i < num_cycles; i++)
    {
        for (int j = 0; j < num_timers; j++)
        {
            time_sum[j] += data[i * num_timers + j];
        }
    }
    for (int j = 0; j < num_timers; j++)
    {
        time_avg[j] = time_sum[j]/num_cycles;
        data_ges[j] = time_avg[j];
    }
    for (int i = 0; i < num_cycles; i++)
    {
        for (int j = 0; j < num_timers; j++)
        {
            time_sd[j] += pow(data[i * num_timers + j] - time_avg[j], 2);
        }
    }
    for (int j = 0; j < num_timers; j++)
    {
        time_sd[j] = sqrt(time_sd[j]/(num_cycles-1));
    }

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
    for (int i = 0; i < num_timers; i++)
    {
        file << timers_names[i] << ": " << time_sum[i] << " (avg.: " << time_avg[i] << "; SD.: " << time_sd[i] << ")\n"; 
    }
    file << "\n";
    file << "\n";
    file << "Values:\n";
    file << "\n";
    for (int i = 0; i < num_cycles; i++)
    {
        file << "Frame" << i + 1 << ": ";
        for (int j = 0; j < num_timers; j++)
        {
            file << data[i * num_timers + j] << "; ";
        }
        file << "\n";
    }
    file.close();
}

void storeResultGes(std::string path, double time_ges, double* data_ges, std::shared_ptr<SceneData>* scene)
{
    double* time_sum = new double[num_timers];
    double* time_avg = new double[num_timers];
    double* time_sd  = new double[num_timers];
    for (int i = 0; i < num_cams; i++)
    {
        for (int j = 0; j < num_timers; j++)
        {
            time_sum[j] += data_ges[i * num_timers + j];
        }
    }
    for (int j = 0; j < num_timers; j++)
    {
        time_avg[j] = time_sum[j] / num_cams;
    }
    for (int i = 0; i < num_cams; i++)
    {
        for (int j = 0; j < num_timers; j++)
        {
            time_sd[j] += pow(data_ges[i * num_timers + j] - time_avg[j], 2);
        }
    }
    for (int j = 0; j < num_timers; j++)
    {
        time_sd[j] = sqrt(time_sd[j] / (num_cams - 1));
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
    for (int i = 0; i < num_timers; i++)
    {
        file << timers_names[i] << ": " << time_sum[i] * num_cycles << " (avg.: " << time_avg[i] << "; SD.: " << time_sd[i] << ")\n";
    }
    file << "\n";
    file << "\n";
    file << "Values(Avg.):\n";
    file << "\n";
    for (int i = 0; i < num_cams; i++)
    {
        file << "Camera" << selected_cam - num_cams + i << ": ";
        for (int j = 0; j < num_timers; j++)
        {
            file << data_ges[i * num_timers + j] << "; ";
        }
        file << "\n";
    }
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
    //Load Scene and Create Renderer
    //---
    std::cout << "*Loading Scene " << scene_dir << "*" << std::endl;
    std::shared_ptr<SceneData> scene = std::make_shared<SceneData>(scene_dir);
    std::cout << "*Create Renderer*" << std::endl;
    std::unique_ptr<RealTimeRenderer> neural_renderer = std::make_unique<RealTimeRenderer>(scene);
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

    std::filesystem::create_directories("speedMeasurements/");
    std::string dir = "speedMeasurements/" + scene->scene_name + "/";
    std::filesystem::create_directories(dir);
    std::string date = GetCurrentTimeForFileName() + "/";
    std::filesystem::create_directories(dir + date);
    double* data = new double[num_cycles * num_timers];
    double* data_ges = new double[num_cams * num_timers];

    std::cout << "*Render Scene " << scene_dir << "*" << std::endl;
    Timer timer_ges;
    timer_ges.start();
    for (int i = 0; i < num_cams; i++)
    {
        //---
        // Redner and Measure
        //---
        Timer timer_render;
        timer_render.start();
        renderScene(&neural_renderer, &scene, data);
        timer_render.stop();
        double time_render = timer_render.getTimeMS();


        //---
        // Safe Result
        //---
        storeResult(dir + date, time_render, data, &scene, &data_ges[i * num_timers]);

        std::string path = dir + date + "scene_example_camera" + std::to_string(selected_cam) + ".png";
        auto frame       = neural_renderer->DownloadRender();
        frame.save(path);
        if (i == 0)
        {
            std::cout << "\nProgress: 00%";
        }
        if (int((i+1)*100.0f / num_cams) < 10)
        {
            std::cout << "\b\b" << int((i+1)*100.0f / num_cams) << "%";
        }
        else
        {
            std::cout << "\b\b\b" << int((i+1)*100.0f / num_cams) << "%";
        }

        selected_cam++;
    }
    timer_ges.stop();
    double time_ges = timer_ges.getTimeMS();
    if (num_cams > 1)
    {
        storeResultGes(dir + date, time_ges, data_ges, &scene);
    }
    std::cout << "\n\n*Stored result to " << dir + date << "*" << std::endl;

    std::cout << std::endl;
    std::cout << "===============" << std::endl;
    std::cout << "*Test Finished*" << std::endl;
    std::cout << "===============" << std::endl;
}


