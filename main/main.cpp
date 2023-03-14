#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <time.h>
#include "vine.h"
#include "tools.h"
#include <boost/filesystem.hpp>
#include <boost/timer.hpp>
#include <boost/program_options.hpp>
using namespace std;
using namespace cv;

int main(int argc,char *argv[])
{
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
        ("help", "watch all optional command")
        ("input", boost::program_options::value<std::string>()->default_value("../testforseed"), "the folder to process, it can contain many images")
        ("length", boost::program_options::value<int>()->default_value(1), "the pixel cuboid's top side length")
        ("num_sp", boost::program_options::value<int>()->default_value(10), "the number of superpixels")
        ("seeds_size", boost::program_options::value<int>()->default_value(2), "the size of drawed seeds")
        ("alpha", boost::program_options::value<double>()->default_value(0.005), "paramter of velocity--alpha")
        ("lambda", boost::program_options::value<double>()->default_value(20), "paramter of velocity--lambda")
        ("beta", boost::program_options::value<double>()->default_value(30), "paramter of velocity--beta")
        ("tau", boost::program_options::value<double>()->default_value(7), "paramter of velocity--tau")
        ("output", boost::program_options::value<std::string>()->default_value("./output"), "the folder to save final results")
        ("output_seeds", "if save the images that contain initial seeds by qiqi method")
        ("output_sp", "if save the images that contain superpixels")
        ("output_label", "if save the superpixel label")
        ("output_mean", "if save mean color");
    boost::program_options::positional_options_description positionals;
    positionals.add("input", 1);
    
    boost::program_options::variables_map parameters;
    boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).positional(positionals).run(), parameters);
    boost::program_options::notify(parameters);

    if (parameters.find("help") != parameters.end()) 
    {
        std::cout << desc << std::endl;
        return 1;
    }
    
    boost::filesystem::path outputDir(parameters["output"].as<std::string>());
    if (!boost::filesystem::is_directory(outputDir)) 
    {
        boost::filesystem::create_directory(outputDir);
    }
    
    boost::filesystem::path inputDir(parameters["input"].as<std::string>());
    if (!boost::filesystem::is_directory(inputDir)) 
    {
        std::cout << "Input directory not found, please check input path carefully!" << std::endl;
        return 1;
    }

    std::vector<boost::filesystem::path> pathVector;
    std::vector<boost::filesystem::path> images;
    
    std::copy(boost::filesystem::directory_iterator(inputDir), boost::filesystem::directory_iterator(), std::back_inserter(pathVector));

    std::sort(pathVector.begin(), pathVector.end());
    
    std::string extension;
    int count = 0;
    
    for (std::vector<boost::filesystem::path>::const_iterator iterator(pathVector.begin()); iterator != pathVector.end(); ++iterator) 
    {
        if (boost::filesystem::is_regular_file(*iterator)) 
        {
            // Check supported file extensions.
            extension = iterator->extension().string();
            std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
            if (extension == ".png" || extension == ".jpg" || extension == ".jpeg")
            {
                images.push_back(*iterator);
                ++count;
            }
        }
    }
    std::cout << "Find " << count << " images" << std::endl;

    int length = parameters["length"].as<int>();
    int num_sp = parameters["num_sp"].as<int>();
    int seeds_size = parameters["seeds_size"].as<int>();
    double alpha = parameters["alpha"].as<double>();
    double lambda = parameters["lambda"].as<double>();
    double beta = parameters["beta"].as<double>();
    double tau = parameters["tau"].as<double>();

    boost::timer timer;
    double total_runtime = 0;

    for(std::vector<boost::filesystem::path>::iterator iterator = images.begin(); iterator != images.end(); ++iterator)
    {
        Mat image = imread(iterator->string());
        //seeds objseeds(image,length,num_sp);
        timer.restart();
        cout<<iterator->string()<<endl;
        seeds objseeds(image,length,num_sp);
        growth objgrowth(objseeds.pixel_S, objseeds.seed_X, objseeds.seed_Y, objseeds.height, objseeds.width, image, alpha, lambda, beta, tau);
        total_runtime += timer.elapsed();
        if(parameters.find("output_seeds") != parameters.end())
        {
            boost::filesystem::path extension = iterator->filename().extension();
            int position = iterator->filename().string().find(extension.string());
            std::string store = outputDir.string() +  "/" + iterator->filename().string().substr(0,position) + "_seeds.png";
            //cout << store <<endl;

            draw objdraw;
            Mat out_seeds = objdraw.DrawSeeds(objseeds.seed_X, objseeds.seed_Y, image, seeds_size);
            imwrite(store, out_seeds);
            
        }
        if(parameters.find("output_sp") != parameters.end())
        {
            boost::filesystem::path extension = iterator->filename().extension();
            int position = iterator->filename().string().find(extension.string());
            std::string store = outputDir.string() +  "/" + iterator->filename().string().substr(0,position) + "_sp.png";
            //cout << store <<endl;

            draw objdraw;
            Mat out_sp = objdraw.DrawSuperpixelEdge(objgrowth.lable, image);
            imwrite(store, out_sp);
            
        }

        if(parameters.find("output_label") != parameters.end())
        {
            boost::filesystem::path extension = iterator->filename().extension();
            int position = iterator->filename().string().find(extension.string());
            std::string store = outputDir.string() +  "/" + iterator->filename().string().substr(0,position) + ".csv";
            //cout << store <<endl;

            draw objdraw;
            objdraw.OutputLabel(image, objgrowth.lable, store);  
        }

        if(parameters.find("output_mean") != parameters.end())
        {
            boost::filesystem::path extension = iterator->filename().extension();
            int position = iterator->filename().string().find(extension.string());
            std::string store = outputDir.string() +  "/" + iterator->filename().string().substr(0,position) + "_mean.png";
            //cout << store <<endl;

            draw objdraw;
            Mat meanimg = objdraw.DrawMeancolor(objgrowth.lable, objgrowth.soil_mean_color, image);
            imwrite(store, meanimg);
        }
        
    }
    cout << "Average runtime: " << total_runtime/images.size() << "s" << endl;
    std::string store = outputDir.string() +  "/" + "Averaging time.txt";
    ofstream outfile;
    outfile.open(store);
    double averuntime = total_runtime/images.size();
    outfile << averuntime;
    outfile.close();
}