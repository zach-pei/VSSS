#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <queue>
#include <vector>
#include <list>

using namespace std;
using namespace cv;

class seeds
{
    public:
        Mat img;
        int pixel_side_length;//the pixel cuboid's top side length
        double qiqi_capacity;
        double S;//area of all pixels
        int K;//the number of superpixels
        int height;//the image's height
        int width;//the image's width
        double **pixel_S;
        int *seed_X;
        int *seed_Y;
        void set_pixel_side_length(int length);
        void initial_pixel_S();
        void initial_seed_P(int K);
        double refine_euclidean(double* pixel1, double* pixel2);
        double caculate_S();
        double caculate_qiqi_capacity();
        double relu(double value);
        double slide_windows_area(int x, int y);
        void run_seeds();
        void run_seeds1();
        void run_seeds2();
        
        seeds(const Mat img, int length, int K);
        ~seeds();
};

class growth{
    public:
        Mat img;
        int** lable;//pixel lable
        double*** gh;//unlabeled nodes' growth height
        double*** h;//all the Adjacency nodes’ height difference
        int*** ad_state;//all nodes' Adjacency state
        double** soil_quality;
        double** soil_mean_color;//mean RGB value of superpixels
        int soil_mean_color_number;
        vector<int > num_of_nodes_in_each_sp;//store all labeled nodes' X coordinate(坐标) in each superpixel
        //vector<int > Seeds_X;//extensible seeds' X coordinate
        //vector<int > Seeds_Y;//extensible seeds' Y coordinate
        list<int> Seeds_X;
        list<int> Seeds_Y;

        void push_new_seeds(int X, int Y);
        void update_soil_mean(int X, int Y);
        void update(int height, int width, double alpha, double lambda, double beta, double tau);
        void initial_seeds_lable(int* X, int* Y, int height, int width);
        void initial_gh(int height, int width);
        double refine_euclidean(double* pixel1, double* pixel2);
        double* returnRGB(int X, int Y);
        void initial_h(int height, int width, Mat img);
        void initial_state(int height, int width);
        void caculate_soil_quality(int height, int width, double** pixel_S);
        void combine_small_sp(int num);
        void merging(int num);
        double velocity(double alpha, double lambda, double beta, double theta, int X, int Y, int num);
        growth(double** pixel_s, int* seed_X, int* seed_Y, int height, int width, Mat image, double alpha, double lambda, double beta, double tau);
        ~growth();

};