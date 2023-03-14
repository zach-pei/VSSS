#include "vine.h"
#include "tools.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
#include <list>
using namespace std;
using namespace cv;




//**************************class seeds*********************************
void seeds::set_pixel_side_length(int length)
{
    this->pixel_side_length = length;
    //cout << this->pixel_side_length <<endl;
}

void seeds::initial_seed_P(int K)
{
    this->K = K;
    int *c = new int[this->K];
    for(int i = 0; i < this->K; ++i)
    {
        c[i] = -1;
    }
    this->seed_X = c;

    int *b = new int[this->K];
    for(int i = 0; i < this->K; ++i)
    {
        b[i] = -1;
    }
    this->seed_Y = b;

}

void seeds::initial_pixel_S()
{
    double **a =new double* [this->height];
    for(int i = 0; i < this->height; ++i)
    {
        a[i] = new double[this->width];
    }
    
    for(int i = 0; i < this->height; ++i)
    {
        for(int j = 0; j < this->width; ++j)
        {
            a[i][j] = 0;
        }
    }
    
    this->pixel_S = a; 
}

double seeds::relu(double value)
{
    double output = 0;
    if(value <= 0)
    {
        output = 0;
    }
    else
    {
        output = value;
    }
    return output;
}

double seeds::refine_euclidean(double* pixel1, double* pixel2)
{
    double M = 0;
    M = 1.732*sqrt(pow(0.299,2) * pow((pixel1[2] - pixel2[2]), 2) + pow(0.587,2) * pow((pixel1[1] - pixel2[1]), 2) + pow(0.114,2) * pow((pixel1[0] - pixel2[0]), 2));
    return M;
}


double seeds::caculate_S()
{
    int move[4][2]={{1,0},{-1,0},{0,-1},{0,1}};
    Mat copyimg = this->img.clone();
    //cvtColor(img,copyimg,CV_BGR2GRAY);
    for(int i = 0; i < this->height; ++i)
    {
        for(int j = 0; j < this->width; ++j)
        {
            double* middle = new double[3]; middle[0] = copyimg.at<cv::Vec3b>(i,j)[0]; middle[1] = copyimg.at<cv::Vec3b>(i,j)[1]; middle[2] = copyimg.at<cv::Vec3b>(i,j)[2];
            double* down = new double[3]; double* right = new double[3]; double* left = new double[3]; double* up = new double[3]; 
            //double middle = copyimg.at<uchar>(i,j);
            if(i+move[0][0] < this->height)
            {
                down[0] = copyimg.at<cv::Vec3b>(i+move[0][0],j+move[0][1])[0]; down[1] = copyimg.at<cv::Vec3b>(i+move[0][0],j+move[0][1])[1];
                down[2] = copyimg.at<cv::Vec3b>(i+move[0][0],j+move[0][1])[2];
            }
            if(j+move[3][1] < this->width)
            {
                right[0] = copyimg.at<cv::Vec3b>(i+move[3][0],j+move[3][1])[0]; right[1] = copyimg.at<cv::Vec3b>(i+move[3][0],j+move[3][1])[1];
                right[2] = copyimg.at<cv::Vec3b>(i+move[3][0],j+move[3][1])[2];
            }
            if(j+move[2][1] >= 0)
            {
                left[0] = copyimg.at<cv::Vec3b>(i+move[2][0],j+move[2][1])[0]; left[1] = copyimg.at<cv::Vec3b>(i+move[2][0],j+move[2][1])[1];
                left[2] = copyimg.at<cv::Vec3b>(i+move[2][0],j+move[2][1])[2]; 
            }
            if(i+move[1][0] >= 0)
            {
                up[0] = copyimg.at<cv::Vec3b>(i+move[1][0],j+move[1][1])[0]; up[1] = copyimg.at<cv::Vec3b>(i+move[1][0],j+move[1][1])[1];
                up[2] = copyimg.at<cv::Vec3b>(i+move[1][0],j+move[1][1])[2];
            }

            //cout << middle <<" ";
            if(i == 0||j == 0||i == this->height-1||j == this->width-1)
            {
                if(i == 0&&j == 0)
                {
                    this->pixel_S[i][j] = pow(this->pixel_side_length,2) + (refine_euclidean(middle, down) + refine_euclidean(middle, right))*this->pixel_side_length;
                }
                else if(i == 0&&j == this->width-1)
                {
                    this->pixel_S[i][j] = pow(this->pixel_side_length,2) + (refine_euclidean(middle, down) + refine_euclidean(middle, left))*this->pixel_side_length;
                }
                else if(i == this->height-1&&j == 0)
                {
                    this->pixel_S[i][j] = pow(this->pixel_side_length,2) + (refine_euclidean(middle, up) + refine_euclidean(middle, right))*this->pixel_side_length;
                }
                else if(i == this->height-1&&j == this->width-1)
                {
                    this->pixel_S[i][j] = pow(this->pixel_side_length,2) + (refine_euclidean(middle, up) + refine_euclidean(middle, left))*this->pixel_side_length;
                }
                else if(i == 0)
                {
                    this->pixel_S[i][j] = pow(this->pixel_side_length,2) + (refine_euclidean(middle, down) + refine_euclidean(middle, right) + refine_euclidean(middle, left))*this->pixel_side_length;
                }
                else if(i == this->height-1)
                {
                    this->pixel_S[i][j] = pow(this->pixel_side_length,2) + (refine_euclidean(middle, up) + refine_euclidean(middle, right) + refine_euclidean(middle, left))*this->pixel_side_length;
                }
                else if(j == 0)
                {
                    this->pixel_S[i][j] = pow(this->pixel_side_length,2) + (refine_euclidean(middle, down) + refine_euclidean(middle, right) + refine_euclidean(middle, up))*this->pixel_side_length;
                }
                else if(j == this->width-1)
                {
                    this->pixel_S[i][j] = pow(this->pixel_side_length,2) + (refine_euclidean(middle, up) + refine_euclidean(middle, down) + refine_euclidean(middle, left))*this->pixel_side_length;
                }
            }
            else
            {
                this->pixel_S[i][j] = pow(this->pixel_side_length,2) + (refine_euclidean(middle, up) + refine_euclidean(middle, down) + refine_euclidean(middle, left) + refine_euclidean(middle, right))*this->pixel_side_length;
            }
            this->S +=  this->pixel_S[i][j];
            delete[] middle; delete[] up; delete[] down; delete[] left; delete[] right;
        }
    }

    return this->S;
}

double seeds::caculate_qiqi_capacity()
{
    this->qiqi_capacity = this->S/double(this->K);
    return this->qiqi_capacity;
}

double seeds::slide_windows_area(int x, int y)
{
    double area = 0;
    if(x + 1 < this->height && y + 1 < this->width && x - 1 >= 0 && y - 1 >= 0)
    {
        area = sqrt(pow(this->pixel_S[x + 1][y] - this->pixel_S[x - 1][y], 2)) + sqrt(pow(this->pixel_S[x][y + 1] - this->pixel_S[x][y - 1], 2));
    }
    else
    {
        area = 50;
    }
    return area;
}

void seeds::run_seeds()
{
    this->initial_seed_P(this->K);
    double MinV = 99999;
    double T = 0;//temp capacity
    int k = 0;
    int range = 1;//if set min gradient
    for(int i = 0; i < this->height; ++i)
    {
        for(int j = 0; j <this->width; ++j)
        {
            if(T >= this->qiqi_capacity)
            {
                T = T - this->qiqi_capacity;
                for(int p = -range; p <= range; ++p)
                {
                    for(int q = -range; q <= range; ++q)
                    {
                        if(j + p >= 0 && j + p < this->width && i + p >= 0 && i + p < this->height)
                        {
                            if(MinV > slide_windows_area(i + p, j + p))
                            {
                                MinV = slide_windows_area(i + p, j + p);
                                this->seed_X[k] = i + p;
                                this->seed_Y[k] = j + p;
                            }
                        } 
                    }
                }
                k++;
                MinV = 99999;
            }
            else
            {
                T += this->pixel_S[i][j];
            }
        }
    }
}

void seeds::run_seeds1()
{
    
    int T = 0;//temp capacity
    int k = 0;
    int GV = this->height*this->width/this->K;

    for(int i = 0; i < this->height; ++i)
    {
        for(int j = 0; j <this->width; ++j)
        {
            if(T >= GV)
            {
                T = 0;
                
                
                this->seed_X[k] = i;
                this->seed_Y[k] = j;
                k++;
            }
            else
            {
                T += 1;
                
            }
        }
    }
}
void seeds::run_seeds2()//SLIC等距离初始化
{
    double D = sqrt(this->height*this->width/this->K);
    //int remainder_x = this->height%D;
    //int remainder_y = this->width%D;
    int x_1 = (this->height)/int(D)+2;
    int y_1 = (this->width)/int(D)+2;
    this->initial_seed_P(x_1*y_1);
    int k = 0;
    for(int i = D/2; i < this->height; i += D)
    {
        for(int j = D/2; j < this->width; j += D)
        {
            //cout<<i<<" "<<j<<endl;
            this->seed_X[k] = i;
            this->seed_Y[k] = j;
            k++;
        }
    }
    cout<<"seeds: "<<k+1<<endl;
}


seeds::seeds(const Mat img, int length, int K)//initial operation
{
    this->img = img;
    this->K = K;
    this->height = img.rows;
    this->width = img.cols;
    this->S = 0;
    this->initial_pixel_S();
    this->set_pixel_side_length(length);
    this->caculate_S();
    this->caculate_qiqi_capacity();
    this->run_seeds();
}
seeds::~seeds()
{
    
    for(int i = 0; i < height; i++)
    {
        delete[] this->pixel_S[i];
    }
    delete[] this->pixel_S;
    delete[] this->seed_X;
    delete[] this->seed_Y;
    //cout<<"delete seeeds class memory successfully!"<<endl;
    //delete[] &this->img;
}

//**************************class seeds*********************************



//**************************class growth*********************************

void growth::initial_seeds_lable(int* X, int* Y, int height, int width)
{
    int**a = new int* [height];
    for(int k = 0; k < height; ++k)
    {
        a[k] = new int[width];
    }
    for(int i = 0; i < height; ++i)
    {
        for(int j = 0; j < width; ++j)
        {
            a[i][j] = -1;
        }
    }
    this->lable = a;

    int k = 0;
    while(X[k] != -1)
    {
        k++;
    }
    this->soil_mean_color_number = k;
    double**b = new double* [k];
    for(int j = 0; j < k; ++j)
    {
        b[j] = new double[3];
    }
    for(int p = 0; p < k; ++p)
    {
        for(int q = 0; q < 3; ++q)
        {
            b[p][q] = 0;
        }
    }

    this->soil_mean_color = b;


    int i = 0;
    while(X[i] != -1)
    {
        this->lable[X[i]][Y[i]] = i;
        this->num_of_nodes_in_each_sp.push_back(0);
        this->push_new_seeds(X[i],Y[i]);
        i++;
    }
}

void growth::initial_gh(int height, int width)
{
    int Adjacency = 8;
    double ***a = new double** [height];
    for(int i = 0; i < height; ++i)
    {
        a[i] = new double* [width];
        for(int j = 0; j < width; ++j)
        {
            a[i][j] = new double [Adjacency];
        }
    }
    
    for(int i = 0; i < height; ++i)
    {
        for(int j = 0; j < width; ++j)
        {
            for(int k = 0; k < Adjacency; ++k)
            {
                a[i][j][k] = 0;
            }
        }
    }
    this->gh = a;
}

double growth::refine_euclidean(double* pixel1, double* pixel2)
{
    double M = 0;
    M = 1.3*sqrt(pow(0.299,2) * pow((pixel1[2] - pixel2[2]), 2) + pow(0.587,2) * pow((pixel1[1] - pixel2[1]), 2) + pow(0.114,2) * pow((pixel1[0] - pixel2[0]), 2));
    return M;
}


void growth::initial_h(int height, int width, Mat img)
{
    int Adjacency = 8;
    double ***a = new double** [height];
    for(int i = 0; i < height; ++i)
    {
        a[i] = new double* [width];
        for(int j = 0; j < width; ++j)
        {
            a[i][j] = new double [Adjacency];
        }
    }
    for(int i = 0; i < height; ++i)
    {
        for(int j = 0; j < width; ++j)
        {
            for(int k = 0; k < Adjacency; ++k)
            {
                a[i][j][k] = -1;
            }
        }
    }
    this->h = a;
    int channel = 3;
    int move[8][2]={{1,0},{-1,0},{0,-1},{0,1},{-1,-1},{-1,1},{1,-1},{1,1}};
    for(int i = 0; i < height; ++i)
    {
        for(int j = 0; j < width; ++j)
        {
            double* middle = new double[channel];
            middle[0] = img.at<cv::Vec3b>(i,j)[0]; middle[1] = img.at<cv::Vec3b>(i,j)[1]; middle[2] = img.at<cv::Vec3b>(i,j)[2];
            double* down = new double[channel];
            if(i+move[0][0] < height)
            {
                down[0] = img.at<cv::Vec3b>(i+move[0][0],j+move[0][1])[0]; down[1] = img.at<cv::Vec3b>(i+move[0][0],j+move[0][1])[1]; down[2] = img.at<cv::Vec3b>(i+move[0][0],j+move[0][1])[2];
            }
            double* right = new double[channel];
            if(j+move[3][1] < width)
            {
                right[0] = img.at<cv::Vec3b>(i+move[3][0],j+move[3][1])[0]; right[1] = img.at<cv::Vec3b>(i+move[3][0],j+move[3][1])[1]; right[2] = img.at<cv::Vec3b>(i+move[3][0],j+move[3][1])[2];
            }
            double* left = new double[channel];
            if(j+move[2][1] >= 0)
            {
                left[0] = img.at<cv::Vec3b>(i+move[2][0],j+move[2][1])[0]; left[1] = img.at<cv::Vec3b>(i+move[2][0],j+move[2][1])[1]; left[2] = img.at<cv::Vec3b>(i+move[2][0],j+move[2][1])[2];
            }
            double* up = new double[channel];
            if(i+move[1][0] >= 0)
            {
                up[0] = img.at<cv::Vec3b>(i+move[1][0],j+move[1][1])[0]; up[1] = img.at<cv::Vec3b>(i+move[1][0],j+move[1][1])[1]; up[2] = img.at<cv::Vec3b>(i+move[1][0],j+move[1][1])[2];
            }
            
            double* right_down = new double[channel];
            if(i+move[7][0] < height && j+move[7][1] < width)
            {
                right_down[0] = img.at<cv::Vec3b>(i+move[7][0],j+move[7][1])[0]; right_down[1] = img.at<cv::Vec3b>(i+move[7][0],j+move[7][1])[1]; right_down[2] = img.at<cv::Vec3b>(i+move[7][0],j+move[7][1])[2];
            }
            double* right_up = new double[channel];
            if(i+move[5][0] >= 0 && j+move[5][1] < width)
            {
                right_up[0] = img.at<cv::Vec3b>(i+move[5][0],j+move[5][1])[0]; right_up[1] = img.at<cv::Vec3b>(i+move[5][0],j+move[5][1])[1]; right_up[2] = img.at<cv::Vec3b>(i+move[5][0],j+move[5][1])[2];
            }
            double* left_down = new double[channel];
            if(i+move[6][0] < height && j+move[6][1] >= 0)
            {
                left_down[0] = img.at<cv::Vec3b>(i+move[6][0],j+move[6][1])[0]; left_down[1] = img.at<cv::Vec3b>(i+move[6][0],j+move[6][1])[1]; left_down[2] = img.at<cv::Vec3b>(i+move[6][0],j+move[6][1])[2];
            }
            double* left_up = new double[channel];
            if(i+move[4][0] >= 0 && j+move[4][1] >= 0)
            {
                left_up[0] = img.at<cv::Vec3b>(i+move[4][0],j+move[4][1])[0]; left_up[1] = img.at<cv::Vec3b>(i+move[4][0],j+move[4][1])[1]; left_up[2] = img.at<cv::Vec3b>(i+move[4][0],j+move[4][1])[2];
            }
            //cout << middle <<" ";
            if(i == 0||j == 0||i == height-1||j == width-1)
            {
                if(i == 0&&j == 0)
                {
                    this->h[i][j][0] = -1;//up
                    this->h[i][j][1] = refine_euclidean(middle, down);//down
                    this->h[i][j][2] = -1;//left
                    this->h[i][j][3] = refine_euclidean(middle, right);//right

                    this->h[i][j][4] = -1;//left_up
                    this->h[i][j][5] = -1;//right_up
                    this->h[i][j][6] = -1;//left_down
                    this->h[i][j][7] = refine_euclidean(middle, right_down);//right_down
                }
                else if(i == 0&&j == width-1)
                {
                    this->h[i][j][0] = -1;//up
                    this->h[i][j][1] = refine_euclidean(middle, down);//down
                    this->h[i][j][2] = refine_euclidean(middle, left);//left
                    this->h[i][j][3] = -1;//right

                    this->h[i][j][4] = -1;//left_up
                    this->h[i][j][5] = -1;//right_up
                    this->h[i][j][6] = refine_euclidean(middle, left_down);//left_down
                    this->h[i][j][7] = -1;//right_down
                }
                else if(i == height-1&&j == 0)
                {
                    this->h[i][j][0] = refine_euclidean(middle, up);//up
                    this->h[i][j][1] = -1;//down
                    this->h[i][j][2] = -1;//left
                    this->h[i][j][3] = refine_euclidean(middle, right);//right

                    this->h[i][j][4] = -1;//left_up
                    this->h[i][j][5] = refine_euclidean(middle, right_up);//right_up
                    this->h[i][j][6] = -1;//left_down
                    this->h[i][j][7] = -1;//right_down
                }
                else if(i == height-1&&j == width-1)
                {
                    this->h[i][j][0] = refine_euclidean(middle, up);//up
                    this->h[i][j][1] = -1;//down
                    this->h[i][j][2] = refine_euclidean(middle, left);//left
                    this->h[i][j][3] = -1;//right

                    this->h[i][j][4] = refine_euclidean(middle, left_up);//left_up
                    this->h[i][j][5] = -1;//right_up
                    this->h[i][j][6] = -1;//left_down
                    this->h[i][j][7] = -1;//right_down
                }
                else if(i == 0)
                {
                    this->h[i][j][0] = -1;//up
                    this->h[i][j][1] = refine_euclidean(middle, down);//down
                    this->h[i][j][2] = refine_euclidean(middle, left);//left
                    this->h[i][j][3] = refine_euclidean(middle, right);//right

                    this->h[i][j][4] = -1;//left_up
                    this->h[i][j][5] = -1;//right_up
                    this->h[i][j][6] = refine_euclidean(middle, left_down);//left_down
                    this->h[i][j][7] = refine_euclidean(middle, right_down);//right_down
                }
                else if(i == height-1)
                {
                    this->h[i][j][0] = refine_euclidean(middle, up);//up
                    this->h[i][j][1] = -1;//down
                    this->h[i][j][2] = refine_euclidean(middle, left);//left
                    this->h[i][j][3] = refine_euclidean(middle, right);//right

                    this->h[i][j][4] = refine_euclidean(middle, left_up);//left_up
                    this->h[i][j][5] = refine_euclidean(middle, right_up);//right_up
                    this->h[i][j][6] = -1;//left_down
                    this->h[i][j][7] = -1;//right_down
                }
                else if(j == 0)
                {
                    this->h[i][j][0] = refine_euclidean(middle, up);//up
                    this->h[i][j][1] = refine_euclidean(middle, down);//down
                    this->h[i][j][2] = -1;//left
                    this->h[i][j][3] = refine_euclidean(middle, right);//right

                    this->h[i][j][4] = -1;//left_up
                    this->h[i][j][5] = refine_euclidean(middle, right_up);//right_up
                    this->h[i][j][6] = -1;//left_down
                    this->h[i][j][7] = refine_euclidean(middle, right_down);//right_down
                }
                else if(j == width-1)
                {
                    this->h[i][j][0] = refine_euclidean(middle, up);//up
                    this->h[i][j][1] = refine_euclidean(middle, down);//down
                    this->h[i][j][2] = refine_euclidean(middle, left);//left
                    this->h[i][j][3] = -1;//right

                    this->h[i][j][4] = refine_euclidean(middle, left_up);//left_up
                    this->h[i][j][5] = -1;//right_up
                    this->h[i][j][6] = refine_euclidean(middle, left_down);//left_down
                    this->h[i][j][7] = -1;//right_down
                }
            }
            else
            {
                this->h[i][j][0] = refine_euclidean(middle, up);//up
                this->h[i][j][1] = refine_euclidean(middle, down);//down
                this->h[i][j][2] = refine_euclidean(middle, left);//left
                this->h[i][j][3] = refine_euclidean(middle, right);//right

                this->h[i][j][4] = refine_euclidean(middle, left_up);//left_up
                this->h[i][j][5] = refine_euclidean(middle, right_up);//right_up
                this->h[i][j][6] = refine_euclidean(middle, left_down);//left_down
                this->h[i][j][7] = refine_euclidean(middle, right_down);//right_down
            }
            delete[] middle;
            delete[] up;
            delete[] left;
            delete[] down;
            delete[] right;
            delete[] left_up;
            delete[] left_down;
            delete[] right_down;
            delete[] right_up;
            
        }
    }
}



void growth::initial_state(int height, int width)
{
    int Adjacency = 8;
    int ***a = new int** [height];
    for(int i = 0; i < height; ++i)
    {
        a[i] = new int* [width];
        for(int j = 0; j < width; ++j)
        {
            a[i][j] = new int[Adjacency];
        }
    }
    for(int i = 0; i < height; ++i)
    {
        for(int j = 0; j < width; ++j)
        {
            for(int k = 0; k < Adjacency; ++k)
            {
                a[i][j][k] = -2;//-2 means the wall, -1 means unlabled node, 0-K means 0-K superpixel labels
            }
        }
    }
    this->ad_state = a;
    int move[8][2]={{-1,0},{1,0},{0,-1},{0,1},{-1,-1},{-1,1},{1,-1},{1,1}};//control up, down, left, right, left_up, right_up, left_down, right_down
    for(int i = 0; i < height; ++i)
    {
        for(int j = 0; j < width; ++j)
        {
            for(int k = 0; k < Adjacency; ++k)
            {
                if(i + move[k][0] >= 0 && j + move[k][1] >= 0 && i + move[k][0] < height && j + move[k][1] < width)
                {
                    this->ad_state[i][j][k] = this->lable[i + move[k][0]][j + move[k][1]];
                }
            }
        }
    }

}

void growth::push_new_seeds(int seed_X, int seed_Y)
{
    this->Seeds_X.push_back(seed_X);
    this->Seeds_Y.push_back(seed_Y);
    int label = this->lable[seed_X][seed_Y];

    this->num_of_nodes_in_each_sp[label]++;
    double R = this->img.at<cv::Vec3b>(seed_X,seed_Y)[2];
    double G = this->img.at<cv::Vec3b>(seed_X,seed_Y)[1];
    double B = this->img.at<cv::Vec3b>(seed_X,seed_Y)[0];
    this->soil_mean_color[label][0] = ((this->num_of_nodes_in_each_sp[label]-1)*this->soil_mean_color[label][0] + R)/this->num_of_nodes_in_each_sp[label];
    this->soil_mean_color[label][1] = ((this->num_of_nodes_in_each_sp[label]-1)*this->soil_mean_color[label][1] + G)/this->num_of_nodes_in_each_sp[label];
    this->soil_mean_color[label][2] = ((this->num_of_nodes_in_each_sp[label]-1)*this->soil_mean_color[label][2] + B)/this->num_of_nodes_in_each_sp[label];
}



void growth::caculate_soil_quality(int height, int width, double** pixel_S)
{
    double** a = new double* [height];
    for(int x = 0; x < height; ++x)
    {
        a[x] = new double[width];
        for(int y = 0; y < width; ++y)
        {
            a[x][y] = 0;
        }
    }
    this->soil_quality = a;

    for(int x = 0; x < height; ++x)
    {
        for(int y = 0; y < width; ++y)
        {   double area = 0;
            if(x - 1 >= 0 && y - 1 >= 0)
            {
                area += fabs(pixel_S[x][y] - pixel_S[x - 1][y - 1]);
            }
            if(x - 1 >= 0)
            {
                area += fabs(pixel_S[x][y] - pixel_S[x - 1][y]);
            }
            if(x - 1 >= 0 && y + 1 < width)
            {
                area += fabs(pixel_S[x][y] - pixel_S[x - 1][y + 1]);
            }
            if(y - 1 >= 0)
            {
                area += fabs(pixel_S[x][y] - pixel_S[x][y - 1]);
            }
            if(y + 1 < width)
            {
                area += fabs(pixel_S[x][y] - pixel_S[x][y + 1]);
            }
            if(x + 1 < height && y - 1 >= 0)
            {
                area += fabs(pixel_S[x][y] - pixel_S[x + 1][y - 1]);
            }
            if(x + 1 < height)
            {
                area += fabs(pixel_S[x][y] - pixel_S[x + 1][y]);
            }
            if(x + 1 < height && y + 1 < width)
            {
                area += fabs(pixel_S[x][y] - pixel_S[x + 1][y + 1]);
            }
            this->soil_quality[x][y] = area;
        }
    }
    
}

double* growth::returnRGB(int X, int Y)
{
    double* a = new double[3];
    a[0] = this->img.at<cv::Vec3b>(X, Y)[2];
    a[1] = this->img.at<cv::Vec3b>(X, Y)[1];
    a[2] = this->img.at<cv::Vec3b>(X, Y)[0];
    return a;
}



double growth::velocity(double alpha, double lambda, double beta, double theta, int X, int Y, int num)
{
    return alpha*pow(this->gh[X][Y][num]-beta,2)+exp(-this->h[X][Y][num]/lambda);
}



void growth::update(int height, int width, double alpha, double lambda, double beta, double tau)
{
    double step = 1.5;
    double theta = 0;
    while(!this->Seeds_X.empty())
    {
        for(list<int>::iterator iter1 = this->Seeds_X.begin(), iter2 = this->Seeds_Y.begin(); iter1 != this->Seeds_X.end();)
        {   
            int X = *iter1;
            int Y = *iter2;
            //cout<<X<<"  "<<Y<<endl;
            if(this->ad_state[X][Y][0] != -1 && this->ad_state[X][Y][1] != -1 && this->ad_state[X][Y][2] != -1 && this->ad_state[X][Y][3] != -1)
            {
                list<int>::iterator tmp_iter1, tmp_iter2;
                tmp_iter1 = iter1;
                tmp_iter2 = iter2;
                ++iter1;
                ++iter2;
                this->Seeds_X.erase(tmp_iter1);
                this->Seeds_Y.erase(tmp_iter2);
            }
            else
            {
                ++iter1;
                ++iter2;
            }
        }

        for(list<int>::iterator iter3 = this->Seeds_X.begin(), iter4 = this->Seeds_Y.begin(); iter3 != this->Seeds_X.end();)
        {   
            int X = *iter3;
            int Y = *iter4;
            if(this->ad_state[X][Y][0] == -1)//up
            {
                int label = this->lable[X][Y];
                double* RGB = this->returnRGB(X - 1, Y);
                
                this->h[X - 1][Y][1] = this->refine_euclidean(this->soil_mean_color[label],RGB);
                delete[] RGB;
                
                this->gh[X - 1][Y][1] = this->gh[X - 1][Y][1] + step* velocity(alpha, lambda, beta, theta, X-1, Y, 1);//update gh
                
                if(this->gh[X - 1][Y][1] >= this->h[X - 1][Y][1] || this->h[X - 1][Y][1] <= tau)//judge if produce new seed
                {
                    this->lable[X - 1][Y] = this->lable[X][Y];//update label
                    //update ad_state for up down left right left_up right_up left_down right_down
                    this->ad_state[X][Y][0] = this->lable[X][Y];
                    if(X - 2 >= 0)
                    {
                        this->ad_state[X - 2][Y][1] = this->lable[X][Y];
                    }
                    if(Y - 1 >= 0 && X - 1 >= 0)
                    {
                        this->ad_state[X - 1][Y - 1][3] = this->lable[X][Y];
                    }
                    if(X - 1 >= 0 && Y + 1 < width)
                    {
                        this->ad_state[X - 1][Y + 1][2] = this->lable[X][Y];
                    }
                    if(X - 2 >= 0 && Y - 1 >= 0)
                    {
                        this->ad_state[X - 2][Y - 1][7] = this->lable[X][Y];
                    }
                    if(X - 2 >= 0 && Y + 1 < width)
                    {
                        this->ad_state[X - 2][Y + 1][6] = this->lable[X][Y];
                    }
                    if(Y - 1 >= 0)
                    {
                        this->ad_state[X][Y - 1][5] = this->lable[X][Y];
                    }
                    if(Y + 1 < width)
                    {
                        this->ad_state[X][Y + 1][4] = this->lable[X][Y];
                    }
                    this->push_new_seeds(X - 1, Y);

                }
            }

            if(this->ad_state[X][Y][1] == -1)//down
            {
                int label = this->lable[X][Y];
                double* RGB = this->returnRGB(X + 1, Y);
                
                this->h[X + 1][Y][0] = this->refine_euclidean(this->soil_mean_color[label],RGB);
                
                delete[] RGB;
                

                this->gh[X + 1][Y][0] = this->gh[X + 1][Y][0] + step*velocity(alpha, lambda, beta, theta, X+1, Y, 0);//update gh
               
                if(this->gh[X + 1][Y][0] >= this->h[X + 1][Y][0] || this->h[X + 1][Y][0] <= tau)//judge if produce new seed
                {
                    this->lable[X + 1][Y] = this->lable[X][Y];//update label
                    this->ad_state[X][Y][1] = this->lable[X][Y];//update ad_state
                    if(X + 2 < height)
                    {
                        this->ad_state[X + 2][Y][0] = this->lable[X][Y];
                    }
                    if(Y - 1 >= 0 && X + 1 < height)
                    {
                        this->ad_state[X + 1][Y - 1][3] = this->lable[X][Y];
                    }
                    if(X + 1 < height && Y + 1 < width)
                    {
                        this->ad_state[X + 1][Y + 1][2] = this->lable[X][Y];
                    }
                    if(Y - 1 >= 0)
                    {
                        this->ad_state[X][Y - 1][7] = this->lable[X][Y];
                    }
                    if(Y + 1 < width)
                    {
                        this->ad_state[X][Y + 1][6] = this->lable[X][Y];
                    }
                    if(X + 2 < height && Y - 1 >= 0)
                    {
                        this->ad_state[X + 2][Y - 1][5] = this->lable[X][Y];
                    }
                    if(X + 2 < height && Y + 1 < width)
                    {
                        this->ad_state[X + 2][Y + 1][4] = this->lable[X][Y];
                    }
                    this->push_new_seeds(X + 1, Y);
                }
            }

            if(this->ad_state[X][Y][2] == -1)//left
            {
                int label = this->lable[X][Y];
                
                double* RGB = this->returnRGB(X, Y - 1);
                
                this->h[X][Y - 1][3] = this->refine_euclidean(this->soil_mean_color[label],RGB);
                
                delete[] RGB;
                

                this->gh[X][Y - 1][3] = this->gh[X][Y - 1][3] + step*velocity(alpha, lambda, beta, theta, X, Y-1, 3);
                
                if(this->gh[X][Y - 1][3] >= this->h[X][Y - 1][3] || this->h[X][Y - 1][3] <= tau)//judge if produce new seed
                {
                    this->lable[X][Y - 1] = this->lable[X][Y];//update label
                    this->ad_state[X][Y][2] = this->lable[X][Y];//update ad_state
                    if(Y - 2 >= 0)
                    {
                        this->ad_state[X][Y - 2][3] = this->lable[X][Y];
                    }
                    if(Y - 1 >= 0 && X - 1 >= 0)
                    {
                        this->ad_state[X - 1][Y - 1][1] = this->lable[X][Y];
                    }
                    if(X + 1 < height && Y - 1 >= 0)
                    {
                        this->ad_state[X + 1][Y - 1][0] = this->lable[X][Y];
                    }
                    if(X - 1 >= 0 && Y - 2 >= 0)
                    {
                        this->ad_state[X - 1][Y - 2][7] = this->lable[X][Y];
                    }
                    if(X - 1 >= 0)
                    {
                        this->ad_state[X - 1][Y][6] = this->lable[X][Y];
                    }
                    if(X + 1 < height && Y - 2 >= 0)
                    {
                        this->ad_state[X + 1][Y - 2][5] = this->lable[X][Y];
                    }
                    if(X + 1 < height)
                    {
                        this->ad_state[X + 1][Y][4] = this->lable[X][Y];
                    }
                    this->push_new_seeds(X, Y - 1);
                }
            }

            if(this->ad_state[X][Y][3] == -1)//right
            {
                int label = this->lable[X][Y];
                double* RGB = this->returnRGB(X, Y + 1);
                
                this->h[X][Y + 1][2] = this->refine_euclidean(this->soil_mean_color[label],RGB);
                
                delete[] RGB;
               

                this->gh[X][Y + 1][2] = this->gh[X][Y + 1][2] + step*velocity(alpha, lambda, beta, theta, X, Y+1, 2);
                
                if(this->gh[X][Y + 1][2] >= this->h[X][Y + 1][2] || this->h[X][Y + 1][2] <= tau)//judge if produce new seed
                {
                    this->lable[X][Y + 1] = this->lable[X][Y];//update label
                    this->ad_state[X][Y][3] = this->lable[X][Y];//update ad_state
                    if(Y + 2 < width)
                    {
                        this->ad_state[X][Y + 2][2] = this->lable[X][Y];
                    }
                    if(Y + 1 < width && X - 1 >= 0)
                    {
                        this->ad_state[X - 1][Y + 1][1] = this->lable[X][Y];
                    }
                    if(X + 1 < height && Y + 1 < width)
                    {
                        this->ad_state[X + 1][Y + 1][0] = this->lable[X][Y];
                    }
                    if(X - 1 >= 0)
                    {
                        this->ad_state[X - 1][Y][7] = this->lable[X][Y];
                    }
                    if(X - 1 >= 0 && Y + 2 < width)
                    {
                        this->ad_state[X - 1][Y + 2][6] = this->lable[X][Y];
                    }
                    if(X + 1 < height)
                    {
                        this->ad_state[X + 1][Y][5] = this->lable[X][Y];
                    }
                    if(X + 1 < height && Y + 2 < width)
                    {
                        this->ad_state[X + 1][Y + 2][4] = this->lable[X][Y];
                    }
                    this->push_new_seeds(X, Y + 1);
                }
            }

            if(this->ad_state[X][Y][4] == -1)//left_up
            {
                int label = this->lable[X][Y];
                double* RGB = this->returnRGB(X - 1, Y - 1);
                
                this->h[X - 1][Y - 1][7] = this->refine_euclidean(this->soil_mean_color[label],RGB);
                
                delete[] RGB;
                

                this->gh[X - 1][Y - 1][7] = this->gh[X - 1][Y - 1][7] + step*velocity(alpha, lambda, beta, theta, X-1, Y-1, 7);
            
                if(this->gh[X - 1][Y - 1][7] >= this->h[X - 1][Y - 1][7] || this->h[X - 1][Y - 1][7] <= tau)//judge if produce new seed
                {
                    this->lable[X - 1][Y - 1] = this->lable[X][Y];//update label
                    //update ad_state for up down left right left_up right_up left_down right_down
                    this->ad_state[X][Y][4] = this->lable[X][Y];
                    if(X - 2 >= 0 && Y - 1 >= 0)
                    {
                        this->ad_state[X - 2][Y - 1][1] = this->lable[X][Y];
                    }
                    if(Y - 1 >= 0)
                    {
                        this->ad_state[X][Y - 1][0] = this->lable[X][Y];
                    }
                    if(X - 1 >= 0 && Y - 2 >= 0)
                    {
                        this->ad_state[X - 1][Y - 2][3] = this->lable[X][Y];
                    }
                    if(X - 1 >= 0)
                    {
                        this->ad_state[X - 1][Y][2] = this->lable[X][Y];
                    }
                    if(X - 2 >= 0 && Y - 2 >= 0)
                    {
                        this->ad_state[X - 2][Y - 2][7] = this->lable[X][Y];
                    }
                    if(X - 2 >= 0)
                    {
                        this->ad_state[X - 2][Y][6] = this->lable[X][Y];
                    }
                    if(Y - 2 >= 0)
                    {
                        this->ad_state[X][Y - 2][5] = this->lable[X][Y];
                    }
                    this->push_new_seeds(X - 1, Y - 1);

                }
            }

            if(this->ad_state[X][Y][5] == -1)//right_up
            {
                int label = this->lable[X][Y];
                double* RGB = this->returnRGB(X - 1, Y + 1);
                
                this->h[X - 1][Y + 1][6] = this->refine_euclidean(this->soil_mean_color[label],RGB);
                
                delete[] RGB;

                this->gh[X - 1][Y + 1][6] = this->gh[X - 1][Y + 1][6] + step*velocity(alpha, lambda, beta, theta, X-1, Y+1, 6);
               
                if(this->gh[X - 1][Y + 1][6] >= this->h[X - 1][Y + 1][6] || this->h[X - 1][Y + 1][6] <= tau)//judge if produce new seed
                {
                    this->lable[X - 1][Y + 1] = this->lable[X][Y];//update label
                    //update ad_state for up down left right left_up right_up left_down right_down
                    this->ad_state[X][Y][5] = this->lable[X][Y];
                    if(X - 2 >= 0 && Y + 1 < width)
                    {
                        this->ad_state[X - 2][Y + 1][1] = this->lable[X][Y];
                    }
                    if(Y + 1 < width)
                    {
                        this->ad_state[X][Y + 1][0] = this->lable[X][Y];
                    }
                    if(X - 1 >= 0)
                    {
                        this->ad_state[X - 1][Y][3] = this->lable[X][Y];
                    }
                    if(X - 1 >= 0 && Y + 1 < width)
                    {
                        this->ad_state[X - 1][Y + 1][2] = this->lable[X][Y];
                    }
                    if(X - 2 >= 0)
                    {
                        this->ad_state[X - 2][Y][7] = this->lable[X][Y];
                    }
                    if(X - 2 >= 0 && Y + 2 < width)
                    {
                        this->ad_state[X - 2][Y + 2][6] = this->lable[X][Y];
                    }
                    if(Y + 2 < width)
                    {
                        this->ad_state[X][Y + 2][4] = this->lable[X][Y];
                    }
                    this->push_new_seeds(X - 1, Y + 1);

                }
            }

            if(this->ad_state[X][Y][6] == -1)//left_down
            {
                int label = this->lable[X][Y];
                double* RGB = this->returnRGB(X + 1, Y - 1);
                
                this->h[X + 1][Y - 1][5] = this->refine_euclidean(this->soil_mean_color[label],RGB);
               
                delete[] RGB;


                this->gh[X + 1][Y - 1][5] = this->gh[X + 1][Y - 1][5] + step*velocity(alpha, lambda, beta, theta, X+1, Y-1, 5);
            
                if(this->gh[X + 1][Y - 1][5] >= this->h[X + 1][Y - 1][5] || this->h[X + 1][Y - 1][5] <= tau)//judge if produce new seed
                {
                    this->lable[X + 1][Y - 1] = this->lable[X][Y];//update label
                    //update ad_state for up down left right left_up right_up left_down right_down
                    this->ad_state[X][Y][6] = this->lable[X][Y];
                    if(Y - 1 >= 0)
                    {
                        this->ad_state[X][Y - 1][1] = this->lable[X][Y];
                    }
                    if(X + 2 < height && Y - 1 >= 0)
                    {
                        this->ad_state[X + 2][Y - 1][0] = this->lable[X][Y];
                    }
                    if(X + 1 < height && Y - 2 >= 0)
                    {
                        this->ad_state[X + 1][Y - 2][3] = this->lable[X][Y];
                    }
                    if(X + 1 < height)
                    {
                        this->ad_state[X + 1][Y][2] = this->lable[X][Y];
                    }
                    if(Y - 2 >= 0)
                    {
                        this->ad_state[X][Y - 2][7] = this->lable[X][Y];
                    }
                    if(X + 2 < height && Y - 2 >= 0)
                    {
                        this->ad_state[X + 2][Y - 2][5] = this->lable[X][Y];
                    }
                    if(X + 2 < height)
                    {
                        this->ad_state[X + 2][Y][4] = this->lable[X][Y];
                    }
                    this->push_new_seeds(X + 1, Y - 1);

                }
            }

            if(this->ad_state[X][Y][7] == -1)//right_down
            {
                int label = this->lable[X][Y];
                double* RGB = this->returnRGB(X + 1, Y + 1);
                
                this->h[X + 1][Y + 1][4] = this->refine_euclidean(this->soil_mean_color[label],RGB);
                
                delete[] RGB;

                this->gh[X + 1][Y + 1][4] = this->gh[X + 1][Y + 1][4] + step*velocity(alpha, lambda, beta, theta, X+1, Y+1, 4);
                if(this->gh[X + 1][Y + 1][4] >= this->h[X + 1][Y + 1][4] || this->h[X + 1][Y + 1][4] <= tau)//judge if produce new seed
                {
                    this->lable[X + 1][Y + 1] = this->lable[X][Y];//update label
                    //update ad_state for up down left right left_up right_up left_down right_down
                    this->ad_state[X][Y][7] = this->lable[X][Y];
                    if(Y + 1 < width)
                    {
                        this->ad_state[X][Y + 1][1] = this->lable[X][Y];
                    }
                    if(X + 2 < height && Y + 1 < width)
                    {
                        this->ad_state[X + 2][Y + 1][0] = this->lable[X][Y];
                    }
                    if(X + 1 < height)
                    {
                        this->ad_state[X + 1][Y][3] = this->lable[X][Y];
                    }
                    if(X + 1 < height && Y + 2 < width)
                    {
                        this->ad_state[X + 1][Y + 2][2] = this->lable[X][Y];
                    }
                    if(Y + 2 < width)
                    {
                        this->ad_state[X][Y + 2][6] = this->lable[X][Y];
                    }
                    if(X + 2 < height && Y + 2 < width)
                    {
                        this->ad_state[X + 2][Y + 2][4] = this->lable[X][Y];
                    }
                    if(X + 2 < height)
                    {
                        this->ad_state[X + 2][Y][5] = this->lable[X][Y];
                    }
                    this->push_new_seeds(X + 1, Y + 1);

                }
            }

            if(this->ad_state[X][Y][0] != -1 && this->ad_state[X][Y][1] != -1 && this->ad_state[X][Y][2] != -1 && this->ad_state[X][Y][3] != -1 && this->ad_state[X][Y][4] != -1 && this->ad_state[X][Y][5] != -1 && this->ad_state[X][Y][6] != -1 && this->ad_state[X][Y][7] != -1)
            {
                list<int>::iterator tmp_iter3, tmp_iter4;
                tmp_iter3 = iter3;
                tmp_iter4 = iter4;
                ++iter3;
                ++iter4;
                this->Seeds_X.erase(tmp_iter3);
                this->Seeds_Y.erase(tmp_iter4);
            }
            else
            {
                ++iter3;
                ++iter4;
            }
            
        }
        
    }
}


void growth::update_soil_mean(int seed_X, int seed_Y)
{
    int label = this->lable[seed_X][seed_Y];
    this->num_of_nodes_in_each_sp[label]++;
    double R = this->img.at<cv::Vec3b>(seed_X,seed_Y)[2];
    double G = this->img.at<cv::Vec3b>(seed_X,seed_Y)[1];
    double B = this->img.at<cv::Vec3b>(seed_X,seed_Y)[0];
    this->soil_mean_color[label][0] = ((this->num_of_nodes_in_each_sp[label]-1)*this->soil_mean_color[label][0] + R)/this->num_of_nodes_in_each_sp[label];
    this->soil_mean_color[label][1] = ((this->num_of_nodes_in_each_sp[label]-1)*this->soil_mean_color[label][1] + G)/this->num_of_nodes_in_each_sp[label];
    this->soil_mean_color[label][2] = ((this->num_of_nodes_in_each_sp[label]-1)*this->soil_mean_color[label][2] + B)/this->num_of_nodes_in_each_sp[label];
}


void growth::merging(int num)
{
    vector<vector<Point> > sp_pixel(this->num_of_nodes_in_each_sp.size());
    int height = this->img.rows;
    int width = this->img.cols;
    for(int i = 0; i < height; ++i)
    {
        for(int j = 0; j < width; ++j)
        {
            int label = this->lable[i][j];
            sp_pixel[label].push_back(Point(i,j));
        }
    }

    /*//*********debug*********************
    int k = 0;
    for(int i = 0; i < this->num_of_nodes_in_each_sp.size(); ++i)
    {
        if(num_of_nodes_in_each_sp[i] != 0)
        {
            k++;
        }
        
    }
    int u = 0;
    for(int i = 0; i< sp_pixel.size();++i)
    {
        if(!sp_pixel[i].empty())
        {
            u++;
        }
    }
    //*************debug**********************/
    for(int i = 0; i < height; ++i)
    {
        for(int j = 0; j < width - 1; ++j)
        {
            int labelp1 = this->lable[i][j];
            int labelp2 = this->lable[i][j+1];
            if(labelp1 == labelp2)
            {
                continue;
            }
            else
            {
                int R = this->soil_mean_color[labelp1][0]-this->soil_mean_color[labelp2][0];
                int G = this->soil_mean_color[labelp1][1]-this->soil_mean_color[labelp2][1];
                int B = this->soil_mean_color[labelp1][2]-this->soil_mean_color[labelp2][2];
                double M = 1.732*sqrt(pow(R,2)*pow(0.299,2) + pow(G,2)*pow(0.587,2) + pow(B,2)*pow(0.114,2));
                if(M <= num)
                {
                    
                    this->soil_mean_color[labelp1][0] = (this->soil_mean_color[labelp1][0]*this->num_of_nodes_in_each_sp[labelp1] + this->soil_mean_color[labelp2][0]*this->num_of_nodes_in_each_sp[labelp2])/(this->num_of_nodes_in_each_sp[labelp1]+this->num_of_nodes_in_each_sp[labelp2]);
                    this->soil_mean_color[labelp1][1] = (this->soil_mean_color[labelp1][1]*this->num_of_nodes_in_each_sp[labelp1] + this->soil_mean_color[labelp2][1]*this->num_of_nodes_in_each_sp[labelp2])/(this->num_of_nodes_in_each_sp[labelp1]+this->num_of_nodes_in_each_sp[labelp2]);
                    this->soil_mean_color[labelp1][2] = (this->soil_mean_color[labelp1][2]*this->num_of_nodes_in_each_sp[labelp1] + this->soil_mean_color[labelp2][2]*this->num_of_nodes_in_each_sp[labelp2])/(this->num_of_nodes_in_each_sp[labelp1]+this->num_of_nodes_in_each_sp[labelp2]);
            
                    this->num_of_nodes_in_each_sp[labelp1] = this->num_of_nodes_in_each_sp[labelp2] + this->num_of_nodes_in_each_sp[labelp1];
                    this->num_of_nodes_in_each_sp[labelp2] = 0;
                    this->soil_mean_color[labelp2][0] = 0;
                    this->soil_mean_color[labelp2][1] = 0;
                    this->soil_mean_color[labelp2][2] = 0;

                    
                    for(int k = 0; k < sp_pixel[labelp2].size(); ++k)
                    {
                        this->lable[sp_pixel[labelp2][k].x][sp_pixel[labelp2][k].y] = labelp1;
                        sp_pixel[labelp1].push_back(Point(sp_pixel[labelp2][k].x,sp_pixel[labelp2][k].y));
                    }
                    sp_pixel[labelp2].clear();
                }
                
            }
        }
    }
    for(int i = 0; i < width; ++i)
    {
        for(int j = 0; j < height - 1; ++j)
        {
            int labelp1 = this->lable[j][i];
            int labelp2 = this->lable[j+1][i];
            if(labelp1 == labelp2)
            {
                continue;
            }
            else
            {
                int R = this->soil_mean_color[labelp1][0]-this->soil_mean_color[labelp2][0];
                int G = this->soil_mean_color[labelp1][1]-this->soil_mean_color[labelp2][1];
                int B = this->soil_mean_color[labelp1][2]-this->soil_mean_color[labelp2][2];
                double M = 1.732*sqrt(pow(R,2)*pow(0.299,2) + pow(G,2)*pow(0.587,2) + pow(B,2)*pow(0.114,2));
                
                

                if(M <= num)
                {
                    
                    this->soil_mean_color[labelp1][0] = (this->soil_mean_color[labelp1][0]*this->num_of_nodes_in_each_sp[labelp1] + this->soil_mean_color[labelp2][0]*this->num_of_nodes_in_each_sp[labelp2])/(this->num_of_nodes_in_each_sp[labelp1]+this->num_of_nodes_in_each_sp[labelp2]);
                    this->soil_mean_color[labelp1][1] = (this->soil_mean_color[labelp1][1]*this->num_of_nodes_in_each_sp[labelp1] + this->soil_mean_color[labelp2][1]*this->num_of_nodes_in_each_sp[labelp2])/(this->num_of_nodes_in_each_sp[labelp1]+this->num_of_nodes_in_each_sp[labelp2]);
                    this->soil_mean_color[labelp1][2] = (this->soil_mean_color[labelp1][2]*this->num_of_nodes_in_each_sp[labelp1] + this->soil_mean_color[labelp2][2]*this->num_of_nodes_in_each_sp[labelp2])/(this->num_of_nodes_in_each_sp[labelp1]+this->num_of_nodes_in_each_sp[labelp2]);
            
                    this->num_of_nodes_in_each_sp[labelp1] = this->num_of_nodes_in_each_sp[labelp2] + this->num_of_nodes_in_each_sp[labelp1];
                    this->num_of_nodes_in_each_sp[labelp2] = 0;
                    this->soil_mean_color[labelp2][0] = 0;
                    this->soil_mean_color[labelp2][1] = 0;
                    this->soil_mean_color[labelp2][2] = 0;

                    
                    for(int k = 0; k < sp_pixel[labelp2].size(); ++k)
                    {
                        this->lable[sp_pixel[labelp2][k].x][sp_pixel[labelp2][k].y] = labelp1;
                        sp_pixel[labelp1].push_back(Point(sp_pixel[labelp2][k].x,sp_pixel[labelp2][k].y));
                    }
                    sp_pixel[labelp2].clear();
                }
                
            }
        }
    }
    int u = 0;
    for(int i = 0; i< sp_pixel.size();++i)
    {
        if(!sp_pixel[i].empty())
        {
            u++;
        }
    }
    
    cout<<"real_sp: "<<u<<endl;

}

void growth::combine_small_sp(int num)
{
    vector<vector<Point> > sp_pixel(this->num_of_nodes_in_each_sp.size());
    vector<set<int> > neibor(this->num_of_nodes_in_each_sp.size());
    int height = this->img.rows;
    int width = this->img.cols;
    int move[8][2]={{-1,0},{1,0},{0,-1},{0,1},{-1,-1},{-1,1},{1,-1},{1,1}};
    int min_d = 999999;
    for(int i = 0; i < height; ++i)
    {
        for(int j = 0; j < width; ++j)
        {
            int label = this->lable[i][j];
            sp_pixel[label].push_back(Point(i,j));
            for(int m = 0; m < 8; ++m)
            {
                int tx = i + move[m][0];
                int ty = j + move[m][1];
                
                if(tx < 0 || tx >= height || ty < 0 || ty >= width)
                {
                    continue;
                }
                if(this->lable[i][j] != this->lable[tx][ty])
                {
                    neibor[label].insert(this->lable[tx][ty]);
                }
            }
            
        }
    }


    for(int i = 0; i < this->num_of_nodes_in_each_sp.size(); ++i)
    {
        if(num_of_nodes_in_each_sp[i] < num)
        {
            set<int>::iterator j = neibor[i].begin();
            int aim = 0;
            while(j != neibor[i].end())
            {
                int R = this->soil_mean_color[i][0]-this->soil_mean_color[*j][0];
                int G = this->soil_mean_color[i][1]-this->soil_mean_color[*j][1];
                int B = this->soil_mean_color[i][2]-this->soil_mean_color[*j][2];
                double M = 1.732*sqrt(pow(R,2)*pow(0.299,2) + pow(G,2)*pow(0.587,2) + pow(B,2)*pow(0.114,2));
                if(M < min_d)
                {
                    min_d = M;
                    aim = *j;
                }
                ++j;
            }
            min_d = 999999;
            this->soil_mean_color[aim][0] = (this->soil_mean_color[i][0]*this->num_of_nodes_in_each_sp[i] + this->soil_mean_color[aim][0]*this->num_of_nodes_in_each_sp[aim])/(this->num_of_nodes_in_each_sp[i]+this->num_of_nodes_in_each_sp[aim]);
            this->soil_mean_color[aim][1] = (this->soil_mean_color[i][1]*this->num_of_nodes_in_each_sp[i] + this->soil_mean_color[aim][1]*this->num_of_nodes_in_each_sp[aim])/(this->num_of_nodes_in_each_sp[i]+this->num_of_nodes_in_each_sp[aim]);
            this->soil_mean_color[aim][2] = (this->soil_mean_color[i][2]*this->num_of_nodes_in_each_sp[i] + this->soil_mean_color[aim][2]*this->num_of_nodes_in_each_sp[aim])/(this->num_of_nodes_in_each_sp[i]+this->num_of_nodes_in_each_sp[aim]);
    
            this->num_of_nodes_in_each_sp[aim] = this->num_of_nodes_in_each_sp[aim] + this->num_of_nodes_in_each_sp[i];
            this->num_of_nodes_in_each_sp[i] = 0;
            this->soil_mean_color[i][0] = 0;
            this->soil_mean_color[i][1] = 0;
            this->soil_mean_color[i][2] = 0;


            for(int k = 0; k < sp_pixel[i].size(); ++k)
            {
                this->lable[sp_pixel[i][k].x][sp_pixel[i][k].y] = aim;
                sp_pixel[aim].push_back(Point(sp_pixel[i][k].x,sp_pixel[i][k].y));
            }
        }
    }
    int k = 0;
    for(int i = 0; i < this->num_of_nodes_in_each_sp.size(); ++i)
    {
        if(num_of_nodes_in_each_sp[i] == 0)
         continue;
        k++;
    }
}


growth::growth(double** pixel_s, int* seed_X, int* seed_Y, int height, int width, Mat image, double alpha, double lambda, double beta, double tau)
{
    this->img = image;
    this->initial_seeds_lable(seed_X, seed_Y, height, width);
    this->initial_gh(height, width);
    this->initial_h(height, width, image);
    this->initial_state(height, width);
    this->update(height, width, alpha, lambda, beta, tau);
    //this->combine_small_sp(20);
    //this->merging(0);
        

}

growth::~growth()
{
    
    
    for(int i = 0; i < img.rows; i++)
    {
        for(int j = 0; j < img.cols; j++)
        {
            delete[] this->gh[i][j];
            delete[] this->h[i][j];
            delete[] this->ad_state[i][j];
        }
        delete[] this->gh[i];
        delete[] this->h[i];
        delete[] this->ad_state[i];
    }
    
    delete[] this->gh;
    delete[] this->h;
    delete[] this->ad_state;
    for(int i = 0; i < img.rows; i++)
    {
        delete[] this->lable[i];
        
        //delete[] this->soil_quality[i];
    }
    for(int i = 0; i < this->soil_mean_color_number; i++)
    {
        delete[] this->soil_mean_color[i];
    }
    delete[] this->lable;
    //delete[] this->soil_quality;
    delete[] this->soil_mean_color;
    //delete[] &this->img;
    vector<int>().swap(this->num_of_nodes_in_each_sp);
    //cout<<"delete growth class memory successfully!"<<endl;
}
