#include <iostream>
#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;

class draw{
    public:
        Mat DrawSeeds(int* X, int* Y, Mat img, int radiu);
        Mat DrawSuperpixelEdge(int **lable, Mat img);
        void OutputLabel(Mat image, int **label, string address);
        Mat DrawMeancolor(int **lable, double **meancolor, Mat img);

};