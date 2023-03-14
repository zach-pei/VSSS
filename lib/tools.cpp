#include <iostream>
#include <opencv2/opencv.hpp>
#include "tools.h"

using namespace std;
using namespace cv;

Mat draw::DrawSeeds(int* X, int* Y, Mat img, int radiu)
{
    int k = 0;
    Mat image = img.clone();
    while(X[k] != -1)
    {
        int i = X[k];
        int j = Y[k];
        for(int x = i - radiu; x <= i + radiu; x++)
        {
            for(int y = j - radiu; y <= j + radiu; y++)
            {
                if(x >= 0 && x < img.rows && y >= 0 && y < img.cols)
                {
                    image.at<cv::Vec3b>(x,y)[0] = 0;
                    image.at<cv::Vec3b>(x,y)[1] = 0;
                    image.at<cv::Vec3b>(x,y)[2] = 0;
                    //cout<<x<<" "<<y<< " ";
                }
            }
            //cout<<endl;
        }
        k++;
    }
    return image;
}

Mat draw::DrawSuperpixelEdge(int** lable, Mat img)
{
    int height = img.rows;
    int width = img.cols;
    Mat outimg = img.clone();
    int colorRGB[3] = {0, 0, 255};
    for(int i = 0; i < height; ++i)
    {
        for(int j = 0; j < width; ++j)
        {
            if((i - 1) >= 0 && (j - 1) >= 0)
            {
                if(lable[i - 1][j - 1] != lable[i][j] || lable[i][j - 1] != lable[i][j] || lable[i - 1][j] != lable[i][j])
                {
                    outimg.at<cv::Vec3b>(i,j)[0] = colorRGB[0];
                    outimg.at<cv::Vec3b>(i,j)[1] = colorRGB[1];
                    outimg.at<cv::Vec3b>(i,j)[2] = colorRGB[2];

                    outimg.at<cv::Vec3b>(i,j+1)[0] = colorRGB[0];
                    outimg.at<cv::Vec3b>(i,j+1)[1] = colorRGB[1];
                    outimg.at<cv::Vec3b>(i,j+1)[2] = colorRGB[2];
                }
            }
            else if((i - 1) >= 0 && (j + 1) < width)
            {
                if(lable[i - 1][j + 1] != lable[i][j] || lable[i][j + 1] != lable[i][j])
                {
                    outimg.at<cv::Vec3b>(i,j)[0] = colorRGB[0];
                    outimg.at<cv::Vec3b>(i,j)[1] = colorRGB[1];
                    outimg.at<cv::Vec3b>(i,j)[2] = colorRGB[2];

                    outimg.at<cv::Vec3b>(i,j+1)[0] = colorRGB[0];
                    outimg.at<cv::Vec3b>(i,j+1)[1] = colorRGB[1];
                    outimg.at<cv::Vec3b>(i,j+1)[2] = colorRGB[2];
                }
            }
            else if((i + 1) < height && (j - 1) >= 0)
            {
                if(lable[i + 1][j - 1] != lable[i][j] || lable[i + 1][j] != lable[i][j])
                {
                    outimg.at<cv::Vec3b>(i,j)[0] = colorRGB[0];
                    outimg.at<cv::Vec3b>(i,j)[1] = colorRGB[1];
                    outimg.at<cv::Vec3b>(i,j)[2] = colorRGB[2];

                    outimg.at<cv::Vec3b>(i,j+1)[0] = colorRGB[0];
                    outimg.at<cv::Vec3b>(i,j+1)[1] = colorRGB[1];
                    outimg.at<cv::Vec3b>(i,j+1)[2] = colorRGB[2];
                }
            }
            else if((i + 1) < height && (j + 1) < width)
            {
                if(lable[i + 1][j + 1] != lable[i][j])
                {
                    outimg.at<cv::Vec3b>(i,j)[0] = colorRGB[0];
                    outimg.at<cv::Vec3b>(i,j)[1] = colorRGB[1];
                    outimg.at<cv::Vec3b>(i,j)[2] = colorRGB[2];

                    outimg.at<cv::Vec3b>(i,j+1)[0] = colorRGB[0];
                    outimg.at<cv::Vec3b>(i,j+1)[1] = colorRGB[1];
                    outimg.at<cv::Vec3b>(i,j+1)[2] = colorRGB[2];
                }
            }
        }
    }
    return outimg;
}

void draw::OutputLabel(Mat image, int **label, string address)
{
    ofstream fout(address);
	if (!fout.is_open())
	{
		cout<<address<<" could not open "<<endl;
		return;
	}

    for(int i = 0; i < image.rows; ++i)
    {
        for(int j = 0; j < image.cols; ++j)
        {
            fout << label[i][j];
            fout << ',';
        }
        fout << endl;
    }
	

	fout.close();
}

Mat draw::DrawMeancolor(int **lable, double **meancolor, Mat img)
{
    Mat mean = img.clone();
    for(int i = 0; i < img.rows; ++i)
    {
        for(int j = 0; j < img.cols; ++j)
        {
            int k = lable[i][j];
            mean.at<cv::Vec3b>(i,j)[2] = meancolor[k][0];
            mean.at<cv::Vec3b>(i,j)[1] = meancolor[k][1];
            mean.at<cv::Vec3b>(i,j)[0] = meancolor[k][2];
        }
        
    }
    return mean;
}