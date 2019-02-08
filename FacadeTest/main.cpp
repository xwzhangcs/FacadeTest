#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <Windows.h>
#include <fstream>
#include <sstream>
#include "gdal_priv.h"
#include "cpl_conv.h" // for CPLMalloc()
#include "ogrsf_frmts.h"
#include <time.h> 

using namespace cv;
using namespace std;


/// Global variables

int threshold_value = 0;
int threshold_type = 0;;
int const max_value = 255;
int const max_type = 4;
int const max_BINARY_value = 255;

cv::Mat src, src_gray, dst;
char* window_name = "Threshold Demo";

char* trackbar_type = "Type:";
char* trackbar_value = "Value";

/// Function headers
void Threshold_Demo(int, void*);
void Threshold_Demo(int threshold_type, int threshold, char* output);
/**
* @function main
*/
int main(int argc, char** argv)
{
	// hist equalized
	cv::Mat src, dst;
	std::cout << "argv[1] is " << argv[1] << std::endl;
	src = cv::imread(argv[1], 1);
	//Convert pixel values to other color spaces.
	cv::Mat hsv;
	cvtColor(src, hsv, COLOR_BGR2HSV);
	std::vector<cv::Mat> bgr;   //destination array
	cv::split(hsv, bgr);//split source 
	for (int i = 0; i < 3; i++)
		cv::equalizeHist(bgr[i], bgr[i]);
	//cv::merge(bgr, dst);	
	/// Load an image
	src_gray = bgr[2];

	/// Create a window to display results
	namedWindow(window_name, CV_WINDOW_AUTOSIZE);

	/// Create Trackbar to choose type of Threshold
	createTrackbar(trackbar_type,
		window_name, &threshold_type,
		max_type, Threshold_Demo);

	createTrackbar(trackbar_value,
		window_name, &threshold_value,
		max_value, Threshold_Demo);
	Threshold_Demo(0, 0);
	/// Wait until user finishes program
	while (true)
	{
		int c;
		c = waitKey(20);
		if ((char)c == 27)
		{
			break;
		}
	}

	//Threshold_Demo(THRESH_BINARY, 32, "../data/output_8.png");

}


/**
* @function Threshold_Demo
*/
void Threshold_Demo(int, void*)
{
	/* 0: Binary
	1: Binary Inverted
	2: Threshold Truncated
	3: Threshold to Zero
	4: Threshold to Zero Inverted
	*/

	threshold(src_gray, dst, threshold_value, max_BINARY_value, threshold_type);

	imshow(window_name, dst);
}

void Threshold_Demo(int threshold_type, int threshold, char* output){

	cv::threshold(src_gray, dst, threshold, max_BINARY_value, threshold_type);
	imshow(window_name, dst);
	cv::imwrite(output, dst);
}

void correct(cv::Mat &img){
	for (int i = 0; i < img.size().height; i++){
		for (int j = 0; j < img.size().width; j++){
			//noise
			if (img.at<cv::Vec3b>(i, j)[0] > 200){
				img.at<cv::Vec3b>(i, j)[0] = 255;
				img.at<cv::Vec3b>(i, j)[1] = 0;
				img.at<cv::Vec3b>(i, j)[2] = 0;
			}
			if (img.at<cv::Vec3b>(i, j)[1] > 200){
				img.at<cv::Vec3b>(i, j)[0] = 0;
				img.at<cv::Vec3b>(i, j)[1] = 255;
				img.at<cv::Vec3b>(i, j)[2] = 0;
			}
			if (img.at<cv::Vec3b>(i, j)[2] > 200){
				img.at<cv::Vec3b>(i, j)[0] = 0;
				img.at<cv::Vec3b>(i, j)[1] = 0;
				img.at<cv::Vec3b>(i, j)[2] = 255;
			}
		}
	}
}
