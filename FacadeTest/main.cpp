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
int threshold_type = 0;
int const max_value = 255;
int const max_type = 4;
int const max_BINARY_value = 255;

cv::Mat src, src_gray, dst;
char* window_name = "Threshold Demo";

char* trackbar_type = "Type:";
char* trackbar_value = "Value";

/// Function headers
void Threshold_Demo(int, void*);
void Threshold_Demo(int threshold_type, int threshold, string output);
/// Function header
std::vector<string> get_all_files_names_within_folder(string folder);
float compute_win_percentage(string img_name);
void find_threshold(string img_name);
/**
* @function main
*/
int main(int argc, char** argv)
{
	find_threshold("../data/2.png");
	return 0;
	// hist equalized
	cv::Mat src, dst;
	std::cout << "argv[1] is " << argv[1] << std::endl;
	// get all files from the folder
	/// Function header
	std::vector<string> names = get_all_files_names_within_folder(argv[1]);
	for (int i = 0; i < names.size(); i++){
		src = cv::imread(argv[1] + names[i], CV_LOAD_IMAGE_COLOR);
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
				char buffer[10];
				std::cout << "threshold_value is " << threshold_value << std::endl;
				//std::strcat(argv[2], _itoa(threshold_value, buffer, 10));
				//std::strcat(argv[2], ".png");
				//std::cout << "argv[2] is " << threshold_value << std::endl;
				Threshold_Demo(THRESH_BINARY, threshold_value, argv[2] + std::to_string(i) + "_" + std::to_string(threshold_value) + ".png");
				break;
			}
		}
	}
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

void Threshold_Demo(int threshold_type, int threshold, string output){

	cv::threshold(src_gray, dst, threshold, max_BINARY_value, threshold_type);
	imshow(window_name, dst);
	cv::imwrite(output, dst);
}

void find_threshold(string img_name){
	// hist equalized
	cv::Mat src, src_gray;
	src = cv::imread(img_name, CV_LOAD_IMAGE_COLOR);
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
	std::vector<int> thresholds;
	std::vector<float> results;
	for (int threshold = 40; threshold < 160; threshold += 5){
		std::cout << "Try threshold: " << threshold << std::endl;
		thresholds.push_back(threshold);
		cv::Mat dst;
		cv::threshold(src_gray, dst, threshold, max_BINARY_value, threshold_type);
		cv::imwrite("../data/result/" + std::to_string(threshold) + ".png", dst);
		cv::Mat tmp = cv::imread("../data/result/" + std::to_string(threshold) + ".png", CV_LOAD_IMAGE_COLOR);
		int count = 0;
		for (int i = 0; i < tmp.size().height; i++){
			for (int j = 0; j < tmp.size().width; j++){
				//noise
				if (tmp.at<cv::Vec3b>(i, j)[0] == 0 && tmp.at<cv::Vec3b>(i, j)[1] == 0 && tmp.at<cv::Vec3b>(i, j)[2] == 0){
					count++;
				}
			}
		}
		float percentage = count * 1.0 / (tmp.size().height * tmp.size().width);
		std::cout << "-----percentage: " << percentage << std::endl;
		results.push_back(percentage);
	}
	for (int i = 0; i < thresholds.size(); i++){
		std::cout << thresholds[i] << ", " << results[i] << std::endl;
	}
	
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

float compute_win_percentage(string img_name){
	src = cv::imread(img_name, CV_LOAD_IMAGE_COLOR);
	int count = 0;
	for (int i = 0; i < src.size().height; i++){
		for (int j = 0; j < src.size().width; j++){
			//noise
			if (src.at<cv::Vec3b>(i, j)[0] == 0 && src.at<cv::Vec3b>(i, j)[1] == 0 && src.at<cv::Vec3b>(i, j)[2] == 0){
				count++;
			}
		}
	}
	return count * 1.0 / (src.size().height * src.size().width);
}

std::vector<string> get_all_files_names_within_folder(string folder)
{
	vector<string> names;
	string search_path = folder + "/*.*";
	WIN32_FIND_DATA fd;
	HANDLE hFind = ::FindFirstFile(search_path.c_str(), &fd);
	if (hFind != INVALID_HANDLE_VALUE) {
		do {
			// read all (real) files in current folder
			// , delete '!' read other 2 default folder . and ..
			if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
				names.push_back(fd.cFileName);
			}
		} while (::FindNextFile(hFind, &fd));
		::FindClose(hFind);
	}
	return names;
}