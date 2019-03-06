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
#include <numeric>
#include "../rapidjson/document.h"
#include "../rapidjson/writer.h"
#include "../rapidjson/stringbuffer.h"
#include "../rapidjson/filereadstream.h"
#include "../rapidjson/filewritestream.h"

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
int find_threshold(cv::Mat src, bool bground);
void process_chip(string img_name);

double readNumber(const rapidjson::Value& node, const char* key, double default_value) {
	if (node.HasMember(key) && node[key].IsDouble()) {
		return node[key].GetDouble();
	}
	else if (node.HasMember(key) && node[key].IsInt()) {
		return node[key].GetInt();
	}
	else {
		return default_value;
	}
}

std::vector<double> read1DArray(const rapidjson::Value& node, const char* key) {
	std::vector<double> array_values;
	if (node.HasMember(key)) {
		const rapidjson::Value& data = node[key];
		array_values.resize(data.Size());
		for (int i = 0; i < data.Size(); i++)
			array_values[i] = data[i].GetDouble();
		return array_values;
	}
	else {
		return array_values;
	}
}

bool readBoolValue(const rapidjson::Value& node, const char* key, bool default_value) {
	if (node.HasMember(key) && node[key].IsBool()) {
		return node[key].GetBool();
	}
	else {
		return default_value;
	}
}

std::string readStringValue(const rapidjson::Value& node, const char* key) {
	if (node.HasMember(key) && node[key].IsString()) {
		return node[key].GetString();
	}
	else {
		throw "Could not read string from node";
	}
}

void generate_score_file(std::string metafiles, std::string output_path);

void reshape_chips(std::string meta_data, double target_width, double target_height);

void find_avg_colors(std::string filename);

std::vector<double> compute_distribution_l1(cv::Mat img, int index);
std::vector<double> compute_distribution_l2(cv::Mat img);
cv::Mat draw_grids(cv::Mat img, int level);

std::pair<double, double> compute_stats(std::vector<double> v){
	double sum = std::accumulate(std::begin(v), std::end(v), 0.0);
	double m = sum / v.size();

	double accum = 0.0;
	std::for_each(std::begin(v), std::end(v), [&](const double d) {
		accum += (d - m) * (d - m);
	});

	double stdev = sqrt(accum / (v.size() - 1));

	return std::make_pair(m, stdev);
}

/// Global variables
Mat erosion_dst, dilation_dst;

int erosion_elem = 0;
int erosion_size = 0;
int dilation_elem = 1;
int dilation_size = 1;
int const max_elem = 2;
int const max_kernel_size = 21;

/** Function Headers */
void Erosion(int, void*);
void Dilation(int, void*);

/**
* @function main
*/
int main(int argc, char** argv)
{
	{
		/// Load an image
		src = imread("../data/1.png");

		if (!src.data)
		{
			return -1;
		}

		/// Create windows
		namedWindow("Erosion Demo", CV_WINDOW_AUTOSIZE);
		namedWindow("Dilation Demo", CV_WINDOW_AUTOSIZE);
		cvMoveWindow("Dilation Demo", src.cols, 0);

		/// Create Erosion Trackbar
		createTrackbar("Element:\n 0: Rect \n 1: Cross \n 2: Ellipse", "Erosion Demo",
			&erosion_elem, max_elem,
			Erosion);

		createTrackbar("Kernel size:\n 2n +1", "Erosion Demo",
			&erosion_size, max_kernel_size,
			Erosion);

		/// Create Dilation Trackbar
		createTrackbar("Element:\n 0: Rect \n 1: Cross \n 2: Ellipse", "Dilation Demo",
			&dilation_elem, max_elem,
			Dilation);

		createTrackbar("Kernel size:\n 2n +1", "Dilation Demo",
			&dilation_size, max_kernel_size,
			Dilation);

		/// Default start
		Erosion(0, 0);
		Dilation(1, 0);

		waitKey(0);
	}
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

/**  @function Erosion  */
void Erosion(int, void*)
{
	int erosion_type;
	if (erosion_elem == 0){ erosion_type = MORPH_RECT; }
	else if (erosion_elem == 1){ erosion_type = MORPH_CROSS; }
	else if (erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }

	Mat element = getStructuringElement(erosion_type,
		Size(2 * erosion_size + 1, 2 * erosion_size + 1),
		Point(erosion_size, erosion_size));

	/// Apply the erosion operation
	erode(src, erosion_dst, element);
	imshow("Erosion Demo", erosion_dst);
}

/** @function Dilation */
void Dilation(int, void*)
{
	int dilation_type;
	if (dilation_elem == 0){ dilation_type = MORPH_RECT; }
	else if (dilation_elem == 1){ dilation_type = MORPH_CROSS; }
	else if (dilation_elem == 2) { dilation_type = MORPH_ELLIPSE; }

	Mat element = getStructuringElement(dilation_type,
		Size(2 * dilation_size + 1, 2 * dilation_size + 1),
		Point(dilation_size, dilation_size));
	/// Apply the dilation operation
	dilate(src, dilation_dst, element);
	imshow("Dilation Demo", dilation_dst);
	imwrite("../data/dilation.png", dilation_dst);
}

void process_chip(string img_name){
	// load image
	cv::Mat src, dst_ehist, dst_classify;
	src = cv::imread(img_name, 1);
	cv::Mat hsv;
	cvtColor(src, hsv, cv::COLOR_BGR2HSV);
	std::vector<cv::Mat> bgr;   //destination array
	cv::split(hsv, bgr);//split source 
	for (int i = 0; i < 3; i++)
		cv::equalizeHist(bgr[i], bgr[i]);
	dst_ehist = bgr[2];
	// threshold classification
	int threshold = find_threshold(src, false);
	std::cout << "thrshold is " << threshold << std::endl;
	cv::threshold(dst_ehist, dst_classify, threshold, max_BINARY_value, cv::THRESH_BINARY);
	cv::imwrite("../data/dst_classify.png", dst_classify);
	// generate input image for DNN
	cv::Scalar bg_color(255, 255, 255); // white back ground
	cv::Scalar window_color(0, 0, 0); // black for windows
	cv::Mat scale_img;
	cv::resize(dst_classify, scale_img, cv::Size(224, 224));
	// correct the color
	for (int i = 0; i < scale_img.size().height; i++){
		for (int j = 0; j < scale_img.size().width; j++){
			//noise
			if ((int)scale_img.at<uchar>(i, j) < 255){
				scale_img.at<uchar>(i, j) = 0;
			}
		}
	}
	// add padding
	int padding_size = 10;
	int borderType = cv::BORDER_CONSTANT;
	cv::Scalar value(255, 255, 255);
	cv::Mat grid_dst;
	cv::copyMakeBorder(scale_img, scale_img, padding_size, padding_size, padding_size, padding_size, borderType, value);
	cv::imwrite("../data/scale_img_padding.png", scale_img);
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	/// Detect edges using canny
	//cv::Canny(scale_img, canny_output, 0, 0, 5);
	cv::findContours(scale_img, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	cv::Mat drawing(scale_img.size(), CV_8UC3, bg_color);
	RNG rng(12345);
	for (int i = 0; i< contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 0), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(drawing, contours, i, color, 1, 8, hierarchy, 0, Point());
	}
	std::vector<cv::Rect> boundRect(contours.size());
	std::vector<cv::RotatedRect> minRect(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		boundRect[i] = cv::boundingRect(cv::Mat(contours[i]));
		minRect[i] = minAreaRect(cv::Mat(contours[i]));
	}
	cv::Mat dnn_img(scale_img.size(), CV_8UC3, bg_color);
	for (int i = 1; i< contours.size(); i++)
	{
		if (hierarchy[i][3] != 0) continue;
		//cv::rectangle(dnn_img, cv::Point(boundRect[i].tl().x + 1, boundRect[i].tl().y + 1), cv::Point(boundRect[i].br().x - 1, boundRect[i].br().y - 1), window_color, -1);
		//cv::rectangle(dnn_img, cv::Point(boundRect[i].tl().x + 1, boundRect[i].tl().y + 1), cv::Point(boundRect[i].br().x, boundRect[i].br().y), window_color, -1);
		cv::Point2f rect_points[4]; 
		minRect[i].points(rect_points);
		cv::Point vertices[4];
		for (int i = 0; i < 4; ++i){
			vertices[i] = rect_points[i];
		}
		cv::fillConvexPoly(dnn_img,
			vertices,
			4,
			window_color);
	}
	// remove padding
	dnn_img = dnn_img(cv::Rect(padding_size, padding_size, 224, 224));
	cv::imwrite("../data/dnn_input.png", dnn_img);
	cv::imwrite("../data/drawing.png", drawing);
}

cv::Mat draw_grids(cv::Mat img, int level){
	int thickness = 1;
	int lineType = 8;
	int segs = (int)pow(2, level);
	for (int i = 1; i < (int)pow(2, level); i++){
		cv::Point l1_start(0, i * img.size().height / segs);
		cv::Point l1_end(img.size().width, i * img.size().height / segs);
		cv::line(img,
			l1_start,
			l1_end,
			cv::Scalar(0, 0, 255),
			thickness,
			lineType);
	}
	for (int i = 1; i < (int)pow(2, level); i++){
		cv::Point l1_start(i * img.size().width / segs, 0);
		cv::Point l1_end(i * img.size().width / segs, img.size().height);
		cv::line(img,
			l1_start,
			l1_end,
			cv::Scalar(0, 0, 255),
			thickness,
			lineType);
	}
	return img;
}

std::vector<double> compute_distribution_l1(cv::Mat img, int index){
	// divide whole img to 2 by 2 grid
	std::vector<double> result;
	int x_start = 0;
	int y_start = 0;
	int grid_width = img.size().width / 2;
	int grid_height = img.size().height / 2;
	RNG rng(12345);
	for (int i = 0; i < 2; i++){
		x_start += i * grid_width;
		y_start = 0;
		for (int j = 0; j < 2; j++){
			y_start += j * grid_height;
			cv::Mat grid = img(cv::Rect(x_start, y_start, grid_width, grid_height)).clone();
			int top = 5;
			int bottom = 5;
			int left = 5;
			int right = 5;
			int borderType = cv::BORDER_CONSTANT;
			cv::Scalar value(255, 255, 255);
			cv::Mat grid_dst;
			cv::copyMakeBorder(grid, grid_dst, top, bottom, left, right, borderType, value);
			cv::Mat grid_gray;
			cvtColor(grid_dst, grid_gray, cv::COLOR_BGR2GRAY);
			//cv::imwrite("../data/grid_" + std::to_string(index * 4 + i * 2 + j) + ".png", grid_gray);
			std::vector<std::vector<cv::Point> > contours;
			std::vector<cv::Vec4i> hierarchy;
			cv::findContours(grid_gray, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
			cv::Mat drawing = Mat::zeros(grid_gray.size(), CV_8UC3);
			/*for (int i = 0; i< contours.size(); i++)
			{
				Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
				drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
			}
			cv::imwrite("../data/grid_contour_" + std::to_string(index * 4 + i * 2 + j) + ".png", drawing);*/
			result.push_back(contours.size() - 1);
		}
	}
	return result;
}

std::vector<double> compute_distribution_l2(cv::Mat img){
	cv::Mat output(img.size(), CV_8UC3, cv::Scalar(255, 255, 255));
	std::vector<double> result;
	// divide whole img to 2 by 2 grid
	int x_start = 0;
	int y_start = 0;
	int grid_width = img.size().width / 2;
	int grid_height = img.size().height / 2;
	for (int i = 0; i < 2; i++){
		x_start += i * grid_width;
		y_start = 0;
		for (int j = 0; j < 2; j++){
			y_start += j * grid_height;
			cv::Mat grid = img(cv::Rect(x_start, y_start, grid_width, grid_height)).clone();
			int top = 5;
			int bottom = 5;
			int left = 5;
			int right = 5;
			int borderType = cv::BORDER_CONSTANT;
			cv::Scalar value(255, 255, 255);
			cv::Mat grid_dst;
			cv::copyMakeBorder(grid, grid_dst, top, bottom, left, right, borderType, value);
			std::vector<double> tmp = compute_distribution_l1(grid_dst, i * 2 + j);
			result.insert(result.end(), tmp.begin(), tmp.end());
		}
	}
	return result;
}

void find_avg_colors(std::string src_filename, std::string classify_filename, std::string dst_filename, std::string output_filename){
	cv::Mat dst_classify = cv::imread(classify_filename, CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat src = cv::imread(src_filename, CV_LOAD_IMAGE_COLOR);
	cv::Scalar bg_avg_color(0, 0, 0);
	cv::Scalar win_avg_color(0, 0, 0);
	int bg_count = 0;
	int win_count = 0;
	for (int i = 0; i < dst_classify.size().height; i++){
		for (int j = 0; j < dst_classify.size().width; j++){
			if ((int)dst_classify.at<uchar>(i, j) == 0){
				win_avg_color.val[0] += src.at<cv::Vec3b>(i, j)[0];
				win_avg_color.val[1] += src.at<cv::Vec3b>(i, j)[1];
				win_avg_color.val[2] += src.at<cv::Vec3b>(i, j)[2];
				win_count++;
			}
			else{
				bg_avg_color.val[0] += src.at<cv::Vec3b>(i, j)[0];
				bg_avg_color.val[1] += src.at<cv::Vec3b>(i, j)[1];
				bg_avg_color.val[2] += src.at<cv::Vec3b>(i, j)[2];
				bg_count++;
			}
		}
	}
	win_avg_color.val[0] = win_avg_color.val[0] / win_count;
	win_avg_color.val[1] = win_avg_color.val[1] / win_count;
	win_avg_color.val[2] = win_avg_color.val[2] / win_count;

	bg_avg_color.val[0] = bg_avg_color.val[0] / bg_count;
	bg_avg_color.val[1] = bg_avg_color.val[1] / bg_count;
	bg_avg_color.val[2] = bg_avg_color.val[2] / bg_count;

	std::cout << "win_avg_color is " << win_avg_color << std::endl;
	std::cout << "bg_avg_color is " << bg_avg_color << std::endl;
	// generate a test image
	cv::Mat result(dst_classify.size().height, dst_classify.size().width, CV_8UC3);
	for (int i = 0; i < dst_classify.size().height; i++){
		for (int j = 0; j < dst_classify.size().width; j++){
			if ((int)dst_classify.at<uchar>(i, j) == 0){
				result.at<cv::Vec3b>(i, j)[0] = win_avg_color.val[0];
				result.at<cv::Vec3b>(i, j)[1] = win_avg_color.val[1];
				result.at<cv::Vec3b>(i, j)[2] = win_avg_color.val[2];
			}
			else{
				result.at<cv::Vec3b>(i, j)[0] = bg_avg_color.val[0];
				result.at<cv::Vec3b>(i, j)[1] = bg_avg_color.val[1];
				result.at<cv::Vec3b>(i, j)[2] = bg_avg_color.val[2];
			}
		}
	}
	cv::imwrite("../data/result.png", result);
}

void reshape_chips(std::string meta_data, double target_width, double target_height){
	// read image json file
	FILE* fp = fopen(meta_data.c_str(), "rb"); // non-Windows use "r"
	char readBuffer[1024];
	rapidjson::FileReadStream is(fp, readBuffer, sizeof(readBuffer));
	rapidjson::Document doc;
	doc.ParseStream(is);
	// score
	double score = readNumber(doc, "score", 0.2);
	std::cout << "score is " << score << std::endl;
	// size of chip
	std::vector<double> facChip_size = read1DArray(doc, "size");
	std::cout << "facChip_size is " << facChip_size[0] << ", " << facChip_size[1] << std::endl;
	// image name
	std::string img_fileName = readStringValue(doc, "imagename");
	std::size_t found = img_fileName.find_first_of("/");
	img_fileName = "../data/" + img_fileName.substr(found + 1);
	std::cout << "img_fileName is " << img_fileName << std::endl;

	int type = 0;
	if (facChip_size[0] < 30.0 && facChip_size[1] < 30.0 && score > 0.8)
		type = 1;
	else if (facChip_size[0] > 30.0 && facChip_size[1] < 30.0 && score > 0.7)
		type = 2;
	else if (facChip_size[0] < 30.0 && facChip_size[1] > 30.0 && score > 0.7)
		type = 3;
	else if (facChip_size[0] > 30.0 && facChip_size[1] > 30.0 && score > 0.3)
		type = 4;
	else {
		// do nothing
	}
	double ratio_width, ratio_height;
	if (type == 1){
		src = imread(img_fileName);
		ratio_width = target_width / facChip_size[0] - 1;
		ratio_height = target_height / facChip_size[1] - 1;
		std::cout << "ratio_width is " << ratio_width << std::endl;
		std::cout << "ratio_height is " << ratio_height << std::endl;
		int top = (int)(ratio_height * src.rows);
		int bottom = 0;
		int left = 0; 
		int right = (int)(ratio_width * src.cols);
		int borderType = BORDER_REFLECT_101;
		cv::Scalar value(0, 0, 0);
		cv::copyMakeBorder(src, dst, top, bottom, left, right, borderType, value);
		cv::imwrite("../data/test.png", dst);
	}
	else if (type == 2){
		src = imread(img_fileName);
		int times = ceil(facChip_size[0] / target_width);
		ratio_width = (times * target_width - facChip_size[0]) / facChip_size[0];
		ratio_height = target_height / facChip_size[1] - 1;
		std::cout << "ratio_width is " << ratio_width << std::endl;
		std::cout << "ratio_height is " << ratio_height << std::endl;
		int top = (int)(ratio_height * src.rows);
		int bottom = 0;
		int left = 0;
		int right = (int)(ratio_width * src.cols);
		int borderType = BORDER_REFLECT_101;
		cv::Scalar value(0, 0, 0);
		cv::copyMakeBorder(src, dst, top, bottom, left, right, borderType, value);
		cv::imwrite("../data/test.png", dst);
		// crop 30 * 30
		cv::Mat croppedImage = dst(cv::Rect(dst.size().width * 0.1, 0, dst.size().width / times, dst.size().height));
		cv::imwrite("../data/test_cropped.png", croppedImage);
	}
	else if (type == 3){
		src = imread(img_fileName);
		int times = ceil(facChip_size[1] / target_height);
		ratio_height = (times * target_height - facChip_size[1]) / facChip_size[1];
		ratio_width = target_width / facChip_size[0] - 1;
		std::cout << "ratio_width is " << ratio_width << std::endl;
		std::cout << "ratio_height is " << ratio_height << std::endl;
		int top = (int)(ratio_height * src.rows);
		int bottom = 0;
		int left = 0;
		int right = (int)(ratio_width * src.cols);
		int borderType = BORDER_REFLECT_101;
		cv::Scalar value(0, 0, 0);
		cv::copyMakeBorder(src, dst, top, bottom, left, right, borderType, value);
		cv::imwrite("../data/test.png", dst);
		// crop 30 * 30
		cv::Mat croppedImage = dst(cv::Rect(0, ratio_height * src.rows, dst.size().width, dst.size().height / times));
		cv::imwrite("../data/test_cropped.png", croppedImage);
	}
	else if (type == 4){
		src = imread(img_fileName);
		int times_width = ceil(facChip_size[0] / target_width);
		int times_height = ceil(facChip_size[1] / target_height);
		ratio_width = (times_width * target_width - facChip_size[0]) / facChip_size[0];
		ratio_height = (times_height * target_height - facChip_size[1]) / facChip_size[1];
		std::cout << "ratio_width is " << ratio_width << std::endl;
		std::cout << "ratio_height is " << ratio_height << std::endl;
		int top = (int)(ratio_height * src.rows);
		int bottom = 0;
		int left = 0;
		int right = (int)(ratio_width * src.cols);
		int borderType = BORDER_REFLECT_101;
		cv::Scalar value(0, 0, 0);
		cv::copyMakeBorder(src, dst, top, bottom, left, right, borderType, value);
		cv::imwrite("../data/test.png", dst);
		// crop 30 * 30
		for (int slice = 0; slice < times_height; slice++){
			cv::Mat croppedImage = dst(cv::Rect(dst.size().width * 0.1, ratio_height * src.rows, dst.size().width / times_width, dst.size().height / times_height));
			cv::imwrite("../data/test_cropped.png", croppedImage);
		}
	}
	else{
		// do nothing
	}
}

void generate_score_file(std::string meta_folder, std::string output_path){
	std::vector<string> metafiles = get_all_files_names_within_folder(meta_folder);
	std::ofstream out_param(output_path + "/parameters.txt");
	for (int i = 0; i < metafiles.size(); i++){
		std::string metafileName = meta_folder + "/" + metafiles[i];
		// read score
		// read image json file
		FILE* fp = fopen(metafileName.c_str(), "rb"); // non-Windows use "r"
		char readBuffer[1024];
		rapidjson::FileReadStream is(fp, readBuffer, sizeof(readBuffer));
		rapidjson::Document doc;
		doc.ParseStream(is);
		// score
		double score = readNumber(doc, "score", 0.2);
		// size of chip
		std::vector<double> facChip_size = read1DArray(doc, "size");
		// image of the chip
		std::string img_fileName = readStringValue(doc, "imagename");
		std::size_t found = img_fileName.find_first_of("/");
		// write to parameters.txt
		{
			// normalize for NN training
			out_param << metafiles[i];
			out_param << ",";
			out_param << img_fileName.substr(found + 1);
			out_param << ",";
			out_param << score;
			out_param << ",";
			out_param << facChip_size[0];
			out_param << ",";
			out_param << facChip_size[1];
			out_param << "\n";
		}
		fclose(fp);
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

int find_threshold(cv::Mat src, bool bground) {
	//Convert pixel values to other color spaces.
	cv::Mat hsv;
	cvtColor(src, hsv, cv::COLOR_BGR2HSV);
	std::vector<cv::Mat> bgr;   //destination array
	cv::split(hsv, bgr);//split source 
	for (int i = 0; i < 3; i++)
		cv::equalizeHist(bgr[i], bgr[i]);
	/// Load an image
	cv::Mat src_gray = bgr[2];
	for (int threshold = 40; threshold < 160; threshold += 5) {
		cv::Mat dst;
		cv::threshold(src_gray, dst, threshold, max_BINARY_value, cv::THRESH_BINARY);
		int count = 0;
		for (int i = 0; i < dst.size().height; i++) {
			for (int j = 0; j < dst.size().width; j++) {
				//noise
				if ((int)dst.at<uchar>(i, j) == 0) {
					count++;
				}
			}
		}
		float percentage = count * 1.0 / (dst.size().height * dst.size().width);
		std::cout << "percentage is " << percentage << std::endl;
		if (percentage > 0.25 && !bground)
			return threshold;
		if (percentage > 0.30 && bground)
			return threshold;
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