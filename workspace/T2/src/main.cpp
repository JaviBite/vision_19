#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <windows.h> // For Sleep
#include "utils.hpp"
#include <fstream>
#include <string>
#include <vector>


using namespace cv;
using namespace std;


int main(int, char**) {

	Mat image;
	image = imread("files/imagenesT2/reco1.pgm", CV_LOAD_IMAGE_COLOR);   // Read the file
	checkImg(image);

	imshow("Display window", image );                   // Show our image inside it.
	waitKey(0);
	cvDestroyWindow("Display window");

	//Point 1: Get Binary Images

	Mat bin_otsu, bin_adap;

	double maxval = 255;
	cv::max(image, maxval);
	int type = THRESH_BINARY_INV;

	bin_otsu = toBinaryOtsu(image, maxval, type);

	int adap_meth = ADAPTIVE_THRESH_MEAN_C;
	int blocksize = 11;
	int C = 2;
	bin_adap = toBinaryAdapt(image,  maxval, type, adap_meth, blocksize, C);

	imshow("OTSU", bin_otsu );                   // Show our image inside it.
	imshow("ADAPTIVE", bin_adap );
	waitKey(0);
	cvDestroyWindow("ADAPTIVE");
	cvDestroyWindow("OTSU");

	//Point 2: Contours

	image = imread("files/imagenesT2/circulo1.pgm", CV_LOAD_IMAGE_COLOR);   // Read the file
	checkImg(image);

	imshow("Display window", image );                   // Show our image inside it.
	waitKey(0);

	image = toBinaryOtsu(image);

	Mat draw_contours = Mat::zeros( image.size(), CV_8UC3 );
	std::vector<std::vector<Point>> contours;
	std::vector<Vec4i> hierarchy;

	int mode = CV_RETR_TREE;
	int method = CV_CHAIN_APPROX_NONE;

	cv::findContours(image, contours, hierarchy, mode, method);
	//cv::drawContours(draw_contours, contours, -1, cv::COLORMAP_JET);
//	for( int i = 0; i< contours.size(); i++ ) {
//		Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
//		cv::drawContours( draw_contours, contours, i, color, 2, 8, hierarchy, 0, Point() );
//	}

	draw_contours = drawableContours(contours, image.size());

	imshow("Contours", draw_contours);                   // Show our image inside it.
	waitKey(0);
	cvDestroyWindow("Contours");


    return 0;
}
