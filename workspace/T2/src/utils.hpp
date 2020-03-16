#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <functional>
#include <cmath>
#include <vector>
#include <fstream>
#define CV_BGR2YCrCb COLOR_BGR2YCrCb
#define CV_YCrCb2BGR COLOR_YCrCb2BGR
#define CV_RGB2GRAY COLOR_RGB2GRAY
#define CV_BGR2HSV COLOR_BGR2HSV
#define CV_HSV2BGR COLOR_HSV2BGR
#define CV_WINDOW_AUTOSIZE WINDOW_AUTOSIZE
#define CV_TERMCRIT_ITER 1
#define CV_LOAD_IMAGE_COLOR IMREAD_COLOR
#define cvDestroyWindow destroyWindow
#define CV_RETR_TREE RETR_TREE
#define CV_CHAIN_APPROX_NONE CHAIN_APPROX_NONE


using namespace cv;
using namespace std;


int histSize = 256; //from 0 to 255
bool uniform = true;
bool accumulat = false;
bool accumulate_hist = false;

void checkImg(Mat img) {
	if(!img.data ) {                          // Check for invalid input
		cout <<  "Could not open or find the image" << std::endl ;
		exit(-1);
	}
}

Mat toBinaryOtsu(Mat &img, int maxval = 255, int type = THRESH_BINARY_INV) {
	Mat ret;
	cv::cvtColor(img, ret, CV_RGB2GRAY);
	ret.convertTo(ret, CV_8UC1);
	if (maxval <= 0) cv::max(img, maxval);

	cv::threshold(ret, ret, 0, maxval, type | THRESH_OTSU);

	return ret;
}

Mat toBinaryAdapt(Mat &img, int maxval, int type, int method, int blocksize, int C) {
	Mat ret;
	cv::cvtColor(img, ret, CV_RGB2GRAY);
	if (maxval <= 0) cv::max(img, maxval);

	cv::adaptiveThreshold(ret, ret, maxval, method, type, blocksize, C);

	return ret;
}

Mat drawableContours(std::vector<std::vector<Point>> &contours, cv::Size_<int> size) {
	Mat ret = Mat::zeros( size, CV_8UC3 );
	RNG rng(12345);

	for( int i = 0; i< (int)contours.size(); i++ ) {
		Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
		cv::drawContours( ret, contours, i, color);
	}
	return ret;
}

void calculateParameters(vector<vector<Point>> &contours){

	// sacado de aquí: https://docs.opencv.org/3.4/d0/d49/tutorial_moments.html
	vector<Moments> mu(contours.size() );
	for( size_t i = 0; i < contours.size(); i++ )
	{
		mu[i] = moments( contours[i] );
	}

	for( size_t i = 0; i < contours.size(); i++ )
	{
		cout << "CONTORNO " << i << endl;
		cout << "Area: "<<  std::fixed << std::setprecision(2) << mu[i].m00 << endl;
		cout << "Perímetro: " << arcLength( contours[i], true ) << endl;
		double hu[7];
		HuMoments(mu[i],hu);
		cout << "Momento 0: " << hu[0] << endl;
		cout << "Momento 1: " << hu[1] << endl;
		cout << "Momento 2: " << hu[2] << endl << endl;
	}
}

void aprender (String imagen, String objeto) {
	if (!objeto.compare("vagon") &&
		!objeto.compare("rectangulo") &&
		!objeto.compare("triangulo") &&
		!objeto.compare("circulo") &&
		!objeto.compare("rueda") ){
		cout << "Objeto no válido" << endl;
	}
	else{
		// calcular blob objeto
		Mat image;
		image = imread(imagen, CV_LOAD_IMAGE_COLOR);
		checkImg(image);
		imshow("Display window", image );
		waitKey(0);
		cvDestroyWindow("Display window");

		image = toBinaryOtsu(image);

		Mat draw_contours = Mat::zeros( image.size(), CV_8UC3 );
		std::vector<std::vector<Point>> contours;
		std::vector<Vec4i> hierarchy;

		int mode = CV_RETR_TREE;
		int method = CV_CHAIN_APPROX_NONE;

		cv::findContours(image, contours, hierarchy, mode, method);
		drawContours(draw_contours,contours,0, Scalar(255,50,50),-1,8,noArray(), 2, Point() );
		imshow("Contours", draw_contours);
		waitKey(0);

		// calcular media y varianza por el blob
		Scalar mean,dev;
		meanStdDev(draw_contours,mean,dev);
		cout << "media " << mean << " stddev " << dev << endl;

		//escribir en fichero
		ofstream salida;
		salida.open("files/objetos.txt", ios::out | ios::app );
		salida << objeto << " " << mean[0] << " " << mean[1] << " " << dev[0] << " " << dev[1] << endl;
		salida.close();

	}
}


void showHist(Mat &frame) {
	 //Historiogram

	        Mat channels[3];
	        split(frame,channels);
	        float range[] = { 0, (float)histSize } ; //the upper boundary is exclusive
	        const float* histRange = { range };

	        Mat b_hist, g_hist, r_hist;

	        /// Compute the histograms:
	        calcHist( &channels[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulat );
	        calcHist( &channels[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulat );
	        calcHist( &channels[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulat );

	        // Draw the histograms for R, G and B
	        int hist_w = 512; int hist_h = 400;
	        int bin_w = cvRound( (double) hist_w/histSize );

	        Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

	        if (accumulate_hist) {
	        	for( int i = 1; i < histSize; i++ )
				{
					b_hist.at<float>(i) = b_hist.at<float>(i-1) + b_hist.at<float>(i);
					g_hist.at<float>(i) = g_hist.at<float>(i-1) + g_hist.at<float>(i);
					r_hist.at<float>(i) = b_hist.at<float>(i-1) + r_hist.at<float>(i);
				}
			}

	        /// Normalize the result to [ 0, histImage.rows ]
	        normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
	        normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
	        normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

	        /// Draw for each channel
	        for( int i = 1; i < histSize; i++ )
	        {

	            line( histImage, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
	                             Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
	                             Scalar( 255, 0, 0), 2, 8, 0  );
	            line( histImage, Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ) ,
	                             Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
	                             Scalar( 0, 255, 0), 2, 8, 0  );
	            line( histImage, Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ) ,
	                             Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
	                             Scalar( 0, 0, 255), 2, 8, 0  );
	        }

	        namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE );
	        imshow("calcHist Demo", histImage );
}
