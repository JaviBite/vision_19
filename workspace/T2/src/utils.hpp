#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <functional>
#include <cmath>
#include <vector>
#include <fstream>
#include <algorithm>
#include <numeric>

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

const String FILE_ITEMS = "files/objetos.txt";
static const int CONT_MODE = CV_RETR_EXTERNAL;
static const int CONT_METH = CV_CHAIN_APPROX_SIMPLE;


int histSize = 256; //from 0 to 255
bool uniform = true;
bool accumulat = false;
bool accumulate_hist = false;

struct Fig {
	std::string nombre;
	float mean_area;
	float mean_perim;
	float mean_m0;
	float mean_m1;
	float mean_m2;

	float std_area;
	float std_perim;
	float std_m0;
	float std_m1;
	float std_m2;
};

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
		cv::drawContours( ret, contours, i, color,-1,8,noArray(), 2, Point() );
	}
	return ret;
}

vector<vector<float>> calculateParameters(vector<vector<Point>> &contours){

	// sacado de aquí: https://docs.opencv.org/3.4/d0/d49/tutorial_moments.html
	vector<Moments> mu(contours.size() );

	vector<vector<float>> ret(contours.size());
	for( size_t i = 0; i < contours.size(); i++ )
	{
		mu[i] = moments( contours[i] );
	}

	for( size_t i = 0; i < contours.size(); i++ )
	{
		double hu[7];
		HuMoments(mu[i],hu);

		ret[i].resize(5);
		ret[i][0] = mu[i].m00;
		ret[i][1] = arcLength( contours[i], true );
		ret[i][2] = hu[0];
		ret[i][3] = hu[1];
		ret[i][4] = hu[2];
	}

	return ret;
}

// comparison function object
bool compareContourAreas ( std::vector<cv::Point> contour1, std::vector<cv::Point> contour2 ) {
    double i = fabs( contourArea(cv::Mat(contour1)) );
    double j = fabs( contourArea(cv::Mat(contour2)) );
    return ( i > j );
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

		std::vector<std::vector<Point>> contours;
		std::vector<Vec4i> hierarchy;

		int mode = CONT_MODE;
		int method = CONT_METH;

		cv::findContours(image, contours, hierarchy, mode, method);
//		Mat drawCont = drawableContours(contours, image.size());
//		imshow("Contours", drawCont);
//		waitKey(0);
		if (contours.size() > 1) std::sort(contours.begin(), contours.end(), compareContourAreas);

		// calcular parametros
		vector<vector<float>> params = calculateParameters(contours);

		//escribir en fichero
		ofstream salida;
		salida.open(FILE_ITEMS, ios::out | ios::app );
		salida << objeto << " " << params[0][0] << " " << params[0][1] << " "
								<< params[0][2] << " " << params[0][3] << " "
								<< params[0][4] << endl;
		salida.close();

	}
}

//double mahalanobis(Mat x, Fig f) {
//	/* Calcula los descriptores de imagen */
//	double perimeter = arcLength(x, true);
//
//	Moments m = moments(x, true);
//	double inv[7];
//	HuMoments(m, inv);
//	double area = m.m00;
//	double inv1 = inv[1];
//	double inv2 = inv[2];
//
//	double d = pow((area - f.area_media), 2) / ((double) f.area_varianza);
//	d += pow((perimeter - f.perimetro_media), 2)
//			/ ((double) f.perimetro_varianza);
//	d += pow((inv1 - f.m1_media), 2) / ((double) f.m1_varianza);
//	d += pow((inv2 - f.m2_media), 2) / ((double) f.m2_varianza);
//
//	return d;
//}

void reconocer (String imagen) {
	ifstream input;
	input.open(FILE_ITEMS, ios::in | ios::app );
	vector<vector<float>> circulo, vagon, rectangulo, triangulo, rueda;

	std::string nombre;
	vector<float> data(5);
	input >> nombre >> data[0] >> data[1] >> data[2] >> data[3] >> data[4];
		if(nombre == "vagon")
			vagon.push_back(data);
		else if(nombre == "circulo")
			circulo.push_back(data);
		else if(nombre == "triangulo")
			triangulo.push_back(data);
		else if(nombre == "rueda")
			rueda.push_back(data);
		else if(nombre == "rectangulo")
			rectangulo.push_back(data);

	while(!input.eof()) {
		std::string nombre;
		vector<float> data(5);
		input >> nombre >> data[0] >> data[1] >> data[2] >> data[3] >> data[4];
			if(nombre == "vagon")
				vagon.push_back(data);
			else if(nombre == "circulo")
				circulo.push_back(data);
			else if(nombre == "triangulo")
				triangulo.push_back(data);
			else if(nombre == "rueda")
				rueda.push_back(data);
			else if(nombre == "rectangulo")
				rectangulo.push_back(data);
	}
	std::cout << "rectangulo" << endl;;
	int N = rectangulo.size();
	vector<float> area, perim;
	for (vector<float> v : rectangulo) {
		std::cout << "AREA  = " << v[0] << std::endl;
		std::cout << "PERIM = " << v[1] << std::endl;
		std::cout << "M0    = " << v[2] << std::endl;
		std::cout << "M1    = " << v[3] << std::endl;
		std::cout << "M2    = " << v[4] << std::endl;
		std::cout << "----" << endl;
		area.push_back(v[0]);
	}

	double sum = std::accumulate(area.begin(), area.end(), 0.0);
	double mean = sum / area.size();

	double sq_sum = std::inner_product(area.begin(), area.end(), area.begin(), 0.0);
	double stdev = std::sqrt(sq_sum / area.size() - mean * mean);

	std::cout << "MEAN AREA    = " << mean << std::endl;
	std::cout << "STD AREAM    = " << stdev << std::endl;
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
