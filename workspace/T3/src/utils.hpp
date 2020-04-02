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
#define CV_BGR2GRAY COLOR_BGR2GRAY
#define CV_GRAY2BGR COLOR_GRAY2BGR
#define CV_BGR2HSV COLOR_BGR2HSV
#define CV_HSV2BGR COLOR_HSV2BGR
#define CV_WINDOW_AUTOSIZE WINDOW_AUTOSIZE
#define CV_TERMCRIT_ITER 1
#define CV_LOAD_IMAGE_COLOR IMREAD_COLOR
#define CV_LOAD_IMAGE_GRAYSCALE IMREAD_GRAYSCALE
#define cvDestroyWindow destroyWindow
#define CV_RETR_TREE RETR_TREE
#define CV_CHAIN_APPROX_NONE CHAIN_APPROX_NONE

using namespace cv;
using namespace std;

int gauss_k = 1;
int gauss_s = 5;


void checkImg(Mat img) {
	if(!img.data ) {                          // Check for invalid input
		cout <<  "Could not open or find the image" << std::endl ;
		exit(-1);
	}
}

Mat modulo(Mat& img, bool draw) {
	Mat img2;
	GaussianBlur(img, img2, Size2i(gauss_s,gauss_s), gauss_k, gauss_k, BORDER_DEFAULT);
	// calcular gradiente horizontal
	Mat sobelx, sobely;
	Sobel(img2, sobelx,CV_32F, 1, 0);

	// calcular gradiente vertical
	Sobel(img2, sobely,CV_32F, 0, 1);

	// calcular el módulo
	Mat moduloaux = sobelx;
	for (int i = 0; i< sobelx.rows; i++)
		for (int j = 0; j< sobelx.cols; j++)
			moduloaux.at<float>(i,j) = sqrt(pow(sobelx.at<float>(i,j),2) + pow(sobely.at<float>(i,j),2));
	if (draw) {
		double minVal, maxVal;
		minMaxLoc(moduloaux, &minVal, &maxVal);
		Mat modulo;
		moduloaux.convertTo(modulo, CV_8U, 255.0/(maxVal - minVal), -minVal * 255.0/(maxVal - minVal));
		return modulo;
	} else {
		return moduloaux;
	}

}

void splitLines(vector<Vec2f> lines, vector<Vec2f> &lines1, vector<Vec2f> &lines2, float thetaThresh) {
	lines1.clear();
	lines2.clear();
	for( Vec2f line : lines ) {
		float theta = line[1];

		if (theta > thetaThresh && abs(theta - CV_PI) > thetaThresh && 		// Vertical
			(abs(theta - CV_PI/2)) > thetaThresh) { 	// Total Horizontal

			//std::cout << "heta " << theta << std::endl;
			if (theta > CV_PI/2)
				lines1.push_back(line);
			else if (theta < CV_PI/2)
				lines2.push_back(line);
		}
	}
}

Mat drawLines(Mat &draw, vector<Vec2f> lines, Scalar color) {
	for( size_t i = 0; i < lines.size(); i++ ){
		float rho = lines[i][0];
		float theta = lines[i][1];
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		Point pt1(cvRound(x0 + 1000*(-b)),
				  cvRound(y0 + 1000*(a)));
		Point pt2(cvRound(x0 - 1000*(-b)),
				  cvRound(y0 - 1000*(a)));
		line( draw, pt1, pt2, color, 3, 8 );
	}
	return draw;
}

// Finds the intersection of two lines, or returns false.
// The lines are defined by (o1, p1) and (o2, p2).
// https://answers.opencv.org/question/9511/how-to-find-the-intersection-point-of-two-lines/
bool intersection(Point2f o1, Point2f p1, Point2f o2, Point2f p2, Point2f &r)
{
    Point2f x = o2 - o1;
    Point2f d1 = p1 - o1;
    Point2f d2 = p2 - o2;

    float cross = d1.x*d2.y - d1.y*d2.x;
    if (abs(cross) < /*EPS*/1e-8)
        return false;

    double t1 = (x.x * d2.y - x.y * d2.x)/cross;
    r = o1 + d1 * t1;
    return true;
}

Point2f fugePoint(vector<Vec2f> lines1, vector<Vec2f> lines2, int pointSize) {
	vector<Point2f> votes;

	int size = min(lines1.size(), lines2.size());

	for (int j = 0; j < size; j++) {
		Vec2f l[2];
		l[0] = lines1[j];
		l[1] = lines2[j];

		Point2f r, o[2], p[2];

		for (int i = 0; i < 2; i++) {
			double a = cos(l[i][1]), b = sin(l[i][1]);
			double x0 = a*l[i][0], y0 = b*l[i][0];
			o[i] = Point2f(cvRound(x0 + 1000*(-b)), cvRound(y0 + 1000*(a)));
			p[i] = Point2f(cvRound(x0 - 1000*(-b)), cvRound(y0 - 1000*(a)));
		}


		if(intersection(o[0], p[0], o[1], p[1], r)) {
			votes.push_back(r);
		}
	}

//	for (auto v : votes) std::cout << "Point " << v << std::endl;

	cv::Point2f sum  = std::accumulate(votes.begin(), votes.end(), Point2f(0,0));
	Point2f mean_point(sum * (1.0f / votes.size()));

	return mean_point;
}
