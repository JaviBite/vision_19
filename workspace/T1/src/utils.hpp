#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <functional>
#include <cmath>

using namespace cv;
using namespace std;



// Aux Functions

const uchar average[3] = {197, 140, 133};
const uchar pale[3] = {236, 188, 180};
const uchar pale_tan[3] = {209, 163, 164};
const uchar tanned[3] = {161, 102, 94};
const uchar black[3] = {80, 51, 53};
const uchar brown[3] = {89, 47, 42};

int skin_cmp(const uchar skin_tone[3], uchar compare[3]) {
	return abs((int)skin_tone[0] - (int)compare[0])
			+ abs((int)skin_tone[1] - (int)compare[1])
			+ abs((int)skin_tone[2] - (int)compare[2]);
}

bool is_skin(float threshold, uchar r, uchar g, uchar b) {
	uchar skin[3] = {r, g, b};
	if ( skin_cmp(average, skin) < threshold ||
			skin_cmp(pale, skin) < threshold ||
			skin_cmp(pale_tan, skin) < threshold ||
			skin_cmp(tanned, skin) < threshold ||
			//skin_cmp(black, skin) < threshold ||
			//skin_cmp(brown, skin) < threshold ||
			false) return true;
	else return false;
}

// Effects

uchar negative(uchar& c, uchar* end, double aux[]) {
	return 255 - c;
}

//void negative(uchar& b, uchar& g, uchar& r, int aux[]){
//   r = 255 - r;
//   g = 255 - g;
//   b = 255 - b;
//}

void alien_blue(uchar& b, uchar& g, uchar& r, double threshold){
   if (is_skin(threshold,r,g,b)) {
	   r = r * ( 1 - 0.7);
	   g = g * ( 1 - 0.7);
	   b = b * ( 1 + 1.5);
	   if (r < 35) r = 35;
	   if (g < 35) g = 35;
	   if (b > 150) b = 150;
   }
}

uchar take_on_me(uchar &a, uchar* end, double aux[]){
	int nCols = aux[1];
	int nChannels = aux[2];
	int threshold = aux[3];
	uchar* main = &a;
	uchar* next = &a + 3;
	uchar* down = &a + (nChannels * nCols);
	if ( (next < end && abs((int)*main - (int)*next) < threshold) ||
		 (down < end && abs((int)*main - (int)*down) < threshold)    ) {
			 return 255;
		 }
		 else return 1;
}



// Functions

Mat apply_effect(Mat I, function<uchar (uchar&, uchar*, double[])> effect, double threshold) {
	double nRows = I.rows;
	double nChannels = I.channels();
	double nCols = I.cols * nChannels;
	double info[4] = {nRows, nCols, nChannels, threshold};
	uchar* end = I.ptr(I.size);
	uchar* p;
	if (I.isContinuous()){
		nCols *= nRows;
		nRows = 1;
	}
	for( int i = 0; i < nRows; ++i)
		{
			p = I.ptr<uchar>(i);
			for ( int j = 0; j < nCols; ++j)
			{
				p[j] = effect(p[j], end, info);
			}
		}
	return I;
}

Mat apply_effect_rgb(Mat I, function<void (uchar&, uchar&, uchar&, float)> effect, float threshold) {
	int nRows = I.rows;
	int nChannels = I.channels();
	int nCols = I.cols * nChannels;
	uchar* p;
	if (I.isContinuous()){
		nCols *= nRows;
		nRows = 1;
	}
	for( int i = 0; i < nRows; ++i)
		{
			p = I.ptr<uchar>(i);
			for ( int j = 0; j < nCols; j = j + nChannels)
			{

				effect(p[j], p[j+1], p[j+2], threshold);
			}
		}
	return I;
}


Vec3b generarAlienPixel(Vec3b color){
	int hue = color[0];
	int saturation = color[1];
	int value = color[2];

	if( hue > 0 && hue < 20 && saturation > 48 && value > 80 ) {
		color[0] = 55;
		color[1] = 255;
		color[2] = 255;
	}
	return color;
}

void generarAlien(Mat& matriz)
{
	for (int i = 0; i < matriz.rows; i++)
	{
		for (int j = 0; j < matriz.cols; j++)
		{
			matriz.at<Vec3b>(i,j) = generarAlienPixel(matriz.at<Vec3b>(i,j));
		}
	}
}

uchar reducirColorF(uchar &a, uchar* end, double aux[]){
	if(a < 80) return 0;
	if(a < 150) return 80;
	return 255;
}

uchar contrastF(uchar &a, uchar* end, double aux[]){
	double contrast = aux[3];
	int ret = a * contrast;
	if (ret > 255) ret = 255;
	if (ret < 0) ret = 0;
	return ret;
}

void generarDistorsion(Mat& matriz)
{
	Mat in = matriz.clone();
	float k1 = -0.5; // para cojín
	//float k1 = 1;  // para barril
	float k2 = 0.0;
	float p1 = 0.0;
	float p2 = 0.0;
	Mat distCoeffs = Mat(4,1,CV_32FC1);
	distCoeffs.at<float>(0,0) = k1;
	distCoeffs.at<float>(1,0) = k2;
	distCoeffs.at<float>(2,0) = p1;
	distCoeffs.at<float>(3,0) = p2;

	Mat cam = Mat(3,3,CV_32FC1);
	cam.at<float>(0,2) = matriz.cols/2;
	cam.at<float>(1,2) = matriz.rows/2;
	cam.at<float>(0,0) = matriz.cols;
	cam.at<float>(1,1) = matriz.rows;
	cam.at<float>(2,2) = 1;

	undistort(in,matriz,cam,distCoeffs);
}
