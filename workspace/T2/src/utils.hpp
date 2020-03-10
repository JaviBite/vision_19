#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <functional>
#include <cmath>
#define CV_BGR2YCrCb COLOR_BGR2YCrCb
#define CV_YCrCb2BGR COLOR_YCrCb2BGR
#define CV_RGB2GRAY COLOR_RGB2GRAY
#define CV_BGR2HSV COLOR_BGR2HSV
#define CV_HSV2BGR COLOR_HSV2BGR


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

//bool is_skin(float threshold, uchar r, uchar g, uchar b) {
void isSkin (uchar& hue, uchar& saturation, uchar& value, float aux) {
	//	uchar skin[3] = {r, g, b};
	//	if ( skin_cmp(average, skin) < threshold ||
	//			skin_cmp(pale, skin) < threshold ||
	//			skin_cmp(pale_tan, skin) < threshold ||
	//			skin_cmp(tanned, skin) < threshold ||
	//			//skin_cmp(black, skin) < threshold ||
	//			//skin_cmp(brown, skin) < threshold ||
	//			false) return true;
	//	else return false;
	//if( hue > 0 && hue < 20 && saturation > 48 && value > 80 ) {
	 if (hue > 10 && hue < 25 && saturation > 48 && saturation < 255 && value > 80 ) {
		 hue = 155; saturation = 155; value = 155;
	 }
	 else {
		 hue = 0; saturation = 0; value = 0;
	 }
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

void alienBlue(uchar& b, uchar& g, uchar& r){
	   r = saturate_cast< uchar >(r * ( 1 - 0.5));
	   g = saturate_cast< uchar >(g * ( 1 - 0.5));
	   b = saturate_cast< uchar >(b * ( 1 + 0.5));

}

void alienRed(uchar& b, uchar& g, uchar& r){
	   r = saturate_cast< uchar >(r * ( 1 + 0.5));
	   g = saturate_cast< uchar >(g * ( 1 - 0.5));
	   b = saturate_cast< uchar >(b * ( 1 - 0.5));

}

void alienGreen(uchar& b, uchar& g, uchar& r){
	   r = saturate_cast< uchar >(r * ( 1 - 0.5));
	   g = saturate_cast< uchar >(g * ( 1 + 0.5));
	   b = saturate_cast< uchar >(b * ( 1 - 0.5));

}

uchar take_on_me(uchar &a, uchar* end, double aux[]){
	int nCols = aux[1];
	int nChannels = aux[2];
	int threshold = aux[3];
	uchar* main = &a;
	uchar* next = &a + nChannels;
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
//YCrCb threshold
const uchar Y_MIN  = 50;		//0
const uchar Y_MAX  = 255;		//255
const uchar Cr_MIN = 133;		//133
const uchar Cr_MAX = 170;		//173
const uchar Cb_MIN = 77;		//77
const uchar Cb_MAX = 127;		//127

Mat skinMat(const Mat& img){
	Mat channels[3];
	Mat ret, no_noise;
	GaussianBlur(img, no_noise, Size2i(5,5), 10, 10, BORDER_DEFAULT);
	cvtColor(img,ret,cv::COLOR_BGR2YCrCb);
	inRange(ret,cv::Scalar(Y_MIN,Cr_MIN,Cb_MIN),cv::Scalar(Y_MAX,Cr_MAX,Cb_MAX),ret);
	//split(apply_effect_rgb(ret,isSkin,0),channels);

	return ret;
}

Mat generarAlien(Mat& skin, Mat& I, int mode) {
	int nRows = I.rows;
	int nChannels = I.channels();
	int nCols = I.cols * nChannels;
	uchar* p;
	uchar* s;
	int j2;
	if (I.isContinuous()){
		nCols *= nRows;
		nRows = 1;
	}
	for( int i = 0; i < nRows; ++i)	{
			p = I.ptr<uchar>(i);
			s = skin.ptr<uchar>(i);
			j2 = 0;
			for ( int j = 0; j < nCols; j = j + nChannels) {
				switch(mode){
				case 0:
					if(s[j2] > 0) alienRed(p[j], p[j+1], p[j+2]);
					break;

				case 1:
					if(s[j2] > 0) alienGreen(p[j], p[j+1], p[j+2]);
					break;

				case 2:
					if(s[j2] > 0) alienBlue(p[j], p[j+1], p[j+2]);
					break;
				}

				j2++;
			}
		}
	return I;
}


uchar reducirColorF(uchar &a, uchar* end, double aux[]){
	int factor = aux[3];
	return saturate_cast< uchar >(a/factor*factor + factor/2);
}

uchar contrastF(uchar &a, uchar* end, double aux[]){
	double contrast = aux[3];
	int ret = a * contrast;
	if (ret > 255) ret = 255;
	if (ret < 0) ret = 0;
	return ret;
}

Mat equalizarCV(Mat in) {
	Mat ycrcb;

	cvtColor(in,ycrcb,CV_BGR2YCrCb);

	vector<Mat> channels;
	split(ycrcb,channels);

	equalizeHist(channels[0], channels[0]);

	Mat result;
	merge(channels,ycrcb);

	cvtColor(ycrcb,result,CV_YCrCb2BGR);

	return result;
}

int countPixels(const cv::Mat &image, cv::Scalar color) {

    Mat binary_image;
    cv::inRange(image, color, color, binary_image);
        return cv::countNonZero(binary_image);
}

Mat equalizarOurs(Mat& in) {
	Mat result;

	Mat hcv;
	vector<Mat> channels;
	cvtColor(in,hcv,CV_BGR2YCrCb);
	split(hcv,channels);

	float px[255];
	int size = in.rows * in.cols;
	for (int i = 0; i < 255; ++i)
		px[i] = countPixels(channels[0],(Scalar)i);

//	for (int i = 0;i < 256 ; i++) cout << "valuee " << px[i] << endl;
//	cv::waitKey();

	float cdf[256];
	cdf[0] = px[0];
	float min = 255;
	for (int i = 1; i < 255; ++i) {
		 cdf[i] = ( cdf[i-1] + px[i] );
		 if (cdf[i] < min ) min = cdf[i];
	}

//	for (int i = 0;i < 256 ; i++) cout << "cdfvaluee " << cdf[i] << endl;
//	cv::waitKey();

	Mat lookUpTable(1, 256, CV_8U);
	uchar* p = lookUpTable.ptr();
	for (int i = 0; i < 256; ++i)
		p[i] = round(((cdf[i] - min ) / (float)(size - min) ) * 255.0f);

//	for (int i = 0;i < 256 ; i++) cout << "valueel " << (int)p[i] << endl;
//    cv::waitKey();

	LUT(channels[0], lookUpTable,  channels[0]);

	merge(channels,hcv);

	cvtColor(hcv,result, CV_YCrCb2BGR);
	return result;
}

void generarDistorsion(Mat& matriz, int mode, float k1) {
	// https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html
	Mat in = matriz.clone();
	switch(mode){
		case 0:
			k1 = abs(k1); // para cojín
			break;

		case 1:
			k1 = -abs(k1); // para cojín
			break;
	}

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
