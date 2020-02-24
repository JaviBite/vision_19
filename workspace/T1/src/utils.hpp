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

uchar negative(uchar& c, uchar* end, int aux[]) {
	return 255 - c;
}

//void negative(uchar& b, uchar& g, uchar& r, int aux[]){
//   r = 255 - r;
//   g = 255 - g;
//   b = 255 - b;
//}

void alien_blue(uchar& b, uchar& g, uchar& r, float threshold){
   if (is_skin(threshold,r,g,b)) {
	   r = r * ( 1 - 0.7);
	   g = g * ( 1 - 0.7);
	   b = b * ( 1 + 1.5);
	   if (r < 35) r = 35;
	   if (g < 35) g = 35;
	   if (b > 150) b = 150;
   }
}

uchar take_on_me(uchar &a, uchar* end, int aux[]){
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

Mat apply_effect(Mat I, function<uchar (uchar&, uchar*, int[])> effect, float threshold) {
	int nRows = I.rows;
	int nChannels = I.channels();
	int nCols = I.cols * nChannels;
	int info[4] = {nRows, nCols, nChannels, threshold};
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
