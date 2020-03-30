#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <windows.h> // For Sleep
#include "utils.hpp"
#include <fstream>
#include <string>
#include <vector>
#include <iomanip>


using namespace cv;
using namespace std;

int gauss_k = 1;
int gauss_s = 5;

int sobel_k_size = 3;


int main(int argc, char *argv[]) {
	if (strcmp(argv[1], "1") == 0) {
		Mat image;
		image = imread("files/imagenesT3/poster.pgm", CV_LOAD_IMAGE_GRAYSCALE);   // Read the file
		checkImg(image);

		// filtro gaussiano
		GaussianBlur(image, image, Size2i(gauss_s,gauss_s), gauss_k, gauss_k, BORDER_DEFAULT);

		imshow("Filtro gaussiano", image );                   // Show our image inside it.
		waitKey(0);
		cvDestroyWindow("Filtro gaussiano");

		Mat sobelx;
		Sobel(image, sobelx,CV_32F, 1, 0);

		double minVal, maxVal;
		minMaxLoc(sobelx, &minVal, &maxVal); //find minimum and maximum intensities
		cout << "Gradiente vertical " << endl << "minVal : " << minVal << endl << "maxVal : " << maxVal << endl;

		Mat draw;
		sobelx.convertTo(draw, CV_8U, 255.0/(maxVal - minVal), -minVal * 255.0/(maxVal - minVal));

		imshow("Gradiente vertical", draw );                   // Show our image inside it.
		waitKey(0);
		cvDestroyWindow("Gradiente vertical");


		Mat sobely;
		Sobel(image, sobely,CV_32F, 0, 1);

		minMaxLoc(sobely, &minVal, &maxVal); //find minimum and maximum intensities
		cout << "Gradiente horizontal (la coordenada y va hacia arriba)" << endl << "minVal : " << minVal << endl << "maxVal : " << maxVal << endl;

		sobely.convertTo(draw, CV_8U, 255.0/(maxVal - minVal), -minVal * 255.0/(maxVal - minVal));

		imshow("Gradiente horizontal", draw );                   // Show our image inside it.
		waitKey(0);
		cvDestroyWindow("Gradiente horizontal");

	}
	else if (strcmp(argv[1], "2") == 0) {

		}
	else {
		std::cout << "Usage: " << argv[0] << " <1 | 2> " << std::endl;
		std::cout << "\t 1 : Gradiente, m�dulo y orientaci�n" << std::endl;
		std::cout << "\t 2 : Detecci�n del punto central" << std::endl;
	}
    return 0;
}
