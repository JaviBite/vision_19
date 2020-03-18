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

	image = imread("files/imagenesT2/reco1.pgm", CV_LOAD_IMAGE_COLOR);   // Read the file
	checkImg(image);

	imshow("Display window", image );                   // Show our image inside it.
	waitKey(0);

	image = toBinaryOtsu(image);

	Mat draw_contours = Mat::zeros( image.size(), CV_8UC3 );
	std::vector<std::vector<Point>> contours;
	std::vector<Vec4i> hierarchy;

	int mode = CONT_MODE;
	int method = CONT_METH;

	cv::findContours(image, contours, hierarchy, mode, method);
	std::sort(contours.begin(), contours.end(), compareContourAreas);
//	for( size_t i = 0; i < contours.size(); i++ )
//	{
//		drawContours(draw_contours,contours,i, Scalar(255,50,50),-1,8,noArray(), 2, Point() );
//		imshow("Contours", draw_contours);
//		waitKey(0);
//	}
	Mat drawCont = drawableContours(contours, image.size());
	imshow("Contours", drawCont);
	waitKey(0);

	cvDestroyWindow("Contours");

	// Point 3: Parameters

	std::vector<vector<float>> params = calculateParameters(contours);
	int i = 1;
	for (std::vector<float> contorno: params) {
		std::cout << "CONTORNO " << i << std::endl;
		std::cout << "AREA  = " << contorno[0] << std::endl;
		std::cout << "PERIM = " << contorno[1] << std::endl;
		std::cout << "M0    = " << contorno[2] << std::endl;
		std::cout << "M1    = " << contorno[3] << std::endl;
		std::cout << "M2    = " << contorno[4] << std::endl;
		i++;
	}


	// Punto 4: Aprendizaje supervisado


	aprender("files/imagenesT2/circulo1.pgm","circulo");
	aprender("files/imagenesT2/circulo2.pgm","circulo");
	aprender("files/imagenesT2/circulo3.pgm","circulo");
	aprender("files/imagenesT2/circulo4.pgm","circulo");
	aprender("files/imagenesT2/circulo5.pgm","circulo");

	aprender("files/imagenesT2/rectangulo1.pgm","rectangulo");
	aprender("files/imagenesT2/rectangulo2.pgm","rectangulo");
	aprender("files/imagenesT2/rectangulo3.pgm","rectangulo");
	aprender("files/imagenesT2/rectangulo4.pgm","rectangulo");
	aprender("files/imagenesT2/rectangulo5.pgm","rectangulo");

	aprender("files/imagenesT2/triangulo1.pgm","triangulo");
	aprender("files/imagenesT2/triangulo2.pgm","triangulo");
	aprender("files/imagenesT2/triangulo3.pgm","triangulo");
	aprender("files/imagenesT2/triangulo4.pgm","triangulo");
	aprender("files/imagenesT2/triangulo5.pgm","triangulo");

	aprender("files/imagenesT2/vagon1.pgm","vagon");
	aprender("files/imagenesT2/vagon2.pgm","vagon");
	aprender("files/imagenesT2/vagon3.pgm","vagon");
	aprender("files/imagenesT2/vagon4.pgm","vagon");
	aprender("files/imagenesT2/vagon5.pgm","vagon");

	aprender("files/imagenesT2/rueda1.pgm","rueda");
	aprender("files/imagenesT2/rueda2.pgm","rueda");
	aprender("files/imagenesT2/rueda3.pgm","rueda");
	aprender("files/imagenesT2/rueda4.pgm","rueda");
	aprender("files/imagenesT2/rueda5.pgm","rueda");

	// Punto 5: Reconocer

	reconocer("files/imagenesT2/rectangulo1.pgm");

    return 0;
}
