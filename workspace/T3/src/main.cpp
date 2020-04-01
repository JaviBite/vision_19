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

int main(int argc, char *argv[]) {
	if (strcmp(argv[1], "1") == 0) {
		Mat image;
		image = imread("files/imagenesT3/poster.pgm", CV_LOAD_IMAGE_GRAYSCALE);
		checkImg(image);

		// filtro gaussiano
		GaussianBlur(image, image, Size2i(gauss_s,gauss_s), gauss_k, gauss_k, BORDER_DEFAULT);

		imshow("Filtro gaussiano", image );
		waitKey(0);


		// calcular gradiente horizontal
		Mat sobelx;
		Sobel(image, sobelx,CV_32F, 1, 0);

		double minVal, maxVal;
		minMaxLoc(sobelx, &minVal, &maxVal);
		cout << "Gradiente vertical " << endl << "minVal : " << minVal << endl << "maxVal : " << maxVal << endl << endl;
		// dibujar gradiente horizontal
		Mat drawx;
		sobelx.convertTo(drawx, CV_8U, 255.0/(maxVal - minVal), -minVal * 255.0/(maxVal - minVal));

		imshow("Gradiente vertical", drawx );
		waitKey(0);


		// calcular gradiente vertical
		Mat sobely;
		Sobel(image, sobely,CV_32F, 0, 1);

		minMaxLoc(sobely, &minVal, &maxVal);
		cout << "Gradiente horizontal (la coordenada \"y\" va hacia abajo)" << endl << "minVal : " << minVal << endl << "maxVal : " << maxVal << endl << endl;
		// dibujar gradiente vertical
		Mat drawy;
		sobely.convertTo(drawy, CV_8U, 255.0/(maxVal - minVal), -minVal * 255.0/(maxVal - minVal));

		imshow("Gradiente horizontal", drawy );
		waitKey(0);



		// calcular el módulo
		Mat moduloaux = sobelx;
		for (int i = 0; i< sobelx.rows; i++)
			for (int j = 0; j< sobelx.cols; j++)
				moduloaux.at<float>(i,j) = sqrt(pow(sobelx.at<float>(i,j),2) + pow(sobely.at<float>(i,j),2));
		minMaxLoc(moduloaux, &minVal, &maxVal);
		cout << "Modulo" << endl << "minVal : " << minVal << endl << "maxVal : " << maxVal << endl << endl;
		// convertir módulo a rango [0,255] y dibujarlo
		Mat modulo;
		moduloaux.convertTo(modulo, CV_8U, 255.0/(maxVal - minVal), -minVal * 255.0/(maxVal - minVal));

		imshow("Modulo", modulo );
		waitKey(0);


		// calcular la orientación
		Mat orientacionaux = sobelx;
		for (int i = 0; i< sobelx.rows; i++)
			for (int j = 0; j< sobelx.cols; j++)
				orientacionaux.at<float>(i,j) = atan2(sobely.at<float>(i,j) , sobelx.at<float>(i,j));
		minMaxLoc(orientacionaux, &minVal, &maxVal);
		cout << "Orientacion" << endl << "minVal : " << minVal << endl << "maxVal : " << maxVal << endl << endl;
		// convertir orientación a rango [0,255] y dibujarla
		Mat orientacion;
		orientacionaux.convertTo(orientacion, CV_8U, 255.0/(maxVal - minVal), -minVal * 255.0/(maxVal - minVal));

		imshow("Orientacion", orientacion );
		waitKey(0);
		destroyAllWindows();

	}
	else if (strcmp(argv[1], "2") == 0) {
		 Mat src, dst, color_dst;
		    if( argc != 2 || !(src=imread("files/ImagenesT3/pasillo2.pgm", 0)).data)
		        return -1;

		    //dst = modulo(src, true);
		    Canny( src, dst, 50, 200, 3 );
		    cvtColor( dst, color_dst, CV_GRAY2BGR );

		    std::vector<Vec2f> lines, lines1, lines2;
		    HoughLines( dst, lines, 1, CV_PI/180, 100 );

		    splitLines(lines, lines1, lines2, 0.3);

		    drawLines(color_dst, lines1, Scalar(255,0,0));
		    drawLines(color_dst, lines2, Scalar(0,0,255));

		    namedWindow( "Source", 1 );
		    imshow( "Source", src );

		    namedWindow( "Detected Lines", 1 );
		    imshow( "Detected Lines", color_dst );

		    Point fuge = fugePoint(lines1, lines2, 1);

		    Mat fugeDraw = src;
		    cv::drawMarker(fugeDraw, fuge,  cv::Scalar(0, 0, 255), MARKER_CROSS, 10, 2);
		    namedWindow( "Fuge point", 1 );
		    imshow( "Fuge point", fugeDraw );

		    std::cout << "Fuge point : " << fuge << std::endl;

		    waitKey(0);
		    return 0;

		}
	else {
		std::cout << "Usage: " << argv[0] << " <1 | 2> " << std::endl;
		std::cout << "\t 1 : Gradiente, módulo y orientación" << std::endl;
		std::cout << "\t 2 : Detección del punto central" << std::endl;
	}
    return 0;
}
