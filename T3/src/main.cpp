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



		// calcular el m�dulo
		Mat moduloaux = sobelx;
		for (int i = 0; i< sobelx.rows; i++)
			for (int j = 0; j< sobelx.cols; j++)
				moduloaux.at<float>(i,j) = sqrt(pow(sobelx.at<float>(i,j),2) + pow(sobely.at<float>(i,j),2));
		minMaxLoc(moduloaux, &minVal, &maxVal);
		cout << "Modulo" << endl << "minVal : " << minVal << endl << "maxVal : " << maxVal << endl << endl;
		// convertir m�dulo a rango [0,255] y dibujarlo
		Mat modulo;
		moduloaux.convertTo(modulo, CV_8U, 255.0/(maxVal - minVal), -minVal * 255.0/(maxVal - minVal));

		imshow("Modulo", modulo );
		waitKey(0);


		// calcular la orientaci�n
		Mat orientacionaux = sobelx;
		for (int i = 0; i< sobelx.rows; i++)
			for (int j = 0; j< sobelx.cols; j++)
				orientacionaux.at<float>(i,j) = atan2(sobely.at<float>(i,j) , sobelx.at<float>(i,j));
		minMaxLoc(orientacionaux, &minVal, &maxVal);
		cout << "Orientacion" << endl << "minVal : " << minVal << endl << "maxVal : " << maxVal << endl << endl;
		// convertir orientaci�n a rango [0,255] y dibujarla
		Mat orientacion;
		orientacionaux.convertTo(orientacion, CV_8U, 255.0/(maxVal - minVal), -minVal * 255.0/(maxVal - minVal));

		imshow("Orientacion", orientacion );
		waitKey(0);
		destroyAllWindows();

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
