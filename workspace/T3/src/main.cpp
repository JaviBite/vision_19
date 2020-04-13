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

const int sigmaCanny = 1;

int main(int argc, char *argv[]) {
	if (strcmp(argv[1], "1") == 0) {

		Mat image;
		image = imread("files/imagenesT3/poster.pgm", CV_LOAD_IMAGE_GRAYSCALE);
		checkImg(image);

		imshow("Original", image );
		waitKey(0);
		double minVal, maxVal;
		minMaxLoc(image, &minVal, &maxVal);
		cout << "Valores mínimos y máximos de la imagen: " << endl << "minVal : " << minVal << endl << "maxVal : " << maxVal << endl << endl;

		cout << endl << endl << "CANNY" << endl << endl;
		// cambiar el tipo de la imagen para poder hacer operaciones con ella
		image.convertTo(image, CV_32F);

		// obtener el tamaño del kernel a partir de sigma
		int n = sigmaCanny * 5;
		if (n%2 == 0) n++;

		double kernel[n];
		double K1 = 0, K2 = 0;

		// generar el primer kernel y pasarlo por la imagen (filtro gaussiano vertical)
		Mat cannyx;
		for (int i = 0; i<n; i++){
			kernel[i] = gaussiana(i-(n/2),sigmaCanny);
			if(kernel[i] > 0){
				K1 += kernel[i];
			}
		}
		cannyx = pasarFiltro(image,kernel,n,true);

		minMaxLoc(cannyx, &minVal, &maxVal);
		cout << "Valores mínimos y máximos de la gaussiana: " << endl << "minVal : " << minVal << endl << "maxVal : " << maxVal << endl << endl;

		// generar segundo kernel (derivada de la gaussiana) y pasarlo a la imagen
		for (int i = 0; i<n; i++){
			kernel[i] = derivadaGaussiana(i-(n/2),sigmaCanny);
			if(kernel[i] > 0){
				K2 += kernel[i];
			}
		}
		cannyx = pasarFiltro(cannyx,kernel,n,false);
		// dividir por los valores positivos
		cannyx = cannyx / (K1*K2);

		minMaxLoc(cannyx, &minVal, &maxVal);
		cout << "Valores mínimos y máximos del gradiente horizontal final: " << endl << "minVal : " << minVal << endl << "maxVal : " << maxVal << endl << endl;
		Mat cdrawx = cannyx;
		cdrawx = cannyx / 2 + 128;
		cdrawx.convertTo(cdrawx,CV_8U );
		imshow("Gradiente horizontal Canny", cdrawx);
		waitKey(0);


		K1 = 0; K2 = 0;

		// generar el primer kernel y pasarlo por la imagen (filtro de derivada de gaussiana vertical)
		Mat cannyy;
		for (int i = 0; i<n; i++){
			kernel[i] = derivadaGaussiana(i-(n/2),sigmaCanny);
			if(kernel[i] > 0){
				K1 += kernel[i];
			}
		}
		cannyy = pasarFiltro(image,kernel,n,true);

		minMaxLoc(cannyx, &minVal, &maxVal);
		cout << "Valores mínimos y máximos de la derivada gaussiana: " << endl << "minVal : " << minVal << endl << "maxVal : " << maxVal << endl << endl;

		// generar segundo kernel (gaussiana) y pasarlo a la imagen
		for (int i = 0; i<n; i++){
			kernel[i] = gaussiana(i-(n/2),sigmaCanny);
			if(kernel[i] > 0){
				K2 += kernel[i];
			}
		}
		cannyy = pasarFiltro(cannyy,kernel,n,false);
		// dividir por los valores positivos
		cannyy = cannyy / (K1*K2);

		minMaxLoc(cannyx, &minVal, &maxVal);
		cout << "Valores mínimos y máximos del gradiente vertical final: " << endl << "minVal : " << minVal << endl << "maxVal : " << maxVal << endl << endl;

		Mat cdrawy = cannyy;
		cdrawy = cannyy / 2 + 128;
		cdrawy.convertTo(cdrawy,CV_8U );
		imshow("Gradiente vertical Canny", cdrawy);
		waitKey(0);

		// Calcular el módulo
		Mat cmoduloaux = cannyx.clone();
		for (int i = 0; i< cannyx.rows; i++)
			for (int j = 0; j< cannyx.cols; j++)
				cmoduloaux.at<float>(i,j) = sqrt(pow(cannyx.at<float>(i,j),2) + pow(cannyy.at<float>(i,j),2));
		minMaxLoc(cmoduloaux, &minVal, &maxVal);
		cout << "Valores mínimos y máximos del módulo:" << endl << "minVal : " << minVal << endl << "maxVal : " << maxVal << endl << endl;
		// convertir módulo al tipo 8bits y dibujarlo
		Mat cmodulo;
		cmoduloaux.convertTo(cmodulo, CV_8U, 255.0/(maxVal - minVal), -minVal * 255.0/(maxVal - minVal));
		imshow("Modulo Canny", cmodulo );
		waitKey(0);


		// calcular la orientación
		Mat corientacionaux = cannyx;
		for (int i = 0; i< cannyx.rows; i++)
			for (int j = 0; j< cannyx.cols; j++)
				corientacionaux.at<float>(i,j) = atan2(cannyy.at<float>(i,j) , cannyx.at<float>(i,j));
		minMaxLoc(corientacionaux, &minVal, &maxVal);
		cout << "Valores mínimos y máximos de la orientación: " << endl << "minVal : " << minVal << endl << "maxVal : " << maxVal << endl << endl;
		// convertir orientación a rango [0,255] y dibujarla
		Mat corientacion;
		corientacionaux.convertTo(corientacion, CV_8U, 255.0/(maxVal - minVal), -minVal * 255.0/(maxVal - minVal));

		imshow("Orientacion Canny", corientacion );
		waitKey(0);












		cout << endl << endl << "SOBEL" << endl << endl;
		image = imread("files/imagenesT3/poster.pgm", CV_LOAD_IMAGE_GRAYSCALE);
		checkImg(image);

		// filtro gaussiano
		GaussianBlur(image, image, Size2i(gauss_s,gauss_s), gauss_k, gauss_k, BORDER_DEFAULT);

		imshow("Filtro gaussiano antes de Sobel", image );
		waitKey(0);


		// calcular gradiente horizontal
		Mat sobelx;
		Sobel(image, sobelx,CV_32F, 1, 0);

		minMaxLoc(sobelx, &minVal, &maxVal);
		cout << "Valores mínimos y máximos del gradiente horizontal: " << endl << "minVal : " << minVal << endl << "maxVal : " << maxVal << endl << endl;
		// dibujar gradiente horizontal
		Mat drawx;
		sobelx.convertTo(drawx, CV_8U, 255.0/(maxVal - minVal), -minVal * 255.0/(maxVal - minVal));

		imshow("Gradiente horizontal Sobel", drawx );
		waitKey(0);




		// calcular gradiente vertical
		Mat sobely;
		Sobel(image, sobely,CV_32F, 0, 1);

		minMaxLoc(sobely, &minVal, &maxVal);
		cout << "Valores mínimos y máximos del gradiente horizontal: " << endl << "minVal : " << minVal << endl << "maxVal : " << maxVal << endl << endl;
		// dibujar gradiente vertical
		Mat drawy;
		sobely.convertTo(drawy, CV_8U, 255.0/(maxVal - minVal), -minVal * 255.0/(maxVal - minVal));

		imshow("Gradiente vertical Sobel (la coordenada \"y\" va hacia abajo)", drawy );
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
	else if (strcmp(argv[1], "3") == 0){
		char tipka;
		Mat frame;
		//--- INITIALIZE VIDEOCAPTURE
		VideoCapture cap;
		// open the default camera using default API
		cap.open(0);
		// OR advance usage: select any API backend
		int deviceID = 0;             // 0 = open default camera
		int apiID = cv::CAP_ANY;      // 0 = autodetect default API
									  // open selected camera using selected API
		cap.open(deviceID + apiID);
		// check if we succeeded
		if (!cap.isOpened()) {
			cerr << "ERROR! Unable to open camera\n";
			return -1;
		}

		for(;;){
			// wait for a new frame from camera and store it into 'frame'
			cap.read(frame);

			if (frame.empty()) {
				cerr << "ERROR! blank frame grabbed\n";
				break;
			}

			Sleep(5); // Sleep is mandatory - for no leg!





			Mat dst, color_dst;


			Canny( frame, dst, 50, 200, 3 );
			cvtColor( dst, color_dst, CV_GRAY2BGR );

			std::vector<Vec2f> lines, lines1, lines2;
			HoughLines( dst, lines, 1, CV_PI/180, 100 );

			splitLines(lines, lines1, lines2, 0.3);

			drawLines(color_dst, lines1, Scalar(255,0,0));
			drawLines(color_dst, lines2, Scalar(0,0,255));

			Point fuge = fugePoint(lines1, lines2, 1);

			Mat fugeDraw = frame;
			cv::drawMarker(fugeDraw, fuge,  cv::Scalar(0, 0, 255), MARKER_CROSS, 10, 2);




			imshow("CAMERA 1", fugeDraw);
			//imshow("CAMERA 1", color_dst);

			tipka = cv::waitKey(30);

			if (tipka == 'q'){
				break;
			}

		}
	}
	else {
		std::cout << "Usage: " << argv[0] << " <1 | 2> " << std::endl;
		std::cout << "\t 1 : Gradiente, módulo y orientación" << std::endl;
		std::cout << "\t 2 : Detección del punto central" << std::endl;
	}
    return 0;
}
