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

//		int morph_size = 1;
//		Mat element = getStructuringElement(MORPH_RECT, Size(2*morph_size + 1, 2*morph_size+1),
//					Point(morph_size, morph_size));
		std::vector<std::vector<Point>> contours;
		std::vector<Vec4i> hierarchy;

		//morphologyEx(image, image, MORPH_CLOSE, element);
		cv::findContours(image, contours, hierarchy,CONT_MODE, CONT_METH);
		std::sort(contours.begin(), contours.end(), compareContourAreas);

		Mat drawCont = drawableContours(contours, hierarchy, image.size());
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

		std::cout << "--------------------------------------" << std::endl;
	}
	else if (strcmp(argv[1], "2") == 0) {
		if (argc >= 4) {
			aprender(argv[2], argv[3]);
		}
		else {

			// Punto 4 y 5: Aprendizaje supervisado y Regularización

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
		}

		std::cout << "DONE!" << std::endl;

	}
	else if (strcmp(argv[1], "reconocer") == 0) {

		// Punto 5: Reconocer

		vector<Fig> figuras = modelo();

		for (Fig f: figuras) {
						std::cout << "Figura = " << f.nombre << std::endl;
						std::cout << "MEAN AREA = " << f.mean_area << std::endl;
						std::cout << "MEAN PERIM = " << f.mean_perim << std::endl;
						std::cout << "MEAN M0 = " << f.mean_m0 << std::endl;
						std::cout << "MEAN M1 = " << f.mean_m1 << std::endl;
						std::cout << "MEAN M2 = " << f.mean_m2 << std::endl;
						std::cout << "STD AREA = " << f.std_area << std::endl;
						std::cout << "STD PERIM = " << f.std_perim << std::endl;
						std::cout << "STD M0 = " << f.std_m0 << std::endl;
						std::cout << "STD M1 = " << f.std_m1 << std::endl;
						std::cout << "STD M2 = " << f.std_m2 << std::endl << std::endl;
					}

		std::cout << "Código identificación: " << std::endl;
		for (int i = 0; i < (int)figuras.size() ; i++) {
			std::cout << std::setw(10) <<  figuras[i].nombre << "\t" << colours[i] << std::endl;
		}
		std::cout << std::setw(10) << "Desconocido" << "\t" << Scalar(125,125,125) << std::endl;
		std::cout << std::endl;

		vector<String> files;

		if (argc == 3) {
			files.push_back(argv[2]);
		}
		else {

			files.push_back("files/imagenesT2/reco1.pgm");
			files.push_back("files/imagenesT2/reco2.pgm");
			files.push_back("files/imagenesT2/reco3.pgm");
		}

			vector<String> ret;
			Mat drawCont;
			int i = 0;
			for(String file: files) {
				ret = reconocer(file, figuras, drawCont);

				Mat image = imread(file, CV_LOAD_IMAGE_COLOR);   // Read the file
				checkImg(image);

				imshow("Display window", image );
				imshow("Contours", drawCont);

				std::cout << "Figura " << i << " : ";
				for(String s: ret) std::cout << s << ", ";
				std::cout << std::endl;

				waitKey(0);
				cvDestroyWindow("Display window");
				cvDestroyWindow("Contours");
				i++;
			}
		}
	else {
		std::cout << "Usage: " << argv[0] << " <1 | 2> " << std::endl;
		std::cout << "\t 1 : Gradiente, módulo y orientación" << std::endl;
		std::cout << "\t 2 : Detección del punto central" << std::endl;
	}
    return 0;
}
