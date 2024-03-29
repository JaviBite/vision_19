#include <opencv2/opencv.hpp>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <iostream>
#include <stdio.h>
#include <windows.h> // For Sleep
#include <iostream>
#include <stdio.h>
#include <windows.h> // For Sleep
#include <fstream>
#include <string>
#include <vector>
#include <list>
#include <thread>

#include "utils.hpp"
// #include <iomanip>

#define HAVE_OPENCV_XFEATURES2D

#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/xfeatures2d.hpp"

using namespace cv;
using namespace cv::xfeatures2d;


// HYPER PARAMS
#define HIEGHT 720
#define WIDTH 960
#define LIVE_SHOT_TIME 1000

void cameraTimer(bool *shotCamera, bool *end) {
	while (!*end) {
		delay(LIVE_SHOT_TIME);
		*shotCamera = true;
	}
}


void help() {
	std::cout << "Usage: " << "program " << "<1|2|3>" << std::endl;
	std::cout << "\t1: Comparativa de detectores" << std::endl;
	std::cout << "\t2: Creaci�n de panoramas a partir de im�genes" << std::endl;
	std::cout << "\t3: Creaci�n de panoramas a partir de la webcam" << std::endl;
	exit(-1);
}

int main(int argc, char *argv[]) {

	if (strcmp(argv[1], "1") == 0) { 	// @suppress("Invalid arguments")
		bool manual;
		int type, matType;
		cout << "Comparativa de detectores, manual (1) o automatica (0): ";
		cin >> manual;

		const int N_MATCHERS = 2;
		cv::DescriptorMatcher::MatcherType matchers[N_MATCHERS];
		cv::String matNames[N_MATCHERS];
		matchers[0] = DescriptorMatcher::BRUTEFORCE_SL2;								matNames[0] = "BRUTE_FORCE_L2";
		matchers[1] = DescriptorMatcher::FLANNBASED;									matNames[1] = "FLANN";

		if (true) {
			cout << "Seleccionar matcher:" << std::endl;
			for (int i = 0; i < N_MATCHERS; i++)
				cout << i << ") " << matNames[i] << std::endl;
			cout << "===========" << std::endl;
			cin >> matType;
		}

		const int N_DETECTORS = 4;
		cv::Ptr<Feature2D> detectors[N_DETECTORS];
		cv::String detecNames[N_DETECTORS];
		detectors[0] = xfeatures2d::SURF::create();											detecNames[0] = "SURF";
		detectors[1] = xfeatures2d::SIFT::create();											detecNames[1] = "SIFT";
		detectors[2] = cv::AKAZE::create();													detecNames[2] = "AKAZE";
		detectors[3] = cv::BRISK::create();													detecNames[3] = "BRISK";

		if (manual == 1) {
			cout << "Seleccionar detector:" << std::endl;
			for (int i = 0; i < N_DETECTORS; i++)
				cout << i << ") " << detecNames[i] << std::endl;
			cout << "===========" << std::endl;
			cin >> type;
		}

		std::vector<cv::String> files;
		files.push_back("files/panorama/out_1.jpg");
		files.push_back("files/panorama/out_2.jpg");
		files.push_back("files/panorama/out_3.jpg");
		files.push_back("files/panorama/out_4.jpg");
		files.push_back("files/panorama/out_5.jpg");

		for (int i = 0; i < N_DETECTORS; i++) {
			if ((!manual || (manual && i == type)) && !((i < 3) && matType == 2)) {
				Mat pan = imread(files[0]);

				resize(pan, pan, Size(WIDTH, HIEGHT));

				clock_t begin = clock();
				for (int j = 1; j < int(files.size()); j++) {

					Mat im_2 = imread(files[j]);
					resize(im_2, im_2, Size(WIDTH, HIEGHT));

					pan = panorama(pan, im_2, false, detectors[i], matchers[matType]);

				}
				clock_t end = clock();
				cout << "Tiempo de CPU para " << detecNames[i] << " "
					 <<  double(end - begin) / CLOCKS_PER_SEC << " segundos" << endl;
				imwrite("files/outs/panorama_" + detecNames[i] + "_" + matNames[matType] + ".jpg",pan);

			}
		}

		std::cout << "Fin" << std::endl;


	}

	else if (strcmp(argv[1], "2") == 0){	 	// @suppress("Invalid arguments")
		bool poster = false, proceso = true, hom = false;
		cout << "Formando panorama con 5 fotos, �ver emparejamientos? (1 -> si/ 0 -> no): ";
		cin >> hom;

		if (!hom) {
			cout << "Ver proceso? (1 -> si/ 0 -> no): ";
			cin >> proceso;
		}

		cout << "Poster(0) o escena (1)?:  ";
		cin >> poster;

		std::list<cv::String> files;

		if (poster == 1) {
			files.push_back("files/panorama/out_1.jpg");
			files.push_back("files/panorama/out_2.jpg");
			files.push_back("files/panorama/out_3.jpg");
			files.push_back("files/panorama/out_4.jpg");
			files.push_back("files/panorama/out_5.jpg");
		}
		else {
			files.push_back("files/poster/poster_1.jpg");
			files.push_back("files/poster/poster_2.jpg");
			files.push_back("files/poster/poster_3.jpg");
			files.push_back("files/poster/poster_4.jpg");
		}

		// Read in the image.
		Mat im_1 = imread(files.back());
		files.pop_back();

		resize(im_1, im_1, Size(WIDTH, HIEGHT));

		int i = 1;

		for (cv::String file : files) {
			i++;

			Mat im_2 = imread(file);
			resize(im_2, im_2, Size(WIDTH, HIEGHT));

			clock_t begin = clock();
			im_1 = panorama(im_1, im_2, hom, xfeatures2d::SURF::create(), DescriptorMatcher::FLANNBASED);
			clock_t end = clock();

			cout << "Tiempo de CPU para " << i << " im�genes: " <<  double(end - begin) / CLOCKS_PER_SEC << " segundos" << endl;

			if (proceso) {
				namedWindow("Imagen panor�mica", 0);
				imshow("Imagen panor�mica", im_1);

				namedWindow("Imagen a�adida", 0);
				imshow("Imagen a�adida", im_2);
				waitKey(0);
			}


		}


		std::cout << "Fin" << std::endl;

		destroyAllWindows();

		imwrite("files/outs/result.jpg", im_1);
	}
	else if(strcmp(argv[1], "3") == 0) {

		int manual;
		cout << "En vivo o manual (tecla para capturar)? (1 -> vivo/ 0 -> manual): ";
		cin >> manual;

		Mat i1,frame, showframe;
		namedWindow("Camara",1);

		VideoCapture cap(0);

		cout << "Presionar INTRO para capturar la primera imagen" << endl;

		//namedWindow("Panorama", 1);

		while(true){
			cap >> frame;
			flip(frame,frame,1);
			imshow("Camara",frame);
			if(waitKey(30) == 13){
				break;
			}
		}

		cap >> i1;
		flip(i1,i1,1);

		if (manual == 0)
			cout << "Presionar INTRO para capturar imagen" << endl;
		cout << "Presionar ESCAPE para terminar" << endl;

		bool *shot = new bool;
		bool *end = new bool;
		*shot = false;
		*end = false;

		std::thread timer;
		if (manual == 1)
			timer = std::thread(cameraTimer, shot, end);

		while(!*end) {
			cap >> frame;
			flip(frame,frame,1);
			imshow("Camara",frame);

			namedWindow("Panorama", 1);

			resize(i1, showframe, Size(WIDTH, HIEGHT));
			imshow("Panorama", showframe);
			//imshow("Panorama", i1);

			int wait = waitKey(20);

			// AUTO or ENTER
			if(wait == 13 && manual == 0){
				*shot = true;
			}
			// ESC
			else if(wait == 27) {
				*end = true;
				*shot = false;
			}

			if(*shot) {
				cout << "Imagen tomada" << endl;
				cap >> frame;
				flip(frame,frame,1);
				*shot = false;

				//Using akaze, the fastest detector probed and FALNN, fastest matcher
				i1 = panorama(i1, frame, false, cv::AKAZE::create(), DescriptorMatcher::FLANNBASED);
			}


		}

		if (manual == 1)
			timer.join();

		destroyWindow("Camara");
		cap.release();

		imwrite("files/out/panorama_camara.jpg", i1);
		namedWindow("Panorama", 1);

		resize(i1, showframe, Size(WIDTH, HIEGHT));
		imshow("Panorama", showframe);
		//imshow("Panorama",i1);

		waitKey(0);
		destroyAllWindows();

	}

	else {
		help();
	}

    return 0;
}

#else
int main()
{
    std::cout << "This program needs the xfeatures2d contrib module to run." << std::endl;
    return 0;
}
#endif
