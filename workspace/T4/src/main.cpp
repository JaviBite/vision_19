#include <opencv2/opencv.hpp>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include <iostream>
#include <stdio.h>
// #include <windows.h> // For Sleep
#include <iostream>
#include <stdio.h>
#include <windows.h> // For Sleep
#include "utils.hpp"
#include <fstream>
#include <string>
#include <vector>
#include <list>
// #include <iomanip>

#define HAVE_OPENCV_XFEATURES2D

#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/xfeatures2d.hpp"

using namespace cv;
using namespace cv::xfeatures2d;


void help() {
	std::cout << "Usage: " << "program " << "<params>" << std::endl;
	exit(-1);
}

int main(int argc, char *argv[]) {

	if (strcmp(argv[1], "1") == 0) { 	// @suppress("Invalid arguments")
		std::vector<cv::String> files;
		std::vector<std::vector<KeyPoint>> keypoints_lists;

		files.push_back("files/panorama/out_1.jpg");
		files.push_back("files/panorama/out_2.jpg");
		files.push_back("files/panorama/out_3.jpg");

		for (cv::String file : files) {
			Mat src, src_original, dst, color_dst;
			src_original = cv::imread(file, 0);	// @suppress("Invalid arguments")

			Size size(960,720);
			resize(src_original,src,size);
			checkImg(src_original);


			//cv::cvtColor(src, src, CV_BGR2GRAY);

		    //-- Step 1: Detect the keypoints using SURF Detector
		    int minHessian = 400;
		    Ptr<SURF> detector = SURF::create( minHessian );
		    std::vector<KeyPoint> keypoints;
		    detector->detect( src, keypoints ); // @suppress("Invalid arguments")

		    //-- Draw keypoints
		    Mat img_keypoints;
		    cv::drawKeypoints( src, keypoints, img_keypoints ); // @suppress("Invalid arguments")

		    //-- Show detected (drawn) keypoints
		    imshow("SURF Keypoints", img_keypoints );
		    waitKey();
		}

//		for (std::vector<KeyPoint> keyPointList : keypoints_lists) {
//			for (std::vector<KeyPoint> keyPointList2 : keypoints_lists) {
//				if (&keyPointList2 != &keyPointList) {
//					Mat homography;
//					homography = cv::findHomography(keyPointList, keyPointList2, CV_RANSAC, 3);
//				}
//			}
//		}


	}

	else if (strcmp(argv[1], "2") == 0){	 	// @suppress("Invalid arguments")
		int hom;
		cout << "Formando panorama con 5 fotos, ¿ver emparejamientos? (1 -> si/ 0 -> no): ";
		cin >> hom;

		if (hom != 1 && hom != 0) hom = 0;

		int width = 960;
		int height = 720;

		std::list<cv::String> files;
		files.push_back("files/panorama/out_1.jpg");
		files.push_back("files/panorama/out_2.jpg");
		files.push_back("files/panorama/out_3.jpg");
		files.push_back("files/panorama/out_4.jpg");
		files.push_back("files/panorama/out_5.jpg");

		// Read in the image.
		Mat im_1 = imread(files.back());
		files.pop_back();

		resize(im_1, im_1, Size(width, height));

		int i = 1;

		for (cv::String file : files) {
			i++;

			Mat im_2 = imread(file);
			resize(im_2, im_2, Size(width, height));

			clock_t begin = clock();
			im_1 = panorama(im_2, im_1, hom);
			clock_t end = clock();

			cout << "Tiempo de CPU para " << i << " imágenes: " <<  double(end - begin) / CLOCKS_PER_SEC << " segundos" << endl;

			namedWindow("Imagen panorámica", 0);
			imshow("Imagen panorámica", im_1);

			namedWindow("Imagen añadida", 0);
			imshow("Imagen añadida", im_2);
			waitKey(0);


		}


		std::cout << "Fin" << std::endl;

		imwrite("result.jpg", im_1);
	}
	else if(strcmp(argv[1], "3") == 0) {

		Mat i1,frame;
		namedWindow("Camara",1);

		VideoCapture cap(0);

		cout << "Presionar INTRO para capturar la primera imagen" << endl;

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

		cout << "Presionar INTRO para capturar imagen" << endl;
		cout << "Presionar ESCAPE para terminar" << endl;
		while(true){
			cap >> frame;
			flip(frame,frame,1);
			imshow("Camara",frame);
			int wait = waitKey(20);
			if(wait == 13){
				cout << "Imagen tomada" << endl;
				cap >> frame;
				flip(frame,frame,1);
				i1 = panorama(frame,i1,2);
			}
			if(wait == 27){
				break;
			}
		}
		destroyAllWindows();
		cap.release();

		imwrite("panorama_camara.jpg",i1);
		imshow("Panorama", i1);
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
    std::cout << "This program needs the xfeatures2d contrib module to be run." << std::endl;
    return 0;
}
#endif
