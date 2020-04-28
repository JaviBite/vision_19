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
	else if (strcmp(argv[1], "2") == 0) { 	// @suppress("Invalid arguments")

		int width = 640 * 3;
		int height = 480 * 3;

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

		translateImg(im_1, width/2, height/2); // 2000 is usual

		namedWindow("translated 1st image", 0);
		imshow("translated 1st image", im_1);
		waitKey(0);

		for (cv::String file : files) {
			Mat im_2 = imread(file);
			resize(im_2, im_2, Size(width, height));

			warp_crops(im_1, im_2);

			namedWindow("translated 1st image", 0);
			imshow("translated 1st image", im_1);
			waitKey(0);

		}


		std::cout << "Fin" << std::endl;
		waitKey(0);

		imwrite("result.jpg", im_1);

		if (argc > 2 && strcmp(argv[2], "horizontal") == 0) { 	 	// @suppress("Invalid arguments")

		}

	}
	else if (strcmp(argv[1], "3") == 0){	 	// @suppress("Invalid arguments")

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
