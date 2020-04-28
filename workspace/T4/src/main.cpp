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
		    drawKeypoints( src, keypoints, img_keypoints ); // @suppress("Invalid arguments")


		    //-- Show detected (drawn) keypoints
		    imshow("SURF Keypoints", img_keypoints );
		    waitKey();
		    return 0;
		}


	}
	else if (strcmp(argv[1], "2") == 0) { 	// @suppress("Invalid arguments")

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
