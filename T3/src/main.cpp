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
		image = imread("files/imagenesT3/poster.pgm", CV_LOAD_IMAGE_GRAYSCALE);   // Read the file
		checkImg(image);

		imshow("Display window", image );                   // Show our image inside it.
		waitKey(0);
		cvDestroyWindow("Display window");

	}
	else if (strcmp(argv[1], "2") == 0) {

		}
	else {
		std::cout << "Usage: " << argv[0] << " <1 | 2> " << std::endl;
		std::cout << "\t 1 : Gradiente, módulo y orientación" << std::endl;
		std::cout << "\t 2 : Detección del punto central" << std::endl;
	}
    return 0;
}
