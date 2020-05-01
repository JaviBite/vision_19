#pragma once

#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <functional>
#include <cmath>
#include <vector>
#include <fstream>
#include <algorithm>
#include <numeric>
#include "opencv2/xfeatures2d.hpp"

#define CV_BGR2YCrCb COLOR_BGR2YCrCb
#define CV_YCrCb2BGR COLOR_YCrCb2BGR
#define CV_RGB2GRAY COLOR_RGB2GRAY
#define CV_BGR2GRAY COLOR_BGR2GRAY
#define CV_GRAY2BGR COLOR_GRAY2BGR
#define CV_BGR2HSV COLOR_BGR2HSV
#define CV_HSV2BGR COLOR_HSV2BGR
#define CV_WINDOW_AUTOSIZE WINDOW_AUTOSIZE
#define CV_TERMCRIT_ITER 1
#define CV_LOAD_IMAGE_COLOR IMREAD_COLOR
#define CV_LOAD_IMAGE_GRAYSCALE IMREAD_GRAYSCALE
#define cvDestroyWindow destroyWindow
#define CV_RETR_TREE RETR_TREE
#define CV_CHAIN_APPROX_NONE CHAIN_APPROX_NONE

#ifdef CV_RANSAC
#else
	#define CV_RANSAC RANSAC
#endif

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;


// HYPER PARAMS
#define FILTER_RATIO 0.5F
#define BLENDING 1
#define GOOD_MATCHES 20

// Convert type (matrix) to string
string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

//Check if the image mat is valid
void checkImg(Mat img) {
	if(!img.data or img.empty()) {                          // Check for invalid input
		cout <<  "Could not open or find the image" << std::endl ;
		exit(-1);
	}
}

// Join the im_1 and im_2 creating a panorama image
Mat panorama(Mat &im_1, Mat &im_2, bool showHomograpy = false,
		cv::Ptr<Feature2D> detector = xfeatures2d::SURF::create(),
		int matcherType = DescriptorMatcher::FLANNBASED){
	Mat im_1aux, im_2aux, inliers, result;

	// Create a grayscale images from the source
	cvtColor(im_1,im_1aux,CV_BGR2GRAY);
	cvtColor(im_2,im_2aux,CV_BGR2GRAY);

	// Variables
	vector< KeyPoint > kp_1, kp_2;
	Mat detects1, detects2;
	vector< vector <DMatch> > matches;
	vector < DMatch > filtrados;
	vector< Point2f > obj, escena;

	// Detecting interest points ("corners")
	detector->clear();
	detector->detectAndCompute(im_1aux, cv::noArray(), kp_1, detects1);	// @suppress("Invalid arguments")
	detector->detectAndCompute(im_2aux, cv::noArray(), kp_2, detects2);	// @suppress("Invalid arguments")

	// Convert type for matchers like FLANN
	if(detects1.type() != CV_32F)
	    detects1.convertTo(detects1, CV_32F);

	if(detects2.type() != CV_32F)
	    detects2.convertTo(detects2, CV_32F);

	// Pair the matches and filter them with a ratio
	Ptr<DescriptorMatcher> matcher =  DescriptorMatcher::create(matcherType);

	if (!detects1.empty() && !detects2.empty())
		matcher->knnMatch(detects1, detects2, matches, 2, cv::noArray(), false); //K = 2 //@suppress("Invalid arguments")

	for (int i = 0; i < int(matches.size()); i++) {
		// Filter ratio
		if(matches[i][0].distance < FILTER_RATIO * matches[i][1].distance) {
			filtrados.push_back(matches[i][0]);
		}
	}

	// Good matches
	if(filtrados.size() > GOOD_MATCHES) {
		for(unsigned int i = 0; i < filtrados.size(); i++) {
			obj.push_back(kp_1[filtrados[i].queryIdx].pt);
			escena.push_back(kp_2[filtrados[i].trainIdx].pt);
		}

		Mat mask;

		// Homograpy between obj (im_1) and scene (im_2)
		Mat homography = findHomography(obj, escena, CV_RANSAC, 3, mask);		// @suppress("Invalid arguments")

		// Corners for image translation/transformation
		vector <Point2f> corners;
		corners.push_back(Point2f(0, 0));
		corners.push_back(Point2f(0, im_1aux.rows));
		corners.push_back(Point2f(im_1aux.cols, 0));
		corners.push_back(Point2f(im_1aux.cols, im_1aux.rows));

		vector < Point2f > scene_corners;
		perspectiveTransform(corners, scene_corners, homography);		// @suppress("Invalid arguments")

		float maxCols = 0, maxRows = 0;
		float minCols = 0, minRows = 0;

		// Check the new image corners
		for(unsigned int i = 0; i < scene_corners.size(); i++){
			if(maxRows < scene_corners.at(i).y){
				maxRows = scene_corners.at(i).y;
			}
			if(minRows > scene_corners.at(i).y){
				minRows = scene_corners.at(i).y;
			}
			if(maxCols < scene_corners.at(i).x){
				maxCols = scene_corners.at(i).x;
			}
			if(minCols > scene_corners.at(i).x){
				minCols = scene_corners.at(i).x;
			}
		}

		//New image size (panorama)
		Size panSize = Size(max(im_2.cols-minCols,maxCols),max(im_2.rows-minRows,maxRows));

		// The transformation if the new image needs more space
		Mat euclid = Mat::eye(3,3,homography.type());
		euclid.at<double>(0,2) = -minCols;
		euclid.at<double>(1,2) = -minRows;

		Mat i_emparejamientos;
		if(showHomograpy) {
			// Mostrar emparejamientos
			namedWindow("Emparejamientos filtrados", 1);
			drawMatches(im_1aux, kp_1, im_2aux, kp_2, filtrados, i_emparejamientos);		// @suppress("Invalid arguments")
			resize(i_emparejamientos, i_emparejamientos, Size(600 * 2, 600));
			imshow("Emparejamientos filtrados", i_emparejamientos);
		}

		#if BLENDING

		//Mask of the image to be blend
		Mat mask1, mask2;
		cv::threshold(im_1, mask1, 0, 255, THRESH_BINARY);
		cv::cvtColor(mask1, mask1, cv::COLOR_BGR2GRAY);

		cv::threshold(im_2, mask2, 0, 255, THRESH_BINARY);
		cv::cvtColor(mask2, mask2, cv::COLOR_BGR2GRAY);

		// Transformed images (result)
		Mat im_1r, im_2r;

		// Transform the images
		warpPerspective(im_1, im_1r, euclid * homography, panSize, INTER_LINEAR, BORDER_REFLECT_101);
		warpPerspective(im_2, im_2r, euclid, panSize, INTER_LINEAR, BORDER_REFLECT_101);

		// Transform the masks for blending
		warpPerspective(mask1, mask1, euclid * homography, panSize, INTER_LINEAR, BORDER_CONSTANT);
		warpPerspective(mask2, mask2, euclid, panSize, INTER_LINEAR, BORDER_CONSTANT);

		//Create blender
		detail::FeatherBlender blender(0.02);	// Sharpness = 0.02

		// Multiband dont work, it creates artifacts
		//detail::MultiBandBlender blender(false, 5);

		// Feed images and the mask areas to blend
		blender.prepare(Rect(0, 0, panSize.width, panSize.height));

		im_1r.convertTo(im_1r, CV_16SC3);
		im_2r.convertTo(im_2r, CV_16SC3);

		blender.feed(im_1r, mask1, Point2f (0,0));
		blender.feed(im_2r, mask2, Point2f (0,0));

		// Prepare resulting size of image
		Mat result_s, result_mask;

		//Blend
		blender.blend(result_s, result_mask);

		// Convert the result to something showable
		result_s.convertTo(result_s, (result_s.type() / 8) * 8);
		result = result_s;

		#else

			warpPerspective(im_2,result,euclid,Size(max(im_2.cols-minCols,maxCols),max(im_2.rows-minRows,maxRows)),INTER_LINEAR,BORDER_CONSTANT,0);
			warpPerspective(im_1,result,euclid*homography,Size(max(im_2.cols-minCols,maxCols),max(im_2.rows-minRows,maxRows)),INTER_LINEAR,BORDER_TRANSPARENT,0);



		#endif

		return result;
	}
	// Less than GOOD_MATCHES
	else{
		std::cerr << "No ha sido posible combinar las imágenes" << std::endl;
		return im_2;
	}
}
