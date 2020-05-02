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

// For sleep
#if defined(__WIN32__) || defined(_WIN32) || defined(WIN32) || defined(__WINDOWS__) || defined(__TOS_WIN__)

  #include <windows.h>

  inline void delay( unsigned long ms )
    {
    Sleep( ms );
    }

#else  /* presume POSIX */

  #include <unistd.h>

  inline void delay( unsigned long ms )
    {
    usleep( ms * 1000 );
    }

#endif

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

// Join the new_image and origin_image creating a panorama image
Mat panorama(Mat &origin_image, Mat &new_image, bool showHomograpy = false,
		cv::Ptr<Feature2D> detector = xfeatures2d::SURF::create(),
		cv::DescriptorMatcher::MatcherType matcherType = DescriptorMatcher::FLANNBASED){
	Mat new_imageAux, origin_imageAux, inliers, result;

	// Create a grayscale images from the source
	cvtColor(new_image, new_imageAux,CV_BGR2GRAY);
	cvtColor(origin_image,origin_imageAux,CV_BGR2GRAY);

	// Variables
	vector< KeyPoint > new_img_keyp, orig_img_keyp;
	Mat new_img_detects, orig_img_detects;
	vector< vector <DMatch> > matches;
	vector < DMatch > filtrados;
	vector< Point2f > obj, escena;

	// Detecting interest points ("corners")
	detector->clear();
	detector->detectAndCompute(new_imageAux, cv::noArray(), new_img_keyp, new_img_detects);	// @suppress("Invalid arguments")
	detector->detectAndCompute(origin_imageAux, cv::noArray(), orig_img_keyp, orig_img_detects);	// @suppress("Invalid arguments")

	// Convert type for matchers like FLANN
	if(new_img_detects.type() != CV_32F)
	    new_img_detects.convertTo(new_img_detects, CV_32F);

	if(orig_img_detects.type() != CV_32F)
	    orig_img_detects.convertTo(orig_img_detects, CV_32F);

	// Pair the matches and filter them with a ratio
	Ptr<DescriptorMatcher> matcher =  DescriptorMatcher::create(matcherType);

	if (!new_img_detects.empty() && !orig_img_detects.empty())
		matcher->knnMatch(new_img_detects, orig_img_detects, matches, 2, cv::noArray(), false); //K = 2 //@suppress("Invalid arguments")

	for (int i = 0; i < int(matches.size()); i++) {
		// Filter ratio
		if(matches[i][0].distance < FILTER_RATIO * matches[i][1].distance) {
			filtrados.push_back(matches[i][0]);
		}
	}

	// Good matches
	if(filtrados.size() > GOOD_MATCHES) {
		for(unsigned int i = 0; i < filtrados.size(); i++) {
			obj.push_back(new_img_keyp[filtrados[i].queryIdx].pt);
			escena.push_back(orig_img_keyp[filtrados[i].trainIdx].pt);
		}

		Mat mask;

		// Homograpy between obj (new_image) and scene (origin_image)
		Mat homography = findHomography(obj, escena, CV_RANSAC, 3, mask);		// @suppress("Invalid arguments")

		// Corners for image translation/transformation
		vector <Point2f> corners;
		corners.push_back(Point2f(0, 0));
		corners.push_back(Point2f(0, new_imageAux.rows));
		corners.push_back(Point2f(new_imageAux.cols, 0));
		corners.push_back(Point2f(new_imageAux.cols, new_imageAux.rows));

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
		Size panSize = Size(max(origin_image.cols-minCols, maxCols), max(origin_image.rows-minRows, maxRows));

		// The transformation if the new image needs more space
		Mat euclid = Mat::eye(3, 3, homography.type());
		euclid.at<double>(0,2) = -minCols;
		euclid.at<double>(1,2) = -minRows;

		Mat i_emparejamientos;
		if(showHomograpy) {
			// Mostrar emparejamientos
			namedWindow("Emparejamientos filtrados", 1);
			drawMatches(new_imageAux, new_img_keyp, origin_imageAux, orig_img_keyp, filtrados, i_emparejamientos);		// @suppress("Invalid arguments")
			resize(i_emparejamientos, i_emparejamientos, Size(600 * 2, 600));
			imshow("Emparejamientos filtrados", i_emparejamientos);
		}

		#if BLENDING

		//Mask of the image to be blend
		Mat mask_new, mask_orig;
		cv::threshold(new_image, mask_new, 0, 255, THRESH_BINARY);
		cv::cvtColor(mask_new, mask_new, cv::COLOR_BGR2GRAY);

		cv::threshold(origin_image, mask_orig, 0, 255, THRESH_BINARY);
		cv::cvtColor(mask_orig, mask_orig, cv::COLOR_BGR2GRAY);

		// Transformed images (result)
		Mat new_imageR, origin_imageR;

		// Transform the images
		warpPerspective(new_image, new_imageR, euclid * homography, panSize, INTER_LINEAR, BORDER_REFLECT_101);
		warpPerspective(origin_image, origin_imageR, euclid, panSize, INTER_LINEAR, BORDER_REFLECT_101);

		// Transform the masks for blending
		warpPerspective(mask_new, mask_new, euclid * homography, panSize, INTER_LINEAR, BORDER_CONSTANT);
		warpPerspective(mask_orig, mask_orig, euclid, panSize, INTER_LINEAR, BORDER_CONSTANT);

		//Create blender
		detail::FeatherBlender blender(0.02);	// Sharpness = 0.02

		// Multiband dont work, it creates artifacts
		//detail::MultiBandBlender blender(false, 5);

		// Feed images and the mask areas to blend
		blender.prepare(Rect(0, 0, panSize.width, panSize.height));

		new_imageR.convertTo(new_imageR, CV_16SC3);
		origin_imageR.convertTo(origin_imageR, CV_16SC3);

		blender.feed(new_imageR, mask_new, Point2f (0,0));
		blender.feed(origin_imageR, mask_orig, Point2f (0,0));

		// Prepare resulting size of image
		Mat result_s, result_mask;

		//Blend
		blender.blend(result_s, result_mask);

		// Convert the result to something showable
		result_s.convertTo(result_s, (result_s.type() / 8) * 8);
		result = result_s;

		#else

			warpPerspective(origin_image,result,euclid,Size(max(origin_image.cols-minCols,maxCols),max(origin_image.rows-minRows,maxRows)),INTER_LINEAR,BORDER_CONSTANT,0);
			warpPerspective(new_image,result,euclid*homography,Size(max(origin_image.cols-minCols,maxCols),max(origin_image.rows-minRows,maxRows)),INTER_LINEAR,BORDER_TRANSPARENT,0);



		#endif

		return result;
	}
	// Less than GOOD_MATCHES
	else{
		std::cerr << "No ha sido posible combinar las imágenes" << std::endl;
		return origin_image;
	}
}
