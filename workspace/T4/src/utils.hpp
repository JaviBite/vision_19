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

void checkImg(Mat img) {
	if(!img.data or img.empty()) {                          // Check for invalid input
		cout <<  "Could not open or find the image" << std::endl ;
		exit(-1);
	}
}

void translateImg(Mat &img, int offsetx, int offsety) {
	Mat T = (Mat_<double>(2, 3) << 1, 0, offsetx,
								   0, 1, offsety);

	warpAffine(img, img, T, Size(img.cols * 3, img.rows * 3)); // 3,4 is usual
}

Mat panorama(Mat &i1, Mat &i2, bool info,
		cv::Ptr<Feature2D> detector = xfeatures2d::SURF::create(),
		int matcherType = DescriptorMatcher::FLANNBASED){
	Mat i1g, i2g, d1, d2, i_matches, inliers, result;
	cvtColor(i1,i1g,CV_BGR2GRAY);
	cvtColor(i2,i2g,CV_BGR2GRAY);
	vector< KeyPoint > kp1, kp2;
	vector< vector <DMatch> > matches;
	vector < DMatch > filtrados, ransac;
	vector< Point2f > obj, scene;

	/* Detectar puntos de interes */
	//SurfFeatureDetector detector(400);
	detector->clear();
	detector->detectAndCompute(i1g, cv::noArray(), kp1, d1);
	detector->detectAndCompute(i2g, cv::noArray(), kp2, d2);
//	detector->detect(i1g, kp1);
//	detector->detect(i2g, kp2);
//
//
//	/* Obtiene los descriptores de cada punto de interes */
//	//Ptr<SURF> extractor = detector;
//	//extractor->clear();
//	detector->compute(i1g,kp1,d1);
//	detector->compute(i2g,kp2,d2);

	if(d1.type() != CV_32F)
	    d1.convertTo(d1, CV_32F);

	if(d2.type() != CV_32F)
	    d2.convertTo(d2, CV_32F);

	/* Realiza los emparejamientos, con filtro de ratio */
	//Ptr<DescriptorMatcher> matcher =  DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE_SL2);
	Ptr<DescriptorMatcher> matcher =  DescriptorMatcher::create(matcherType);

	if (!d1.empty() && !d2.empty())
		matcher->knnMatch(d1,d2,matches,2);

	for(unsigned int i = 0; i < matches.size(); i++){
		/* Aplica el filtro de ratio */
		if(matches[i][0].distance < 0.5*matches[i][1].distance){
			filtrados.push_back(matches[i][0]);
		}
	}

	// Good matches
	if(filtrados.size()>10){

		for(unsigned int i = 0; i < filtrados.size(); i++){
			obj.push_back(kp1[ filtrados[i].queryIdx ].pt);
			scene.push_back(kp2[ filtrados[i].trainIdx ].pt);
		}

		Mat mask;
		Mat homography = findHomography(obj,scene,CV_RANSAC,3,mask);

		/* Calculo de los inliers */
		for(unsigned int i = 0; i < filtrados.size(); i++){
			if((int)mask.at<uchar>(i,0) == 1){
				ransac.push_back(filtrados[i]);
			}
		}

		vector <Point2f> corners;

		corners.push_back(Point2f(0,0));
		corners.push_back(Point2f(0,i1g.rows));
		corners.push_back(Point2f(i1g.cols,0));
		corners.push_back(Point2f(i1g.cols,i1g.rows));

		vector < Point2f > scene_corners;
		perspectiveTransform(corners, scene_corners, homography);

		float maxCols(0),maxRows(0),minCols(0),minRows(0);

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

		Mat euclid = Mat::eye(3,3,homography.type());
		euclid.at<double>(0,2) = -minCols;
		euclid.at<double>(1,2) = -minRows;

		if(info){
			/* Muestra los emparejamientos */
			namedWindow("Emparejamientos filtrados",1);
			drawMatches(i1g,kp1,i2g,kp2,filtrados,i_matches);
			resize(i_matches, i_matches, Size(600 * 2, 600));
			imshow("Emparejamientos filtrados", i_matches);

			/* Muestra los inliers
			namedWindow("Inliers",1);
			drawMatches(i1g,kp1,i2g,kp2,ransac,inliers);
			imshow("Inliers", inliers);
			waitKey(0);*/
		}

		#if 1

		//Mask of the image to be combined so you can get resulting mask
		Mat mask1, mask2;
		cv::threshold(i1, mask1, 0, 255, THRESH_BINARY);
		cv::cvtColor(mask1, mask1, cv::COLOR_BGR2GRAY);

		cv::threshold(i2, mask2, 0, 255, THRESH_BINARY);
		cv::cvtColor(mask2, mask2, cv::COLOR_BGR2GRAY);

		Mat i1r, i2r;

		warpPerspective(i2,i2r,euclid,Size(max(i2.cols-minCols,maxCols),max(i2.rows-minRows,maxRows)),INTER_LINEAR,BORDER_REFLECT_101,Scalar(155,155,155));
		warpPerspective(i1,i1r,euclid*homography,Size(max(i2.cols-minCols,maxCols),max(i2.rows-minRows,maxRows)),INTER_LINEAR,BORDER_REFLECT_101,Scalar(155,155,155));

		warpPerspective(mask2,mask2,euclid,Size(max(mask2.cols-minCols,maxCols),max(mask2.rows-minRows,maxRows)),INTER_LINEAR,BORDER_CONSTANT,0);
		warpPerspective(mask1,mask1,euclid*homography,Size(max(mask2.cols-minCols,maxCols),max(mask2.rows-minRows,maxRows)),INTER_LINEAR,BORDER_CONSTANT,0);

//		GaussianBlur(mask1, mask1, Size(177,177), 10, 0, BORDER_DEFAULT );
//		GaussianBlur(mask2, mask2, Size(177,177), 10, 0, BORDER_DEFAULT );

//		Mat mask11, mask22;
//		resize(i2r, mask22, Size(600, 600));
//		resize(i1r, mask11, Size(600, 600));
//		imshow("TEST", mask11);
//		imshow("TEST2", mask22);
//
//		waitKey();
//		destroyWindow("TEST");
//		destroyWindow("TEST2");
//
//		resize(mask2, mask22, Size(600, 600));
//		resize(mask1, mask11, Size(600, 600));
//		imshow("TEST", mask11);
//		imshow("TEST2", mask22);
//
//		waitKey();
//		destroyWindow("TEST");
//		destroyWindow("TEST2");

		//create blender
		detail::FeatherBlender blender(0.02);
		//detail::MultiBandBlender blender(false, 5);
		//feed images and the mask areas to blend
		blender.prepare(Rect(0, 0, max(i2.cols-minCols,maxCols), max(i2.rows-minRows,maxRows)));

		i1r.convertTo(i1r, CV_16SC3);
		i2r.convertTo(i2r, CV_16SC3);

		blender.feed(i1r, mask1, Point2f (0,0));
		blender.feed(i2r, mask2, Point2f (0,0));
		//prepare resulting size of image
		Mat result_s, result_mask;
		//blend
		blender.blend(result_s, result_mask);

		result_s.convertTo(result_s, (result_s.type() / 8) * 8);

		return result_s;

		#else

			warpPerspective(i2,result,euclid,Size(max(i2.cols-minCols,maxCols),max(i2.rows-minRows,maxRows)),INTER_LINEAR,BORDER_CONSTANT,0);
			warpPerspective(i1,result,euclid*homography,Size(max(i2.cols-minCols,maxCols),max(i2.rows-minRows,maxRows)),INTER_LINEAR,BORDER_TRANSPARENT,0);

			return result;

		#endif
	}
	else{
		cerr << "Las imágenes no se han podido asociar" << endl;
		return i2;
	}
}
