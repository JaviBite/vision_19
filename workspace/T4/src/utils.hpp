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
#define CV_RANSAC RANSAC
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

void warp_crops(Mat& im_1, const Mat& im_2)
{
	cv::Ptr<Feature2D> f2d = xfeatures2d::SURF::create();


	// Step 1: Detect the keypoints:
	std::vector<KeyPoint> keypoints_1, keypoints_2;
	f2d->detect( im_1, keypoints_1 );
	f2d->detect( im_2, keypoints_2 );

	// Step 2: Calculate descriptors (feature vectors)
	Mat descriptors_1, descriptors_2;
	f2d->compute( im_1, keypoints_1, descriptors_1 );
	f2d->compute( im_2, keypoints_2, descriptors_2 );

	// Step 3: Matching descriptor vectors using BFMatcher :
	FlannBasedMatcher matcher;
	std::vector< DMatch > matches;
	matcher.match( descriptors_1, descriptors_2, matches );

	// Keep best matches only to have a nice drawing.
	// We sort distance between descriptor matches
	Mat index;
	int nbMatch = int(matches.size());
	Mat tab(nbMatch, 1, CV_32F);
	for (int i = 0; i < nbMatch; i++)
		tab.at<float>(i, 0) = matches[i].distance;
	sortIdx(tab, index, SORT_EVERY_COLUMN + SORT_ASCENDING);
	vector<DMatch> bestMatches;

	for (int i = 0; i < 200; i++)
		bestMatches.push_back(matches[index.at < int > (i, 0)]);


	// 1st image is the destination image and the 2nd image is the src image
	std::vector<Point2f> dst_pts;                   //1st
	std::vector<Point2f> source_pts;                //2nd

	for (vector<DMatch>::iterator it = bestMatches.begin(); it != bestMatches.end(); ++it) {
		cout << it->queryIdx << "\t" <<  it->trainIdx << "\t"  <<  it->distance << "\n";
		//-- Get the keypoints from the good matches
		dst_pts.push_back( keypoints_1[ it->queryIdx ].pt );
		source_pts.push_back( keypoints_2[ it->trainIdx ].pt );
	}

	 Mat img_matches;
	 drawMatches( im_1, keypoints_1, im_2, keypoints_2,
	           bestMatches, img_matches, Scalar::all(-1), Scalar::all(-1),
	           vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

	 //-- Show detected matches
	 resize(img_matches, img_matches, Size(500, 500));
	 imshow( "Good_Matches.jpg", img_matches );



	Mat H = findHomography( source_pts, dst_pts, CV_RANSAC );
	Mat wim_2;
	cout << "Homo = " << H << endl;

	warpPerspective(im_2, wim_2, H, im_1.size());

	for (int i = 0; i < im_1.cols; i++)
		for (int j = 0; j < im_1.rows; j++) {
			Vec3b color_im1 = im_1.at<Vec3b>(Point(i, j));
			Vec3b color_im2 = wim_2.at<Vec3b>(Point(i, j));
			if (norm(color_im1) == 0)
				im_1.at<Vec3b>(Point(i, j)) = color_im2;

		}

}

Mat panorama(Mat &im_1, Mat &im_2, int homografia){
	Mat im_1aux, im_2aux, inliers, result;

	cvtColor(im_1,im_1aux,CV_BGR2GRAY);
	cvtColor(im_2,im_2aux,CV_BGR2GRAY);

	vector< KeyPoint > kp_1, kp_2;

	// Detección de puntos de interés
	cv::Ptr<Feature2D> detector = xfeatures2d::SURF::create();
	detector->detect( im_1aux, kp_1 );
	detector->detect( im_2aux, kp_2 );

	Mat d1, d2;

	// Calculo de los descriptores
	Ptr<SURF> extractor = SURF::create();
	extractor->compute(im_1aux,kp_1,d1);
	extractor->compute(im_2aux,kp_2,d2);

	vector< vector <DMatch> > emparejamientos;
	// Realización de emparejamientos
	BFMatcher matcher(NORM_L2);
	matcher.knnMatch(d1,d2,emparejamientos,2);

	vector < DMatch > filtrados, ransac;
	vector< Point2f > obj, escena;

	for(unsigned int i = 0; i < emparejamientos.size(); i++){

		// Aplicar el filtro de ratio
		if(emparejamientos[i][0].distance < 0.5*emparejamientos[i][1].distance){
			filtrados.push_back(emparejamientos[i][0]);
		}
	}
	if(filtrados.size()>10){

		for(unsigned int i = 0; i < filtrados.size(); i++){
			obj.push_back(kp_1[ filtrados[i].queryIdx ].pt);
			escena.push_back(kp_2[ filtrados[i].trainIdx ].pt);
		}

		Mat mask;
		Mat homography = findHomography(obj,escena,CV_RANSAC,3,mask);

		// Cálculo de inliners
		for(unsigned int i = 0; i < filtrados.size(); i++){
			if((int)mask.at<uchar>(i,0) == 1){
				ransac.push_back(filtrados[i]);
			}
		}

		vector <Point2f> corners;

		corners.push_back(Point2f(0,0));
		corners.push_back(Point2f(0,im_1aux.rows));
		corners.push_back(Point2f(im_1aux.cols,0));
		corners.push_back(Point2f(im_1aux.cols,im_1aux.rows));

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

		Mat i_emparejamientos;

		if(homografia == 1){
			// Nostrar emparejamientos
			namedWindow("Emparejamientos filtrados",1);
			drawMatches(im_1aux,kp_1,im_2aux,kp_2,filtrados,i_emparejamientos);
			resize(i_emparejamientos, i_emparejamientos, Size(600 * 2, 600));
			imshow("Emparejamientos filtrados", i_emparejamientos);

			/*// Mostrar inliners
			namedWindow("Inliers",1);
			drawemparejamientos(im_1aux,kp_1,im_2aux,kp_2,ransac,inliers);
			imshow("Inliers", inliers);
			waitKey(0);*/
		}

		#if 1

		//Mask of the image to be combined so you can get resulting mask
		Mat mask1, mask2;
		cv::threshold(im_1, mask1, 0, 255, THRESH_BINARY);
		cv::cvtColor(mask1, mask1, cv::COLOR_BGR2GRAY);

		cv::threshold(im_2, mask2, 0, 255, THRESH_BINARY);
		cv::cvtColor(mask2, mask2, cv::COLOR_BGR2GRAY);

		Mat im_1r, im_2r;

		warpPerspective(im_2,im_2r,euclid,Size(max(im_2.cols-minCols,maxCols),max(im_2.rows-minRows,maxRows)),INTER_LINEAR,BORDER_REFLECT_101,Scalar(155,155,155));
		warpPerspective(im_1,im_1r,euclid*homography,Size(max(im_2.cols-minCols,maxCols),max(im_2.rows-minRows,maxRows)),INTER_LINEAR,BORDER_REFLECT_101,Scalar(155,155,155));

		warpPerspective(mask2,mask2,euclid,Size(max(mask2.cols-minCols,maxCols),max(mask2.rows-minRows,maxRows)),INTER_LINEAR,BORDER_CONSTANT,0);
		warpPerspective(mask1,mask1,euclid*homography,Size(max(mask2.cols-minCols,maxCols),max(mask2.rows-minRows,maxRows)),INTER_LINEAR,BORDER_CONSTANT,0);

//		GaussianBlur(mask1, mask1, Size(177,177), 10, 0, BORDER_DEFAULT );
//		GaussianBlur(mask2, mask2, Size(177,177), 10, 0, BORDER_DEFAULT );

//		Mat mask11, mask22;
//		resize(im_2r, mask22, Size(600, 600));
//		resize(im_1r, mask11, Size(600, 600));
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
		blender.prepare(Rect(0, 0, max(im_2.cols-minCols,maxCols), max(im_2.rows-minRows,maxRows)));

		im_1r.convertTo(im_1r, CV_16SC3);
		im_2r.convertTo(im_2r, CV_16SC3);

		blender.feed(im_1r, mask1, Point2f (0,0));
		blender.feed(im_2r, mask2, Point2f (0,0));
		//prepare resulting size of image
		Mat result_s, result_mask;
		//blend
		blender.blend(result_s, result_mask);

		result_s.convertTo(result_s, (result_s.type() / 8) * 8);

		return result_s;

		#else

			warpPerspective(im_2,result,euclid,Size(max(im_2.cols-minCols,maxCols),max(im_2.rows-minRows,maxRows)),INTER_LINEAR,BORDER_CONSTANT,0);
			warpPerspective(im_1,result,euclid*homography,Size(max(im_2.cols-minCols,maxCols),max(im_2.rows-minRows,maxRows)),INTER_LINEAR,BORDER_TRANSPARENT,0);

			return result;

		#endif
	}
	else{
		cerr << "Las imágenes no se han podido asociar" << endl;
		return im_2;
	}
}
