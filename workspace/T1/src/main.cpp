#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <iostream>
#include <stdio.h>
#include <windows.h> // For Sleep
#include "utils.hpp"
#include <fstream>
#include <string>
#define CV_WINDOW_AUTOSIZE WINDOW_AUTOSIZE
using namespace cv;
using namespace std;


int ct = 0;
char tipka;
char filename[100]; // For filename
int  c = 1; // For filename

int main(int, char**)
{
	// VARIABLES
	double contrast = 1;
	int numeroColores = 8;
	int alien_mode = 0;
	int dist_mode = 0;
	int take_mode = 2;
	int gauss_k = 1;
	int gauss_s = 5;
	int pixels = 64;
	int interpolation = INTER_NEAREST;
	float dist_k = 0.5;

    int histSize = 256; //from 0 to 255


	bool contraste = false;
	bool reduccionColores = false;
	bool efectoAlien = false;
	bool distorsion = false;
	bool take_effect = false;
	bool hist_eq = false;
	bool hist_eq_ours = false;
	bool gauss = false;
	bool negative_effect = false;
	bool gray_scale = false;
	bool pixel = false;
	bool k_means = false;

	bool uniform = true;
	bool accumulate = false;
	bool accumulate_hist = false;

    Mat frame;
    //--- INITIALIZE VIDEOCAPTURE
    VideoCapture cap;
    // open the default camera using default API
    cap.open(0);
    // OR advance usage: select any API backend
    int deviceID = 0;             // 0 = open default camera
    int apiID = cv::CAP_ANY;      // 0 = autodetect default API
                                  // open selected camera using selected API
    cap.open(deviceID + apiID);
    // check if we succeeded
    if (!cap.isOpened()) {
        cerr << "ERROR! Unable to open camera\n";
        return -1;
    }
    //--- GRAB AND WRITE LOOP
    cout << "Start grabbing" << endl
         << "Press q to terminate" << endl
    << "Press a to alternate accumulated/normal histogram" << endl
    << "Press s to alternate max of histogram (250/256)" << endl
    << "Press g to switch gaussian blur" << endl
	<< "Press p to switch pixel mode (press +/- modify num of pixels and press * to switch interpolations" << endl
    << "EFFECTS:\n-------------" << endl
    << "0 to reset effects" << endl
    << "1 contrast (press z to increase)" << endl
    << "2 color reduction (press x to increase, press k for kmeans)" << endl
    << "3 alien (press c to change colour)" << endl
    << "4 distortion (press v to switch dist, and ,/. modify k)" << endl
    << "5 take on me (press b to increase threshold)" << endl
    << "6 equalization openCV " << endl
    << "7 equalization ours" << endl
    << "8 negative" << endl
	<< "9 gray scale" << endl
	<< "COOL COMBINATIONS\n---------------" << endl
	<< "8 + 5 + b + g" << endl
	<< "4 + v + v + 9 + p + *" << endl
    << "p + '+'7 + 2 + x5" << endl;

    for (;;)
    {
        // wait for a new frame from camera and store it into 'frame'
        cap.read(frame);

        if (frame.empty()) {
            cerr << "ERROR! blank frame grabbed\n";
            break;
        }

        Sleep(5); // Sleep is mandatory - for no leg!

        //apply filter
        //Mat I = apply_effect_rgb(frame, alien_blue, 50);
        //Mat I = apply_effect(frame, take_on_me, 10);

        //EFECTOS
        if (gauss) {
        	GaussianBlur(frame, frame, Size2i(gauss_s,gauss_s), gauss_k, gauss_k, BORDER_DEFAULT);
        }
        if (gray_scale) {
        	cvtColor(frame, frame, CV_RGB2GRAY);
        }
		if (contraste) {

			Mat ycrcb;

			cvtColor(frame,ycrcb,CV_BGR2HSV);

			vector<Mat> channels;
			split(ycrcb,channels);

			channels[2] = apply_effect(channels[2], contrastF, contrast);
			merge(channels,ycrcb);

			cvtColor(ycrcb,frame,CV_HSV2BGR);

			//frame = apply_effect(frame, contrastF, contrast);
			putText(frame,to_string(contrast),Point2f(16,20),FONT_HERSHEY_PLAIN, 1,  Scalar(0,0,255), 2 , 8 , false);
		}
		if (hist_eq) frame = equalizarCV(frame);
		if (hist_eq_ours) frame = equalizarOurs(frame);
		if (reduccionColores) {
			if (k_means && !gray_scale) {
				Mat data;
				frame.convertTo(data,CV_32F);
				data = data.reshape(1,data.total());

				Mat labels, centers;
				kmeans(data, 8, labels, TermCriteria(CV_TERMCRIT_ITER, 10, 1.0), 3,
				       KMEANS_PP_CENTERS, centers);

				centers = centers.reshape(3,centers.rows);
				data = data.reshape(3,data.rows);


				Vec3f *p = data.ptr<Vec3f>();

				for (size_t i=  0; (int)i < data.rows; i++) {
				   int center_id = labels.at<int>(i);
				   p[i] = centers.at<Vec3f>(center_id);
				}


				frame = data.reshape(3, frame.rows);
				frame.convertTo(frame, CV_8U);
			}
			else {
				frame = apply_effect(frame, reducirColorF, 256 / cbrt(numeroColores));
				putText(frame,to_string(numeroColores),Point2f(16,20),FONT_HERSHEY_PLAIN, 1,  Scalar(0,0,255), 2 , 8 , false);
			}
		}

		if (efectoAlien) {
			Mat skin = skinMat(frame);
			//frame = skin;
			frame = generarAlien(skin,frame, alien_mode);

		}

		if (take_effect) {
			frame = apply_effect(frame, take_on_me, take_mode);
			putText(frame,to_string(take_mode),Point2f(16,20),FONT_HERSHEY_PLAIN, 1,  Scalar(0,0,255), 2 , 8 , false);
		}
		if (negative_effect) {
			frame = apply_effect(frame, negative, 0);
		}
		if (pixel) {
			Mat temp;
			resize(frame, temp, Size2i(pixels, pixels), 0, 0, INTER_LINEAR);
			resize(temp, frame, Size2i(frame.cols, frame.rows), 0, 0, interpolation);
		}
		if (distorsion) generarDistorsion(frame, dist_mode, dist_k);

        // show live and wait for a key with timeout long enough to show images
        imshow("CAMERA 1", frame);  // Window name


        tipka = cv::waitKey(30);

        switch (tipka) {
        case '1':
        	contraste = !contraste;
        	break;
        case '2':
            reduccionColores = !reduccionColores;
            break;
        case '3':
        	efectoAlien = !efectoAlien;
        	if (efectoAlien) gray_scale = false;
            break;
        case '4':
        	distorsion = !distorsion;
            break;
        case '5':
        	take_effect = !take_effect;
        	break;
        case '6':
			hist_eq = !hist_eq;
			if (hist_eq) gray_scale = false;
			break;
        case '7':
			hist_eq_ours = !hist_eq_ours;
			if (hist_eq_ours) gray_scale = false;
			break;
        case '8':
			negative_effect = !negative_effect;
			break;
        case '9':
			gray_scale = !gray_scale;
			if (gray_scale) {
				efectoAlien = false;
				hist_eq_ours = false;
				hist_eq = false;
			}

			break;
        case 'p':
        	pixel = !pixel;
        	break;
        case 'g':
			gauss = !gauss;
			break;




        case 'v':
				dist_mode = (dist_mode + 1) % 2;
				break;
        case 'c':
				alien_mode = (alien_mode + 1) % 3;
				break;
        case 'z':
				contrast = contrast + 0.2;
				if (contrast > 4) contrast = 0;
				break;
        case 'x':
				//numeroColores = (numeroColores + 1) % 64;
        		numeroColores = pow((cbrt(numeroColores) + 1),3);
        		if (numeroColores > 1331) numeroColores = 8;
				break;
        case 'b':
				take_mode = (take_mode + 2) % 20 + 2;
				break;


        case '\'':
				gauss_k += 1;
				break;
        case '�':
				gauss_k -= 1;
				break;
        case '?':
				gauss_s += 1;
				break;
		case '�':
				gauss_s -= 1;
				break;
		case '+':
				pixels = pixels + 10;
				break;
		case '-':
				pixels = pixels - 10;
				if (pixels < 1) pixels = 1;
				break;
		case '*':
				if (interpolation == INTER_NEAREST) interpolation = INTER_LINEAR;
				else interpolation = INTER_NEAREST;
				break;
		case 'k':
				k_means = !k_means;
				break;
		case ',':
				dist_k += 0.25;
				break;
		case '.':
				dist_k -= 0.25;
				if (dist_k < 0) dist_k = 0;
				break;


		case 's':
			if (histSize >= 256 ) histSize = 250;
			else histSize = 256;
			break;



        case '0':
            contraste = false;
            reduccionColores = false;
            efectoAlien = false;
            distorsion = false;
            take_effect = false;
            hist_eq = false;
			hist_eq_ours = false;
			gauss = false;
			negative_effect = false;
			gray_scale = false;
			pixel = false;

			contrast = 1;
			numeroColores = 8;
			alien_mode = 0;
			dist_mode = 0;
			take_mode = 2;
			gauss_k = 1;
			gauss_s = 5;
			pixels = 16;
        }

        if (tipka == 'd') {

            sprintf(filename, "C://Frame_%d.jpg", c); // select your folder - filename is "Frame_n"
            ofstream f_out("C://Frame.jpg");
            f_out << frame;
            f_out.close();
            cv::waitKey(10);

            imshow("CAMERA 1", frame);
            imwrite(filename, frame);
            cout << "Frame_" << c << endl;
            c++;
        }

        if (tipka == 'q') {
            cout << "Terminating..." << endl;
            Sleep(2000);
            break;
        }

        if (tipka == 'a') {
        	accumulate_hist = !accumulate_hist;
		}

        //Historiogram

        Mat channels[3];
        split(frame,channels);
        float range[] = { 0, (float)histSize } ; //the upper boundary is exclusive
        const float* histRange = { range };

        Mat b_hist, g_hist, r_hist;

        /// Compute the histograms:
        calcHist( &channels[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
        calcHist( &channels[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
        calcHist( &channels[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );

        // Draw the histograms for R, G and B
        int hist_w = 512; int hist_h = 400;
        int bin_w = cvRound( (double) hist_w/histSize );

        Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

        if (accumulate_hist) {
        	for( int i = 1; i < histSize; i++ )
			{
				b_hist.at<float>(i) = b_hist.at<float>(i-1) + b_hist.at<float>(i);
				g_hist.at<float>(i) = g_hist.at<float>(i-1) + g_hist.at<float>(i);
				r_hist.at<float>(i) = b_hist.at<float>(i-1) + r_hist.at<float>(i);
			}
		}

        /// Normalize the result to [ 0, histImage.rows ]
        normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
        normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
        normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

        /// Draw for each channel
        for( int i = 1; i < histSize; i++ )
        {

            line( histImage, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
                             Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
                             Scalar( 255, 0, 0), 2, 8, 0  );
            line( histImage, Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ) ,
                             Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
                             Scalar( 0, 255, 0), 2, 8, 0  );
            line( histImage, Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ) ,
                             Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
                             Scalar( 0, 0, 255), 2, 8, 0  );
        }

        namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE );
        imshow("calcHist Demo", histImage );

    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}
