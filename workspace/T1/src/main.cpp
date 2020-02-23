#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <windows.h> // For Sleep
#include "utils.hpp"
#include <fstream>

using namespace cv;
using namespace std;


int ct = 0;
char tipka;
char filename[100]; // For filename
int  c = 1; // For filename

int main(int, char**)
{


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
        << "Press a to terminate" << endl;
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
        Mat I = apply_effect(frame, take_on_me, 9);

        // show live and wait for a key with timeout long enough to show images
        imshow("CAMERA 1", I);  // Window name


        tipka = cv::waitKey(30);


        if (tipka == 'q') {

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


        if (tipka == 'a') {
            cout << "Terminating..." << endl;
            Sleep(2000);
            break;
        }


    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}
