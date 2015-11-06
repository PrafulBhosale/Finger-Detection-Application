#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>
#include "dirent.h"

using namespace std;
using namespace cv;


// functions defined later
int runonFolder(const CascadeClassifier cascade, const string folder);
void runonVideo(const CascadeClassifier cascade);


int detect(Mat frame, CascadeClassifier cascade) {
	Mat frame_gray;
	vector<Rect> detections;


	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	//fastNlMeansDenoising(frame_gray, frame_gray, 3, 7, 21);
	// equalizeHist(frame_gray, frame_gray);
	//  medianBlur(frame_gray, frame_gray, 5);
	cascade.detectMultiScale(frame_gray, detections,
		1.1, 8, 0 | CV_HAAR_SCALE_IMAGE, Size(40, 40));
	int detected = (int)detections.size();
	for (int i = 0; i < detected; i++) {
		int width2 = detections[i].width / 2;
		int height2 = detections[i].height / 2;
		Point center(detections[i].x + width2 - 1, detections[i].y + height2 - 1);
		ellipse(frame, center, Size(width2, height2), 0, 0, 360,
			Scalar(255, 0, 255), 2, 8, 0);
	}
	return(detected);
}

int runonFolder(const CascadeClassifier cascade, string folder) {
	if (folder.at(folder.length() - 1) != '/') folder += '/';
	DIR *dir = opendir(folder.c_str());
	if (dir == NULL) {
		cerr << "Can't open folder " << folder << endl;
		exit(1);
	}
	bool finish = false;
	string windowName;
	struct dirent *entry;
	int detections = 0;
	while (!finish && (entry = readdir(dir)) != NULL) {
		char *name = entry->d_name;
		string dname = folder + name;
		Mat img = imread(dname.c_str(), CV_LOAD_IMAGE_UNCHANGED);
		if (!img.empty()) {
			int d = detect(img, cascade);
			cerr << d << " detections" << endl;
			detections += d;
			//if (!windowName.empty()) destroyWindow(windowName);
			windowName = name;
			namedWindow(windowName.c_str(), CV_WINDOW_AUTOSIZE);
			imshow(windowName.c_str(), img);
			int key = waitKey(0); // Wait for a keystroke
			switch (key) {
			case 27: // <Esc>
				finish = true; break;
			default:
				break;
			}
		} // if image is available
	}
	closedir(dir);
	return(detections);
}

void runonVideo(const CascadeClassifier cascade) {
	VideoCapture videocapture(0);
	if (!videocapture.isOpened()) {
		cerr << "Can't open default video camera" << endl;
		exit(1);
	}
	string windowName = "Live Video";
	namedWindow("video", CV_WINDOW_AUTOSIZE);
	Mat frame;
	bool finish = false;
	while (!finish) {
		if (!videocapture.read(frame)) {
			cout << "Can't capture frame" << endl;
			break;
		}
		detect(frame, cascade);
		imshow("video", frame);
		if (waitKey(30) >= 0) finish = true;
	}
}

int main(int argc, char** argv) {
	if (argc != 2 && argc != 3) {
		cerr << argv[0] << ": "
			<< "got " << argc - 1
			<< " arguments. Expecting one or two: cascade [image-folder]"
			<< endl
			<< "Example: cascade.xml ../folderimages" << endl
			<< "Example: cascade.xml" << endl;
		return(-1);
	}

	string cascade_name = argv[1];
	CascadeClassifier cascade;
	if (!cascade.load(cascade_name)) {
		cerr << cascade_name << " is not in a proper cascade format" << endl;
		return(-1);
	}
	string foldername = (argc == 2) ? "" : argv[2];

	if (argc == 2) runonVideo(cascade);
	else { //(argc == 3) 
		int detections = runonFolder(cascade, foldername);
		cout << "Total of " << detections << " detections" << endl;
	}
	waitKey(0);
	return(0);
}

