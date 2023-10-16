#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
using namespace cv;
using namespace std;

typedef struct parametri {
	int rho, theta;
}PARAMETRI;


#define NUM_THETAS 500
#define NUM_THETAS_LF 500.0
#define NUM_RHOS   500
#define SCALING 10.0

//bez parametara
std::vector<Point> houghLinesP(Mat input, int threshold, int num_lines) {

	int cols = input.cols;
	int rows = input.rows;
	//cout << "cols=" << cols << "," << "rows=" << rows << endl;

	//short(*transformisano_arr)[NUM_RHOS] = (short(*)[NUM_RHOS])calloc(NUM_THETAS * NUM_RHOS, sizeof(short));

	Mat m(NUM_THETAS, NUM_RHOS, CV_16U, Scalar(0));

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			if (input.at<uchar>(i, j) != 0) {
				// radi transformaciju
				for (int k = 0; k < NUM_THETAS; k++) {
					int rho =  (NUM_RHOS/2)+ (j * cos(k / NUM_THETAS_LF * CV_PI) - i * sin(k / NUM_THETAS_LF * CV_PI))/SCALING;

					m.at<ushort>(k, rho)++;
				}
			}
		}
	}



	Mat t;
	normalize(m, t, 0,255, NORM_MINMAX,CV_8U );

	int neighborhoodSize = 5;
	Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * neighborhoodSize + 1, 2 * neighborhoodSize + 1));
	cv::Mat dilated;
	cv::dilate(t, dilated, element);

	int thresholdValue = 210; // Adjust this threshold value as needed
	Mat thresholded;
	cv::threshold(dilated, thresholded, thresholdValue, 255, cv::THRESH_BINARY);
	Mat eroded;
	erode(thresholded, eroded, getStructuringElement(MORPH_RECT, Size(11, 11)));
	//cv::imshow("Local Maxima", eroded);

	std::vector<Point> tacke;

	for (int i = 0; i < NUM_THETAS; i++) {
		for (int j = 0; j < NUM_RHOS; j++) {
			if (eroded.at<uchar>(i, j) != 0) {
				if (sin(i * CV_PI / NUM_THETAS_LF) < 0.9) {
					//printf("x=%d,y=%d\n", i, j);
					tacke.push_back(Point(i, j));
				}
				
			}
		}
	}
	// Create a binary mask where values greater than the threshold are set to 255 (white) and others to 0 (black)
	//cv::Mat thresholded;
	//cv::compare(t, 100, thresholded, cv::CMP_GT);

	//// Find coordinates (points) of values greater than the threshold
	//std::vector<cv::Point> points;
	//for (int y = 0; y < thresholded.rows; y++) {
	//	for (int x = 0; x < thresholded.cols; x++) {
	//		if (thresholded.at<uchar>(y, x) > 0) {
	//			points.push_back(cv::Point(x, y));
	//		}
	//	}
	//}
	//imshow("transformisano", t);

	return tacke;
}



int main(int argc, char** argv)
{
	// Declare the output variables
	Mat dst, cdst;
	const char* default_file = "../../../../resources/images/working_image.png";
	const char* filename = argc >= 2 ? argv[1] : default_file;
	// Loads an image
	Mat src_0 = imread(samples::findFile(filename), IMREAD_COLOR);
	Mat src;
	cvtColor(src_0, src, COLOR_RGB2GRAY);
	// Check if image is loaded fine
	if (src.empty()) {
		printf(" Error opening image\n");
		printf(" Program Arguments: [image_name -- default %s] \n", default_file);
		return -1;
	}
	Mat blured;
	GaussianBlur(src, blured, Size(3, 3),0);
	// Edge detection
	Canny(blured, dst, 150, 200, 3);

	//imshow("edged", dst);

	std::vector<Point> tacke = houghLinesP(dst, 10, 10);
	for (Point p : tacke) {
		int x0 = 500,x1 = 880, y0,y1;
		y0 = x0 * tan(p.x * ((3.14) / (NUM_THETAS_LF))) + SCALING * ((p.y - ((NUM_RHOS) / (2))) / (cos(p.x * ((3.14) / (NUM_THETAS_LF)))));
		y1 = x1 * tan(p.x * ((3.14) / (NUM_THETAS_LF))) + SCALING * ((p.y - ((NUM_RHOS) / (2))) / (cos(p.x * ((3.14) / (NUM_THETAS_LF)))));
		line(src_0, Point(y0, x0), Point(y1, x1), Scalar(0, 0, 255),3);
		//cout << "Point0(x=" << x0 << ",y=" << y0 << ")" << endl;
		//cout << "Point1(x=" << x1 << ",y=" << y1 << ")" << endl;
	}

	imshow("lined",src_0);
	
	

	//// Copy edges to the images that will display the results in BGR
	//cvtColor(dst, cdst, COLOR_GRAY2BGR);
	//Mat cdstP = cdst.clone();



	//Rect roi = Rect(cvRound(1*(dst.cols / 10)), cvRound(11*(dst.rows / 20)), cvRound(8*(dst.cols / 10)), cvRound(9*(dst.rows / 20)));

	//Mat cropped_gray = dst(roi);
	//Mat cropped_colored = src_0(roi);
	////cvtColor(cropped_gray, cropped_colored, COLOR_GRAY2BGR);

	//
	//// Standard Hough Line Transform
	//vector<Vec2f> lines; // will hold the results of the detection
	//HoughLines(cropped_gray, lines, 1, CV_PI / 180, 200, 0, 0); // runs the actual detection
	//// Draw the lines


	//for (size_t i = 0; i < lines.size(); i++)
	//{
	//	float rho = lines[i][0], theta = lines[i][1];
	//	Point pt1, pt2;
	//	double a = cos(theta), b = sin(theta);
	//	if (b > 0.9)
	//		continue;
	//	double x0 = a * rho, y0 = b * rho;
	//	pt1.x = cvRound(x0 + 10000 * (-b));
	//	pt1.y = cvRound(y0 + 10000 * (a));
	//	pt2.x = cvRound(x0 - 10000 * (-b));
	//	pt2.y = cvRound(y0 - 10000 * (a));
	//	line(cropped_colored, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);
	//}

	//cropped_colored.copyTo(src_0(roi));

	// Show results
	//imshow("Source", src_0);
	//imshow("asdf", cropped_colored);
	//imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst);


	// Probabilistic Line Transform
	//vector<Vec4i> linesP; // will hold the results of the detection
	//HoughLinesP(dst, linesP, 1, CV_PI / 180, 100, 50, 10); // runs the actual detection
	//// Draw the lines
	//for (size_t i = 0; i < linesP.size(); i++)
	//{
	//	Vec4i l = linesP[i];
	//	line(cdstP, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, LINE_AA);
	//}
	//imshow("Detected Lines (in red) - Probabilistic Hough Line Transform", cdstP);
	// Wait and Exit
	waitKey();
	return 0;
}