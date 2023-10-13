#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <stdlib.h>
#include <cmath>
using namespace cv;
using namespace std;

typedef struct parametri {
	int rho, theta;
}PARAMETRI;


#define NUM_THETAS 1000
#define NUM_THETAS_LF 1000.0
#define NUM_RHOS   4000

//bez parametara
void houghLinesP(Mat input, int threshold, int num_lines) {

	int cols = input.cols;
	int rows = input.rows;

	//short(*transformisano_arr)[NUM_RHOS] = (short(*)[NUM_RHOS])calloc(NUM_THETAS * NUM_RHOS, sizeof(short));

	Mat m(NUM_THETAS, NUM_RHOS, CV_16U, Scalar(0));

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			if (input.at<uchar>(i, j) != 0) {
				// radi transformaciju
				for (int k = 0; k < NUM_THETAS; k++) {
					//int rho = (NUM_RHOS / 2) + (NUM_RHOS / 2-1) * std::cos(k / 100.0 * 3.14);// -(j * sin(k / NUM_THETAS)));
					int rho =  (NUM_RHOS/2)+ (j * cos(k / NUM_THETAS_LF * CV_PI) - i * sin(k / NUM_THETAS_LF * CV_PI));
					//transformisano_arr[k][rho]++;
					m.at<ushort>(k, rho)++;
				}
			}
		}
	}

	

	
	
	//int rho = 0;
	//for (int k = 0; k < NUM_THETAS; k++) {
	//	int rho =75+(74) * std::cos(k / 100.0 * 3.14);// -(j * sin(k / NUM_THETAS)));
	//	printf("%d\n", rho);

	//	//m.at<uchar>(k,rho) = 100;
	//	transformisano_arr[k][rho] = 100;
	//}
	
	//Mat m(NUM_THETAS, NUM_RHOS, CV_16U, transformisano_arr);

	Mat t;
	normalize(m, t, 0,255, NORM_MINMAX,CV_8U );

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
	imshow("transformisano", t);


}

// sa parametrima
void houghLinesP(Mat input, PARAMETRI* param, int* size_param, int threshold, int num_lines, int num_thetas, int num_rhos) {

	int cols = input.cols;
	int rows = input.rows;

	uchar** transformisano_arr = (uchar**)calloc(num_thetas * num_rhos, sizeof(short));

	for (int i = 0; i < cols; i++) {
		for (int j = 0; j < rows; j++) {
			if (input.at<uchar>(i, j) != 0) {
				// radi transformaciju
				for (int k = 0; k < num_thetas; k++) {
					int rho = (j * cos(k / num_thetas)) - (i * sin(k / num_thetas));
					transformisano_arr[k][rho]++;
				}
			}
		}
	}

	Mat m(rows, cols, CV_8U, transformisano_arr);
	imshow("transformisano", m);


}



int main(int argc, char** argv)
{
	// Declare the output variables
	Mat dst, cdst;
	const char* default_file = "../../../resources/images/working_image.png";
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

	imshow("edged", dst);

	houghLinesP(dst, 10, 10);

	// Copy edges to the images that will display the results in BGR
	cvtColor(dst, cdst, COLOR_GRAY2BGR);
	Mat cdstP = cdst.clone();



	Rect roi = Rect(cvRound(1*(dst.cols / 10)), cvRound(11*(dst.rows / 20)), cvRound(8*(dst.cols / 10)), cvRound(9*(dst.rows / 20)));

	Mat cropped_gray = dst(roi);
	Mat cropped_colored = src_0(roi);
	//cvtColor(cropped_gray, cropped_colored, COLOR_GRAY2BGR);

	
	// Standard Hough Line Transform
	vector<Vec2f> lines; // will hold the results of the detection
	HoughLines(cropped_gray, lines, 1, CV_PI / 180, 200, 0, 0); // runs the actual detection
	// Draw the lines


	for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0], theta = lines[i][1];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		if (b > 0.9)
			continue;
		double x0 = a * rho, y0 = b * rho;
		pt1.x = cvRound(x0 + 10000 * (-b));
		pt1.y = cvRound(y0 + 10000 * (a));
		pt2.x = cvRound(x0 - 10000 * (-b));
		pt2.y = cvRound(y0 - 10000 * (a));
		line(cropped_colored, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);
	}

	cropped_colored.copyTo(src_0(roi));

	// Show results
	//imshow("Source", src_0);
	//imshow("asdf", cropped_colored);
	//imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst);


	// Probabilistic Line Transform
	vector<Vec4i> linesP; // will hold the results of the detection
	HoughLinesP(dst, linesP, 1, CV_PI / 180, 100, 50, 10); // runs the actual detection
	// Draw the lines
	for (size_t i = 0; i < linesP.size(); i++)
	{
		Vec4i l = linesP[i];
		line(cdstP, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, LINE_AA);
	}
	//imshow("Detected Lines (in red) - Probabilistic Hough Line Transform", cdstP);
	// Wait and Exit
	waitKey();
	return 0;
}