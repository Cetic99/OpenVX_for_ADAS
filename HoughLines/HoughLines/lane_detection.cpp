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


#define NUM_THETAS 		(200)
#define NUM_THETAS_LF 	(200.0)
#define NUM_THETAS_HALF (100)
#define NUM_RHOS   		(200)
#define SCALING 		(20.0)

#define ROI_X			(350)
#define ROI_Y			(350)
#define ROI_WIDTH		(1280-450-350)
#define ROI_HEIGHT		(720-230-350)

#define WIDTH			(1280)
#define HEIGHT			(720)

//bez parametara
std::vector<Point> houghLinesP(Mat input, int threshold, int num_lines) {

	int cols = input.cols;
	int rows = input.rows;
	//cout << "cols=" << cols << "," << "rows=" << rows << endl;

	//short(*transformisano_arr)[NUM_RHOS] = (short(*)[NUM_RHOS])calloc(NUM_THETAS * NUM_RHOS, sizeof(short));

	Rect line0 = Rect(0, cols / 5, rows - 20,0);
    float_t k0 = ((float_t)line0.height - (float_t)line0.y) / ((float_t)line0.width - (float_t)line0.x);
    float_t n0 = (int32_t)line0.y - k0 * (int32_t)line0.x;

    Rect line1 = Rect(0, 4*cols / 5, rows - 20,cols);
    float_t k1 = ((float_t)line1.height - (float_t)line1.y) / ((float_t)line1.width - (float_t)line1.x);
    float_t n1 = (int32_t)line1.y - k1 * (int32_t)line1.x;
    int32_t curr_yy0;
    int32_t curr_yy1;

	uint32_t x, y, i, j;
    bool switched = false;
    for (y = 0; y < rows; y++)
    {
        switched = false;
        uint8_t state = 0;
        curr_yy0 = k0 * y + n0;
        curr_yy1 = k1 * y + n1;
        for (x = 0; x < cols; x++)
        {
            if (x == curr_yy0 || x == curr_yy1)
            {
                if(switched == true)
                    switched = false;
                else
                    switched = true;
            }
            if (switched == false)
            {
                input.at<uchar>(y,x) = state;
            }
        }
    }
	imshow("roired_img",input);

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



	Mat normalized;
	normalize(m, normalized, 0,255, NORM_MINMAX,CV_8U );
	cv::imshow("Normalized", normalized);

	std::vector<Point> tacke;
 	int32_t left_max = 0, right_max = 0;
    Point left_coord, right_coord;
    bool left = false, right = false;
	for (int i = 0; i < NUM_THETAS_HALF; i++) {
		for (int j = 0; j < NUM_RHOS; j++) {
			uint8_t pixel0 = normalized.at<uchar>(i,j);
			uint8_t pixel1 = normalized.at<uchar>(i+NUM_THETAS_HALF,j);
			float_t theta1 = (float_t)i * CV_PI / NUM_THETAS_LF;
            float_t theta2 = ((float_t)i + NUM_THETAS_HALF) * CV_PI / NUM_THETAS_LF;
			if (pixel0 > left_max && sin(theta1) < 0.8)
            {
                left_max = pixel0;
				left_coord = Point(i,j);
                left = true;
            }
            if (pixel1 > right_max && sin(theta2) < 0.8)
            {
                right_max = pixel1;
                right_coord = Point(i+NUM_THETAS_HALF,j);
                right = true;
            }
		}
	}
	tacke.push_back(left_coord);
	tacke.push_back(right_coord);
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
	// Working roi
	Rect roi(ROI_X,ROI_Y,ROI_WIDTH,ROI_HEIGHT);	
	
	// intermediate Mat objects
	Mat src;
	Mat src_roi;
	Mat gray_src(720,1280,CV_8UC3);
	Mat gray_roi;
	Mat blured0;
	Mat blured1;
	Mat edged;
	
	if(!strcmp("--image",argv[1])){
		resize(imread(argv[2]),src,Size(WIDTH,HEIGHT));
		/***************************************************
		*		PROCEDURES
		****************************************************/
		// 1. Converting to grayscale
		cvtColor(src, gray_src, COLOR_RGB2GRAY);
		gray_roi = gray_src(roi);
		imshow("grayed_roi", gray_roi);
		// 2. Bluring image
		GaussianBlur(gray_roi, blured0, Size(3, 3),0);
		GaussianBlur(blured0, blured1, Size(3, 3),0);
		imshow("blured_images",blured1);
		// 3. Detecting edges
		Canny(blured1, edged, 150, 200, 3);
		imshow("edged",edged);

		std::vector<Point> tacke = houghLinesP(edged, 10, 10);
		src_roi = src(roi);
		for (Point p : tacke) {
			int x0 = 0,x1 = ROI_HEIGHT, y0,y1;
			float tann = tan(p.x * ((3.14) / (NUM_THETAS_LF)));
			float shifted = p.y - NUM_RHOS / 2;
			float coss = (cos(p.x * ((3.14) / (NUM_THETAS_LF))));
			y0 = x0 * tan(p.x * ((3.14) / (NUM_THETAS_LF))) + SCALING * ((p.y - ((NUM_RHOS) / (2))) / (cos(p.x * ((3.14) / (NUM_THETAS_LF)))));
			y1 = x1 * tan(p.x * ((3.14) / (NUM_THETAS_LF))) + SCALING * ((p.y - ((NUM_RHOS) / (2))) / (cos(p.x * ((3.14) / (NUM_THETAS_LF)))));
			line(src_roi, Point(y0, x0), Point(y1, x1), Scalar(0, 0, 255),3);
			cout << "Point0(x=" << x0 << ",y=" << y0 << ")" << endl;
			cout << "Point1(x=" << x1 << ",y=" << y1 << ")" << endl;
		}

		imshow("lined",src);
	
	
	}
	else if(!strcmp("--video",argv[1])){
		       VideoCapture cap(argv[2]);
        if (!cap.isOpened()) {
            printf("Unable to open camera\n");
            return 0;
        }
        for(;;) {
            cap >> src;
            resize(src, src, Size(WIDTH,HEIGHT));
            if(waitKey(30) >= 0) break;
			/***************************************************
			*		PROCEDURES
			****************************************************/
			// 1. Converting to grayscale
			cvtColor(src, gray_src, COLOR_RGB2GRAY);
			gray_roi = gray_src(roi);
			imshow("grayed_roi", gray_roi);
			// 2. Bluring image
			GaussianBlur(gray_roi, blured0, Size(3, 3),0);
			GaussianBlur(blured0, blured1, Size(3, 3),0);
			imshow("blured_images",blured1);
			// 3. Detecting edges
			Canny(blured1, edged, 150, 200, 3);
			imshow("edged",edged);

			std::vector<Point> tacke = houghLinesP(edged, 10, 10);
			src_roi = src(roi);
			for (Point p : tacke) {
				int x0 = 0,x1 = ROI_HEIGHT, y0,y1;
				float tann = tan(p.x * ((3.14) / (NUM_THETAS_LF)));
				float shifted = p.y - NUM_RHOS / 2;
				float coss = (cos(p.x * ((3.14) / (NUM_THETAS_LF))));
				y0 = x0 * tan(p.x * ((3.14) / (NUM_THETAS_LF))) + SCALING * ((p.y - ((NUM_RHOS) / (2))) / (cos(p.x * ((3.14) / (NUM_THETAS_LF)))));
				y1 = x1 * tan(p.x * ((3.14) / (NUM_THETAS_LF))) + SCALING * ((p.y - ((NUM_RHOS) / (2))) / (cos(p.x * ((3.14) / (NUM_THETAS_LF)))));
				line(src_roi, Point(y0, x0), Point(y1, x1), Scalar(0, 0, 255),3);
				cout << "Point0(x=" << x0 << ",y=" << y0 << ")" << endl;
				cout << "Point1(x=" << x1 << ",y=" << y1 << ")" << endl;
			}

			imshow("lined",src);
            /*=================== drawing lines =============================================*/

            /*============================================================================*/
            //imshow( "CannyDetect", mat );
            //imshow("inputWindow", input);
            if(waitKey(30) >= 0) break;
		}
	}
	else {
		printf(" Error opening image\n");
		printf("usage --image <path_to_image>\n");
		printf("usage --video <path_to_video>\n");
		return -1;
	}

	
	// Wait and Exit
	waitKey();
	return 0;
}