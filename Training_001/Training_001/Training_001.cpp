// Training_001.cpp : このファイルには 'main' 関数が含まれています。プログラム実行の開始と終了がそこで行われます。
//

#include "pch.h"
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace std;
using namespace cv;

int main()
{
	// 入力画像の取得
	Mat img_1 = imread("D:\\hanamoto\\VS2017\\Projects\\OpenCV_01\\Sample\\lena.png", IMREAD_GRAYSCALE);
	Mat img_2 = img_1.clone();
	if (!img_1.data || !img_2.data) {
		std::cout << "画像がよみこめません" << std::endl; return -1;
	}
	int minHessian = 400;
	Ptr < xfeatures2d::SURF>detectorSURF = xfeatures2d::SURF::create(minHessian);
	Ptr < xfeatures2d::SIFT>detectorSIFT = xfeatures2d::SIFT::create(minHessian);
	vector <KeyPoint>keypoints_1, keypoints_2;
	detectorSURF->detect(img_1, keypoints_1);
	detectorSIFT->detect(img_2, keypoints_2);

	Mat img_1_keypoints;
	Mat img_2_keypoints;
	drawKeypoints(img_1, keypoints_1, img_1_keypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	drawKeypoints(img_2, keypoints_2, img_2_keypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

	imshow("INPUT_IMG", img_1);
	imshow("SURF_IMG", img_1_keypoints);
	imshow("SIFT_IMG", img_2_keypoints);

	waitKey(0);

	return 0;
}
