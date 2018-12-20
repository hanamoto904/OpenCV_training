// Training_001.cpp : このファイルには 'main' 関数が含まれています。プログラム実行の開始と終了がそこで行われます。
//

#include "pch.h"
#include <iostream>

#include <opencv2/opencv.hpp>

using namespace cv;

int main()
{
    std::cout << "Hello World!\n";

	Mat image = Mat::zeros(100, 100, CV_8UC3);
	imshow("", image);
	waitKey(0);

}
