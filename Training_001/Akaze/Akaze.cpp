
#include "pch.h"
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>

using namespace std;
using namespace cv;

int main()
{
	//比較用画像を読み込む (アルファチャンネル非対応のため、IMREAD_COLORで強制する)
	Mat img001 = imread( "D:\\hanamoto\\VS2017\\Projects\\OpenCV_01\\Sample\\001.jpg", IMREAD_COLOR);
	Mat img002 = imread( "D:\\hanamoto\\VS2017\\Projects\\OpenCV_01\\Sample\\002.jpg", IMREAD_COLOR);
	if (img001.empty() || img002.empty()) return -1;

	//アルゴリズムにAKAZEを使用する
	auto algorithm = AKAZE::create( AKAZE::DESCRIPTOR_MLDB, 0, 3, 0.001f, 4, 4, KAZE::DIFF_PM_G2);

	// 特徴点抽出
	vector<KeyPoint> keypoint1, keypoint2;
	algorithm->detect( img001, keypoint1 );
	algorithm->detect( img002, keypoint2 );
	if (keypoint1.size() == 0 || keypoint2.size() == 0) return -1;

	// 特徴記述
	Mat descriptor1, descriptor2;
	algorithm->compute( img001, keypoint1, descriptor1 );
	algorithm->compute( img002, keypoint2, descriptor2 );

	// マッチング (アルゴリズムにはBruteForceを使用)
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");
	vector<DMatch> match, match12, match21;
	matcher->match(descriptor1, descriptor2, match12);
	matcher->match(descriptor2, descriptor1, match21);

	//クロスチェック(1→2と2→1の両方でマッチしたものだけを残して精度を高める)
	for (size_t i=0; i<match12.size(); i++)
	{
		DMatch forward = match12[i];
		DMatch backward = match21[forward.trainIdx];
		if (backward.trainIdx != forward.queryIdx) continue;

		if (forward.distance < 300.0f)  match.push_back(forward);
	}

	// マッチング結果の描画
	Mat dest;
	drawMatches( img001, keypoint1, img002, keypoint2, match, dest );

	Mat img001_keypoints;
	Mat img002_keypoints;
	drawKeypoints( img001, keypoint1, img001_keypoints, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	drawKeypoints( img002, keypoint2, img002_keypoints, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	resize(img001_keypoints, img001_keypoints, Size(), 0.5, 0.5 );
	resize(img002_keypoints, img002_keypoints, Size(), 0.5, 0.5);

	imshow( "img001", img001_keypoints );
	imshow( "img002", img002_keypoints );

	resize( dest, dest, Size(), 0.5, 0.5);
	imshow( "Matching", dest );

	waitKey(0);

	//マッチング結果の書き出し
	//imwrite( "D:\\hanamoto\\VS2017\\Projects\\OpenCV_01\\Sample\\output.png", dest );

	cout << "end" << endl;
}
