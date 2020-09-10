#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

struct CalibrateInfo
{
	float fx;
	float fy;
	float cx;
	float cy;
	float err;
	float dist[5];
};

class Calibrater
{
public:
	//std::vector<cv::Mat> images;
	cv::Size boardSize;	

	cv::Size imageSize;
	int imageCount;
	int validImageCount;
	cv::Mat cameraMatrix;
	cv::Mat distCoeffs;
	std::vector<cv::Mat> tvecsMat;
	std::vector<cv::Mat> rvecsMat;
	float total_err = 0.0; /* 所有图像的平均误差的总和 */

	std::vector<std::vector<cv::Point2f>> imagePointsSeq;

	Calibrater(cv::Size _boardSize) : boardSize(_boardSize) 
	{
		cameraMatrix = cv::Mat(3, 3, CV_32FC1, cv::Scalar::all(0));
		distCoeffs = cv::Mat(1, 5, CV_32FC1, cv::Scalar::all(0));
		validImageCount = 0;
	}

	bool pushImage(cv::Mat& image)
	{
		imageSize.width = image.cols;
		imageSize.height = image.rows;
		//提取角点
		std::vector<cv::Point2f> imagePointsBuf;
		if (0 == cv::findChessboardCorners(image, boardSize, imagePointsBuf))
		{
			//std::cout << "can not find chessboard corners!\n";
		}
		else
		{

			cv::Mat imageGray;
			cv::cvtColor(image, imageGray, CV_RGB2GRAY);
			cv::find4QuadCornerSubpix(imageGray, imagePointsBuf, cv::Size(5, 5)); //对粗提取的角点进行精确化
			//cornerSubPix(view_gray,image_points_buf,Size(5,5),Size(-1,-1),TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER,30,0.1));
			if(imagePointsBuf.size() == boardSize.height * boardSize.width)
			{
				validImageCount++;
				imagePointsSeq.push_back(imagePointsBuf);  //保存亚像素角点
				return true;
			}
			/* 在图像上显示角点位置 */
			//cv::drawChessboardCorners(imageGray, boardSize, imagePointsBuf, false); //用于在图片中标记角点
			//cv::imshow("Camera Calibration", imageGray);//显示图片
			//cv::waitKey(500);//暂停0.5S
		}
		return false;

	}

	void calibrate()
	{		

		imageCount = validImageCount;
		
		cv::Size square_size = cv::Size(10, 10);  /* 实际测量得到的标定板上每个棋盘格的大小 */
		std::vector<std::vector<cv::Point3f>> object_points; /* 保存标定板上角点的三维坐标 */		
		
		std::vector<int> point_counts;  // 每幅图像中角点的数量				
		/* 初始化标定板上角点的三维坐标 */
		int i, j, t;
		for (t = 0; t < validImageCount; t++)
		{
			std::vector<cv::Point3f> tempPointSet;
			for (i = 0; i < boardSize.height; i++)
			{
				for (j = 0; j < boardSize.width; j++)
				{
					cv::Point3f realPoint;
					/* 假设标定板放在世界坐标系中z=0的平面上 */
					realPoint.x = i * square_size.width;
					realPoint.y = j * square_size.height;
					realPoint.z = 0;
					tempPointSet.push_back(realPoint);
				}
			}
			object_points.push_back(tempPointSet);
		}
		/* 初始化每幅图像中的角点数量，假定每幅图像中都可以看到完整的标定板 */
		for (i = 0; i < validImageCount; i++)
		{
			point_counts.push_back(imagePointsSeq[i].size());
			//point_counts.push_back(boardSize.width*boardSize.height);
		}
		/* 开始标定 */
		calibrateCamera(object_points, imagePointsSeq, imageSize, cameraMatrix, distCoeffs, rvecsMat, tvecsMat, 0);
		//std::cout << "标定完成！\n";
		//对标定结果进行评价
		//std::cout << "开始评价标定结果………………\n";
		
		float err = 0.0; /* 每幅图像的平均误差 */
		std::vector<cv::Point2f> image_points2; /* 保存重新计算得到的投影点 */
		//std::cout << "\t每幅图像的标定误差：\n";		
		for (i = 0; i < validImageCount; i++)
		{
			std::vector<cv::Point3f> tempPointSet = object_points[i];
			/* 通过得到的摄像机内外参数，对空间的三维点进行重新投影计算，得到新的投影点 */
			projectPoints(tempPointSet, rvecsMat[i], tvecsMat[i], cameraMatrix, distCoeffs, image_points2);
			/* 计算新的投影点和旧的投影点之间的误差*/
			std::vector<cv::Point2f> tempImagePoint = imagePointsSeq[i];
			cv::Mat tempImagePointMat = cv::Mat(1, tempImagePoint.size(), CV_32FC2);
			cv::Mat image_points2Mat = cv::Mat(1, image_points2.size(), CV_32FC2);
			for (int j = 0; j < tempImagePoint.size(); j++)
			{
				image_points2Mat.at<cv::Vec2f>(0, j) = cv::Vec2f(image_points2[j].x, image_points2[j].y);
				tempImagePointMat.at<cv::Vec2f>(0, j) = cv::Vec2f(tempImagePoint[j].x, tempImagePoint[j].y);
			}
			err = cv::norm(image_points2Mat, tempImagePointMat, cv::NORM_L2);
			total_err += err /= point_counts[i];
			//std::cout << "第" << i + 1 << "幅图像的平均误差：" << err << "像素" << std::endl;
			
		}
		//std::cout << "总体平均误差：" << total_err / validImageCount << "像素" << std::endl;
		
		//std::cout << "评价完成！" << std::endl;
		//保存定标结果  	
		//std::cout << "标定结果………………" << std::endl;
		//cv::Mat rotation_Matrix = cv::Mat(3, 3, CV_32FC1, cv::Scalar::all(0)); /* 保存每幅图像的旋转矩阵 */
		//std::cout << "相机内参数矩阵：" << std::endl;
		//std::cout << cameraMatrix << std::endl << std::endl;
		//std::cout << "畸变系数：\n";
		//std::cout << distCoeffs << std::endl << std::endl << std::endl;

		//for (int i = 0; i < imageCount; i++)
		//{
		//	std::cout << "第" << i + 1 << "幅图像的旋转向量：" << std::endl;
		//	std::cout << tvecsMat[i] << std::endl;
		//	/* 将旋转向量转换为相对应的旋转矩阵 */
		//	cv::Rodrigues(rvecsMat[i], rotation_Matrix);
		//	std::cout << "第" << i + 1 << "幅图像的旋转矩阵：" << std::endl;
		//	std::cout << rotation_Matrix << std::endl;
		//	std::cout << "第" << i + 1 << "幅图像的平移向量：" << std::endl;
		//	std::cout << tvecsMat[i] << std::endl << std::endl;
		//}
		
		//std::cout << std::endl;

	}

	void restart()
	{
		imagePointsSeq.clear();
		validImageCount = 0;
	}

};