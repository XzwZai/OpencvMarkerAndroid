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
	float total_err = 0.0; /* ����ͼ���ƽ�������ܺ� */

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
		//��ȡ�ǵ�
		std::vector<cv::Point2f> imagePointsBuf;
		if (0 == cv::findChessboardCorners(image, boardSize, imagePointsBuf))
		{
			//std::cout << "can not find chessboard corners!\n";
		}
		else
		{

			cv::Mat imageGray;
			cv::cvtColor(image, imageGray, CV_RGB2GRAY);
			cv::find4QuadCornerSubpix(imageGray, imagePointsBuf, cv::Size(5, 5)); //�Դ���ȡ�Ľǵ���о�ȷ��
			//cornerSubPix(view_gray,image_points_buf,Size(5,5),Size(-1,-1),TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER,30,0.1));
			if(imagePointsBuf.size() == boardSize.height * boardSize.width)
			{
				validImageCount++;
				imagePointsSeq.push_back(imagePointsBuf);  //���������ؽǵ�
				return true;
			}
			/* ��ͼ������ʾ�ǵ�λ�� */
			//cv::drawChessboardCorners(imageGray, boardSize, imagePointsBuf, false); //������ͼƬ�б�ǽǵ�
			//cv::imshow("Camera Calibration", imageGray);//��ʾͼƬ
			//cv::waitKey(500);//��ͣ0.5S
		}
		return false;

	}

	void calibrate()
	{		

		imageCount = validImageCount;
		
		cv::Size square_size = cv::Size(10, 10);  /* ʵ�ʲ����õ��ı궨����ÿ�����̸�Ĵ�С */
		std::vector<std::vector<cv::Point3f>> object_points; /* ����궨���Ͻǵ����ά���� */		
		
		std::vector<int> point_counts;  // ÿ��ͼ���нǵ������				
		/* ��ʼ���궨���Ͻǵ����ά���� */
		int i, j, t;
		for (t = 0; t < validImageCount; t++)
		{
			std::vector<cv::Point3f> tempPointSet;
			for (i = 0; i < boardSize.height; i++)
			{
				for (j = 0; j < boardSize.width; j++)
				{
					cv::Point3f realPoint;
					/* ����궨�������������ϵ��z=0��ƽ���� */
					realPoint.x = i * square_size.width;
					realPoint.y = j * square_size.height;
					realPoint.z = 0;
					tempPointSet.push_back(realPoint);
				}
			}
			object_points.push_back(tempPointSet);
		}
		/* ��ʼ��ÿ��ͼ���еĽǵ��������ٶ�ÿ��ͼ���ж����Կ��������ı궨�� */
		for (i = 0; i < validImageCount; i++)
		{
			point_counts.push_back(imagePointsSeq[i].size());
			//point_counts.push_back(boardSize.width*boardSize.height);
		}
		/* ��ʼ�궨 */
		calibrateCamera(object_points, imagePointsSeq, imageSize, cameraMatrix, distCoeffs, rvecsMat, tvecsMat, 0);
		//std::cout << "�궨��ɣ�\n";
		//�Ա궨�����������
		//std::cout << "��ʼ���۱궨���������������\n";
		
		float err = 0.0; /* ÿ��ͼ���ƽ����� */
		std::vector<cv::Point2f> image_points2; /* �������¼���õ���ͶӰ�� */
		//std::cout << "\tÿ��ͼ��ı궨��\n";		
		for (i = 0; i < validImageCount; i++)
		{
			std::vector<cv::Point3f> tempPointSet = object_points[i];
			/* ͨ���õ������������������Կռ����ά���������ͶӰ���㣬�õ��µ�ͶӰ�� */
			projectPoints(tempPointSet, rvecsMat[i], tvecsMat[i], cameraMatrix, distCoeffs, image_points2);
			/* �����µ�ͶӰ��;ɵ�ͶӰ��֮������*/
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
			//std::cout << "��" << i + 1 << "��ͼ���ƽ����" << err << "����" << std::endl;
			
		}
		//std::cout << "����ƽ����" << total_err / validImageCount << "����" << std::endl;
		
		//std::cout << "������ɣ�" << std::endl;
		//���涨����  	
		//std::cout << "�궨���������������" << std::endl;
		//cv::Mat rotation_Matrix = cv::Mat(3, 3, CV_32FC1, cv::Scalar::all(0)); /* ����ÿ��ͼ�����ת���� */
		//std::cout << "����ڲ�������" << std::endl;
		//std::cout << cameraMatrix << std::endl << std::endl;
		//std::cout << "����ϵ����\n";
		//std::cout << distCoeffs << std::endl << std::endl << std::endl;

		//for (int i = 0; i < imageCount; i++)
		//{
		//	std::cout << "��" << i + 1 << "��ͼ�����ת������" << std::endl;
		//	std::cout << tvecsMat[i] << std::endl;
		//	/* ����ת����ת��Ϊ���Ӧ����ת���� */
		//	cv::Rodrigues(rvecsMat[i], rotation_Matrix);
		//	std::cout << "��" << i + 1 << "��ͼ�����ת����" << std::endl;
		//	std::cout << rotation_Matrix << std::endl;
		//	std::cout << "��" << i + 1 << "��ͼ���ƽ��������" << std::endl;
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