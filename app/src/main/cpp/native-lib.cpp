#include "native-lib.h"
#include <opencv2/opencv.hpp>

extern "C"
{



void SharedObject1::showImage(char* imageDataIn, int width, int height)
{
    cv::Mat img(height, width, CV_8UC4, imageDataIn);
    cv::imshow("1", img);
    cv::waitKey(0);
}

void SharedObject1::cvtImage(char* imageDataIn, char* imageDataOut, int width, int height)
{
    cv::Mat img(height, width, CV_8UC4, imageDataIn);
    cv::cvtColor(img, img, CV_RGBA2BGR);
    cv::flip(img, img, 0);
    cv::cvtColor(img, img, CV_BGR2GRAY);
    cv::cvtColor(img, img, CV_GRAY2BGR);
    cv::cvtColor(img, img, CV_BGR2BGRA);
    memcpy(imageDataOut, img.data, width * height * 4);
    //cv::imshow("3", img);
    //cv::waitKey(0);
}

bool SharedObject1::PushImageToCalibrater(void *pointer, char *imageDataIn, int width, int height)
{
    Calibrater* calibrater = (Calibrater*)pointer;
    cv::Mat image(height, width, CV_8UC4, imageDataIn);
    cv::cvtColor(image, image, CV_RGBA2BGR);
    cv::flip(image, image, 0);
    cv::transpose(image, image);
    cv::flip(image, image, 1);
    return calibrater->pushImage(image);
}

void* SharedObject1::CreateCalibrater(int boardWidth, int boardHeight)
{
    return new Calibrater(cv::Size(boardWidth,boardHeight));
}

void SharedObject1::Calibrate(void* pointer,CalibrateInfo* info)
{
    Calibrater* calibrater = (Calibrater*)pointer;
    calibrater->calibrate();
    info->err = calibrater->total_err / calibrater->validImageCount;
    info->fx = calibrater->cameraMatrix.at<double>(0, 0);
    info->fy = calibrater->cameraMatrix.at<double>(1, 1);
    info->cx = calibrater->cameraMatrix.at<double>(0, 2);
    info->cy = calibrater->cameraMatrix.at<double>(1, 2);
    for(int i = 0;i < 5;i++)
    {
        info->dist[i] = calibrater->distCoeffs.at<double>(0,i);
    }
}

void SharedObject1::RestartCalibrater(void* pointer)
{
    Calibrater* calibrater = (Calibrater*)pointer;
    calibrater->restart();
}

//void SharedObject1::calibrate(char * imagesDataIn, int imageCount, int width, int height,int boardWidth, int boardHeight,float* fx, float* fy,float* dx, float* dy, float* err)
//{
//    std::vector<cv::Mat> images;
//    int step = width * height * 4;
//    for (int i = 0; i < imageCount; i++)
//    {
//        cv::Mat img(height, width, CV_8UC4, imagesDataIn + step * i);
//        cv::cvtColor(img, img, CV_RGBA2BGR);
//        cv::flip(img, img, 0);
//        images.push_back(img);
//    }
//    Calibrater calibrater(cv::Size(boardWidth, boardHeight));
//    calibrater.calibrate(images);
//    *err = calibrater.total_err / imageCount;
//    *fx = calibrater.cameraMatrix.at<double>(0, 0);
//    *fy = calibrater.cameraMatrix.at<double>(1, 1);
//    *dx = calibrater.cameraMatrix.at<double>(0, 2);
//    *dy = calibrater.cameraMatrix.at<double>(1, 2);
//
//}

void SharedObject1::estimate(void* pointer, char* imageDataIn, int width, int height, float* rotation, float* translation, float* rVec, int* markerCount)
{
    cv::Mat image(height, width, CV_8UC4, imageDataIn);
    cv::cvtColor(image, image, CV_RGBA2BGR);
    cv::flip(image, image, 0);
    MarkerDetector* md = (MarkerDetector*)pointer;
    std::vector<Marker> markers;
    //md->findMarkers(image, markers);
    md->getMarkersPos(image, markers);
    *markerCount = markers.size();
    if (markers.size() != 0)
    {
        for (int i = 0; i < 9; i++)
        {
            *(rotation + i) = markers[0].transformation.r().data[i];
        }
        for (int i = 0; i < 3; i++)
        {
            *(translation + i) = markers[0].transformation.t().data[i];
        }
        for (int i = 0; i < 3; i++)
        {
            *(rVec + i) = markers[0].rVec[i];
        }
    }

}

void* SharedObject1::CreateMD(float f, float dx, float dy, float* distortion)
{
    float dis[5];
    for (int i = 0; i < 5; i++)
    {
        dis[i] = distortion[5];
    }
    cv::Mat distortion_coefficients = cv::Mat(5, 1, CV_32FC1, dis);
    return new MarkerDetector(f, dx, dy, distortion_coefficients);

}

void* SharedObject1::GetMarkers(void* pointer, char* imageDataIn, int width, int height, int* markerCount)
{
    MarkerDetector* md = (MarkerDetector*)pointer;
    cv::Mat image(height, width, CV_8UC4, imageDataIn);
    cv::cvtColor(image, image, CV_RGBA2BGR);
    cv::flip(image, image, 0);
    cv::transpose(image, image);
    cv::flip(image, image, 1);
    std::vector<Marker> markers;
    md->getMarkersPos(image, markers);
    *markerCount = markers.size();
    MarkerInfo* p = (MarkerInfo*)malloc(sizeof(MarkerInfo) * (*markerCount));
    //MarkerInfo* p = new MarkerInfo[*markerCount];
    for (int i = 0; i < *markerCount; i++)
    {
        p[i].id = markers[i].id;
        for (int j = 0; j < 9; j++)
        {
            p[i].rotation[j] = markers[i].transformation.r().data[j];
        }
        for (int j = 0; j < 3; j++)
        {
            p[i].translation[j] = markers[i].transformation.t().data[j];
        }
        float area = contourArea(markers[i].points);
        float confidence = 1 / (0.0001*area) / (0.0001*area) - 1 > 0 ? 1 / (0.0001*area) / (0.0001*area) - 1 : 0;
        p[i].confidence = confidence;
        /*for (int j = 0; j < 3; j++)
        {
            p[i].rVec[j] = markers[i].rVec[j];
        }*/
    }
    return p;
}

void SharedObject1::FreeNewMemory(void* pointer)
{
    pointer = (MarkerInfo*)pointer;
    if (pointer != NULL)
    {
        free(pointer);
        //delete(pointer);
        pointer = NULL;
    }
}

void SharedObject1::setIntrinsic(void* pointer, float f, float dx, float dy, float* distortion)
{
    float dis[5];
    for (int i = 0; i < 5; i++)
    {
        dis[i] = distortion[i];
    }
    cv::Mat distortion_coefficients = cv::Mat(1, 5, CV_32FC1, dis);
    MarkerDetector* md = (MarkerDetector*)pointer;
    //md->setIntrinsic(f, dx, dy);
    md->setIntrinsic(f, dx, dy, distortion_coefficients);
}

void SharedObject1::setCannyThreshold(void* pointer, int threshold)
{
    MarkerDetector* md = (MarkerDetector*)pointer;
    md->cannyThreshold = threshold;
}

void SharedObject1::setMode(void* pointer,int mode)
{
    MarkerDetector* md = (MarkerDetector*)pointer;
    md->mode = mode;
}

 void SharedObject1::getIntrinsic(void* pointer, CalibrateInfo* info)
{
    MarkerDetector* md = (MarkerDetector*)pointer;
    info->fx = md->camMatrix.at<float>(0, 0);
    info->fy = md->camMatrix.at<float>(1, 1);
    info->cx = md->camMatrix.at<float>(0, 2);
    info->cy = md->camMatrix.at<float>(1, 2);
    info->err = 10;
    for (int i = 0; i < 5; i++)
    {
        info->dist[i] = md->distCoeff.at<float>(0, i);
    }

}

}