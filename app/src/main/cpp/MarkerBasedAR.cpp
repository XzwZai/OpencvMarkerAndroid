// MarkerBasedAR.cpp : 定义 DLL 应用程序的导出函数。
//

#include "MarkerBasedAR.h"

MarkerDetector *md;
Calibrater *calibrater;

cv::Size imageSize(1280,960);
cv::Mat mapx = cv::Mat(imageSize, CV_32FC1);
cv::Mat mapy = cv::Mat(imageSize, CV_32FC1);
cv::Mat R = cv::Mat::eye(3, 3, CV_32F);
extern  "C"
{

    void InitMarkerDetector(float f, float dx, float dy, float *distortion) {
        float dis[5];
        for (int i = 0; i < 5; i++) {
            dis[i] = distortion[5];
        }
        cv::Mat distortion_coefficients = cv::Mat(1, 5, CV_32FC1, dis);
        md = new MarkerDetector(f, dx, dy, distortion_coefficients);
        imageSize = cv::Size(dx * 2, dy * 2);
        initUndistortRectifyMap(md->camMatrix, md->distCoeff, R, md->camMatrix, imageSize, CV_32FC1,
                                mapx, mapy);
    }

    void *GetMarkers(char *imageDataIn, int width, int height, int *markerCount, bool tracking, int* trackStatus) {
        cv::Mat image(height, width, CV_8UC4, imageDataIn);
        cv::cvtColor(image, image, CV_RGBA2BGR);
        cv::flip(image, image, 0);
        cv::transpose(image, image);
        cv::flip(image, image, 1);
        std::vector<Marker> markers;
        if(tracking)
        {
            *trackStatus = md->track(image, markers);
        }
        else{
            //md->findMarkers(image, markers);
            md->getMarkersPos(image, markers);
        }
        *markerCount = markers.size();
        //MarkerInfo* p = new MarkerInfo[*markerCount];
        MarkerInfo *p = (MarkerInfo *) malloc(sizeof(MarkerInfo) * (*markerCount));
        for (int i = 0; i < *markerCount; i++) {
            p[i].id = markers[i].id;
            for (int j = 0; j < 9; j++) {
                p[i].rotation[j] = markers[i].transformation.r().data[j];
            }
            for (int j = 0; j < 3; j++) {
                p[i].translation[j] = markers[i].transformation.t().data[j];
            }
            float area = contourArea(markers[i].points);
            float confidence = 1 / (0.0001 * area) / (0.0001 * area) - 1 > 0 ?
                               1 / (0.0001 * area) / (0.0001 * area) - 1 : 0;
            p[i].confidence = confidence;
        }
        return p;
    }

    void drawMarker(char *imageDataIn, char *imageDataOut, int width, int height) {

        cv::Mat img(height, width, CV_8UC4, imageDataIn);
        cv::cvtColor(img, img, CV_RGBA2BGR);
        cv::flip(img, img, 0);

        for (int i = 0; i < md->markers.size(); i++) {
            md->markers[i].draw(img);
        }
        cv::flip(img, img, 0);
        cv::cvtColor(img, img, CV_BGR2BGRA);
        memcpy(imageDataOut, img.data, width * height * 4);
        /*cv::imshow("3", dstImg);
        cv::waitKey(0);*/
    }

    void FreeNewMemory(void *pointer) {
        pointer = (MarkerInfo *) pointer;
        if (pointer != NULL) {
            free(pointer);
            //delete(pointer);
            pointer = NULL;
        }
    }

    void getIntrinsic(CalibrateInfo *info) {
        info->fx = md->camMatrix.at<float>(0, 0);
        info->fy = md->camMatrix.at<float>(1, 1);
        info->cx = md->camMatrix.at<float>(0, 2);
        info->cy = md->camMatrix.at<float>(1, 2);
        info->err = 10;
        for (int i = 0; i < 5; i++) {
            info->dist[i] = md->distCoeff.at<float>(0, i);
        }
    }

    void setIntrinsic(float f, float dx, float dy, float *distortion) {
        float dis[5];
        for (int i = 0; i < 5; i++) {
            dis[i] = distortion[i];
        }

        cv::Mat distortion_coefficients = cv::Mat(1, 5, CV_32FC1, dis);
        md->setIntrinsic(f, dx, dy, distortion_coefficients);
        imageSize = cv::Size(dx * 2, dy * 2);
        initUndistortRectifyMap(md->camMatrix, md->distCoeff, R, md->camMatrix, imageSize, CV_32FC1,
                                mapx, mapy);
    }

    void setCannyThreshold(int threshold) {
        md->cannyThreshold = threshold;
    }

    void setMode(int mode) {
        md->mode = mode;
    }

    void InitCalibrater(int boardWidth, int boardHeight) {
        calibrater = new Calibrater(cv::Size(boardWidth, boardHeight));
    }

    bool PushImageToCalibrater(char *imageDataIn, int width, int height) {
        cv::Mat image(height, width, CV_8UC4, imageDataIn);
        cv::cvtColor(image, image, CV_RGBA2BGR);
        cv::flip(image, image, 0);
        cv::transpose(image, image);
        cv::flip(image, image, 1);
        return calibrater->pushImage(image);
    }

    void Calibrate(CalibrateInfo *info) {
        calibrater->calibrate();
        info->err = calibrater->total_err / calibrater->validImageCount;
        info->fx = calibrater->cameraMatrix.at<double>(0, 0);
        info->fy = calibrater->cameraMatrix.at<double>(1, 1);
        info->cx = calibrater->cameraMatrix.at<double>(0, 2);
        info->cy = calibrater->cameraMatrix.at<double>(1, 2);
        for (int i = 0; i < 5; i++) {
            info->dist[i] = calibrater->distCoeffs.at<double>(0, i);
        }
    }

    void RestartCalibrater() {
        calibrater->restart();
    }

    void unDistortImage(char *imageDataIn, int width, int height) {
        cv::Mat image(height, width, CV_8UC4, imageDataIn);
        //cv::cvtColor(image, image, CV_RGBA2BGR);
        //cv::flip(image, image, 0);

        /*std::string s = "info: " + std::to_string(width) + " " + std::to_string(height) + std::to_string(imageSize.width) + std::to_string(imageSize.height);
        cv::Mat img = cv::Mat(600, 1500, CV_8UC3, cv::Scalar(255, 255, 255));
        cv::putText(img, s, cv::Point(200, 300), cv::FONT_HERSHEY_COMPLEX, 2, cv::Scalar(0, 0, 0), 2, 8, 0);
        cv::imshow("1", img);*/
        cv::remap(image, image, mapx, mapy, cv::INTER_LINEAR);

        //cv::flip(image, image, 0);
        //cv::cvtColor(image, image, CV_BGR2BGRA);
        memcpy(imageDataIn, image.data, width * height * 4);
    }

}
