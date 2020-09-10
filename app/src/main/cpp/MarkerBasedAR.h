#pragma once

#include "MarkerDetector.h"
#include "Calibrater.h"

struct MarkerInfo
{
    int id;
    float confidence;
    float rotation[9];
    float translation[3];
};

//Marker
extern "C" void InitMarkerDetector(float f, float dx, float dy, float* distortion);
extern "C" void drawMarker(char* imageDataIn, char* imageDataOut, int width, int height);
extern "C" void* GetMarkers(char* imageDataIn, int width, int height, int* markerCount, bool tracking, int* trackStatus);
extern "C" void FreeNewMemory(void* pointer);
extern "C" void setIntrinsic(float f, float dx, float dy, float* distortion);
extern "C" void setCannyThreshold(int threshold);
extern "C" void setMode(int mode);
extern "C" void getIntrinsic(CalibrateInfo* info);

//Calibrate
extern "C" void InitCalibrater(int boardWidth, int boardHeight);
extern "C" bool PushImageToCalibrater(char* imageDataIn, int width, int height);
extern "C" void Calibrate(CalibrateInfo* info);
extern "C" void RestartCalibrater();

//UnDistort
extern "C" void unDistortImage(char* imageDataIn, int width, int height);