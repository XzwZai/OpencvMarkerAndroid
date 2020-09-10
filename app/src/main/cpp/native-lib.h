#pragma once
#include "MarkerDetector.h"
#include "Calibrater.h"

extern "C"
{
	namespace SharedObject1
	{
		struct MarkerInfo
		{
			int id;
			float confidence;
			float rotation[9];
			float translation[3];
			//float rVec[3];
		};

		void showImage(char* imageDataIn, int width, int height);
		void cvtImage(char* imageDataIn, char* imageDataOut, int width, int height);
		//void drawMarker(char* imageDataIn, char* imageDataOut, int width, int height);
		void calibrate(char* imagesDataIn, int imageCount, int width, int height, int boardWidth, int boardHeight, float* fx, float* fy, float* dx, float* dy, float* err);
		void  estimate(void* pointer, char* imageDataIn, int width, int height, float* rotation, float* translation, float* rVec, int* markerCount);
		void*  CreateMD(float f, float dx, float dy, float* distortion);
		void* GetMarkers(void* pointer, char* imageDataIn, int width, int height, int* markerCount);
		void FreeNewMemory(void* pointer);
		void setIntrinsic(void* pointer, float f, float dx, float dy, float* distortion);
		void setCannyThreshold(void* pointer, int threshold);
		void setMode(void* pointer, int mode);
        void  getIntrinsic(void* pointer, CalibrateInfo* info);

        void* CreateCalibrater(int boardWidth, int boardHeight);
        bool PushImageToCalibrater(void* pointer, char* imageDataIn, int width, int height);
        void Calibrate(void* pointer,CalibrateInfo* info);
        void RestartCalibrater(void* pointer);


	}
}

