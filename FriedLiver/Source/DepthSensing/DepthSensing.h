#pragma once

#include "RGBDSensor.h"
#include "TrajectoryManager.h"
#include "OnlineBundler.h"
#include "ConditionManager.h"

int startDepthSensing(OnlineBundler* bundler, RGBDSensor* sensor, CUDAImageManager* imageManager);

//==panrj
int startDepthSensing2(CUDAImageManager* imageManager);
int startDepthSensingCPU(CUDAImageManager* imageManager);
void FrameRender(int i, vector<mat4f> trajectory);
void ExtractIsoSurfaceMC_CPU();
//==panrj