////////////////////////////////////////////////////////////////////////////
//	File:		SiftMatch.cpp
//	Author:		Changchang Wu
//	Description :	implementation of SiftMatchGPU
//
//
//	Copyright (c) 2007 University of North Carolina at Chapel Hill
//	All Rights Reserved
//
//	Permission to use, copy, modify and distribute this software and its
//	documentation for educational, research and non-profit purposes, without
//	fee, and without a written agreement is hereby granted, provided that the
//	above copyright notice and the following paragraph appear in all copies.
//	
//	The University of North Carolina at Chapel Hill make no representations
//	about the suitability of this software for any purpose. It is provided
//	'as is' without express or implied warranty. 
//
//	Please send BUG REPORTS to ccwu@cs.unc.edu
//
////////////////////////////////////////////////////////////////////////////
#include "stdafx.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <sstream>
#include <algorithm>
using namespace std;
#include <string.h>

#include <cuda_runtime.h>
#include <cutil_inline.h>

#include "GlobalUtil.h"
#include "SiftMatch.h"

#include "CuTexImage.h"
#include "ProgramCU.h"
#include "CUDATimer.h"



SiftMatchGPU::SiftMatchGPU(int max_sift)
{
	_num_sift[0] = _num_sift[1] = 0;
	_id_sift[0] = _id_sift[1] = 0;
	_have_loc[0] = _have_loc[1] = 0;
	_max_sift = max_sift <= 0 ? 4096 : ((max_sift + 31) / 32 * 32);
	_initialized = 0;

	d_rowMatchDistances = NULL;

	_timer = new CUDATimer();
}

SiftMatchGPU::~SiftMatchGPU()
{
	if (d_rowMatchDistances) cutilSafeCall(cudaFree(d_rowMatchDistances));

	if (_timer) delete _timer;
}


void SiftMatchGPU::InitSiftMatch()
{
	//if (!CheckCudaDevice(GlobalUtil::_DeviceIndex)) {
	//	std::cout << "ERROR checking cuda device" << std::endl;
	//	return;
	//}

	if (_initialized) return;
	_initialized = 1;

	cutilSafeCall(cudaMalloc(&d_rowMatchDistances, sizeof(float) * 4096));
}

//void* SiftMatchGPU::operator new (size_t  size){
//	void * p = malloc(size);
//	if (p == 0)
//	{
//		const std::bad_alloc ba;
//		throw ba;
//	}
//	return p;
//}

//int  SiftMatchGPU::CheckCudaDevice(int device)
//{
//	return ProgramCU::CheckCudaDevice(device);
//}

void SiftMatchGPU::SetDescriptors(int index, int num, unsigned char* d_descriptors, int id)
{
	if (_initialized == 0) return;
	if (index > 1) index = 1;
	if (index < 0) index = 0;
	_have_loc[index] = 0;
	//the same feature is already set
	if (id != -1 && id == _id_sift[index]) return;
	_id_sift[index] = id;
	if (num > _max_sift) num = _max_sift;
	_num_sift[index] = num;
	_texDes[index].setImageData(8 * num, 1, 4, d_descriptors);
}

void SiftMatchGPU::SetDescriptorsFromCPU(int index, int num, const unsigned char* descriptors, int id)
{
	if (_initialized == 0) return;
	if (index > 1) index = 1;
	if (index < 0) index = 0;
	_have_loc[index] = 0;
	//the same feature is already set
	if (id != -1 && id == _id_sift[index]) return;
	_id_sift[index] = id;
	if (num > _max_sift) num = _max_sift;
	_num_sift[index] = num;
	_texDes[index].InitTexture(8 * num, 1, 4);
	_texDes[index].CopyFromHost((void*)descriptors);
}

void SiftMatchGPU::SetDescriptorsFromCPU(int index, int num, const float* descriptors, int id)
{
	if (_initialized == 0) return;
	if (index > 1) index = 1;
	if (index < 0) index = 0;
	if (num > _max_sift) num = _max_sift;

	sift_buffer.resize(num * 128 / 4);
	unsigned char * pub = (unsigned char*)&sift_buffer[0];
	for (int i = 0; i < 128 * num; ++i)
	{
		pub[i] = int(512 * descriptors[i] + 0.5);
	}
	SetDescriptorsFromCPU(index, num, pub, id);
}

void SiftMatchGPU::SetFeautreLocation(int index, const float* locations, int gap)
{
	if (_num_sift[index] <= 0) return;
	_texLoc[index].InitTexture(_num_sift[index], 1, 2);
	if (gap == 0)
	{
		_texLoc[index].CopyFromHost(locations);
	}
	else
	{
		sift_buffer.resize(_num_sift[index] * 2);
		float* pbuf = (float*)(&sift_buffer[0]);
		for (int i = 0; i < _num_sift[index]; ++i)
		{
			pbuf[i * 2] = *locations++;
			pbuf[i * 2 + 1] = *locations++;
			locations += gap;
		}
		_texLoc[index].CopyFromHost(pbuf);
	}
	_have_loc[index] = 1;

}

void SiftMatchGPU::GetOpenCVMatch(int num1, int num2, float* des1, cv::cuda::GpuMat & descriptors2GPU, ImagePairMatch& imagePairMatch, uint2 keyPointOffset)
{
	cv::cuda::GpuMat descriptors1GPU(num1, 128, CV_32F, des1);
	//cv::cuda::GpuMat descriptors2GPU(num2, 128, CV_32F, des2);
	cv::Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_L2);
	// NORM_L1, NORM_L2 are preferable choices for SIFT and SURF descriptors
	// NORM_HAMMING should be used with ORB, BRISK and BRIEF
	cv::cuda::Stream stream = cv::cuda::Stream::Stream();
	cv::cuda::GpuMat match_gm; //2行n列的GpuMat，其中n为匹配的对数，第0行是与该列匹配的点的索引号，第1行是它们的距离。
	matcher->matchAsync(descriptors1GPU, descriptors2GPU, match_gm, cv::noArray(), stream);
	//matcher->knnMatchAsync(descriptors1GPU, descriptors2GPU, match_gm, 2, cv::noArray(), stream);
	stream.waitForCompletion();
	//在GPU中将GpuMat里的特征点匹配信息保存在ImagePairMatch中
	ProgramCU::GetOpenCVMatch(match_gm, match_gm, imagePairMatch.d_keyPointIndices, imagePairMatch.d_distances, imagePairMatch.d_numMatches, keyPointOffset);
	//ProgramCU::GetOpenCVKnnMatch(match_gm, match_gm, imagePairMatch.d_keyPointIndices, imagePairMatch.d_distances, imagePairMatch.d_numMatches, keyPointOffset);
	match_gm.release();
}

void SiftMatchGPU::GetSiftMatch(int max_match, ImagePairMatch& imagePairMatch, uint2 keyPointOffset, float distmax, float ratiomax, int mutual_best_match)
{
	if (_initialized == 0 || _num_sift[0] <= 0 || _num_sift[1] <= 0) {
		cudaMemset(imagePairMatch.d_numMatches, 0, sizeof(int));
		return;
	}
	if (GlobalUtil::_EnableDetailedTimings) { _timer->startEvent("MultiplyDescriptor");	}
	ProgramCU::MultiplyDescriptor(_texDes, _texDes + 1, &_texDot, (mutual_best_match ? &_texCRT : NULL));
	if (GlobalUtil::_EnableDetailedTimings) {_timer->endEvent(); }

	GetBestMatch(max_match, imagePairMatch, distmax, ratiomax, keyPointOffset);//, mutual_best_match);
}

void SiftMatchGPU::GetBestMatch(int max_match, ImagePairMatch& imagePairMatch,  float distmax, float ratiomax, uint2 keyPointOffset)//, int mbm)
{
	_texMatch[0].InitTexture(_num_sift[0], 1);
	if (GlobalUtil::_EnableDetailedTimings) {_timer->startEvent("GetRowMatch"); }
	ProgramCU::GetRowMatch(&_texDot, _texMatch, d_rowMatchDistances, distmax, ratiomax);
	if (GlobalUtil::_EnableDetailedTimings) {_timer->endEvent();}

	//_texMatch[1].InitTexture(_num_sift[1], 1);
	if (GlobalUtil::_EnableDetailedTimings) {_timer->startEvent("GetColMatch");	}
	ProgramCU::GetColMatch(&_texCRT, distmax, ratiomax, &_texMatch[0], d_rowMatchDistances, 
		imagePairMatch.d_keyPointIndices, imagePairMatch.d_distances, imagePairMatch.d_numMatches, keyPointOffset);
	if (GlobalUtil::_EnableDetailedTimings) {_timer->endEvent(); }
}

void SiftMatchGPU::EvaluateTimings()
{
	if (!GlobalUtil::_EnableDetailedTimings) {
		std::cout << "Error timings not enabled" << std::endl;
		return;
	}
	else {
		_timer->evaluate(true);
	}
}

SiftMatchGPU* CreateNewSiftMatchGPU(int max_sift)
{
	return new SiftMatchGPU(max_sift);
}

