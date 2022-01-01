#pragma once

#include "GlobalAppState.h"

#ifdef STRUCTURE_CORE

#include "RGBDSensor.h"

#include <iostream>
#include <ST/CaptureSession.h>


#include <string>

using namespace std;
using namespace ST;



class StructureCore : public RGBDSensor
{
public:
	//! Constructor; allocates CPU memory and creates handles
	StructureCore();

	//! Destructor; releases allocated ressources
	~StructureCore();

	//! Initializes the sensor
	void createFirstConnected();

	//! Processes the depth & color data
	bool processDepth();


	//! processing happends in processdepth()
	bool processColor() {
		return true;
	}

	string getSensorName() const {
		return "StructureCore";
	}

	void stopReceivingFrames() { 
		m_bIsReceivingFrames = false; 	
		//相机停止扫描
		session.stopStreaming();
	}

	//实现CaptureSessionDelegate结构体中的函数
	struct SessionDelegate : ST::CaptureSessionDelegate {
		//当相机连接状态改变时触发，输出相机状态信息
		void captureSessionEventDidOccur(ST::CaptureSession *session, ST::CaptureSessionEventId event) override {
			printf("Received capture session event %d (%s)\n", (int)event, ST::CaptureSessionSample::toString(event));
			switch (event) {
			case ST::CaptureSessionEventId::Booting: break;
			case ST::CaptureSessionEventId::Connected:
				printf("Starting streams...\n");
				printf("Sensor Serial Number is %s \n ", session->sensorInfo().serialNumber);
				//当相机连接成功时，开始扫描
				session->startStreaming();
				break;
			case ST::CaptureSessionEventId::Disconnected:
			case ST::CaptureSessionEventId::Error:
				printf("Capture session error\n");
				exit(1);
				break;
			default:
				printf("Capture session event unhandled\n");
			}
		}
		//当相机扫描到信息时触发，保存最后一帧数据
		void captureSessionDidOutputSample(ST::CaptureSession *, const ST::CaptureSessionSample& sample) override {
			cout << "正在扫描第  " << countFrame << " 帧......" << endl;
			//保存最后一帧的深度和彩色数据
			df_p = sample.depthFrame.depthInMeters();
			cf_p = sample.visibleFrame.rgbData();
			countFrame++;
		}
		//df_p、cf_p、countFrame改为用结构体成员存储
		//用指针存储最后一帧的深度和彩色数据
		const float* df_p ;
		const uint8_t* cf_p ;
		//统计扫描的帧数
		int countFrame = 0;
	};
private:

	//session、settings、delegate改为用类成员存储
	ST::CaptureSession session;
	ST::CaptureSessionSettings settings;
	SessionDelegate delegate;


	unsigned int color_width;
	unsigned int color_height;
	unsigned int depth_width;
	unsigned int depth_height;
	unsigned int frame_rate;
	float m_depth_scale;
};

#endif
