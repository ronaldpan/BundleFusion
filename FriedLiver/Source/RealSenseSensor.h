#pragma once

#include "GlobalAppState.h"

#ifdef REAL_SENSE

#include "RGBDSensor.h"

#include <iostream>
#include <string>
#include <librealsense2/rs.hpp>


using namespace std;
using namespace rs2;

//��������ʹ��D435i+T265ͬʱɨ��
#define T265

class RealSenseSensor : public RGBDSensor
{
public:
	//! Constructor; allocates CPU memory and creates handles
	RealSenseSensor();

	//! Destructor; releases allocated ressources
	~RealSenseSensor();

	//! Initializes the sensor
	void createFirstConnected();

	//! Processes the depth & color data
	bool processDepth();

	//! processing happends in processdepth()
	bool processColor() {
		return true;
	}

	string getSensorName() const {
		return "RealSense";
	}

	void stopReceivingFrames();

	void savePoseFile(const string fileName);

private:
	rs2::pipeline pipe;
	rs2::colorizer map;
#ifdef T265	
	//��T265�н���imu����
	rs2::pipeline pipe_t265;
	//�������һ֡��imu����
	//rs2_pose pose_data;
	vector<array<double, 8> > m_vPose;
#endif // T265


	size_t   m_poseIndex { 0 };
};
#endif