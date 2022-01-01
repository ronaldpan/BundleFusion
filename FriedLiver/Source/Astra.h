#pragma once

#include "GlobalAppState.h"

#ifdef ASTRA

#include "RGBDSensor.h"

#include <iostream>

using namespace std;

#include <astra/astra.hpp>
#include <astra_core/astra_core.hpp>
//#include <LitDepthVisualizer.hpp>
//#include <SFML/Graphics.hpp>
//#include <chrono>


class Astra : public RGBDSensor
{
public:
	//! Constructor; allocates CPU memory and creates handles
	Astra();

	//! Destructor; releases allocated ressources
	~Astra();

	//! Initializes the sensor
	void createFirstConnected();

	//! Processes the depth & color data
	bool processDepth();

	//! processing happends in processdepth()
	bool processColor() {
		return true;
	}

	string getSensorName() const {
		return "Astra";
	}

	void stopReceivingFrames();
private:

	unsigned int color_width;
	unsigned int color_height;
	unsigned int depth_width;
	unsigned int depth_height;
	float m_depth_scale;
	float m_frame_rate;
	//标识是否为第一帧
	bool isFirstFrame;

	astra::StreamSet streamset;
	astra::StreamReader reader;
};

#endif
