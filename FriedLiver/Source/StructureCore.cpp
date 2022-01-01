#include "stdafx.h"

#include "StructureCore.h"

#ifdef STRUCTURE_CORE


using namespace std;
using namespace ST;


//将代理线程放在类的构造函数中，使其可以一直在后台运行
StructureCore::StructureCore() {

	color_width = 640;
	color_height = 480;
	depth_width = 640;
	depth_height = 480;

	frame_rate = 30;

	m_depth_scale = 1.0;
	RGBDSensor::init(depth_width, depth_height, color_width, color_height, 1);

	//设置相机参数
	settings.source = ST::CaptureSessionSourceId::StructureCore;
	//仅用深度和彩色图像
	settings.structureCore.depthEnabled = true;
	settings.structureCore.visibleEnabled = true;
	settings.structureCore.infraredEnabled = false;
	settings.structureCore.accelerometerEnabled = false;
	settings.structureCore.gyroscopeEnabled = false;
	//可设置扫描深度的模式
	settings.structureCore.depthRangeMode = StructureCoreDepthRangeMode::VeryShort;

	settings.structureCore.depthResolution = ST::StructureCoreDepthResolution::VGA;
	settings.structureCore.imuUpdateRate = ST::StructureCoreIMUUpdateRate::AccelAndGyro_200Hz;
	//使用代理线程实时扫描，主线程继续执行后续处理操作
	session.setDelegate(&delegate);
	//准备开始扫描
	if (!session.startMonitoring(settings)) {
		printf("Failed to initialize capture session!\n");
		exit(1);
	}
	//等待相机初始化
	std::this_thread::sleep_for(std::chrono::milliseconds(2000));

}

StructureCore::~StructureCore() {

}

void StructureCore::createFirstConnected() {

	//初始化相机内参
	initializeDepthIntrinsics(557.135, 557.135, 298, 234.25);
	initializeColorIntrinsics(443.882, 443.577, 326.266, 258.954);
	// Extrinsics
	initializeDepthExtrinsics(mat4f::identity());
	initializeColorExtrinsics(mat4f::identity());
	return;
}

bool StructureCore::processDepth() {

	float* depth = getDepthFloat();
	for (int j = 0; j < (int)getDepthWidth()*(int)getDepthHeight(); j++)
	{
		//获取最后一帧的深度和彩色数据
		const float d = delegate.df_p[j];
		if (d == 0)
			depth[j] = -std::numeric_limits<float>::infinity();
		else {
			depth[j] = (float)d * m_depth_scale;
		}
	}
	for (int j = 0; j < (int)getColorWidth()*(int)getColorHeight(); j++)
	{
		m_colorRGBX[j] = vec4uc(delegate.cf_p[j * 3], delegate.cf_p[j * 3 + 1], delegate.cf_p[j * 3 + 2], 255);
	}
	return true;
}

#endif
