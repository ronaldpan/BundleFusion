
#include "stdafx.h"

#include "Astra.h"

#ifdef ASTRA

using namespace std;

Astra::Astra() {

	//从参数文件中读取相机分辨率、帧数、深度值的尺度
	color_width = GlobalAppState::get().s_sensorColorWidth;
	color_height = GlobalAppState::get().s_sensorColorHeight;
	depth_width = GlobalAppState::get().s_sensorDepthWidth;
	depth_height = GlobalAppState::get().s_sensorDepthHeight;
	m_depth_scale = GlobalAppState::get().s_depthScale;
	m_frame_rate = GlobalAppState::get().s_frameRate;
	//是第一帧
	isFirstFrame = 1;
	RGBDSensor::init(depth_width, depth_height, color_width, color_height, 1);

}

Astra::~Astra() {

}

void Astra::createFirstConnected() {

	//初始化Astra相机
	astra::initialize();
	reader = streamset.create_reader();

	//获取深度流
	auto depthStream = reader.stream<astra::DepthStream>();

	//配置深度流分辨率、格式和帧率
	astra::ImageStreamMode depthMode;
	//深度流的分辨率从参数文件中读取，一般为640x480
	depthMode.set_width(depth_width);
	depthMode.set_height(depth_height);
	depthMode.set_pixel_format(astra_pixel_formats::ASTRA_PIXEL_FORMAT_DEPTH_MM);
	//深度流的帧率从参数文件中读取，一般为30
	depthMode.set_fps(m_frame_rate);

	//设置深度流模式并开启深度流
	depthStream.set_mode(depthMode);
	//关闭默认的镜像模式
	depthStream.enable_mirroring(0);
	depthStream.start();

	//获取彩色流
	auto colorStream = reader.stream<astra::ColorStream>();

	//配置彩色流的分辨率、格式和帧率
	astra::ImageStreamMode colorMode;
	//彩色流的分辨率从参数文件中读取，一般为640x480
	colorMode.set_width(color_width);
	colorMode.set_height(color_height);
	colorMode.set_pixel_format(astra_pixel_formats::ASTRA_PIXEL_FORMAT_RGB888);
	//彩色流的帧率从参数文件中读取，一般为30
	colorMode.set_fps(m_frame_rate);

	//设置彩色流模式并开启彩色流
	colorStream.set_mode(colorMode);
	//关闭默认的镜像模式
	colorStream.enable_mirroring(0);
	colorStream.start();

	//初始化相机内参，其中四个参数fx，fy，cx，cy由相机对应的OpenNI SDK文件中如下代码得到
	/*Status rc = OpenNI::initialize();
	Device device;
	rc = device.open(ANY_DEVICE);
	OBCameraParams cameraParam;
	int dataSize = sizeof(cameraParam);
	memset(&cameraParam, 0, sizeof(cameraParam));
	openni::Status r3 =device.getProperty(openni::OBEXTENSION_ID_CAM_PARAMS, (uint8_t
		*)&cameraParam, &dataSize);
	int fx = cameraParam.l_intr_p[0];
	int fy = cameraParam.l_intr_p[1];
	int cx = cameraParam.l_intr_p[2];
	int cy = cameraParam.l_intr_p[3];*/
	initializeDepthIntrinsics(577, 577, 320, 248);
	initializeColorIntrinsics(577, 577, 320, 248);
	//用单位矩阵初始化相机外参
	initializeDepthExtrinsics(mat4f::identity());
	initializeColorExtrinsics(mat4f::identity());

	return;


}

bool Astra::processDepth() {
	//因为相机第一帧的彩色图像数据存在错误，导致前10帧匹配不上，故采用if对是否是第一帧进行判断，若是第一帧，则不进行保存图像数据的操作，跳过该帧，处理下一帧
	//若是第一帧
	if (isFirstFrame) {
		//循环更新相机信息直至存在新的一帧
		while (1) {
			astra_update();
			if (reader.has_new_frame())
				break;
		}
		//获取最后一帧
		auto frame = reader.get_latest_frame();

		//获取深度帧数据
		auto depthFrame = frame.get<astra::DepthFrame>();
		auto depth = depthFrame.data();

		//获取彩色帧数据
		auto colorFrame = frame.get<astra::ColorFrame>();
		auto color = colorFrame.data();

		//第一帧结束，紧接着处理下一帧
		isFirstFrame = 0;

	}

	//循环更新相机信息直至存在新的一帧
	while (1) {
		astra_update();
		if (reader.has_new_frame())
			break;
	}

	//获取最后一帧
	auto frame = reader.get_latest_frame();

	//获取深度帧数据
	auto depthFrame = frame.get<astra::DepthFrame>();
	auto depth = depthFrame.data();

	//保存深度帧数据
	if (depthFrame.is_valid())
	{
		float* depth1 = getDepthFloat();
		for (int j = 0; j < (int)getDepthWidth()*(int)getDepthHeight(); j++)
		{
			const USHORT& d = depth[j];
			if (d == 0)
				depth1[j] = -std::numeric_limits<float>::infinity();
			else {
				depth1[j] = (float)d * m_depth_scale;
			}
		}
	}

	//获取彩色帧数据
	auto colorFrame = frame.get<astra::ColorFrame>();
	auto color = colorFrame.data();

	//保存深度帧数据
	if (colorFrame.is_valid())
	{
		for (int j = 0; j < (int)getColorWidth()*(int)getColorHeight(); j++)
		{
			m_colorRGBX[j] = vec4uc((UCHAR)color[j].r, (UCHAR)color[j].g, (UCHAR)color[j].b, 255);
		}
	}

	return true;
}

void Astra::stopReceivingFrames()
{
	m_bIsReceivingFrames = false;
	astra::terminate();
}
#endif
