#include "stdafx.h"
#include "RealSenseSensor.h"
#include <librealsense2/rs-utils.hpp>

#ifdef REAL_SENSE

RealSenseSensor::RealSenseSensor()
{
	m_colorWidth = GlobalAppState::get().s_sensorColorWidth; //640;
	m_colorHeight = GlobalAppState::get().s_sensorColorHeight; //480;
	m_depthWidth = GlobalAppState::get().s_sensorDepthWidth; //640;
	m_depthHeight = GlobalAppState::get().s_sensorDepthHeight; //480;
	m_frame_rate = GlobalAppState::get().s_frameRate; //30;
	m_depth_scale = GlobalAppState::get().s_depthScale; //1.0; //depth���׶�Ӧm_depth_scale 1
	RGBDSensor::init(m_colorWidth, m_colorHeight, m_depthWidth, m_depthHeight,1);
}


RealSenseSensor::~RealSenseSensor()
{
}

//! Initializes the sensor
void RealSenseSensor::createFirstConnected()
{
	std::string depthAndColorSensorSerialNumber;
	if (!device_with_streams({ RS2_STREAM_DEPTH, RS2_STREAM_COLOR }, depthAndColorSensorSerialNumber))
	{
		cout << "Unable to find RealSenseSensor!" << endl;
		return;
	}

	std::string poseSensorSerialNumber;
	if (device_with_streams({ RS2_STREAM_POSE }, poseSensorSerialNumber))
	{
		cout << "find pose RealSenseSensor!" << endl;
		m_hasPoseSensor = true;
	}

#if 0
	// Realsense device list
	rs2::context ctx;
	auto devs = ctx.query_devices();
	int device_num = devs.size(); //�����������Ŀ
	if (device_num <= 0) {
		cout << "Unable to find RealSenseSensor!" << endl;
		return;
	}
	// Realsense device numbers
	cout << "RealSenseSensor found,device num: " << device_num << endl;

	// first device
	rs2::device dev = devs[0];
	// serial number
	char serial_number[100] = { 0 };
	strcpy(serial_number, dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER));
	cout << "device serial_number: " << serial_number << endl;
	if (device_num > 1) //������
	{
		rs2::device dev2 = devs[1];
		// serial number
		char serial_number2[100] = { 0 };
		strcpy(serial_number2, dev2.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER));
		cout << "device2 serial_number: " << serial_number2 << endl;
	}
#endif

	//D435i��� 
	// Realsense pipeline
	rs2::pipeline pipe_init;
	pipe = pipe_init;
	// pipeline for color and depth
	rs2::config pipe_config;
	//�������к�ȷ��D435i���
	pipe_config.enable_device(depthAndColorSensorSerialNumber);
	// color stream:640*480,BGR,30 fps
	//BundleFusionʹ��RGB��ɫ�洢��ʽ����ɫ������
	pipe_config.enable_stream(RS2_STREAM_COLOR, m_colorWidth, m_colorHeight, RS2_FORMAT_RGB8, m_frame_rate);
	// depth stream:640*480,Z16,30 fps�����������
	pipe_config.enable_stream(RS2_STREAM_DEPTH, m_depthWidth, m_depthHeight, RS2_FORMAT_Z16, m_frame_rate);
	// return profile of pipeline
	rs2::pipeline_profile profile = pipe.start(pipe_config);

	// depth scale
	auto sensor = profile.get_device().first<rs2::depth_sensor>();
	m_depth_scale = sensor.get_depth_scale(); //�õ������depth scale

	// data stream
	auto depth_stream = profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
	auto color_stream = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();

	// intrinsics
	auto intrinDepth = depth_stream.get_intrinsics();
	auto intrinColor = color_stream.get_intrinsics();
	initializeDepthIntrinsics(intrinDepth.fx, intrinDepth.fy, intrinDepth.ppx, intrinDepth.ppy);
	initializeColorIntrinsics(intrinColor.fx, intrinColor.fy, intrinColor.ppx, intrinColor.ppy);
	// Extrinsics
	initializeDepthExtrinsics(mat4f::identity());
	initializeColorExtrinsics(mat4f::identity());

	if (m_hasPoseSensor)
	{
		//T265���
		rs2::pipeline pipe_init_t265;
		pipe_t265 = pipe_init_t265;
		rs2::config pipe_config_t265;
		//�������к�ȷ��T265���
		pipe_config_t265.enable_device(poseSensorSerialNumber);
		//imu������
		pipe_config_t265.enable_stream(RS2_STREAM_POSE, RS2_FORMAT_6DOF);
		//T265���ͨ���ص�������ȡimu���ݣ��������һ֡����Ϊpose_data
		auto callback_t265 = [&](const rs2::pose_frame& frame)
		{
			double timeStamp = frame.as<rs2::pose_frame>().get_timestamp() / 1000000.0; //΢�� to ��
			rs2_pose pose_data = frame.as<rs2::pose_frame>().get_pose_data();
			array<double, 8> pose = { timeStamp , pose_data.rotation.w, pose_data.rotation.x, pose_data.rotation.y, pose_data.rotation.z, 
				pose_data.translation.x, pose_data.translation.y, pose_data.translation.z };
			m_vPose.push_back(pose);
		};
		//�Իص���������ʽ����T265�����imu������
		rs2::pipeline_profile profile_t265 = pipe_t265.start(pipe_config_t265, callback_t265);
	}

	return;
}

//! Processes the depth & color data
bool RealSenseSensor::processDepth()
{
	rs2::frameset frames;
	if(pipe.poll_for_frames(&frames)){
		// be full of depth api
		rs2::frame depth_frame = frames.first(RS2_STREAM_DEPTH);
		const USHORT* ddata = (USHORT*)depth_frame.get_data();
		float* depth = getDepthFloat();
		for (int j = 0; j < (int)getDepthWidth()*(int)getDepthHeight(); j++)
		{
			const USHORT& d = ddata[j];
			if (d == 0)//��Ч���ֵ
				depth[j] = -std::numeric_limits<float>::infinity();
			else
				depth[j] = (float)d * m_depth_scale;
		}

		// be full of color api
		rs2::frame color_frame = frames.first(RS2_STREAM_COLOR);
		const UCHAR* cdata = (UCHAR*)color_frame.get_data();;
		for (int j = 0; j < (int)getColorWidth()*(int)getColorHeight(); j++)
		{
			m_colorRGBX[j] = vec4uc(cdata[j * 3], cdata[j * 3 + 1], cdata[j * 3 + 2], 255);
		}

		m_timeStampDepth = depth_frame.get_timestamp() / 1000000.0;//�������һ֡���ͼ��ʱ�������λ��΢��ת��Ϊ��
		m_timeStampColor = color_frame.get_timestamp() / 1000000.0;
	}

	if (m_hasPoseSensor)
	{
		while (m_timeStampDepth < m_vPose[m_poseIndex][0] && m_poseIndex< m_vPose.size()-1)
			m_poseIndex++;

		//����T265����е�imu����
		//����m_pose�ڻ���RGBDSensor�ж��壬Ϊ�뱾������ͼ��֡ʱ��ͬ����imu֡�����ݾ�������õ��Ĵ�T265�������ϵ����������ϵ�ı任����
		//��imu�����е���Ԫ������õ��任�����е���ת����
		float w = m_vPose[m_poseIndex][1];
		float x = m_vPose[m_poseIndex][2];
		float y = m_vPose[m_poseIndex][3];
		float z = m_vPose[m_poseIndex][4];

		m_pose._m00 = 1 - 2 * y*y - 2 * z*z;
		m_pose._m01 = 2 * x*y - 2 * w*z;
		m_pose._m02 = 2 * x*z + 2 * w*y;

		m_pose._m10 = 2 * x*y + 2 * w*z;
		m_pose._m11 = 1 - 2 * x*x - 2 * z*z;
		m_pose._m12 = 2 * y*z - 2 * w*x;

		m_pose._m20 = 2 * x*z - 2 * w*y;
		m_pose._m21 = 2 * y*z + 2 * w*x;
		m_pose._m22 = 1 - 2 * x*x - 2 * y*y;

		//��imu�����е�ƽ�����õ��任�����е�ƽ�ƾ���
		m_pose._m03 = m_vPose[m_poseIndex][5];
		m_pose._m13 = m_vPose[m_poseIndex][6];
		m_pose._m23 = m_vPose[m_poseIndex][7];

		//��λ��任����
		m_pose._m30 = 0;	m_pose._m31 = 0;	m_pose._m32 = 0;	m_pose._m33 = 1;
	}
	return true;
}

void RealSenseSensor::stopReceivingFrames()
{
	m_bIsReceivingFrames = false;
	pipe.stop();
	if (m_hasPoseSensor) {
		pipe_t265.stop();
		savePoseFile("t265_pose.txt");
	}
}

void RealSenseSensor::savePoseFile(const string fileName)
{
	std::ofstream outFile(fileName);
	outFile.setf(ios::fixed, ios::floatfield);
	//���þ���
	outFile.precision(16);
	//д���ͷ
	//outFile<< "#timestamp [s]" << ',' << "a_RS_S_x [m s^-2]" << ',' << "a_RS_S_y [m s^-2]" << ',' << "a_RS_S_z [m s^-2]" << std::endl;
	//д��ÿ����Ϣ
	for (array<double, 8> element : m_vPose) {
		for (int j = 0; j < 7; j++)
			outFile << element[j] << ',';
		outFile << element[7] << std::endl;
	}
	//�ر��ļ�
	outFile.close();

	return;
}
#endif