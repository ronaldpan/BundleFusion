
#include "stdafx.h"

#include "Astra.h"

#ifdef ASTRA

using namespace std;

Astra::Astra() {

	//�Ӳ����ļ��ж�ȡ����ֱ��ʡ�֡�������ֵ�ĳ߶�
	color_width = GlobalAppState::get().s_sensorColorWidth;
	color_height = GlobalAppState::get().s_sensorColorHeight;
	depth_width = GlobalAppState::get().s_sensorDepthWidth;
	depth_height = GlobalAppState::get().s_sensorDepthHeight;
	m_depth_scale = GlobalAppState::get().s_depthScale;
	m_frame_rate = GlobalAppState::get().s_frameRate;
	//�ǵ�һ֡
	isFirstFrame = 1;
	RGBDSensor::init(depth_width, depth_height, color_width, color_height, 1);

}

Astra::~Astra() {

}

void Astra::createFirstConnected() {

	//��ʼ��Astra���
	astra::initialize();
	reader = streamset.create_reader();

	//��ȡ�����
	auto depthStream = reader.stream<astra::DepthStream>();

	//����������ֱ��ʡ���ʽ��֡��
	astra::ImageStreamMode depthMode;
	//������ķֱ��ʴӲ����ļ��ж�ȡ��һ��Ϊ640x480
	depthMode.set_width(depth_width);
	depthMode.set_height(depth_height);
	depthMode.set_pixel_format(astra_pixel_formats::ASTRA_PIXEL_FORMAT_DEPTH_MM);
	//�������֡�ʴӲ����ļ��ж�ȡ��һ��Ϊ30
	depthMode.set_fps(m_frame_rate);

	//���������ģʽ�����������
	depthStream.set_mode(depthMode);
	//�ر�Ĭ�ϵľ���ģʽ
	depthStream.enable_mirroring(0);
	depthStream.start();

	//��ȡ��ɫ��
	auto colorStream = reader.stream<astra::ColorStream>();

	//���ò�ɫ���ķֱ��ʡ���ʽ��֡��
	astra::ImageStreamMode colorMode;
	//��ɫ���ķֱ��ʴӲ����ļ��ж�ȡ��һ��Ϊ640x480
	colorMode.set_width(color_width);
	colorMode.set_height(color_height);
	colorMode.set_pixel_format(astra_pixel_formats::ASTRA_PIXEL_FORMAT_RGB888);
	//��ɫ����֡�ʴӲ����ļ��ж�ȡ��һ��Ϊ30
	colorMode.set_fps(m_frame_rate);

	//���ò�ɫ��ģʽ��������ɫ��
	colorStream.set_mode(colorMode);
	//�ر�Ĭ�ϵľ���ģʽ
	colorStream.enable_mirroring(0);
	colorStream.start();

	//��ʼ������ڲΣ������ĸ�����fx��fy��cx��cy�������Ӧ��OpenNI SDK�ļ������´���õ�
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
	//�õ�λ�����ʼ��������
	initializeDepthExtrinsics(mat4f::identity());
	initializeColorExtrinsics(mat4f::identity());

	return;


}

bool Astra::processDepth() {
	//��Ϊ�����һ֡�Ĳ�ɫͼ�����ݴ��ڴ��󣬵���ǰ10֡ƥ�䲻�ϣ��ʲ���if���Ƿ��ǵ�һ֡�����жϣ����ǵ�һ֡���򲻽��б���ͼ�����ݵĲ�����������֡��������һ֡
	//���ǵ�һ֡
	if (isFirstFrame) {
		//ѭ�����������Ϣֱ�������µ�һ֡
		while (1) {
			astra_update();
			if (reader.has_new_frame())
				break;
		}
		//��ȡ���һ֡
		auto frame = reader.get_latest_frame();

		//��ȡ���֡����
		auto depthFrame = frame.get<astra::DepthFrame>();
		auto depth = depthFrame.data();

		//��ȡ��ɫ֡����
		auto colorFrame = frame.get<astra::ColorFrame>();
		auto color = colorFrame.data();

		//��һ֡�����������Ŵ�����һ֡
		isFirstFrame = 0;

	}

	//ѭ�����������Ϣֱ�������µ�һ֡
	while (1) {
		astra_update();
		if (reader.has_new_frame())
			break;
	}

	//��ȡ���һ֡
	auto frame = reader.get_latest_frame();

	//��ȡ���֡����
	auto depthFrame = frame.get<astra::DepthFrame>();
	auto depth = depthFrame.data();

	//�������֡����
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

	//��ȡ��ɫ֡����
	auto colorFrame = frame.get<astra::ColorFrame>();
	auto color = colorFrame.data();

	//�������֡����
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
