﻿
#include "stdafx.h"


#include "DepthSensing.h"



#include <windows.h>
#include <d3d11.h>
#include <xnamath.h>
#include "DX11Utils.h"

#include "GlobalAppState.h"
#include "TimingLogDepthSensing.h"
#include "StdOutputLogger.h"
#include "Util.h"


#include "DXUT.h"
#include "DXUTcamera.h"
#include "DXUTgui.h"
#include "DXUTsettingsDlg.h"
#include "SDKmisc.h"

#include "DX11RGBDRenderer.h"
#include "DX11QuadDrawer.h"
#include "DX11CustomRenderTarget.h"
#include "DX11PhongLighting.h"

#include "CUDASceneRepHashSDF.h"
#include "CUDARayCastSDF.h"
#include "CUDAMarchingCubesHashSDF.h"
#include "CUDAHistogramHashSDF.h"
#include "CUDASceneRepChunkGrid.h"
#include "CUDAImageManager.h"

#include "../BinaryDumpReader.h"
#include "../StructureSensor.h"
#include "../SensorDataReader.h"
#include "../TimingLog.h"

#include <iomanip>


//--------------------------------------------------------------------------------------
// UI control IDs
//--------------------------------------------------------------------------------------
#define IDC_TOGGLEFULLSCREEN      1
#define IDC_TOGGLEREF             3
#define IDC_CHANGEDEVICE          4
#define IDC_TEST                  5



//--------------------------------------------------------------------------------------
// Forward declarations 
//--------------------------------------------------------------------------------------

bool CALLBACK		ModifyDeviceSettings(DXUTDeviceSettings* pDeviceSettings, void* pUserContext);
void CALLBACK		OnFrameMove(double fTime, float fElapsedTime, void* pUserContext);
LRESULT CALLBACK	MsgProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam, bool* pbNoFurtherProcessing, void* pUserContext);
void CALLBACK		OnGUIEvent(UINT nEvent, int nControlID, CDXUTControl* pControl, void* pUserContext);
void CALLBACK		OnKeyboard(UINT nChar, bool bKeyDown, bool bAltDown, void* pUserContext);
bool CALLBACK		IsD3D11DeviceAcceptable(const CD3D11EnumAdapterInfo *AdapterInfo, UINT Output, const CD3D11EnumDeviceInfo *DeviceInfo, DXGI_FORMAT BackBufferFormat, bool bWindowed, void* pUserContext);
HRESULT CALLBACK	OnD3D11CreateDevice(ID3D11Device* pd3dDevice, const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc, void* pUserContext);
HRESULT CALLBACK	OnD3D11ResizedSwapChain(ID3D11Device* pd3dDevice, IDXGISwapChain* pSwapChain, const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc, void* pUserContext);
void CALLBACK		OnD3D11ReleasingSwapChain(void* pUserContext);
void CALLBACK		OnD3D11DestroyDevice(void* pUserContext);
void CALLBACK		OnD3D11FrameRender(ID3D11Device* pd3dDevice, ID3D11DeviceContext* pd3dImmediateContext, double fTime, float fElapsedTime, void* pUserContext);

void renderToFile(ID3D11DeviceContext* pd3dImmediateContext, const mat4f& lastRigidTransform, bool trackingLost);
void renderTopDown(ID3D11DeviceContext* pd3dImmediateContext, const mat4f& lastRigidTransform, bool trackingLost);
void renderFrustum(const mat4f& transform, const mat4f& cameraMatrix, const vec4f& color);

void RenderText();
void RenderHelp();


//--------------------------------------------------------------------------------------
// Global variables
//--------------------------------------------------------------------------------------

CDXUTDialogResourceManager	g_DialogResourceManager; // manager for shared resources of dialogs
CDXUTTextHelper*            g_pTxtHelper = NULL;
bool						g_renderText = false;
bool						g_bRenderHelp = true;

CModelViewerCamera          g_Camera;               // A model viewing camera
DX11RGBDRenderer			g_RGBDRenderer;
DX11CustomRenderTarget		g_CustomRenderTarget;
DX11CustomRenderTarget		g_RenderToFileTarget;

CUDARayCastSDF*				g_rayCast = NULL;
CUDAMarchingCubesHashSDF*	g_marchingCubesHashSDF = NULL;
CUDAHistrogramHashSDF*		g_historgram = NULL;
CUDASceneRepChunkGrid*		g_chunkGrid = NULL;
CUDASceneRepHashSDF*		g_sceneRep = NULL;

DepthCameraParams			g_depthCameraParams;
mat4f						g_lastRigidTransform = mat4f::identity();

//managed externally
CUDAImageManager*			g_CudaImageManager = NULL;
RGBDSensor*					g_depthSensingRGBDSensor = NULL;
OnlineBundler*				g_depthSensingBundler = NULL;

mat4f g_transformWorld = mat4f::identity();

//panrj
vector<mat4f> g_vImuPose;//保存每帧从t265相机得到的位姿
vector<mat4f> g_vImuPoseTransformed;//保存每帧从t265相机得到的位姿,转换到D415坐标系
int g_iLocalMatchFailed = 0, g_iGlobalMatchFailed = 0;
//panrj


void ResetDepthSensing();
void StopScanningAndExtractIsoSurfaceMC(const std::string& filename = "./scans/scan.ply", bool overwriteExistingFile = false);
//void DumpinputManagerData(const std::string& filename = "./dump/dump.sensor");

ml::mat4f getRigidTransformFromPose(int reference, int current) 
{
	//d435到t265的变化矩阵
	//mat4f t_d = {
	//0.999968402, -0.006753626, -0.004188075 ,-0.015890727,
	//-0.006685408, -0.999848172, 0.016093893 ,0.028273059,
	//-0.004296131, -0.016065384 ,-0.999861654, -0.009375589,
	//0,0,0,1
	//};

	//d415到t265的变化矩阵
	mat4f t_d = {
	0.999968402, -0.006753626, -0.004188075 ,-(0.015890727+0.0025),
	-0.006685408, -0.999848172, 0.016093893 ,0.028273059-0.001,
	-0.004296131, -0.016065384 ,-0.999861654, -(0.009375589-0.005),
	0,0,0,1
	};

	//mat4f t_d = {
	//1.0, 0.0, 0.0 , -0.03348218453390381,
	//0.0, -1.0, 0.0, 0.01412251894042127,
	//0.0, 0.0, -1.0,  -0.001145679235499528,
	//0,0,0,1
	//};

	mat4f t_d_inv = t_d.getInverse();
	//第refence帧的t265位姿(world<--T265)
	mat4f w_t0 = g_vImuPose[reference];
	mat4f w_t0_inv = w_t0.getInverse();
	//计算当前帧相对于第refence帧的变换
	mat4f transform = t_d_inv * w_t0_inv* g_vImuPose[current] * t_d;
	return transform;
	//return mat4f::identity();
}

//main()中call
int startDepthSensing(OnlineBundler* bundler, RGBDSensor* sensor, CUDAImageManager* imageManager)
{
	g_depthSensingRGBDSensor = sensor;
	g_CudaImageManager = imageManager;
	g_depthSensingBundler = bundler;
	if (GlobalAppState::get().s_generateVideo) g_transformWorld = GlobalAppState::get().s_topVideoTransformWorld;

	// Set DXUT callbacks
	DXUTSetCallbackDeviceChanging(ModifyDeviceSettings);
	DXUTSetCallbackMsgProc(MsgProc);
	DXUTSetCallbackKeyboard(OnKeyboard);
	DXUTSetCallbackFrameMove(OnFrameMove);

	DXUTSetCallbackD3D11DeviceAcceptable(IsD3D11DeviceAcceptable);
	DXUTSetCallbackD3D11DeviceCreated(OnD3D11CreateDevice); //创建各种全局对象
	DXUTSetCallbackD3D11SwapChainResized(OnD3D11ResizedSwapChain);
	DXUTSetCallbackD3D11FrameRender(OnD3D11FrameRender); //调用imageManager的process()
	DXUTSetCallbackD3D11SwapChainReleasing(OnD3D11ReleasingSwapChain);
	DXUTSetCallbackD3D11DeviceDestroyed(OnD3D11DestroyDevice);

	DXUTInit(true, true); // Parse the command line, show msgboxes on error, and an extra cmd line param to force REF for now
	DXUTSetCursorSettings(true, true); // Show the cursor and clip it when in full screen
	DXUTCreateWindow(GlobalAppState::get().s_windowWidth, GlobalAppState::get().s_windowHeight, L"Fried Liver", false);

	DXUTSetIsInGammaCorrectMode(false);	//gamma fix (for kinect color)

	DXUTCreateDevice(D3D_FEATURE_LEVEL_11_0, true, GlobalAppState::get().s_windowWidth, GlobalAppState::get().s_windowHeight);
	DXUTMainLoop(); // Enter into the DXUT render loop, 进入事件循环，OnD3D11FrameRender


	return DXUTGetExitCode();
}

//--------------------------------------------------------------------------------------
// Called right before creating a D3D9 or D3D10 device, allowing the app to modify the device settings as needed
//--------------------------------------------------------------------------------------
bool CALLBACK ModifyDeviceSettings(DXUTDeviceSettings* pDeviceSettings, void* pUserContext)
{
	// For the first device created if its a REF device, optionally display a warning dialog box
	static bool s_bFirstTime = true;
	if (s_bFirstTime)
	{
		s_bFirstTime = false;
		if ((DXUT_D3D9_DEVICE == pDeviceSettings->ver && D3DDEVTYPE_REF == pDeviceSettings->d3d9.DeviceType) ||
			(DXUT_D3D11_DEVICE == pDeviceSettings->ver && D3D_DRIVER_TYPE_REFERENCE == pDeviceSettings->d3d11.DriverType)
			)//使用参考驱动程序，精度高，速度慢
		{
			DXUTDisplaySwitchingToREFWarning(pDeviceSettings->ver); //切换至参考驱动程序时提示
		}
	}

	return true;
}

//--------------------------------------------------------------------------------------
// Handle updates to the scene
//--------------------------------------------------------------------------------------
void CALLBACK OnFrameMove(double fTime, float fElapsedTime, void* pUserContext)
{
	g_Camera.FrameMove(fElapsedTime);
	// Update the camera's position based on user input 
}

//--------------------------------------------------------------------------------------
// Render the statistics text
//--------------------------------------------------------------------------------------
void RenderText()
{
	g_pTxtHelper->Begin();
	g_pTxtHelper->SetInsertionPos(2, 0);
	g_pTxtHelper->SetForegroundColor(D3DXCOLOR(1.0f, 1.0f, 0.0f, 1.0f));
	g_pTxtHelper->DrawTextLine(DXUTGetFrameStats(DXUTIsVsyncEnabled()));
	g_pTxtHelper->DrawTextLine(DXUTGetDeviceStats());
	if (!g_bRenderHelp) {
		g_pTxtHelper->SetForegroundColor(D3DXCOLOR(1.0f, 1.0f, 1.0f, 1.0f));
		g_pTxtHelper->DrawTextLine(L"\tPress F1 for help");
	}
	g_pTxtHelper->End();


	if (g_bRenderHelp) {
		RenderHelp();
	}
}

void RenderHelp()
{
	g_pTxtHelper->Begin();
	g_pTxtHelper->SetInsertionPos(2, 40);
	g_pTxtHelper->SetForegroundColor(D3DXCOLOR(1.0f, 0.0f, 0.0f, 1.0f));
	g_pTxtHelper->DrawTextLine(L"Controls ");
	g_pTxtHelper->DrawTextLine(L"  \tF1:\t Hide help");
	g_pTxtHelper->DrawTextLine(L"  \tF2:\t Screenshot");
	g_pTxtHelper->DrawTextLine(L"  \t'R':\t Reset scan");
	g_pTxtHelper->DrawTextLine(L"  \t'9':\t Extract geometry (Marching Cubes)");
	g_pTxtHelper->DrawTextLine(L"  \t'8':\t Save recorded input data to sensor file (if enabled)");
	g_pTxtHelper->DrawTextLine(L"  \t'7':\t Stop scanning");
	g_pTxtHelper->DrawTextLine(L"  \t'6':\t Print Timings");
	g_pTxtHelper->DrawTextLine(L"  \t'<tab>':\t Switch to free-view mode");
	g_pTxtHelper->DrawTextLine(L"  \t");
	g_pTxtHelper->DrawTextLine(L"  \t'1':\t Visualize reconstruction (default)");
	g_pTxtHelper->DrawTextLine(L"  \t'2':\t Visualize input depth");
	g_pTxtHelper->DrawTextLine(L"  \t'3':\t Visualize input color");
	g_pTxtHelper->DrawTextLine(L"  \t'4':\t Visualize input normals");
	g_pTxtHelper->DrawTextLine(L"  \t'5':\t Visualize phong shaded");
	g_pTxtHelper->DrawTextLine(L"  \t'H':\t GPU hash statistics");
	g_pTxtHelper->DrawTextLine(L"  \t'T':\t Print detailed timings");
	g_pTxtHelper->DrawTextLine(L"  \t'M':\t Debug hash");
	g_pTxtHelper->DrawTextLine(L"  \t'N':\t Save hash to file");
	g_pTxtHelper->DrawTextLine(L"  \t'N':\t Load hash from file");
	g_pTxtHelper->End();
}


//--------------------------------------------------------------------------------------
// Handle messages to the application
//--------------------------------------------------------------------------------------
LRESULT CALLBACK MsgProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam, bool* pbNoFurtherProcessing,
	void* pUserContext)
{
	// Pass messages to dialog resource manager calls so GUI state is updated correctly
	*pbNoFurtherProcessing = g_DialogResourceManager.MsgProc(hWnd, uMsg, wParam, lParam);
	if (*pbNoFurtherProcessing)
		return 0;

	g_Camera.HandleMessages(hWnd, uMsg, wParam, lParam);

	return 0;
}
/*
void DumpinputManagerData(const std::string& filename)
{
TrajectoryManager* tm = g_depthSensingBundler->getTrajectoryManager();
unsigned int numFrames = g_depthSensingBundler->getNumProcessedFrames();
numFrames = std::min(numFrames, (unsigned int)tm->getFrames().size());
numFrames = std::min(numFrames, (unsigned int)g_CudaImageManager->getCurrFrameNumber() + 1);

if (numFrames == 0) return;

std::string folder = util::directoryFromPath(filename);
if (!util::directoryExists(folder)) {
util::makeDirectory(folder);
}

std::string actualFilename = filename;
while (util::fileExists(actualFilename)) {
std::string path = util::directoryFromPath(actualFilename);
std::string curr = util::fileNameFromPath(actualFilename);
std::string ext = util::getFileExtension(curr);
curr = util::removeExtensions(curr);
std::string base = util::getBaseBeforeNumericSuffix(curr);
unsigned int num = util::getNumericSuffix(curr);
if (num == (unsigned int)-1) {
num = 0;
}
actualFilename = path + base + std::to_string(num + 1) + "." + ext;
}

CalibratedSensorData cs;
cs.m_DepthImageWidth = g_CudaImageManager->getIntegrationWidth();
cs.m_DepthImageHeight = g_CudaImageManager->getIntegrationHeight();
cs.m_ColorImageWidth = g_CudaImageManager->getIntegrationWidth();
cs.m_ColorImageHeight = g_CudaImageManager->getIntegrationHeight();
cs.m_DepthNumFrames = numFrames;
cs.m_ColorNumFrames = numFrames;

cs.m_CalibrationDepth.m_Intrinsic = g_CudaImageManager->getDepthIntrinsics();
cs.m_CalibrationDepth.m_Extrinsic = g_CudaImageManager->getDepthExtrinsics();
cs.m_CalibrationDepth.m_IntrinsicInverse = g_CudaImageManager->getDepthIntrinsicsInv();
cs.m_CalibrationDepth.m_ExtrinsicInverse = g_CudaImageManager->getDepthExtrinsicsInv();
if (GlobalAppState::get().s_bUseCameraCalibration) {
cs.m_CalibrationDepth.m_Intrinsic = g_CudaImageManager->getColorIntrinsics();
cs.m_CalibrationDepth.m_IntrinsicInverse = g_CudaImageManager->getColorIntrinsicsInv();
}

cs.m_CalibrationColor.m_Intrinsic = g_CudaImageManager->getColorIntrinsics();
cs.m_CalibrationColor.m_Extrinsic = g_CudaImageManager->getDepthExtrinsics();
cs.m_CalibrationColor.m_IntrinsicInverse = g_CudaImageManager->getColorIntrinsicsInv();
cs.m_CalibrationColor.m_ExtrinsicInverse = g_CudaImageManager->getDepthExtrinsicsInv();

cs.m_DepthImages.resize(cs.m_DepthNumFrames);
cs.m_ColorImages.resize(cs.m_ColorNumFrames);
cs.m_trajectory.resize(cs.m_DepthNumFrames);

tm->lockUpdateTransforms();
for (unsigned int i = 0; i < numFrames; i++) {
const float* depth = g_CudaImageManager->getIntegrateFrame(i).getDepthFrameCPU();
const uchar4* color = g_CudaImageManager->getIntegrateFrame(i).getColorFrameCPU();

cs.m_DepthImages[i] = (float*)depth;	// this non-const cast is hacky
cs.m_ColorImages[i] = (vec4uc*)color;	// this non-const cast is hacky

const auto&f = tm->getFrames()[i];
if (f.type == TrajectoryManager::TrajectoryFrame::Invalid) {
assert(f.frameIdx == i);
for (unsigned int k = 0; k < 16; k++) {
cs.m_trajectory[i][k] = -std::numeric_limits<float>::infinity();
}
}
else {
cs.m_trajectory[i] = f.optimizedTransform;
}
}
tm->unlockUpdateTransforms();

std::cout << cs << std::endl;
std::cout << "dumping recorded frames (" << numFrames << ")... ";

BinaryDataStreamFile outStream(actualFilename, true);
//BinaryDataStreamZLibFile outStream(filename, true);
outStream << cs;
std::cout << "done" << std::endl;

//make sure we don't accidentally delete the data that doesn't belong to us
for (unsigned int i = 0; i < numFrames; i++) {
cs.m_DepthImages[i] = NULL;
cs.m_ColorImages[i] = NULL;
}
}
*/
void StopScanningAndExtractIsoSurfaceMC(const std::string& filename, bool overwriteExistingFile /*= false*/)
{
	//g_sceneRep->debugHash();
	//g_chunkGrid->debugCheckForDuplicates();
	if (GlobalAppState::get().s_sensorIdx == 7) { //! hack for structure sensor
		std::cout << "[marching cubes] stopped receiving frames from structure sensor" << std::endl;
		g_depthSensingRGBDSensor->stopReceivingFrames();
	}
	std::cout << "running marching cubes..." << std::endl;

	Timer t;

	g_marchingCubesHashSDF->clearMeshBuffer();
	if (!GlobalAppState::get().s_streamingEnabled) {
		//std::cout << "running marching cubes...streaming disenabled" << std::endl;
		//g_chunkGrid->stopMultiThreading();
		//g_chunkGrid->streamInToGPUAll();
		if (!(g_sceneRep->getCPUEnable())) //GPU
		{
			g_marchingCubesHashSDF->extractIsoSurface(g_sceneRep->getHashData(), /*g_sceneRep->getHashParams(),*/ g_rayCast->getRayCastData());
			//g_chunkGrid->startMultiThreading();
		}
		else //CPU
		{
			SAFE_DELETE(g_marchingCubesHashSDF);
			SAFE_DELETE(g_rayCast);
			g_marchingCubesHashSDF = new CUDAMarchingCubesHashSDF(false);
			g_rayCast = new CUDARayCastSDF(CUDARayCastSDF::parametersFromGlobalAppStateCpu(GlobalAppState::get(), g_CudaImageManager->getDepthIntrinsics(), g_CudaImageManager->getDepthIntrinsicsInv()));
			g_marchingCubesHashSDF->extractIsoSurfaceCPU(g_sceneRep->getHashDataCPU(), g_sceneRep->getHashParams(), g_rayCast->getRayCastDataCPU());
		}
	}
	else {
		std::cout << "running marching cubes...streaming enabled" << std::endl;
		vec4f posWorld = vec4f(g_lastRigidTransform*GlobalAppState::get().s_streamingPos, 1.0f); // trans lags one frame
		vec3f p(posWorld.x, posWorld.y, posWorld.z);
		g_marchingCubesHashSDF->extractIsoSurface(*g_chunkGrid, g_rayCast->getRayCastData(), p, GlobalAppState::getInstance().s_streamingRadius);
	}

	const mat4f& rigidTransform = mat4f::identity();//g_lastRigidTransform
	g_marchingCubesHashSDF->saveMesh(filename, &rigidTransform, overwriteExistingFile);

	std::cout << "Mesh generation time " << t.getElapsedTime() << " seconds" << std::endl;

	//g_sceneRep->debugHash();
	//g_chunkGrid->debugCheckForDuplicates();
}

void ResetDepthSensing()
{
	g_sceneRep->reset();
	g_Camera.Reset();
	if (g_chunkGrid) {
		g_chunkGrid->reset();
	}
}


void StopScanningAndSaveSDFHash(const std::string& filename = "test.hash") {
	//g_sceneRep->debugHash();
	//g_chunkGrid->debugCheckForDuplicates();

	Timer t;

	vec4f posWorld = vec4f(g_lastRigidTransform*GlobalAppState::get().s_streamingPos, 1.0f); // trans lags one frame
	vec3f p(posWorld.x, posWorld.y, posWorld.z);

	if (g_chunkGrid) {
		g_chunkGrid->saveToFile(filename, g_rayCast->getRayCastData(), p, GlobalAppState::getInstance().s_streamingRadius);
	}
	else {
		throw MLIB_EXCEPTION("chunk grid not initialized");
	}
	std::cout << "Saving Time " << t.getElapsedTime() << " seconds" << std::endl;

	//g_sceneRep->debugHash();
	//g_chunkGrid->debugCheckForDuplicates();
}


void StopScanningAndLoadSDFHash(const std::string& filename = "test.hash") {
	//g_sceneRep->debugHash();
	//g_chunkGrid->debugCheckForDuplicates();

	Timer t;

	vec4f posWorld = vec4f(g_lastRigidTransform*GlobalAppState::get().s_streamingPos, 1.0f); // trans lags one frame
	vec3f p(posWorld.x, posWorld.y, posWorld.z);

	ResetDepthSensing();
	g_chunkGrid->loadFromFile(filename, g_rayCast->getRayCastData(), p, GlobalAppState::getInstance().s_streamingRadius);

	std::cout << "Loading Time " << t.getElapsedTime() << " seconds" << std::endl;

	GlobalAppState::get().s_integrationEnabled = false;
	std::cout << "Integration enabled == false" << std::endl;
	GlobalAppState::get().s_trackingEnabled = false;
	std::cout << "Tracking enabled == false" << std::endl;

	//g_sceneRep->debugHash();
	//g_chunkGrid->debugCheckForDuplicates();
}

//--------------------------------------------------------------------------------------
// Handle key presses
//--------------------------------------------------------------------------------------
static int whichScreenshot = 0;

void StopScanningAndExit(bool aborted = false);
void CALLBACK OnKeyboard(UINT nChar, bool bKeyDown, bool bAltDown, void* pUserContext)
{
	if (!bKeyDown) return;
	//if (bKeyDown) 
	{
		wchar_t sz[200];
		switch (nChar)
		{
		case VK_F1:
			g_bRenderHelp = !g_bRenderHelp;
			break;
		case VK_F2:
			swprintf_s(sz, 200, L"screenshot%d.bmp", whichScreenshot++);
			DXUTSnapD3D11Screenshot(sz, D3DX11_IFF_BMP);
			std::wcout << std::wstring(sz) << std::endl;
			break;
		case '\t':
			g_renderText = !g_renderText;
			break;
		case '1':
			GlobalAppState::get().s_RenderMode = 1;
			break;
		case '2':
			GlobalAppState::get().s_RenderMode = 2;
			break;
		case '3':
			GlobalAppState::get().s_RenderMode = 3;
			break;
		case '4':
			GlobalAppState::get().s_RenderMode = 4;
			break;
		case '5':
		{
			if (GlobalBundlingState::get().s_enableGlobalTimings || GlobalBundlingState::get().s_enablePerFrameTimings) 
				TimingLog::printAllTimings();
			else 
				std::cout << "Cannot print timings: enable \"s_enableGlobalTimings\" or \"s_enablePerFrameTimings\" in parameter file" << std::endl;
		} break;
		case '6':
			GlobalAppState::get().s_timingsTotalEnabled = !GlobalAppState::get().s_timingsTotalEnabled;
			break;
		case '7':
			//g_depthSensingRGBDSensor->stopReceivingFrames();
			//break;
		case '8':
		{
			//if (GlobalAppState::getInstance().s_recordData) { //保存sens文件
				//if (GlobalAppState::get().s_sensorIdx == 7) { //! hack for structure sensor
				//	std::cout << "[dump frames] stopped receiving frames from sensor" << std::endl;
				//	g_depthSensingRGBDSensor->stopReceivingFrames();
				//}
				//std::vector<mat4f> trajectory;
				//g_depthSensingBundler->getTrajectoryManager()->getOptimizedTransforms(trajectory);
				//g_depthSensingRGBDSensor->saveRecordedFramesToFile(GlobalAppState::getInstance().s_recordDataFile, trajectory);
			//}
			//else {
			//	std::cout << "Cannot save recording: enable \"s_recordData\" in parameter file" << std::endl;
			//}
		} //break;
		case '9':
			std::cout << "[dump frames] stopped receiving frames from sensor" << std::endl;
			g_depthSensingRGBDSensor->stopReceivingFrames();
			//StopScanningAndExtractIsoSurfaceMC();
			//StopScanningAndExit();
			break;
		case 'T':
			GlobalAppState::get().s_timingsDetailledEnabled = !GlobalAppState::get().s_timingsDetailledEnabled;
			break;
		case 'Z': //save out correspondences and trajectory
		{
			//g_depthSensingBundler->saveGlobalSparseCorrsToFile(util::removeExtensions(GlobalAppState::get().s_binaryDumpSensorFile) + ".corrs");
			//std::vector<mat4f> trajectory; 
			//g_depthSensingBundler->getTrajectoryManager()->getOptimizedTransforms(trajectory);
			//BinaryDataStreamFile s(util::removeExtensions(GlobalAppState::get().s_binaryDumpSensorFile) + ".traj", true);
			//s << trajectory; 
			//s.close();
			//std::cout << "saved trajectory and sparse global corrs" << std::endl;
			////保存优化后的相机位姿 to file
			//std::string  trajFileName = util::removeExtensions(GlobalAppState::get().s_binaryDumpSensorFile) + ".traj";
			//std::ofstream poseFile(trajFileName);
			//for (auto & pose : trajectory)
			//{
			//	poseFile << pose;
			//}
			//poseFile.close();
			//std::cout << "保存的trajectory文件：" << trajFileName << std::endl;

		} break;
		case 'R':
			ResetDepthSensing();
			break;
		case 'H'://直方图
			g_historgram->computeHistrogram(g_sceneRep->getHashData(), g_sceneRep->getHashParams());
			break;
		case 'M':
			g_sceneRep->debugHash();
			if (g_chunkGrid)	g_chunkGrid->debugCheckForDuplicates();
			break;
		case 'E':
		{ //TODO this is just a hack to be removed
			if (GlobalAppState::get().s_sensorIdx == 3 || GlobalAppState::get().s_sensorIdx == 8) {
				std::vector<mat4f> trajectory;
				g_depthSensingBundler->getTrajectoryManager()->getOptimizedTransforms(trajectory);
				if (GlobalAppState::get().s_sensorIdx == 3) ((BinaryDumpReader*)g_depthSensingRGBDSensor)->evaluateTrajectory(trajectory);
				else										((SensorDataReader*)g_depthSensingRGBDSensor)->evaluateTrajectory(trajectory);
				BinaryDataStreamFile s("debug/opt.trajectory", true);
				s << trajectory; s.close();
				std::cout << "press key to continue" << std::endl; getchar();
			}
			else {
				std::cout << "Cannot evaluate trajectory (need sensorIdx == 3 or sensorIdx == 8)" << std::endl;
			}
		} break;
		case 'C':
		{
			std::vector<mat4f> trajectory;
			g_depthSensingBundler->getTrajectoryManager()->getOptimizedTransforms(trajectory);
			unsigned int numValidTransforms = PoseHelper::countNumValidTransforms(trajectory);
			std::cout << "#opt transforms = " << numValidTransforms << " of " << trajectory.size() << std::endl;
		} break;
		case 'N':
			StopScanningAndSaveSDFHash("test.hash");
			break;
		case 'B':
			StopScanningAndLoadSDFHash("test.hash");
			break;
		case 'I':
		{
			GlobalAppState::get().s_integrationEnabled = !GlobalAppState::get().s_integrationEnabled;
			if (GlobalAppState::get().s_integrationEnabled)		std::cout << "integration enabled" << std::endl;
			else std::cout << "integration disabled" << std::endl;
		} break;

		default:
			break;
		}
	}
}

//--------------------------------------------------------------------------------------
// Handles the GUI events
//--------------------------------------------------------------------------------------
void CALLBACK OnGUIEvent(UINT nEvent, int nControlID, CDXUTControl* pControl, void* pUserContext)
{
	switch (nControlID)
	{
		// Standard DXUT controls
	case IDC_TOGGLEFULLSCREEN:
		DXUTToggleFullScreen();
		break;
	case IDC_TOGGLEREF:
		DXUTToggleREF();
		break;
	case IDC_TEST:
		break;
	}
}

//--------------------------------------------------------------------------------------
// Reject any D3D11 devices that aren't acceptable by returning false
//--------------------------------------------------------------------------------------
bool CALLBACK IsD3D11DeviceAcceptable(const CD3D11EnumAdapterInfo *AdapterInfo, UINT Output, const CD3D11EnumDeviceInfo *DeviceInfo, DXGI_FORMAT BackBufferFormat, bool bWindowed, void* pUserContext)
{
	return true;
}

//--------------------------------------------------------------------------------------
// Create any D3D11 resources that aren't dependent on the back buffer
//--------------------------------------------------------------------------------------
HRESULT CALLBACK OnD3D11CreateDevice(ID3D11Device* pd3dDevice, const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc, void* pUserContext)
{
	HRESULT hr = S_OK;

	V_RETURN(GlobalAppState::get().OnD3D11CreateDevice(pd3dDevice));

	ID3D11DeviceContext* pd3dImmediateContext = DXUTGetD3D11DeviceContext();

	V_RETURN(g_DialogResourceManager.OnD3D11CreateDevice(pd3dDevice, pd3dImmediateContext));
	g_pTxtHelper = new CDXUTTextHelper(pd3dDevice, pd3dImmediateContext, &g_DialogResourceManager, 15);


	V_RETURN(DX11QuadDrawer::OnD3D11CreateDevice(pd3dDevice));
	V_RETURN(DX11PhongLighting::OnD3D11CreateDevice(pd3dDevice));

	TimingLogDepthSensing::init();

	std::vector<DXGI_FORMAT> formats;
	formats.push_back(DXGI_FORMAT_R32_FLOAT);
	formats.push_back(DXGI_FORMAT_R32G32B32A32_FLOAT);
	formats.push_back(DXGI_FORMAT_R32G32B32A32_FLOAT);
	formats.push_back(DXGI_FORMAT_R32G32B32A32_FLOAT);

	V_RETURN(g_RGBDRenderer.OnD3D11CreateDevice(pd3dDevice, GlobalAppState::get().s_rayCastWidth, GlobalAppState::get().s_rayCastHeight));
	V_RETURN(g_CustomRenderTarget.OnD3D11CreateDevice(pd3dDevice, GlobalAppState::get().s_rayCastWidth, GlobalAppState::get().s_rayCastHeight, formats));

	D3DXVECTOR3 vecEye(0.0f, 0.0f, 0.0f);
	D3DXVECTOR3 vecAt(0.0f, 0.0f, 1.0f);
	g_Camera.SetViewParams(&vecEye, &vecAt);


	g_sceneRep = new CUDASceneRepHashSDF(CUDASceneRepHashSDF::parametersFromGlobalAppState(GlobalAppState::get()));
	//g_rayCast = new CUDARayCastSDF(CUDARayCastSDF::parametersFromGlobalAppState(GlobalAppState::get(), g_CudaImageManager->getColorIntrinsics(), g_CudaImageManager->getColorIntrinsicsInv()));
	g_rayCast = new CUDARayCastSDF(CUDARayCastSDF::parametersFromGlobalAppState(GlobalAppState::get(), g_CudaImageManager->getDepthIntrinsics(), g_CudaImageManager->getDepthIntrinsicsInv()));

	//g_marchingCubesHashSDF = new CUDAMarchingCubesHashSDF(CUDAMarchingCubesHashSDF::parametersFromGlobalAppState(GlobalAppState::get()));
	g_marchingCubesHashSDF = new CUDAMarchingCubesHashSDF(true);
	g_historgram = new CUDAHistrogramHashSDF(g_sceneRep->getHashParams());

	if (GlobalAppState::get().s_streamingEnabled) {
		g_chunkGrid = new CUDASceneRepChunkGrid(g_sceneRep,
			GlobalAppState::get().s_streamingVoxelExtents,
			GlobalAppState::get().s_streamingGridDimensions,
			GlobalAppState::get().s_streamingMinGridPos,
			GlobalAppState::get().s_streamingInitialChunkListSize,
			GlobalAppState::get().s_streamingEnabled,
			GlobalAppState::get().s_streamingOutParts);
	}

	if (!GlobalAppState::get().s_reconstructionEnabled) {
		GlobalAppState::get().s_RenderMode = 2;
	}


	g_depthCameraParams.fx = g_CudaImageManager->getDepthIntrinsics()(0, 0);//TODO check intrinsics
	g_depthCameraParams.fy = g_CudaImageManager->getDepthIntrinsics()(1, 1);
	g_depthCameraParams.mx = g_CudaImageManager->getDepthIntrinsics()(0, 2);
	g_depthCameraParams.my = g_CudaImageManager->getDepthIntrinsics()(1, 2);
	g_depthCameraParams.m_sensorDepthWorldMin = GlobalAppState::get().s_renderDepthMin;
	g_depthCameraParams.m_sensorDepthWorldMax = GlobalAppState::get().s_renderDepthMax;
	g_depthCameraParams.m_imageWidth = g_CudaImageManager->getIntegrationWidth();
	g_depthCameraParams.m_imageHeight = g_CudaImageManager->getIntegrationHeight();
	DepthCameraData::updateParams(g_depthCameraParams);

	std::vector<DXGI_FORMAT> rtfFormat;
	rtfFormat.push_back(DXGI_FORMAT_R8G8B8A8_UNORM); // _SRGB
	V_RETURN(g_RenderToFileTarget.OnD3D11CreateDevice(pd3dDevice, GlobalAppState::get().s_rayCastWidth, GlobalAppState::get().s_rayCastHeight, rtfFormat));

	g_CudaImageManager->OnD3D11CreateDevice(pd3dDevice);

	if (GlobalAppState::get().s_sensorIdx == 7) { // structure sensor
		g_depthSensingRGBDSensor->startReceivingFrames();
	}

	return hr;
}

//--------------------------------------------------------------------------------------
// Release D3D11 resources created in OnD3D10CreateDevice 
//--------------------------------------------------------------------------------------
void CALLBACK OnD3D11DestroyDevice(void* pUserContext)
{
	g_DialogResourceManager.OnD3D11DestroyDevice();
	DXUTGetGlobalResourceCache().OnDestroyDevice();
	SAFE_DELETE(g_pTxtHelper);

	DX11QuadDrawer::OnD3D11DestroyDevice();
	DX11PhongLighting::OnD3D11DestroyDevice();
	GlobalAppState::get().OnD3D11DestroyDevice();

	g_RGBDRenderer.OnD3D11DestroyDevice();
	g_CustomRenderTarget.OnD3D11DestroyDevice();

	SAFE_DELETE(g_rayCast);
	SAFE_DELETE(g_marchingCubesHashSDF);
	SAFE_DELETE(g_historgram);
	SAFE_DELETE(g_chunkGrid);
	SAFE_DELETE(g_sceneRep);

	g_RenderToFileTarget.OnD3D11DestroyDevice();

	TimingLogDepthSensing::destroy();
}

//--------------------------------------------------------------------------------------
// Create any D3D11 resources that depend on the back buffer
//--------------------------------------------------------------------------------------
HRESULT CALLBACK OnD3D11ResizedSwapChain(ID3D11Device* pd3dDevice, IDXGISwapChain* pSwapChain,
	const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc, void* pUserContext)
{
	HRESULT hr = S_OK;

	V_RETURN(g_DialogResourceManager.OnD3D11ResizedSwapChain(pd3dDevice, pBackBufferSurfaceDesc));

	// Setup the camera's projection parameters
	g_Camera.SetWindow(pBackBufferSurfaceDesc->Width, pBackBufferSurfaceDesc->Height);
	g_Camera.SetButtonMasks(MOUSE_MIDDLE_BUTTON, MOUSE_WHEEL, MOUSE_LEFT_BUTTON);

	//g_Camera.SetRotateButtons(true, false, false);

	float fAspectRatio = pBackBufferSurfaceDesc->Width / (FLOAT)pBackBufferSurfaceDesc->Height;
	//D3DXVECTOR3 vecEye ( 0.0f, 0.0f, 0.0f );
	//D3DXVECTOR3 vecAt ( 0.0f, 0.0f, 1.0f );
	//g_Camera.SetViewParams( &vecEye, &vecAt );
	g_Camera.SetProjParams(D3DX_PI / 4, fAspectRatio, 0.1f, 10.0f);


	V_RETURN(DX11PhongLighting::OnResize(pd3dDevice, pBackBufferSurfaceDesc->Width, pBackBufferSurfaceDesc->Height));

	return hr;
}

//--------------------------------------------------------------------------------------
// Release D3D11 resources created in OnD3D10ResizedSwapChain 
//--------------------------------------------------------------------------------------
void CALLBACK OnD3D11ReleasingSwapChain(void* pUserContext)
{
	g_DialogResourceManager.OnD3D11ReleasingSwapChain();
}

//集成有效帧数据
void integrate(const DepthCameraData& depthCameraData, const mat4f& transformation)
{
	if (GlobalAppState::get().s_streamingEnabled) {
		vec4f posWorld = transformation*vec4f(GlobalAppState::getInstance().s_streamingPos, 1.0f); // trans laggs one frame *trans
		vec3f p(posWorld.x, posWorld.y, posWorld.z);

		g_chunkGrid->streamOutToCPUPass0GPU(p, GlobalAppState::get().s_streamingRadius, true, true);
		g_chunkGrid->streamInToGPUPass1GPU(true);
	}

	if (GlobalAppState::get().s_integrationEnabled) {
		unsigned int* d_bitMask = NULL;
		if (g_chunkGrid) d_bitMask = g_chunkGrid->getBitMaskGPU();
		//g_sceneRep->integrate(g_transformWorld * transformation, depthCameraData, g_depthCameraParams, d_bitMask);
		//==panrj
		if (g_sceneRep->getCPUEnable()) //CPU
			g_sceneRep->integrateCPU(g_transformWorld * transformation, depthCameraData, g_depthCameraParams, d_bitMask);
		else //GPU
			g_sceneRep->integrate(g_transformWorld * transformation, depthCameraData, g_depthCameraParams, d_bitMask);
		//==panrj
	}
	//else {
	//	//compactification is required for the ray cast splatting
	//	g_sceneRep->setLastRigidTransformAndCompactify(transformation);	//TODO check this
	//}

}

void deIntegrate(const DepthCameraData& depthCameraData, const mat4f& transformation)
{
	if (GlobalAppState::get().s_streamingEnabled) {
		vec4f posWorld = transformation*vec4f(GlobalAppState::getInstance().s_streamingPos, 1.0f); // trans laggs one frame *trans
		vec3f p(posWorld.x, posWorld.y, posWorld.z);

		g_chunkGrid->streamOutToCPUPass0GPU(p, GlobalAppState::get().s_streamingRadius, true, true);
		g_chunkGrid->streamInToGPUPass1GPU(true);
	}

	if (GlobalAppState::get().s_integrationEnabled) {
		unsigned int* d_bitMask = NULL;
		if (g_chunkGrid) d_bitMask = g_chunkGrid->getBitMaskGPU();
		g_sceneRep->deIntegrate(g_transformWorld * transformation, depthCameraData, g_depthCameraParams, d_bitMask);
	}
	//else {
	//	//compactification is required for the ray cast splatting
	//	g_sceneRep->setLastRigidTransformAndCompactify(transformation);	//TODO check this
	//}
}

void reintegrate()
{
	const unsigned int maxPerFrameFixes = GlobalAppState::get().s_maxFrameFixes;
	TrajectoryManager* tm = g_depthSensingBundler->getTrajectoryManager();
	if (tm->getNumActiveOperations() < maxPerFrameFixes) {
		tm->generateUpdateLists();
		if (tm->getNumActiveOperations() == 0) return;
		//if (GlobalBundlingState::get().s_verbose) {
		//	if (tm->getNumActiveOperations() == 0) 
		//		std::cout << __FUNCTION__ << " :  no more work (everything is reintegrated)" << std::endl;
		//}
	}

	for (unsigned int fixes = 0; fixes < maxPerFrameFixes; fixes++) {

		mat4f newTransform = mat4f::zero();
		mat4f oldTransform = mat4f::zero();
		unsigned int frameIdx = (unsigned int)-1;

		if (tm->getTopFromDeIntegrateList(oldTransform, frameIdx)) {
			auto& f = g_CudaImageManager->getIntegrateFrame(frameIdx);
			DepthCameraData depthCameraData(f.getDepthFrameGPU(), f.getColorFrameGPU());
			MLIB_ASSERT(!isnan(oldTransform[0]) && oldTransform[0] != -std::numeric_limits<float>::infinity());
			deIntegrate(depthCameraData, oldTransform);
			continue;
		}
		else if (tm->getTopFromToIntegrateList(newTransform, frameIdx)) {
			auto& f = g_CudaImageManager->getIntegrateFrame(frameIdx);
			DepthCameraData depthCameraData(f.getDepthFrameGPU(), f.getColorFrameGPU());
			MLIB_ASSERT(!isnan(newTransform[0]) && newTransform[0] != -std::numeric_limits<float>::infinity());
			integrate(depthCameraData, newTransform);
			tm->confirmIntegration(frameIdx);
			continue;
		}
		else if (tm->getTopFromReIntegrateList(oldTransform, newTransform, frameIdx)) {
			auto& f = g_CudaImageManager->getIntegrateFrame(frameIdx);
			DepthCameraData depthCameraData(f.getDepthFrameGPU(), f.getColorFrameGPU());
			MLIB_ASSERT(!isnan(oldTransform[0]) && !isnan(newTransform[0]) && oldTransform[0] != -std::numeric_limits<float>::infinity() && newTransform[0] != -std::numeric_limits<float>::infinity());
			deIntegrate(depthCameraData, oldTransform);
			integrate(depthCameraData, newTransform);
			tm->confirmIntegration(frameIdx);
			continue;
		}
		else {
			break; //no more work to do
		}
	}
	g_sceneRep->garbageCollect();
}

void visualizeFrame(ID3D11DeviceContext* pd3dImmediateContext, ID3D11Device* pd3dDevice, const mat4f& transform, bool trackingLost)
{
	if (GlobalAppState::get().s_generateVideo) return; // no need for vis here

	// If the settings dialog is being shown, then render it instead of rendering the app's scene
	//if(g_D3DSettingsDlg.IsActive())
	//{
	//	g_D3DSettingsDlg.OnRender(fElapsedTime);
	//	return;
	//}


	// Clear the back buffer
	static float ClearColor[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
	ID3D11RenderTargetView* pRTV = DXUTGetD3D11RenderTargetView();
	ID3D11DepthStencilView* pDSV = DXUTGetD3D11DepthStencilView();
	pd3dImmediateContext->ClearRenderTargetView(pRTV, ClearColor);
	pd3dImmediateContext->ClearDepthStencilView(pDSV, D3D11_CLEAR_DEPTH, 1.0f, 0);

	mat4f view = MatrixConversion::toMlib(*g_Camera.GetViewMatrix());
	view.setIdentity();	//wanna disable that for the video..
	mat4f t = mat4f::identity();
	t(1, 1) *= -1.0f;	view = t * view * t;	//t is self-inverse

	if (g_sceneRep->getNumIntegratedFrames() > 0) {
		g_sceneRep->setLastRigidTransformAndCompactify(transform);	//TODO check that
		g_rayCast->render(g_sceneRep->getHashData(), g_sceneRep->getHashParams(), transform); //光线投射
	}

	if (GlobalAppState::get().s_RenderMode == 1)	{//光线投射的深度
		//default render mode (render ray casted depth)
		const mat4f& renderIntrinsics = g_rayCast->getIntrinsics();

		//always render, irrespective whether there is a new depth frame available
		g_CustomRenderTarget.Clear(pd3dImmediateContext);
		g_CustomRenderTarget.Bind(pd3dImmediateContext);
		g_RGBDRenderer.RenderDepthMap(pd3dImmediateContext, g_rayCast->getRayCastData().d_depth, g_rayCast->getRayCastData().d_colors, 
			g_rayCast->getRayCastParams().m_width, g_rayCast->getRayCastParams().m_height, g_rayCast->getIntrinsicsInv(), 
			view, renderIntrinsics, g_CustomRenderTarget.getWidth(), g_CustomRenderTarget.getHeight(), 
			GlobalAppState::get().s_renderingDepthDiscontinuityThresOffset, GlobalAppState::get().s_renderingDepthDiscontinuityThresLin);
		g_CustomRenderTarget.Unbind(pd3dImmediateContext);

		vec3f overlayColor = vec3f(0.0f, 0.0f, 0.0f);
		if (trackingLost) { //track识败时,用特殊颜色显示
			overlayColor += vec3f(-1.0f, 0.0f, 0.0f);
		}
		DX11PhongLighting::render(pd3dImmediateContext, g_CustomRenderTarget.GetSRV(1), g_CustomRenderTarget.GetSRV(2), g_CustomRenderTarget.GetSRV(3), 
			false, g_CustomRenderTarget.getWidth(), g_CustomRenderTarget.getHeight(), overlayColor);
		DX11QuadDrawer::RenderQuad(pd3dImmediateContext, DX11PhongLighting::GetColorsSRV(), 1.0f);
#ifdef STRUCTURE_SENSOR
		if (GlobalAppState::get().s_sensorIdx == 7 && GlobalBundlingState::get().s_sendUplinkFeedbackImage) {
			ID3D11Texture2D* pSurface;
			HRESULT hr = DXUTGetDXGISwapChain()->GetBuffer(0, __uuidof(ID3D11Texture2D), reinterpret_cast<void**>(&pSurface));
			if (pSurface) {
				float* tex = (float*)CreateAndCopyToDebugTexture2D(pd3dDevice, pd3dImmediateContext, pSurface, true); //!!! TODO just copy no create
				((StructureSensor*)g_depthSensingRGBDSensor)->updateFeedbackImage((BYTE*)tex);
				SAFE_DELETE_ARRAY(tex);
			}
			SAFE_RELEASE(pSurface);
		}
#endif
	}
	else if (GlobalAppState::get().s_RenderMode == 2) {//光线投射的彩色
		//default render mode (render ray casted color)
		const mat4f& renderIntrinsics = g_rayCast->getIntrinsics();

		g_CustomRenderTarget.Clear(pd3dImmediateContext);
		g_CustomRenderTarget.Bind(pd3dImmediateContext);
		g_RGBDRenderer.RenderDepthMap(pd3dImmediateContext, g_rayCast->getRayCastData().d_depth, g_rayCast->getRayCastData().d_colors, g_rayCast->getRayCastParams().m_width, g_rayCast->getRayCastParams().m_height, g_rayCast->getIntrinsicsInv(), view, renderIntrinsics, g_CustomRenderTarget.getWidth(), g_CustomRenderTarget.getHeight(), GlobalAppState::get().s_renderingDepthDiscontinuityThresOffset, GlobalAppState::get().s_renderingDepthDiscontinuityThresLin);
		g_CustomRenderTarget.Unbind(pd3dImmediateContext);

		DX11PhongLighting::render(pd3dImmediateContext, g_CustomRenderTarget.GetSRV(1), g_CustomRenderTarget.GetSRV(2), g_CustomRenderTarget.GetSRV(3), true, g_CustomRenderTarget.getWidth(), g_CustomRenderTarget.getHeight());
		DX11QuadDrawer::RenderQuad(pd3dImmediateContext, DX11PhongLighting::GetColorsSRV(), 1.0f);
	}
	else if (GlobalAppState::get().s_RenderMode == 3) {//相机捕获的彩色
		//color input
		const uchar4* d_color = g_CudaImageManager->getLastIntegrateFrame().getColorFrameGPU();
		DX11QuadDrawer::RenderQuadDynamicUCHAR4(DXUTGetD3D11Device(), pd3dImmediateContext, d_color, g_CudaImageManager->getIntegrationWidth(), g_CudaImageManager->getIntegrationHeight());
	}
	else if (GlobalAppState::get().s_RenderMode == 4) {//相机捕获的深度
		const float* d_depth = g_CudaImageManager->getLastIntegrateFrame().getDepthFrameGPU();
		const float minDepth = GlobalAppState::get().s_sensorDepthMin;	//not that this is not the render depth!
		const float maxDepth = GlobalAppState::get().s_sensorDepthMax;	//not that this is not the render depth!
		DX11QuadDrawer::RenderQuadDynamicDEPTHasHSV(DXUTGetD3D11Device(), pd3dImmediateContext, d_depth, minDepth, maxDepth, g_CudaImageManager->getIntegrationWidth(), g_CudaImageManager->getIntegrationHeight());
	}
	else {
		std::cout << "Unknown render mode " << GlobalAppState::get().s_RenderMode << std::endl;
	}
}

void StopScanningAndExit(bool aborted /*= false*/)
{
#ifdef EVALUATE_SPARSE_CORRESPONDENCES
	g_depthSensingBundler->finishCorrespondenceEvaluatorLogging();
	std::vector<mat4f> trajectory;
	g_depthSensingBundler->getTrajectoryManager()->getOptimizedTransforms(trajectory);
	if (GlobalAppState::get().s_sensorIdx == 8) ((SensorDataReader*)g_depthSensingRGBDSensor)->evaluateTrajectory(trajectory);
#endif
#ifdef PRINT_MEM_STATS
	unsigned int heapOccCount = g_sceneRep->getHashParams().m_numSDFBlocks - g_sceneRep->getHeapFreeCount();
	std::cout << "=============== RECONSTRUCTION ===============" << std::endl;
	std::cout << "#hash buckets = " << g_sceneRep->getHashParams().m_hashNumBuckets << std::endl;
	std::cout << "#voxel blocks = " << heapOccCount << std::endl;
	std::cout << "=============== OPTIMIZATION ===============" << std::endl;
	g_depthSensingBundler->printMemStats();
#endif

	std::cout << "[ stop scanning and exit ]" << std::endl;
	if (!aborted) {
		//estimate validity of reconstruction
		bool valid = true;
		unsigned int heapFreeCount = g_sceneRep->getHeapFreeCount();
		if (heapFreeCount < 800) valid = false; // probably a messed up reconstruction (used up all the heap...)
		unsigned int numValidTransforms = 0, numTransforms = 0;
		//write trajectory
		std::vector<mat4f> trajectory;
		g_depthSensingBundler->getTrajectoryManager()->getOptimizedTransforms(trajectory);
		numValidTransforms = PoseHelper::countNumValidTransforms(trajectory);		
		numTransforms = (unsigned int)trajectory.size();
		if (numValidTransforms < (unsigned int)std::round(0.5f * numTransforms)) 
			valid = false; // not enough valid transforms
		std::cout << "#VALID TRANSFORMS = " << numValidTransforms << std::endl;

		std::string recordDataFile = GlobalAppState::getInstance().s_recordDataFile;
		if (GlobalAppState::get().s_sensorIdx != 8)
		{
			//const std::string saveFile = GlobalAppState::get().s_binaryDumpSensorFile;
			//((SensorDataReader*)g_depthSensingRGBDSensor)->saveToFile(saveFile, trajectory); //overwrite the original file
			g_depthSensingRGBDSensor->saveRecordedFramesToFile(recordDataFile, trajectory);
			std::cout << "保存的原始数据文件：" << recordDataFile << std::endl;

			if (g_depthSensingRGBDSensor->m_hasPoseSensor)
			{
				std::string actualFilename = recordDataFile;
				std::string path = util::directoryFromPath(actualFilename);
				std::string curr = util::fileNameFromPath(actualFilename);
				curr = util::removeExtensions(curr);
				std::string ext = "pose";
				actualFilename = path + curr + "." + ext;
				//保存从t265相机得到的位姿 to file
				std::ofstream poseFile(actualFilename);
				for (auto & pose:g_vImuPose)
				{
					poseFile << pose;
				}				
				poseFile.close();
				std::cout << "保存的pose文件：" << actualFilename << std::endl;
			}
			if (1)
			{
				//保存优化后的相机位姿 to file
				std::string  trajFileName = util::removeExtensions(recordDataFile) + ".traj";
				std::ofstream poseFile(trajFileName);
				for (auto & pose : trajectory)
				{
					poseFile << pose;
				}
				poseFile.close();
				std::cout << "保存的trajectory文件：" << trajFileName << std::endl;
			}
		}
		
		if (1)
		{
			//保存优化后的相机位姿 to file, 转换到D415中
			std::string  trajFileName = util::removeExtensions(recordDataFile) + "_pose_transformed.traj";
			std::ofstream poseFile(trajFileName);
			for (auto & pose : g_vImuPoseTransformed)
			{
				poseFile << pose;
			}
			poseFile.close();
			std::cout << "保存的trajectory文件：" << trajFileName << std::endl;
		}

		//if (GlobalAppState::get().s_sensorIdx == 8) ((SensorDataReader*)g_depthSensingRGBDSensor)->evaluateTrajectory(trajectory);

		//save ply
		std::cout << "[marching cubes] ";
		StopScanningAndExtractIsoSurfaceMC();
		//StopScanningAndExtractIsoSurfaceMC(util::removeExtensions(GlobalAppState::get().s_binaryDumpSensorFile) + ".ply", true); //force overwrite and existing plys
		//StopScanningAndExtractIsoSurfaceMC("debug/" + util::removeExtensions(util::fileNameFromPath(GlobalAppState::get().s_binaryDumpSensorFile)) + ".ply", true);
		std::cout << "done!" << std::endl;

		//write out confirmation file
		std::ofstream s(util::directoryFromPath(GlobalAppState::get().s_binaryDumpSensorFile) + "processed.txt");
		if (valid)  s << "valid = true" << std::endl;
		else		s << "valid = false" << std::endl;
		s << "heapFreeCount = " << heapFreeCount << std::endl;
		s << "numValidOptTransforms = " << numValidTransforms << std::endl;
		s << "numTransforms = " << numTransforms << std::endl;
		s.close();

		//==panrj //在CPU中提取mesh, 从内存中将位姿接过去 20210618 Jing
		if (GlobalAppState::get().s_SDF_CPUEnable) {
			startDepthSensingCPU(g_CudaImageManager);
			std::cout << "正在处理";
			for (int i = 0; i < trajectory.size(); i++) {
			//for (int i = 0; i < 10; i++) {
				//std::cout << "正在处理第" << i << "帧......" << std::endl;
				if (i%10==0) std::cout << ".";
				FrameRender(i, trajectory);
			}
			std::cout << std::endl;
			StopScanningAndExtractIsoSurfaceMC();
		}
		//==panrj
	}
	else {//中止
		std::ofstream s(util::directoryFromPath(GlobalAppState::get().s_binaryDumpSensorFile) + "processed.txt");
		s << "valid = false" << std::endl;
		s << "ABORTED" << std::endl; // can only be due to invalid first chunk (i think)
		s.close();
	}
	fflush(stdout);
	//exit
	//exit(0);
	//==panrj
	DXUTShutdown();//Closes down the window.  When the window closes, it will cleanup everything

	//exit(0);
	//==panrj
}

//--------------------------------------------------------------------------------------
// Render the scene using the D3D11 device
//--------------------------------------------------------------------------------------
void CALLBACK OnD3D11FrameRender(ID3D11Device* pd3dDevice, ID3D11DeviceContext* pd3dImmediateContext, double fTime, float fElapsedTime, void* pUserContext)
{
	if (ConditionManager::getExit()) {
		std::cout << "用户中止处理数据。"<<std::endl;
		StopScanningAndExit(true); //用户中止
	}

	Timer t;
	//double timeReconstruct = 0.0f;	double timeVisualize = 0.0f;	double timeReintegrate = 0.0f;

	//Start Timing
	if (GlobalBundlingState::get().s_enablePerFrameTimings) { GlobalAppState::get().WaitForGPU(); GlobalAppState::get().s_Timer.start(); }

	///////////////////////////////////////
	// Read Input
	///////////////////////////////////////
#ifdef RUN_MULTITHREADED
	ConditionManager::lockImageManagerFrameReady(ConditionManager::Recon);
	while (g_CudaImageManager->hasBundlingFrameRdy()) { //wait until bundling is done with previous frame
		ConditionManager::waitImageManagerFrameReady(ConditionManager::Recon);
	}
	bool bGotDepth = g_CudaImageManager->process();
	if (bGotDepth) {
		if (g_depthSensingRGBDSensor->m_hasPoseSensor)
			g_vImuPose.push_back(g_depthSensingRGBDSensor->m_pose);//保存每帧从t265相机得到的位姿

		g_CudaImageManager->setBundlingFrameRdy();					//ready for bundling thread
		ConditionManager::unlockAndNotifyImageManagerFrameReady(ConditionManager::Recon);
	}
	if (!g_depthSensingRGBDSensor->isReceivingFrames()) { //sequence is done
		if (bGotDepth) throw MLIB_EXCEPTION("ERROR bGotDepth = true but sequence is done");

		g_CudaImageManager->setBundlingFrameRdy();				// let bundling still optimize after scanning done
		ConditionManager::unlockAndNotifyImageManagerFrameReady(ConditionManager::Recon);
	}
#else
	bool bGotDepth = g_CudaImageManager->process();
	g_depthSensingBundler->processInput();
#endif

	///////////////////////////////////////
	// Fix old frames
	///////////////////////////////////////
	if (GlobalBundlingState::get().s_enableGlobalTimings) { GlobalAppState::get().WaitForGPU(); cudaDeviceSynchronize(); t.start(); }
	if (!GlobalAppState::get().s_binaryDumpSensorUseTrajectory)
		reintegrate(); 
	if (GlobalBundlingState::get().s_enableGlobalTimings) { 
		GlobalAppState::get().WaitForGPU(); cudaDeviceSynchronize(); t.stop(); TimingLog::getFrameTiming(true).timeReIntegrate = t.getElapsedTimeMS(); }


#ifdef RUN_MULTITHREADED
	//wait until the bundling thread is done with: sift extraction, sift matching, and key point filtering
	ConditionManager::lockBundlerProcessedInput(ConditionManager::Recon);
	while (!g_depthSensingBundler->hasProcssedInputFrame()) ConditionManager::waitBundlerProcessedInput(ConditionManager::Recon);

	if (!g_depthSensingRGBDSensor->isReceivingFrames()) { // let bundling still optimize after scanning done
		g_depthSensingBundler->confirmProcessedInputFrame();
		ConditionManager::unlockAndNotifyBundlerProcessedInput(ConditionManager::Recon);
	}
#endif

	/////////////////////////////////////// 
	// Reconstruction of current frame
	///////////////////////////////////////	
	if (GlobalBundlingState::get().s_enableGlobalTimings) { GlobalAppState::get().WaitForGPU(); cudaDeviceSynchronize(); t.start(); }
	bool bValidTransform = true; 
	bool bGlobalTrackingLost = false;
	if (bGotDepth) {
		mat4f transformation = mat4f::zero();
		unsigned int frameIdx;
		bValidTransform = g_depthSensingBundler->getCurrentIntegrationFrame(transformation, frameIdx, bGlobalTrackingLost);
#ifdef RUN_MULTITHREADED
		//allow bundler to process new frame
		g_depthSensingBundler->confirmProcessedInputFrame();
		ConditionManager::unlockAndNotifyBundlerProcessedInput(ConditionManager::Recon);
#endif

		if (GlobalAppState::get().s_binaryDumpSensorUseTrajectory && g_depthSensingRGBDSensor->m_hasPoseSensor )
			// && GlobalAppState::get().s_sensorIdx == 3)  //3=BinaryDumpReader
		{ 
			//overwrite transform and use given trajectory in this case
			transformation = getRigidTransformFromPose(0, frameIdx);
			g_vImuPoseTransformed.push_back(transformation);
			bValidTransform = true;
			bGlobalTrackingLost = false;
		}

		if (GlobalAppState::getInstance().s_recordData) {//记录frame
			g_depthSensingRGBDSensor->recordFrame(); //保存原始帧
		}

		if (bValidTransform && GlobalAppState::get().s_reconstructionEnabled) {
			DepthCameraData depthCameraData(g_CudaImageManager->getIntegrateFrame(frameIdx).getDepthFrameGPU(), g_CudaImageManager->getIntegrateFrame(frameIdx).getColorFrameGPU());
			integrate(depthCameraData, transformation); //集成到TSDF
			//保存到m_trajectoryManager中
			g_depthSensingBundler->getTrajectoryManager()->addFrame(TrajectoryManager::TrajectoryFrame::Integrated, transformation, g_CudaImageManager->getCurrFrameNumber());
		}
		else {
			g_depthSensingBundler->getTrajectoryManager()->addFrame(TrajectoryManager::TrajectoryFrame::NotIntegrated_NoTransform, 
				mat4f::zero(-std::numeric_limits<float>::infinity()), g_CudaImageManager->getCurrFrameNumber());
		}

		if (bValidTransform) {
			g_lastRigidTransform = transformation;
		}
	}
	if (GlobalBundlingState::get().s_enableGlobalTimings) { 
		GlobalAppState::get().WaitForGPU(); cudaDeviceSynchronize(); t.stop(); TimingLog::getFrameTiming(true).timeReconstruct = t.getElapsedTimeMS(); 
	}

	///////////////////////////////////////
	// Render with view of current frame 绘制
	///////////////////////////////////////
	if (GlobalBundlingState::get().s_enableGlobalTimings) { t.start(); } // just sync-ed //{ GlobalAppState::get().WaitForGPU(); cudaDeviceSynchronize(); t.start(); }
	bool trackingLost = bGotDepth && (!bValidTransform || bGlobalTrackingLost); //tracking lost when local frame has tracking lost or global frame has tracking lost
	visualizeFrame(pd3dImmediateContext, pd3dDevice, g_transformWorld * g_lastRigidTransform, trackingLost);
	if (GlobalBundlingState::get().s_enableGlobalTimings) { 
		GlobalAppState::get().WaitForGPU(); cudaDeviceSynchronize(); t.stop(); TimingLog::getFrameTiming(true).timeVisualize = t.getElapsedTimeMS(); 
	}

	///////////////////////////////////////////
	////// Bundling Optimization
	///////////////////////////////////////////
#ifndef RUN_MULTITHREADED
	g_depthSensingBundler->process(GlobalBundlingState::get().s_numLocalNonLinIterations, GlobalBundlingState::get().s_numLocalLinIterations,
		GlobalBundlingState::get().s_numGlobalNonLinIterations, GlobalBundlingState::get().s_numGlobalLinIterations);
	//g_depthSensingBundler->resetDEBUG(true);
#endif

	// Stop Timing
	std::cout << "<<< [Frame: " << g_CudaImageManager->getCurrFrameNumber() << " ] ";//<< g_sceneRep->getHeapFreeCount() 
	if (GlobalBundlingState::get().s_enablePerFrameTimings) {
		GlobalAppState::get().WaitForGPU(); 
		GlobalAppState::get().s_Timer.stop();
		TimingLog::addTotalFrameTime(GlobalAppState::get().s_Timer.getElapsedTimeMS());
		std::cout << " Frame Time:\t " << GlobalAppState::get().s_Timer.getElapsedTimeMS() << " [ms]";
	}
	std::cout << " >>>" << std::endl;

	//std::cout << VAR_NAME(timeReconstruct) << " : " << timeReconstruct << " [ms]" << std::endl;
	//std::cout << VAR_NAME(timeVisualize) << " : " << timeVisualize << " [ms]" << std::endl;
	//std::cout << VAR_NAME(timeReintegrate) << " : " << timeReintegrate << " [ms]" << std::endl;
	//std::cout << std::endl;
	//std::cout << "<<HEAP FREE>> " << g_sceneRep->getHeapFreeCount() << std::endl;
	//TimingLogDepthSensing::printTimings();

	if (g_renderText) RenderText();

	if (GlobalAppState::get().s_generateVideo) { // still renders the frames during the end optimize
		//StopScanningAndExtractIsoSurfaceMC();
		//getchar();
		//renderToFile(pd3dImmediateContext, g_transformWorld * g_lastRigidTransform, trackingLost); //TODO fix can't run these at the same time
		//std::cout << g_transformWorld << std::endl;
		renderTopDown(pd3dImmediateContext, g_transformWorld * g_lastRigidTransform, trackingLost);
		//std::cout << "waiting..." << std::endl; getchar();
	}
	if (!g_depthSensingRGBDSensor->isReceivingFrames() && !GlobalAppState::get().s_printTimingsDirectory.empty()) {
		const std::string outDir = GlobalAppState::get().s_printTimingsDirectory;
		if (!util::directoryExists(outDir)) util::makeDirectory(outDir);
		TimingLog::printAllTimings(outDir); // might skip the last frames but whatever
		//exit(1); 
	}
	// s_sensorIdx: 8=SensorDataReader (for offline processing) 读取存储的sens文件
	if (!g_depthSensingRGBDSensor->isReceivingFrames() //&& GlobalAppState::get().s_sensorIdx == 8 
		&& GlobalAppState::get().s_numSolveFramesBeforeExit != (unsigned int)-1) { //todo something better?
		static unsigned int countPastLast = 0;
		if (countPastLast >= GlobalAppState::get().s_numSolveFramesBeforeExit + 1) 
		{
			TrajectoryManager* tm = g_depthSensingBundler->getTrajectoryManager();
			if (!GlobalAppState::get().s_binaryDumpSensorUseTrajectory)
				tm->generateUpdateLists();
			//std::cout << "[NumActiveOperations] " << tm->getNumActiveOperations() << std::endl;
			if (tm->getNumActiveOperations() == 0) 
			{
				std::cout << "[no more reintegration ops] " << countPastLast << " frames past end" << std::endl;
				StopScanningAndExit();
			}
		}
		countPastLast++;
	}

	DXUT_EndPerfEvent();
}

void renderToFile(ID3D11DeviceContext* pd3dImmediateContext, const mat4f& lastRigidTransform, bool trackingLost)
{
	static unsigned int frameNumber = 0;
	std::string baseFolder = GlobalAppState::get().s_generateVideoDir;
	if (!util::directoryExists(baseFolder)) util::makeDirectory(baseFolder);
	//const std::string inputColorDir = baseFolder + "input_color/"; if (!util::directoryExists(inputColorDir)) util::makeDirectory(inputColorDir);
	//const std::string inputDepthDir = baseFolder + "input_depth/"; if (!util::directoryExists(inputDepthDir)) util::makeDirectory(inputDepthDir);
	const std::string reconstructionDir = baseFolder + "self_reconstruction/"; if (!util::directoryExists(reconstructionDir)) util::makeDirectory(reconstructionDir);
	const std::string reconstructColorDir = baseFolder + "self_reconstruction_color/"; if (!util::directoryExists(reconstructColorDir)) util::makeDirectory(reconstructColorDir);

	//reset intrinsics (no need to render at hi res)
	const mat4f origRayCastIntrinsics = g_rayCast->getIntrinsics();
	const mat4f origRayCastIntrinsicsInv = g_rayCast->getIntrinsicsInv();
	const unsigned int origRayCastWidth = g_rayCast->getRayCastParams().m_width; const unsigned int origRayCastHeight = g_rayCast->getRayCastParams().m_height;
	//g_rayCast->setRayCastIntrinsics(g_CudaImageManager->getIntegrationWidth(), g_CudaImageManager->getIntegrationHeight(), g_CudaImageManager->getColorIntrinsics(), g_CudaImageManager->getColorIntrinsicsInv());
	g_rayCast->setRayCastIntrinsics(g_CudaImageManager->getIntegrationWidth(), g_CudaImageManager->getIntegrationHeight(), g_CudaImageManager->getDepthIntrinsics(), g_CudaImageManager->getDepthIntrinsicsInv());
	g_sceneRep->setLastRigidTransformAndCompactify(lastRigidTransform);	//TODO check that
	g_rayCast->render(g_sceneRep->getHashData(), g_sceneRep->getHashParams(), lastRigidTransform);

	std::stringstream ssFrameNumber;	unsigned int numCountDigits = 6;
	for (unsigned int i = std::max(1u, (unsigned int)std::ceilf(std::log10f((float)frameNumber + 1))); i < numCountDigits; i++) ssFrameNumber << "0";
	ssFrameNumber << frameNumber;

	mat4f view = mat4f::identity();
	{	// reconstruction
		const mat4f renderIntrinsics = g_rayCast->getIntrinsics();
		g_CustomRenderTarget.Clear(pd3dImmediateContext);
		g_CustomRenderTarget.Bind(pd3dImmediateContext);
		g_RGBDRenderer.RenderDepthMap(pd3dImmediateContext, g_rayCast->getRayCastData().d_depth, g_rayCast->getRayCastData().d_colors,
			g_rayCast->getRayCastParams().m_width, g_rayCast->getRayCastParams().m_height,
			g_rayCast->getIntrinsicsInv(), view, renderIntrinsics, g_CustomRenderTarget.getWidth(), g_CustomRenderTarget.getHeight(),
			GlobalAppState::get().s_renderingDepthDiscontinuityThresOffset, GlobalAppState::get().s_renderingDepthDiscontinuityThresLin);
		g_CustomRenderTarget.Unbind(pd3dImmediateContext);

		bool colored = false;
		// tracking lost
		vec3f overlayColor = vec3f(0.0f, 0.0f, 0.0f);
		if (trackingLost) {
			overlayColor += vec3f(-1.0f, 0.0f, 0.0f);
		}

		DX11PhongLighting::render(pd3dImmediateContext, g_CustomRenderTarget.GetSRV(1), g_CustomRenderTarget.GetSRV(2), g_CustomRenderTarget.GetSRV(3),
			colored, g_CustomRenderTarget.getWidth(), g_CustomRenderTarget.getHeight(), overlayColor);

		g_RenderToFileTarget.Clear(pd3dImmediateContext);
		g_RenderToFileTarget.Bind(pd3dImmediateContext);
		DX11QuadDrawer::RenderQuad(pd3dImmediateContext, DX11PhongLighting::GetColorsSRV(), 1.0f);
		g_RenderToFileTarget.Unbind(pd3dImmediateContext);

		BYTE* data; unsigned int bytesPerElement;
		g_RenderToFileTarget.copyToHost(data, bytesPerElement);
		ColorImageR8G8B8A8 image(g_RenderToFileTarget.getWidth(), g_RenderToFileTarget.getHeight(), (vec4uc*)data);
		for (unsigned int i = 0; i < image.getWidth()*image.getHeight(); i++) {
			if (image.getData()[i].x > 0 || image.getData()[i].y > 0 || image.getData()[i].z > 0)
				image.getData()[i].w = 255;
		}
		LodePNG::save(image, reconstructionDir + ssFrameNumber.str() + ".png");
		SAFE_DELETE_ARRAY(data);
	}
	{	// reconstruction color
		const mat4f renderIntrinsics = g_rayCast->getIntrinsics();

		g_CustomRenderTarget.Clear(pd3dImmediateContext);
		g_CustomRenderTarget.Bind(pd3dImmediateContext);
		g_RGBDRenderer.RenderDepthMap(pd3dImmediateContext, g_rayCast->getRayCastData().d_depth, g_rayCast->getRayCastData().d_colors,
			g_rayCast->getRayCastParams().m_width, g_rayCast->getRayCastParams().m_height,
			g_rayCast->getIntrinsicsInv(), view, renderIntrinsics, g_CustomRenderTarget.getWidth(), g_CustomRenderTarget.getHeight(),
			GlobalAppState::get().s_renderingDepthDiscontinuityThresOffset, GlobalAppState::get().s_renderingDepthDiscontinuityThresLin);
		g_CustomRenderTarget.Unbind(pd3dImmediateContext);

		bool colored = true;

		DX11PhongLighting::render(pd3dImmediateContext, g_CustomRenderTarget.GetSRV(1), g_CustomRenderTarget.GetSRV(2), g_CustomRenderTarget.GetSRV(3),
			colored, g_CustomRenderTarget.getWidth(), g_CustomRenderTarget.getHeight());

		g_RenderToFileTarget.Clear(pd3dImmediateContext);
		g_RenderToFileTarget.Bind(pd3dImmediateContext);
		DX11QuadDrawer::RenderQuad(pd3dImmediateContext, DX11PhongLighting::GetColorsSRV(), 1.0f);
		g_RenderToFileTarget.Unbind(pd3dImmediateContext);

		BYTE* data; unsigned int bytesPerElement;
		g_RenderToFileTarget.copyToHost(data, bytesPerElement);
		ColorImageR8G8B8A8 image(g_RenderToFileTarget.getWidth(), g_RenderToFileTarget.getHeight(), (vec4uc*)data);
		for (unsigned int i = 0; i < image.getWidth()*image.getHeight(); i++) {
			if (image.getData()[i].x > 0 || image.getData()[i].y > 0 || image.getData()[i].z > 0)
				image.getData()[i].w = 255;
		}
		LodePNG::save(image, reconstructColorDir + ssFrameNumber.str() + ".png");
		SAFE_DELETE_ARRAY(data);
	}

	//// for input color/depth
	//{	// input color
	//	ColorImageR8G8B8A8 image(g_depthSensingRGBDSensor->getColorWidth(), g_depthSensingRGBDSensor->getColorHeight(), (vec4uc*)g_depthSensingRGBDSensor->getColorRGBX());
	//	for (unsigned int i = 0; i < image.getWidth()*image.getHeight(); i++) {
	//		if (image.getData()[i].x > 0 || image.getData()[i].y > 0 || image.getData()[i].z > 0)
	//			image.getData()[i].w = 255;
	//	}
	//	//FreeImageWrapper::saveImage(inputColorDir + ssFrameNumber.str() + ".png", image);
	//	LodePNG::save(image, inputColorDir + ssFrameNumber.str() + ".png");
	//}
	//{	// input depth
	//	DepthImage32 depthImage(g_depthSensingRGBDSensor->getDepthWidth(), g_depthSensingRGBDSensor->getDepthHeight(), g_depthSensingRGBDSensor->getDepthFloat());
	//	ColorImageR32G32B32A32 image(depthImage);
	//	//for (unsigned int i = 0; i < image.getWidth()*image.getHeight(); i++) {
	//	//	if (image.getData()[i].x > 0 || image.getData()[i].y > 0 || image.getData()[i].z > 0)
	//	//		image.getData()[i].w = 1.0f;
	//	//}
	//	//FreeImageWrapper::saveImage(inputDepthDir + ssFrameNumber.str() + ".png", image);
	//	ColorImageR8G8B8A8 imageU(image.getWidth(), image.getHeight());
	//	for (unsigned int i = 0; i < image.getNumPixels(); i++) {
	//		imageU.getData()[i] = vec4uc(image.getData()[i] * 255.0f);
	//	}
	//	LodePNG::save(imageU, inputDepthDir + ssFrameNumber.str() + ".png");
	//}

	//reset
	g_rayCast->setRayCastIntrinsics(origRayCastWidth, origRayCastHeight, origRayCastIntrinsics, origRayCastIntrinsicsInv);
	frameNumber++;
	if (!g_depthSensingRGBDSensor->isReceivingFrames()) {
		static unsigned int pastEndCounter = 0;
		if (pastEndCounter == 40) {
			std::cout << "DONE DONE DONE" << std::endl;
			exit(1);
		}
		pastEndCounter++;
	}

	std::cout << "waiting..." << std::endl; getchar();
}

void renderTopDown(ID3D11DeviceContext* pd3dImmediateContext, const mat4f& lastRigidTransform, bool trackingLost)
{
	static unsigned int frameNumber = 0;
	std::string baseFolder = GlobalAppState::get().s_generateVideoDir;
	if (!util::directoryExists(baseFolder)) util::makeDirectory(baseFolder);
	const std::string inputColorDir = baseFolder + "input_color/"; if (!util::directoryExists(inputColorDir)) util::makeDirectory(inputColorDir);
	const std::string inputDepthDir = baseFolder + "input_depth/"; if (!util::directoryExists(inputDepthDir)) util::makeDirectory(inputDepthDir);
	const std::string reconstructionDir = baseFolder + "reconstruction/"; if (!util::directoryExists(reconstructionDir)) util::makeDirectory(reconstructionDir);
	const std::string reconstructColorDir = baseFolder + "reconstruction_color/"; if (!util::directoryExists(reconstructColorDir)) util::makeDirectory(reconstructColorDir);

	std::stringstream ssFrameNumber;	unsigned int numCountDigits = 6;
	for (unsigned int i = std::max(1u, (unsigned int)std::ceilf(std::log10f((float)frameNumber + 1))); i < numCountDigits; i++) ssFrameNumber << "0";
	ssFrameNumber << frameNumber;

	mat4f view = mat4f::identity();

	const vec2f viewRange = GlobalAppState::get().s_topVideoMinMax;
	const vec4f pose = GlobalAppState::get().s_topVideoCameraPose;
	mat4f transform = mat4f::translation(pose[1], pose[2], pose[3]) * mat4f::rotationZ(pose[0]);

	const mat4f renderIntrinsics = g_rayCast->getIntrinsics();
	g_sceneRep->setLastRigidTransformAndCompactify(transform);
	unsigned int numOccupiedBlocks = g_sceneRep->getHashParams().m_numOccupiedBlocks;
	if (g_sceneRep->getHashParams().m_numOccupiedBlocks == 0) {
		std::cout << "ERROR nothing in the scene!" << std::endl;
		getchar();
	}
	g_rayCast->updateRayCastMinMax(viewRange.x, viewRange.y);
	g_rayCast->render(g_sceneRep->getHashData(), g_sceneRep->getHashParams(), transform);
	//g_rayCast->updateRayCastMinMax(GlobalAppState::get().s_renderDepthMin, GlobalAppState::get().s_renderDepthMax); // not technically necessary

	ColorImageR8G8B8A8 imageFrustum(g_RenderToFileTarget.getWidth(), g_RenderToFileTarget.getHeight());
	{	//frustum
		g_RenderToFileTarget.Clear(pd3dImmediateContext);
		g_RenderToFileTarget.Bind(pd3dImmediateContext);
		RGBColor c(74, 196, 237);
		if (trackingLost) c = RGBColor(127, 127, 127);
		renderFrustum(lastRigidTransform, transform.getInverse(), c.toVec4f());
		g_RenderToFileTarget.Unbind(pd3dImmediateContext);

		BYTE* data; unsigned int bytesPerElement;
		g_RenderToFileTarget.copyToHost(data, bytesPerElement);
		vec4uc* cdata = (vec4uc*)data;
		for (unsigned int y = 0; y < imageFrustum.getHeight(); y++) {
			for (unsigned int x = 0; x < imageFrustum.getWidth(); x++) {
				const vec4uc& v = cdata[y * imageFrustum.getWidth() + x];
				if (v.x > 0 || v.y > 0 || v.z > 0)
					imageFrustum(x, imageFrustum.getHeight() - y - 1) = vec4uc(v.x, v.y, v.z, 255);
			}
		}
		SAFE_DELETE_ARRAY(data);
	}
	{	// reconstruction
		g_CustomRenderTarget.Clear(pd3dImmediateContext);
		g_CustomRenderTarget.Bind(pd3dImmediateContext);
		g_RGBDRenderer.RenderDepthMap(pd3dImmediateContext, g_rayCast->getRayCastData().d_depth, g_rayCast->getRayCastData().d_colors,
			g_rayCast->getRayCastParams().m_width, g_rayCast->getRayCastParams().m_height,
			g_rayCast->getIntrinsicsInv(),
			view, renderIntrinsics, g_CustomRenderTarget.getWidth(), g_CustomRenderTarget.getHeight(),
			0.02f, 0.01f);
		//GlobalAppState::get().s_renderingDepthDiscontinuityThresOffset, GlobalAppState::get().s_renderingDepthDiscontinuityThresLin);
		g_CustomRenderTarget.Unbind(pd3dImmediateContext);

		bool colored = false;

		// tracking lost
		vec3f overlayColor = vec3f(0.0f, 0.0f, 0.0f);
		//if (trackingLost) { overlayColor += vec3f(-1.0f, 0.0f, 0.0f); }

		DX11PhongLighting::render(pd3dImmediateContext, g_CustomRenderTarget.GetSRV(1), g_CustomRenderTarget.GetSRV(2), g_CustomRenderTarget.GetSRV(3), 
			colored, g_CustomRenderTarget.getWidth(), g_CustomRenderTarget.getHeight(), overlayColor);

		g_RenderToFileTarget.Clear(pd3dImmediateContext);
		g_RenderToFileTarget.Bind(pd3dImmediateContext);
		DX11QuadDrawer::RenderQuad(pd3dImmediateContext, DX11PhongLighting::GetColorsSRV(), 1.0f);
		g_RenderToFileTarget.Unbind(pd3dImmediateContext);

		BYTE* data; unsigned int bytesPerElement;
		g_RenderToFileTarget.copyToHost(data, bytesPerElement);
		ColorImageR8G8B8A8 image(g_RenderToFileTarget.getWidth(), g_RenderToFileTarget.getHeight(), (vec4uc*)data);
		for (unsigned int i = 0; i < image.getWidth()*image.getHeight(); i++) {
			if (image.getData()[i].x > 0 || image.getData()[i].y > 0 || image.getData()[i].z > 0)
				image.getData()[i].w = 255;
			if (imageFrustum.getData()[i].x != 0 || imageFrustum.getData()[i].y != 0 || imageFrustum.getData()[i].z != 0)
				image.getData()[i] = imageFrustum.getData()[i];
		}
		LodePNG::save(image, reconstructionDir + ssFrameNumber.str() + ".png");
		SAFE_DELETE_ARRAY(data);
	}
	{	// reconstruction color
		g_CustomRenderTarget.Clear(pd3dImmediateContext);
		g_CustomRenderTarget.Bind(pd3dImmediateContext);
		g_RGBDRenderer.RenderDepthMap(pd3dImmediateContext, g_rayCast->getRayCastData().d_depth, g_rayCast->getRayCastData().d_colors,
			g_rayCast->getRayCastParams().m_width, g_rayCast->getRayCastParams().m_height,
			g_rayCast->getIntrinsicsInv(),
			view, renderIntrinsics, g_CustomRenderTarget.getWidth(), g_CustomRenderTarget.getHeight(),
			0.02f, 0.01f);
		//GlobalAppState::get().s_renderingDepthDiscontinuityThresOffset, GlobalAppState::get().s_renderingDepthDiscontinuityThresLin);
		g_CustomRenderTarget.Unbind(pd3dImmediateContext);

		bool colored = true;

		DX11PhongLighting::render(pd3dImmediateContext, g_CustomRenderTarget.GetSRV(1), g_CustomRenderTarget.GetSRV(2), g_CustomRenderTarget.GetSRV(3), colored, g_CustomRenderTarget.getWidth(), g_CustomRenderTarget.getHeight());

		g_RenderToFileTarget.Clear(pd3dImmediateContext);
		g_RenderToFileTarget.Bind(pd3dImmediateContext);
		DX11QuadDrawer::RenderQuad(pd3dImmediateContext, DX11PhongLighting::GetColorsSRV(), 1.0f);
		g_RenderToFileTarget.Unbind(pd3dImmediateContext);

		BYTE* data; unsigned int bytesPerElement;
		g_RenderToFileTarget.copyToHost(data, bytesPerElement);
		ColorImageR8G8B8A8 image(g_RenderToFileTarget.getWidth(), g_RenderToFileTarget.getHeight(), (vec4uc*)data);
		for (unsigned int i = 0; i < image.getWidth()*image.getHeight(); i++) {
			if (image.getData()[i].x > 0 || image.getData()[i].y > 0 || image.getData()[i].z > 0)
				image.getData()[i].w = 255;
			if (imageFrustum.getData()[i].x != 0 || imageFrustum.getData()[i].y != 0 || imageFrustum.getData()[i].z != 0)
				image.getData()[i] = imageFrustum.getData()[i];
		}
		//FreeImageWrapper::saveImage(reconstructColorDir + ssFrameNumber.str() + ".png", image);
		LodePNG::save(image, reconstructColorDir + ssFrameNumber.str() + ".png");
		SAFE_DELETE_ARRAY(data);
	}

	// for input color/depth
	{	// input color
		ColorImageR8G8B8A8 image(g_depthSensingRGBDSensor->getColorWidth(), g_depthSensingRGBDSensor->getColorHeight(), (vec4uc*)g_depthSensingRGBDSensor->getColorRGBX());
		for (unsigned int i = 0; i < image.getWidth()*image.getHeight(); i++) {
			if (image.getData()[i].x > 0 || image.getData()[i].y > 0 || image.getData()[i].z > 0)
				image.getData()[i].w = 255;
		}
		LodePNG::save(image, inputColorDir + ssFrameNumber.str() + ".png");
	}
	{	// input depth
		DepthImage32 depthImage(g_depthSensingRGBDSensor->getDepthWidth(), g_depthSensingRGBDSensor->getDepthHeight(), g_depthSensingRGBDSensor->getDepthFloat());
		ColorImageR32G32B32A32 image(depthImage);
		ColorImageR8G8B8A8 imageU(image.getWidth(), image.getHeight());
		for (unsigned int i = 0; i < image.getNumPixels(); i++) {
			imageU.getData()[i] = vec4uc(image.getData()[i] * 255.0f);
		}
		LodePNG::save(imageU, inputDepthDir + ssFrameNumber.str() + ".png");
	}
	//std::cout << "waiting..." << std::endl; getchar();

	frameNumber++;
	if (!g_depthSensingRGBDSensor->isReceivingFrames()) {
		static unsigned int pastEndCounter = 0;
		if (pastEndCounter == 40) {
			std::cout << "DONE DONE DONE" << std::endl;
			exit(1);
		}
		pastEndCounter++;
	}
}

void renderFrustum(const mat4f& transform, const mat4f& cameraMatrix, const vec4f& color) {
	std::vector<LineSegment3f> frustum;

	//float maxDepth = GlobalAppState::get().s_SDFMaxIntegrationDistance;
	float maxDepth = 0.30f;	//in m

	vec3f eye = vec3f(0, 0, 0);
	vec3f farPlane[4];
	const mat4f& intrinsicsInv = g_CudaImageManager->getDepthIntrinsicsInv();
	farPlane[0] = intrinsicsInv * (maxDepth * vec3f(0, 0, 1.0f));
	farPlane[1] = intrinsicsInv * (maxDepth * vec3f(g_CudaImageManager->getIntegrationWidth() - 1.0f, 0, 1.0f));
	farPlane[2] = intrinsicsInv * (maxDepth * vec3f(g_CudaImageManager->getIntegrationWidth() - 1.0f, g_CudaImageManager->getIntegrationHeight() - 1.0f, 1.0f));
	farPlane[3] = intrinsicsInv * (maxDepth * vec3f(0, g_CudaImageManager->getIntegrationHeight() - 1.0f, 1.0f));

	for (unsigned int i = 0; i < 4; i++) {
		frustum.push_back(LineSegment3f(farPlane[i], farPlane[(i + 1) % 4]));
		frustum.push_back(LineSegment3f(farPlane[i], eye));
	}

	//transform to world space
	for (auto& line : frustum) {
		line = LineSegment3f(transform * line.p0(), transform * line.p1());
	}

	float fx = g_rayCast->getIntrinsics()(0, 0);
	float fy = g_rayCast->getIntrinsics()(1, 1);
	mat4f proj = Cameraf::visionToGraphicsProj(g_RenderToFileTarget.getWidth(), g_RenderToFileTarget.getHeight(), fx, fy, 1.0f, 25.0f);

	ml::D3D11GraphicsDevice g;	g.init(DXUTGetD3D11Device(), DXUTGetD3D11DeviceContext(), DXUTGetDXGISwapChain(), DXUTGetD3D11RenderTargetView(), DXUTGetD3D11DepthStencilView());

	struct ConstantBuffer	{ ml::mat4f worldViewProj; };
	ml::D3D11ConstantBuffer<ConstantBuffer> m_constants;
	m_constants.init(g);
	ConstantBuffer cBuffer;	cBuffer.worldViewProj = proj * cameraMatrix;
	m_constants.updateAndBind(cBuffer, 0);

	MeshDataf debugMesh;

	float radius = 0.01f;
	for (auto& line : frustum) {
		auto triMesh = ml::Shapesf::cylinder(line.p0(), line.p1(), radius, 10, 10, color);
		debugMesh.merge(triMesh.computeMeshData());

		ml::D3D11TriMesh renderLine;
		renderLine.init(g, triMesh);

		g.getShaderManager().bindShaders("defaultBasic");
		renderLine.render();
	}
}

//==panrj
int startDepthSensing2(CUDAImageManager* imageManager)
{
	g_CudaImageManager = imageManager;
	if (GlobalAppState::get().s_generateVideo) g_transformWorld = GlobalAppState::get().s_topVideoTransformWorld;

	// Set DXUT callbacks
	DXUTSetCallbackDeviceChanging(ModifyDeviceSettings);
	DXUTSetCallbackMsgProc(MsgProc);
	DXUTSetCallbackKeyboard(OnKeyboard);
	DXUTSetCallbackFrameMove(OnFrameMove);

	DXUTSetCallbackD3D11DeviceAcceptable(IsD3D11DeviceAcceptable);
	DXUTSetCallbackD3D11DeviceCreated(OnD3D11CreateDevice);
	DXUTSetCallbackD3D11SwapChainResized(OnD3D11ResizedSwapChain);
	DXUTSetCallbackD3D11SwapChainReleasing(OnD3D11ReleasingSwapChain);
	DXUTSetCallbackD3D11DeviceDestroyed(OnD3D11DestroyDevice);

	DXUTInit(true, true); // Parse the command line, show msgboxes on error, and an extra cmd line param to force REF for now
	DXUTSetCursorSettings(true, true); // Show the cursor and clip it when in full screen
	DXUTCreateWindow(GlobalAppState::get().s_windowWidth, GlobalAppState::get().s_windowHeight, L"Fried Liver", false);

	DXUTSetIsInGammaCorrectMode(false);	//gamma fix (for kinect color)

	DXUTCreateDevice(D3D_FEATURE_LEVEL_11_0, false, GlobalAppState::get().s_windowWidth, GlobalAppState::get().s_windowHeight);
	//DXUTMainLoop(); // Enter into the DXUT render loop


	//新建CUDASceneRepHashSDF，用来存储sdf信息进行integrate
	//if (!GlobalAppState::get().s_SDF_CPUEnable)
	//	g_sceneRep = new CUDASceneRepHashSDF(CUDASceneRepHashSDF::parametersFromGlobalAppState(GlobalAppState::get()));
	//else
		g_sceneRep = new CUDASceneRepHashSDF(CUDASceneRepHashSDF::parametersFromGlobalAppStateCPU(GlobalAppState::get()));
	//读取基本的相机参数
	g_depthCameraParams.fx = g_CudaImageManager->getDepthIntrinsics()(0, 0);//TODO check intrinsics
	g_depthCameraParams.fy = g_CudaImageManager->getDepthIntrinsics()(1, 1);
	g_depthCameraParams.mx = g_CudaImageManager->getDepthIntrinsics()(0, 2);
	g_depthCameraParams.my = g_CudaImageManager->getDepthIntrinsics()(1, 2);
	g_depthCameraParams.m_sensorDepthWorldMin = GlobalAppState::get().s_renderDepthMin;
	g_depthCameraParams.m_sensorDepthWorldMax = GlobalAppState::get().s_renderDepthMax;
	g_depthCameraParams.m_imageWidth = g_CudaImageManager->getIntegrationWidth();
	g_depthCameraParams.m_imageHeight = g_CudaImageManager->getIntegrationHeight();
	g_sceneRep->setCameraParams(g_depthCameraParams);
	return 0;
}
int startDepthSensingCPU(CUDAImageManager* imageManager)
{
	g_CudaImageManager = imageManager;
	if (GlobalAppState::get().s_generateVideo) g_transformWorld = GlobalAppState::get().s_topVideoTransformWorld;

	//新建CUDASceneRepHashSDF，用来存储sdf信息进行integrate
	SAFE_DELETE(g_sceneRep);
	g_sceneRep = new CUDASceneRepHashSDF(CUDASceneRepHashSDF::parametersFromGlobalAppStateCPU(GlobalAppState::get()));
	//读取基本的相机参数
	g_depthCameraParams.fx = g_CudaImageManager->getDepthIntrinsics()(0, 0);//TODO check intrinsics
	g_depthCameraParams.fy = g_CudaImageManager->getDepthIntrinsics()(1, 1);
	g_depthCameraParams.mx = g_CudaImageManager->getDepthIntrinsics()(0, 2);
	g_depthCameraParams.my = g_CudaImageManager->getDepthIntrinsics()(1, 2);
	g_depthCameraParams.m_sensorDepthWorldMin = GlobalAppState::get().s_renderDepthMin;
	g_depthCameraParams.m_sensorDepthWorldMax = GlobalAppState::get().s_renderDepthMax;
	g_depthCameraParams.m_imageWidth = g_CudaImageManager->getIntegrationWidth();
	g_depthCameraParams.m_imageHeight = g_CudaImageManager->getIntegrationHeight();
	g_sceneRep->setCameraParams(g_depthCameraParams);
	return 0;
}

void FrameRender(int i, vector<mat4f> trajectory) 
{
	//对g_CudaImageManager中的彩色深度信息进行处理
	g_CudaImageManager->process();
	//用g_CudaImageManager中处理过的彩色深度信息构造depthCameraData
	DepthCameraData depthCameraData(g_CudaImageManager->getIntegrateFrame(i).getDepthFrameGPU(), g_CudaImageManager->getIntegrateFrame(i).getColorFrameGPU());
	//用depthCameraData中的彩色深度信息以及从sens文件中得到的该帧的位姿进行integrate
	integrate(depthCameraData, trajectory[i]);
}

void ExtractIsoSurfaceMC_CPU()
{
	g_rayCast = new CUDARayCastSDF(CUDARayCastSDF::parametersFromGlobalAppState(GlobalAppState::get(), g_CudaImageManager->getDepthIntrinsics(), g_CudaImageManager->getDepthIntrinsicsInv()));
	g_marchingCubesHashSDF = new CUDAMarchingCubesHashSDF(false);
	//g_marchingCubesHashSDF->clearMeshBuffer();
	//g_marchingCubesHashSDF->extractIsoSurfaceCPU(g_sceneRep->getHashDataCPU(), g_sceneRep->getHashParams(), g_rayCast->getRayCastDataCPU());
	StopScanningAndExtractIsoSurfaceMC();
	/*const mat4f& rigidTransform = mat4f::identity();
	g_marchingCubesHashSDF->saveMesh("sens.ply", &rigidTransform, false);*/


	//对生成的sdf进行debug
	//g_sceneRep->debugHash();
}
//==panrj
