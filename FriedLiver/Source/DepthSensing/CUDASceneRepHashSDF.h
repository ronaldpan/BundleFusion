#pragma once

#include <cutil_inline.h>
#include <cutil_math.h>

#include "MatrixConversion.h"
#include "VoxelUtilHashSDF.h"
#include "DepthCameraUtil.h"
#include "CUDAScan.h"
#include "CUDATimer.h"

#include "GlobalAppState.h"
#include "TimingLogDepthSensing.h"

//==panrj
#define T_PER_BLOCK 8
#define COMPACTIFY_HASH_THREADS_PER_BLOCK 256
#define SDF_BLOCK_SIZE 8
#define MINF -std::numeric_limits<float>::infinity()
#define PINF +std::numeric_limits<float>::infinity()
//==panrj

extern "C" void resetCUDA(HashDataStruct& hashData, const HashParams& hashParams);
extern "C" void resetHashBucketMutexCUDA(HashDataStruct& hashData, const HashParams& hashParams);
extern "C" void allocCUDA(HashDataStruct& hashData, const HashParams& hashParams, const DepthCameraData& depthCameraData, const DepthCameraParams& depthCameraParams, const unsigned int* d_bitMask);
extern "C" void fillDecisionArrayCUDA(HashDataStruct& hashData, const HashParams& hashParams);
extern "C" void compactifyHashCUDA(HashDataStruct& hashData, const HashParams& hashParams);
extern "C" unsigned int compactifyHashAllInOneCUDA(HashDataStruct& hashData, const HashParams& hashParams);
extern "C" void integrateDepthMapCUDA(HashDataStruct& hashData, const HashParams& hashParams, const DepthCameraData& depthCameraData, const DepthCameraParams& depthCameraParams);
extern "C" void deIntegrateDepthMapCUDA(HashDataStruct& hashData, const HashParams& hashParams, const DepthCameraData& depthCameraData, const DepthCameraParams& depthCameraParams);
extern "C" void bindInputDepthColorTextures(const DepthCameraData& depthCameraData, unsigned int width, unsigned int height);

extern "C" void starveVoxelsKernelCUDA(HashDataStruct& hashData, const HashParams& hashParams);
extern "C" void garbageCollectIdentifyCUDA(HashDataStruct& hashData, const HashParams& hashParams);
extern "C" void garbageCollectFreeCUDA(HashDataStruct& hashData, const HashParams& hashParams);

class CUDASceneRepHashSDF
{
public:
	CUDASceneRepHashSDF(const HashParams& params) {
		s_SDF_CPUEnable = params.m_bSDF_CPUEnable;//是否启用cpu
		if (s_SDF_CPUEnable)
			std::cout << "放在cpu上执行" << std::endl;
		else
			std::cout << "放在gpu上执行" << std::endl;
		create(params);
	}

	~CUDASceneRepHashSDF() {
		//destroy();
		m_hashData.free();
	}

	static HashParams parametersFromGlobalAppState(const GlobalAppState& gas) {
		HashParams params;
		params.m_rigidTransform.setIdentity();
		params.m_rigidTransformInverse.setIdentity();
		params.m_hashNumBuckets = gas.s_hashNumBuckets;
		params.m_hashBucketSize = HASH_BUCKET_SIZE;
		params.m_hashMaxCollisionLinkedListSize = gas.s_hashMaxCollisionLinkedListSize;
		params.m_SDFBlockSize = SDF_BLOCK_SIZE;
		params.m_numSDFBlocks = gas.s_hashNumSDFBlocks;
		params.m_virtualVoxelSize = gas.s_SDFVoxelSize;
		params.m_maxIntegrationDistance = gas.s_SDFMaxIntegrationDistance;
		params.m_truncation = gas.s_SDFTruncation;
		params.m_truncScale = gas.s_SDFTruncationScale;
		params.m_integrationWeightSample = gas.s_SDFIntegrationWeightSample;
		params.m_integrationWeightMax = gas.s_SDFIntegrationWeightMax;
		params.m_streamingVoxelExtents = MatrixConversion::toCUDA(gas.s_streamingVoxelExtents);
		params.m_streamingGridDimensions = MatrixConversion::toCUDA(gas.s_streamingGridDimensions);
		params.m_streamingMinGridPos = MatrixConversion::toCUDA(gas.s_streamingMinGridPos);
		params.m_streamingInitialChunkListSize = gas.s_streamingInitialChunkListSize;

		//==panrj
		params.m_bSDF_CPUEnable = false;
		//==panrj
		return params;
	}

	static HashParams parametersFromGlobalAppStateCPU(const GlobalAppState& gas) {
		HashParams params;

		params.m_rigidTransform.setIdentity();
		params.m_rigidTransformInverse.setIdentity();
		params.m_hashNumBuckets = gas.s_hashNumBucketsC;
		params.m_hashBucketSize = HASH_BUCKET_SIZE;
		params.m_hashMaxCollisionLinkedListSize = gas.s_hashMaxCollisionLinkedListSizeC;
		params.m_SDFBlockSize = SDF_BLOCK_SIZE;
		params.m_numSDFBlocks = gas.s_hashNumSDFBlocksC;
		params.m_virtualVoxelSize = gas.s_SDFVoxelSizeC;
		params.m_maxIntegrationDistance = gas.s_SDFMaxIntegrationDistanceC;
		params.m_truncation = gas.s_SDFTruncationC;
		params.m_truncScale = gas.s_SDFTruncationScaleC;
		params.m_integrationWeightSample = gas.s_SDFIntegrationWeightSampleC;
		params.m_integrationWeightMax = gas.s_SDFIntegrationWeightMaxC;
		params.m_streamingVoxelExtents = MatrixConversion::toCUDA(gas.s_streamingVoxelExtents);
		params.m_streamingGridDimensions = MatrixConversion::toCUDA(gas.s_streamingGridDimensions);
		params.m_streamingMinGridPos = MatrixConversion::toCUDA(gas.s_streamingMinGridPos);
		params.m_streamingInitialChunkListSize = gas.s_streamingInitialChunkListSize;

		params.m_bSDF_CPUEnable = true;
		std::cout << "读取cpu配置..." << std::endl;

		return params;
	}

	void bindDepthCameraTextures(const DepthCameraData& depthCameraData, const DepthCameraParams& depthCameraParams) {
		bindInputDepthColorTextures(depthCameraData, depthCameraParams.m_imageWidth, depthCameraParams.m_imageHeight);//CUDA函数
	}

	void integrate(const mat4f& lastRigidTransform, const DepthCameraData& depthCameraData, const DepthCameraParams& depthCameraParams, unsigned int* d_bitMask) {
		
		bindDepthCameraTextures(depthCameraData, depthCameraParams);

		setLastRigidTransform(lastRigidTransform);

		//allocate all hash blocks which are corresponding to depth map entries
		alloc(depthCameraData, depthCameraParams, d_bitMask);

		//generate a linear hash array with only occupied entries
		compactifyHashEntries();

		//volumetrically integrate the depth data into the depth SDFBlocks
		integrateDepthMap(depthCameraData, depthCameraParams);

		//garbageCollect();

		m_numIntegratedFrames++;
	}

	void deIntegrate(const mat4f& lastRigidTransform, const DepthCameraData& depthCameraData, const DepthCameraParams& depthCameraParams, unsigned int* d_bitMask) {

		bindDepthCameraTextures(depthCameraData, depthCameraParams);

		if (GlobalAppState::get().s_streamingEnabled == true) {
			MLIB_WARNING("s_streamingEnabled is no compatible with deintegration");
		}

		setLastRigidTransform(lastRigidTransform);

		//generate a linear hash array with only occupied entries
		compactifyHashEntries();

		//volumetrically integrate the depth data into the depth SDFBlocks
		deIntegrateDepthMap(depthCameraData, depthCameraParams);

		//garbageCollect();

		//DepthImage32 test(depthCameraParams.m_imageWidth, depthCameraParams.m_imageHeight);
		//cudaMemcpyFromArray(test.getData(), depthCameraData.d_depthArray, 0, 0, sizeof(float)*depthCameraParams.m_imageWidth *depthCameraParams.m_imageHeight, cudaMemcpyDeviceToHost);
		//FreeImageWrapper::saveImage("test_deint_depth" + std::to_string(m_numIntegratedFrames) + " .png", ColorImageR32G32B32(test), true);

		m_numIntegratedFrames--;
	}


	void garbageCollect() {
		//only perform if enabled by global app state
		if (GlobalAppState::get().s_garbageCollectionEnabled) {

			//if (m_numIntegratedFrames > 0 && m_numIntegratedFrames % GlobalAppState::get().s_garbageCollectionStarve == 0) {
			//	starveVoxelsKernelCUDA(m_hashData, m_hashParams);

			//	MLIB_WARNING("starving voxel weights is incompatible with bundling");
			//}

			if (m_hashParams.m_numOccupiedBlocks > 0) {
				garbageCollectIdentifyCUDA(m_hashData, m_hashParams);
				resetHashBucketMutexCUDA(m_hashData, m_hashParams);	//needed if linked lists are enabled -> for memeory deletion
				garbageCollectFreeCUDA(m_hashData, m_hashParams);
			}
		}
	}

	void setLastRigidTransform(const mat4f& lastRigidTransform) {
		m_hashParams.m_rigidTransform = MatrixConversion::toCUDA(lastRigidTransform);
		m_hashParams.m_rigidTransformInverse = m_hashParams.m_rigidTransform.getInverse();

		//make the rigid transform available on the GPU
		m_hashData.updateParams(m_hashParams);
	}

	void setLastRigidTransformAndCompactify(const mat4f& lastRigidTransform) {
		setLastRigidTransform(lastRigidTransform);
		compactifyHashEntries();
	}
	
	const mat4f getLastRigidTransform() const {
		return MatrixConversion::toMlib(m_hashParams.m_rigidTransform);
	}

	//! resets the hash to the initial state (i.e., clears all data)
	void reset() {
		m_numIntegratedFrames = 0;
		m_hashParams.m_rigidTransform.setIdentity();
		m_hashParams.m_rigidTransformInverse.setIdentity();
		m_hashParams.m_numOccupiedBlocks = 0;
		m_hashData.updateParams(m_hashParams);
		if (!getCPUEnable())
			resetCUDA(m_hashData, m_hashParams); //GPU
		else
			resetCPU();
	}
	
	HashDataStruct& getHashData() {
		return m_hashData;
	} 

	const HashParams& getHashParams() const {
		return m_hashParams;
	}
	
	//==panrj
	void integrateCPU(const mat4f& lastRigidTransform, const DepthCameraData& depthCameraData, const DepthCameraParams& depthCameraParams, unsigned int* d_bitMask) {
		m_hashParamsSRH.m_rigidTransform = MatrixConversion::toCUDA(lastRigidTransform);
		m_hashParamsSRH.m_rigidTransformInverse = m_hashParamsSRH.m_rigidTransform.getInverse();
		allocCPU(depthCameraData, depthCameraParams, d_bitMask);
		//HashDataStruct hdscpu0 = getHashDataCPU();
		compactifyHashEntriesCPU();
		//HashDataStruct hdscpu1 = getHashDataCPU();
		integrateDepthMapCPU(depthCameraData, depthCameraParams, false);
		//HashDataStruct hdscpu2 = getHashDataCPU();
	}

	void deIntegrateCPU(const mat4f& lastRigidTransform, const DepthCameraData& depthCameraData, const DepthCameraParams& depthCameraParams, unsigned int* d_bitMask) {
		compactifyHashEntriesCPU();
		//HashDataStruct hdscpu1 = getHashDataCPU();
		integrateDepthMapCPU(depthCameraData, depthCameraParams, true);
	}

	unsigned int getHeapFreeCountCPU() {
		unsigned int count;
		count = unsigned int(m_hashDataCPU.d_heapCounter);//jfwrongs 强制转换，另外d_heapCounter在这里没有修改值
		return count + 1;
	}

	bool isSDFBlockStreamedOutCPU(const int3& sdfBlock, const HashDataStruct& hashData, const unsigned int* d_bitMask)
	{
		if (!d_bitMask) return false;	//TODO can statically disable streaming??
	}

	HashDataStruct& getHashDataCPU() {
		return m_hashDataCPU;
	}

	void setCameraParams(DepthCameraParams dcp) {
		cameraParamsSRH = dcp;
	}

	void setCPUEnable(bool flag) {
		s_SDF_CPUEnable = flag;
	}

	bool getCPUEnable() {
		return s_SDF_CPUEnable;
	}

	void resetCPU()
	{
		for (int block = 0; block < (m_hashParams.m_numSDFBlocks + (T_PER_BLOCK*T_PER_BLOCK) - 1) / (T_PER_BLOCK*T_PER_BLOCK); block++) {
			for (int thread = 0; thread < T_PER_BLOCK*T_PER_BLOCK; thread++) {
				unsigned int idx = block * T_PER_BLOCK*T_PER_BLOCK + thread;

				if (idx == 0) {
					m_hashDataCPU.d_heapCounter[0] = m_hashParams.m_numSDFBlocks - 1;	//points to the last element of the array
				}

				if (idx < m_hashParams.m_numSDFBlocks) {

					m_hashDataCPU.d_heap[idx] = m_hashParams.m_numSDFBlocks - idx - 1;
					uint blockSize = SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;
					uint base_idx = idx * blockSize;
					for (uint i = 0; i < blockSize; i++) {
						m_hashDataCPU.deleteVoxelCPU(base_idx + i);
					}
				}
			}
		}

		for (int block = 0; block < (HASH_BUCKET_SIZE * m_hashParams.m_hashNumBuckets + (T_PER_BLOCK*T_PER_BLOCK) - 1) / (T_PER_BLOCK*T_PER_BLOCK); block++) {
			for (int thread = 0; thread < (T_PER_BLOCK*T_PER_BLOCK); thread++) {
				const unsigned int idx = block * T_PER_BLOCK*T_PER_BLOCK + thread;
				if (idx < m_hashParams.m_hashNumBuckets * HASH_BUCKET_SIZE) {
					m_hashDataCPU.deleteHashEntryCPU(m_hashDataCPU.d_hash[idx]);
					m_hashDataCPU.deleteHashEntryCPU(m_hashDataCPU.d_hashCompactified[idx]);
				}
			}
		}

		for (int block = 0; block < (m_hashParams.m_hashNumBuckets + (T_PER_BLOCK*T_PER_BLOCK) - 1) / (T_PER_BLOCK*T_PER_BLOCK); block++) {
			for (int thread = 0; thread < (T_PER_BLOCK*T_PER_BLOCK); thread++) {
				const unsigned int idx = block * (T_PER_BLOCK*T_PER_BLOCK) + thread;
				if (idx < m_hashParams.m_hashNumBuckets) {
					m_hashDataCPU.d_hashBucketMutex[idx] = FREE_ENTRY;
				}
			}
		}
		m_hashParamsSRH = m_hashParams;
		m_hashDataCPU.hashParams_hd = m_hashParamsSRH;
	}

	void allocCPU(const DepthCameraData& depthCameraData, const DepthCameraParams& depthCameraParams, const unsigned int* d_bitMask) {
		float* m_depth = new float[320 * 240];
		//cudaMalloc(&d_inputDepthRaw, sizeof(float)*m_inputDepthWidth*m_inputDepthHeight)
		cudaMemcpy(m_depth, depthCameraData.d_depthData, depthCameraParams.m_imageHeight*depthCameraParams.m_imageWidth * sizeof(float), cudaMemcpyDeviceToHost);
		//std::cout << m_hashDataCPU.d_hash[0].ptr << std::endl; 
		unsigned int prevFree = getHeapFreeCountCPU();
		while (1) {
			//第一个函数resetHashBucketMutexCUDA改写
			for (int block = 0; block < (m_hashParamsSRH.m_hashNumBuckets + (T_PER_BLOCK*T_PER_BLOCK) - 1) / (T_PER_BLOCK*T_PER_BLOCK); block++) {
				for (int thread = 0; thread < T_PER_BLOCK*T_PER_BLOCK; thread++) {
					const unsigned int idx = block * T_PER_BLOCK*T_PER_BLOCK + thread;
					if (idx < m_hashParamsSRH.m_hashNumBuckets) {
						m_hashDataCPU.d_hashBucketMutex[idx] = FREE_ENTRY;
					}//if
				}//thread
			}//block

			//第二个函数allocCUDA改写
			for (int block1 = 0; block1 < (depthCameraParams.m_imageWidth + T_PER_BLOCK - 1) / T_PER_BLOCK; block1++) {
				for (int block2 = 0; block2 < (depthCameraParams.m_imageHeight + T_PER_BLOCK - 1) / T_PER_BLOCK; block2++) {
					for (int thread1 = 0; thread1 < T_PER_BLOCK; thread1++) {
						for (int thread2 = 0; thread2 < T_PER_BLOCK; thread2++) {
							const unsigned int x = block1 * T_PER_BLOCK + thread1;
							const unsigned int y = block2 * T_PER_BLOCK + thread2;
							if (x < cameraParamsSRH.m_imageWidth && y < cameraParamsSRH.m_imageHeight)
							{

								//float d = tex2D(depthTextureRef, x, y);
								float d = m_depth[y*depthCameraParams.m_imageWidth + x];//jfwrongs
								//std::cout << y * depthCameraParams.m_imageWidth + x << "------" << d << std::endl;

								if (d == MINF || d == 0.0f)	continue;

								//std::cout << d << std::endl;
								if (d >= m_hashParamsSRH.m_maxIntegrationDistance) continue;

								float t = m_hashDataCPU.getTruncationCPU(d, m_hashParamsSRH);
								//std::cout << d-t << std::endl;

								float minDepth = d - t;
								//printf("%f\n", minDepth);
								float maxDepth = d + t;
								if (minDepth >= maxDepth) continue;

								float3 rayMin = DepthCameraData::kinectDepthToSkeletonCPU(x, y, minDepth, cameraParamsSRH);
								rayMin = m_hashParamsSRH.m_rigidTransform * rayMin;
								float3 rayMax = DepthCameraData::kinectDepthToSkeletonCPU(x, y, maxDepth, cameraParamsSRH);
								rayMax = m_hashParamsSRH.m_rigidTransform * rayMax;

								//printf("%f  %f  %f\n", rayMin.x, rayMin.y, rayMin.z);
								float3 rayDir = normalize(rayMax - rayMin);

								int3 idCurrentVoxel = m_hashDataCPU.worldToSDFBlockCPU(rayMin);
								int3 idEnd = m_hashDataCPU.worldToSDFBlockCPU(rayMax);

								float3 step = make_float3(sign(rayDir));
								float3 boundaryPos = m_hashDataCPU.SDFBlockToWorldCPU(idCurrentVoxel + make_int3(clamp(step, 0.0, 1.0f))) - 0.5f*m_hashParamsSRH.m_virtualVoxelSize;
								float3 tMax = (boundaryPos - rayMin) / rayDir;
								float3 tDelta = (step*SDF_BLOCK_SIZE*m_hashParamsSRH.m_virtualVoxelSize) / rayDir;
								int3 idBound = make_int3(make_float3(idEnd) + step);

								if (rayDir.x == 0.0f) { tMax.x = PINF; tDelta.x = PINF; }
								if (boundaryPos.x - rayMin.x == 0.0f) { tMax.x = PINF; tDelta.x = PINF; }

								if (rayDir.y == 0.0f) { tMax.y = PINF; tDelta.y = PINF; }
								if (boundaryPos.y - rayMin.y == 0.0f) { tMax.y = PINF; tDelta.y = PINF; }

								if (rayDir.z == 0.0f) { tMax.z = PINF; tDelta.z = PINF; }
								if (boundaryPos.z - rayMin.z == 0.0f) { tMax.z = PINF; tDelta.z = PINF; }


								unsigned int iter = 0; // iter < g_MaxLoopIterCount
								unsigned int g_MaxLoopIterCount = 1024;
#pragma unroll 1
								while (iter < g_MaxLoopIterCount) {

									//检查是否在视锥内
									//std::cout << idCurrentVoxel.x<<"    "<< idCurrentVoxel.y << "    "<<idCurrentVoxel.z <<  std::endl;
									if (m_hashDataCPU.isSDFBlockInCameraFrustumApproxCPU(idCurrentVoxel, m_hashParamsSRH, cameraParamsSRH) && !isSDFBlockStreamedOutCPU(idCurrentVoxel, m_hashDataCPU, d_bitMask)) {//jfwrongs 这个函数简化后的chunk里的mask必为空
										//std::cout << iter << std::endl;
										m_hashDataCPU.allocBlockCPU(idCurrentVoxel, m_hashParamsSRH);
									}
									/*		else {
												std::cout << iter << std::endl;
											}*/
											//std::cout << x << "    " << y << std::endl;
											// Traverse voxel grid
									if (tMax.x < tMax.y && tMax.x < tMax.z) {
										idCurrentVoxel.x += step.x;
										if (idCurrentVoxel.x == idBound.x) break;
										tMax.x += tDelta.x;
									}
									else if (tMax.z < tMax.y) {
										idCurrentVoxel.z += step.z;
										if (idCurrentVoxel.z == idBound.z) break;
										tMax.z += tDelta.z;
									}
									else {
										idCurrentVoxel.y += step.y;
										if (idCurrentVoxel.y == idBound.y) break;
										tMax.y += tDelta.y;
									}

									iter++;
								}
								//std::cout << iter << std::endl;
							}
						}
					}
				}
			}
			unsigned int curFree = getHeapFreeCountCPU();
			if (prevFree != curFree) {
				prevFree = curFree;
			}
			else {//使用了内存就记录新内存开销否则跳出循环
				break;
			}
		}
		delete[] m_depth;
	}

	void compactifyHashEntriesCPU() {
		//cudaMemcpyFromSymbol(&m_hashParamsSRH, &c_hashParams, sizeof(HashParams), 0, cudaMemcpyDeviceToHost);
		//cudaMemcpyFromSymbol(&cameraParamsSRH, &c_depthCameraParams, sizeof(DepthCameraParams), 0, cudaMemcpyDeviceToHost);
		memset(m_hashDataCPU.d_hashCompactifiedCounter, 0, sizeof(int));
		//改写第一个函数compactifyHashAllInOneCUDA
		const unsigned int threadsPerBlock = COMPACTIFY_HASH_THREADS_PER_BLOCK;
		for (int block = 0; block < (HASH_BUCKET_SIZE * m_hashParamsSRH.m_hashNumBuckets + threadsPerBlock - 1) / threadsPerBlock; block++) {
			for (int thread = 0; thread < threadsPerBlock; thread++) {
				const unsigned int idx = block * threadsPerBlock + thread;
				if (idx < m_hashParamsSRH.m_hashNumBuckets * HASH_BUCKET_SIZE) {
					if (m_hashDataCPU.d_hash[idx].ptr != FREE_ENTRY) {
						//std::cout << "-----------" << std::endl;
						if (m_hashDataCPU.isSDFBlockInCameraFrustumApproxCPU(m_hashDataCPU.d_hash[idx].pos, m_hashParamsSRH, cameraParamsSRH))
						{
							//int addr = atomicAdd(hashData.d_hashCompactifiedCounter, 1);
							int addr = *(m_hashDataCPU.d_hashCompactifiedCounter);
							(*m_hashDataCPU.d_hashCompactifiedCounter)++;
							m_hashDataCPU.d_hashCompactified[addr] = m_hashDataCPU.d_hash[idx];
						}//if
					}//if
				}//if
			}//thread
		}//block
		m_hashParamsSRH.m_numOccupiedBlocks = *m_hashDataCPU.d_hashCompactifiedCounter;
		//std::cout <<"*****"<<m_hashParamsSRH.m_numOccupiedBlocks << std::endl;
		/*在这里对于每一帧都要更新CUDA缓冲区中的常量hash字段，但现在使用的是统一的hash字段，在下一帧进来的时候势必复制的是GPU版本的字段
		两种解决方法，CPU另设字段，要么就是手动将上一帧计算出的m_numOccupiedBlocks放进去,这里先采用后者，节约空间，也较为方便看看效果对不对 jfwrongs*/
		//m_hashParamsSRH = m_hashParams;
	}

	void integrateDepthMapCPU(const DepthCameraData& depthCameraData, const DepthCameraParams& depthCameraParams, bool deIntegrate) {
		//改写第一个函数integrateDepthMapCUDA
		float* m_depth = new float[320 * 240];
		uchar4* m_color = new uchar4[320 * 240];
		cudaMemcpy(m_depth, depthCameraData.d_depthData, depthCameraParams.m_imageHeight*depthCameraParams.m_imageWidth * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(m_color, depthCameraData.d_colorData, depthCameraParams.m_imageHeight*depthCameraParams.m_imageWidth * sizeof(uchar4), cudaMemcpyDeviceToHost);
		const unsigned int threadsPerBlock = SDF_BLOCK_SIZE * SDF_BLOCK_SIZE*SDF_BLOCK_SIZE;
		for (int block = 0; block < m_hashParamsSRH.m_numOccupiedBlocks; block++) {
			for (int thread = 0; thread < threadsPerBlock; thread++) {
				const HashEntry& entry = m_hashDataCPU.d_hashCompactified[block];

				int3 pi_base = m_hashDataCPU.SDFBlockToVirtualVoxelPosCPU(entry.pos);

				uint i = thread;	//inside of an SDF block
				int3 pi = pi_base + make_int3(m_hashDataCPU.delinearizeVoxelIndexCPU(i));
				float3 pf = m_hashDataCPU.virtualVoxelPosToWorldCPU(pi);

				pf = m_hashParamsSRH.m_rigidTransformInverse * pf;
				uint2 screenPos = make_uint2(depthCameraData.cameraToKinectScreenIntCPU(pf, cameraParamsSRH));
				//std::cout << "-----------" << std::endl;

				if (screenPos.x < cameraParamsSRH.m_imageWidth && screenPos.y < cameraParamsSRH.m_imageHeight) {	//on screen
					//float depth = g_InputDepth[screenPos];
					//float depth = tex2D(depthTextureRef, screenPos.x, screenPos.y);
					float depth = m_depth[screenPos.y*depthCameraParams.m_imageWidth + screenPos.x];//jfwrongs
					float4 color = make_float4(MINF, MINF, MINF, MINF);
					if (depthCameraData.d_colorData) {
						//uchar4 color_uc = tex2D(colorTextureRef, screenPos.x, screenPos.y);
						uchar4 color_uc = m_color[screenPos.y*depthCameraParams.m_imageWidth + screenPos.x];//jfwrongs
						color = make_float4(color_uc.x, color_uc.y, color_uc.z, color_uc.w);
						//color = bilinearFilterColor(cameraData.cameraToKinectScreenFloat(pf));
					}

					if (color.x != MINF && depth != MINF) { // valid depth and color
					//if (depth != MINF) {	//valid depth
						if (depth < m_hashParamsSRH.m_maxIntegrationDistance) {
							float depthZeroOne = depthCameraData.cameraToKinectProjZCPU(depth, cameraParamsSRH);

							float sdf = depth - pf.z;
							float truncation = m_hashDataCPU.getTruncationCPU(depth, m_hashParamsSRH);
							//if (sdf > -truncation) 
							if (abs(sdf) < truncation)
							{
								if (sdf >= 0.0f) {
									sdf = fminf(truncation, sdf);
								}
								else {
									sdf = fmaxf(-truncation, sdf);
								}

								float weightUpdate = max(m_hashParamsSRH.m_integrationWeightSample * 1.5f * (1.0f - depthZeroOne), 1.0f);
								weightUpdate = 1.0f;	//TODO remove that again

								Voxel curr;	//construct current voxel
								curr.sdf = sdf;
								curr.weight = weightUpdate;

								if (depthCameraData.d_colorData) {
									curr.color = make_uchar4(color.x, color.y, color.z, 255);
								}
								else {
									curr.color = make_uchar4(0, 255, 0, 0);
								}

								uint idx = entry.ptr + i;

								const Voxel& oldVoxel = m_hashDataCPU.d_SDFBlocks[idx];
								Voxel newVoxel;


								float3 currColor = make_float3(curr.color.x, curr.color.y, curr.color.z);
								float3 oldColor = make_float3(oldVoxel.color.x, oldVoxel.color.y, oldVoxel.color.z);

								if (!deIntegrate) {	//integration
									//hashData.combineVoxel(hashData.d_SDFBlocks[idx], curr, newVoxel);
									float3 res;
									if (oldVoxel.weight == 0) res = currColor;
									//else res = (currColor + oldColor) / 2;
									else res = 0.2f * currColor + 0.8f * oldColor;
									//float3 res = (currColor*curr.weight + oldColor*oldVoxel.weight) / (curr.weight + oldVoxel.weight);
									res = make_float3(round(res.x), round(res.y), round(res.z));
									res = fmaxf(make_float3(0.0f), fminf(res, make_float3(254.5f)));
									//newVoxel.color.x = (uchar)(res.x + 0.5f);	newVoxel.color.y = (uchar)(res.y + 0.5f);	newVoxel.color.z = (uchar)(res.z + 0.5f);
									newVoxel.color = make_uchar4(res.x, res.y, res.z, 255);
									newVoxel.sdf = (curr.sdf*curr.weight + oldVoxel.sdf*oldVoxel.weight) / (curr.weight + oldVoxel.weight);
									newVoxel.weight = min((float)m_hashParamsSRH.m_integrationWeightMax, curr.weight + oldVoxel.weight);
								}
								else {				//deintegration
									//float3 res = 2 * c0 - c1;
									float3 res = (oldColor*oldVoxel.weight - currColor * curr.weight) / (oldVoxel.weight - curr.weight);
									res = make_float3(round(res.x), round(res.y), round(res.z));
									res = fmaxf(make_float3(0.0f), fminf(res, make_float3(254.5f)));
									//newVoxel.color.x = (uchar)(res.x + 0.5f);	newVoxel.color.y = (uchar)(res.y + 0.5f);	newVoxel.color.z = (uchar)(res.z + 0.5f);
									newVoxel.color = make_uchar4(res.x, res.y, res.z, 255);
									newVoxel.sdf = (oldVoxel.sdf*oldVoxel.weight - curr.sdf*curr.weight) / (oldVoxel.weight - curr.weight);
									newVoxel.weight = max(0.0f, oldVoxel.weight - curr.weight);
									if (newVoxel.weight <= 0.001f) {
										newVoxel.sdf = 0.0f;
										newVoxel.color = make_uchar4(0, 0, 0, 0);
										newVoxel.weight = 0.0f;
									}
								}

								m_hashDataCPU.d_SDFBlocks[idx] = newVoxel;
							}
						}
					}
				}
			}//thread
		}//block
	
		delete[] m_depth;
		delete[] m_color;
	}
	//==panrj

	//! debug only!
	unsigned int getHeapFreeCount() {
		unsigned int count;
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(&count, m_hashData.d_heapCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost));
		return count+1;	//there is one more free than the address suggests (0 would be also a valid address)
	}

	unsigned int getNumIntegratedFrames() const {
		return m_numIntegratedFrames;
	}

	//! debug only!
	void debugHash() {
		HashEntry* hashCPU = new HashEntry[m_hashParams.m_hashBucketSize*m_hashParams.m_hashNumBuckets];
		unsigned int* heapCPU = new unsigned int[m_hashParams.m_numSDFBlocks];
		unsigned int heapCounterCPU;

		MLIB_CUDA_SAFE_CALL(cudaMemcpy(&heapCounterCPU, m_hashData.d_heapCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost));
		heapCounterCPU++;	//points to the first free entry: number of blocks is one more

		MLIB_CUDA_SAFE_CALL(cudaMemcpy(heapCPU, m_hashData.d_heap, sizeof(unsigned int)*m_hashParams.m_numSDFBlocks, cudaMemcpyDeviceToHost));
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(hashCPU, m_hashData.d_hash, sizeof(HashEntry)*m_hashParams.m_hashBucketSize*m_hashParams.m_hashNumBuckets, cudaMemcpyDeviceToHost));

		Voxel* sdfBlocksCPU = new Voxel[m_hashParams.m_numSDFBlocks*SDF_BLOCK_SIZE*SDF_BLOCK_SIZE*SDF_BLOCK_SIZE];
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(sdfBlocksCPU, m_hashData.d_SDFBlocks, sizeof(Voxel)*SDF_BLOCK_SIZE*SDF_BLOCK_SIZE*SDF_BLOCK_SIZE*m_hashParams.m_numSDFBlocks, cudaMemcpyDeviceToHost));


		//Check for duplicates
		class myint3Voxel {
		public:
			myint3Voxel() {}
			~myint3Voxel() {}
			bool operator<(const myint3Voxel& other) const {
				if (x == other.x) {
					if (y == other.y) {
						return z < other.z;
					}
					return y < other.y;
				}
				return x < other.x;
			}

			bool operator==(const myint3Voxel& other) const {
				return x == other.x && y == other.y && z == other.z;
			}

			int x,y,z, i;
			int offset;
			int ptr;
		}; 


		std::unordered_set<unsigned int> pointersFreeHash;
		std::vector<unsigned int> pointersFreeVec(m_hashParams.m_numSDFBlocks, 0);
		for (unsigned int i = 0; i < heapCounterCPU; i++) {
			pointersFreeHash.insert(heapCPU[i]);
			pointersFreeVec[heapCPU[i]] = FREE_ENTRY;
		}
		if (pointersFreeHash.size() != heapCounterCPU) {
			throw MLIB_EXCEPTION("ERROR: duplicate free pointers in heap array");
		}
		 

		unsigned int numOccupied = 0;
		unsigned int numMinusOne = 0;
		unsigned int listOverallFound = 0;

		PointCloudf voxelBlocksPC;
		PointCloudf voxelPC;
		std::list<myint3Voxel> l;
		BoundingBox3<int> bboxBlocks;
		//std::vector<myint3Voxel> v;
		
		for (unsigned int i = 0; i < m_hashParams.m_hashBucketSize*m_hashParams.m_hashNumBuckets; i++) {
			if (hashCPU[i].ptr == -1) {
				numMinusOne++;
			}

			if (hashCPU[i].ptr != -2) {
				numOccupied++;	// != FREE_ENTRY
				myint3Voxel a;	
				a.x = hashCPU[i].pos.x;
				a.y = hashCPU[i].pos.y;
				a.z = hashCPU[i].pos.z;
				l.push_back(a);
				//v.push_back(a);

				unsigned int linearBlockSize = m_hashParams.m_SDFBlockSize*m_hashParams.m_SDFBlockSize*m_hashParams.m_SDFBlockSize;
				if (pointersFreeHash.find(hashCPU[i].ptr / linearBlockSize) != pointersFreeHash.end()) {
					throw MLIB_EXCEPTION("ERROR: ptr is on free heap, but also marked as an allocated entry");
				}
				pointersFreeVec[hashCPU[i].ptr / linearBlockSize] = LOCK_ENTRY;

				voxelBlocksPC.m_points.push_back(vec3f((float)a.x, (float)a.y, (float)a.z));
				bboxBlocks.include(vec3i(a.x, a.y, a.z));

				for (unsigned int z = 0; z < SDF_BLOCK_SIZE; z++) {
					for (unsigned int y = 0; y < SDF_BLOCK_SIZE; y++) {
						for (unsigned int x = 0; x < SDF_BLOCK_SIZE; x++) {
							unsigned int linearOffset = z*SDF_BLOCK_SIZE*SDF_BLOCK_SIZE + y*SDF_BLOCK_SIZE + x;
							const Voxel& v = sdfBlocksCPU[hashCPU[i].ptr + linearOffset];
							if (v.weight > 0 && std::abs(v.sdf) <= m_hashParams.m_virtualVoxelSize) {
								vec3f pos = vec3f(vec3i(hashCPU[i].pos.x, hashCPU[i].pos.y, hashCPU[i].pos.z) * SDF_BLOCK_SIZE + vec3i(x, y, z));
								pos = pos * m_hashParams.m_virtualVoxelSize;
								voxelPC.m_points.push_back(pos);

								std::cout << "voxel weight " << v.weight << std::endl;
								std::cout << "voxel sdf " << v.sdf << std::endl;
							}
						}
					}
				}
			}
		}

		std::cout << "valid blocks found " << voxelBlocksPC.m_points.size() << std::endl;
		std::cout << "valid voxel found " << voxelPC.m_points.size() << std::endl;

		unsigned int numHeapFree = 0;
		unsigned int numHeapOccupied = 0;
		for (unsigned int i = 0; i < m_hashParams.m_numSDFBlocks; i++) {
			if		(pointersFreeVec[i] == FREE_ENTRY) numHeapFree++;
			else if (pointersFreeVec[i] == LOCK_ENTRY) numHeapOccupied++;
			else {
				throw MLIB_EXCEPTION("memory leak detected: neither free nor allocated");
			}
		}
		if (numHeapFree + numHeapOccupied == m_hashParams.m_numSDFBlocks) std::cout << "HEAP OK!" << std::endl;
		else throw MLIB_EXCEPTION("HEAP CORRUPTED");

		l.sort();
		size_t sizeBefore = l.size();
		l.unique();
		size_t sizeAfter = l.size();


		std::cout << "diff: " << sizeBefore - sizeAfter << std::endl;
		std::cout << "minOne: " << numMinusOne << std::endl;
		std::cout << "numOccupied: " << numOccupied << "\t numFree: " << getHeapFreeCount() << std::endl;
		std::cout << "numOccupied + free: " << numOccupied + getHeapFreeCount() << std::endl;
		std::cout << "numInFrustum: " << m_hashParams.m_numOccupiedBlocks << std::endl;

		SAFE_DELETE_ARRAY(heapCPU);
		SAFE_DELETE_ARRAY(hashCPU);

		SAFE_DELETE_ARRAY(sdfBlocksCPU);
		//getchar();
	}

private:

	void create(const HashParams& params) {
		m_hashParams = params;
		
		if (!getCPUEnable())
			m_hashData.allocate(m_hashParams);
			//m_hashData.allocate(m_hashParams,false); //有错误
		else
			m_hashDataCPU.allocate(m_hashParams, false);//jfwrongs 仍使用原有字段会不会有问题
		reset();
	}

	//void destroy() {
	//	m_hashData.free();
	//}

	void alloc(const DepthCameraData& depthCameraData, const DepthCameraParams& depthCameraParams, const unsigned int* d_bitMask) {
		//Start Timing
		if(GlobalAppState::get().s_timingsDetailledEnabled) { cutilSafeCall(cudaDeviceSynchronize()); m_timer.start(); }

		//resetHashBucketMutexCUDA(m_hashData, m_hashParams);
		//allocCUDA(m_hashData, m_hashParams, depthCameraData, depthCameraParams, d_bitMask);
		 
		unsigned int prevFree = getHeapFreeCount();
		while (1) {
			resetHashBucketMutexCUDA(m_hashData, m_hashParams);
			allocCUDA(m_hashData, m_hashParams, depthCameraData, depthCameraParams, d_bitMask);

			unsigned int currFree = getHeapFreeCount();

			if (prevFree != currFree) {
				prevFree = currFree;
			}
			else {
				break;
			}
		}

		// Stop Timing
		if (GlobalAppState::get().s_timingsDetailledEnabled) { cutilSafeCall(cudaDeviceSynchronize()); m_timer.stop(); TimingLogDepthSensing::totalTimeAlloc += m_timer.getElapsedTimeMS(); TimingLogDepthSensing::countTimeAlloc++; }
	}


	void compactifyHashEntries() {
		//Start Timing
		if (GlobalAppState::get().s_timingsDetailledEnabled) { cutilSafeCall(cudaDeviceSynchronize()); m_timer.start(); }

		//CUDATimer t;
		
		////t.startEvent("fillDecisionArray");
		//fillDecisionArrayCUDA(m_hashData, m_hashParams);
		////t.endEvent();

		////t.startEvent("prefixSum");
		//m_hashParams.m_numOccupiedBlocks = 
		//	m_cudaScan.prefixSum(
		//		m_hashParams.m_hashNumBuckets*m_hashParams.m_hashBucketSize,
		//		m_hashData.d_hashDecision,
		//		m_hashData.d_hashDecisionPrefix);
		////t.endEvent();

		////t.startEvent("compactifyHash");
		//m_hashData.updateParams(m_hashParams);	//make sure numOccupiedBlocks is updated on the GPU
		//compactifyHashCUDA(m_hashData, m_hashParams);
		////t.endEvent();

		//t.startEvent("compactifyAllInOne");
		m_hashParams.m_numOccupiedBlocks = compactifyHashAllInOneCUDA(m_hashData, m_hashParams);
		m_hashData.updateParams(m_hashParams);	//make sure numOccupiedBlocks is updated on the GPU
		//t.endEvent();
		//t.evaluate();

		// Stop Timing
		if (GlobalAppState::get().s_timingsDetailledEnabled) { cutilSafeCall(cudaDeviceSynchronize()); m_timer.stop(); TimingLogDepthSensing::totalTimeCompactifyHash += m_timer.getElapsedTimeMS(); TimingLogDepthSensing::countTimeCompactifyHash++; }

		//std::cout << "numOccupiedBlocks: " << m_hashParams.m_numOccupiedBlocks << std::endl;
	}

	void integrateDepthMap(const DepthCameraData& depthCameraData, const DepthCameraParams& depthCameraParams) {
		//Start Timing
		if(GlobalAppState::get().s_timingsDetailledEnabled) { cutilSafeCall(cudaDeviceSynchronize()); m_timer.start(); }

		integrateDepthMapCUDA(m_hashData, m_hashParams, depthCameraData, depthCameraParams);

		// Stop Timing
		if (GlobalAppState::get().s_timingsDetailledEnabled) { cutilSafeCall(cudaDeviceSynchronize()); m_timer.stop(); 
		TimingLogDepthSensing::totalTimeIntegrate += m_timer.getElapsedTimeMS(); TimingLogDepthSensing::countTimeIntegrate++; }
	}

	void deIntegrateDepthMap(const DepthCameraData& depthCameraData, const DepthCameraParams& depthCameraParams) {
		//Start Timing
		if (GlobalAppState::get().s_timingsDetailledEnabled) { cutilSafeCall(cudaDeviceSynchronize()); m_timer.start(); }

		deIntegrateDepthMapCUDA(m_hashData, m_hashParams, depthCameraData, depthCameraParams);

		// Stop Timing
		if (GlobalAppState::get().s_timingsDetailledEnabled) { cutilSafeCall(cudaDeviceSynchronize()); m_timer.stop(); TimingLogDepthSensing::totalTimeDeIntegrate += m_timer.getElapsedTimeMS(); TimingLogDepthSensing::countTimeDeIntegrate++; }
	}

	HashParams		m_hashParams;
	HashDataStruct	m_hashData;

	CUDAScan		m_cudaScan;

	unsigned int	m_numIntegratedFrames;	//used for garbage collect

	static Timer m_timer;

	//==panrj
	bool s_SDF_CPUEnable;
	HashDataStruct m_hashDataCPU;
	HashParams		m_hashParamsSRH;
	DepthCameraParams cameraParamsSRH;
	//==panrj
};
