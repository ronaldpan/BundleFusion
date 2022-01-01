#pragma once

#include "GlobalAppState.h"
#include "VoxelUtilHashSDF.h"
#include "MarchingCubesSDFUtil.h"
#include "CUDASceneRepChunkGrid.h"

class CUDAMarchingCubesHashSDF
{
public:
	//CUDAMarchingCubesHashSDF(const MarchingCubesParams& params) {	//create(params);}
	CUDAMarchingCubesHashSDF(bool bGpu) {
		if (bGpu)
			create(parametersFromGlobalAppState(GlobalAppState::get()));
		else
			create(parametersFromGlobalAppStateCpu(GlobalAppState::get()));
	}

	~CUDAMarchingCubesHashSDF(void) {
		destroy();
	}

	static MarchingCubesParams parametersFromGlobalAppState(const GlobalAppState& gas) {
		MarchingCubesParams params;
		params.m_maxNumTriangles = gas.s_marchingCubesMaxNumTriangles;
		params.m_threshMarchingCubes = gas.s_SDFMarchingCubeThreshFactor*gas.s_SDFVoxelSize;
		params.m_threshMarchingCubes2 = gas.s_SDFMarchingCubeThreshFactor*gas.s_SDFVoxelSize;
		params.m_sdfBlockSize = SDF_BLOCK_SIZE;
		params.m_hashBucketSize = HASH_BUCKET_SIZE;
		params.m_hashNumBuckets = gas.s_hashNumBuckets;
		return params;
	}
	
	static MarchingCubesParams parametersFromGlobalAppStateCpu(const GlobalAppState& gas) {
		MarchingCubesParams params;
		params.m_maxNumTriangles = gas.s_marchingCubesMaxNumTriangles;
		params.m_threshMarchingCubes = gas.s_SDFMarchingCubeThreshFactorC*gas.s_SDFVoxelSizeC;
		params.m_threshMarchingCubes2 = gas.s_SDFMarchingCubeThreshFactorC*gas.s_SDFVoxelSizeC;
		params.m_sdfBlockSize = SDF_BLOCK_SIZE;
		params.m_hashBucketSize = HASH_BUCKET_SIZE;
		params.m_hashNumBuckets = gas.s_hashNumBucketsC;
		return params;
	}

	void clearMeshBuffer(void) {
		m_meshData.clear();
	}

	//! copies the intermediate result of extract isoSurfaceCUDA to the CPU and merges it with meshData
	void copyTrianglesToCPU();
	void saveMesh(const std::string& filename, const mat4f *transform = NULL, bool overwriteExistingFile = false);

	void extractIsoSurface(const HashDataStruct& hashData, /*const HashParams& hashParams,*/ const RayCastData& rayCastData, 
		const vec3f& minCorner = vec3f(0.0f, 0.0f, 0.0f), const vec3f& maxCorner = vec3f(0.0f, 0.0f, 0.0f), bool boxEnabled = false);
	//void extractIsoSurfaceCPU(const HashData& hashData, const HashParams& hashParams, const RayCastData& rayCastData);
	void extractIsoSurface(CUDASceneRepChunkGrid& chunkGrid, const RayCastData& rayCastData, const vec3f& camPos, float radius);
	//==panrj
	void extractIsoSurfaceCPU(const HashDataStruct& hashData, const HashParams& hashParams, RayCastData& rayCastData);
	//==panrj

private:
	void create(const MarchingCubesParams& params);
	void destroy(void)
	{
		m_data.free();
	}

	MarchingCubesParams m_params;
	MarchingCubesData	m_data;
	MeshDataf m_meshData; //mLib÷–

	Timer m_timer;
};