/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\
*  Copyright (c) 2019 <GTEP> - All Rights Reserved                            *
*  This file is part of HERMES Project.                                       *
*  Unauthorized copying of this file, via any medium is strictly prohibited.  *
*  Proprietary and confidential.                                              *
*                                                                             *
*  Developers:                                                                *
*     - Bismarck G. Souza Jr <bismarck@puc-rio.br>                            *
*     - Nelson Inoue <inoue@puc-rio.br>                                       *
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#include "GPUInfo.h"

int GPUInfo::nGPUs;
vector<cudaDeviceProp> GPUInfo::devices;


GPUInfo::GPUInfo()
{
	ReadGpuSpecification();
	//ReportGpuSpecification();
	//ReportGpuMemory();
}

void GPUInfo::ReadGpuSpecification()
{
	cudaGetDeviceCount(&nGPUs);
	devices.resize(nGPUs);

	for(int gpu=0; gpu < nGPUs; ++gpu) {
		cudaSetDevice(gpu);
		cudaGetDeviceProperties(&GPUInfo::devices[gpu], gpu);
	}
}

void GPUInfo::ReportGpuSpecification(ostream* out)
{
	int driverVersion, runtimeVersion;
	cudaDeviceProp deviceProp;
	string line = " ==================================================================";

	for (int dev=0; dev < nGPUs; ++dev) {
		cudaDriverGetVersion(&driverVersion);
		cudaRuntimeGetVersion(&runtimeVersion);
		cudaGetDeviceProperties(&deviceProp, dev);

		(*out) << endl << line << endl << endl
		     << " Device " << dev << ": " << deviceProp.name << endl << endl
		     << "   CUDA Driver Version / Runtime Version:         " << (driverVersion / 1000)
			                                                 << "." << (driverVersion % 100) / 10
													       << " / " << (runtimeVersion / 1000)
														     << "." << ((runtimeVersion % 100) / 10) << endl
		     << "   CUDA Capability Major/Minor version number:    " << deviceProp.major
															 << "." << deviceProp.minor <<endl
		     << "   Total amount of constant memory:               " << deviceProp.totalConstMem << " bytes\n"
			 << "   Total amount of shared memory per block:       " << deviceProp.sharedMemPerBlock << " bytes\n"
			 << "   Total number of registers available per block: " << deviceProp.regsPerBlock << endl
			 << "   Warp size:                                     " << deviceProp.warpSize << endl
			 << "   Maximum number of threads per multiprocessor:  " << deviceProp.maxThreadsPerMultiProcessor << endl
			 << "   Maximum number of threads per block:           " << deviceProp.maxThreadsPerBlock << endl
			 << "   Max dimension size of a thread block (x,y,z): (" << deviceProp.maxThreadsDim[0]
															<< ", " << deviceProp.maxThreadsDim[1]
															<< ", " << deviceProp.maxThreadsDim[2] << ")\n"
			 << "   Max dimension size of a grid size    (x,y,z): (" << deviceProp.maxGridSize[0]
															<< ", " << deviceProp.maxGridSize[1]
															<< ", " << deviceProp.maxGridSize[2] << ")\n"
			 << "   Concurrent copy and kernel execution:          " << (deviceProp.deviceOverlap ? "Yes" : "No")
														 << " with " << deviceProp.asyncEngineCount << " copy engine(s)\n"
			 << "   Run time limit on kernels:                     " << (deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No") << endl
			 << "   Integrated GPU sharing Host Memory:            " << (deviceProp.integrated ? "Yes" : "No") << endl
			 << "   Device supports Unified Addressing (UVA):      " << (deviceProp.unifiedAddressing ? "Yes" : "No") << endl
			 << "   Device supports Compute Preemption:            " << (deviceProp.computePreemptionSupported ? "Yes" : "No") << endl
			 << "   Supports Cooperative Kernel Launch:            " << (deviceProp.cooperativeLaunch ? "Yes" : "No") << endl
			 << "   Supports MultiDevice Co-op Kernel Launch:      " << (deviceProp.cooperativeMultiDeviceLaunch ? "Yes" : "No") << endl;
	}

	 (*out) << endl << line << endl << endl;
}

void GPUInfo::ReportGpuMemory(ostream* out)
{
	size_t free_byte, total_byte;
	double free_Gb, total_Gb, sum_free=0.0, sum_total=0.0;
	double b_2_Gb = 1.0/(1 << 30);
	vector<double> freeMem, totalMem;
	char line[100];

	(*out) << endl
	       << "         GPU global memory report [Gb]\n"
	       << "         =========================================\n\n"
	       << "                   Free   Used  Total\n";

	
	for(int dev=0; dev < nGPUs; ++dev) {
		cudaSetDevice(dev);
		cudaMemGetInfo( &free_byte, &total_byte );
		free_Gb = free_byte*b_2_Gb;
		total_Gb = total_byte*b_2_Gb;
		sprintf(line, "          GPU %d: %6.2f %6.2f %6.2f\n",
			dev,  free_Gb, total_Gb-free_Gb, total_Gb);
		(*out) << line;
		
		sum_free += free_Gb;
		sum_total += total_Gb;
	}

	sprintf(line, "          Total: %6.2f %6.2f %6.2f\n", 
		sum_free, (sum_total-sum_free), sum_total);

	(*out) << "                  -----  -----  -----\n" << line << endl <<endl; 
}

