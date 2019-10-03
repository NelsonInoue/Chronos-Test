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
#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cuda_runtime.h>

using namespace std;


class GPUInfo
{

public:
	static int nGPUs;
	static vector<cudaDeviceProp> devices;
	
	GPUInfo();
	static void ReadGpuSpecification();
	static void ReportGpuSpecification(ostream* out=&cout);
	static void ReportGpuMemory(ostream* out=&cout);
};

