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
#include <fstream>
#include <iostream>
#include <string>
#include <strstream>
#include <cmath>
#include <iomanip>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>     // abs()
#include <vector>
#include "FemOneGpu.h"
#include <omp.h>
#include <cuda_runtime.h>
#include "defs.h"
#include <vector>

using namespace Chronos;

// Utilities and system includes

#define streq(s1,s2)	((s1[0]==s2[0]) && strcmp(s1,s2)==0)

//----------------------------------------------------
// External Functions for Multi GPUs Implementation						 
//----------------------------------------------------

extern "C" void EvaluateMmatrix(int Id, int BlockSizeX, int _iNumMeshNodes, int _iNumDofNode, int _inumDiaPart, double *K, double *M);

extern "C" void AssemblyStiffnessMatrixColor(dim3 blocksPerGrid, dim3 threadsPerBlock, int Id, int _iNumMeshNodes, int _iNumMeshElem, int _iNumDofNode, int _iNumElasMat, int numelcolor, int numelcolorprv,
			int *connect, double *coord, double *prop, double *K, int *offsets, int offsets_size); 

extern "C" void EvaluateStrainStateColor(dim3 blocksPerGrid, dim3 threadsPerBlock, int Id, int _iNumMeshNodes, int _iNumMeshElem, int _iNumDofNode, int _iNumElasMat, int numelcolor, int numelcolorprv,
			int *connect, double *coord, double *X, double *strain);

extern "C" void EvaluateStressState(int Id, int BlockSizeX, int _iNumMeshElem, int _iNumElasMat, int *connect, int *LinkMeshColor, double *prop, double *strain, double *stress); 
                                         
extern "C" void ImpositionBoundaryCondition(int Id, int BlockSizeX, int _iNumMeshNodes, int _iNumSuppNodes, int _inumDiaPart, int *supp, double *K, double *B);

extern "C" void ImpositionBoundaryConditionNeumann(int Id, int BlockSizeX, int _iNumMeshNodes, int _iNumSuppNodes, int *supp, double *B);

extern "C" void SolveLinearSystem(int Id, int _inumDiaPart, int _iNumDofNode, int BlockSizeX, int BlockMultiProcPerGpu, int _iNumMeshNodes, double *B, double *M, double *K, int *off, double *t, double *CGTime, int *CGIter, double *CGError, double *X, double *R, double *D, double *Q, double *S,
		double *delta_1_aux, double *delta_new, double *dTq, double *delta_old, double *delta_new_h);

extern "C" void EvaluateNodalForceColor(dim3 blocksPerGrid, dim3 threadsPerBlock, int Id, int _iNumMeshNodes, int _iNumMeshElem, int _iNumDofNode, int _iNumElasMat, int numelcolornodalforce, int numelcolornodalforceprv,
			int *connect, double *coord, int *LinkMeshMeshColor, int *LinkMeshCellColor, double *dP, double *B);

//=============================================================================

void cFemOneGPU::ReadCHR(string filename)
{
	ReadFile_CHR* chrData_tmp = new ReadFile_CHR();
	chrData_tmp->read_file(filename);

	SetChrData(chrData_tmp);
}

void cFemOneGPU::SetChrData(ReadFile_CHR *chrData_)
{
	
	chrData = chrData_;
	std::vector<int> sizes = chrData->get_extension_sizes();
	nx = chrData->get_reservoir_size(X);
	ny = chrData->get_reservoir_size(Y);
	nz = chrData->get_reservoir_size(Z);
	nsi1 = sizes[LEFT];
	nsi2 = sizes[RIGHT];
	nsj1 = sizes[FRONT];
	nsj2 = sizes[BACK];
	nov = sizes[UP];
	nun = sizes[DOWN];
	nNodes = chrData->get_nNodes();
	nOffsets = chrData->get_nOffsets();
	nDofNode = chrData->get_nDofNode();
	nElements = chrData->get_nElements();
	nSupports = chrData->get_nSupports();
	nMaterials = chrData->get_nMaterials();

	Id = chrData->get_gpu_id(0);
}

//========================================================================================================
void cFemOneGPU::AnalyzeFemOneGPU(int ii, int jj, int GridCoord, double *dP_h, double *DeltaStrainChr_h, double *DeltaStressChr_h)
{
	double time;

	// ========= Read Data =========

	if(ii == 0 && jj ==0) {

		printf("         Read input file             \n");
		printf("         ========================================= \n");

		// ========= Prepare input data =========

		PrepareInputData();

		//SplitNodesAndElements();  // Split the number of nodes for Reservoir models

		// ========= Allocate memory =========

		printf("\n\n         Allocate and Copy Memory              \n");
		printf("         ========================================= \n");
		AllocateAndCopyVectors();      

		// ========================================================================================================================================================
		// ========= Functions for Partial Coupling  =========

		AllocateAndCopyVectorsPartialCoupling(); 

		LinkMeshGridMapping(GridCoord);  // Calculate nodal force from pore pressure

		EvalColoringMeshBrick8Struct();  // Apply coloring algorithm in reservoir elements

		EvaluateNodalForce(dP_h);  // Evaluate nodal force from pore pressure variation

		// ========================================================================================================================================================
		// ========= Assembly the stiffness matrix on one GPU =========

		printf("\n\n         Assembly Stiffness Matrix         \n");
		printf("         ========================================= \n");

		AssemblyStiffnessMatrix();  

		//PrintKDia();

		// ========= Null displacement boundary condition imposition  =========

		printf("\n\n         Boundary Condition Imposition       \n");
		printf("         ========================================= \n");

		ImpositionBoundaryConditionClass();

		// ========= Evaluate M matrix =========

		printf("\n\n         Evaluate M matrix            \n");
		printf("         ========================================= \n");

		EvaluateMmatrixClass();

	}
	else {

		// ====================================================================================================================================================
		// ========= Functions for Partial Coupling  =========

		EvaluateNodalForce(dP_h);  // Evaluate nodal force from pore pressure variation

		ImpositionBoundaryConditionNeumannClass();

		// ====================================================================================================================================================

	}

	// ========= Solve linear system of equation CG method  =========

	printf("\n\n         Solve Linear System CG          \n");
	printf("         ========================================= \n");

	//SolveLinearSystemClass(VectorChr_X_h);

	//PrintDispl();

	if (DEBUGGING) PrintTimeCG();

	// ========= Evaluate strain state =========

	printf("\n\n         Evaluate Strain State         \n");
	printf("         ========================================= \n");

	EvaluateStrainState(DeltaStrainChr_h);  // Average values

	// ========= Write strain state =========

	/*printf("\n\n         Write Strain State         \n");
	printf("         ========================================= \n");

	WriteAverageStrainState(nElements, strain);*/

	// ========= Evaluate strain state =========

	printf("\n\n         Evaluate Strain State         \n");
	printf("         ========================================= \n");

	EvaluateStressStateClass(DeltaStressChr_h);  // Average values

	// ========= Write stress state =========

	/*printf("\n\n         Write Stress State         \n");
	printf("         ========================================= \n");

	WriteAverageStressState(nElements, stress);*/

}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============
void cFemOneGPU::ImpositionBoundaryConditionClass()
{
	ImpositionBoundaryCondition(Id, BlockSizeX, nNodes, nSupports, nOffsets, supp, K, B);
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============
void cFemOneGPU::EvaluateMmatrixClass()
{
	EvaluateMmatrix(Id, BlockSizeX, nNodes, nDofNode, nOffsets, K, M);
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============
void cFemOneGPU::ImpositionBoundaryConditionNeumannClass()
{
	ImpositionBoundaryConditionNeumann(Id, BlockSizeX, nNodes, nSupports, supp, B);
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============
void cFemOneGPU::SolveLinearSystemClass(double *VectorChr_X_h)
{
	SolveLinearSystem(Id, nOffsets, nDofNode, BlockSizeX, BlockMultiProcPerGpu, nNodes, B, M, K, off, t, &CGTime, &CGIter, &CGError, U, R, D, Q, S,
		delta_1_aux, delta_new, dTq, delta_old, delta_new_h);

	cudaSetDevice(Id);
	cudaMemcpy(VectorChr_X_h, U, sizeof(double)*nNodes*nDofNode, cudaMemcpyDeviceToHost);

}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============
void cFemOneGPU::EvaluateStressStateClass(double *DeltaStressChr_h)
{
	EvaluateStressState(Id, BlockSizeX, nElements, nMaterials, connect, LinkMeshColor, prop, strain, stress);

	cudaSetDevice(Id);
	cudaMemcpy(DeltaStressChr_h, stress, sizeof(double)*nElements*6, cudaMemcpyDeviceToHost);

}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============
void cFemOneGPU::EvaluateStrainState(double *DeltaStrainChr_h)
{
	int i;
	double time;

	cudaSetDevice(Id);
	cudaMemset(strain, 0, sizeof(double)*nElements*6);

	time = clock();

	for(i=0; i<numcolor; i++) {

		dim3 threadsPerBlock(BlockSizeX, BlockSizeX);
		dim3 blocksPerGrid(int(sqrt(double(numelcolor[i]))/BlockSizeX)+1, int(sqrt(double(numelcolor[i]))/BlockSizeX)+1);		

		EvaluateStrainStateColor(blocksPerGrid, threadsPerBlock, Id, nNodes, nElements, nDofNode, nMaterials, numelcolor[i], numelcolorprv[i],
			connect, coord, U, strain);

	}

	cudaMemcpy(DeltaStrainChr_h, strain, sizeof(double)*nElements*6, cudaMemcpyDeviceToHost);

	//printf("         Time Execution : %0.3f s \n", (clock()-time)/CLOCKS_PER_SEC);

}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============
void cFemOneGPU::AssemblyStiffnessMatrix()
{
	int i;
	double time;

	cudaSetDevice(Id);
	cudaMemset(K, 0, sizeof(double)*nDofNode*nNodes*nOffsets);

	time = clock();

	for(i=0; i<numcolor; i++) {

		dim3 threadsPerBlock(BlockSizeX, BlockSizeX);
		dim3 blocksPerGrid(int(sqrt(double(numelcolor[i]))/BlockSizeX)+1, int(sqrt(double(numelcolor[i]))/BlockSizeX)+1);		

		AssemblyStiffnessMatrixColor(blocksPerGrid, threadsPerBlock, Id, nNodes, nElements, nDofNode, nMaterials, numelcolor[i], numelcolorprv[i],
			connect, coord, prop, K, off, nOffsets);

	}

	//printf("         Time Execution : %0.3f s \n", (clock()-time)/CLOCKS_PER_SEC);

}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============
string _get_time()
{
	time_t now = time(0);
	tm* t = localtime(&now);
	char tmp[80];
	strftime(tmp, sizeof(tmp), "# %Y.%m.%d - %X", t);

    return tmp;
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============
void cFemOneGPU::PrintTimeCG()
{
	using namespace std;
	ofstream outfile;
	std::string fileName2 = "PrintTimeCG";
	fileName2 += ".dat";
	outfile.open(fileName2.c_str(), ios::app);

	outfile << _get_time() << "  ## Number of nodes = "<< nNodes << endl; 

	outfile << "## Assembly Stiffness Matrix" << endl;
	outfile << "   Sum K = " << setprecision(15) << (double) sumK << endl;

    std::cout << std::setprecision(8);
	outfile << " Time Execution = " << StffTime << endl;

	outfile << "## Solve Conjugate Gradient" << endl;
	outfile << " T0 = " << t[0] << " T1 = " << t[1] << " T2 = " << t[2] << " T3 = " << t[3] << " T4 = " << t[4] << " T5 = " << t[5] << " T6 = " << t[6] << " T7 = " << t[7] << " T8 = " << t[8] << " T9 = " << t[9];
	outfile << " T10 = " << t[10] << " T11 = " << t[11] << " T12 = " << t[12] << " T13 = " << t[13] << " T14 = " << t[14] << endl;
	outfile << " Time Execution = " << CGTime << "  Iteration Number = " << CGIter  << " Error = " << CGError << endl;
	outfile << "------------------------------------------------------------------------------------------------------------------------------------- " << endl  << endl;

	outfile.close();

}

//==============================================================================
void cFemOneGPU::AllocateAndCopyVectors()
{
	int i, j, GPU_Id, size;
	
	PRINT_TIME("Allocating memory to GPU...");

	// ----------------------------------------------------------------------------------------------------------------
	// Allocating memory for the Assembly of the stiffness matrix

	cudaSetDevice(Id);  // Sets device "Id" as the current device

	cudaMalloc((void **)&coord, sizeof(double)*nDofNode*nNodes);
	cudaMalloc((void **)&connect, sizeof(int)*nElements*11);
	cudaMalloc((void **)&K, sizeof(double)*nDofNode*nNodes*nOffsets);
	cudaMalloc((void **)&prop, sizeof(double)*2*nMaterials);
	cudaMalloc((void **)&stress, sizeof(double)*nElements*6);  // 6 stresses and 8 integration points
	cudaMalloc((void **)&strain, sizeof(double)*nElements*6);
	//cudaMalloc((void **)&con, sizeof(double)*numel*9);

	// ----------------------------------------------------------------------------------------------------------------
	// Allocating memory for the CG

	cudaSetDevice(Id);  // Sets device "Id" as the current device

	cudaMalloc((void **)&U, sizeof(double)*nDofNode*nNodes);
	cudaMalloc((void **)&R, sizeof(double)*nDofNode*nNodes);
	cudaMalloc((void **)&D, sizeof(double)*nDofNode*nNodes);
	cudaMalloc((void **)&D_Full, sizeof(double)*nDofNode*nNodes);
	cudaMalloc((void **)&X_Full, sizeof(double)*nDofNode*nNodes);
	cudaMalloc((void **)&Q, sizeof(double)*nDofNode*nNodes);
	cudaMalloc((void **)&M, sizeof(double)*nDofNode*nNodes);
	cudaMalloc((void **)&S, sizeof(double)*nDofNode*nNodes);
	cudaMalloc((void **)&B, sizeof(double)*nDofNode*nNodes);

	cudaMalloc((void **)&delta_new, sizeof(double));
	cudaMalloc((void **)&delta_old, sizeof(double));
	cudaMalloc((void **)&dTq, sizeof(double));

	cudaMalloc((void **) &delta_1_aux, sizeof(double)*BlockMultiProcPerGpu);

	cudaMalloc((void **)&off, sizeof(int)*nOffsets);
	cudaMalloc((void **)&supp, sizeof(int)*(nDofNode+1)*nSupports);

	delta_new_h = (double *)malloc(sizeof(double));
	X_h = (double *)malloc(sizeof(double)*nDofNode*nNodes);
	strain_h = (double *)malloc(sizeof(double)*nElements*6*8);
	stress_h = (double *)malloc(sizeof(double)*nElements*6*8);

	t = (double *)malloc(sizeof(double)*20);

	//printf("         Time - Allocate GPU Memory: %0.3f s \n", (clock()-time)/CLOCKS_PER_SEC);

	//=============== x =============== x =============== x =============== x =============== x =============== x ===============
	// Copy vectors CPU->GPU:

	PRINT_TIME("Copying memory to GPU...");

	cudaSetDevice(Id);  // Sets device "Id" as the current device

	// Offsets
	cudaMemcpy(off, chrData->ptr_offsets(), sizeof(int)*nOffsets, cudaMemcpyHostToDevice);

	// Coordinates x, y and z
	for (int dir=0; dir < 3; ++dir)
		cudaMemcpy(coord+dir*nNodes, chrData->ptr_coord(dir), sizeof(double)*nNodes, cudaMemcpyHostToDevice);

	// Connections (element index, material index, 1, nodes)
	cudaMemcpy(connect, &chrData->get_connects()[0], sizeof(int)*nElements*11, cudaMemcpyHostToDevice);

	// Elastic properties
	for (int p=0; p < 2; ++p)
		cudaMemcpy(prop+p*nMaterials, chrData->ptr_prop(p), sizeof(double)*nMaterials, cudaMemcpyHostToDevice);

	// Supports
	cudaMemcpy(supp, &chrData->get_supports()[0], sizeof(int)*(nDofNode+1)*nSupports, cudaMemcpyHostToDevice);

	END_TIME();

	//=============== x =============== x =============== x =============== x =============== x =============== x ===============
	// GPU global memory information:
	GPUInfo::ReportGpuMemory();
}

// ============================ SplitNodesAndElements ==============================
void cFemOneGPU::PrepareInputData()
{
	int i, BolockNumber;

	// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
	// xxxxxxxxxxxx Element by GPU by Color
	int gpu = 0;
	numcolor = chrData->get_nColorGroups(gpu); 
	numelcolor = (int *)malloc(sizeof(int)*numcolor);
	numelcolorprv = (int *)malloc(sizeof(int)*numcolor);

	for(i=0; i<numcolor; i++) {
		numelcolor[i] = chrData->get_color_group(gpu, i).size();

		if(i==0) numelcolorprv[i] = 0;
		else     numelcolorprv[i] = numelcolorprv[i-1] + chrData->get_color_group(gpu, i-1).size();
	}

	// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

	multiProcessorCount     = GPUInfo::devices[gpu].multiProcessorCount;
	maxThreadsPerBlock      = GPUInfo::devices[gpu].maxThreadsPerBlock;
	maxThreadsPerMultiProc  = GPUInfo::devices[gpu].maxThreadsPerMultiProcessor;
	ThreadsMultiProcPerGpu  = maxThreadsPerMultiProc * multiProcessorCount;  // maxThreadsPerMultiProc X multiProcessorCount
	BlockSizeX              = int(sqrt(double(maxThreadsPerBlock))); 
	BlockSizeY              = BlockSizeX;

	// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

	BolockNumber = int(ThreadsMultiProcPerGpu / maxThreadsPerBlock);
	if(BolockNumber>  0 && BolockNumber<= 16) BlockMultiProcPerGpu =  16;
	if(BolockNumber> 16 && BolockNumber<= 32) BlockMultiProcPerGpu =  32;
	if(BolockNumber> 32 && BolockNumber<= 64) BlockMultiProcPerGpu =  64;
	if(BolockNumber> 64 && BolockNumber<=128) BlockMultiProcPerGpu = 128;
	if(BolockNumber>128 && BolockNumber<=256) BlockMultiProcPerGpu = 256;

}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============
void cFemOneGPU::FreeMemory()
{
	/*cudaFree(coord_h);
	cudaFree(supp_h);
	cudaFree(MatType);
	cudaFree(MatElastIdx);
	cudaFree(prop_h);
	cudaFree(connect_h);
	cudaFree(GPUData);
	cudaFree(off_h);
	cudaFree(KDia);
	cudaFree(B_h);
	cudaFree(M_full);*/
}

// ============================ Print KDia ==============================
void cFemOneGPU::PrintKDia()
{
	int i, j;
	using namespace std;

	double *K_h;
	K_h = (double *)malloc(nDofNode*nNodes*nOffsets*sizeof(double));

	cudaSetDevice(Id);
	cudaMemcpy(K_h, K, sizeof(double)*nDofNode*nNodes*nOffsets, cudaMemcpyDeviceToHost);

	ofstream outfile;
	std::string fileName = "KDia";

	fileName += ".dat";
	outfile.open(fileName.c_str());

// --------------------------------------------------------------------------------------------------------------------

	outfile << endl;

	for(j=0; j<3*nNodes; j++ ) {

		outfile << j << " - " ;

		for(i=0; i<nOffsets; i++ ) {

			if(fabs(K_h[j + i*3*nNodes]) > 0.01)
				outfile << K_h[j + i*3*nNodes] << " " ;
		}

		outfile << endl;

	}

}

// ============================ Print KDia ==============================
void cFemOneGPU::PrintDispl()
{
	int i, j, k;
	using namespace std;

	double *U_h;
	U_h = (double *)malloc(nDofNode*nNodes*sizeof(double));

	cudaSetDevice(Id);
	cudaMemcpy(U_h, U, sizeof(double)*nDofNode*nNodes, cudaMemcpyDeviceToHost);

	ofstream outfile;
	std::string fileName = "Displ";

	fileName += ".dat";
	outfile.open(fileName.c_str());

// --------------------------------------------------------------------------------------------------------------------

	outfile << endl;

	int cont=0;

	for(k=0; k<13; k++) {

		for(j=0; j<22; j++) {

			for(i=0; i<22; i++) {

				outfile << U_h[3*cont+2] << " ";

				cont++;

			}

			outfile << endl;
	
		}

		outfile << endl;

	}

	outfile.close();

}

// ============================ Print PrintRate ==============================
void cFemOneGPU::PrintRate(double *rate)
{
	int i, j, k;
	using namespace std;

	double *rate_h;
	rate_h = (double *)malloc(6*nElements*sizeof(double));

	cudaSetDevice(Id);
	cudaMemcpy(rate_h, rate, sizeof(double)*6*nElements, cudaMemcpyDeviceToHost);

	ofstream outfile;
	std::string fileName = "Rate";

	fileName += ".dat";
	outfile.open(fileName.c_str());

// --------------------------------------------------------------------------------------------------------------------

	outfile << endl;

	int cont=0;

	for(k=0; k<12; k++) {

		for(j=0; j<21; j++) {

			for(i=0; i<21; i++) {

				outfile << rate_h[cont+2*nElements] << " ";

				cont++;

			}

			outfile << endl;
	
		}

		outfile << endl;

	}

	outfile.close();

}

// ============================ Write Average Strain ==============================

void cFemOneGPU::WriteAverageStrainState(int numel, double *strain)
{
	int i, j;
	double time, E[6];
	FILE *outFile;

	time = clock();

	cudaMemcpy(strain_h, strain, sizeof(double)*nElements*6*8, cudaMemcpyDeviceToHost);
    
    outFile = fopen(OUTPUT_STRAIN, "w");
    
    fprintf(outFile, "\n%Results - Strain State at Integration Points \n");
	fprintf(outFile, "\n");

	// Print results

	for(i=0; i<numel; i++) {

		E[0]=0; E[1]=0; E[2]=0; E[3]=0; E[4]=0; E[5]=0; 

		for(j=0; j<8; j++) {  // 8 = number of integration points

			E[0] += strain_h[i+0*numel+j*numel*6]/8;
			E[1] += strain_h[i+1*numel+j*numel*6]/8;
			E[2] += strain_h[i+2*numel+j*numel*6]/8;
			E[3] += strain_h[i+3*numel+j*numel*6]/8;
			E[4] += strain_h[i+4*numel+j*numel*6]/8;
			E[5] += strain_h[i+5*numel+j*numel*6]/8;

		}

		fprintf(outFile, " %d  %+0.8e %+0.8e %+0.8e %+0.8e %+0.8e %+0.8e \n", i+1, E[0], E[1], E[2], E[3], E[4], E[5]);
		
	}
	
	fclose(outFile);

	printf("         Time Execution : %0.3f s \n", (clock()-time)/CLOCKS_PER_SEC);

}

// ============================ Write Average Strain ==============================

void cFemOneGPU::WriteAverageStressState(int numel, double *stress)
{
	int i, j;
	double time, S[6];
	FILE *outFile;

	time = clock();

	cudaMemcpy(stress_h, stress, sizeof(double)*nElements*6*8, cudaMemcpyDeviceToHost);
    
    outFile = fopen(OUTPUT_STRESS, "w");
    
    fprintf(outFile, "\n%Results - Stress State at Integration Points \n");
	fprintf(outFile, "\n");

	// Print results

	for(i=0; i<numel; i++) {

		S[0]=0; S[1]=0; S[2]=0; S[3]=0; S[4]=0; S[5]=0; 

		for(j=0; j<8; j++) {  // 8 = number of integration points

			S[0] += stress_h[i+0*numel+j*numel*6]/8;
			S[1] += stress_h[i+1*numel+j*numel*6]/8;
			S[2] += stress_h[i+2*numel+j*numel*6]/8;
			S[3] += stress_h[i+3*numel+j*numel*6]/8;
			S[4] += stress_h[i+4*numel+j*numel*6]/8;
			S[5] += stress_h[i+5*numel+j*numel*6]/8;

		}

		fprintf(outFile, " %d  %+0.8e %+0.8e %+0.8e %+0.8e %+0.8e %+0.8e \n", i+1, S[0], S[1], S[2], S[3], S[4], S[5]);
		
	}
	
	fclose(outFile);

	printf("         Time Execution : %0.3f s \n", (clock()-time)/CLOCKS_PER_SEC);

}

// ============================ Write Average Strain ==============================

void cFemOneGPU::LinkMeshGridMapping(int GridCoord)
{
	int  Ti, Tj, Tk, PosMesh, PosCell, cont, i, j, k;
	
	nsi1 = chrData->get_extension_size(LEFT);
	nsj1 = chrData->get_extension_size(FRONT);
	nun = chrData->get_extension_size(DOWN);

	Ti = nsi1 + nx + nsi2;
	Tj = nsj1 + ny + nsj2;
	Tk = nov + nz + nun;

	// --------------------------------------------------------------------------------------------------------------------------------------------------------

	// Link local mesh - global mesh position

	LinkMeshMesh_h = (int *)malloc(nx*ny*nz*sizeof(int));

	cont = 0;

	for(k=0; k<nz; k++) {

		for(j=0; j<ny; j++) {

			for(i=0; i<nx; i++) {

				PosMesh = Ti*Tj*(nun+k) + Ti*(nsj1+j) + nsi1 + i;

				// --------------------------------------------------------------------------------

				LinkMeshMesh_h[cont] = PosMesh;  // Begin in 0

				cont++;

				// --------------------------------------------------------------------------------

			}

		}

	}

	// --------------------------------------------------------------------------------------------------------------------------------------------------------

	// Link local mesh - cell position

	LinkMeshCell_h = (int *)malloc(nx*ny*nz*sizeof(int));

	cont = 0;

	for(k=0; k<nz; k++) {

		for(j=0; j<ny; j++) {

			for(i=0; i<nx; i++) {

				switch(GridCoord) { 

				case 0:  // Positive cross product
					PosCell = nx*ny*(nz-1-k) + nx*(ny-1-j) + i;  // Namorado
					break; 

				case 1: // Negative cross product
					PosCell = nx*ny*(nz-1-k) + Ti*j + i;       // Campo B
					break; 

				}

				// --------------------------------------------------------------------------------

				LinkMeshCell_h[cont] = PosCell;  // Begin in 0

				cont++;

				// --------------------------------------------------------------------------------

			}

		}

	}

	// ============================================================
	//IF_DEBUGGING { 
	/*	using namespace std;
		ofstream outfile;
		std::string fileName2 = "LinkMeshGridMapping";
		fileName2 += ".dat";
		outfile.open(fileName2.c_str());
		using std::ios;
		using std::setw;
		outfile.setf(ios::fixed,ios::floatfield);
		outfile.precision(2);

		cont = 0;

		outfile << "LinkMeshMesh_h" << endl;

		for(int k=0; k<_nz; k++) {
			
			for(int j=0; j<_ny; j++) {

				for(int i=0; i<_nx; i++) {

					outfile << LinkMeshMesh_h[cont] << " ";
					cont++;

				}

				outfile << endl;
			}
		}

		outfile << endl;

		// ----------------------------------------------------------------------------------------

		cont = 0;

		outfile << "LinkMeshCell_h" << endl;

		for(int k=0; k<_nz; k++) {
			
			for(int j=0; j<_ny; j++) {

				for(int i=0; i<_nx; i++) {

					outfile << LinkMeshCell_h[cont] << " ";
					cont++;

				}

				outfile << endl;
			}
		}

		outfile.close();*/
	//}
	// ============================================================
}

//==============================================================================
void cFemOneGPU::EvalColoringMeshBrick8Struct()
{
	int i, j, k, l;
	int pos, add_i, add_ij;
	int color_beg[8], i_beg[8], j_beg[8], k_beg[8];


	color_beg[0] = 0;
	color_beg[0] = 0;
	color_beg[1] = 1;
	color_beg[2] = nx;
	color_beg[3] = nx+1;

	color_beg[4] = 0+nx*ny;
	color_beg[5] = 1+nx*ny;
	color_beg[6] = nx+nx*ny;
	color_beg[7] = nx+1+nx*ny;

	i_beg[0] = 0;
	i_beg[1] = 1;
	i_beg[2] = 0;
	i_beg[3] = 1;

	i_beg[4] = 0;
	i_beg[5] = 1;
	i_beg[6] = 0;
	i_beg[7] = 1;

	j_beg[0] = 0;
	j_beg[1] = 0;
	j_beg[2] = 1;
	j_beg[3] = 1;

	j_beg[4] = 0;
	j_beg[5] = 0;
	j_beg[6] = 1;
	j_beg[7] = 1;

	k_beg[0] = 0;
	k_beg[1] = 0;
	k_beg[2] = 0;
	k_beg[3] = 0;

	k_beg[4] = 1;
	k_beg[5] = 1;
	k_beg[6] = 1;
	k_beg[7] = 1;


	ElemColorByGPU.resize(8);

	for(l=0; l<8; l++) {

		pos = color_beg[l];
		add_i = 0;
		add_ij = 0;

		for(k=k_beg[l]; k<nz; k+=2) {

			add_i = 0;

			for(j=j_beg[l]; j<ny; j+=2) {

				for(i=i_beg[l]; i<nx; i+=2) {

					ElemColorByGPU[l].push_back(pos);  // Begin in 0

					pos += 2;

				}

				add_i += 2*nx;
				pos = color_beg[l] + add_i + add_ij;

			}

			add_ij += 2*nx*ny;
			pos = color_beg[l] + add_ij;

		}

	}

	// Check coloring algorighm:

	int cont=0;

	for(l=0; l<8; l++)
		cont += ElemColorByGPU[l].size();

	// --------------------------------------------------------------------------------------------
	/*
	int numelem = nx*ny*nz;  // Total number of elements

	if(numelem != cont)
		printf("\n         Problem with the coloring algorihm!\n");
	else
		printf("\n         Coloring algorihm performed well.\n" );
	*/

	// --------------------------------------------------------------------------------------------

	int *LinkMeshMeshColorAux_h;
	LinkMeshMeshColorAux_h = (int *)malloc(nx*ny*nz*sizeof(int));

	cont = 0;

	for(i=0; i<ElemColorByGPU.size(); i++) {

		for(j=0; j<ElemColorByGPU[i].size(); j++) {

			LinkMeshMeshColorAux_h[cont] = LinkMeshMesh_h[ElemColorByGPU[i][j]];  // Begin in 0
			LinkMeshCellColor_h[cont] = LinkMeshCell_h[ElemColorByGPU[i][j]];     // Begin in 0

			cont++;

		}

	}

	// --------------------------------------------------------------------------------------------

	for(i=0; i<nElements; i++) {

		for(j=0; j<nx*ny*nz; j++) {

			if(chrData->get_element_pos(i)-1 == LinkMeshMeshColorAux_h[j]) 
				LinkMeshMeshColor_h[j] = i;  // Begin in 0

		}

	}

	// --------------------------------------------------------------------------------------------

	numelcolornodalforce = (int *)malloc(sizeof(int)*numcolor);
	numelcolornodalforceprv = (int *)malloc(sizeof(int)*numcolor);

	for(i=0; i<numcolor; i++) {
		numelcolornodalforce[i] = ElemColorByGPU[i].size();

		if(i==0) numelcolornodalforceprv[i] = 0;
		else     numelcolornodalforceprv[i] = numelcolornodalforceprv[i-1] + ElemColorByGPU[i-1].size();
	}

	// --------------------------------------------------------------------------------------------

	cudaSetDevice(Id);  // Sets device "Id" as the current device

	cudaMemcpy(LinkMeshMeshColor, LinkMeshMeshColor_h, sizeof(int)*nx*ny*nz, cudaMemcpyHostToDevice);
	cudaMemcpy(LinkMeshCellColor, LinkMeshCellColor_h, sizeof(int)*nx*ny*nz, cudaMemcpyHostToDevice);

	// ============================================================
	//IF_DEBUGGING { 
		/*using namespace std;
		ofstream outfile;
		std::string fileName2 = "LinkMeshMeshColor_h";
		fileName2 += ".dat";
		outfile.open(fileName2.c_str());
		using std::ios;
		using std::setw;
		outfile.setf(ios::fixed,ios::floatfield);
		outfile.precision(2);

		cont = 0;

		outfile << "LinkMeshMeshColor_h" << endl;

		for(int k=0; k<_nz; k++) {
			
			for(int j=0; j<_ny; j++) {

				for(int i=0; i<_nx; i++) {

					outfile << LinkMeshMeshColor_h[cont] << " ";
					cont++;

				}

				outfile << endl;
			}
		}

		outfile << endl;

		// ----------------------------------------------------------------------------------------

		cont = 0;

		outfile << "LinkMeshCellColor_h" << endl;

		for(int k=0; k<_nz; k++) {
			
			for(int j=0; j<_ny; j++) {

				for(int i=0; i<_nx; i++) {

					outfile << LinkMeshCellColor_h[cont] << " ";
					cont++;

				}

				outfile << endl;
			}
		}

		outfile.close();*/
	//}
	// ============================================================

}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============
void cFemOneGPU::AllocateAndCopyVectorsPartialCoupling()
{
	int numelres = nx*ny*nz;
	
	LinkMeshMeshColor_h = (int *)malloc(numelres*sizeof(int));
	LinkMeshCellColor_h = (int *)malloc(numelres*sizeof(int));
	LinkMeshColor_h = (int *)malloc(nElements*sizeof(int));

	cudaSetDevice(Id);  // Sets device "Id" as the current device

	cudaMalloc((void **)&dP, sizeof(double)*numelres);
	cudaMalloc((void **)&LinkMeshMeshColor, sizeof(int)*numelres);
	cudaMalloc((void **)&LinkMeshCellColor, sizeof(int)*numelres);
	cudaMalloc((void **)&LinkMeshColor, sizeof(int)*nElements);

	//F_h = (double *)malloc(sizeof(double)*nDofNode*nNodes);

}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============
void cFemOneGPU::EvaluateNodalForce(double *dP_h)
{
	int i, j, k;
	//double time;

	PRINT_TIME("Evaluating nodal forces");

	cudaSetDevice(Id);

	// Copying the pore pressure from CPU to GPU memory:
	cudaMemcpy(dP, dP_h, sizeof(double)*nx*ny*nz, cudaMemcpyHostToDevice);

	// Cleaning vectors:
	cudaMemset(B, 0, sizeof(double)*nDofNode*nNodes);
	
	for(i=0; i<numcolor; i++) {

		dim3 threadsPerBlock(BlockSizeX, BlockSizeX);
		dim3 blocksPerGrid(int(sqrt(double(numelcolornodalforce[i]))/BlockSizeX)+1, int(sqrt(double(numelcolornodalforce[i]))/BlockSizeX)+1);		

		EvaluateNodalForceColor(blocksPerGrid, threadsPerBlock, Id, nNodes, nElements, nDofNode, nMaterials, numelcolornodalforce[i], numelcolornodalforceprv[i],
			connect, coord, LinkMeshMeshColor, LinkMeshCellColor, dP, B);

	}

	std::vector<double> F_h(nDofNode*nNodes);
	//cudaMemcpy(&F_h[0], B, sizeof(double)*nDofNode*nNodes, cudaMemcpyDeviceToHost);

	// ============================================================
	//IF_DEBUGGING { 
	/*using namespace std;
	ofstream outfile;
	std::string fileName2 = "F_h";
	fileName2 += ".dat";
	outfile.open(fileName2.c_str());

	int cont=0;

	for(k=0; k<13; k++) {

		for(j=0; j<22; j++) {

			for(i=0; i<22; i++) {

				if(F_h[3*cont+2] != 0.) 
					outfile << F_h[3*cont+2] << " ";
				else
					outfile << "0" << " ";

				cont++;

			}

			outfile << endl;
	
		}

		outfile << endl;

	}

	outfile.close();*/

	//}

	// ----------------------------------------------------------------------------------------

	//printf("         Time Execution : %0.3f s \n", (clock()-time)/CLOCKS_PER_SEC);

	END_TIME();
}

//==============================================================================
// Evalauting initial stress state:
void cFemOneGPU::EvaluateInitialStressState(double *StressTotalChr_h)
{

	int i, j, k, l, kaux, Ti, Tj, Tk, ElemIdx, ElemColorIdx;
	double StressInitialAux, densCurr, densPrev, K0x, K0y;  
	//  dg = grain density, ds = saturated density,K0x = relation between Sx and Sz, K0y = relation between Sy and Sz
	double dw = 9810., ZcenterCurr, ZcenterPrev, Ztop;

	Ti = nsi1 + nx + nsi2;
	Tj = nsj1 + ny + nsj2;
	Tk = nov + nz + nun;
	
	// ------------------------------------------------------------------

	for(j=0; j<Tj; j++) {                        // Loop elements in x direcion

		for(i=0; i<Ti; i++) {                    // Loop elements in y direcion

			StressInitialAux = 0.;

			for(kaux=0; kaux<Tk; kaux++) {       // Vertical Direction

				k = Tk - 1 - kaux;

				ElemIdx = k*(Ti*Tj) + j*Ti + i;

				ElemColorIdx = LinkMeshColor_h[ElemIdx];

				// Material density
				densCurr = chrData->get_prop(ElemColorIdx, DENSITY); 
				K0x      = chrData->get_prop(ElemColorIdx, K0X); 
				K0y      = chrData->get_prop(ElemColorIdx, K0Y);
				
				// --------------------------------------------------------------------------------

				Ztop = 0.;

				for(int j2=4; j2<8; j2++) {  

					Ztop += chrData->get_coord(ElemColorIdx, j2, Z)/4;

				}

				// --------------------------------------------------------------------------------

				ZcenterCurr = 0.;

				for(int j2=0; j2<8; j2++) { 

					ZcenterCurr += chrData->get_coord(ElemColorIdx, j2, Z)/8;  

				}

				// --------------------------------------------------------------------------------

				if(kaux == 0)      // First layer of elements
					StressInitialAux = densCurr*(ZcenterCurr-Ztop) + dw*Ztop ;  

				else               // secound layer of elements and son on
					StressInitialAux += densCurr*(ZcenterCurr-Ztop) + densPrev*(Ztop - ZcenterPrev);

				// --------------------------------------------------------------------------------

				ZcenterPrev = ZcenterCurr;
				densPrev = densCurr;
		
				// (-) negative stress = rock/soil mechanics
				StressTotalChr_h[ElemIdx+0*nElements] = K0x*StressInitialAux;  // Sx
				StressTotalChr_h[ElemIdx+1*nElements] = K0y*StressInitialAux;  // Sy
				StressTotalChr_h[ElemIdx+2*nElements] =     StressInitialAux;  // Sz

				// --------------------------------------------------------------------------------

			}                                    // Vertical Direction

		}

	}

}

//==============================================================================
// Evalauting initial stress state:
void cFemOneGPU::LinkColorMapping()
{

	int i, j;

	// ------------------------------------------------------------------

	for(i=0; i<nElements; i++) {                    

		// Searching in the coloring element list
		for(j=0; j<nElements; j++) { 
			if((i+1) == chrData->get_element_pos(j)) {
				LinkMeshColor_h[i] = j;
				break;
			}
		}

	}

	// ------------------------------------------------------------------

	cudaMemcpy(LinkMeshColor, LinkMeshColor_h, sizeof(int)*nElements, cudaMemcpyHostToDevice);

	// ------------------------------------------------------------------

	/*using namespace std;
	ofstream outfile;
	std::string fileName2 = "LinkMeshColor_h";
	fileName2 += ".dat";
	outfile.open(fileName2.c_str());

	for(j=0; j<nElements; j++) { 
		
		outfile << j+1 << "  " << LinkMeshColor_h[j]+1 << endl;

	}*/

}
