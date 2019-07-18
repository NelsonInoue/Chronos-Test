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
#include "FemOneGPU.h"
#include <omp.h>
#include <cuda_runtime.h>
#include "defs.h"


// Utilities and system includes

#define streq(s1,s2)	((s1[0]==s2[0]) && strcmp(s1,s2)==0)

//----------------------------------------------------
// External Functions for Multi GPUs Implementation						 
//----------------------------------------------------

extern "C" void EvaluateMmatrix(int Id, int BlockSizeX, int _iNumMeshNodes, int _iNumDofNode, int _inumDiaPart, double *K, double *M);

extern "C" void AssemblyStiffnessMatrixColor(dim3 blocksPerGrid, dim3 threadsPerBlock, int Id, int _iNumMeshNodes, int _iNumMeshElem, int _iNumDofNode, int _iNumElasMat, int numelcolor, int numelcolorprv,
			int *connect, double *coord, double *prop, double *K, int *offfull); 

extern "C" void EvaluateStrainStateColor(dim3 blocksPerGrid, dim3 threadsPerBlock, int Id, int _iNumMeshNodes, int _iNumMeshElem, int _iNumDofNode, int _iNumElasMat, int numelcolor, int numelcolorprv,
			int *connect, double *coord, double *X, double *strain);

extern "C" void EvaluateStressState(int Id, int BlockSizeX, int _iNumMeshElem, int _iNumElasMat, int *connect, double *prop, double *strain, double *stress); 
                                           
extern "C" void ImpositionBoundaryCondition(int Id, int BlockSizeX, int _iNumMeshNodes, int _iNumSuppNodes, int _inumDiaPart, int *supp, double *K);

extern "C" void SolveLinearSystem(int Id, int _inumDiaPart, int _iNumDofNode, int BlockSizeX, int BlockMultiProcPerGpu, int _iNumMeshNodes, double *B, double *M, double *K, int *off, double *t, double *CGTime, int *CGIter, double *CGError, double *X, double *R, double *D, double *Q, double *S,
		double *delta_1_aux, double *delta_new, double *dTq, double *delta_old, double *delta_new_h);


//=============================================================================
cFemOneGPU::cFemOneGPU()
{

}

//========================================================================================================
void cFemOneGPU::AnalyzeFemOneGPU()
{
	double time;

	// ========= Read Data =========
	
	printf("         Read input file             \n");
	printf("         ========================================= \n");

	in = new cInput(); 

	in->ReadInputFile();         // input file

	// ----------------------------------------------------------------------------------------------------------------

	// ========= Prepare input data =========

	PrepareInputData();

	//SplitNodesAndElements();  // Split the number of nodes for Reservoir models
	
	// ========= Allocate memory =========

	printf("\n\n         Allocate and Copy Memory              \n");
	printf("         ========================================= \n");
	AllocateAndCopyVectors();      

	// ========= Assembly the stiffness matrix on one GPU =========

	printf("\n\n         Assembly Stiffness Matrix         \n");
	printf("         ========================================= \n");

	AssemblyStiffnessMatrix();  

	//PrintKDia();

	// ========= Null displacement boundary condition imposition  =========

	printf("\n\n         Boundary Condition Imposition       \n");
	printf("         ========================================= \n");

	ImpositionBoundaryCondition(Id, BlockSizeX, in->_iNumMeshNodes, in->_iNumSuppNodes, in->_inumDiaPart, supp, K);

	// ========= Evaluate M matrix =========

	printf("\n\n         Evaluate M matrix            \n");
	printf("         ========================================= \n");

	EvaluateMmatrix(Id, BlockSizeX, in->_iNumMeshNodes, in->_iNumDofNode, in->_inumDiaPart, K, M);

	// ========= Solve linear system of equation CG method  =========

	printf("\n\n         Solve Linear System CG          \n");
	printf("         ========================================= \n");

	SolveLinearSystem(Id, in->_inumDiaPart, in->_iNumDofNode, BlockSizeX, BlockMultiProcPerGpu, in->_iNumMeshNodes, B, M, K, off, t, &CGTime, &CGIter, &CGError, X, R, D, Q, S,
		delta_1_aux, delta_new, dTq, delta_old, delta_new_h);

	IF_DEBUGGING PrintTimeCG();

	// ========= Evaluate strain state =========

	printf("\n\n         Evaluate Strain State         \n");
	printf("         ========================================= \n");

	EvaluateStrainState();  // Average values

	// ========= Write strain state =========

	printf("\n\n         Write Strain State         \n");
	printf("         ========================================= \n");

	WriteAverageStrainState(in->_iNumMeshElem, strain);

	// ========= Evaluate strain state =========

	printf("\n\n         Evaluate Strain State         \n");
	printf("         ========================================= \n");

	EvaluateStressState(Id, BlockSizeX, in->_iNumMeshElem, in->_iNumElasMat, connect, prop, strain, stress);  // Average values

	// ========= Write stress state =========

	printf("\n\n         Write Stress State         \n");
	printf("         ========================================= \n");

	WriteAverageStressState(in->_iNumMeshElem, stress);

}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============
void cFemOneGPU::EvaluateStrainState()
{
	int i;
	double time;

	cudaSetDevice(Id);
	cudaMemset(K, 0, sizeof(double)*in->_iNumDofNode*in->_iNumMeshNodes*in->_inumDiaPart);

	time = clock();

	for(i=0; i<numcolor; i++) {

		dim3 threadsPerBlock(BlockSizeX, BlockSizeX);
		dim3 blocksPerGrid(int(sqrt(double(numelcolor[i]))/BlockSizeX)+1, int(sqrt(double(numelcolor[i]))/BlockSizeX)+1);		

		EvaluateStrainStateColor(blocksPerGrid, threadsPerBlock, Id, in->_iNumMeshNodes, in->_iNumMeshElem, in->_iNumDofNode, in->_iNumElasMat, numelcolor[i], numelcolorprv[i],
			connect, coord, X, strain);

	}

	printf("         Time Execution : %0.3f s \n", (clock()-time)/CLOCKS_PER_SEC);

}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============
void cFemOneGPU::AssemblyStiffnessMatrix()
{
	int i;
	double time;

	cudaSetDevice(Id);
	cudaMemset(K, 0, sizeof(double)*in->_iNumDofNode*in->_iNumMeshNodes*in->_inumDiaPart);

	time = clock();

	for(i=0; i<numcolor; i++) {

		dim3 threadsPerBlock(BlockSizeX, BlockSizeX);
		dim3 blocksPerGrid(int(sqrt(double(numelcolor[i]))/BlockSizeX)+1, int(sqrt(double(numelcolor[i]))/BlockSizeX)+1);		

		AssemblyStiffnessMatrixColor(blocksPerGrid, threadsPerBlock, Id, in->_iNumMeshNodes, in->_iNumMeshElem, in->_iNumDofNode, in->_iNumElasMat, numelcolor[i], numelcolorprv[i],
			connect, coord, prop, K, offfull);

	}

	printf("         Time Execution : %0.3f s \n", (clock()-time)/CLOCKS_PER_SEC);

}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============
string get_time()
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

	outfile << get_time() << "  ## Number of nodes = "<< in->_iNumMeshNodes << endl; 

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
	int i, j, GPU_Id;
	double time, FreeMem[4], UsedMem[4], TotalMem[4];
	size_t free_byte, total_byte;

	time = clock();

	// ----------------------------------------------------------------------------------------------------------------
	// Allocating memory for the Assembly of the stiffness matrix

	cudaSetDevice(Id);  // Sets device "Id" as the current device

	cudaMalloc((void **)&coord, sizeof(double)*in->_iNumDofNode*in->_iNumMeshNodes);
	cudaMalloc((void **)&connect, sizeof(int)*in->_iNumMeshElem*11);
	cudaMalloc((void **)&K, sizeof(double)*in->_iNumDofNode*in->_iNumMeshNodes*in->_inumDiaPart);
	cudaMalloc((void **)&prop, sizeof(double)*2*in->_iNumElasMat);
	cudaMalloc((void **)&offfull, sizeof(int)*in->_inumDiaFull);
	cudaMalloc((void **)&stress, sizeof(double)*in->_iNumMeshElem*6*8);  // 6 stresses and 8 integration points
	cudaMalloc((void **)&strain, sizeof(double)*in->_iNumMeshElem*6*8);
	//cudaMalloc((void **)&con, sizeof(double)*numel*9);

	// ----------------------------------------------------------------------------------------------------------------
	// Allocating memory for the CG

	cudaSetDevice(Id);  // Sets device "Id" as the current device

	cudaMalloc((void **)&X, sizeof(double)*in->_iNumDofNode*in->_iNumMeshNodes);
	cudaMalloc((void **)&R, sizeof(double)*in->_iNumDofNode*in->_iNumMeshNodes);
	cudaMalloc((void **)&D, sizeof(double)*in->_iNumDofNode*in->_iNumMeshNodes);
	cudaMalloc((void **)&D_Full, sizeof(double)*in->_iNumDofNode*in->_iNumMeshNodes);
	cudaMalloc((void **)&X_Full, sizeof(double)*in->_iNumDofNode*in->_iNumMeshNodes);
	cudaMalloc((void **)&Q, sizeof(double)*in->_iNumDofNode*in->_iNumMeshNodes);
	cudaMalloc((void **)&M, sizeof(double)*in->_iNumDofNode*in->_iNumMeshNodes);
	cudaMalloc((void **)&S, sizeof(double)*in->_iNumDofNode*in->_iNumMeshNodes);
	cudaMalloc((void **)&B, sizeof(double)*in->_iNumDofNode*in->_iNumMeshNodes);

	cudaMalloc((void **)&delta_new, sizeof(double));
	cudaMalloc((void **)&delta_old, sizeof(double));
	cudaMalloc((void **)&dTq, sizeof(double));

	cudaMalloc((void **) &delta_1_aux, sizeof(double)*BlockMultiProcPerGpu);

	cudaMalloc((void **)&off, sizeof(int)*in->_inumDiaPart);
	cudaMalloc((void **)&supp, sizeof(int)*(in->_iNumDofNode+1)*in->_iNumSuppNodes);

	delta_new_h = (double *)malloc(sizeof(double));
	X_h = (double *)malloc(sizeof(double)*in->_iNumDofNode*in->_iNumMeshNodes);
	strain_h = (double *)malloc(sizeof(double)*in->_iNumMeshElem*6*8);
	stress_h = (double *)malloc(sizeof(double)*in->_iNumMeshElem*6*8);

	t = (double *)malloc(sizeof(double)*20);

	printf("         Time - Allocate GPU Memory: %0.3f s \n", (clock()-time)/CLOCKS_PER_SEC);

	//=============== x =============== x =============== x =============== x =============== x =============== x ===============
	// Copy vectors CPU->GPU:

	time = clock();

	cudaSetDevice(Id);  // Sets device "Id" as the current device

	cudaMemcpy(off, in->off_h, sizeof(int)*in->_inumDiaPart, cudaMemcpyHostToDevice);
	cudaMemcpy(B, in->B_h, sizeof(double)*in->_iNumDofNode*in->_iNumMeshNodes, cudaMemcpyHostToDevice);
	cudaMemcpy(coord, in->coord_h, sizeof(double)*in->_iNumDofNode*in->_iNumMeshNodes, cudaMemcpyHostToDevice);
	cudaMemcpy(connect, in->connect_h, sizeof(int)*in->_iNumMeshElem*11, cudaMemcpyHostToDevice);
	cudaMemcpy(offfull, in->offfull_h, sizeof(int)*in->_inumDiaFull, cudaMemcpyHostToDevice);
	cudaMemcpy(prop, in->prop_h, sizeof(double)*2*in->_iNumElasMat, cudaMemcpyHostToDevice);
	cudaMemcpy(supp, in->supp_h, sizeof(int)*(in->_iNumDofNode+1)*in->_iNumSuppNodes, cudaMemcpyHostToDevice);

	printf("         Time - Copy Vectors CPU->GPU: %0.3f \n", (clock()-time)/CLOCKS_PER_SEC);

	//=============== x =============== x =============== x =============== x =============== x =============== x ===============
	// GPU global memory information:

	printf("\n");
	printf("         GPU global memory report \n");
	printf("         ========================================= ");
	printf("\n");

	for(i=0; i<in->deviceCount; i++) {
		cudaSetDevice(i);  // Sets device "0" as the current device
		cudaMemGetInfo( &free_byte, &total_byte );

		FreeMem[i] = free_byte/1073741824;
		TotalMem[i] = total_byte/1073741824;
		UsedMem[i] = TotalMem[i]-FreeMem[i];

	}

	printf("\n");
	printf("                       GPU 0    GPU 1    GPU 2    GPU 3\n");
	printf("         Free Memory:  %0.2f Gb  %0.2f Gb  %0.2f Gb  %0.2f Gb \n", FreeMem[0], FreeMem[1], FreeMem[2], FreeMem[3]); 
	printf("         Used Memory:  %0.2f Gb  %0.2f Gb  %0.2f Gb  %0.2f Gb \n", UsedMem[0], UsedMem[1], UsedMem[2], UsedMem[3]);
	printf("         Memory:       %0.2f Gb  %0.2f Gb  %0.2f Gb  %0.2f Gb \n", TotalMem[0], TotalMem[1], TotalMem[2], TotalMem[3]);
	printf("\n");
	printf("         Total Free Memory:  %0.2f Gb \n", FreeMem[0]+FreeMem[1]+FreeMem[2]+FreeMem[3]); 
	printf("         Total Used Memory:  %0.2f Gb \n", UsedMem[0]+UsedMem[1]+UsedMem[2]+UsedMem[3]);
	printf("         Total Memory:       %0.2f Gb \n", TotalMem[0]+TotalMem[1]+TotalMem[2]+TotalMem[3]);

}

// ============================ SplitNodesAndElements ==============================
void cFemOneGPU::PrepareInputData()
{
	int i, BolockNumber;

	// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
	// xxxxxxxxxxxx Element by GPU by Color

	numcolor = in->NumColorbyGPU[0];
	numelcolor = (int *)malloc(sizeof(int)*numcolor);
	numelcolorprv = (int *)malloc(sizeof(int)*numcolor);

	for(i=0; i<numcolor; i++) {
		numelcolor[i] = in->NumElemByGPUbyColor[0][i];

		if(i==0) numelcolorprv[i] = 0;
		else     numelcolorprv[i] = numelcolorprv[i-1] + in->NumElemByGPUbyColor[0][i-1];
	}

	// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

	multiProcessorCount     = in->GPUData[Id*6+0];
	maxThreadsPerBlock      = in->GPUData[Id*6+4];
	maxThreadsPerMultiProc  = in->GPUData[Id*6+5];
	ThreadsMultiProcPerGpu  = in->GPUData[Id*6+5] * in->GPUData[Id*6+0];  // maxThreadsPerMultiProc X multiProcessorCount
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
	cudaFree(offfull_h);
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
	K_h = (double *)malloc(in->_iNumDofNode*in->_iNumMeshNodes*in->_inumDiaPart*sizeof(double));

	cudaSetDevice(Id);
	cudaMemcpy(K_h, K, sizeof(double)*in->_iNumDofNode*in->_iNumMeshNodes*in->_inumDiaPart, cudaMemcpyDeviceToHost);

	ofstream outfile;
	std::string fileName = "KDia";

	fileName += ".dat";
	outfile.open(fileName.c_str());

// --------------------------------------------------------------------------------------------------------------------

	outfile << endl;

	for(j=0; j<3*in->_iNumMeshNodes; j++ ) {

		outfile << j << " - " ;

		for(i=0; i<in->_inumDiaPart; i++ ) {

			if(fabs(K_h[j + i*3*in->_iNumMeshNodes]) > 0.01)
				outfile << K_h[j + i*3*in->_iNumMeshNodes] << " " ;
		}

		outfile << endl;

	}

}

// ============================ Write Average Strain ==============================

void cFemOneGPU::WriteAverageStrainState(int numel, double *strain)
{
	int i, j;
	double time, E[6];
	FILE *outFile;

	time = clock();

	cudaMemcpy(strain_h, strain, sizeof(double)*in->_iNumMeshElem*6*8, cudaMemcpyDeviceToHost);
    
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

	cudaMemcpy(stress_h, stress, sizeof(double)*in->_iNumMeshElem*6*8, cudaMemcpyDeviceToHost);
    
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