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

extern "C" void EvaluateStressState(int Id, int BlockSizeX, int _iNumMeshElem, int _iNumElasMat, int *connect, int *LinkMeshColor, double *prop, double *strain, double *stress); 
                                           
extern "C" void ImpositionBoundaryCondition(int Id, int BlockSizeX, int _iNumMeshNodes, int _iNumSuppNodes, int _inumDiaPart, int *supp, double *K, double *B);

extern "C" void ImpositionBoundaryConditionNeumann(int Id, int BlockSizeX, int _iNumMeshNodes, int _iNumSuppNodes, int *supp, double *B);

extern "C" void SolveLinearSystem(int Id, int _inumDiaPart, int _iNumDofNode, int BlockSizeX, int BlockMultiProcPerGpu, int _iNumMeshNodes, double *B, double *M, double *K, int *off, double *t, double *CGTime, int *CGIter, double *CGError, double *X, double *R, double *D, double *Q, double *S,
		double *delta_1_aux, double *delta_new, double *dTq, double *delta_old, double *delta_new_h);

extern "C" void EvaluateNodalForceColor(dim3 blocksPerGrid, dim3 threadsPerBlock, int Id, int _iNumMeshNodes, int _iNumMeshElem, int _iNumDofNode, int _iNumElasMat, int numelcolornodalforce, int numelcolornodalforceprv,
			int *connect, double *coord, int *LinkMeshMeshColor, int *LinkMeshCellColor, double *dP, double *B);

//=============================================================================
cFemOneGPU::cFemOneGPU()
{

}

//========================================================================================================
void cFemOneGPU::AnalyzeFemOneGPU(int ii, int jj, int GridCoord, double *dP_h, double *DeltaStrainChr_h, double *DeltaStressChr_h)
{
	double time;

	// ========= Read Data =========

	if(ii == 0 && jj ==0) {

		printf("         Read input file             \n");
		printf("         ========================================= \n");

		in = new cInput(); 

		//in->ReadInputFile();         // Read input file

		// ========= Prepare input data =========

		PrepareInputData();

		//SplitNodesAndElements();  // Split the number of nodes for Reservoir models

		// ========= Allocate memory =========

		printf("\n\n         Allocate and Copy Memory              \n");
		printf("         ========================================= \n");
		AllocateAndCopyVectors();      

		// ========================================================================================================================================================
		// ========= Functions for Partial Coupling  =========

		in->ReadMeshGeometry();  // Read geometry data for the coupling

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

	IF_DEBUGGING PrintTimeCG();

	// ========= Evaluate strain state =========

	printf("\n\n         Evaluate Strain State         \n");
	printf("         ========================================= \n");

	EvaluateStrainState(DeltaStrainChr_h);  // Average values

	// ========= Write strain state =========

	/*printf("\n\n         Write Strain State         \n");
	printf("         ========================================= \n");

	WriteAverageStrainState(in->_iNumMeshElem, strain);*/

	// ========= Evaluate strain state =========

	printf("\n\n         Evaluate Strain State         \n");
	printf("         ========================================= \n");

	EvaluateStressStateClass(DeltaStressChr_h);  // Average values

	// ========= Write stress state =========

	/*printf("\n\n         Write Stress State         \n");
	printf("         ========================================= \n");

	WriteAverageStressState(in->_iNumMeshElem, stress);*/

}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============
void cFemOneGPU::ImpositionBoundaryConditionClass()
{
	ImpositionBoundaryCondition(Id, BlockSizeX, in->_iNumMeshNodes, in->_iNumSuppNodes, in->_inumDiaPart, supp, K, B);
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============
void cFemOneGPU::EvaluateMmatrixClass()
{
	EvaluateMmatrix(Id, BlockSizeX, in->_iNumMeshNodes, in->_iNumDofNode, in->_inumDiaPart, K, M);
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============
void cFemOneGPU::ImpositionBoundaryConditionNeumannClass()
{
	ImpositionBoundaryConditionNeumann(Id, BlockSizeX, in->_iNumMeshNodes, in->_iNumSuppNodes, supp, B);
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============
void cFemOneGPU::SolveLinearSystemClass(double *VectorChr_X_h)
{
	SolveLinearSystem(Id, in->_inumDiaPart, in->_iNumDofNode, BlockSizeX, BlockMultiProcPerGpu, in->_iNumMeshNodes, B, M, K, off, t, &CGTime, &CGIter, &CGError, X, R, D, Q, S,
		delta_1_aux, delta_new, dTq, delta_old, delta_new_h);

	cudaSetDevice(Id);
	cudaMemcpy(VectorChr_X_h, X, sizeof(double)*in->_iNumMeshNodes*in->_iNumDofNode, cudaMemcpyDeviceToHost);

}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============
void cFemOneGPU::EvaluateStressStateClass(double *DeltaStressChr_h)
{
	EvaluateStressState(Id, BlockSizeX, in->_iNumMeshElem, in->_iNumElasMat, connect, LinkMeshColor, prop, strain, stress);

	cudaSetDevice(Id);
	cudaMemcpy(DeltaStressChr_h, stress, sizeof(double)*in->_iNumMeshElem*6, cudaMemcpyDeviceToHost);

}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============
void cFemOneGPU::EvaluateStrainState(double *DeltaStrainChr_h)
{
	int i;
	double time;

	cudaSetDevice(Id);
	cudaMemset(strain, 0, sizeof(double)*in->_iNumMeshElem*6);

	time = clock();

	for(i=0; i<numcolor; i++) {

		dim3 threadsPerBlock(BlockSizeX, BlockSizeX);
		dim3 blocksPerGrid(int(sqrt(double(numelcolor[i]))/BlockSizeX)+1, int(sqrt(double(numelcolor[i]))/BlockSizeX)+1);		

		EvaluateStrainStateColor(blocksPerGrid, threadsPerBlock, Id, in->_iNumMeshNodes, in->_iNumMeshElem, in->_iNumDofNode, in->_iNumElasMat, numelcolor[i], numelcolorprv[i],
			connect, coord, X, strain);

	}

	cudaMemcpy(DeltaStrainChr_h, strain, sizeof(double)*in->_iNumMeshElem*6, cudaMemcpyDeviceToHost);

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

	outfile << _get_time() << "  ## Number of nodes = "<< in->_iNumMeshNodes << endl; 

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
	cudaMalloc((void **)&stress, sizeof(double)*in->_iNumMeshElem*6);  // 6 stresses and 8 integration points
	cudaMalloc((void **)&strain, sizeof(double)*in->_iNumMeshElem*6);
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
	//cudaMemcpy(B, in->B_h, sizeof(double)*in->_iNumDofNode*in->_iNumMeshNodes, cudaMemcpyHostToDevice);
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

// ============================ Print KDia ==============================
void cFemOneGPU::PrintDispl()
{
	int i, j, k;
	using namespace std;

	double *X_h;
	X_h = (double *)malloc(in->_iNumDofNode*in->_iNumMeshNodes*sizeof(double));

	cudaSetDevice(Id);
	cudaMemcpy(X_h, X, sizeof(double)*in->_iNumDofNode*in->_iNumMeshNodes, cudaMemcpyDeviceToHost);

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

				outfile << X_h[3*cont+2] << " ";

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
	rate_h = (double *)malloc(6*in->_iNumMeshElem*sizeof(double));

	cudaSetDevice(Id);
	cudaMemcpy(rate_h, rate, sizeof(double)*6*in->_iNumMeshElem, cudaMemcpyDeviceToHost);

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

				outfile << rate_h[cont+2*in->_iNumMeshElem] << " ";

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

// ============================ Write Average Strain ==============================

void cFemOneGPU::LinkMeshGridMapping(int GridCoord)
{
	int _nx, _ny, _nz, _nsi1, _nsi2, _nsj1, _nsj2, _nov, _nun, _Ti, _Tj, _Tk;
	int PosMesh, PosCell, cont;
	int i, j, k;

	_nx = in->nx; _ny = in->ny; _nz = in->nz; _nsi1 = in->nsi1; _nsi2 = in->nsi2; _nsj1 = in->nsj1; _nsj2 = in->nsj2; _nov = in->nov; _nun = in->nun;

	_Ti = _nsi1 + _nx + _nsi2;
	_Tj = _nsj1 + _ny + _nsj2;
	_Tk = _nov  + _nz + _nun;

	// --------------------------------------------------------------------------------------------------------------------------------------------------------

	// Link local mesh - global mesh position

	LinkMeshMesh_h = (int *)malloc(_nx*_ny*_nz*sizeof(int));

	cont = 0;

	for(k=0; k<_nz; k++) {

		for(j=0; j<_ny; j++) {

			for(i=0; i<_nx; i++) {

				PosMesh = _Ti*_Tj*(_nun+k) + _Ti*(_nsj1+j) + _nsi1 + i;

				// --------------------------------------------------------------------------------

				LinkMeshMesh_h[cont] = PosMesh;  // Begin in 0

				cont++;

				// --------------------------------------------------------------------------------

			}

		}

	}

	// --------------------------------------------------------------------------------------------------------------------------------------------------------

	// Link local mesh - cell position

	LinkMeshCell_h = (int *)malloc(_nx*_ny*_nz*sizeof(int));

	cont = 0;

	for(k=0; k<_nz; k++) {

		for(j=0; j<_ny; j++) {

			for(i=0; i<_nx; i++) {

				switch(GridCoord) { 

				case 0:  // Positive cross product
					PosCell = _nx*_ny*(_nz-1-k) + _nx*(_ny-1-j) + i;  // Namorado
					break; 

				case 1: // Negative cross product
					PosCell = _nx*_ny*(_nz-1-k) + _Ti*j + i;       // Campo B
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
	int _nx, _ny, _nz;
	int pos, add_i, add_ij;

	int color_beg[8], i_beg[8], j_beg[8], k_beg[8];

	_nx = in->nx; _ny = in->ny; _nz = in->nz;

	color_beg[0] = 0;

	color_beg[0] = 0;
	color_beg[1] = 1;
	color_beg[2] = _nx;
	color_beg[3] = _nx+1;

	color_beg[4] = 0+_nx*_ny;
	color_beg[5] = 1+_nx*_ny;
	color_beg[6] = _nx+_nx*_ny;
	color_beg[7] = _nx+1+_nx*_ny;

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

		for(k=k_beg[l]; k<_nz; k+=2) {

			add_i = 0;

			for(j=j_beg[l]; j<_ny; j+=2) {

				for(i=i_beg[l]; i<_nx; i+=2) {

					ElemColorByGPU[l].push_back(pos);  // Begin in 0

					pos += 2;

				}

				add_i += 2*_nx;
				pos = color_beg[l] + add_i + add_ij;

			}

			add_ij += 2*_nx*_ny;
			pos = color_beg[l] + add_ij;

		}

	}

	// Check coloring algorighm:

	int cont=0;

	for(l=0; l<8; l++)
		cont += ElemColorByGPU[l].size();

	// --------------------------------------------------------------------------------------------

	int numelem = _nx*_ny*_nz;  // Total number of elements

	if(numelem != cont)
		printf("\n         Problem with the coloring algorihm!\n");
	else
		printf("\n         Coloring algorihm performed well.\n" );

	// --------------------------------------------------------------------------------------------

	int *LinkMeshMeshColorAux_h;
	LinkMeshMeshColorAux_h = (int *)malloc(_nx*_ny*_nz*sizeof(int));

	cont = 0;

	for(i=0; i<ElemColorByGPU.size(); i++) {

		for(j=0; j<ElemColorByGPU[i].size(); j++) {

			LinkMeshMeshColorAux_h[cont] = LinkMeshMesh_h[ElemColorByGPU[i][j]];  // Begin in 0
			LinkMeshCellColor_h[cont] = LinkMeshCell_h[ElemColorByGPU[i][j]];     // Begin in 0

			cont++;

		}

	}

	// --------------------------------------------------------------------------------------------

	for(i=0; i<in->_iNumMeshElem; i++) {

		for(j=0; j<in->nx*in->ny*in->nz; j++) {

			if(in->connect_h[i]-1 == LinkMeshMeshColorAux_h[j]) LinkMeshMeshColor_h[j] = i;  // Begin in 0

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

	cudaMemcpy(LinkMeshMeshColor, LinkMeshMeshColor_h, sizeof(int)*in->nx*in->ny*in->nz, cudaMemcpyHostToDevice);
	cudaMemcpy(LinkMeshCellColor, LinkMeshCellColor_h, sizeof(int)*in->nx*in->ny*in->nz, cudaMemcpyHostToDevice);

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
	int numelres = in->nx*in->ny*in->nz;
	
	LinkMeshMeshColor_h = (int *)malloc(numelres*sizeof(int));
	LinkMeshCellColor_h = (int *)malloc(numelres*sizeof(int));
	LinkMeshColor_h = (int *)malloc(in->_iNumMeshElem*sizeof(int));

	cudaSetDevice(Id);  // Sets device "Id" as the current device

	cudaMalloc((void **)&dP, sizeof(double)*numelres);
	cudaMalloc((void **)&LinkMeshMeshColor, sizeof(int)*numelres);
	cudaMalloc((void **)&LinkMeshCellColor, sizeof(int)*numelres);
	cudaMalloc((void **)&LinkMeshColor, sizeof(int)*in->_iNumMeshElem);

	//F_h = (double *)malloc(sizeof(double)*in->_iNumDofNode*in->_iNumMeshNodes);

}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============
void cFemOneGPU::EvaluateNodalForce(double *dP_h)
{
	int i, j, k;
	double time;

	cudaSetDevice(Id);

	// Copying the pore pressure from CPU to GPU memory:
	cudaMemcpy(dP, dP_h, sizeof(double)*in->nx*in->ny*in->nz, cudaMemcpyHostToDevice);

	// Cleaning vectors:
	cudaMemset(B, 0, sizeof(double)*in->_iNumDofNode*in->_iNumMeshNodes);
	
	time = clock();

	for(i=0; i<numcolor; i++) {

		dim3 threadsPerBlock(BlockSizeX, BlockSizeX);
		dim3 blocksPerGrid(int(sqrt(double(numelcolornodalforce[i]))/BlockSizeX)+1, int(sqrt(double(numelcolornodalforce[i]))/BlockSizeX)+1);		

		EvaluateNodalForceColor(blocksPerGrid, threadsPerBlock, Id, in->_iNumMeshNodes, in->_iNumMeshElem, in->_iNumDofNode, in->_iNumElasMat, numelcolornodalforce[i], numelcolornodalforceprv[i],
			connect, coord, LinkMeshMeshColor, LinkMeshCellColor, dP, B);

	}

	std::vector<double> F_h(in->_iNumDofNode*in->_iNumMeshNodes);
	cudaMemcpy(&F_h[0], B, sizeof(double)*in->_iNumDofNode*in->_iNumMeshNodes, cudaMemcpyDeviceToHost);

	printf("kkk");
	int aaa = 0;
	for (int i=0; i< F_h.size(); ++i){
		if (F_h[i] != 0)
			aaa = 2;
	}
	aaa=3;

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

	printf("         Time Execution : %0.3f s \n", (clock()-time)/CLOCKS_PER_SEC);

}

//==============================================================================
// Evalauting initial stress state:
void cFemOneGPU::EvaluateInitialStressState(double *StressTotalChr_h)
{

	int i, j, k, l, kaux;
	double StressInitialAux;

	int _nx, _ny, _nz, _nsi1, _nsi2, _nsj1, _nsj2, _nov, _nun, Ti, Tj, Tk;

	int ElemIdx, ElemColorIdx;
	
	double densCurr, densPrev, K0x, K0y;  //  dg = grain density, ds = saturated density,K0x = relation between Sx and Sz, K0y = relation between Sy and Sz
	double dw = 9810., Z;
	int pos;

	double ZcenterCurr, ZcenterPrev, Ztop;

	_nx = in->nx; _ny = in->ny; _nz = in->nz; _nsi1 = in->nsi1; _nsi2 = in->nsi2; _nsj1 = in->nsj1; _nsj2 = in->nsj2; _nov = in->nov; _nun = in->nun;

	Ti = _nsi1 + _nx + _nsi2;
	Tj = _nsj1 + _ny + _nsj2;
	Tk = _nov  + _nz + _nun;

	// ------------------------------------------------------------------

	for(j=0; j<Tj; j++) {                        // Loop elements in x direcion

		for(i=0; i<Ti; i++) {                    // Loop elements in y direcion

			StressInitialAux = 0.;

			for(kaux=0; kaux<Tk; kaux++) {       // Vertical Direction

				k = Tk - 1 - kaux;

				ElemIdx = k*(Ti*Tj) + j*Ti + i;

				ElemColorIdx = LinkMeshColor_h[ElemIdx];

				// Material density

				densCurr = in->Material_Density_h[in->connect_h[ElemColorIdx+1*in->_iNumMeshElem]-1+0*in->_iNumMeshMat];
				K0x      = in->Material_Density_h[in->connect_h[ElemColorIdx+1*in->_iNumMeshElem]-1+1*in->_iNumMeshMat];
				K0y      = in->Material_Density_h[in->connect_h[ElemColorIdx+1*in->_iNumMeshElem]-1+2*in->_iNumMeshMat];

				// --------------------------------------------------------------------------------

				Ztop = 0.;

				for(int j2=7; j2<11; j2++) {  

					Z = in->coord_h[in->connect_h[ElemColorIdx + j2*in->_iNumMeshElem]-1 + 2*in->_iNumMeshNodes];

					Ztop += Z/4;

				}

				// --------------------------------------------------------------------------------

				ZcenterCurr = 0.;

				for(int j2=3; j2<11; j2++) { 

					Z = in->coord_h[in->connect_h[ElemColorIdx + j2*in->_iNumMeshElem]-1 + 2*in->_iNumMeshNodes];

					ZcenterCurr += Z/8;  

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
				StressTotalChr_h[ElemIdx+0*in->_iNumMeshElem] = K0x*StressInitialAux;  // Sx
				StressTotalChr_h[ElemIdx+1*in->_iNumMeshElem] = K0y*StressInitialAux;  // Sy
				StressTotalChr_h[ElemIdx+2*in->_iNumMeshElem] =     StressInitialAux;  // Sz

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

	for(i=0; i<in->_iNumMeshElem; i++) {                    

		// Searching in the coloring element list
		for(j=0; j<in->_iNumMeshElem; j++) { 
			if((i+1) == in->connect_h[j]) {
				LinkMeshColor_h[i] = j;
				break;
			}
		}

	}

	// ------------------------------------------------------------------

	cudaMemcpy(LinkMeshColor, LinkMeshColor_h, sizeof(int)*in->_iNumMeshElem, cudaMemcpyHostToDevice);

	// ------------------------------------------------------------------

	/*using namespace std;
	ofstream outfile;
	std::string fileName2 = "LinkMeshColor_h";
	fileName2 += ".dat";
	outfile.open(fileName2.c_str());

	for(j=0; j<in->_iNumMeshElem; j++) { 
		
		outfile << j+1 << "  " << LinkMeshColor_h[j]+1 << endl;

	}*/

}