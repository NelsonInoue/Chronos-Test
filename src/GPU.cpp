#include <fstream>
#include <iostream>
#include <string>
#include <strstream>
#include <cmath>
#include <iomanip>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>     // abs()
#include "GPU.h"
#include <omp.h>
#include <cuda_runtime.h>

#pragma warning(disable:4244)
#pragma warning(disable:4996)

//----------------------------------------------------
// External Functions for Multi GPUs Implementation						 
//----------------------------------------------------

extern "C" void GPU_AssemblyStiffnessHexahedron(int numno, int numdof, int RowSize, int NumThread, int BlockSizeX, int BlockSizeY, double *Matrix_K_d, 
double *Matrix_K_Aux_d, double *Matrix_M_d, double *MatPropQ1_d, double *MatPropQ2_d, double *MatPropQ3_d, double *MatPropQ4_d, double *MatPropQ5_d, 
double *MatPropQ6_d, double *MatPropQ7_d, double *MatPropQ8_d, double *CoordQ1_d, double *CoordQ2_d, double *CoordQ3_d, double *CoordQ4_d, double *CoordQ5_d, 
double *CoordQ6_d, double *CoordQ7_d, double *CoordQ8_d, int *iiPosition_d);

extern "C" void AssemblyStiffnessGPUTetrahedron(int numdof, int numdofel, int numno, int RowSize,
int maxThreadsPerBlock, int Block33X, int BlockDimY, double *Matrix_K_h, double *Matrix_K_d, double *Matrix_K_Aux_d, int NumThread,
double *abcParam_h, double *abcParam_d, int MaxNodeAround, int MaxElemPerEdge);

extern "C" void SolveLinearSystemEquationGPU(int DeviceId, int numno, int numdof, int NumMaxThreadsBlock, int NumMultiProcessor, int RowSize, int BlockSizeX, int BlockSizeY,
double *Matrix_A_d, int *Vector_I_d, double *Vector_B_d, double *Matrix_M_d, double *Vector_R_d,  
double *Vector_0_D_Global_d, double *Vector_1_D_Global_d, double *Vector_2_D_Global_d, double *Vector_3_D_Global_d,
double *Vector_Q_d, double *Vector_S_d, double *delta_new_d, double *delta_aux_d, double *delta_new_V_d, double *delta_aux_V_d, double *delta_old_V_d, double *delta_new_V_h, 
double *delta_GPU_d, 
int ,  int ,  int ,  int , double *, double *, double *, double *, double *, double *, double *, double *, double *,
double *Vector_0_X_d, double *Vector_1_X_d, double *Vector_2_X_d, double *Vector_3_X_d);

extern "C" void AssemblyStiffnessHexahedronGPU(int DeviceId, int Localnumno, int numdof, int RowSize, int GPUNumThread, int GPUBlockSizeX, int GPUBlockSizeY, 
double *Matrix_K_Aux_d, double *Matrix_M_d,
double *MatProp_Q1_d, double *MatProp_Q2_d, double *MatProp_Q3_d, double *MatProp_Q4_d, double *MatProp_Q5_d, double *MatProp_Q6_d, double *MatProp_Q7_d, double *MatProp_Q8_d, 
double *Coord_Q1_d, double *Coord_Q2_d, double *Coord_Q3_d, double *Coord_Q4_d, double *Coord_Q5_d, double *Coord_Q6_d, double *Coord_Q7_d, double *Coord_Q8_d, int *iiPosition_d);

extern "C" void SolveLinearSystemEquationCPUOpenMP(GPU_Struct *plan, int numdof, int RowSize, 
int GPUmaxThreadsPerBlock_0, int GPUmultiProcessorCount_0, int GPUBlockSizeX_0, int GPUBlockSizeY_0, double *Matrix_0_K_aux_d, int *Vector_0_I_d, double *Vector_0_B_d, 
double *Vector_0_X_d, double *Matrix_0_M_d, double *Vector_0_R_d, double *Vector_0_D_d, double *Vector_0_D_Global_d, double *Vector_0_Q_d, double *Vector_0_S_d, double *delta_0_new_d, double *delta_0_aux_d, 
double *delta_0_new_V_d, double *delta_0_aux_V_d, double *delta_0_old_V_d, double *delta_0_new_V_h, double *delta_0_GPU_d, double *Result_0_d, double *Vector_0_X_Global_d,

int GPUmaxThreadsPerBlock_1, int GPUmultiProcessorCount_1, int GPUBlockSizeX_1, int GPUBlockSizeY_1, double *Matrix_1_K_aux_d, int *Vector_1_I_d, double *Vector_1_B_d,
double *Vector_1_X_d, double *Matrix_1_M_d, double *Vector_1_R_d, double *Vector_1_D_d, double *Vector_1_D_Global_d, double *Vector_1_Q_d, double *Vector_1_S_d, double *delta_1_new_d, double *delta_1_aux_d, 
double *delta_1_new_V_d, double *delta_1_aux_V_d, double *delta_1_old_V_d, double *delta_1_new_V_h, double *delta_1_GPU_d, double *Result_1_d, double *Vector_1_X_Global_d,

int GPUmaxThreadsPerBlock_2, int GPUmultiProcessorCount_2, int GPUBlockSizeX_2, int GPUBlockSizeY_2, double *Matrix_2_K_aux_d, int *Vector_2_I_d, double *Vector_2_B_d,
double *Vector_2_X_d, double *Matrix_2_M_d, double *Vector_2_R_d, double *Vector_2_D_d, double *Vector_2_D_Global_d, double *Vector_2_Q_d, double *Vector_2_S_d, double *delta_2_new_d, double *delta_2_aux_d, 
double *delta_2_new_V_d, double *delta_2_aux_V_d, double *delta_2_old_V_d, double *delta_2_new_V_h, double *delta_2_GPU_d, double *Result_2_d, double *Vector_2_X_Global_d,

int GPUmaxThreadsPerBlock_3, int GPUmultiProcessorCount_3, int GPUBlockSizeX_3, int GPUBlockSizeY_3, double *Matrix_3_K_aux_d, int *Vector_3_I_d, double *Vector_3_B_d,
double *Vector_3_X_d, double *Matrix_3_M_d, double *Vector_3_R_d, double *Vector_3_D_d, double *Vector_3_D_Global_d, double *Vector_3_Q_d, double *Vector_3_S_d, double *delta_3_new_d, double *delta_3_aux_d, 
double *delta_3_new_V_d, double *delta_3_aux_V_d, double *delta_3_old_V_d, double *delta_3_new_V_h, double *delta_3_GPU_d, double *Result_3_d, double *Vector_3_X_Global_d);

extern "C" void  AssemblyStiffnessHexahedronCPUOpenMP(GPU_Struct *plan, int numdof, int RowSize, 
int GPUNumThread_0, int GPUBlockSizeX_0, int GPUBlockSizeY_0, double *Matrix_0_K_Aux_d, double *Matrix_0_M_d,
double *MatProp_0_Q1_d, double *MatProp_0_Q2_d, double *MatProp_0_Q3_d, double *MatProp_0_Q4_d, double *MatProp_0_Q5_d, double *MatProp_0_Q6_d, double *MatProp_0_Q7_d, double *MatProp_0_Q8_d, 
double *Coord_0_Q1_d, double *Coord_0_Q2_d, double *Coord_0_Q3_d, double *Coord_0_Q4_d, double *Coord_0_Q5_d, double *Coord_0_Q6_d, double *Coord_0_Q7_d, double *Coord_0_Q8_d, int *iiPosition_0_d,
		
int GPUNumThread_1, int GPUBlockSizeX_1, int GPUBlockSizeY_1, double *Matrix_1_K_Aux_d, double *Matrix_1_M_d,
double *MatProp_1_Q1_d, double *MatProp_1_Q2_d, double *MatProp_1_Q3_d, double *MatProp_1_Q4_d, double *MatProp_1_Q5_d, double *MatProp_1_Q6_d, double *MatProp_1_Q7_d, double *MatProp_1_Q8_d, 
double *Coord_1_Q1_d, double *Coord_1_Q2_d, double *Coord_1_Q3_d, double *Coord_1_Q4_d, double *Coord_1_Q5_d, double *Coord_1_Q6_d, double *Coord_1_Q7_d, double *Coord_1_Q8_d, int *iiPosition_1_d,
		
int GPUNumThread_2, int GPUBlockSizeX_2, int GPUBlockSizeY_2, double *Matrix_2_K_Aux_d, double *Matrix_2_M_d,
double *MatProp_2_Q1_d, double *MatProp_2_Q2_d, double *MatProp_2_Q3_d, double *MatProp_2_Q4_d, double *MatProp_2_Q5_d, double *MatProp_2_Q6_d, double *MatProp_2_Q7_d, double *MatProp_2_Q8_d, 
double *Coord_2_Q1_d, double *Coord_2_Q2_d, double *Coord_2_Q3_d, double *Coord_2_Q4_d, double *Coord_2_Q5_d, double *Coord_2_Q6_d, double *Coord_2_Q7_d, double *Coord_2_Q8_d, int *iiPosition_2_d,
		
int GPUNumThread_3, int GPUBlockSizeX_3, int GPUBlockSizeY_3, double *Matrix_3_K_Aux_d, double *Matrix_3_M_d,
double *MatProp_3_Q1_d, double *MatProp_3_Q2_d, double *MatProp_3_Q3_d, double *MatProp_3_Q4_d, double *MatProp_3_Q5_d, double *MatProp_3_Q6_d, double *MatProp_3_Q7_d, double *MatProp_3_Q8_d, 
double *Coord_3_Q1_d, double *Coord_3_Q2_d, double *Coord_3_Q3_d, double *Coord_3_Q4_d, double *Coord_3_Q5_d, double *Coord_3_Q6_d, double *Coord_3_Q7_d, double *Coord_3_Q8_d, int *iiPosition_3_d);

extern "C" void  EvaluateStrainStateCPUOpenMP(GPU_Struct *plan, int numdof, int RowSize,
int GPUBlockSizeX_0, int GPUBlockSizeY_0, int *Connect_0_d, double *CoordElem_0_d, double *Vector_0_X_Global_d, double *Strain_0_d,
int GPUBlockSizeX_1, int GPUBlockSizeY_1, int *Connect_1_d, double *CoordElem_1_d, double *Vector_1_X_Global_d, double *Strain_1_d,
int GPUBlockSizeX_2, int GPUBlockSizeY_2, int *Connect_2_d, double *CoordElem_2_d, double *Vector_2_X_Global_d, double *Strain_2_d,
int GPUBlockSizeX_3, int GPUBlockSizeY_3, int *Connect_3_d, double *CoordElem_3_d, double *Vector_3_X_Global_d, double *Strain_3_d);

extern "C" void EvaluateStrainStateGPU(int DeviceId, int Localnumel, int RowSize, int GPUBlockSizeX, int GPUBlockSizeY, 
int *Connect_d, double *CoordElem_d, double *Vector_X_Global_d, double *Strain_d);

extern "C" void EvaluateStressStateGPU(int numel, int BlockDimX, int BlockDimY, double *Material_Data_d, double *Strain_d, double *Stress_d);

extern "C" int GPU_ReadSize(int &numno, int &numel, int &nummat, int &numprop, int &numdof, int &numnoel, int &numstr, int &numbc, int &numnl,
int &imax, int &jmax, int &kmax, char GPU_arqaux[80]);

extern "C" int GPU_ReadInPutFile(int numno, int numel, int nummat, int numprop, int numdof, int numnoel, int numstr, int numbc, int numnl, char GPU_arqaux[80],
double *X_h, double *Y_h, double *Z_h, int *BC_h, double *Material_Param_h, int *Connect_h, int *Material_Id_h);

void GPU_Information(int & , double * );

extern "C" void ReadConcLoad(int numno, double *Vector_B_h, int *BC_h);

void EvaluateGPUMaterialProperty(int imax, int jmax, int kmax, int numno, double *MatPropQ1_h, double *MatPropQ2_h, double *MatPropQ3_h, double *MatPropQ4_h, 
double *MatPropQ5_h, double *MatPropQ6_h, double *MatPropQ7_h, double *MatPropQ8_h, double *Material_Data_h, int *BC_h);

void EvaluateGPUCoordinate(int imax, int jmax, int kmax, double *X_h, double *Y_h, double *Z_h, int numno, double *CoordQ1_h, double *CoordQ2_h, 
double *CoordQ3_h, double *CoordQ4_h, double *CoordQ5_h, double *CoordQ6_h, double *CoordQ7_h, double *CoordQ8_h);

void EvaluateGPURowNumbering(int imax, int jmax, int kmax, int RowSize, int *Vector_I_h);

void EvaluateiiPositionMultiGPU(int numdof, int numno, int , int , int , int, int RowSize, int *Vector_I_h, int *iiPosition_h);

void WriteDisplacement(int numno, double *Vector_X_h);

void WriteStrainState(int numel, double *Strain_h);

void WriteStressState(int numel, double *Stress_h);

void EvaluateCoordStrainState(int numel, int numdof, int numnoel, int *Connect_h,  double *X_h, 
double *Y_h,  double *Z_h,  double *CoordElem_h);

extern "C" void Launch_SUBT_V_V(dim3 BlockSizeVector, dim3 ThreadsPerBlockXY, int numdof, int numno, double *Vector_B_d, double *Vector_R_d);
extern "C" void Launch_CLEAR_V(dim3 BlockSizeVector, dim3 ThreadsPerBlockXY, int numdof, int numno, double *Vector_X_d);
extern "C" void Launch_MULT_DM_V(dim3 , dim3 , int , int , double *, double *, double *);
extern "C" void Launch_MULT_V_V(int , int , int , int , double *, double *, double *);
extern "C" void Launch_SUM_V(dim3 , dim3 , int , int, int , int , double *, double *, double *);
extern "C" void Launch_MULT_SM_V_128(dim3 BlockSizeMatrix, dim3 ThreadsPerBlockXY, int NumMaxThreadsBlock, int numdof, int numno, int RowSize, double *Matrix_A_d, 
									 double *Vector_D_d, int *Vector_I_d, double *Vector_Q_d);
extern "C" void Launch_MULT_SM_V_Text_128(dim3 BlockSizeMatrix, dim3 ThreadsPerBlockXY, int numdof, int numno, int RowSize, double *Matrix_A_d, double *Vector_D_d, int *Vector_I_d, double *Vector_Q_d);
extern "C" void Launch_ADD_V_V_X (dim3 BlockSizeVector, dim3 ThreadsPerBlockXY, int numdof, int numno, double *Vector_X_d, double *Vector_D_d, double *delta_new_V_d, double *delta_aux_V_d, 
								  double *Vector_R_d, double *Vector_Q_d); // =============> {x} = {x} + alfa{d}
								                                         // =============> {r} = {r} - alfa{q}
extern "C" void Launch_UPDATE_DELTA(dim3 BlockSizeVector, dim3 ThreadsPerBlockXY, int numdof, int numno, double *delta_new_V_d, double *delta_old_V_d);
extern "C" void Launch_ADD_V_V_D(dim3 BlockSizeVector, dim3 ThreadsPerBlockXY, int numdof, int numno, double *Vector_S_d, double *Vector_D_d,
								 double *delta_new_V_d, double *delta_old_V_d);
extern "C" void Launch_SUM_GPU_4(dim3 BlockSizeVector, dim3 ThreadsPerBlockXY, int numdof, int numno, double *delta_GPU_d, double *delta_new_V_d);
extern "C" void Launch_Copy_Local_Global(dim3 BlockSizeVector, dim3 ThreadsPerBlockXY, int numdof, int numno, int position, double *Vector_D_d, double *Vector_D_Global_d);
extern "C" void Launch_Copy_for_Vector_4(double *, double *);
extern "C" void Launch_Clean_Global_Vector(dim3 , dim3 , int numdof, int , double *);

extern "C" void Launch_AssemblyStiffness3DQ1(dim3 , dim3 , int , int , double * , double * , double * );
extern "C" void Launch_AssemblyStiffness3DQ2(dim3 , dim3 , int , int , double * , double * , double * );
extern "C" void Launch_AssemblyStiffness3DQ3(dim3 , dim3 , int , int , double * , double * , double * );
extern "C" void Launch_AssemblyStiffness3DQ4(dim3 , dim3 , int , int , double * , double * , double * );
extern "C" void Launch_AssemblyStiffness3DQ5(dim3 , dim3 , int , int , double * , double * , double * );
extern "C" void Launch_AssemblyStiffness3DQ6(dim3 , dim3 , int , int , double * , double * , double * );
extern "C" void Launch_AssemblyStiffness3DQ7(dim3 , dim3 , int , int , double * , double * , double * );
extern "C" void Launch_AssemblyStiffness3DQ8(dim3 , dim3 , int , int , double * , double * , double * );

extern "C" void Launch_TrasnsposeCoalesced(dim3 , dim3 , int , double * , double * );
extern "C" void Launch_EvaluateMatrixM(dim3 , dim3 , int , int , int * , double * , double * );

extern "C" void Launch_EvaluateStrainState(dim3 numBlocksStrain, dim3 threadsPerBlockStrain, int Localnumel, int RowSize, 
double *CoordElem_d, int *Connect_d, double *Vector_X_d, double *Strain_d);

using namespace std;
pcGPU cGPU::GPU = NULL;


//==============================================================================
cGPU::cGPU()
{





}

//========================================================================================================
int cGPU::AnalyzeMultiGPU()
{

	// ========= Coordinate Allocation =========

	int numno, numel, nummat, numprop, numdof, numnoel, numstr, numbc, numnl, imax, jmax, kmax, numdofel;

	// numno            Number of nodes of the mesh          
	// numel            Number of elements on mesh
	// nummat           Number of materials on mesh
	// numprop          Number of properties parameter for material
	// numdof           Number of freedom of degree (DOF) per node
	// numnoel          Number of nodes per element
	// numstr           Number of stress component
	// numbc            Number of boundary conditions
	// numnl            Number of nodes with nodal load
	// imax             Number of nodes in x direction
	// jmax             Number of nodes in y direction
	// kmax             Number of nodes in z direction

	GPU_Struct plan[4];
	
	int i, j, BlockSizeX, BlockSizeY;
	int NumThread, RowSize, NumMultProc;
	double time, SumNo, SumEl;
	int NumTerms, BlockDimX, BlockDimY;
	char GPU_arqaux[80];
	int CPUTheadId;  // Cpu thread Id
	int NumRowLocal, NumRowGlobal, NumTermLocal;

	size_t free_byte;
	size_t total_byte;
	double FreeMem[4], UsedMem[4], TotalMem[4];

	// **************  Declaring GPU Information:  **************

	double TotalFreeGlobalMem, FreeGlobalMem, TotalGlobalMem, GlobalMem;
	int multiProcessorCount, maxThreadsPerBlock;
	int deviceCount, idev;

	double *GPUData;

	int gpuBaseTerm, gpuBaseRow, gpuBaseMatProp, gpuBaseCoord, gpuBaseiiPos, gpuBaseElem;
	int numno_0, numno_1, numno_2, numno_3;

	int GPUNumThread_0, GPUmultiProcessorCount_0, GPUmaxThreadsPerBlock_0, GPUBlockSizeX_0, GPUBlockSizeY_0;
	int GPUNumThread_1, GPUmultiProcessorCount_1, GPUmaxThreadsPerBlock_1, GPUBlockSizeX_1, GPUBlockSizeY_1;
	int GPUNumThread_2, GPUmultiProcessorCount_2, GPUmaxThreadsPerBlock_2, GPUBlockSizeX_2, GPUBlockSizeY_2;
	int GPUNumThread_3, GPUmultiProcessorCount_3, GPUmaxThreadsPerBlock_3, GPUBlockSizeX_3, GPUBlockSizeY_3;

	// **************  Declaring Matrix Stiffness Variables:  **************

	int   *Vector_I_h, *BC_h, *Connect_h, *Material_Id_h;
	double *Vector_B_h, *Vector_X_h;
	double *Material_Data_h, *Material_Param_h, *CoordElem_h, *Strain_h, *Stress_h;
	
	double dE, dNu;
	double *X_h, *Y_h, *Z_h; 

	// Host Allocation:
	double *MatPropQ1_h, *MatPropQ2_h, *MatPropQ3_h, *MatPropQ4_h, *MatPropQ5_h, *MatPropQ6_h, *MatPropQ7_h, *MatPropQ8_h;
	double *CoordQ1_h, *CoordQ2_h, *CoordQ3_h, *CoordQ4_h, *CoordQ5_h, *CoordQ6_h, *CoordQ7_h, *CoordQ8_h;
	int   *iiPosition_h;  // Evaluating M matrix = 1/K:

	// Device Allocation:
	int   *iiPosition_d;  // Evaluating M matrix = 1/K:

	// **************  Declaring Variables for Stiffness Matrix:  **************

	// Device Allocation:
	// Device 0:
	double *MatProp_0_Q1_d, *MatProp_0_Q2_d, *MatProp_0_Q3_d, *MatProp_0_Q4_d, *MatProp_0_Q5_d, *MatProp_0_Q6_d, *MatProp_0_Q7_d, *MatProp_0_Q8_d;
	double *Coord_0_Q1_d, *Coord_0_Q2_d, *Coord_0_Q3_d, *Coord_0_Q4_d, *Coord_0_Q5_d, *Coord_0_Q6_d, *Coord_0_Q7_d, *Coord_0_Q8_d;
	int   *iiPosition_0_d;  // Evaluating M matrix = 1/K:
	// Device 1:
	double *MatProp_1_Q1_d, *MatProp_1_Q2_d, *MatProp_1_Q3_d, *MatProp_1_Q4_d, *MatProp_1_Q5_d, *MatProp_1_Q6_d, *MatProp_1_Q7_d, *MatProp_1_Q8_d;
	double *Coord_1_Q1_d, *Coord_1_Q2_d, *Coord_1_Q3_d, *Coord_1_Q4_d, *Coord_1_Q5_d, *Coord_1_Q6_d, *Coord_1_Q7_d, *Coord_1_Q8_d;
	int   *iiPosition_1_d;  // Evaluating M matrix = 1/K:
	// Device 2:
	double *MatProp_2_Q1_d, *MatProp_2_Q2_d, *MatProp_2_Q3_d, *MatProp_2_Q4_d, *MatProp_2_Q5_d, *MatProp_2_Q6_d, *MatProp_2_Q7_d, *MatProp_2_Q8_d;
	double *Coord_2_Q1_d, *Coord_2_Q2_d, *Coord_2_Q3_d, *Coord_2_Q4_d, *Coord_2_Q5_d, *Coord_2_Q6_d, *Coord_2_Q7_d, *Coord_2_Q8_d;
	int   *iiPosition_2_d;  // Evaluating M matrix = 1/K:
	// Device 3:
	double *MatProp_3_Q1_d, *MatProp_3_Q2_d, *MatProp_3_Q3_d, *MatProp_3_Q4_d, *MatProp_3_Q5_d, *MatProp_3_Q6_d, *MatProp_3_Q7_d, *MatProp_3_Q8_d;
	double *Coord_3_Q1_d, *Coord_3_Q2_d, *Coord_3_Q3_d, *Coord_3_Q4_d, *Coord_3_Q5_d, *Coord_3_Q6_d, *Coord_3_Q7_d, *Coord_3_Q8_d;
	int   *iiPosition_3_d;  // Evaluating M matrix = 1/K:

	// **************  Declaring CGA Variables:  **************

	// Host Allocation:
	double *delta_new_h, *delta_aux_h, *delta_new_V_h, *delta_aux_V_h;
	double *Vector_R_h, *Vector_D_h, *Vector_Q_h, *Vector_S_h;
	double *delta_0_new_V_h, *delta_1_new_V_h, *delta_2_new_V_h, *delta_3_new_V_h;

	// Host Allocation:
	double *Vector_R_d, *Vector_D_d, *Vector_S_d;
	double *delta_aux_d, *delta_new_V_d, *delta_aux_V_d, *delta_old_V_d;

	// Device Allocation:
	// Device 0:
	int   *Vector_0_I_d, *Connect_0_d;
	double *Matrix_0_K_Aux_d, *Vector_0_X_d, *Matrix_0_M_d, *Vector_0_B_d, *Result_0_d;
	double *Vector_0_R_d, *Vector_0_D_d, *Vector_0_D_Global_d, *Vector_0_Q_d, *Vector_0_S_d;
	double *delta_0_new_d, *delta_0_aux_d, *delta_0_new_V_d, *delta_0_aux_V_d, *delta_0_old_V_d, *delta_0_GPU_d;
	double *CoordElem_0_d, *Strain_0_d, *Vector_0_X_Global_d;
	// Device 1:
	int   *Vector_1_I_d, *Connect_1_d;
	double *Matrix_1_K_Aux_d, *Vector_1_X_d, *Matrix_1_M_d, *Vector_1_B_d, *Result_1_d;
	double *Vector_1_R_d, *Vector_1_D_d, *Vector_1_D_Global_d, *Vector_1_Q_d, *Vector_1_S_d;
	double *delta_1_new_d, *delta_1_aux_d, *delta_1_new_V_d, *delta_1_aux_V_d, *delta_1_old_V_d, *delta_1_GPU_d;
	double *CoordElem_1_d, *Strain_1_d, *Vector_1_X_Global_d;
	// Device 2:
	int   *Vector_2_I_d, *Connect_2_d;
	double *Matrix_2_K_Aux_d, *Vector_2_X_d, *Matrix_2_M_d, *Vector_2_B_d, *Result_2_d;
	double *Vector_2_R_d, *Vector_2_D_d, *Vector_2_D_Global_d, *Vector_2_Q_d, *Vector_2_S_d;
	double *delta_2_new_d, *delta_2_aux_d, *delta_2_new_V_d, *delta_2_aux_V_d, *delta_2_old_V_d, *delta_2_GPU_d;
	double *CoordElem_2_d, *Strain_2_d, *Vector_2_X_Global_d;
	// Device 3:
	int   *Vector_3_I_d, *Connect_3_d;
	double *Matrix_3_K_Aux_d, *Vector_3_X_d, *Matrix_3_M_d, *Vector_3_B_d, *Result_3_d;
	double *Vector_3_R_d, *Vector_3_D_d, *Vector_3_D_Global_d, *Vector_3_Q_d, *Vector_3_S_d;
	double *delta_3_new_d, *delta_3_aux_d, *delta_3_new_V_d, *delta_3_aux_V_d, *delta_3_old_V_d, *delta_3_GPU_d;
	double *CoordElem_3_d, *Strain_3_d, *Vector_3_X_Global_d;

	// ========= Reset Device =========

	// Explicitly destroys and cleans up all resources associated with the current device in the current process.

	// Reset device 0:
	cudaSetDevice(0); cudaDeviceReset();

	// Reset device 1:
	cudaSetDevice(1); cudaDeviceReset(); 

	// Reset device 2:
	cudaSetDevice(2); cudaDeviceReset();

	// Reset device 3:
	cudaSetDevice(3); cudaDeviceReset();

	// ========= initialization =========

	numno = 0; numel = 0; nummat = 0; numdof = 0; numnoel = 0; numstr = 0; numbc = 0; numnl = 0;
	imax = 0; jmax = 0; kmax = 0;

	// Defining RowSize
	RowSize = 128;     // 3D Problem = Total number of terms per line is 81 (hexahedrol element) 128 to do a parallel reduction

	// ========= Reading vector size =========

	if(GPU_ReadSize(numno, numel, nummat, numprop, numdof, numnoel, numstr, numbc, numnl, imax, jmax, kmax, GPU_arqaux) == 0) {
		printf("Invalid neutral file or label END doesn't exist" );
	}

	numdofel = numdof*numnoel;  // Number of DOF per element

	// ========= Getting GPU information =========

	deviceCount = 0;
	if(cudaGetDeviceCount(&deviceCount) != cudaSuccess) {
		printf("         cudaGetDeviceCount FAILED CUDA Driver and Runtime version may be mismatched.\n");
		printf("\n         FAILED\n");
	}

	GPUData = (double *)malloc(5*deviceCount*sizeof(double));

	GPU_Information(deviceCount, GPUData);
   
	// GPUData[dev*5+0] = Total number of multiprocessor
	// GPUData[dev*5+1] = Total amount of shared memory per block
	// GPUData[dev*5+2] = Total number of registers available per block
	// GPUData[dev*5+3] = Warp size
	// GPUData[dev*5+4] = Maximum number of threads per block

	// ========= Allocating Vectors =========         

	// Allocating memory space for host global node coordinate
	X_h  = (double *)malloc(numno*sizeof(double)); 
	Y_h  = (double *)malloc(numno*sizeof(double)); 
	Z_h  = (double *)malloc(numno*sizeof(double)); 

	// Allocating memory space for host boundary condition vector
	BC_h = (int *)malloc(numno*(sizeof(int))); for(i=0; i<numno; i++) BC_h[i] = 0;

	// Allocating memory space for host material parameters vector
	Material_Param_h = (double *)malloc(nummat*numprop*(sizeof(int))); 

	// Allocating memory space for host element conectivity vector

	Material_Id_h = (int *)malloc(numel*(sizeof(int)));  
	Connect_h  = (int *)malloc(numel*numnoel*(sizeof(int)));  // numnoel = Number of nodes per element 
	
	// ========= Reading input file =========

	printf("\n");
	printf("         Reading input data file \n");
	printf("         ========================================= ");

	time = clock();

	GPU_ReadInPutFile(numno, numel, nummat, numprop, numdof, numnoel, numstr, numbc, numnl, GPU_arqaux, X_h, Y_h, Z_h, BC_h, Material_Param_h, Connect_h, Material_Id_h);

	time = clock()-time;
	printf("\n");
	printf("         Input File Reading Time: %0.3f s \n", time/CLOCKS_PER_SEC);

	// ========= Allocating material property per element =========

	//------------------------------ Evaluating data preparing time
	time = clock();
	//------------------------------

	Material_Data_h = (double *)malloc(numprop*numel*sizeof(double));            // Allocating menoty space for host

	for(i=0; i<numel; i++) {

		dE  = Material_Param_h[ Material_Id_h[i]-1         ];
		dNu = Material_Param_h[ Material_Id_h[i]-1 + nummat];

		Material_Data_h[i  ]     = dE;
		Material_Data_h[i+numel] = dNu;

	}

	//cudaMemcpy(Material_Data_d, Material_Data_h, sizeof(double)*numprop*numel, cudaMemcpyHostToDevice);  // To copy from host memory to device memory;

	// ------------------------------------------------------------------
	// Allocating menory space for host

	Vector_I_h   = (int *)malloc(numdof*numno*RowSize*(sizeof(int)));             
	iiPosition_h = (int *)malloc(numdof*numno*(sizeof(int)));

	Vector_B_h  = (double *)malloc(numdof*numno*sizeof(double));           
	Vector_X_h  = (double *)malloc(numdof*numno*sizeof(double));          

	for(i=0; i<numdof*numno*RowSize; i++)
		Vector_I_h[i]=0;

	for(i=0; i<numdof*numno; i++)
		Vector_B_h[i]=0.;

	// ------------------------------------------------------------------

	// numdof   =   Number of freedom of degree (DOF) per node
	// numnoel = Number of nodes per element

	// Allocating vector to storage coord x, y, and z for each element
	CoordElem_h  = (double *)malloc(numel*numdof*numnoel*sizeof(double));  

	for(i=0; i<numel*numdof*numnoel; i++) CoordElem_h[i] = 0.;

	// ------------------------------------------------------------------

	// Strain state vector memory allocation
	Strain_h  = (double *)malloc(numel*numstr*8*sizeof(double));      // numstr = 6 = number of components of strain
	
	// Stress state vector memory allocation
	Stress_h  = (double *)malloc(numel*numstr*8*sizeof(double));      // numstr = 6 = number of components of strain

	// =================================================================

	// Allocating Host memory:  
	
	MatPropQ1_h = (double *)malloc(numno*numprop*sizeof(double));  // 2 = _dE and _dNu
	MatPropQ2_h = (double *)malloc(numno*numprop*sizeof(double));
	MatPropQ3_h = (double *)malloc(numno*numprop*sizeof(double));
	MatPropQ4_h = (double *)malloc(numno*numprop*sizeof(double));
	MatPropQ5_h = (double *)malloc(numno*numprop*sizeof(double));
	MatPropQ6_h = (double *)malloc(numno*numprop*sizeof(double));
	MatPropQ7_h = (double *)malloc(numno*numprop*sizeof(double));
	MatPropQ8_h = (double *)malloc(numno*numprop*sizeof(double));
	
	for(i=0; i<numno*2; i++) {  // Initialization
		MatPropQ1_h[i] = 0.; MatPropQ2_h[i] = 0.; MatPropQ3_h[i] = 0.; MatPropQ4_h[i] = 0.;
		MatPropQ5_h[i] = 0.; MatPropQ6_h[i] = 0.; MatPropQ7_h[i] = 0.; MatPropQ8_h[i] = 0.;
	}

	// =================================================================

	// NumThread = // It is associated with number of nodes!!!!!!!!!!!!!!!! 
	
	CoordQ1_h = (double *)malloc(numno*numdof*numnoel*sizeof(double));  // numdof*numnoel = Number of DOF per element
	CoordQ2_h = (double *)malloc(numno*numdof*numnoel*sizeof(double));
	CoordQ3_h = (double *)malloc(numno*numdof*numnoel*sizeof(double));
	CoordQ4_h = (double *)malloc(numno*numdof*numnoel*sizeof(double));
	CoordQ5_h = (double *)malloc(numno*numdof*numnoel*sizeof(double));
	CoordQ6_h = (double *)malloc(numno*numdof*numnoel*sizeof(double));
	CoordQ7_h = (double *)malloc(numno*numdof*numnoel*sizeof(double));
	CoordQ8_h = (double *)malloc(numno*numdof*numnoel*sizeof(double));


	for(i=0; i<numno*numdof*numnoel; i++) {
		CoordQ1_h[i] = 0.; CoordQ2_h[i] = 0.; CoordQ3_h[i] = 0.; CoordQ4_h[i] = 0.;
		CoordQ5_h[i] = 0.; CoordQ6_h[i] = 0.; CoordQ7_h[i] = 0.; CoordQ8_h[i] = 0.;
	}           

	// =================================================================

	// Dividing the number of nodes among GPUs: 

	SumNo = 0;
	SumEl = 0;

	//Subdividing input data across GPUs;
    //Get the number of lines of the verctor for each GPU:
	for(i=0; i<deviceCount; i++) {
		
		if(i == deviceCount-1) {
			plan[i].Localnumno  = numno-SumNo;
			plan[i].Localnumel  = numel-SumEl;

		}
		else {
			plan[i].Localnumno  = int(numno/deviceCount);  // number of nodes for each GPU
			SumNo += plan[i].Localnumno;

			plan[i].Localnumel  = int(numel/deviceCount);  // number of elements for each GPU
			SumEl += plan[i].Localnumel;
		}

	}

	// =================================================================

	// Dividing the number of elements among GPUs: 

	// =================================================================

	// Evaluate appropriate Material Properties with BCs:

	EvaluateGPUMaterialProperty(imax, jmax, kmax, numno, MatPropQ1_h, MatPropQ2_h, MatPropQ3_h, MatPropQ4_h, MatPropQ5_h, MatPropQ6_h, MatPropQ7_h, MatPropQ8_h,
	Material_Data_h, BC_h);

	// Evaluate appropriate GPU coordinate to assembly Stiffness Matrix:	
		
	EvaluateGPUCoordinate(imax, jmax, kmax, X_h, Y_h, Z_h, numno, CoordQ1_h, CoordQ2_h, CoordQ3_h, CoordQ4_h, CoordQ5_h, CoordQ6_h, CoordQ7_h, CoordQ8_h);

	// ========= Evaluate GPU Row Numbering  =========
		
	// Call function to calculate global numbering in stiffness matrix 
	EvaluateGPURowNumbering(imax, jmax, kmax, RowSize, Vector_I_h);

	// ========= Evaluate GPU diagonal position  =========
		
	EvaluateiiPositionMultiGPU(numdof, numno, plan[0].Localnumno, plan[1].Localnumno, plan[2].Localnumno, plan[3].Localnumno,
	RowSize, Vector_I_h, iiPosition_h);

	// =========  Evaluate coordinate for each element  =========
	EvaluateCoordStrainState(numel, numdof, numnoel, Connect_h, X_h, Y_h, Z_h, CoordElem_h);
	
	// =================================================================

	//Get the number of terms of the matrix for each GPU:
		
	//Assign data ranges to GPUs
    gpuBaseMatProp = 0;
	gpuBaseCoord   = 0;
	gpuBaseiiPos   = 0;

    for(i=0; i<deviceCount; i++){
        
		plan[i].MatPropQ1_h_Pt = MatPropQ1_h + gpuBaseMatProp;
		plan[i].MatPropQ2_h_Pt = MatPropQ2_h + gpuBaseMatProp;
		plan[i].MatPropQ3_h_Pt = MatPropQ3_h + gpuBaseMatProp;
		plan[i].MatPropQ4_h_Pt = MatPropQ4_h + gpuBaseMatProp;
		plan[i].MatPropQ5_h_Pt = MatPropQ5_h + gpuBaseMatProp;
		plan[i].MatPropQ6_h_Pt = MatPropQ6_h + gpuBaseMatProp;
		plan[i].MatPropQ7_h_Pt = MatPropQ7_h + gpuBaseMatProp;
		plan[i].MatPropQ8_h_Pt = MatPropQ8_h + gpuBaseMatProp;

		plan[i].CoordQ1_h_Pt = CoordQ1_h + gpuBaseCoord;
		plan[i].CoordQ2_h_Pt = CoordQ2_h + gpuBaseCoord;
		plan[i].CoordQ3_h_Pt = CoordQ3_h + gpuBaseCoord;
		plan[i].CoordQ4_h_Pt = CoordQ4_h + gpuBaseCoord;
		plan[i].CoordQ5_h_Pt = CoordQ5_h + gpuBaseCoord;
		plan[i].CoordQ6_h_Pt = CoordQ6_h + gpuBaseCoord;
		plan[i].CoordQ7_h_Pt = CoordQ7_h + gpuBaseCoord;
		plan[i].CoordQ8_h_Pt = CoordQ8_h + gpuBaseCoord;

		plan[i].iiPosition_h_Pt = iiPosition_h + gpuBaseiiPos;

		gpuBaseMatProp += plan[i].Localnumno;
		gpuBaseCoord   += plan[i].Localnumno;
		gpuBaseiiPos   += plan[i].Localnumno*numdof;

	}

	// =================================================================

	// Allocating Device memory:

	// Device 0:
	cudaSetDevice(0);              // Sets device 0 as the current device  

	// Allocating Material Properties data: 
	cudaMalloc((void **) &MatProp_0_Q1_d, plan[0].Localnumno*numprop*sizeof(double));
	cudaMalloc((void **) &MatProp_0_Q2_d, plan[0].Localnumno*numprop*sizeof(double));
	cudaMalloc((void **) &MatProp_0_Q3_d, plan[0].Localnumno*numprop*sizeof(double));
	cudaMalloc((void **) &MatProp_0_Q4_d, plan[0].Localnumno*numprop*sizeof(double));
	cudaMalloc((void **) &MatProp_0_Q5_d, plan[0].Localnumno*numprop*sizeof(double));
	cudaMalloc((void **) &MatProp_0_Q6_d, plan[0].Localnumno*numprop*sizeof(double));
	cudaMalloc((void **) &MatProp_0_Q7_d, plan[0].Localnumno*numprop*sizeof(double));
	cudaMalloc((void **) &MatProp_0_Q8_d, plan[0].Localnumno*numprop*sizeof(double));

	// Allocating Coodinate data:
	cudaMalloc((void **) &Coord_0_Q1_d, plan[0].Localnumno*numdof*numnoel*sizeof(double));
	cudaMalloc((void **) &Coord_0_Q2_d, plan[0].Localnumno*numdof*numnoel*sizeof(double));
	cudaMalloc((void **) &Coord_0_Q3_d, plan[0].Localnumno*numdof*numnoel*sizeof(double));
	cudaMalloc((void **) &Coord_0_Q4_d, plan[0].Localnumno*numdof*numnoel*sizeof(double));
	cudaMalloc((void **) &Coord_0_Q5_d, plan[0].Localnumno*numdof*numnoel*sizeof(double));
	cudaMalloc((void **) &Coord_0_Q6_d, plan[0].Localnumno*numdof*numnoel*sizeof(double));
	cudaMalloc((void **) &Coord_0_Q7_d, plan[0].Localnumno*numdof*numnoel*sizeof(double));
	cudaMalloc((void **) &Coord_0_Q8_d, plan[0].Localnumno*numdof*numnoel*sizeof(double));

	// Allocating ii position:
	cudaMalloc((void **) &iiPosition_0_d, numdof*plan[0].Localnumno*sizeof(int));          

	for(i=0; i<numprop; i++) {

		// Copying Material Properties data from host to device:
		cudaMemcpy(MatProp_0_Q1_d + i*plan[0].Localnumno, plan[0].MatPropQ1_h_Pt + i*numno, plan[0].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  // To copy from host memory to device memory;
		cudaMemcpy(MatProp_0_Q2_d + i*plan[0].Localnumno, plan[0].MatPropQ2_h_Pt + i*numno, plan[0].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  
		cudaMemcpy(MatProp_0_Q3_d + i*plan[0].Localnumno, plan[0].MatPropQ3_h_Pt + i*numno, plan[0].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  
		cudaMemcpy(MatProp_0_Q4_d + i*plan[0].Localnumno, plan[0].MatPropQ4_h_Pt + i*numno, plan[0].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  
		cudaMemcpy(MatProp_0_Q5_d + i*plan[0].Localnumno, plan[0].MatPropQ5_h_Pt + i*numno, plan[0].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  
		cudaMemcpy(MatProp_0_Q6_d + i*plan[0].Localnumno, plan[0].MatPropQ6_h_Pt + i*numno, plan[0].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  
		cudaMemcpy(MatProp_0_Q7_d + i*plan[0].Localnumno, plan[0].MatPropQ7_h_Pt + i*numno, plan[0].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  
		cudaMemcpy(MatProp_0_Q8_d + i*plan[0].Localnumno, plan[0].MatPropQ8_h_Pt + i*numno, plan[0].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  

	}

	for(i=0; i<numdof*numnoel; i++) {

		// Copying Coordinate data from host to device:
		cudaMemcpy(Coord_0_Q1_d + i*plan[0].Localnumno, plan[0].CoordQ1_h_Pt + i*numno, plan[0].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  // To copy from host memory to device memory;
		cudaMemcpy(Coord_0_Q2_d + i*plan[0].Localnumno, plan[0].CoordQ2_h_Pt + i*numno, plan[0].Localnumno*sizeof(double), cudaMemcpyHostToDevice); 
		cudaMemcpy(Coord_0_Q3_d + i*plan[0].Localnumno, plan[0].CoordQ3_h_Pt + i*numno, plan[0].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  
		cudaMemcpy(Coord_0_Q4_d + i*plan[0].Localnumno, plan[0].CoordQ4_h_Pt + i*numno, plan[0].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  
		cudaMemcpy(Coord_0_Q5_d + i*plan[0].Localnumno, plan[0].CoordQ5_h_Pt + i*numno, plan[0].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  
		cudaMemcpy(Coord_0_Q6_d + i*plan[0].Localnumno, plan[0].CoordQ6_h_Pt + i*numno, plan[0].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  
		cudaMemcpy(Coord_0_Q7_d + i*plan[0].Localnumno, plan[0].CoordQ7_h_Pt + i*numno, plan[0].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  
		cudaMemcpy(Coord_0_Q8_d + i*plan[0].Localnumno, plan[0].CoordQ8_h_Pt + i*numno, plan[0].Localnumno*sizeof(double), cudaMemcpyHostToDevice);

	}

	// Copying ii position data from host to device:
	cudaMemcpy(iiPosition_0_d, plan[0].iiPosition_h_Pt, numdof*plan[0].Localnumno*sizeof(int), cudaMemcpyHostToDevice);    // To copy from host memory to device memory;

	// -----------------------------------------------------------------

	// Device 1:
	cudaSetDevice(1);              // Sets device 1 as the current device  

	// Allocating Material Properties data: 
	cudaMalloc((void **) &MatProp_1_Q1_d, plan[1].Localnumno*numprop*sizeof(double));
	cudaMalloc((void **) &MatProp_1_Q2_d, plan[1].Localnumno*numprop*sizeof(double));
	cudaMalloc((void **) &MatProp_1_Q3_d, plan[1].Localnumno*numprop*sizeof(double));
	cudaMalloc((void **) &MatProp_1_Q4_d, plan[1].Localnumno*numprop*sizeof(double));
	cudaMalloc((void **) &MatProp_1_Q5_d, plan[1].Localnumno*numprop*sizeof(double));
	cudaMalloc((void **) &MatProp_1_Q6_d, plan[1].Localnumno*numprop*sizeof(double));
	cudaMalloc((void **) &MatProp_1_Q7_d, plan[1].Localnumno*numprop*sizeof(double));
	cudaMalloc((void **) &MatProp_1_Q8_d, plan[1].Localnumno*numprop*sizeof(double));

	// Allocating Coordinate data:
	cudaMalloc((void **) &Coord_1_Q1_d, plan[1].Localnumno*numdof*numnoel*sizeof(double));
	cudaMalloc((void **) &Coord_1_Q2_d, plan[1].Localnumno*numdof*numnoel*sizeof(double));
	cudaMalloc((void **) &Coord_1_Q3_d, plan[1].Localnumno*numdof*numnoel*sizeof(double));
	cudaMalloc((void **) &Coord_1_Q4_d, plan[1].Localnumno*numdof*numnoel*sizeof(double));
	cudaMalloc((void **) &Coord_1_Q5_d, plan[1].Localnumno*numdof*numnoel*sizeof(double));
	cudaMalloc((void **) &Coord_1_Q6_d, plan[1].Localnumno*numdof*numnoel*sizeof(double));
	cudaMalloc((void **) &Coord_1_Q7_d, plan[1].Localnumno*numdof*numnoel*sizeof(double));
	cudaMalloc((void **) &Coord_1_Q8_d, plan[1].Localnumno*numdof*numnoel*sizeof(double));

	// Allocating ii position:
	cudaMalloc((void **) &iiPosition_1_d, numdof*plan[1].Localnumno*sizeof(int));          

	for(i=0; i<numprop; i++) {

		// Copying Material Properties data from host to device:
		cudaMemcpy(MatProp_1_Q1_d + i*plan[1].Localnumno, plan[1].MatPropQ1_h_Pt + i*numno, plan[1].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  // To copy from host memory to device memory;
		cudaMemcpy(MatProp_1_Q2_d + i*plan[1].Localnumno, plan[1].MatPropQ2_h_Pt + i*numno, plan[1].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  
		cudaMemcpy(MatProp_1_Q3_d + i*plan[1].Localnumno, plan[1].MatPropQ3_h_Pt + i*numno, plan[1].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  
		cudaMemcpy(MatProp_1_Q4_d + i*plan[1].Localnumno, plan[1].MatPropQ4_h_Pt + i*numno, plan[1].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  
		cudaMemcpy(MatProp_1_Q5_d + i*plan[1].Localnumno, plan[1].MatPropQ5_h_Pt + i*numno, plan[1].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  
		cudaMemcpy(MatProp_1_Q6_d + i*plan[1].Localnumno, plan[1].MatPropQ6_h_Pt + i*numno, plan[1].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  
		cudaMemcpy(MatProp_1_Q7_d + i*plan[1].Localnumno, plan[1].MatPropQ7_h_Pt + i*numno, plan[1].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  
		cudaMemcpy(MatProp_1_Q8_d + i*plan[1].Localnumno, plan[1].MatPropQ8_h_Pt + i*numno, plan[1].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  

	}

	for(i=0; i<numdof*numnoel; i++) {

		// Copying Coordinate data from host to device:
		cudaMemcpy(Coord_1_Q1_d + i*plan[1].Localnumno, plan[1].CoordQ1_h_Pt + i*numno, plan[1].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  // To copy from host memory to device memory;
		cudaMemcpy(Coord_1_Q2_d + i*plan[1].Localnumno, plan[1].CoordQ2_h_Pt + i*numno, plan[1].Localnumno*sizeof(double), cudaMemcpyHostToDevice); 
		cudaMemcpy(Coord_1_Q3_d + i*plan[1].Localnumno, plan[1].CoordQ3_h_Pt + i*numno, plan[1].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  
		cudaMemcpy(Coord_1_Q4_d + i*plan[1].Localnumno, plan[1].CoordQ4_h_Pt + i*numno, plan[1].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  
		cudaMemcpy(Coord_1_Q5_d + i*plan[1].Localnumno, plan[1].CoordQ5_h_Pt + i*numno, plan[1].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  
		cudaMemcpy(Coord_1_Q6_d + i*plan[1].Localnumno, plan[1].CoordQ6_h_Pt + i*numno, plan[1].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  
		cudaMemcpy(Coord_1_Q7_d + i*plan[1].Localnumno, plan[1].CoordQ7_h_Pt + i*numno, plan[1].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  
		cudaMemcpy(Coord_1_Q8_d + i*plan[1].Localnumno, plan[1].CoordQ8_h_Pt + i*numno, plan[1].Localnumno*sizeof(double), cudaMemcpyHostToDevice);

	}

	// Copying ii position data from host to device:
	cudaMemcpy(iiPosition_1_d, plan[1].iiPosition_h_Pt, numdof*plan[1].Localnumno*sizeof(int), cudaMemcpyHostToDevice);    // To copy from host memory to device memory;

	// -----------------------------------------------------------------

	// Device 2:
	cudaSetDevice(2);              // Sets device 2 as the current device  

	// Allocating Material Properties data: 
	cudaMalloc((void **) &MatProp_2_Q1_d, plan[2].Localnumno*numprop*sizeof(double));
	cudaMalloc((void **) &MatProp_2_Q2_d, plan[2].Localnumno*numprop*sizeof(double));
	cudaMalloc((void **) &MatProp_2_Q3_d, plan[2].Localnumno*numprop*sizeof(double));
	cudaMalloc((void **) &MatProp_2_Q4_d, plan[2].Localnumno*numprop*sizeof(double));
	cudaMalloc((void **) &MatProp_2_Q5_d, plan[2].Localnumno*numprop*sizeof(double));
	cudaMalloc((void **) &MatProp_2_Q6_d, plan[2].Localnumno*numprop*sizeof(double));
	cudaMalloc((void **) &MatProp_2_Q7_d, plan[2].Localnumno*numprop*sizeof(double));
	cudaMalloc((void **) &MatProp_2_Q8_d, plan[2].Localnumno*numprop*sizeof(double));

	// Allocating Coodinate data:
	cudaMalloc((void **) &Coord_2_Q1_d, plan[2].Localnumno*numdof*numnoel*sizeof(double));
	cudaMalloc((void **) &Coord_2_Q2_d, plan[2].Localnumno*numdof*numnoel*sizeof(double));
	cudaMalloc((void **) &Coord_2_Q3_d, plan[2].Localnumno*numdof*numnoel*sizeof(double));
	cudaMalloc((void **) &Coord_2_Q4_d, plan[2].Localnumno*numdof*numnoel*sizeof(double));
	cudaMalloc((void **) &Coord_2_Q5_d, plan[2].Localnumno*numdof*numnoel*sizeof(double));
	cudaMalloc((void **) &Coord_2_Q6_d, plan[2].Localnumno*numdof*numnoel*sizeof(double));
	cudaMalloc((void **) &Coord_2_Q7_d, plan[2].Localnumno*numdof*numnoel*sizeof(double));
	cudaMalloc((void **) &Coord_2_Q8_d, plan[2].Localnumno*numdof*numnoel*sizeof(double));

	// Allocating ii position:
	cudaMalloc((void **) &iiPosition_2_d, numdof*plan[2].Localnumno*sizeof(int));         

	// Copying Material Properties data from host to device:
	for(i=0; i<numprop; i++) {

		// Copying Material Properties data from host to device:
		cudaMemcpy(MatProp_2_Q1_d + i*plan[2].Localnumno, plan[2].MatPropQ1_h_Pt + i*numno, plan[2].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  // To copy from host memory to device memory;
		cudaMemcpy(MatProp_2_Q2_d + i*plan[2].Localnumno, plan[2].MatPropQ2_h_Pt + i*numno, plan[2].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  
		cudaMemcpy(MatProp_2_Q3_d + i*plan[2].Localnumno, plan[2].MatPropQ3_h_Pt + i*numno, plan[2].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  
		cudaMemcpy(MatProp_2_Q4_d + i*plan[2].Localnumno, plan[2].MatPropQ4_h_Pt + i*numno, plan[2].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  
		cudaMemcpy(MatProp_2_Q5_d + i*plan[2].Localnumno, plan[2].MatPropQ5_h_Pt + i*numno, plan[2].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  
		cudaMemcpy(MatProp_2_Q6_d + i*plan[2].Localnumno, plan[2].MatPropQ6_h_Pt + i*numno, plan[2].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  
		cudaMemcpy(MatProp_2_Q7_d + i*plan[2].Localnumno, plan[2].MatPropQ7_h_Pt + i*numno, plan[2].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  
		cudaMemcpy(MatProp_2_Q8_d + i*plan[2].Localnumno, plan[2].MatPropQ8_h_Pt + i*numno, plan[2].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  

	}

	for(i=0; i<numdof*numnoel; i++) {

		// Copying Coordinate data from host to device:
		cudaMemcpy(Coord_2_Q1_d + i*plan[2].Localnumno, plan[2].CoordQ1_h_Pt + i*numno, plan[2].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  // To copy from host memory to device memory;
		cudaMemcpy(Coord_2_Q2_d + i*plan[2].Localnumno, plan[2].CoordQ2_h_Pt + i*numno, plan[2].Localnumno*sizeof(double), cudaMemcpyHostToDevice); 
		cudaMemcpy(Coord_2_Q3_d + i*plan[2].Localnumno, plan[2].CoordQ3_h_Pt + i*numno, plan[2].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  
		cudaMemcpy(Coord_2_Q4_d + i*plan[2].Localnumno, plan[2].CoordQ4_h_Pt + i*numno, plan[2].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  
		cudaMemcpy(Coord_2_Q5_d + i*plan[2].Localnumno, plan[2].CoordQ5_h_Pt + i*numno, plan[2].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  
		cudaMemcpy(Coord_2_Q6_d + i*plan[2].Localnumno, plan[2].CoordQ6_h_Pt + i*numno, plan[2].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  
		cudaMemcpy(Coord_2_Q7_d + i*plan[2].Localnumno, plan[2].CoordQ7_h_Pt + i*numno, plan[2].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  
		cudaMemcpy(Coord_2_Q8_d + i*plan[2].Localnumno, plan[2].CoordQ8_h_Pt + i*numno, plan[2].Localnumno*sizeof(double), cudaMemcpyHostToDevice);

	}

	// Copying ii position data from host to device:
	cudaMemcpy(iiPosition_2_d, plan[2].iiPosition_h_Pt, numdof*plan[2].Localnumno*sizeof(int), cudaMemcpyHostToDevice);    // To copy from host memory to device memory;

	// -----------------------------------------------------------------

	// Device 3:
	cudaSetDevice(3);              // Sets device 3 as the current device  

	// Allocating Material Properties data: 
	cudaMalloc((void **) &MatProp_3_Q1_d, plan[3].Localnumno*numprop*sizeof(double));
	cudaMalloc((void **) &MatProp_3_Q2_d, plan[3].Localnumno*numprop*sizeof(double));
	cudaMalloc((void **) &MatProp_3_Q3_d, plan[3].Localnumno*numprop*sizeof(double));
	cudaMalloc((void **) &MatProp_3_Q4_d, plan[3].Localnumno*numprop*sizeof(double));
	cudaMalloc((void **) &MatProp_3_Q5_d, plan[3].Localnumno*numprop*sizeof(double));
	cudaMalloc((void **) &MatProp_3_Q6_d, plan[3].Localnumno*numprop*sizeof(double));
	cudaMalloc((void **) &MatProp_3_Q7_d, plan[3].Localnumno*numprop*sizeof(double));
	cudaMalloc((void **) &MatProp_3_Q8_d, plan[3].Localnumno*numprop*sizeof(double));

	// Allocating Coodinate data:
	cudaMalloc((void **) &Coord_3_Q1_d, plan[3].Localnumno*numdof*numnoel*sizeof(double));
	cudaMalloc((void **) &Coord_3_Q2_d, plan[3].Localnumno*numdof*numnoel*sizeof(double));
	cudaMalloc((void **) &Coord_3_Q3_d, plan[3].Localnumno*numdof*numnoel*sizeof(double));
	cudaMalloc((void **) &Coord_3_Q4_d, plan[3].Localnumno*numdof*numnoel*sizeof(double));
	cudaMalloc((void **) &Coord_3_Q5_d, plan[3].Localnumno*numdof*numnoel*sizeof(double));
	cudaMalloc((void **) &Coord_3_Q6_d, plan[3].Localnumno*numdof*numnoel*sizeof(double));
	cudaMalloc((void **) &Coord_3_Q7_d, plan[3].Localnumno*numdof*numnoel*sizeof(double));
	cudaMalloc((void **) &Coord_3_Q8_d, plan[3].Localnumno*numdof*numnoel*sizeof(double));

	// Allocating ii position:
	cudaMalloc((void **) &iiPosition_3_d, numdof*plan[3].Localnumno*sizeof(int));          

	for(i=0; i<numprop; i++) {

		// Copying Material Properties data from host to device:
		cudaMemcpy(MatProp_3_Q1_d + i*plan[3].Localnumno, plan[3].MatPropQ1_h_Pt + i*numno, plan[3].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  // To copy from host memory to device memory;
		cudaMemcpy(MatProp_3_Q2_d + i*plan[3].Localnumno, plan[3].MatPropQ2_h_Pt + i*numno, plan[3].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  
		cudaMemcpy(MatProp_3_Q3_d + i*plan[3].Localnumno, plan[3].MatPropQ3_h_Pt + i*numno, plan[3].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  
		cudaMemcpy(MatProp_3_Q4_d + i*plan[3].Localnumno, plan[3].MatPropQ4_h_Pt + i*numno, plan[3].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  
		cudaMemcpy(MatProp_3_Q5_d + i*plan[3].Localnumno, plan[3].MatPropQ5_h_Pt + i*numno, plan[3].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  
		cudaMemcpy(MatProp_3_Q6_d + i*plan[3].Localnumno, plan[3].MatPropQ6_h_Pt + i*numno, plan[3].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  
		cudaMemcpy(MatProp_3_Q7_d + i*plan[3].Localnumno, plan[3].MatPropQ7_h_Pt + i*numno, plan[3].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  
		cudaMemcpy(MatProp_3_Q8_d + i*plan[3].Localnumno, plan[3].MatPropQ8_h_Pt + i*numno, plan[3].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  

	}

	for(i=0; i<numdof*numnoel; i++) {

		// Copying Coordinate data from host to device:
		cudaMemcpy(Coord_3_Q1_d + i*plan[3].Localnumno, plan[3].CoordQ1_h_Pt + i*numno, plan[3].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  // To copy from host memory to device memory;
		cudaMemcpy(Coord_3_Q2_d + i*plan[3].Localnumno, plan[3].CoordQ2_h_Pt + i*numno, plan[3].Localnumno*sizeof(double), cudaMemcpyHostToDevice); 
		cudaMemcpy(Coord_3_Q3_d + i*plan[3].Localnumno, plan[3].CoordQ3_h_Pt + i*numno, plan[3].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  
		cudaMemcpy(Coord_3_Q4_d + i*plan[3].Localnumno, plan[3].CoordQ4_h_Pt + i*numno, plan[3].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  
		cudaMemcpy(Coord_3_Q5_d + i*plan[3].Localnumno, plan[3].CoordQ5_h_Pt + i*numno, plan[3].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  
		cudaMemcpy(Coord_3_Q6_d + i*plan[3].Localnumno, plan[3].CoordQ6_h_Pt + i*numno, plan[3].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  
		cudaMemcpy(Coord_3_Q7_d + i*plan[3].Localnumno, plan[3].CoordQ7_h_Pt + i*numno, plan[3].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  
		cudaMemcpy(Coord_3_Q8_d + i*plan[3].Localnumno, plan[3].CoordQ8_h_Pt + i*numno, plan[3].Localnumno*sizeof(double), cudaMemcpyHostToDevice);

	}  

	// Copying ii position data from host to device:
	cudaMemcpy(iiPosition_3_d, plan[3].iiPosition_h_Pt, numdof*plan[3].Localnumno*sizeof(int), cudaMemcpyHostToDevice);    // To copy from host memory to device memory;

	// Releasing CPU memory:
	free(MatPropQ1_h); free(MatPropQ2_h); free(MatPropQ3_h); free(MatPropQ4_h); free(MatPropQ5_h); free(MatPropQ6_h); free(MatPropQ7_h); free(MatPropQ8_h); 
	free(CoordQ1_h); free(CoordQ2_h); free(CoordQ3_h); free(CoordQ4_h); free(CoordQ5_h); free(CoordQ6_h); free(CoordQ7_h); free(CoordQ8_h); 
	free(iiPosition_h); 

	//=============== x =============== x =============== x =============== x =============== x =============== x ===============

	// Reading load vector from coupled analysis:

	ReadConcLoad(numno, Vector_B_h, BC_h);

	//=============== x =============== x =============== x =============== x =============== x =============== x ===============

	//Get the number of terms of the matrix for each GPU:
		
	//Assign data ranges to GPUs
    gpuBaseTerm = gpuBaseRow = gpuBaseElem = 0;

	for(i=0; i<deviceCount; i++){

		plan[i].Vector_I_h_Pt = Vector_I_h + gpuBaseTerm;
		plan[i].Vector_B_h_Pt = Vector_B_h + gpuBaseRow;
		plan[i].Vector_X_h_Pt = Vector_X_h + gpuBaseRow;

		plan[i].CoordElem_h_Pt = CoordElem_h + gpuBaseElem; 
		plan[i].Connect_h_Pt   = Connect_h   + gpuBaseElem;
		plan[i].Strain_h_Pt    = Strain_h    + gpuBaseElem;

		plan[i].DeviceId = i;
		plan[i].DeviceNumber      = deviceCount;
		plan[i].GlobalNumRow      = numno;             //  Total number of rows;
		plan[i].NumMultiProcessor = GPUData[i*5+0];
		plan[i].BlockSize         = GPUData[i*5+4];
		plan[i].RowSize           = RowSize;

		gpuBaseTerm += numdof*plan[i].Localnumno*RowSize;
		gpuBaseRow  += numdof*plan[i].Localnumno;
		gpuBaseElem += plan[i].Localnumel;

	}

	//=============== x =============== x =============== x =============== x =============== x =============== x ===============

	// Gradient Conjugate Method Memory Allocation:

	// Allocating memory space on device "0" 

	cudaSetDevice(0);  // Sets device "0" as the current device

	GPUmultiProcessorCount_0 = GPUData[0*5+0];
	GPUmaxThreadsPerBlock_0  = GPUData[0*5+4];

	GPUBlockSizeX_0 = 32; GPUBlockSizeY_0 = 32;  // 32 x 32 = 1024

	BlockDimX = int(sqrt(double(plan[0].Localnumno))/GPUBlockSizeX_0)+1;
	BlockDimY = int(sqrt(double(plan[0].Localnumno))/GPUBlockSizeY_0)+1;
	GPUNumThread_0 = BlockDimX*BlockDimY*GPUmaxThreadsPerBlock_0;

	// Number of blocks needs to sum the terms of vector:
	if(GPUmultiProcessorCount_0 <= 4)       NumMultProc =  4;
	else if(GPUmultiProcessorCount_0 <=  8) NumMultProc =  8;
	else if(GPUmultiProcessorCount_0 <= 16) NumMultProc = 16;
	else if(GPUmultiProcessorCount_0 <= 32) NumMultProc = 32;

	// ------------------------------------------------------------------------------------

	// Doing the minimum NumRowLocal = GPUBlockSizeX*GPUBlockSizeY

	NumRowLocal  = numdof*plan[0].Localnumno;
	NumTermLocal = NumRowLocal*RowSize;
	NumRowGlobal = numdof*numno;

	// ------------------------------------------------------------------------------------
	
	cudaMalloc((void **) &Matrix_0_K_Aux_d, numdof*GPUNumThread_0*RowSize*sizeof(double));	
	cudaMalloc((void **) &Vector_0_B_d, NumRowLocal*sizeof(double));
	cudaMalloc((void **) &Vector_0_I_d, NumTermLocal*sizeof(int));	
	cudaMalloc((void **) &Matrix_0_M_d, NumRowLocal*sizeof(double));		
	cudaMalloc((void **) &Vector_0_X_d, NumRowLocal*sizeof(double));

	cudaMalloc((void **) &Vector_0_R_d, NumRowLocal*sizeof(double));		
	cudaMalloc((void **) &Vector_0_D_d, NumRowLocal*sizeof(double));	
	cudaMalloc((void **) &Vector_0_D_Global_d, NumRowGlobal*sizeof(double));
	cudaMalloc((void **) &Vector_0_Q_d, NumRowLocal*sizeof(double));		
	cudaMalloc((void **) &Vector_0_S_d, NumRowLocal*sizeof(double));	    
	cudaMalloc((void **) &delta_0_new_d, NumMultProc*sizeof(double));    
	cudaMalloc((void **) &delta_0_aux_d, NumMultProc*sizeof(double));    
	cudaMalloc((void **) &delta_0_new_V_d, NumRowLocal*sizeof(double));  
	cudaMalloc((void **) &delta_0_aux_V_d, NumRowLocal*sizeof(double));   
	cudaMalloc((void **) &delta_0_old_V_d, NumRowLocal*sizeof(double));
	cudaMalloc((void **) &delta_0_GPU_d, plan[0].DeviceNumber*sizeof(double));
	cudaMalloc((void **) &Result_0_d, sizeof(double));

	cudaMalloc((void **) &Connect_0_d, plan[0].Localnumel*numnoel*sizeof(int));
	cudaMalloc((void **) &CoordElem_0_d, plan[0].Localnumel*numdof*numnoel*sizeof(double));
	cudaMalloc((void **) &Strain_0_d, plan[0].Localnumel*numstr*sizeof(double));
	cudaMalloc((void **) &Vector_0_X_Global_d, NumRowGlobal*sizeof(double));

	delta_0_new_V_h = (double *)malloc(NumRowLocal*sizeof(double)); 

	// Copying from Host to Device for device 0:
	cudaMemcpy(Vector_0_I_d, plan[0].Vector_I_h_Pt, numdof*plan[0].Localnumno*RowSize*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(Vector_0_B_d, plan[0].Vector_B_h_Pt, numdof*plan[0].Localnumno*sizeof(double), cudaMemcpyHostToDevice);

	// Copying element node coordinate data from host to device:
	for(i=0; i<numdof*numnoel; i++)
		cudaMemcpy(CoordElem_0_d + i*plan[0].Localnumel, plan[0].CoordElem_h_Pt + i*numel, plan[0].Localnumel*sizeof(double), cudaMemcpyHostToDevice);

	// Copying element conectivity from host to device:
	for(i=0; i<numnoel; i++)
		cudaMemcpy(Connect_0_d + i*plan[0].Localnumel, plan[0].Connect_h_Pt + i*numel, plan[0].Localnumel*sizeof(int), cudaMemcpyHostToDevice);

	// Clearing vectors:
    cudaMemset(Matrix_0_K_Aux_d, 0, numdof*GPUNumThread_0*RowSize*sizeof(double));
	cudaMemset(delta_0_GPU_d, 0, plan[0].DeviceNumber*sizeof(double));
	cudaMemset(Vector_0_D_Global_d, 0, numdof*numno*sizeof(double));
	cudaMemset(Strain_0_d, 0, plan[0].Localnumel*numstr*sizeof(double));
	cudaMemset(Vector_0_X_d, 0, numdof*plan[0].Localnumno*sizeof(double));
	cudaMemset(Vector_0_Q_d, 0, numdof*plan[0].Localnumno*sizeof(double));
	cudaMemset(Vector_0_R_d, 0, numdof*plan[0].Localnumno*sizeof(double));

	//--------------------------------------------------------------------------------------------------------------------------------------

	// Allocating memory space on device "1" 

	cudaSetDevice(1);  // Sets device "1" as the current device

	GPUmultiProcessorCount_1 = GPUData[1*5+0];
	GPUmaxThreadsPerBlock_1  = GPUData[1*5+4];

	GPUBlockSizeX_1 = 32; GPUBlockSizeY_1 = 32;  // 32 x 32 = 1024
	
	BlockDimX = int(sqrt(double(plan[1].Localnumno))/GPUBlockSizeX_1)+1;
	BlockDimY = int(sqrt(double(plan[1].Localnumno))/GPUBlockSizeY_1)+1;
	GPUNumThread_1 = BlockDimX*BlockDimY*GPUmaxThreadsPerBlock_1;

	// Number of blocks needs to sum the terms of vector:
	if(GPUmultiProcessorCount_1 <= 4)       NumMultProc =  4;
	else if(GPUmultiProcessorCount_1 <=  8) NumMultProc =  8;
	else if(GPUmultiProcessorCount_1 <= 16) NumMultProc = 16;
	else if(GPUmultiProcessorCount_1 <= 32) NumMultProc = 32;

	// ------------------------------------------------------------------------------------

	// Doing the minimum NumRowLocal = GPUBlockSizeX*GPUBlockSizeY

	NumRowLocal  = numdof*plan[1].Localnumno;
	NumTermLocal = NumRowLocal*RowSize;
	NumRowGlobal = numdof*numno;

	// ------------------------------------------------------------------------------------

	cudaMalloc((void **) &Matrix_1_K_Aux_d, numdof*GPUNumThread_1*RowSize*sizeof(double));	
	cudaMalloc((void **) &Vector_1_B_d, NumRowLocal*sizeof(double));
	cudaMalloc((void **) &Vector_1_I_d, NumTermLocal*(sizeof(int)));	
	cudaMalloc((void **) &Matrix_1_M_d, NumRowLocal*sizeof(double));		
	cudaMalloc((void **) &Vector_1_X_d, NumRowLocal*sizeof(double));

	cudaMalloc((void **) &Vector_1_R_d, NumRowLocal*sizeof(double));		
	cudaMalloc((void **) &Vector_1_D_d, NumRowLocal*sizeof(double));	
	cudaMalloc((void **) &Vector_1_D_Global_d, NumRowGlobal*sizeof(double));
	cudaMalloc((void **) &Vector_1_Q_d, NumRowLocal*sizeof(double));		
	cudaMalloc((void **) &Vector_1_S_d, NumRowLocal*sizeof(double));	    
	cudaMalloc((void **) &delta_1_new_d, NumMultProc*sizeof(double));    
	cudaMalloc((void **) &delta_1_aux_d, NumMultProc*sizeof(double));    
	cudaMalloc((void **) &delta_1_new_V_d, NumRowLocal*sizeof(double));  
	cudaMalloc((void **) &delta_1_aux_V_d, NumRowLocal*sizeof(double));   
	cudaMalloc((void **) &delta_1_old_V_d, NumRowLocal*sizeof(double));
	cudaMalloc((void **) &delta_1_GPU_d, plan[1].DeviceNumber*sizeof(double));
	cudaMalloc((void **) &Result_1_d, sizeof(double));

	cudaMalloc((void **) &Connect_1_d, plan[1].Localnumel*numnoel*sizeof(int));
	cudaMalloc((void **) &CoordElem_1_d, plan[1].Localnumel*numdof*numnoel*sizeof(double));
	cudaMalloc((void **) &Strain_1_d, plan[1].Localnumel*numstr*sizeof(double));
	cudaMalloc((void **) &Vector_1_X_Global_d, NumRowGlobal*sizeof(double));

	delta_1_new_V_h = (double *)malloc(NumRowLocal*sizeof(double)); 

	// Copying from Host to Device for device 1:
	cudaMemcpy(Vector_1_I_d, plan[1].Vector_I_h_Pt, numdof*plan[1].Localnumno*RowSize*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(Vector_1_B_d, plan[1].Vector_B_h_Pt, numdof*plan[1].Localnumno*sizeof(double), cudaMemcpyHostToDevice);

	// Copying element node coordinate data from host to device:
	for(i=0; i<numdof*numnoel; i++)
		cudaMemcpy(CoordElem_1_d + i*plan[1].Localnumel, plan[1].CoordElem_h_Pt + i*numel, plan[1].Localnumel*sizeof(double), cudaMemcpyHostToDevice);  // To copy from host memory to device memory;
	
	// Copying element conectivity from host to device:
	for(i=0; i<numnoel; i++)
		cudaMemcpy(Connect_1_d + i*plan[1].Localnumel, plan[1].Connect_h_Pt + i*numel, plan[1].Localnumel*sizeof(int), cudaMemcpyHostToDevice);

	cudaMemset(Matrix_1_K_Aux_d, 0, numdof*GPUNumThread_1*RowSize*sizeof(double));
	cudaMemset(delta_1_GPU_d, 0, plan[1].DeviceNumber*sizeof(double));
	cudaMemset(Vector_1_D_Global_d, 0, numdof*numno*sizeof(double));
	cudaMemset(Strain_1_d, 0, plan[1].Localnumel*numstr*sizeof(double));
	cudaMemset(Vector_1_X_d, 0, numdof*plan[1].Localnumno*sizeof(double));
	cudaMemset(Vector_1_Q_d, 0, numdof*plan[1].Localnumno*sizeof(double));
	cudaMemset(Vector_1_R_d, 0, numdof*plan[1].Localnumno*sizeof(double));

	//--------------------------------------------------------------------------------------------------------------------------------------

	// Allocating memory space on device "2" 

	cudaSetDevice(2);  // Sets device "2" as the current device

	GPUmultiProcessorCount_2 = GPUData[2*5+0];
	GPUmaxThreadsPerBlock_2  = GPUData[2*5+4];

	GPUBlockSizeX_2 = 32; GPUBlockSizeY_2 = 32;  // 32 x 32 = 1024

	BlockDimX = int(sqrt(double(plan[2].Localnumno))/GPUBlockSizeX_2)+1;
	BlockDimY = int(sqrt(double(plan[2].Localnumno))/GPUBlockSizeY_2)+1;
	GPUNumThread_2 = BlockDimX*BlockDimY*GPUmaxThreadsPerBlock_2;

	// Number of blocks needs to sum the terms of vector:
	if(GPUmultiProcessorCount_2 <= 4)       NumMultProc =  4;
	else if(GPUmultiProcessorCount_2 <=  8) NumMultProc =  8;
	else if(GPUmultiProcessorCount_2 <= 16) NumMultProc = 16;
	else if(GPUmultiProcessorCount_2 <= 32) NumMultProc = 32;

	// ------------------------------------------------------------------------------------

	// Doing the minimum NumRowLocal = GPUBlockSizeX*GPUBlockSizeY

	NumRowLocal  = numdof*plan[2].Localnumno;
	NumTermLocal = NumRowLocal*RowSize;
	NumRowGlobal = numdof*numno;

	// ------------------------------------------------------------------------------------

	cudaMalloc((void **) &Matrix_2_K_Aux_d, numdof*GPUNumThread_2*RowSize*sizeof(double));	
	cudaMalloc((void **) &Vector_2_B_d, NumRowLocal*sizeof(double));
	cudaMalloc((void **) &Vector_2_I_d, NumTermLocal*(sizeof(int)));	
	cudaMalloc((void **) &Matrix_2_M_d, NumRowLocal*sizeof(double));		
	cudaMalloc((void **) &Vector_2_X_d, NumRowLocal*sizeof(double));

	cudaMalloc((void **) &Vector_2_R_d, NumRowLocal*sizeof(double));		
	cudaMalloc((void **) &Vector_2_D_d, NumRowLocal*sizeof(double));	
	cudaMalloc((void **) &Vector_2_D_Global_d, NumRowGlobal*sizeof(double));
	cudaMalloc((void **) &Vector_2_Q_d, NumRowLocal*sizeof(double));		
	cudaMalloc((void **) &Vector_2_S_d, NumRowLocal*sizeof(double));	    
	cudaMalloc((void **) &delta_2_new_d, NumMultProc*sizeof(double));    
	cudaMalloc((void **) &delta_2_aux_d, NumMultProc*sizeof(double));    
	cudaMalloc((void **) &delta_2_new_V_d, NumRowLocal*sizeof(double));  
	cudaMalloc((void **) &delta_2_aux_V_d, NumRowLocal*sizeof(double));   
	cudaMalloc((void **) &delta_2_old_V_d, NumRowLocal*sizeof(double));
	cudaMalloc((void **) &delta_2_GPU_d, plan[2].DeviceNumber*sizeof(double));
	cudaMalloc((void **) &Result_2_d, sizeof(double));

	cudaMalloc((void **) &Connect_2_d, plan[2].Localnumel*numnoel*sizeof(int));
	cudaMalloc((void **) &CoordElem_2_d, plan[2].Localnumel*numdof*numnoel*sizeof(double));
	cudaMalloc((void **) &Strain_2_d, plan[2].Localnumel*numstr*sizeof(double));
	cudaMalloc((void **) &Vector_2_X_Global_d, NumRowGlobal*sizeof(double));

	delta_2_new_V_h = (double *)malloc(NumRowLocal*sizeof(double)); 

	// Copying from Host to Device for device 2:
	cudaMemcpy(Vector_2_I_d, plan[2].Vector_I_h_Pt, numdof*plan[2].Localnumno*RowSize*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(Vector_2_B_d, plan[2].Vector_B_h_Pt, numdof*plan[2].Localnumno*sizeof(double), cudaMemcpyHostToDevice);

	// Copying element node coordinate data from host to device:
	for(i=0; i<numdof*numnoel; i++)
		cudaMemcpy(CoordElem_2_d + i*plan[2].Localnumel, plan[2].CoordElem_h_Pt + i*numel, plan[2].Localnumel*sizeof(double), cudaMemcpyHostToDevice);  // To copy from host memory to device memory;
	
	// Copying element conectivity from host to device:
	for(i=0; i<numnoel; i++)
		cudaMemcpy(Connect_2_d + i*plan[2].Localnumel, plan[2].Connect_h_Pt + i*numel, plan[2].Localnumel*sizeof(int), cudaMemcpyHostToDevice);

	cudaMemset(Matrix_2_K_Aux_d, 0, numdof*GPUNumThread_2*RowSize*sizeof(double));
	cudaMemset(delta_2_GPU_d, 0, plan[2].DeviceNumber*sizeof(double));
	cudaMemset(Vector_2_D_Global_d, 0, numdof*numno*sizeof(double));
	cudaMemset(Strain_2_d, 0, plan[2].Localnumel*numstr*sizeof(double));
	cudaMemset(Vector_2_X_d, 0, numdof*plan[2].Localnumno*sizeof(double));
	cudaMemset(Vector_2_Q_d, 0, numdof*plan[2].Localnumno*sizeof(double));
	cudaMemset(Vector_2_R_d, 0, numdof*plan[2].Localnumno*sizeof(double));

	//--------------------------------------------------------------------------------------------------------------------------------------

	// Allocating memory space on device "3" 

	cudaSetDevice(3);  // Sets device "3" as the current device

	GPUmultiProcessorCount_3 = GPUData[3*5+0];
	GPUmaxThreadsPerBlock_3  = GPUData[3*5+4];

	GPUBlockSizeX_3 = 32; GPUBlockSizeY_3 = 32;  // 32 x 32 = 1024

	BlockDimX = int(sqrt(double(plan[3].Localnumno))/GPUBlockSizeX_3)+1;
	BlockDimY = int(sqrt(double(plan[3].Localnumno))/GPUBlockSizeY_3)+1;
	GPUNumThread_3 = BlockDimX*BlockDimY*GPUmaxThreadsPerBlock_3;

	// Number of blocks needs to sum the terms of vector:
	if(GPUmultiProcessorCount_3 <= 4)       NumMultProc =  4;
	else if(GPUmultiProcessorCount_3 <=  8) NumMultProc =  8;
	else if(GPUmultiProcessorCount_3 <= 16) NumMultProc = 16;
	else if(GPUmultiProcessorCount_3 <= 32) NumMultProc = 32;

	// ------------------------------------------------------------------------------------

	// Doing the minimum NumRowLocal = GPUBlockSizeX*GPUBlockSizeY

	NumRowLocal  = numdof*plan[3].Localnumno;
	NumTermLocal = NumRowLocal*RowSize;
	NumRowGlobal = numdof*numno;

	// ------------------------------------------------------------------------------------
	
	cudaMalloc((void **) &Matrix_3_K_Aux_d, numdof*GPUNumThread_3*RowSize*sizeof(double));	
	cudaMalloc((void **) &Vector_3_B_d, NumRowLocal*sizeof(double));
	cudaMalloc((void **) &Vector_3_I_d, NumTermLocal*(sizeof(int)));	
	cudaMalloc((void **) &Matrix_3_M_d, NumRowLocal*sizeof(double));		
	cudaMalloc((void **) &Vector_3_X_d, NumRowLocal*sizeof(double));

	cudaMalloc((void **) &Vector_3_R_d, NumRowLocal*sizeof(double));		
	cudaMalloc((void **) &Vector_3_D_d, NumRowLocal*sizeof(double));	
	cudaMalloc((void **) &Vector_3_D_Global_d, NumRowGlobal*sizeof(double));
	cudaMalloc((void **) &Vector_3_Q_d, NumRowLocal*sizeof(double));		
	cudaMalloc((void **) &Vector_3_S_d, NumRowLocal*sizeof(double));	    
	cudaMalloc((void **) &delta_3_new_d, NumMultProc*sizeof(double));    
	cudaMalloc((void **) &delta_3_aux_d, NumMultProc*sizeof(double));    
	cudaMalloc((void **) &delta_3_new_V_d, NumRowLocal*sizeof(double));  
	cudaMalloc((void **) &delta_3_aux_V_d, NumRowLocal*sizeof(double));   
	cudaMalloc((void **) &delta_3_old_V_d, NumRowLocal*sizeof(double));
	cudaMalloc((void **) &delta_3_GPU_d, plan[3].DeviceNumber*sizeof(double));
	cudaMalloc((void **) &Result_3_d, sizeof(double));

	cudaMalloc((void **) &Connect_3_d, plan[3].Localnumel*numnoel*sizeof(int));
	cudaMalloc((void **) &CoordElem_3_d, plan[3].Localnumel*numdof*numnoel*sizeof(double));
	cudaMalloc((void **) &Strain_3_d, plan[3].Localnumel*numstr*sizeof(double));
	cudaMalloc((void **) &Vector_3_X_Global_d, NumRowGlobal*sizeof(double));

	delta_3_new_V_h = (double *)malloc(NumRowLocal*sizeof(double)); 

	// Copying from Host to Device for device 3:
	cudaMemcpy(Vector_3_I_d, plan[3].Vector_I_h_Pt, numdof*plan[3].Localnumno*RowSize*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(Vector_3_B_d, plan[3].Vector_B_h_Pt, numdof*plan[3].Localnumno*sizeof(double), cudaMemcpyHostToDevice);

	// Copying element node coordinate data from host to device:
	for(i=0; i<numdof*numnoel; i++)
		cudaMemcpy(CoordElem_3_d + i*plan[3].Localnumel, plan[3].CoordElem_h_Pt + i*numel, plan[3].Localnumel*sizeof(double), cudaMemcpyHostToDevice);  // To copy from host memory to device memory;
	
	// Copying element conectivity from host to device:
	for(i=0; i<numnoel; i++)
		cudaMemcpy(Connect_3_d + i*plan[3].Localnumel, plan[3].Connect_h_Pt + i*numel, plan[3].Localnumel*sizeof(int), cudaMemcpyHostToDevice);

	cudaMemset(Matrix_3_K_Aux_d, 0, numdof*GPUNumThread_3*RowSize*sizeof(double));
	cudaMemset(delta_3_GPU_d, 0, plan[3].DeviceNumber*sizeof(double));
	cudaMemset(Vector_3_D_Global_d, 0, numdof*numno*sizeof(double));
	cudaMemset(Strain_3_d, 0, plan[3].Localnumel*numstr*sizeof(double));
	cudaMemset(Vector_3_X_d, 0, numdof*plan[3].Localnumno*sizeof(double));
	cudaMemset(Vector_3_Q_d, 0, numdof*plan[3].Localnumno*sizeof(double));
	cudaMemset(Vector_3_R_d, 0, numdof*plan[3].Localnumno*sizeof(double));

	//=============== x =============== x =============== x =============== x =============== x =============== x ===============

	//Assign data ranges to GPUs

	plan[0].Localnumno_0 = plan[0].Localnumno;  plan[0].Localnumno_1 = plan[1].Localnumno;  plan[0].Localnumno_2 = plan[2].Localnumno;  plan[0].Localnumno_3 = plan[3].Localnumno;
	plan[1].Localnumno_0 = plan[0].Localnumno;  plan[1].Localnumno_1 = plan[1].Localnumno;  plan[1].Localnumno_2 = plan[2].Localnumno;  plan[1].Localnumno_3 = plan[3].Localnumno;
	plan[2].Localnumno_0 = plan[0].Localnumno;  plan[2].Localnumno_1 = plan[1].Localnumno;  plan[2].Localnumno_2 = plan[2].Localnumno;  plan[2].Localnumno_3 = plan[3].Localnumno;
	plan[3].Localnumno_0 = plan[0].Localnumno;  plan[3].Localnumno_1 = plan[1].Localnumno;  plan[3].Localnumno_2 = plan[2].Localnumno;  plan[3].Localnumno_3 = plan[3].Localnumno;

	//=============== x =============== x =============== x =============== x =============== x =============== x ===============
	// GPU global memory information:

	printf("\n");
	printf("         GPU global memory report \n");
	printf("         ========================================= ");
	printf("\n");

	for(i=0; i<4; i++) {
		cudaSetDevice(i);  // Sets device "0" as the current device
		cudaMemGetInfo( &free_byte, &total_byte );

		FreeMem[i] = free_byte/1e9;
		TotalMem[i] = total_byte/1e9;
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

	//------------------------------ Evaluating data preparing time
	time = clock()-time;
	printf("\n");
	printf("         Data Preparing Time: %0.3f s \n", time/CLOCKS_PER_SEC);
	//------------------------------

	//=============== x =============== x =============== x =============== x =============== x =============== x ===============

	// Assembly the stiffness matrix for Hexahedron Element on Multi GPUs:

	// To define one CPU thread for each GPU:

	omp_set_num_threads(deviceCount);

#pragma omp parallel 
	{

		CPUTheadId = omp_get_thread_num();

		AssemblyStiffnessHexahedronCPUOpenMP((plan + CPUTheadId), numdof, RowSize, 

		GPUNumThread_0, GPUBlockSizeX_0, GPUBlockSizeY_0, Matrix_0_K_Aux_d, Matrix_0_M_d,
		MatProp_0_Q1_d, MatProp_0_Q2_d, MatProp_0_Q3_d, MatProp_0_Q4_d, MatProp_0_Q5_d, MatProp_0_Q6_d, MatProp_0_Q7_d, MatProp_0_Q8_d, 
		Coord_0_Q1_d, Coord_0_Q2_d, Coord_0_Q3_d, Coord_0_Q4_d, Coord_0_Q5_d, Coord_0_Q6_d, Coord_0_Q7_d, Coord_0_Q8_d, iiPosition_0_d,
		
		GPUNumThread_1, GPUBlockSizeX_1, GPUBlockSizeY_1, Matrix_1_K_Aux_d, Matrix_1_M_d,
		MatProp_1_Q1_d, MatProp_1_Q2_d, MatProp_1_Q3_d, MatProp_1_Q4_d, MatProp_1_Q5_d, MatProp_1_Q6_d, MatProp_1_Q7_d, MatProp_1_Q8_d, 
		Coord_1_Q1_d, Coord_1_Q2_d, Coord_1_Q3_d, Coord_1_Q4_d, Coord_1_Q5_d, Coord_1_Q6_d, Coord_1_Q7_d, Coord_1_Q8_d, iiPosition_1_d,
		
		GPUNumThread_2, GPUBlockSizeX_2, GPUBlockSizeY_2, Matrix_2_K_Aux_d, Matrix_2_M_d,
		MatProp_2_Q1_d, MatProp_2_Q2_d, MatProp_2_Q3_d, MatProp_2_Q4_d, MatProp_2_Q5_d, MatProp_2_Q6_d, MatProp_2_Q7_d, MatProp_2_Q8_d, 
		Coord_2_Q1_d, Coord_2_Q2_d, Coord_2_Q3_d, Coord_2_Q4_d, Coord_2_Q5_d, Coord_2_Q6_d, Coord_2_Q7_d, Coord_2_Q8_d, iiPosition_2_d,
		
		GPUNumThread_3, GPUBlockSizeX_3, GPUBlockSizeY_3, Matrix_3_K_Aux_d, Matrix_3_M_d,
		MatProp_3_Q1_d, MatProp_3_Q2_d, MatProp_3_Q3_d, MatProp_3_Q4_d, MatProp_3_Q5_d, MatProp_3_Q6_d, MatProp_3_Q7_d, MatProp_3_Q8_d, 
		Coord_3_Q1_d, Coord_3_Q2_d, Coord_3_Q3_d, Coord_3_Q4_d, Coord_3_Q5_d, Coord_3_Q6_d, Coord_3_Q7_d, Coord_3_Q8_d, iiPosition_3_d);

	}

	//=============== x =============== x =============== x =============== x =============== x =============== x ===============

	/*cudaSetDevice(0);  // Sets device "0" as the current device

	using namespace std;
	ofstream outfile;
	std::string fileName2 = "StiffnessMatrixGPU";
	fileName2 += ".dat";
	outfile.open(fileName2.c_str());
	using std::ios;
	using std::setw;
	outfile.setf(ios::fixed,ios::floatfield);
	outfile.precision(5);

	BlockDimX = int(sqrt(double(numno))/32)+1;
	BlockDimY = int(sqrt(double(numno))/32)+1;
	NumThread = BlockDimX*BlockDimY*1024;
	
	int NumThreadSpace = RowSize*GPUNumThread_0;

	Matrix_K_h  = (double *)malloc(numdof*GPUNumThread_0*RowSize*sizeof(double));           // Allocating menoty space for host
	cudaMemcpy(Matrix_K_h, Matrix_0_K_Aux_d, sizeof(double)*numdof*GPUNumThread_0*RowSize, cudaMemcpyDeviceToHost);  // mudou NumThread*/

	//int *Aux_h;

	//Aux_h  = (int *)malloc(numdof*NumThread*RowSize*sizeof(int));           // Allocating menoty space for host

	//int *iiPosition_aux_h;
	//iiPosition_aux_h = (int *)malloc(3*2000*(sizeof(int)));

	//cudaMemcpy(Matrix_K_h, Vector_3_B_d, sizeof(double)*numdof*plan[0].Localnumno, cudaMemcpyDeviceToHost);
	//cudaMemcpy(Matrix_K_h, Matrix_3_K_d, sizeof(double)*numdof*plan[3].Localnumno*128, cudaMemcpyDeviceToHost);  // mudou NumThread
	//cudaMemcpy(Matrix_M_h, Matrix_M_d, sizeof(double)*numdof*numno, cudaMemcpyDeviceToHost);
	//cudaMemcpy(Aux_h, iiPosition_0_d, sizeof(int)*numdof*plan[0].Localnumno, cudaMemcpyDeviceToHost);

	/*cudaMemcpy(Matrix_K_h, Matrix_2_K_Aux_d, sizeof(double)*numdof*plan[2].Localnumno*128, cudaMemcpyDeviceToHost);

	// Print Matrix_3_K_Aux_d
	for(i=0; i<GPUNumThread_0; i++) {

		outfile << "line = " << 3*i << endl;
		for(j=0; j<RowSize; j++) {
			if(Matrix_K_h[j*GPUNumThread_0+i] != 0.) {
				if(fabs(Matrix_K_h[j*GPUNumThread_0+i])>0.1)                  outfile << Matrix_K_h[j*GPUNumThread_0+i] << " ";
			}
		}
		outfile << endl; outfile << endl;
		
		outfile << "line = " << 3*i+1 << endl;
		for(j=0; j<RowSize; j++) {	
			if(Matrix_K_h[j*GPUNumThread_0+i+NumThreadSpace] != 0.) {
				if(fabs(Matrix_K_h[j*GPUNumThread_0+i+NumThreadSpace])>0.1)   outfile << Matrix_K_h[j*GPUNumThread_0+i+NumThreadSpace] << " ";
			}
		}
		outfile << endl; outfile << endl;

		outfile << "line = " << 3*i+2 << endl;
		for(j=0; j<RowSize; j++) {	
			if(Matrix_K_h[j*GPUNumThread_0+i+2*NumThreadSpace] != 0.) {
				if(fabs(Matrix_K_h[j*GPUNumThread_0+i+2*NumThreadSpace])>0.1) outfile << Matrix_K_h[j*GPUNumThread_0+i+2*NumThreadSpace] << " ";
			}
		}
		outfile << endl; outfile << endl;
		
	}*/

	// =======================================================================================================================================================


	//Matrix_0_K_d 
	/*for(i=0; i<3*plan[3].Localnumno; i++) {

		outfile << "line = " << i << endl;
		for(j=0; j<RowSize; j++) {
			if(Matrix_K_h[i*RowSize+j] != 0.) {
				if(fabs(Matrix_K_h[i*RowSize+j])>0.1)                  outfile << Matrix_K_h[i*RowSize+j] << " ";
			}
		}
		outfile << endl; outfile << endl;
		
	}*/


	//Matrix_0_M_d
	/*for(i=0; i<plan[0].Localnumno; i++) {

		outfile << "line = " << i << endl;
		for(j=0; j<3; j++) {
			
			outfile << Matrix_K_h[i*3+j] << " ";

		}
		outfile << endl; outfile << endl;
		
	}*/


	//iiPosition_0_d
	/*for(i=0; i<plan[0].Localnumno; i++) {

		outfile << "line = " << i << endl;
		for(j=0; j<3; j++) {
			
				outfile << Aux_h[i*3+j] << " ";

		}
		outfile << endl; outfile << endl;
		
	}*/

	//outfile.close();

	//=============== x =============== x =============== x =============== x =============== x =============== x ===============

	// Solve linear system of equation with conjugate gradient on Multi GPUs:

	// To define one CPU thread for each GPU:

	omp_set_num_threads(deviceCount);

#pragma omp parallel 
	{

		CPUTheadId = omp_get_thread_num();
		
		/*SolveLinearSystemEquationCPUOpenMP((plan + CPUTheadId), numdof, RowSize, 

		GPUmaxThreadsPerBlock_0, GPUmultiProcessorCount_0, GPUBlockSizeX_0, GPUBlockSizeY_0, Matrix_0_K_d, Vector_0_I_d, Vector_0_B_d, Vector_0_X_d, 
		Matrix_0_M_d, Vector_0_R_d, Vector_0_D_d, Vector_0_D_Global_d, Vector_0_Q_d, Vector_0_S_d, delta_0_new_d, delta_0_aux_d, delta_0_new_V_d, 
		delta_0_aux_V_d, delta_0_old_V_d, delta_0_new_V_h, delta_0_GPU_d, Result_0_d, Vector_0_X_Global_d,

		GPUmaxThreadsPerBlock_1, GPUmultiProcessorCount_1, GPUBlockSizeX_1, GPUBlockSizeY_1, Matrix_1_K_d, Vector_1_I_d, Vector_1_B_d, Vector_1_X_d, 
		Matrix_1_M_d, Vector_1_R_d, Vector_1_D_d, Vector_1_D_Global_d, Vector_1_Q_d, Vector_1_S_d, delta_1_new_d, delta_1_aux_d, delta_1_new_V_d, 
		delta_1_aux_V_d, delta_1_old_V_d, delta_1_new_V_h, delta_1_GPU_d, Result_1_d, Vector_1_X_Global_d,

		GPUmaxThreadsPerBlock_2, GPUmultiProcessorCount_2, GPUBlockSizeX_2, GPUBlockSizeY_2, Matrix_2_K_d, Vector_2_I_d, Vector_2_B_d, Vector_2_X_d, 
		Matrix_2_M_d, Vector_2_R_d, Vector_2_D_d, Vector_2_D_Global_d, Vector_2_Q_d, Vector_2_S_d, delta_2_new_d, delta_2_aux_d, delta_2_new_V_d, 
		delta_2_aux_V_d, delta_2_old_V_d, delta_2_new_V_h, delta_2_GPU_d, Result_2_d, Vector_2_X_Global_d,

		GPUmaxThreadsPerBlock_3, GPUmultiProcessorCount_3, GPUBlockSizeX_3, GPUBlockSizeY_3, Matrix_3_K_d, Vector_3_I_d, Vector_3_B_d, Vector_3_X_d, 
		Matrix_3_M_d, Vector_3_R_d, Vector_3_D_d, Vector_3_D_Global_d, Vector_3_Q_d, Vector_3_S_d, delta_3_new_d, delta_3_aux_d, delta_3_new_V_d, 
		delta_3_aux_V_d, delta_3_old_V_d, delta_3_new_V_h, delta_3_GPU_d, Result_3_d, Vector_3_X_Global_d);*/

		SolveLinearSystemEquationCPUOpenMP((plan + CPUTheadId), numdof, RowSize, 

		GPUmaxThreadsPerBlock_0, GPUmultiProcessorCount_0, GPUBlockSizeX_0, GPUBlockSizeY_0, Matrix_0_K_Aux_d, Vector_0_I_d, Vector_0_B_d, Vector_0_X_d, 
		Matrix_0_M_d, Vector_0_R_d, Vector_0_D_d, Vector_0_D_Global_d, Vector_0_Q_d, Vector_0_S_d, delta_0_new_d, delta_0_aux_d, delta_0_new_V_d, 
		delta_0_aux_V_d, delta_0_old_V_d, delta_0_new_V_h, delta_0_GPU_d, Result_0_d, Vector_0_X_Global_d,

		GPUmaxThreadsPerBlock_1, GPUmultiProcessorCount_1, GPUBlockSizeX_1, GPUBlockSizeY_1, Matrix_1_K_Aux_d, Vector_1_I_d, Vector_1_B_d, Vector_1_X_d, 
		Matrix_1_M_d, Vector_1_R_d, Vector_1_D_d, Vector_1_D_Global_d, Vector_1_Q_d, Vector_1_S_d, delta_1_new_d, delta_1_aux_d, delta_1_new_V_d, 
		delta_1_aux_V_d, delta_1_old_V_d, delta_1_new_V_h, delta_1_GPU_d, Result_1_d, Vector_1_X_Global_d,

		GPUmaxThreadsPerBlock_2, GPUmultiProcessorCount_2, GPUBlockSizeX_2, GPUBlockSizeY_2, Matrix_2_K_Aux_d, Vector_2_I_d, Vector_2_B_d, Vector_2_X_d, 
		Matrix_2_M_d, Vector_2_R_d, Vector_2_D_d, Vector_2_D_Global_d, Vector_2_Q_d, Vector_2_S_d, delta_2_new_d, delta_2_aux_d, delta_2_new_V_d, 
		delta_2_aux_V_d, delta_2_old_V_d, delta_2_new_V_h, delta_2_GPU_d, Result_2_d, Vector_2_X_Global_d,

		GPUmaxThreadsPerBlock_3, GPUmultiProcessorCount_3, GPUBlockSizeX_3, GPUBlockSizeY_3, Matrix_3_K_Aux_d, Vector_3_I_d, Vector_3_B_d, Vector_3_X_d, 
		Matrix_3_M_d, Vector_3_R_d, Vector_3_D_d, Vector_3_D_Global_d, Vector_3_Q_d, Vector_3_S_d, delta_3_new_d, delta_3_aux_d, delta_3_new_V_d, 
		delta_3_aux_V_d, delta_3_old_V_d, delta_3_new_V_h, delta_3_GPU_d, Result_3_d, Vector_3_X_Global_d);

	}

	//=============== x =============== x =============== x =============== x =============== x =============== x ===============

	// Evaluate strain state on Multi GPUs:

	// To define one CPU thread for each GPU:

	omp_set_num_threads(deviceCount);

#pragma omp parallel 
	{

		CPUTheadId = omp_get_thread_num();
		
		EvaluateStrainStateCPUOpenMP((plan + CPUTheadId), numdof, RowSize,
		GPUBlockSizeX_0, GPUBlockSizeY_0, Connect_0_d, CoordElem_0_d, Vector_0_X_Global_d, Strain_0_d,
		GPUBlockSizeX_1, GPUBlockSizeY_1, Connect_1_d, CoordElem_1_d, Vector_1_X_Global_d, Strain_1_d,
		GPUBlockSizeX_2, GPUBlockSizeY_2, Connect_2_d, CoordElem_2_d, Vector_2_X_Global_d, Strain_2_d,
		GPUBlockSizeX_3, GPUBlockSizeY_3, Connect_3_d, CoordElem_3_d, Vector_3_X_Global_d, Strain_3_d);

	}

	//=============== x =============== x =============== x =============== x =============== x =============== x ===============

	// Evaluate stress state on GPU:

	//EvaluateStressStateGPU(numel, BlockSizeX, BlockSizeY, Material_Data_d, Strain_d, Stress_d);

	//=============== x =============== x =============== x =============== x =============== x =============== x ===============
	
	// Write displacement field in output file GPUDisplacement.pos
	
	// Device 0:
	cudaSetDevice(0);              // Sets device 0 as the current device
	cudaMemcpy(Vector_X_h, Vector_0_X_Global_d, sizeof(double)*numdof*numno, cudaMemcpyDeviceToHost);

	time = clock();
	
	WriteDisplacement(numno, Vector_X_h);

	time = clock()-time;
	printf("\n");
	printf("         Displacement Field Writing Time: %0.3f s \n", time/CLOCKS_PER_SEC);
		
	// ==================================================================================================================================
	
	// Write Strain state in output file GPUStrainState.pos
	
	// -----------------------------------------------------------------

	// Device 0:
	cudaSetDevice(0);              // Sets device 0 as the current device         

	// Copying Strain State data from device (1/4) to host (global):
	for(i=0; i<numstr; i++)
		cudaMemcpy(plan[0].Strain_h_Pt + i*numel, Strain_0_d + i*plan[0].Localnumel, plan[0].Localnumel*sizeof(double), cudaMemcpyDeviceToHost);
		
	// Device 1:
	cudaSetDevice(1);              // Sets device 1 as the current device         

	// Copying Strain State data from device (1/4) to host (global):
	for(i=0; i<numstr; i++)
		cudaMemcpy(plan[1].Strain_h_Pt + i*numel, Strain_1_d + i*plan[1].Localnumel, plan[1].Localnumel*sizeof(double), cudaMemcpyDeviceToHost);

	// Device 2:
	cudaSetDevice(2);              // Sets device 2 as the current device         

	// Copying Strain State data from device (1/4) to host (global):
	for(i=0; i<numstr; i++)
		cudaMemcpy(plan[2].Strain_h_Pt + i*numel, Strain_2_d + i*plan[2].Localnumel, plan[2].Localnumel*sizeof(double), cudaMemcpyDeviceToHost);

	// Device 3:
	cudaSetDevice(3);              // Sets device 3 as the current device         

	// Copying Strain State data from device (1/4) to host (global):
	for(i=0; i<numstr; i++)
		cudaMemcpy(plan[3].Strain_h_Pt + i*numel, Strain_3_d + i*plan[3].Localnumel, plan[3].Localnumel*sizeof(double), cudaMemcpyDeviceToHost);

	// -----------------------------------------------------------------

	time = clock();

	WriteStrainState(numel, Strain_h);

	time = clock()-time;
	printf("\n");
	printf("         Strain State Writing Time: %0.3f s \n", time/CLOCKS_PER_SEC);

	//=============== x =============== x =============== x =============== x =============== x =============== x ===============
	
	// Write Stress state in output file GPUStressState.pos
	
	//cudaMemcpy(Stress_h, Stress_d, numel*numstr*8*sizeof(double), cudaMemcpyDeviceToHost);   // To copy from device memory to host memory;

	//WriteStressState(numel, Stress_h);

	//=============== x =============== x =============== x =============== x =============== x =============== x ===============
	
	return 1;

}

//==============================================================================
//	Assembly the stiffness matrix for Hexahedron Element on Multi GPUs:

void EvaluateStrainStateCPUOpenMP(GPU_Struct *plan, int numdof, int RowSize,
int GPUBlockSizeX_0, int GPUBlockSizeY_0, int *Connect_0_d, double *CoordElem_0_d, double *Vector_0_X_Global_d, double *Strain_0_d,
int GPUBlockSizeX_1, int GPUBlockSizeY_1, int *Connect_1_d, double *CoordElem_1_d, double *Vector_1_X_Global_d, double *Strain_1_d,
int GPUBlockSizeX_2, int GPUBlockSizeY_2, int *Connect_2_d, double *CoordElem_2_d, double *Vector_2_X_Global_d, double *Strain_2_d,
int GPUBlockSizeX_3, int GPUBlockSizeY_3, int *Connect_3_d, double *CoordElem_3_d, double *Vector_3_X_Global_d, double *Strain_3_d)
{

	switch(plan->DeviceId) {
				
		case 0:  // Run device 0

			EvaluateStrainStateGPU(plan->DeviceId, plan->Localnumel, RowSize, GPUBlockSizeX_0, GPUBlockSizeY_0, 
			Connect_0_d, CoordElem_0_d, Vector_0_X_Global_d, Strain_0_d);
			
			break;
	
		case 1:  // Run device 1

			EvaluateStrainStateGPU(plan->DeviceId, plan->Localnumel, RowSize, GPUBlockSizeX_1, GPUBlockSizeY_1, 
			Connect_1_d, CoordElem_1_d, Vector_1_X_Global_d, Strain_1_d);
		
			break;
			
		case 2:  // Run device 2

			EvaluateStrainStateGPU(plan->DeviceId, plan->Localnumel, RowSize, GPUBlockSizeX_2, GPUBlockSizeY_2, 
			Connect_2_d, CoordElem_2_d, Vector_2_X_Global_d, Strain_2_d);
			
			break;
			
		case 3:  // Run device 3

			EvaluateStrainStateGPU(plan->DeviceId, plan->Localnumel, RowSize, GPUBlockSizeX_3, GPUBlockSizeY_3, 
			Connect_3_d, CoordElem_3_d, Vector_3_X_Global_d, Strain_3_d);
		
			break;  
			
	}

}

//==============================================================================
//
void EvaluateStrainStateGPU(int DeviceId, int Localnumel, int RowSize, int GPUBlockSizeX, int GPUBlockSizeY, 
int *Connect_d, double *CoordElem_d, double *Vector_X_d, double *Strain_d)
{
	double time;
	
	using namespace std;
	
	// ==================================================================================================================================
	// Evaluate Strain state

	// Barrier Synchronization  ===================================================================
#pragma omp barrier
	
	if(DeviceId == 0) {
		printf("\n");
		printf("         Evaluating Strain State \n");
		printf("         ========================================= ");
	}
	
	// =================================================================
	
	dim3 numBlocksStrain(int( sqrt(double(Localnumel)) /GPUBlockSizeX)+1, int( sqrt(double(Localnumel)) /GPUBlockSizeY)+1);
	dim3 threadsPerBlockStrain(GPUBlockSizeX, GPUBlockSizeY);
	
	// To start time count to evaluate strain state:
	time = clock();
		
	Launch_EvaluateStrainState(numBlocksStrain, threadsPerBlockStrain, Localnumel, RowSize, CoordElem_d, Connect_d, Vector_X_d, Strain_d);

	// Barrier Synchronization  ===================================================================
#pragma omp barrier

	if(DeviceId == 0) {
		printf("\n");
		printf("         Evaluating Strain State Time: %0.3f s \n", (clock()-time)/CLOCKS_PER_SEC);
	}

	// Barrier Synchronization  ===================================================================
#pragma omp barrier
	
}

//==============================================================================
//	Assembly the stiffness matrix for Hexahedron Element on Multi GPUs:

void AssemblyStiffnessHexahedronCPUOpenMP(GPU_Struct *plan, int numdof, int RowSize, 

int GPUNumThread_0, int GPUBlockSizeX_0, int GPUBlockSizeY_0, double *Matrix_0_K_Aux_d, double *Matrix_0_M_d,
double *MatProp_0_Q1_d, double *MatProp_0_Q2_d, double *MatProp_0_Q3_d, double *MatProp_0_Q4_d, double *MatProp_0_Q5_d, double *MatProp_0_Q6_d, double *MatProp_0_Q7_d, double *MatProp_0_Q8_d, 
double *Coord_0_Q1_d, double *Coord_0_Q2_d, double *Coord_0_Q3_d, double *Coord_0_Q4_d, double *Coord_0_Q5_d, double *Coord_0_Q6_d, double *Coord_0_Q7_d, double *Coord_0_Q8_d, int *iiPosition_0_d,
		
int GPUNumThread_1, int GPUBlockSizeX_1, int GPUBlockSizeY_1, double *Matrix_1_K_Aux_d, double *Matrix_1_M_d,
double *MatProp_1_Q1_d, double *MatProp_1_Q2_d, double *MatProp_1_Q3_d, double *MatProp_1_Q4_d, double *MatProp_1_Q5_d, double *MatProp_1_Q6_d, double *MatProp_1_Q7_d, double *MatProp_1_Q8_d, 
double *Coord_1_Q1_d, double *Coord_1_Q2_d, double *Coord_1_Q3_d, double *Coord_1_Q4_d, double *Coord_1_Q5_d, double *Coord_1_Q6_d, double *Coord_1_Q7_d, double *Coord_1_Q8_d, int *iiPosition_1_d,
		
int GPUNumThread_2, int GPUBlockSizeX_2, int GPUBlockSizeY_2, double *Matrix_2_K_Aux_d, double *Matrix_2_M_d,
double *MatProp_2_Q1_d, double *MatProp_2_Q2_d, double *MatProp_2_Q3_d, double *MatProp_2_Q4_d, double *MatProp_2_Q5_d, double *MatProp_2_Q6_d, double *MatProp_2_Q7_d, double *MatProp_2_Q8_d, 
double *Coord_2_Q1_d, double *Coord_2_Q2_d, double *Coord_2_Q3_d, double *Coord_2_Q4_d, double *Coord_2_Q5_d, double *Coord_2_Q6_d, double *Coord_2_Q7_d, double *Coord_2_Q8_d, int *iiPosition_2_d,
		
int GPUNumThread_3, int GPUBlockSizeX_3, int GPUBlockSizeY_3, double *Matrix_3_K_Aux_d, double *Matrix_3_M_d,
double *MatProp_3_Q1_d, double *MatProp_3_Q2_d, double *MatProp_3_Q3_d, double *MatProp_3_Q4_d, double *MatProp_3_Q5_d, double *MatProp_3_Q6_d, double *MatProp_3_Q7_d, double *MatProp_3_Q8_d, 
double *Coord_3_Q1_d, double *Coord_3_Q2_d, double *Coord_3_Q3_d, double *Coord_3_Q4_d, double *Coord_3_Q5_d, double *Coord_3_Q6_d, double *Coord_3_Q7_d, double *Coord_3_Q8_d, int *iiPosition_3_d)
{

	switch(plan->DeviceId) {
				
		case 0:  // Run device 0

			AssemblyStiffnessHexahedronGPU(plan->DeviceId, plan->Localnumno, numdof, RowSize, GPUNumThread_0, GPUBlockSizeX_0, GPUBlockSizeY_0, 
			Matrix_0_K_Aux_d, Matrix_0_M_d,
			MatProp_0_Q1_d, MatProp_0_Q2_d, MatProp_0_Q3_d, MatProp_0_Q4_d, MatProp_0_Q5_d, MatProp_0_Q6_d, MatProp_0_Q7_d, MatProp_0_Q8_d, 
			Coord_0_Q1_d, Coord_0_Q2_d, Coord_0_Q3_d, Coord_0_Q4_d, Coord_0_Q5_d, Coord_0_Q6_d, Coord_0_Q7_d, Coord_0_Q8_d, iiPosition_0_d);
			
			break;
	
		case 1:  // Run device 1

			AssemblyStiffnessHexahedronGPU(plan->DeviceId, plan->Localnumno, numdof, RowSize, GPUNumThread_1, GPUBlockSizeX_1, GPUBlockSizeY_1, 
			Matrix_1_K_Aux_d, Matrix_1_M_d,
			MatProp_1_Q1_d, MatProp_1_Q2_d, MatProp_1_Q3_d, MatProp_1_Q4_d, MatProp_1_Q5_d, MatProp_1_Q6_d, MatProp_1_Q7_d, MatProp_1_Q8_d, 
			Coord_1_Q1_d, Coord_1_Q2_d, Coord_1_Q3_d, Coord_1_Q4_d, Coord_1_Q5_d, Coord_1_Q6_d, Coord_1_Q7_d, Coord_1_Q8_d, iiPosition_1_d);
		
			break;
			
		case 2:  // Run device 2

			AssemblyStiffnessHexahedronGPU(plan->DeviceId, plan->Localnumno, numdof, RowSize, GPUNumThread_2, GPUBlockSizeX_2, GPUBlockSizeY_2, 
			Matrix_2_K_Aux_d, Matrix_2_M_d,
			MatProp_2_Q1_d, MatProp_2_Q2_d, MatProp_2_Q3_d, MatProp_2_Q4_d, MatProp_2_Q5_d, MatProp_2_Q6_d, MatProp_2_Q7_d, MatProp_2_Q8_d, 
			Coord_2_Q1_d, Coord_2_Q2_d, Coord_2_Q3_d, Coord_2_Q4_d, Coord_2_Q5_d, Coord_2_Q6_d, Coord_2_Q7_d, Coord_2_Q8_d, iiPosition_2_d);
			
			break;
			
		case 3:  // Run device 3

			AssemblyStiffnessHexahedronGPU(plan->DeviceId, plan->Localnumno, numdof, RowSize, GPUNumThread_3, GPUBlockSizeX_3, GPUBlockSizeY_3, 
			Matrix_3_K_Aux_d, Matrix_3_M_d,
			MatProp_3_Q1_d, MatProp_3_Q2_d, MatProp_3_Q3_d, MatProp_3_Q4_d, MatProp_3_Q5_d, MatProp_3_Q6_d, MatProp_3_Q7_d, MatProp_3_Q8_d, 
			Coord_3_Q1_d, Coord_3_Q2_d, Coord_3_Q3_d, Coord_3_Q4_d, Coord_3_Q5_d, Coord_3_Q6_d, Coord_3_Q7_d, Coord_3_Q8_d, iiPosition_3_d);
		
			break;  
			
	}

}

//==============================================================================
void AssemblyStiffnessHexahedronGPU(int DeviceId, int numno, int numdof, int RowSize, int NumThread, int BlockSizeX, int BlockSizeY, 
double *Matrix_K_Aux_d, double *Matrix_M_d, 
double *MatPropQ1_d, double *MatPropQ2_d, double *MatPropQ3_d, double *MatPropQ4_d, double *MatPropQ5_d, double *MatPropQ6_d, double *MatPropQ7_d, double *MatPropQ8_d, 
double *CoordQ1_d, double *CoordQ2_d, double *CoordQ3_d, double *CoordQ4_d, double *CoordQ5_d, double *CoordQ6_d, double *CoordQ7_d, double *CoordQ8_d, int *iiPosition_d)
{
	double time;
	
	using namespace std;

	// ==================================================================================================================================
	// Sets device as the current device

	cudaSetDevice(DeviceId);   
	
	// ==================================================================================================================================
	// Evaluating Stiffness Matrix (SM):

	if(DeviceId == 0) {
		printf("\n");
		printf("         Assembling Stiffness Matrix \n");
		printf("         ========================================= ");   
	}
	
	// Barrier Synchronization  ===================================================================
#pragma omp barrier

	// To start time count to assembly stiffness matrix:
	time = clock();
		
	// Compute Capability 2.x and 3.0 => Maximum number of threads per block = 1024 (maxThreadsPerBlock)
	
	dim3 numBlocksSM(int(sqrt(double(numno))/BlockSizeX)+1, int(sqrt(double(numno))/BlockSizeY)+1);	
	dim3 threadsPerBlockSM(BlockSizeX, BlockSizeY); 

	// Assembly on GPU the Stiffness Matrix

	Launch_AssemblyStiffness3DQ1(numBlocksSM, threadsPerBlockSM, numno, RowSize, CoordQ1_d, MatPropQ1_d, Matrix_K_Aux_d);
	
	Launch_AssemblyStiffness3DQ2(numBlocksSM, threadsPerBlockSM, numno, RowSize, CoordQ2_d, MatPropQ2_d, Matrix_K_Aux_d);
	
	Launch_AssemblyStiffness3DQ3(numBlocksSM, threadsPerBlockSM, numno, RowSize, CoordQ3_d, MatPropQ3_d, Matrix_K_Aux_d);
	
	Launch_AssemblyStiffness3DQ4(numBlocksSM, threadsPerBlockSM, numno, RowSize, CoordQ4_d, MatPropQ4_d, Matrix_K_Aux_d);

	Launch_AssemblyStiffness3DQ5(numBlocksSM, threadsPerBlockSM, numno, RowSize, CoordQ5_d, MatPropQ5_d, Matrix_K_Aux_d);
	
	Launch_AssemblyStiffness3DQ6(numBlocksSM, threadsPerBlockSM, numno, RowSize, CoordQ6_d, MatPropQ6_d, Matrix_K_Aux_d);
	
	Launch_AssemblyStiffness3DQ7(numBlocksSM, threadsPerBlockSM, numno, RowSize, CoordQ7_d, MatPropQ7_d, Matrix_K_Aux_d);
	
	Launch_AssemblyStiffness3DQ8(numBlocksSM, threadsPerBlockSM, numno, RowSize, CoordQ8_d, MatPropQ8_d, Matrix_K_Aux_d);

	// Barrier Synchronization  ===================================================================
#pragma omp barrier

	if(DeviceId == 0) printf("\n         Assembly Stiffness Matrix Time: %0.3f s \n", (clock()-time)/CLOCKS_PER_SEC);
	
		
	// ==================================================================================================================================
	// Transpose (TR) Stiffness Matrix:

	// To start time count to transpose stiffness matrix: 
	time = clock();  
		
	dim3 numBlocksTR(3*(RowSize/BlockSizeX), NumThread/BlockSizeY);	// X-Direction 3*96 (3 dof * RowSize)
	dim3 threadsPerBlockTR(BlockSizeX, BlockSizeY);
	
	// Transpose on GPU the Stiffness Matrix

	//Launch_TrasnsposeCoalesced(numBlocksTR, threadsPerBlockTR, NumThread, Matrix_K_Aux_d, Matrix_K_d);

	// Barrier Synchronization  ===================================================================
#pragma omp barrier
	
	if(DeviceId == 0) printf("         Transpose Stiffness Matrix Time: %0.3f s \n", (clock()-time)/CLOCKS_PER_SEC);
		
	// ==================================================================================================================================
	// Evaluating M matrix = 1/K:

	// To start time count to transpose stiffness matrix: 
	time = clock();  
	
	// Number of active terms (DOF x number of nodes)
	// numno = Number of nodes of the mesh  
		
	dim3 numBlocksM(int( sqrt(double(numdof*numno)) /BlockSizeX)+1, int( sqrt(double(numdof*numno)) /BlockSizeY)+1);
	dim3 threadsPerBlockM(BlockSizeX, BlockSizeY);

	Launch_EvaluateMatrixM(numBlocksM, threadsPerBlockM, numdof, numno, iiPosition_d, Matrix_K_Aux_d, Matrix_M_d);

	// Barrier Synchronization  ===================================================================
#pragma omp barrier
		
	if(DeviceId == 0) printf("         Evaluating M Matrix Time: %0.3f s \n", (clock()-time)/CLOCKS_PER_SEC);
			
}

//==============================================================================
//	Preconditioned Conjugate Gradient Method - SOLVER for Multi GPUs

void SolveLinearSystemEquationCPUOpenMP(GPU_Struct *plan, int numdof, int RowSize, 
int GPUmaxThreadsPerBlock_0, int GPUmultiProcessorCount_0, int GPUBlockSizeX_0, int GPUBlockSizeY_0, double *Matrix_0_K_Aux_d, int *Vector_0_I_d, double *Vector_0_B_d, 
double *Vector_0_X_d, double *Matrix_0_M_d, double *Vector_0_R_d, double *Vector_0_D_d, double *Vector_0_D_Global_d, double *Vector_0_Q_d, double *Vector_0_S_d, double *delta_0_new_d, double *delta_0_aux_d, 
double *delta_0_new_V_d, double *delta_0_aux_V_d, double *delta_0_old_V_d, double *delta_0_new_V_h, double *delta_0_GPU_d, double *Result_0_d, double *Vector_0_X_Global_d,

int GPUmaxThreadsPerBlock_1, int GPUmultiProcessorCount_1, int GPUBlockSizeX_1, int GPUBlockSizeY_1, double *Matrix_1_K_Aux_d, int *Vector_1_I_d, double *Vector_1_B_d,
double *Vector_1_X_d, double *Matrix_1_M_d, double *Vector_1_R_d, double *Vector_1_D_d, double *Vector_1_D_Global_d, double *Vector_1_Q_d, double *Vector_1_S_d, double *delta_1_new_d, double *delta_1_aux_d, 
double *delta_1_new_V_d, double *delta_1_aux_V_d, double *delta_1_old_V_d, double *delta_1_new_V_h, double *delta_1_GPU_d, double *Result_1_d, double *Vector_1_X_Global_d,

int GPUmaxThreadsPerBlock_2, int GPUmultiProcessorCount_2, int GPUBlockSizeX_2, int GPUBlockSizeY_2, double *Matrix_2_K_Aux_d, int *Vector_2_I_d, double *Vector_2_B_d,
double *Vector_2_X_d, double *Matrix_2_M_d, double *Vector_2_R_d, double *Vector_2_D_d, double *Vector_2_D_Global_d, double *Vector_2_Q_d, double *Vector_2_S_d, double *delta_2_new_d, double *delta_2_aux_d, 
double *delta_2_new_V_d, double *delta_2_aux_V_d, double *delta_2_old_V_d, double *delta_2_new_V_h, double *delta_2_GPU_d, double *Result_2_d, double *Vector_2_X_Global_d,

int GPUmaxThreadsPerBlock_3, int GPUmultiProcessorCount_3, int GPUBlockSizeX_3, int GPUBlockSizeY_3, double *Matrix_3_K_Aux_d, int *Vector_3_I_d, double *Vector_3_B_d,
double *Vector_3_X_d, double *Matrix_3_M_d, double *Vector_3_R_d, double *Vector_3_D_d, double *Vector_3_D_Global_d, double *Vector_3_Q_d, double *Vector_3_S_d, double *delta_3_new_d, double *delta_3_aux_d, 
double *delta_3_new_V_d, double *delta_3_aux_V_d, double *delta_3_old_V_d, double *delta_3_new_V_h, double *delta_3_GPU_d, double *Result_3_d, double *Vector_3_X_Global_d)
{

	int id = plan->DeviceId;

	switch(plan->DeviceId) {
				
		case 0:  // Run device 0

			SolveLinearSystemEquationGPU(plan->DeviceId, plan->Localnumno, numdof, GPUmaxThreadsPerBlock_0, GPUmultiProcessorCount_0, RowSize, GPUBlockSizeX_0, GPUBlockSizeY_0, 
			Matrix_0_K_Aux_d, Vector_0_I_d, Vector_0_B_d, Matrix_0_M_d, Vector_0_R_d, Vector_0_D_Global_d, Vector_1_D_Global_d, Vector_2_D_Global_d, Vector_3_D_Global_d,
			Vector_0_Q_d, Vector_0_S_d, delta_0_new_d, delta_0_aux_d, delta_0_new_V_d, delta_0_aux_V_d, delta_0_old_V_d, delta_0_new_V_h, delta_0_GPU_d, 
			plan->Localnumno_0, plan->Localnumno_1, plan->Localnumno_2, plan->Localnumno_3,
			Result_0_d, Result_1_d, Result_2_d, Result_3_d, Vector_0_D_d, Vector_1_D_d, Vector_2_D_d, Vector_3_D_d, Vector_0_X_Global_d,
			Vector_0_X_d, Vector_1_X_d, Vector_2_X_d, Vector_3_X_d);
			
			break;
		
		case 1:  // Run device 1

			SolveLinearSystemEquationGPU(plan->DeviceId, plan->Localnumno, numdof, GPUmaxThreadsPerBlock_1, GPUmultiProcessorCount_1, RowSize, GPUBlockSizeX_1, GPUBlockSizeY_1, 
			Matrix_1_K_Aux_d, Vector_1_I_d, Vector_1_B_d, Matrix_1_M_d, Vector_1_R_d, Vector_0_D_Global_d, Vector_1_D_Global_d, Vector_2_D_Global_d, Vector_3_D_Global_d,
			Vector_1_Q_d, Vector_1_S_d, delta_1_new_d, delta_1_aux_d, delta_1_new_V_d, delta_1_aux_V_d, delta_1_old_V_d, delta_1_new_V_h, delta_1_GPU_d, 
			plan->Localnumno_0, plan->Localnumno_1, plan->Localnumno_2, plan->Localnumno_3,
			Result_0_d, Result_1_d, Result_2_d, Result_3_d, Vector_0_D_d, Vector_1_D_d, Vector_2_D_d, Vector_3_D_d, Vector_1_X_Global_d,
			Vector_0_X_d, Vector_1_X_d, Vector_2_X_d, Vector_3_X_d);
		
			break;
			
		case 2:  // Run device 2

			SolveLinearSystemEquationGPU(plan->DeviceId, plan->Localnumno, numdof, GPUmaxThreadsPerBlock_2, GPUmultiProcessorCount_2, RowSize, GPUBlockSizeX_2, GPUBlockSizeY_2, 
			Matrix_2_K_Aux_d, Vector_2_I_d, Vector_2_B_d, Matrix_2_M_d, Vector_2_R_d, Vector_0_D_Global_d, Vector_1_D_Global_d, Vector_2_D_Global_d, Vector_3_D_Global_d,
			Vector_2_Q_d, Vector_2_S_d, delta_2_new_d, delta_2_aux_d, delta_2_new_V_d, delta_2_aux_V_d, delta_2_old_V_d, delta_2_new_V_h, delta_2_GPU_d, 
			plan->Localnumno_0, plan->Localnumno_1, plan->Localnumno_2, plan->Localnumno_3,
			Result_0_d, Result_1_d, Result_2_d, Result_3_d, Vector_0_D_d, Vector_1_D_d, Vector_2_D_d, Vector_3_D_d, Vector_2_X_Global_d,
			Vector_0_X_d, Vector_1_X_d, Vector_2_X_d, Vector_3_X_d);
			
			break;
			
		case 3:  // Run device 3

			SolveLinearSystemEquationGPU(plan->DeviceId, plan->Localnumno, numdof, GPUmaxThreadsPerBlock_3, GPUmultiProcessorCount_3, RowSize, GPUBlockSizeX_3, GPUBlockSizeY_3, 
			Matrix_3_K_Aux_d, Vector_3_I_d, Vector_3_B_d, Matrix_3_M_d, Vector_3_R_d, Vector_0_D_Global_d, Vector_1_D_Global_d, Vector_2_D_Global_d, Vector_3_D_Global_d,
			Vector_3_Q_d, Vector_3_S_d, delta_3_new_d, delta_3_aux_d, delta_3_new_V_d, delta_3_aux_V_d, delta_3_old_V_d, delta_3_new_V_h, delta_3_GPU_d, 
			plan->Localnumno_0, plan->Localnumno_1, plan->Localnumno_2, plan->Localnumno_3,
			Result_0_d, Result_1_d, Result_2_d, Result_3_d, Vector_0_D_d, Vector_1_D_d, Vector_2_D_d, Vector_3_D_d, Vector_3_X_Global_d,
			Vector_0_X_d, Vector_1_X_d, Vector_2_X_d, Vector_3_X_d);

			break;  

	}

}

//==============================================================================
//	Preconditioned Conjugate Gradient Method - SOLVER

void SolveLinearSystemEquationGPU(int DeviceId, int numno, int numdof, int NumMaxThreadsBlock, int NumMultiProcessor, int RowSize, int BlockSizeX, int BlockSizeY,
double *Matrix_A_d, int *Vector_I_d, double *Vector_B_d, double *Matrix_M_d, double *Vector_R_d, 
double *Vector_0_D_Global_d, double *Vector_1_D_Global_d, double *Vector_2_D_Global_d, double *Vector_3_D_Global_d,
double *Vector_Q_d, double *Vector_S_d, double *delta_new_d, double *delta_aux_d, double *delta_new_V_d, double *delta_aux_V_d,
double *delta_old_V_d, double *delta_new_V_h, double *delta_GPU_d,
int numno_0, int numno_1, int numno_2, int numno_3,
double *Result_0_d, double *Result_1_d, double *Result_2_d, double *Result_3_d, double *Vector_0_D_d, double *Vector_1_D_d, double *Vector_2_D_d, double *Vector_3_D_d,
double *Vector_X_Global_d, double *Vector_0_X_d, double *Vector_1_X_d, double *Vector_2_X_d, double *Vector_3_X_d)
{ 

	int cont, BlockSizeNumMultiProc, numnoglobal;
	double delta_new;
	double time;
	
	using namespace std;

	// ==================================================================================================================================
	// Sets device as the current device

	cudaSetDevice(DeviceId);         

	// ==================================================================================================================================
	// Solver on GPU of Linear System of Equations - Conjugate Gradient Method
	
	// Number of blocks needs to sum the terms of vector:
	if(NumMultiProcessor <= 4)       BlockSizeNumMultiProc =  4;
	else if(NumMultiProcessor <=  8) BlockSizeNumMultiProc =  8;
	else if(NumMultiProcessor <= 16) BlockSizeNumMultiProc = 16;
	else if(NumMultiProcessor <= 32) BlockSizeNumMultiProc = 32;

	BlockSizeNumMultiProc = 16;
	
	// **************  Evaluating Block and Thread dimesion:  **************
	
	// Block size for local vector operation
	dim3 BlockSizeVector(int(sqrt(double(numdof*numno))/BlockSizeX)+1, int(sqrt(double(numdof*numno))/BlockSizeY)+1);	
	dim3 ThreadsPerBlockXY(BlockSizeX, BlockSizeY);

	// Block size for global vector operation
	numnoglobal = numno_0 + numno_1 + numno_2 + numno_3;
	dim3 BlockSizeVectorGlobal(int(sqrt(double(numdof*numnoglobal))/BlockSizeX)+1, int(sqrt(double(numdof*numnoglobal))/BlockSizeY)+1);	
	
	// Block size for local matrix operation
	dim3 BlockSizeMatrix(int(sqrt(double(numdof*numno*RowSize))/BlockSizeX)+1, int(sqrt(double(numdof*numno*RowSize))/BlockSizeY)+1);	
	
	// **************  This Kernel Subtracts a Vector by a Vector:  **************
		
	Launch_SUBT_V_V(BlockSizeVector, ThreadsPerBlockXY, numdof, numno, Vector_B_d, Vector_R_d); // =======================> {r} = {b} - [A]{x}
		
	// **************  This Kernel Multiplies a Vector by a other Vector:  **************
	
	// Minimum vector size is: NumMultiProcessor * 2 * blocksize

	switch(DeviceId) {

		case 0:  // Run device 0
			// **************  This Kernel Multiplies a Diagonal Matrix by a Vector:  **************
			Launch_MULT_DM_V(BlockSizeVector, ThreadsPerBlockXY, numdof, numno, Matrix_M_d, Vector_R_d, Vector_0_D_d); // =======================> {d} = [M]-1{r}

			Launch_MULT_V_V(BlockSizeNumMultiProc, NumMaxThreadsBlock, numdof, numno, Vector_R_d, Vector_0_D_d, delta_new_d); // =======================> delta_new = {r}T{d}
			// **************  This Kernel add up the tems of the delta_new_d vector:  **************
			Launch_SUM_V(BlockSizeVector, ThreadsPerBlockXY, BlockSizeNumMultiProc, numdof, numno, 0, delta_new_d, delta_GPU_d, Result_0_d);  // =======================> {delta_new_V_d} = {r}T{d}
			break;

		case 1:  // Run device 1
			// **************  This Kernel Multiplies a Diagonal Matrix by a Vector:  **************
			Launch_MULT_DM_V(BlockSizeVector, ThreadsPerBlockXY, numdof, numno, Matrix_M_d, Vector_R_d, Vector_1_D_d); // =======================> {d} = [M]-1{r}

			Launch_MULT_V_V(BlockSizeNumMultiProc, NumMaxThreadsBlock, numdof, numno, Vector_R_d, Vector_1_D_d, delta_new_d); // =======================> delta_new = {r}T{d}
			// **************  This Kernel add up the tems of the delta_new_d vector:  **************
			Launch_SUM_V(BlockSizeVector, ThreadsPerBlockXY, BlockSizeNumMultiProc, numdof, numno, 1, delta_new_d, delta_GPU_d, Result_1_d);  // =======================> {delta_new_V_d} = {r}T{d} 
			break;

		case 2:  // Run device 2
			// **************  This Kernel Multiplies a Diagonal Matrix by a Vector:  **************
			Launch_MULT_DM_V(BlockSizeVector, ThreadsPerBlockXY, numdof, numno, Matrix_M_d, Vector_R_d, Vector_2_D_d); // =======================> {d} = [M]-1{r}

			Launch_MULT_V_V(BlockSizeNumMultiProc, NumMaxThreadsBlock, numdof, numno, Vector_R_d, Vector_2_D_d, delta_new_d); // =======================> delta_new = {r}T{d}
			// **************  This Kernel add up the tems of the delta_new_d vector:  **************
			Launch_SUM_V(BlockSizeVector, ThreadsPerBlockXY, BlockSizeNumMultiProc, numdof, numno, 2, delta_new_d, delta_GPU_d, Result_2_d);  // =======================> {delta_new_V_d} = {r}T{d} 
			break;

		case 3:  // Run device 3
			// **************  This Kernel Multiplies a Diagonal Matrix by a Vector:  **************
			Launch_MULT_DM_V(BlockSizeVector, ThreadsPerBlockXY, numdof, numno, Matrix_M_d, Vector_R_d, Vector_3_D_d); // =======================> {d} = [M]-1{r}

			Launch_MULT_V_V(BlockSizeNumMultiProc, NumMaxThreadsBlock, numdof, numno, Vector_R_d, Vector_3_D_d, delta_new_d); // =======================> delta_new = {r}T{d}
			// **************  This Kernel add up the tems of the delta_new_d vector:  **************
			Launch_SUM_V(BlockSizeVector, ThreadsPerBlockXY, BlockSizeNumMultiProc, numdof, numno, 3, delta_new_d, delta_GPU_d, Result_3_d);  // =======================> {delta_new_V_d} = {r}T{d}
			break;

	}

	// Barrier Synchronization  ===================================================================
#pragma omp barrier

	// **************  Copies memory from one device to memory on another device:  **************

	if(DeviceId == 0)
		cudaMemcpyPeer(delta_GPU_d+1, 0, Result_1_d, 1, sizeof(double)); 
	if(DeviceId == 2)
		cudaMemcpyPeer(delta_GPU_d+3, 2, Result_3_d, 3, sizeof(double));

	#pragma omp barrier

	if(DeviceId == 0)
		cudaMemcpyPeer(delta_GPU_d+2, 0, Result_2_d, 2, sizeof(double));
	if(DeviceId == 1)
		cudaMemcpyPeer(delta_GPU_d+3, 1, Result_3_d, 3, sizeof(double));

	#pragma omp barrier

	if(DeviceId == 0)
		cudaMemcpyPeer(delta_GPU_d+3, 0, Result_3_d, 3, sizeof(double)); 
	if(DeviceId == 1)
		cudaMemcpyPeer(delta_GPU_d+2, 1, Result_2_d, 2, sizeof(double));
		
	#pragma omp barrier

	if(DeviceId == 1) 
		cudaMemcpyPeer(delta_GPU_d+0, 1, Result_0_d, 0, sizeof(double));  
	if(DeviceId == 3)
		cudaMemcpyPeer(delta_GPU_d+2, 3, Result_2_d, 2, sizeof(double));
		
	#pragma omp barrier

	if(DeviceId == 2)
		cudaMemcpyPeer(delta_GPU_d+0, 2, Result_0_d, 0, sizeof(double));
	if(DeviceId == 3)
		cudaMemcpyPeer(delta_GPU_d+1, 3, Result_1_d, 1, sizeof(double));

	#pragma omp barrier

	if(DeviceId == 2)
		cudaMemcpyPeer(delta_GPU_d+1, 2, Result_1_d, 1, sizeof(double));
	if(DeviceId == 3)
		cudaMemcpyPeer(delta_GPU_d+0, 3, Result_0_d, 0, sizeof(double));

		// Barrier Synchronization  ===================================================================
#pragma omp barrier

	// **************  Adding delta_3_new_V_d [0]+[1]+[2]+[3] terms:  **************

	Launch_SUM_GPU_4(BlockSizeVector, ThreadsPerBlockXY, numdof, numno, delta_GPU_d, delta_new_V_d); 

	// Barrier Synchronization  ===================================================================
#pragma omp barrier

	cudaMemcpy(delta_new_V_h, delta_new_V_d, sizeof(double)*numdof*numno, cudaMemcpyDeviceToHost);
		
	delta_new = delta_new_V_h[0];

	// err = cur_err
	double epsilon  = 0.001;
	double err = (double)(delta_new * epsilon * epsilon) ;

	// ************************************************************************************

	cont = 0;

	if(DeviceId == 0) {  // Only device 0 print message

		printf("\n");
		printf("         Solving Linear Equation System on MultiGPU algorithm \n");
		printf("         ======================================================= \n");
		printf("         * Conjugate Gradient Method * \n\n");

	}

	// *************************************************************************************************************************************************************************************************

	// Barrier Synchronization  ===================================================================
#pragma omp barrier

	time = clock();  // Time count

	while(delta_new > err && cont < 4000) {  

		// **************  This Kernel copy data from Vector_D_d to Vector_D_Global_d:  **************

		switch(DeviceId) {

			case 0:  // Run device 0
				Launch_Copy_Local_Global(BlockSizeVector, ThreadsPerBlockXY, numdof, numno, 0,                                Vector_0_D_d, Vector_0_D_Global_d); 
				break;

			case 1:  // Run device 1
				Launch_Copy_Local_Global(BlockSizeVector, ThreadsPerBlockXY, numdof, numno, numdof*numno_0,                   Vector_1_D_d, Vector_1_D_Global_d); 
				break;

			case 2:  // Run device 2
				Launch_Copy_Local_Global(BlockSizeVector, ThreadsPerBlockXY, numdof, numno, numdof*(numno_0+numno_1),         Vector_2_D_d, Vector_2_D_Global_d); 
				break;

			case 3:  // Run device 3
				Launch_Copy_Local_Global(BlockSizeVector, ThreadsPerBlockXY, numdof, numno, numdof*(numno_0+numno_1+numno_2), Vector_3_D_d, Vector_3_D_Global_d); 
				break;

		}

		// Barrier Synchronization  ===================================================================
#pragma omp barrier

		// **************   Assembling global vector Vector_D_Global_d from each GPU information:  **************

		if(DeviceId == 0)
			cudaMemcpyPeer(Vector_0_D_Global_d+numdof*(numno_0),                 0, Vector_1_D_d, 1, numdof*numno_1*sizeof(double));  
		if(DeviceId == 2)
			cudaMemcpyPeer(Vector_2_D_Global_d+numdof*(numno_0+numno_1+numno_2), 2, Vector_3_D_d, 3, numdof*numno_3*sizeof(double));

		#pragma omp barrier

		if(DeviceId == 0)
			cudaMemcpyPeer(Vector_0_D_Global_d+numdof*(numno_0+numno_1),         0, Vector_2_D_d, 2, numdof*numno_2*sizeof(double)); 
		if(DeviceId == 1)
			cudaMemcpyPeer(Vector_1_D_Global_d+numdof*(numno_0+numno_1+numno_2), 1, Vector_3_D_d, 3, numdof*numno_3*sizeof(double)); 

		#pragma omp barrier

		if(DeviceId == 0)
			cudaMemcpyPeer(Vector_0_D_Global_d+numdof*(numno_0+numno_1+numno_2), 0, Vector_3_D_d, 3, numdof*numno_3*sizeof(double)); 
		if(DeviceId == 1)
			cudaMemcpyPeer(Vector_1_D_Global_d+numdof*(numno_0+numno_1),         1, Vector_2_D_d, 2, numdof*numno_2*sizeof(double)); 
		
		#pragma omp barrier

		if(DeviceId == 1) 
			cudaMemcpyPeer(Vector_1_D_Global_d,                                  1, Vector_0_D_d, 0, numdof*numno_0*sizeof(double));  // (dst, dstDevice, src, srcDevice, count)
		if(DeviceId == 3)
			cudaMemcpyPeer(Vector_3_D_Global_d+numdof*(numno_0+numno_1),         3, Vector_2_D_d, 2, numdof*numno_2*sizeof(double));

		#pragma omp barrier	

		if(DeviceId == 2)
			cudaMemcpyPeer(Vector_2_D_Global_d,                                  2, Vector_0_D_d, 0, numdof*numno_0*sizeof(double));
		if(DeviceId == 3)
			cudaMemcpyPeer(Vector_3_D_Global_d+numdof*(numno_0),                 3, Vector_1_D_d, 1, numdof*numno_1*sizeof(double)); 

		#pragma omp barrier	

		if(DeviceId == 2)
			cudaMemcpyPeer(Vector_2_D_Global_d+numdof*(numno_0),                 2, Vector_1_D_d, 1, numdof*numno_1*sizeof(double)); 
		if(DeviceId == 3)
			cudaMemcpyPeer(Vector_3_D_Global_d,                                  3, Vector_0_D_d, 0, numdof*numno_0*sizeof(double)); 

		// Barrier Synchronization  ===================================================================
#pragma omp barrier

		// **************   Multiplying matrix A by global vector Vector_D_Global_d:  **************

		switch(DeviceId) {  

			case 0:  // Run device 0
				Launch_MULT_SM_V_128(BlockSizeMatrix, ThreadsPerBlockXY, NumMaxThreadsBlock, numdof, numno, RowSize, Matrix_A_d, Vector_0_D_Global_d, Vector_I_d, Vector_Q_d); // ======> {q} = [A]{d}
				break;

			case 1:  // Run device 1
				Launch_MULT_SM_V_128(BlockSizeMatrix, ThreadsPerBlockXY, NumMaxThreadsBlock, numdof, numno, RowSize, Matrix_A_d, Vector_1_D_Global_d, Vector_I_d, Vector_Q_d); // ======> {q} = [A]{d}
				break;

			case 2:  // Run device 2
				Launch_MULT_SM_V_128(BlockSizeMatrix, ThreadsPerBlockXY, NumMaxThreadsBlock, numdof, numno, RowSize, Matrix_A_d, Vector_2_D_Global_d, Vector_I_d, Vector_Q_d); // ======> {q} = [A]{d}
				break;

			case 3:  // Run device 3
				Launch_MULT_SM_V_128(BlockSizeMatrix, ThreadsPerBlockXY, NumMaxThreadsBlock, numdof, numno, RowSize, Matrix_A_d, Vector_3_D_Global_d, Vector_I_d, Vector_Q_d); // ======> {q} = [A]{d}
				break;

		}

		// Barrier Synchronization  ===================================================================
#pragma omp barrier

		// **************   This Kernel Multiplies a Vector by a other Vector:  **************

		switch(DeviceId) {  

			case 0:  // Run device 0
				Launch_MULT_V_V(BlockSizeNumMultiProc, NumMaxThreadsBlock, numdof, numno, Vector_0_D_d, Vector_Q_d, delta_aux_d); // =======================> {delta_aux_V_d} = {d}T{q}
				// **************  This Kernel add up the tems of the delta_new_d vector:  **************
				Launch_SUM_V(BlockSizeVector, ThreadsPerBlockXY, BlockSizeNumMultiProc, numdof, numno, 0, delta_aux_d, delta_GPU_d, Result_0_d);  // =======================> {delta_new_V_d} = {r}T{d}
				break;

			case 1:  // Run device 1
				Launch_MULT_V_V(BlockSizeNumMultiProc, NumMaxThreadsBlock, numdof, numno, Vector_1_D_d, Vector_Q_d, delta_aux_d); // =======================> {delta_aux_V_d} = {d}T{q}
				// **************  This Kernel add up the tems of the delta_new_d vector:  **************
				Launch_SUM_V(BlockSizeVector, ThreadsPerBlockXY, BlockSizeNumMultiProc, numdof, numno, 1, delta_aux_d, delta_GPU_d, Result_1_d);  // =======================> {delta_new_V_d} = {r}T{d} 
				break;

			case 2:  // Run device 2
				Launch_MULT_V_V(BlockSizeNumMultiProc, NumMaxThreadsBlock, numdof, numno, Vector_2_D_d, Vector_Q_d, delta_aux_d); // =======================> {delta_aux_V_d} = {d}T{q}
				// **************  This Kernel add up the tems of the delta_new_d vector:  **************
				Launch_SUM_V(BlockSizeVector, ThreadsPerBlockXY, BlockSizeNumMultiProc, numdof, numno, 2, delta_aux_d, delta_GPU_d, Result_2_d);  // =======================> {delta_new_V_d} = {r}T{d} 
				break;

			case 3:  // Run device 3
				Launch_MULT_V_V(BlockSizeNumMultiProc, NumMaxThreadsBlock, numdof, numno, Vector_3_D_d, Vector_Q_d, delta_aux_d); // =======================> {delta_aux_V_d} = {d}T{q}
				// **************  This Kernel add up the tems of the delta_new_d vector:  **************
				Launch_SUM_V(BlockSizeVector, ThreadsPerBlockXY, BlockSizeNumMultiProc, numdof, numno, 3, delta_aux_d, delta_GPU_d, Result_3_d);  // =======================> {delta_new_V_d} = {r}T{d}
				break;

		}

		// Barrier Synchronization  ===================================================================
#pragma omp barrier

	// **************  Copies memory from one device to memory on another device:  **************

		if(DeviceId == 0) {  // Run device 0
			cudaMemcpyPeer(delta_GPU_d+1, 0, Result_1_d, 1, sizeof(double));  // (dst, dstDevice, src, srcDevice, count)
			cudaMemcpyPeer(delta_GPU_d+2, 0, Result_2_d, 2, sizeof(double));
			cudaMemcpyPeer(delta_GPU_d+3, 0, Result_3_d, 3, sizeof(double)); 
		}

		// Barrier Synchronization  ===================================================================
#pragma omp barrier

		if(DeviceId == 1) {  // Run device 1
			cudaMemcpyPeer(delta_GPU_d+0, 1, Result_0_d, 0, sizeof(double));  // (dst, dstDevice, src, srcDevice, count)
			cudaMemcpyPeer(delta_GPU_d+2, 1, Result_2_d, 2, sizeof(double));
			cudaMemcpyPeer(delta_GPU_d+3, 1, Result_3_d, 3, sizeof(double));
		}

		// Barrier Synchronization  ===================================================================
#pragma omp barrier

		if(DeviceId == 2) {  // Run device 2
			cudaMemcpyPeer(delta_GPU_d+0, 2, Result_0_d, 0, sizeof(double));  // (dst, dstDevice, src, srcDevice, count)
			cudaMemcpyPeer(delta_GPU_d+1, 2, Result_1_d, 1, sizeof(double));
			cudaMemcpyPeer(delta_GPU_d+3, 2, Result_3_d, 3, sizeof(double));  
		}

		// Barrier Synchronization  ===================================================================
#pragma omp barrier

		if(DeviceId == 3) {  // Run device 3
			cudaMemcpyPeer(delta_GPU_d+0, 3, Result_0_d, 0, sizeof(double));  // (dst, dstDevice, src, srcDevice, count)
			cudaMemcpyPeer(delta_GPU_d+1, 3, Result_1_d, 1, sizeof(double));
			cudaMemcpyPeer(delta_GPU_d+2, 3, Result_2_d, 2, sizeof(double));  
		}
		
	// Barrier Synchronization  ===================================================================
#pragma omp barrier

	// **************  Adding delta_3_new_V_d [0]+[1]+[2]+[3] terms:  **************

		Launch_SUM_GPU_4(BlockSizeVector, ThreadsPerBlockXY, numdof, numno, delta_GPU_d, delta_aux_V_d);

	// Barrier Synchronization  ===================================================================
#pragma omp barrier

		// **************  This Kernel adds up a Vector by a alfa*Vector:  **************
										                                                                                                                                                        	
		switch(DeviceId) {  

			case 0:  // Run device 0
				Launch_ADD_V_V_X (BlockSizeVector, ThreadsPerBlockXY, numdof, numno, Vector_0_X_d, Vector_0_D_d, delta_new_V_d, delta_aux_V_d, Vector_R_d, Vector_Q_d); // =============> {x} = {x} + alfa{d}                                                                                                                                                    // =============> {r} = {r} - alfa{q}
				break;

			case 1:  // Run device 1
				Launch_ADD_V_V_X (BlockSizeVector, ThreadsPerBlockXY, numdof, numno, Vector_1_X_d, Vector_1_D_d, delta_new_V_d, delta_aux_V_d, Vector_R_d, Vector_Q_d); // =============> {x} = {x} + alfa{d}                                                                                                                                                              // =============> {r} = {r} - alfa{q}
				break;

			case 2:  // Run device 2
				Launch_ADD_V_V_X (BlockSizeVector, ThreadsPerBlockXY, numdof, numno, Vector_2_X_d, Vector_2_D_d, delta_new_V_d, delta_aux_V_d, Vector_R_d, Vector_Q_d); // =============> {x} = {x} + alfa{d}	                                                                                                                                                              // =============> {r} = {r} - alfa{q}
				break;

			case 3:  // Run device 3
				Launch_ADD_V_V_X (BlockSizeVector, ThreadsPerBlockXY, numdof, numno, Vector_3_X_d, Vector_3_D_d, delta_new_V_d, delta_aux_V_d, Vector_R_d, Vector_Q_d); // =============> {x} = {x} + alfa{d}                                                                                                                                                            // =============> {r} = {r} - alfa{q}
				break;

		}

		// **************  This Kernel Multiplies a Diagonal Matrix by a Vector:  **************

		Launch_MULT_DM_V(BlockSizeVector, ThreadsPerBlockXY, numdof, numno, Matrix_M_d, Vector_R_d, Vector_S_d); // =======================> {s} = [M]-1{r}

		// **************  Storing delta_new in delta_old:  **************
				
		Launch_UPDATE_DELTA(BlockSizeVector, ThreadsPerBlockXY, numdof, numno, delta_new_V_d, delta_old_V_d);  // =======================> {delta_old} = {delta_new}
		
		// **************  This Kernel Multiplies a Vector by a other Vector:  **************
								
		Launch_MULT_V_V(BlockSizeNumMultiProc, NumMaxThreadsBlock, numdof, numno, Vector_R_d, Vector_S_d, delta_new_d); // =======================> {delta_new} = {r}T{s}

		// **************  This Kernel add up the tems of the delta_new_d vector:  **************

		switch(DeviceId) {  

			case 0:  // Run device 0
				Launch_SUM_V(BlockSizeVector, ThreadsPerBlockXY, BlockSizeNumMultiProc, numdof, numno, 0, delta_new_d, delta_GPU_d, Result_0_d);  // =======================> {delta_new_V_d} = {r}T{d}
				break;

			case 1:  // Run device 1
				Launch_SUM_V(BlockSizeVector, ThreadsPerBlockXY, BlockSizeNumMultiProc, numdof, numno, 1, delta_new_d, delta_GPU_d, Result_1_d);  // =======================> {delta_new_V_d} = {r}T{d} 
				break;

			case 2:  // Run device 2
				Launch_SUM_V(BlockSizeVector, ThreadsPerBlockXY, BlockSizeNumMultiProc, numdof, numno, 2, delta_new_d, delta_GPU_d, Result_2_d);  // =======================> {delta_new_V_d} = {r}T{d} 
				break;

			case 3:  // Run device 3
				Launch_SUM_V(BlockSizeVector, ThreadsPerBlockXY, BlockSizeNumMultiProc, numdof, numno, 3, delta_new_d, delta_GPU_d, Result_3_d);  // =======================> {delta_new_V_d} = {r}T{d}
				break;

		}

		// Barrier Synchronization  ===================================================================
#pragma omp barrier

		// **************  Copies memory from one device to memory on another device:  **************

		if(DeviceId == 0) {  // Run device 0
			cudaMemcpyPeer(delta_GPU_d+1, 0, Result_1_d, 1, sizeof(double));  // (dst, dstDevice, src, srcDevice, count)
			cudaMemcpyPeer(delta_GPU_d+2, 0, Result_2_d, 2, sizeof(double));
			cudaMemcpyPeer(delta_GPU_d+3, 0, Result_3_d, 3, sizeof(double)); 
		}

		// Barrier Synchronization  ===================================================================
#pragma omp barrier

		if(DeviceId == 1) {  // Run device 1
			cudaMemcpyPeer(delta_GPU_d+0, 1, Result_0_d, 0, sizeof(double));  // (dst, dstDevice, src, srcDevice, count)
			cudaMemcpyPeer(delta_GPU_d+2, 1, Result_2_d, 2, sizeof(double));
			cudaMemcpyPeer(delta_GPU_d+3, 1, Result_3_d, 3, sizeof(double));
		}

		// Barrier Synchronization  ===================================================================
#pragma omp barrier

		if(DeviceId == 2) {  // Run device 2
			cudaMemcpyPeer(delta_GPU_d+0, 2, Result_0_d, 0, sizeof(double));  // (dst, dstDevice, src, srcDevice, count)
			cudaMemcpyPeer(delta_GPU_d+1, 2, Result_1_d, 1, sizeof(double));
			cudaMemcpyPeer(delta_GPU_d+3, 2, Result_3_d, 3, sizeof(double));  
		}

		// Barrier Synchronization  ===================================================================
#pragma omp barrier

		if(DeviceId == 3) {  // Run device 3
			cudaMemcpyPeer(delta_GPU_d+0, 3, Result_0_d, 0, sizeof(double));  // (dst, dstDevice, src, srcDevice, count)
			cudaMemcpyPeer(delta_GPU_d+1, 3, Result_1_d, 1, sizeof(double));
			cudaMemcpyPeer(delta_GPU_d+2, 3, Result_2_d, 2, sizeof(double));  
		}

		// Barrier Synchronization  ===================================================================
#pragma omp barrier

		// **************  Adding delta_new_V_d [0]+[1]+[2]+[3] terms:  **************

		Launch_SUM_GPU_4(BlockSizeVector, ThreadsPerBlockXY, numdof, numno, delta_GPU_d, delta_new_V_d); 

		// Barrier Synchronization  ===================================================================
#pragma omp barrier
	    
	    int inc = 1;
		
		if((cont - int(cont/inc)*inc) == 0) {
			cudaMemcpy(delta_new_V_h, delta_new_V_d, sizeof(double)*1, cudaMemcpyDeviceToHost);
			delta_new = delta_new_V_h[0];
		}

		// Barrier Synchronization  ===================================================================
#pragma omp barrier
				
		// **************  This Kernel adds up a Vector by a beta*Vector:  **************

		switch(DeviceId) {

			case 0:  // Run device 0

				Launch_ADD_V_V_D(BlockSizeVector, ThreadsPerBlockXY, numdof, numno, Vector_S_d, Vector_0_D_d, delta_new_V_d, delta_old_V_d); // =======================> {d} = {s} + beta{d}

				break;

			case 1:  // Run device 1

				Launch_ADD_V_V_D(BlockSizeVector, ThreadsPerBlockXY, numdof, numno, Vector_S_d, Vector_1_D_d, delta_new_V_d, delta_old_V_d); // =======================> {d} = {s} + beta{d}

				break;

			case 2:  // Run device 2

				Launch_ADD_V_V_D(BlockSizeVector, ThreadsPerBlockXY, numdof, numno, Vector_S_d, Vector_2_D_d, delta_new_V_d, delta_old_V_d); // =======================> {d} = {s} + beta{d}

				break;

			case 3:  // Run device 3

				Launch_ADD_V_V_D(BlockSizeVector, ThreadsPerBlockXY, numdof, numno, Vector_S_d, Vector_3_D_d, delta_new_V_d, delta_old_V_d); // =======================> {d} = {s} + beta{d}

				break;

		}
								
		// Barrier Synchronization  ===================================================================
#pragma omp barrier
										
		cont++;

		/*if(DeviceId == 0) {
			printf("\n");
			printf("         Iteration Number : %d \n", cont+1);
		}*/

		// ******************************************************************************************************************************************************
			
	}

	// *************************************************************************************************************************************************************************************************

	if(DeviceId == 0) {
	
		time = clock()-time;
		printf("\n");
		printf("         Time Execution : %0.3f s \n", time/CLOCKS_PER_SEC);
		
		printf("\n");	
		printf("         Iteration Number = %i  \n", cont);
		printf("         Error = %0.7f  \n", delta_new);

	}

	// **************  Assembling global displacement vector from each GPU:  **************

	// Barrier Synchronization  ===================================================================
#pragma omp barrier

	switch(DeviceId) {

			case 0:  // Run device 0
				Launch_Copy_Local_Global(BlockSizeVector, ThreadsPerBlockXY, numdof, numno, 0,                                Vector_0_X_d, Vector_X_Global_d); 
				break;

			case 1:  // Run device 1
				Launch_Copy_Local_Global(BlockSizeVector, ThreadsPerBlockXY, numdof, numno, numdof*numno_0,                   Vector_1_X_d, Vector_X_Global_d); 
				break;

			case 2:  // Run device 2
				Launch_Copy_Local_Global(BlockSizeVector, ThreadsPerBlockXY, numdof, numno, numdof*(numno_0+numno_1),         Vector_2_X_d, Vector_X_Global_d); 
				break;

			case 3:  // Run device 3
				Launch_Copy_Local_Global(BlockSizeVector, ThreadsPerBlockXY, numdof, numno, numdof*(numno_0+numno_1+numno_2), Vector_3_X_d, Vector_X_Global_d); 
				break;

	}

	// Barrier Synchronization  ===================================================================
#pragma omp barrier

	// **************   Assembling global vector Vector_X_Global_d from each GPU information:  **************

	if(DeviceId == 0) {  // Run device 0
		cudaMemcpyPeer(Vector_X_Global_d+numdof*(numno_0),                 0, Vector_1_X_d, 1, numdof*numno_1*sizeof(double));  // (dst, dstDevice, src, srcDevice, count)
		cudaMemcpyPeer(Vector_X_Global_d+numdof*(numno_0+numno_1),         0, Vector_2_X_d, 2, numdof*numno_2*sizeof(double)); 
		cudaMemcpyPeer(Vector_X_Global_d+numdof*(numno_0+numno_1+numno_2), 0, Vector_3_X_d, 3, numdof*numno_3*sizeof(double)); 
	}

	// Barrier Synchronization  ===================================================================
#pragma omp barrier

	if(DeviceId == 1) {  // Run device 1
		cudaMemcpyPeer(Vector_X_Global_d,                                  1, Vector_0_X_d, 0, numdof*numno_0*sizeof(double));  // (dst, dstDevice, src, srcDevice, count)
		cudaMemcpyPeer(Vector_X_Global_d+numdof*(numno_0+numno_1),         1, Vector_2_X_d, 2, numdof*numno_2*sizeof(double)); 
		cudaMemcpyPeer(Vector_X_Global_d+numdof*(numno_0+numno_1+numno_2), 1, Vector_3_X_d, 3, numdof*numno_3*sizeof(double)); 
	}

	// Barrier Synchronization  ===================================================================
#pragma omp barrier

	if(DeviceId == 2) {  // Run device 2
		cudaMemcpyPeer(Vector_X_Global_d,                                  2, Vector_0_X_d, 0, numdof*numno_0*sizeof(double));  // (dst, dstDevice, src, srcDevice, count)
		cudaMemcpyPeer(Vector_X_Global_d+numdof*(numno_0),                 2, Vector_1_X_d, 1, numdof*numno_1*sizeof(double)); 
		cudaMemcpyPeer(Vector_X_Global_d+numdof*(numno_0+numno_1+numno_2), 2, Vector_3_X_d, 3, numdof*numno_3*sizeof(double)); 
	}

	// Barrier Synchronization  ===================================================================
#pragma omp barrier

	if(DeviceId == 3) {  // Run device 3
		cudaMemcpyPeer(Vector_X_Global_d,                                  3, Vector_0_X_d, 0, numdof*numno_0*sizeof(double));  // (dst, dstDevice, src, srcDevice, count)
		cudaMemcpyPeer(Vector_X_Global_d+numdof*(numno_0),                 3, Vector_1_X_d, 1, numdof*numno_1*sizeof(double)); 
		cudaMemcpyPeer(Vector_X_Global_d+numdof*(numno_0+numno_1),         3, Vector_2_X_d, 2, numdof*numno_2*sizeof(double));
	}

	// Barrier Synchronization  ===================================================================
#pragma omp barrier

}

//==============================================================================
void EvaluateiiPositionMultiGPU(int numdof, int numno, int numno_0, int numno_1, int numno_2, int numno_3, 
int RowSize, int *Vector_I_h, int *iiPosition_h)
{
	int i, j;
	
	for(i=0; i<numdof*numno; i++) {
	
		for(j=0; j<RowSize; j++) {
		
			if(Vector_I_h[i*RowSize + j] == i+1) {
				iiPosition_h[i] = i*RowSize + j;
			} 
		
		}
		
	}

	// -----------------------------------------------------------------------

	// Fix numbering second quarter for Multi GPU

	for(i=numdof*numno_0; i<numdof*(numno_0+numno_1); i++) 
		iiPosition_h[i] -= numdof*numno_0*RowSize;

	// Fix numbering third quarter for Multi GPU

	for(i=numdof*(numno_0+numno_1); i<numdof*(numno_0+numno_1+numno_2); i++) 
		iiPosition_h[i] -= numdof*(numno_0+numno_1)*RowSize;

	// Fix numbering fourty quarter for Multi GPU

	for(i=numdof*(numno_0+numno_1+numno_2); i<numdof*(numno_0+numno_1+numno_2+numno_3); i++) 
		iiPosition_h[i] -= numdof*(numno_0+numno_1+numno_2)*RowSize;

}

//==============================================================================
void EvaluateGPURowNumbering(int imax, int jmax, int kmax, int RowSize, int *Vector_I_h)
{
	int i, j, k, l, size, cont, imaxhalos, jmaxhalos, kmaxhalos;
	int aux1, GlobalPos, LocalPos;
	int *GlobNum;

	imaxhalos=imax+2;
	jmaxhalos=jmax+2;
	kmaxhalos=kmax+2;
	
	// Allocating memory space for host 
	size = (imax+2)*(jmax+2)*(kmax+2)*(sizeof(int));
	GlobNum  = (int *)malloc(size);

	for(i=0; i<(imax+2)*(jmax+2)*(kmax+2); i++)
		GlobNum[i] = 0;

	cont = 1;

	for(k=1; k<kmaxhalos-1; k++) {					     // to go along the layer

		for(j=1; j<jmaxhalos-1; j++) {				     // to go along the column
  
			for(i=1; i<imaxhalos-1; i++) {               // to go along the row

				GlobalPos = imaxhalos*jmaxhalos*k + imaxhalos*j + i;

				GlobNum[GlobalPos] = cont;

				cont++;

			}

		}

	}

	// =========================================================================================================
	
	cont = 0;

	for(k=1; k<kmaxhalos-1; k++) {					       // to go along the layer

		for(j=1; j<jmaxhalos-1; j++) {				       // to go along the column
  
			for(i=1; i<imaxhalos-1; i++) {                 // to go along the line
			
				GlobalPos = imaxhalos*jmaxhalos*k + imaxhalos*j + i;
				
				// =================================================================
				// First Layer:
				LocalPos = 0;  // Local Numbering = 0
				aux1 = GlobalPos-imaxhalos-1-imaxhalos*jmaxhalos;

				if(GlobNum[aux1] > 0) {

					for(l=0; l<3; l++) {

						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos  )] = (3*(GlobNum[aux1]-1)  )+1;
						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos+1)] = (3*(GlobNum[aux1]-1)+1)+1;
						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos+2)] = (3*(GlobNum[aux1]-1)+2)+1;

					}

				}
				
				// Local Numbering = 1
				LocalPos = 1;
				aux1 = GlobalPos-imaxhalos-imaxhalos*jmaxhalos;
				
				if(GlobNum[aux1] > 0) {

					for(l=0; l<3; l++) {

						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos  )] = (3*(GlobNum[aux1]-1)  )+1;
						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos+1)] = (3*(GlobNum[aux1]-1)+1)+1;
						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos+2)] = (3*(GlobNum[aux1]-1)+2)+1;

					}

				}
				
				// Local Numbering = 2
				LocalPos = 2;
				aux1 = GlobalPos-imaxhalos+1-imaxhalos*jmaxhalos;
				
				if(GlobNum[aux1] > 0) {

					for(l=0; l<3; l++) {

						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos  )] = (3*(GlobNum[aux1]-1)  )+1;
						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos+1)] = (3*(GlobNum[aux1]-1)+1)+1;
						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos+2)] = (3*(GlobNum[aux1]-1)+2)+1;

					}

				}
				
				// Local Numbering = 3
				LocalPos = 3;
				aux1 = GlobalPos-1-imaxhalos*jmaxhalos;
				
				if(GlobNum[aux1] > 0) {

					for(l=0; l<3; l++) {

						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos  )] = (3*(GlobNum[aux1]-1)  )+1;
						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos+1)] = (3*(GlobNum[aux1]-1)+1)+1;
						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos+2)] = (3*(GlobNum[aux1]-1)+2)+1;

					}

				}
				
				// Local Numbering = 4
				LocalPos = 4;
				aux1 = GlobalPos-imaxhalos*jmaxhalos;
				
				if(GlobNum[aux1] > 0) {

					for(l=0; l<3; l++) {

						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos  )] = (3*(GlobNum[aux1]-1)  )+1;
						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos+1)] = (3*(GlobNum[aux1]-1)+1)+1;
						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos+2)] = (3*(GlobNum[aux1]-1)+2)+1;

					}

				}
				
				// Local Numbering = 5
				LocalPos = 5;
				aux1 = GlobalPos+1-imaxhalos*jmaxhalos;
				
				if(GlobNum[aux1] > 0) {

					for(l=0; l<3; l++) {

						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos  )] = (3*(GlobNum[aux1]-1)  )+1;
						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos+1)] = (3*(GlobNum[aux1]-1)+1)+1;
						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos+2)] = (3*(GlobNum[aux1]-1)+2)+1;

					}

				}
				
				// Local Numbering = 6
				LocalPos = 6;
				aux1 = GlobalPos+imaxhalos-1-imaxhalos*jmaxhalos;
				
				if(GlobNum[aux1] > 0) {

					for(l=0; l<3; l++) {

						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos  )] = (3*(GlobNum[aux1]-1)  )+1;
						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos+1)] = (3*(GlobNum[aux1]-1)+1)+1;
						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos+2)] = (3*(GlobNum[aux1]-1)+2)+1;

					}

				}
				
				// Local Numbering = 7
				LocalPos = 7;
				aux1 = GlobalPos+imaxhalos-imaxhalos*jmaxhalos;
				
				if(GlobNum[aux1] > 0) {

					for(l=0; l<3; l++) {

						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos  )] = (3*(GlobNum[aux1]-1)  )+1;
						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos+1)] = (3*(GlobNum[aux1]-1)+1)+1;
						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos+2)] = (3*(GlobNum[aux1]-1)+2)+1;

					}

				}
				
				// Local Numbering = 8
				LocalPos = 8;
				aux1 = GlobalPos+imaxhalos+1-imaxhalos*jmaxhalos;
				
				if(GlobNum[aux1] > 0) {

					for(l=0; l<3; l++) {

						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos  )] = (3*(GlobNum[aux1]-1)  )+1;
						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos+1)] = (3*(GlobNum[aux1]-1)+1)+1;
						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos+2)] = (3*(GlobNum[aux1]-1)+2)+1;

					}

				}
				
				// =================================================================
				// Second Layer:
				// Local Numbering = 9
				LocalPos = 9;
				aux1 = GlobalPos-imaxhalos-1;
				
				if(GlobNum[aux1] > 0) {

					for(l=0; l<3; l++) {

						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos  )] = (3*(GlobNum[aux1]-1)  )+1;
						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos+1)] = (3*(GlobNum[aux1]-1)+1)+1;
						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos+2)] = (3*(GlobNum[aux1]-1)+2)+1;

					}

				}
				
				// Local Numbering = 10
				LocalPos = 10;
				aux1 = GlobalPos-imaxhalos;
				
				if(GlobNum[aux1] > 0) {

					for(l=0; l<3; l++) {

						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos  )] = (3*(GlobNum[aux1]-1)  )+1;
						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos+1)] = (3*(GlobNum[aux1]-1)+1)+1;
						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos+2)] = (3*(GlobNum[aux1]-1)+2)+1;

					}

				}
	
				// Local Numbering = 11
				LocalPos = 11;
				aux1 = GlobalPos-imaxhalos+1;
				
				if(GlobNum[aux1] > 0) {

					for(l=0; l<3; l++) {

						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos  )] = (3*(GlobNum[aux1]-1)  )+1;
						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos+1)] = (3*(GlobNum[aux1]-1)+1)+1;
						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos+2)] = (3*(GlobNum[aux1]-1)+2)+1;

					}

				}
				
				// Local Numbering = 12
				LocalPos = 12;
				aux1 = GlobalPos-1;
				
				if(GlobNum[aux1] > 0) {

					for(l=0; l<3; l++) {

						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos  )] = (3*(GlobNum[aux1]-1)  )+1;
						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos+1)] = (3*(GlobNum[aux1]-1)+1)+1;
						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos+2)] = (3*(GlobNum[aux1]-1)+2)+1;

					}

				}
				
				// Local Numbering = 13
				LocalPos = 13;
				aux1 = GlobalPos;
				
				if(GlobNum[aux1] > 0) {

					for(l=0; l<3; l++) {

						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos  )] = (3*(GlobNum[aux1]-1)  )+1;
						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos+1)] = (3*(GlobNum[aux1]-1)+1)+1;
						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos+2)] = (3*(GlobNum[aux1]-1)+2)+1;

					}

				}
				
				// Local Numbering = 14
				LocalPos = 14;
				aux1 = GlobalPos+1;
				
				if(GlobNum[aux1] > 0) {

					for(l=0; l<3; l++) {

						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos  )] = (3*(GlobNum[aux1]-1)  )+1;
						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos+1)] = (3*(GlobNum[aux1]-1)+1)+1;
						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos+2)] = (3*(GlobNum[aux1]-1)+2)+1;

					}

				}
				
				// Local Numbering = 15
				LocalPos = 15;
				aux1 = GlobalPos+imaxhalos-1;
				
				if(GlobNum[aux1] > 0) {

					for(l=0; l<3; l++) {

						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos  )] = (3*(GlobNum[aux1]-1)  )+1;
						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos+1)] = (3*(GlobNum[aux1]-1)+1)+1;
						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos+2)] = (3*(GlobNum[aux1]-1)+2)+1;

					}

				}
				
				// Local Numbering = 16
				LocalPos = 16;
				aux1 = GlobalPos+imaxhalos;
				
				if(GlobNum[aux1] > 0) {

					for(l=0; l<3; l++) {

						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos  )] = (3*(GlobNum[aux1]-1)  )+1;
						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos+1)] = (3*(GlobNum[aux1]-1)+1)+1;
						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos+2)] = (3*(GlobNum[aux1]-1)+2)+1;

					}

				}
				
				// Local Numbering = 17
				LocalPos = 17;
				aux1 = GlobalPos+imaxhalos+1;
				
				if(GlobNum[aux1] > 0) {

					for(l=0; l<3; l++) {

						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos  )] = (3*(GlobNum[aux1]-1)  )+1;
						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos+1)] = (3*(GlobNum[aux1]-1)+1)+1;
						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos+2)] = (3*(GlobNum[aux1]-1)+2)+1;

					}

				}
				
				// =================================================================
				// Third Layer:
				// Local Numbering = 18
				LocalPos = 18;
				aux1 = GlobalPos-imaxhalos-1+imaxhalos*jmaxhalos;
				
				if(GlobNum[aux1] > 0) {

					for(l=0; l<3; l++) {

						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos  )] = (3*(GlobNum[aux1]-1)  )+1;
						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos+1)] = (3*(GlobNum[aux1]-1)+1)+1;
						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos+2)] = (3*(GlobNum[aux1]-1)+2)+1;

					}

				}
				
				// Local Numbering = 19
				LocalPos = 19;
				aux1 = GlobalPos-imaxhalos+imaxhalos*jmaxhalos;
				
				if(GlobNum[aux1] > 0) {

					for(l=0; l<3; l++) {

						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos  )] = (3*(GlobNum[aux1]-1)  )+1;
						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos+1)] = (3*(GlobNum[aux1]-1)+1)+1;
						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos+2)] = (3*(GlobNum[aux1]-1)+2)+1;

					}

				}
								
				// Local Numbering = 20
				LocalPos = 20;
				aux1 = GlobalPos-imaxhalos+1+imaxhalos*jmaxhalos;
				
				if(GlobNum[aux1] > 0) {

					for(l=0; l<3; l++) {

						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos  )] = (3*(GlobNum[aux1]-1)  )+1;
						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos+1)] = (3*(GlobNum[aux1]-1)+1)+1;
						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos+2)] = (3*(GlobNum[aux1]-1)+2)+1;

					}

				}
				
				// Local Numbering = 21
				LocalPos = 21;
				aux1 = GlobalPos-1+imaxhalos*jmaxhalos;
				
				if(GlobNum[aux1] > 0) {

					for(l=0; l<3; l++) {

						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos  )] = (3*(GlobNum[aux1]-1)  )+1;
						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos+1)] = (3*(GlobNum[aux1]-1)+1)+1;
						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos+2)] = (3*(GlobNum[aux1]-1)+2)+1;

					}

				}
				
				// Local Numbering = 22
				LocalPos = 22;
				aux1 = GlobalPos+imaxhalos*jmaxhalos;
				
				if(GlobNum[aux1] > 0) {

					for(l=0; l<3; l++) {

						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos  )] = (3*(GlobNum[aux1]-1)  )+1;
						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos+1)] = (3*(GlobNum[aux1]-1)+1)+1;
						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos+2)] = (3*(GlobNum[aux1]-1)+2)+1;

					}

				}
				
				// Local Numbering = 23
				LocalPos = 23;
				aux1 = GlobalPos+1+imaxhalos*jmaxhalos;
				
				if(GlobNum[aux1] > 0) {

					for(l=0; l<3; l++) {

						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos  )] = (3*(GlobNum[aux1]-1)  )+1;
						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos+1)] = (3*(GlobNum[aux1]-1)+1)+1;
						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos+2)] = (3*(GlobNum[aux1]-1)+2)+1;

					}

				}
				
				// Local Numbering = 24
				LocalPos = 24;
				aux1 = GlobalPos+imaxhalos-1+imaxhalos*jmaxhalos;
				
				if(GlobNum[aux1] > 0) {

					for(l=0; l<3; l++) {

						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos  )] = (3*(GlobNum[aux1]-1)  )+1;
						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos+1)] = (3*(GlobNum[aux1]-1)+1)+1;
						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos+2)] = (3*(GlobNum[aux1]-1)+2)+1;

					}

				}
				
				// Local Numbering = 25
				LocalPos = 25;
				aux1 = GlobalPos+imaxhalos+imaxhalos*jmaxhalos;
				
				if(GlobNum[aux1] > 0) {

					for(l=0; l<3; l++) {

						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos  )] = (3*(GlobNum[aux1]-1)  )+1;
						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos+1)] = (3*(GlobNum[aux1]-1)+1)+1;
						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos+2)] = (3*(GlobNum[aux1]-1)+2)+1;

					}

				}

				
				// Local Numbering = 26
				LocalPos = 26;
				aux1 = GlobalPos+imaxhalos+1+imaxhalos*jmaxhalos;
				
				if(GlobNum[aux1] > 0) {

					for(l=0; l<3; l++) {

						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos  )] = (3*(GlobNum[aux1]-1)  )+1;
						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos+1)] = (3*(GlobNum[aux1]-1)+1)+1;
						Vector_I_h[(3*cont+l)*RowSize + (3*LocalPos+2)] = (3*(GlobNum[aux1]-1)+2)+1;

					}

				}

				cont++;
				
			}
				
		}
		
	}
	
	free(GlobNum);

	/*int totalnodes = imax*jmax*kmax;

	for(i=0; i<3*totalnodes; i++) {
	
		printf("Row = %d\n", i);
		
		for(j=0; j<RowSize; j++) {
			printf("%d  ", Vector_I_h[i*RowSize+j]);
		}
		
		printf("\n");

	}*/

}

//==============================================================================

void EvaluateGPUMaterialProperty(int imax, int jmax, int kmax, int numno, double *MatPropQ1_h, double *MatPropQ2_h, double *MatPropQ3_h, double *MatPropQ4_h, 
double *MatPropQ5_h, double *MatPropQ6_h, double *MatPropQ7_h, double *MatPropQ8_h, double *Material_Data_h, int *BC_h)
{

	int i, j, k, cont, e_imax, e_jmax, e_kmax, e_imaxhalos, e_jmaxhalos, e_kmaxhalos;
	int ElemPos, NodePos;
	double *GlobElemMat;
	
	// imax = Number of nodes in X direction
	// jmax = Number of nodes in Y direction
	// kmax = Number of nodes in Y direction

	e_imax = imax-1;
	e_jmax = jmax-1;
	e_kmax = kmax-1;
	
	e_imaxhalos = e_imax+2;
	e_jmaxhalos = e_jmax+2;
	e_kmaxhalos = e_kmax+2;
	
	// Allocating memory space for host 
	GlobElemMat  = (double *)malloc(2*e_imaxhalos*e_jmaxhalos*e_kmaxhalos*sizeof(double));  // 2 why? _dE and _dNu

	for(i=0; i<2*e_imaxhalos*e_jmaxhalos*e_kmaxhalos; i++)
		GlobElemMat[i] = 0;

	cont = 0;

	for(k=1; k<e_kmaxhalos-1; k++) {					   // to go along the layer

		for(j=1; j<e_jmaxhalos-1; j++) {				   // to go along the column
  
			for(i=1; i<e_imaxhalos-1; i++) {               // to go along the row

				ElemPos = e_imaxhalos*e_jmaxhalos*k + e_imaxhalos*j + i;

				GlobElemMat[ElemPos] = Material_Data_h[cont];  // _dE  
				GlobElemMat[ElemPos + e_imaxhalos*e_jmaxhalos*e_kmaxhalos] = Material_Data_h[cont + e_imax*e_jmax*e_kmax];  // _dNu

				cont++;

			}

		}

	}
	
	/*for(k=0; k<e_kmaxhalos; k++) {					   // to go along the layer

		for(j=0; j<e_jmaxhalos; j++) {				       // to go along the column
  
			for(i=0; i<e_imaxhalos; i++) {                 // to go along the row

				ElemPos = e_imaxhalos*e_jmaxhalos*k + e_imaxhalos*j + i;
				
				printf("  %f %f", GlobElemMat[ElemPos], GlobElemMat[ElemPos + e_imaxhalos*e_jmaxhalos*e_kmaxhalos]);

			}
			
			printf("\n");

		}

	}*/

	// =========================================================================================================
		
	cont = 0;

	for(k=0; k<kmax; k++) {					               // to go along the layer

		for(j=0; j<jmax; j++) {				               // to go along the column
  
			for(i=0; i<imax; i++) {                        // to go along the line
			
				if(BC_h[cont]==0) {            // if BC_h = 1 Displacement is constrained at the node 
			
					NodePos = imax*jmax*k + imax*j + i;
					ElemPos = e_imaxhalos*e_jmaxhalos*k + e_imaxhalos*j + i;

					MatPropQ1_h[NodePos        ] = GlobElemMat[ElemPos                                      ];
					MatPropQ1_h[NodePos + numno] = GlobElemMat[ElemPos + e_imaxhalos*e_jmaxhalos*e_kmaxhalos];
				
					MatPropQ2_h[NodePos        ] = GlobElemMat[ElemPos + 1                                      ];
					MatPropQ2_h[NodePos + numno] = GlobElemMat[ElemPos + 1 + e_imaxhalos*e_jmaxhalos*e_kmaxhalos];
				
					MatPropQ3_h[NodePos        ] = GlobElemMat[ElemPos + e_imaxhalos                                      ];
					MatPropQ3_h[NodePos + numno] = GlobElemMat[ElemPos + e_imaxhalos + e_imaxhalos*e_jmaxhalos*e_kmaxhalos];
				
					MatPropQ4_h[NodePos        ] = GlobElemMat[ElemPos + e_imaxhalos + 1                                      ];
					MatPropQ4_h[NodePos + numno] = GlobElemMat[ElemPos + e_imaxhalos + 1 + e_imaxhalos*e_jmaxhalos*e_kmaxhalos];
				
					MatPropQ5_h[NodePos        ] = GlobElemMat[ElemPos + e_imaxhalos*e_jmaxhalos                                      ];
					MatPropQ5_h[NodePos + numno] = GlobElemMat[ElemPos + e_imaxhalos*e_jmaxhalos + e_imaxhalos*e_jmaxhalos*e_kmaxhalos];
				
					MatPropQ6_h[NodePos        ] = GlobElemMat[ElemPos + e_imaxhalos*e_jmaxhalos + 1                                      ];
					MatPropQ6_h[NodePos + numno] = GlobElemMat[ElemPos + e_imaxhalos*e_jmaxhalos + 1 + e_imaxhalos*e_jmaxhalos*e_kmaxhalos];
				
					MatPropQ7_h[NodePos        ] = GlobElemMat[ElemPos + e_imaxhalos*e_jmaxhalos + e_imaxhalos                                      ];
					MatPropQ7_h[NodePos + numno] = GlobElemMat[ElemPos + e_imaxhalos*e_jmaxhalos + e_imaxhalos + e_imaxhalos*e_jmaxhalos*e_kmaxhalos];
				
					MatPropQ8_h[NodePos        ] = GlobElemMat[ElemPos + e_imaxhalos*e_jmaxhalos + e_imaxhalos + 1                                      ];
					MatPropQ8_h[NodePos + numno] = GlobElemMat[ElemPos + e_imaxhalos*e_jmaxhalos + e_imaxhalos + 1 + e_imaxhalos*e_jmaxhalos*e_kmaxhalos];
										
				}
				
				cont++;
					
			}
			
		}
		
	}
	
	/*for(i=0; i<27; i++) {
	
		printf("  %f %f \n", MatPropQ1_h[i], MatPropQ1_h[i+numno]);
		printf("  %f %f \n", MatPropQ2_h[i], MatPropQ2_h[i+numno]);
		printf("  %f %f \n", MatPropQ3_h[i], MatPropQ3_h[i+numno]);
		printf("  %f %f \n", MatPropQ4_h[i], MatPropQ4_h[i+numno]);
		printf("  %f %f \n", MatPropQ5_h[i], MatPropQ5_h[i+numno]);
		printf("  %f %f \n", MatPropQ6_h[i], MatPropQ6_h[i+numno]);
		printf("  %f %f \n", MatPropQ7_h[i], MatPropQ7_h[i+numno]);
		printf("  %f %f \n", MatPropQ8_h[i], MatPropQ8_h[i+numno]);
		printf("\n");
	
	}*/
				
	free(GlobElemMat);


}

//==============================================================================
// Reading nodal concentrate load from external file:

void ReadConcLoad(int numno, double *Vector_B_h, int *BC_h)
{
	int i, nc;
	int id;
	double fx, fy, fz;
	
    FILE *inFile;
    
    inFile = fopen("examples/LoadFile.inc", "r");
    
	if(fscanf(inFile, "%d", &nc) != 1) {
		printf("\nError on reading number of nodal loads !!!\n\n");
	}
	
	if(nc == 0) return;
	
	for(i=0; i<nc; i++) {
	
		if(fscanf(inFile, "%d %lf %lf %lf", &id, &fx, &fy, &fz) != 4) {
			printf("\n Error on reading nodal loads (number %d) !!!\n\n", i+1);
			return;
		}
		
		Vector_B_h[3*(id-1)  ] = fx;
		Vector_B_h[3*(id-1)+1] = fy;
		Vector_B_h[3*(id-1)+2] = fz;

		if(Vector_B_h[3*(id-1)  ] !=0) {
			printf("%d %f %f %f\n", id, Vector_B_h[3*(id-1)  ], Vector_B_h[3*(id-1)+1], Vector_B_h[3*(id-1)+2]);
		}
		
	}
	
	for(i=0; i<numno; i++) {
	
		if(BC_h[i]==1) {            // if BC_h = 1 Displacement is constrained at the node 
		
			Vector_B_h[3*i  ] = 0.;
			Vector_B_h[3*i+1] = 0.;
			Vector_B_h[3*i+2] = 0.;
		
		}
		
		//if(fabs(Vector_B_h[3*i+2]) > 0.01) printf("%d %f \n", 3*i+2, Vector_B_h[3*i+2]);
		
	}
	
	fclose(inFile);
    
}

//==============================================================================
void GPU_Information(int &deviceCount, double *GPUData)
{
	int dev;
	size_t free_byte, total_byte;
	int driverVersion = 0, runtimeVersion = 0;
	double auxN;

	// Printing on output file:  ===============================================

	using namespace std;

	std::string FileName  ="GPUsInformation.out";
	remove(FileName.c_str());

	// Create a new file
	ofstream outfile;
	outfile.open(FileName.c_str(),ios::app);

	outfile << "======================================================== " << endl;
	outfile << endl;
	outfile << "Number of devices " << endl;
	outfile << deviceCount << endl;
	outfile << endl;

	// Printing on Display:  ===================================================

	printf("\n\n");
	printf("         Reading GPU Specification \n");
	printf("         ========================================= \n\n");

	// This function call returns 0 if there are no CUDA capable devices.
	if (deviceCount == 0)
		printf("         There is no device supporting CUDA\n");

	// ======================================================================================================================

	for(dev=0; dev<deviceCount; ++dev) {

		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);

		if(dev == 0) {
			// This function call returns 9999 for both major & minor fields, if no CUDA capable devices are present
			if (deviceProp.major == 9999 && deviceProp.minor == 9999)
				printf("         There is no device supporting CUDA.\n");
			else if (deviceCount == 1)
				printf("         There is 1 device supporting CUDA\n");
			else
				printf("         There are %d devices supporting CUDA\n", deviceCount);
		}
		printf("\n         Device %d: \"%s\"\n", dev, deviceProp.name);

#if CUDART_VERSION >= 2020
		// Console log
		cudaDriverGetVersion(&driverVersion);
		printf("         CUDA Driver Version:                           %d.%d\n", driverVersion/1000, driverVersion%100);
		cudaRuntimeGetVersion(&runtimeVersion);
		printf("         CUDA Runtime Version:                          %d.%d\n", runtimeVersion/1000, runtimeVersion%100);
#endif
		printf("         CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);

		printf("         Total amount of global memory:                 %llu bytes\n", (unsigned long long) deviceProp.totalGlobalMem);

		cudaSetDevice(dev);
		cudaMemGetInfo(&free_byte, &total_byte);  // Returns the free and total amount of memory available for allocation by the device in bytes
		auxN = free_byte;
		printf("         Total amount of global free memory:            %.0f bytes\n", auxN);

		printf("         Total number of multiprocessor:                %d\n", deviceProp.multiProcessorCount);
		GPUData[dev*5+0] = deviceProp.multiProcessorCount;

		printf("         Total amount of constant memory:               %u bytes\n", deviceProp.totalConstMem); 

		printf("         Total amount of shared memory per block:       %u bytes\n", deviceProp.sharedMemPerBlock);
		GPUData[dev*5+1] = deviceProp.sharedMemPerBlock;

		printf("         Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
		GPUData[dev*5+2] = deviceProp.regsPerBlock;

		printf("         Warp size:                                     %d\n", deviceProp.warpSize);
		GPUData[dev*5+3] = deviceProp.warpSize;

		printf("         Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
		GPUData[dev*5+4] = deviceProp.maxThreadsPerBlock;

		printf("         Maximum 1D linear texture size:                %d bytes\n", deviceProp.maxTexture1DLinear);
		
		printf("         Maximum sizes of each dimension of a block:    %d x %d x %d\n",
			deviceProp.maxThreadsDim[0],
			deviceProp.maxThreadsDim[1],
			deviceProp.maxThreadsDim[2]);
		printf("         Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
			deviceProp.maxGridSize[0],
			deviceProp.maxGridSize[1],
			deviceProp.maxGridSize[2]);
		printf("         Clock rate:                                    %.2f GHz\n", deviceProp.clockRate * 1e-6f);

	// *********************************** Print GPUs Information: ***********************************

		outfile << "======================================================== " << endl;
		outfile << endl;
		outfile << "Device " << dev << " ======= " << deviceProp.name << endl;
		outfile << endl;

		outfile << ">> Total amount of global memory " << endl;
		outfile << total_byte << " Bytes" << endl;
		outfile << total_byte / 1e9 << " GBytes" << endl;

		outfile << ">> Total amount of global free memory " << endl;
		outfile << free_byte << " Bytes" << endl;
		outfile << free_byte / 1e9 << " GBytes" << endl;

		outfile << ">> Total number of multiprocessor " << endl;
		outfile << deviceProp.multiProcessorCount << endl;

		outfile << ">> Total amount of shared memory per block " << endl;
		outfile << deviceProp.sharedMemPerBlock << endl;

		outfile << ">> Total number of registers available per block " << endl;
		outfile << deviceProp.regsPerBlock << endl;

		outfile << ">> Warp size " << endl;
		outfile << deviceProp.warpSize << endl;

		outfile << ">> Maximum number of threads per block " << endl;
		outfile << deviceProp.maxThreadsPerBlock << endl;

	}

	outfile.close();

}

//==============================================================================
void InitialInformation()
{
	printf("\n\n");
	printf("\t --------------------------------------------------   \n");
	printf("\t  PONTIFICAL CATHOLIC UNIVERSITY OF RIO DE JANEIRO    \n");
	printf("\t **       DEPARTMENT OF CIVIL ENGINEERING        **   \n");
	printf("\t GTEP - Group of Technology in Petroleum Engineering  \n\n");
	printf("\t               C H R O N O S  -  3  D                 \n\n");
	printf("\t            FINITE ELEMENT METHOD ON GPU              \n");
	printf("\t                             Version: 0.2 - fev/18    \n");
	printf("\t -------------------------------------------------- \n\n");

}

//==============================================================================
int GPU_ReadSize(int &numno, int &numel, int &nummat, int &numprop, int &numdof, int &numnoel, int &numstr, int &numbc, int &numnl,
int &imax, int &jmax, int &kmax, char GPU_arqaux[80])
{
	char GPU_arqinp[80];
	char label[80];
	std::string s;
	FILE *inFile;
	int c;

	printf ( "         Enter input file name [.dat]........: ");
	gets(GPU_arqinp);

	strcpy(GPU_arqaux, GPU_arqinp);
	strcat(GPU_arqaux, ".dat");
	inFile  = fopen(GPU_arqaux, "r");

	if((inFile == NULL)) {
		printf("\n\n\n\t *** Error on reading Input file !!! *** \n");
		return(0);
	}

	// ========= Getting Problem Vectors Size  =========

	
	// numno            Number of nodes of the mesh          
	// numel            Number of elements on mesh
	// nummat           Number of materials on mesh
	// numprop          Number of materials properties on mesh
	// numdof           Number of freedom of degree (DOF) per node
	// numelno          Number of nodes per element
	// numstr           Number of stress component
	// numbc            Number of boundary conditions
	// numnl            Number of nodes with nodal load
	// imax             Number of nodes in x direction
	// jmax             Number of nodes in y direction
	// kmax             Number of nodes in z direction

	while((c = fgetc(inFile)) != EOF) {        // find the end of file 

		fscanf(inFile, "%s", label );          // scan label string 

		s = label;

		if(s == "%PROBLEM.SIZES") {

			if(fscanf(inFile, "%d %d %d %d %d %d %d %d %d %d %d %d", &numno, &numel, &nummat, &numprop, &numdof, &numnoel, &numstr, &numbc, &numnl,
				&imax, &jmax, &kmax) != 12) {
				//fprintf("\nError on reading vector size");
				return(0);
			}
				
			break;

		}
		
	}

	fclose(inFile);

	return(1);

}

//==============================================================================
int GPU_ReadInPutFile(int numno, int numel, int nummat, int numprop, int numdof, int numnoel, int numstr, int numbc, int numnl, char GPU_arqaux[80],
double *X_h, double *Y_h, double *Z_h, int *BC_h, double *Material_Param_h, int *Connect_h, int *Material_Id_h)
{
	char label[80];
	std::string s;
	FILE *inFile;
	int c;
	int i, number, matId, aux, el1, el2, el3, el4, el5, el6, el7, el8;

	int _numno, _numel, _nummat,  _numbc;
	int dx, dy, dz, rx, ry, rz;
	double prop1, prop2;
	
	inFile  = fopen(GPU_arqaux, "r");

	if((inFile == NULL)) {
		printf("\n\n\n\t *** Error on reading Input file !!! *** \n");
		return(0);
	}

	// ========= Reading nodal coordinate data =========


	while(1) { 

		while((c = fgetc(inFile))!='%') {        
			if(c == EOF) break;
		} 

		fscanf(inFile, "%s", label);          // scan label string 

		s = label;

		if(s == "NODE.COORD") {

			if(fscanf(inFile, "%d", &_numno) != 1) {
				printf("\n Error on reading number of nodes !!!\n\n" );
				return(0);
			}

			if(numno != _numno) return(0);

			for(i=0; i<numno; i++) {

				if(fscanf(inFile, "%d %lf %lf %lf", &number, &X_h[i], &Y_h[i], &Z_h[i]) != 4){
					printf("\nError on reading nodes coordinates, node Id = %d", number);
					return(0);
				}

			}

			break;

		}

	}

	// ========= Reading boundary condition data =========

	while(1) { 

		while((c = fgetc(inFile))!='%') {        
			if(c == EOF) break;
		} 

		fscanf(inFile, "%s", label);          // scan label string 

		s = label;

		if(s == "NODE.SUPPORT") {

			if(fscanf(inFile, "%d", &_numbc) != 1) {
				printf("\n Error on reading number of nodes !!!\n\n" );
				return(0);
			}

			if(numbc != _numbc) return(0);

			for(i=0; i<numbc; i++) {

				if(fscanf(inFile, "%d %d %d %d %d %d %d", &number, &dx, &dy, &dz, &rx, &ry, &rz) != 7){
					printf("\nError on reading nodal boundary conditions, node Id = %d", number);
					return(0);
				}

				if(dx == 1 && dy == 1 && dz == 1) {
					BC_h[number-1] = 1;
				}

			}

			break;

		}

	}

	// ========= Reading material data =========

	while(1) { 

		while((c = fgetc(inFile))!='%') {        
			if(c == EOF) break;
		} 

		fscanf(inFile, "%s", label);          // scan label string 

		s = label;

		if(s == "MATERIAL.ISOTROPIC") {

			if(fscanf(inFile, "%d", &_nummat) != 1) {
				printf("\n Error on reading number of nodes !!!\n\n" );
				return(0);
			}

			if(nummat != _nummat) return(0);

			for(i=0; i<nummat; i++) {

				if(fscanf(inFile, "%d %lf %lf", &number, &prop1, &prop2) != 3){
					printf("\nError on reading material property, material Id = %d", number);
					return(0);
				}

				Material_Param_h[i       ] = prop1;
				Material_Param_h[i+nummat] = prop2;

			}

			break;

		}

	}

	// ========= Reading element data =========

	while(1) { 

		while((c = fgetc(inFile))!='%') {        
			if(c == EOF) break;
		} 

		fscanf(inFile, "%s", label);          // scan label string 

		s = label;

		if(s == "ELEMENT.BRICK8") {

			if(fscanf(inFile, "%d", &_numel) != 1) {
				printf("\n Error on reading number of nodes !!!\n\n" );
				return(0);
			}

			if(numel != _numel) return(0);

			for(i=0; i<numel; i++) {

				if(fscanf(inFile, "%d %d %d %d %d %d %d %d %d %d %d", &number, &matId, &aux, &el1, &el2, &el3, &el4, &el5, &el6, &el7, &el8) != 11){
					printf("\nError on reading element connectivity, element Id = %d", number);
					return(0);
				}

				Material_Id_h[i] = matId;

				Connect_h[i + 0*numel] = el1;
				Connect_h[i + 1*numel] = el2;
				Connect_h[i + 2*numel] = el3;
				Connect_h[i + 3*numel] = el4;
				Connect_h[i + 4*numel] = el5;
				Connect_h[i + 5*numel] = el6;
				Connect_h[i + 6*numel] = el7;
				Connect_h[i + 7*numel] = el8;

			}

			break;

		}

	}

	return(1);

}

//==============================================================================
void EvaluateGPUCoordinate(int imax, int jmax, int kmax, double *X_h, double *Y_h, double *Z_h, int numno, double *CoordQ1_h, double *CoordQ2_h, 
double *CoordQ3_h, double *CoordQ4_h, double *CoordQ5_h, double *CoordQ6_h, double *CoordQ7_h, double *CoordQ8_h)
{
	int i, j, k, pos_ijk, cont;
	
	cont=0;

	for(k=0; k<kmax; k++) {					     // to go along the layer

		for(j=0; j<jmax; j++) {				     // to go along the column
  
			for(i=0; i<imax; i++) {              // to go along the line
			
				pos_ijk = ((imax*jmax)*k) + (imax*j) + i;
				
				// ****************************** Evaluating CoordQ1_h ******************************
				
				if(i != 0) {
				
					if(j != 0) {
					
						if(k != 0) { 
											
							// Node 1:
							CoordQ1_h[cont+ 0*numno] = X_h[pos_ijk-(imax*jmax)-imax-1];
							CoordQ1_h[cont+ 1*numno] = Y_h[pos_ijk-(imax*jmax)-imax-1];
							CoordQ1_h[cont+ 2*numno] = Z_h[pos_ijk-(imax*jmax)-imax-1];
							
							// Node 2:
							CoordQ1_h[cont+ 3*numno] = X_h[pos_ijk-(imax*jmax)-imax];
							CoordQ1_h[cont+ 4*numno] = Y_h[pos_ijk-(imax*jmax)-imax];
							CoordQ1_h[cont+ 5*numno] = Z_h[pos_ijk-(imax*jmax)-imax];
							
							// Node 3:
							CoordQ1_h[cont+ 6*numno] = X_h[pos_ijk-(imax*jmax)];
							CoordQ1_h[cont+ 7*numno] = Y_h[pos_ijk-(imax*jmax)];
							CoordQ1_h[cont+ 8*numno] = Z_h[pos_ijk-(imax*jmax)];
							
							// Node 4:
							CoordQ1_h[cont+ 9*numno] = X_h[pos_ijk-(imax*jmax)-1];
							CoordQ1_h[cont+10*numno] = Y_h[pos_ijk-(imax*jmax)-1];
							CoordQ1_h[cont+11*numno] = Z_h[pos_ijk-(imax*jmax)-1];
						
							// Node 5:
							CoordQ1_h[cont+12*numno] = X_h[pos_ijk-imax-1];
							CoordQ1_h[cont+13*numno] = Y_h[pos_ijk-imax-1];
							CoordQ1_h[cont+14*numno] = Z_h[pos_ijk-imax-1];
							
							// Node 6:
							CoordQ1_h[cont+15*numno] = X_h[pos_ijk-imax];
							CoordQ1_h[cont+16*numno] = Y_h[pos_ijk-imax];
							CoordQ1_h[cont+17*numno] = Z_h[pos_ijk-imax];
							
							// Node 7:
							CoordQ1_h[cont+18*numno] = X_h[pos_ijk];
							CoordQ1_h[cont+19*numno] = Y_h[pos_ijk];
							CoordQ1_h[cont+20*numno] = Z_h[pos_ijk];
							
							// Node 8:
							CoordQ1_h[cont+21*numno] = X_h[pos_ijk-1];
							CoordQ1_h[cont+22*numno] = Y_h[pos_ijk-1];
							CoordQ1_h[cont+23*numno] = Z_h[pos_ijk-1];
							
						}
						
					}
					
				}
				
				// ****************************** Evaluating CoordQ2_h ******************************
				
				if(i != imax-1) {
				
					if(j != 0) {
					
						if(k != 0) { 
											
							// Node 1:
							CoordQ2_h[cont+ 0*numno] = X_h[pos_ijk-(imax*jmax)-imax];
							CoordQ2_h[cont+ 1*numno] = Y_h[pos_ijk-(imax*jmax)-imax];
							CoordQ2_h[cont+ 2*numno] = Z_h[pos_ijk-(imax*jmax)-imax];
							
							// Node 2:
							CoordQ2_h[cont+ 3*numno] = X_h[pos_ijk-(imax*jmax)-imax+1];
							CoordQ2_h[cont+ 4*numno] = Y_h[pos_ijk-(imax*jmax)-imax+1];
							CoordQ2_h[cont+ 5*numno] = Z_h[pos_ijk-(imax*jmax)-imax+1];
							
							// Node 3:
							CoordQ2_h[cont+ 6*numno] = X_h[pos_ijk-(imax*jmax)+1];
							CoordQ2_h[cont+ 7*numno] = Y_h[pos_ijk-(imax*jmax)+1];
							CoordQ2_h[cont+ 8*numno] = Z_h[pos_ijk-(imax*jmax)+1];
							
							// Node 4:
							CoordQ2_h[cont+ 9*numno] = X_h[pos_ijk-(imax*jmax)];
							CoordQ2_h[cont+10*numno] = Y_h[pos_ijk-(imax*jmax)];
							CoordQ2_h[cont+11*numno] = Z_h[pos_ijk-(imax*jmax)];
						
							// Node 5:
							CoordQ2_h[cont+12*numno] = X_h[pos_ijk-imax];
							CoordQ2_h[cont+13*numno] = Y_h[pos_ijk-imax];
							CoordQ2_h[cont+14*numno] = Z_h[pos_ijk-imax];
							
							// Node 6:
							CoordQ2_h[cont+15*numno] = X_h[pos_ijk-imax+1];
							CoordQ2_h[cont+16*numno] = Y_h[pos_ijk-imax+1];
							CoordQ2_h[cont+17*numno] = Z_h[pos_ijk-imax+1];
							
							// Node 7:
							CoordQ2_h[cont+18*numno] = X_h[pos_ijk+1];
							CoordQ2_h[cont+19*numno] = Y_h[pos_ijk+1];
							CoordQ2_h[cont+20*numno] = Z_h[pos_ijk+1];
							
							// Node 8:
							CoordQ2_h[cont+21*numno] = X_h[pos_ijk];
							CoordQ2_h[cont+22*numno] = Y_h[pos_ijk];
							CoordQ2_h[cont+23*numno] = Z_h[pos_ijk];
							
						}
						
					}
					
				}
				
				// ****************************** Evaluating CoordQ3_h ******************************
				
				if(i != 0) {
				
					if(j != jmax-1) {
					
						if(k != 0) { 
											
							// Node 1:
							CoordQ3_h[cont+ 0*numno] = X_h[pos_ijk-(imax*jmax)-1];
							CoordQ3_h[cont+ 1*numno] = Y_h[pos_ijk-(imax*jmax)-1];
							CoordQ3_h[cont+ 2*numno] = Z_h[pos_ijk-(imax*jmax)-1];
							
							// Node 2:
							CoordQ3_h[cont+ 3*numno] = X_h[pos_ijk-(imax*jmax)];
							CoordQ3_h[cont+ 4*numno] = Y_h[pos_ijk-(imax*jmax)];
							CoordQ3_h[cont+ 5*numno] = Z_h[pos_ijk-(imax*jmax)];
							
							// Node 3:
							CoordQ3_h[cont+ 6*numno] = X_h[pos_ijk-(imax*jmax)+imax];
							CoordQ3_h[cont+ 7*numno] = Y_h[pos_ijk-(imax*jmax)+imax];
							CoordQ3_h[cont+ 8*numno] = Z_h[pos_ijk-(imax*jmax)+imax];
							
							// Node 4:
							CoordQ3_h[cont+ 9*numno] = X_h[pos_ijk-(imax*jmax)+imax-1];
							CoordQ3_h[cont+10*numno] = Y_h[pos_ijk-(imax*jmax)+imax-1];
							CoordQ3_h[cont+11*numno] = Z_h[pos_ijk-(imax*jmax)+imax-1];
						
							// Node 5:
							CoordQ3_h[cont+12*numno] = X_h[pos_ijk-1];
							CoordQ3_h[cont+13*numno] = Y_h[pos_ijk-1];
							CoordQ3_h[cont+14*numno] = Z_h[pos_ijk-1];
							
							// Node 6:
							CoordQ3_h[cont+15*numno] = X_h[pos_ijk];
							CoordQ3_h[cont+16*numno] = Y_h[pos_ijk];
							CoordQ3_h[cont+17*numno] = Z_h[pos_ijk];
							
							// Node 7:
							CoordQ3_h[cont+18*numno] = X_h[pos_ijk+imax];
							CoordQ3_h[cont+19*numno] = Y_h[pos_ijk+imax];
							CoordQ3_h[cont+20*numno] = Z_h[pos_ijk+imax];
							
							// Node 8:
							CoordQ3_h[cont+21*numno] = X_h[pos_ijk+imax-1];
							CoordQ3_h[cont+22*numno] = Y_h[pos_ijk+imax-1];
							CoordQ3_h[cont+23*numno] = Z_h[pos_ijk+imax-1];
							
						}
						
					}
					
				}
				
				// ****************************** Evaluating CoordQ4_h ******************************
				
				if(i != imax-1) {
				
					if(j != jmax-1) {
					
						if(k != 0) { 
											
							// Node 1:
							CoordQ4_h[cont+ 0*numno] = X_h[pos_ijk-(imax*jmax)];
							CoordQ4_h[cont+ 1*numno] = Y_h[pos_ijk-(imax*jmax)];
							CoordQ4_h[cont+ 2*numno] = Z_h[pos_ijk-(imax*jmax)];
							
							// Node 2:
							CoordQ4_h[cont+ 3*numno] = X_h[pos_ijk-(imax*jmax)+1];
							CoordQ4_h[cont+ 4*numno] = Y_h[pos_ijk-(imax*jmax)+1];
							CoordQ4_h[cont+ 5*numno] = Z_h[pos_ijk-(imax*jmax)+1];
							
							// Node 3:
							CoordQ4_h[cont+ 6*numno] = X_h[pos_ijk-(imax*jmax)+imax+1];
							CoordQ4_h[cont+ 7*numno] = Y_h[pos_ijk-(imax*jmax)+imax+1];
							CoordQ4_h[cont+ 8*numno] = Z_h[pos_ijk-(imax*jmax)+imax+1];
							
							// Node 4:
							CoordQ4_h[cont+ 9*numno] = X_h[pos_ijk-(imax*jmax)+imax];
							CoordQ4_h[cont+10*numno] = Y_h[pos_ijk-(imax*jmax)+imax];
							CoordQ4_h[cont+11*numno] = Z_h[pos_ijk-(imax*jmax)+imax];
						
							// Node 5:
							CoordQ4_h[cont+12*numno] = X_h[pos_ijk];
							CoordQ4_h[cont+13*numno] = Y_h[pos_ijk];
							CoordQ4_h[cont+14*numno] = Z_h[pos_ijk];
							
							// Node 6:
							CoordQ4_h[cont+15*numno] = X_h[pos_ijk+1];
							CoordQ4_h[cont+16*numno] = Y_h[pos_ijk+1];
							CoordQ4_h[cont+17*numno] = Z_h[pos_ijk+1];
							
							// Node 7:
							CoordQ4_h[cont+18*numno] = X_h[pos_ijk+imax+1];
							CoordQ4_h[cont+19*numno] = Y_h[pos_ijk+imax+1];
							CoordQ4_h[cont+20*numno] = Z_h[pos_ijk+imax+1];
							
							// Node 8:
							CoordQ4_h[cont+21*numno] = X_h[pos_ijk+imax];
							CoordQ4_h[cont+22*numno] = Y_h[pos_ijk+imax];
							CoordQ4_h[cont+23*numno] = Z_h[pos_ijk+imax];
							
						}
						
					}
					
				}
				
				// ****************************** Evaluating CoordQ5_h ******************************
				
				if(i != 0) {
				
					if(j != 0) {
					
						if(k != kmax-1) { 
											
							// Node 1:
							CoordQ5_h[cont+ 0*numno] = X_h[pos_ijk-imax-1];
							CoordQ5_h[cont+ 1*numno] = Y_h[pos_ijk-imax-1];
							CoordQ5_h[cont+ 2*numno] = Z_h[pos_ijk-imax-1];
							
							// Node 2:
							CoordQ5_h[cont+ 3*numno] = X_h[pos_ijk-imax];
							CoordQ5_h[cont+ 4*numno] = Y_h[pos_ijk-imax];
							CoordQ5_h[cont+ 5*numno] = Z_h[pos_ijk-imax];
							
							// Node 3:
							CoordQ5_h[cont+ 6*numno] = X_h[pos_ijk];
							CoordQ5_h[cont+ 7*numno] = Y_h[pos_ijk];
							CoordQ5_h[cont+ 8*numno] = Z_h[pos_ijk];
							
							// Node 4:
							CoordQ5_h[cont+ 9*numno] = X_h[pos_ijk-1];
							CoordQ5_h[cont+10*numno] = Y_h[pos_ijk-1];
							CoordQ5_h[cont+11*numno] = Z_h[pos_ijk-1];
						
							// Node 5:
							CoordQ5_h[cont+12*numno] = X_h[pos_ijk+(imax*jmax)-imax-1];
							CoordQ5_h[cont+13*numno] = Y_h[pos_ijk+(imax*jmax)-imax-1];
							CoordQ5_h[cont+14*numno] = Z_h[pos_ijk+(imax*jmax)-imax-1];
							
							// Node 6:
							CoordQ5_h[cont+15*numno] = X_h[pos_ijk+(imax*jmax)-imax];
							CoordQ5_h[cont+16*numno] = Y_h[pos_ijk+(imax*jmax)-imax];
							CoordQ5_h[cont+17*numno] = Z_h[pos_ijk+(imax*jmax)-imax];
							
							// Node 7:
							CoordQ5_h[cont+18*numno] = X_h[pos_ijk+(imax*jmax)];
							CoordQ5_h[cont+19*numno] = Y_h[pos_ijk+(imax*jmax)];
							CoordQ5_h[cont+20*numno] = Z_h[pos_ijk+(imax*jmax)];
							
							// Node 8:
							CoordQ5_h[cont+21*numno] = X_h[pos_ijk+(imax*jmax)-1];
							CoordQ5_h[cont+22*numno] = Y_h[pos_ijk+(imax*jmax)-1];
							CoordQ5_h[cont+23*numno] = Z_h[pos_ijk+(imax*jmax)-1];
							
						}
						
					}
					
				}
				
				// ****************************** Evaluating CoordQ6_h ******************************
				
				if(i != imax-1) {
				
					if(j != 0) {
					
						if(k != kmax-1) { 
											
							// Node 1:
							CoordQ6_h[cont+ 0*numno] = X_h[pos_ijk-imax];
							CoordQ6_h[cont+ 1*numno] = Y_h[pos_ijk-imax];
							CoordQ6_h[cont+ 2*numno] = Z_h[pos_ijk-imax];
							
							// Node 2:
							CoordQ6_h[cont+ 3*numno] = X_h[pos_ijk-imax+1];
							CoordQ6_h[cont+ 4*numno] = Y_h[pos_ijk-imax+1];
							CoordQ6_h[cont+ 5*numno] = Z_h[pos_ijk-imax+1];
							
							// Node 3:
							CoordQ6_h[cont+ 6*numno] = X_h[pos_ijk+1];
							CoordQ6_h[cont+ 7*numno] = Y_h[pos_ijk+1];
							CoordQ6_h[cont+ 8*numno] = Z_h[pos_ijk+1];
							
							// Node 4:
							CoordQ6_h[cont+ 9*numno] = X_h[pos_ijk];
							CoordQ6_h[cont+10*numno] = Y_h[pos_ijk];
							CoordQ6_h[cont+11*numno] = Z_h[pos_ijk];
						
							// Node 5:
							CoordQ6_h[cont+12*numno] = X_h[pos_ijk+(imax*jmax)-imax];
							CoordQ6_h[cont+13*numno] = Y_h[pos_ijk+(imax*jmax)-imax];
							CoordQ6_h[cont+14*numno] = Z_h[pos_ijk+(imax*jmax)-imax];
							
							// Node 6:
							CoordQ6_h[cont+15*numno] = X_h[pos_ijk+(imax*jmax)-imax+1];
							CoordQ6_h[cont+16*numno] = Y_h[pos_ijk+(imax*jmax)-imax+1];
							CoordQ6_h[cont+17*numno] = Z_h[pos_ijk+(imax*jmax)-imax+1];
							
							// Node 7:
							CoordQ6_h[cont+18*numno] = X_h[pos_ijk+(imax*jmax)+1];
							CoordQ6_h[cont+19*numno] = Y_h[pos_ijk+(imax*jmax)+1];
							CoordQ6_h[cont+20*numno] = Z_h[pos_ijk+(imax*jmax)+1];
							
							// Node 8:
							CoordQ6_h[cont+21*numno] = X_h[pos_ijk+(imax*jmax)];
							CoordQ6_h[cont+22*numno] = Y_h[pos_ijk+(imax*jmax)];
							CoordQ6_h[cont+23*numno] = Z_h[pos_ijk+(imax*jmax)];
							
						}
						
					}
					
				}
				
				// ****************************** Evaluating CoordQ7_h ******************************
				
				if(i != 0) {
				
					if(j != jmax-1) {
					
						if(k != kmax-1) { 
											
							// Node 1:
							CoordQ7_h[cont+ 0*numno] = X_h[pos_ijk-1];
							CoordQ7_h[cont+ 1*numno] = Y_h[pos_ijk-1];
							CoordQ7_h[cont+ 2*numno] = Z_h[pos_ijk-1];
							
							// Node 2:
							CoordQ7_h[cont+ 3*numno] = X_h[pos_ijk];
							CoordQ7_h[cont+ 4*numno] = Y_h[pos_ijk];
							CoordQ7_h[cont+ 5*numno] = Z_h[pos_ijk];
							
							// Node 3:
							CoordQ7_h[cont+ 6*numno] = X_h[pos_ijk+imax];
							CoordQ7_h[cont+ 7*numno] = Y_h[pos_ijk+imax];
							CoordQ7_h[cont+ 8*numno] = Z_h[pos_ijk+imax];
							
							// Node 4:
							CoordQ7_h[cont+ 9*numno] = X_h[pos_ijk+imax-1];
							CoordQ7_h[cont+10*numno] = Y_h[pos_ijk+imax-1];
							CoordQ7_h[cont+11*numno] = Z_h[pos_ijk+imax-1];
						
							// Node 5:
							CoordQ7_h[cont+12*numno] = X_h[pos_ijk+(imax*jmax)-1];
							CoordQ7_h[cont+13*numno] = Y_h[pos_ijk+(imax*jmax)-1];
							CoordQ7_h[cont+14*numno] = Z_h[pos_ijk+(imax*jmax)-1];
							
							// Node 6:
							CoordQ7_h[cont+15*numno] = X_h[pos_ijk+(imax*jmax)];
							CoordQ7_h[cont+16*numno] = Y_h[pos_ijk+(imax*jmax)];
							CoordQ7_h[cont+17*numno] = Z_h[pos_ijk+(imax*jmax)];
							
							// Node 7:
							CoordQ7_h[cont+18*numno] = X_h[pos_ijk+(imax*jmax)+imax];
							CoordQ7_h[cont+19*numno] = Y_h[pos_ijk+(imax*jmax)+imax];
							CoordQ7_h[cont+20*numno] = Z_h[pos_ijk+(imax*jmax)+imax];
							
							// Node 8:
							CoordQ7_h[cont+21*numno] = X_h[pos_ijk+(imax*jmax)+imax-1];
							CoordQ7_h[cont+22*numno] = Y_h[pos_ijk+(imax*jmax)+imax-1];
							CoordQ7_h[cont+23*numno] = Z_h[pos_ijk+(imax*jmax)+imax-1];
							
						}
						
					}
					
				}
				
				// ****************************** Evaluating CoordQ8_h ******************************
				
				if(i != imax-1) {
				
					if(j != jmax-1) {
					
						if(k != kmax-1) { 
											
							// Node 1:
							CoordQ8_h[cont+ 0*numno] = X_h[pos_ijk];
							CoordQ8_h[cont+ 1*numno] = Y_h[pos_ijk];
							CoordQ8_h[cont+ 2*numno] = Z_h[pos_ijk];
							
							// Node 2:
							CoordQ8_h[cont+ 3*numno] = X_h[pos_ijk+1];
							CoordQ8_h[cont+ 4*numno] = Y_h[pos_ijk+1];
							CoordQ8_h[cont+ 5*numno] = Z_h[pos_ijk+1];
							
							// Node 3:
							CoordQ8_h[cont+ 6*numno] = X_h[pos_ijk+imax+1];
							CoordQ8_h[cont+ 7*numno] = Y_h[pos_ijk+imax+1];
							CoordQ8_h[cont+ 8*numno] = Z_h[pos_ijk+imax+1];
							
							// Node 4:
							CoordQ8_h[cont+ 9*numno] = X_h[pos_ijk+imax];
							CoordQ8_h[cont+10*numno] = Y_h[pos_ijk+imax];
							CoordQ8_h[cont+11*numno] = Z_h[pos_ijk+imax];
						
							// Node 5:
							CoordQ8_h[cont+12*numno] = X_h[pos_ijk+(imax*jmax)];
							CoordQ8_h[cont+13*numno] = Y_h[pos_ijk+(imax*jmax)];
							CoordQ8_h[cont+14*numno] = Z_h[pos_ijk+(imax*jmax)];
							
							// Node 6:
							CoordQ8_h[cont+15*numno] = X_h[pos_ijk+(imax*jmax)+1];
							CoordQ8_h[cont+16*numno] = Y_h[pos_ijk+(imax*jmax)+1];
							CoordQ8_h[cont+17*numno] = Z_h[pos_ijk+(imax*jmax)+1];
							
							// Node 7:
							CoordQ8_h[cont+18*numno] = X_h[pos_ijk+(imax*jmax)+imax+1];
							CoordQ8_h[cont+19*numno] = Y_h[pos_ijk+(imax*jmax)+imax+1];
							CoordQ8_h[cont+20*numno] = Z_h[pos_ijk+(imax*jmax)+imax+1];
							
							// Node 8:
							CoordQ8_h[cont+21*numno] = X_h[pos_ijk+(imax*jmax)+imax];
							CoordQ8_h[cont+22*numno] = Y_h[pos_ijk+(imax*jmax)+imax];
							CoordQ8_h[cont+23*numno] = Z_h[pos_ijk+(imax*jmax)+imax];
							
						}
						
					}
					
				}
				
				cont++;
				
			}
				
		}
		
	}
	
	/*int totalnodes = imax*jmax*kmax;
	cont =1;
	for(i=0; i<totalnodes; i++) {
		printf("cont1 = %d  \n", i+1);
		for(j=0; j<24; j++) {
			printf("%f  ", CoordQ8_h[i+numno*j]);
			if(cont==3) {
				printf("\n");
				cont=0;
			}
			cont++;

		}

	}*/

}

//==============================================================================
// Writing strain state in output file - GPUStrainState:

void WriteStressState(int numel, double *Stress_h)
{
	int i, j;
	FILE *outFile;
    
    outFile = fopen("GPUStressState.pos", "w");
    
    fprintf(outFile, "\n%Results - Stress State at Integration Points \n");
	fprintf(outFile, "\n");

	// Print results

	for(i=0; i<numel; i++) {
 
		fprintf(outFile, "Element %-4d \n", i+1);

		for(j=0; j<8; j++) {  // 8 = number of integration points

			fprintf(outFile, "  IP %d  %+0.8e %+0.8e %+0.8e %+0.8e %+0.8e %+0.8e", j+1, Stress_h[i+0*numel+j*numel*6], Stress_h[i+1*numel+j*numel*6],
																		           Stress_h[i+2*numel+j*numel*6], Stress_h[i+3*numel+j*numel*6],
																		           Stress_h[i+4*numel+j*numel*6], Stress_h[i+5*numel+j*numel*6]);

			fprintf(outFile, "\n");

		}
		
		fprintf(outFile, "\n");
		
	}
	
	fclose(outFile);
		
}

//==============================================================================
// Writing strain state in output file - GPUStrainState:

void WriteStrainState(int numel, double *Strain_h)
{
	int i, j;
	FILE *outFile;
    
    outFile = fopen("GPUStrainState.pos", "w");
    
    fprintf(outFile, "\n%Results - Strain State at Integration Points \n");
	fprintf(outFile, "\n");

	// Print results

	for(i=0; i<numel; i++) {

		fprintf(outFile, "Element %-4d \n", i+1);

		fprintf(outFile, "  %+0.8e %+0.8e %+0.8e %+0.8e %+0.8e %+0.8e", Strain_h[i+0*numel+j*numel*6], Strain_h[i+1*numel+j*numel*6],
			                                                            Strain_h[i+2*numel+j*numel*6], Strain_h[i+3*numel+j*numel*6],
			                                                            Strain_h[i+4*numel+j*numel*6], Strain_h[i+5*numel+j*numel*6]);

		fprintf(outFile, "\n");
		
	}
	
	fclose(outFile);
		
}

/*void WriteStrainState(int numel, double *Strain_h)
{
	int i, j;
	FILE *outFile;
    
    outFile = fopen("GPUStrainState.pos", "w");
    
    fprintf(outFile, "\n%Results - Strain State at Integration Points \n");
	fprintf(outFile, "\n");

	// Print results

	for(i=0; i<numel; i++) {
 
		fprintf(outFile, "Element %-4d \n", i+1);

		for(j=0; j<8; j++) {  // 8 = number of integration points

			fprintf(outFile, "  IP %d  %+0.8e %+0.8e %+0.8e %+0.8e %+0.8e %+0.8e", j+1, Strain_h[i+0*numel+j*numel*6], Strain_h[i+1*numel+j*numel*6],
																		           Strain_h[i+2*numel+j*numel*6], Strain_h[i+3*numel+j*numel*6],
																		           Strain_h[i+4*numel+j*numel*6], Strain_h[i+5*numel+j*numel*6]);

			fprintf(outFile, "\n");

		}
		
		fprintf(outFile, "\n");
		
	}
	
	fclose(outFile);
		
}*/

//==============================================================================
// Writing nodal displacement in output file - GPUDisplacement:

void WriteDisplacement(int numno, double *Vector_X_h)
{
	int i;
	FILE *outFile;
    
    outFile = fopen("GPUDisplacement.pos", "w");
    
    fprintf(outFile, "\n%%RESULT.CASE.STEP.NODAL.DISPLACEMENT");
	fprintf(outFile, "\n%d  'Displacement'\n", numno);

	// Print results

	for(i=0; i<numno; i++) {
 
		fprintf(outFile, "%-4d ", i+1);
		
		fprintf(outFile, " %+0.8e %+0.8e %+0.8e", Vector_X_h[3*i], Vector_X_h[3*i+1], Vector_X_h[3*i+2]);
		
		fprintf(outFile, "\n");
		
	}
	
	fclose(outFile);
		
}

//==============================================================================
// Evaluating nodal coordinate for each element

void EvaluateCoordStrainState(int numel, int numdof, int numnoel, int *Connect_h,  double *X_h,  double *Y_h,  double *Z_h,  double *CoordElem_h)
{
	int i, j;
	
	for(i=0; i<numel; i++) {
	
		for(j=0; j<numnoel; j++) {
	
			CoordElem_h[i + (3*j)  *numel] = X_h[ Connect_h[i + j*numel]-1 ];
			CoordElem_h[i + (3*j+1)*numel] = Y_h[ Connect_h[i + j*numel]-1 ];
			CoordElem_h[i + (3*j+2)*numel] = Z_h[ Connect_h[i + j*numel]-1 ];
			
			//printf("   %0.1f %0.1f %0.1f", CoordElem_h[i+(3*j)*NumThreadElem], CoordElem_h[i+(3*j+1)*NumThreadElem], CoordElem_h[i+(3*j+2)*NumThreadElem]);
		}
		
		//printf("\n\n");
		
	}

}