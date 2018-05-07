//
//   o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o 	
//   o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o 	
//	
//         C C C C C C	         U U                 U U     D D D D D D D D D                 A A A A A A
//       C C C C C C C C	     U U                 U U     D D D D D D D D D D             A A A A A A A A
//     C C             C C       U U                 U U     D D               D D         A A             A A
//   C C                 C C     U U                 U U     D D                 D D     A A                 A A      
//   C C                         U U                 U U     D D                 D D     A A                 A A             
//   C C                         U U                 U U     D D                 D D     A A                 A A
//   C C                         U U                 U U     D D                 D D     A A A A A A A A A A A A
//   C C                         U U                 U U     D D                 D D     A A A A A A A A A A A A           
//   C C                         U U                 U U     D D                 D D     A A                 A A
//     C C             C C         U U             U U       D D               D D       A A                 A A
//       C C C C C C C C             U U U U U U U U         D D D D D D D D D D         A A                 A A
//         C C C C C C	               U U U U U U           D D D D D D D D D           A A                 A A
//
//   o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o 	
//   o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o 


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <omp.h>
#include <cuda_runtime.h>
 
//---------------------------------------------------------------------------
// Functions declaration									 
//---------------------------------------------------------------------------

extern "C" void GPU_AssemblyStiffnessHexahedron(int numno, int numdof, int RowSize, int NumThread, int BlockSizeX, int BlockSizeY, double *Matrix_K_d, 
double *Matrix_K_Aux_d, double *Matrix_M_d, double *MatPropQ1_d, double *MatPropQ2_d, double *MatPropQ3_d, double *MatPropQ4_d, double *MatPropQ5_d, 
double *MatPropQ6_d, double *MatPropQ7_d, double *MatPropQ8_d, double *CoordQ1_d, double *CoordQ2_d, double *CoordQ3_d, double *CoordQ4_d, double *CoordQ5_d, 
double *CoordQ6_d, double *CoordQ7_d, double *CoordQ8_d, int *iiPosition_d);
 
extern "C" void AssemblyStiffnessGPUTetrahedron(int numdof, int numdofel, int numno, int maxThreadsPerBlock, int RowSize, int BlockDimX, int BlockDimY,
double *Matrix_K_h, double *Matrix_K_d, double *Matrix_K_Aux_d, int NumThread, double *abcParam_h, double *abcParam_d, int MaxNodeAround, int MaxElemPerEdge);

extern "C" void EvaluateStressStateGPU(int numel, int BlockDimX, int BlockDimY, double *Material_Data_d, double *Strain_d, double *Stress_d);
                 
  __global__ void AssemblyStiffness3DQ1(int numno, int RowSize, double *CoordQ1_d, double *MatPropQ1_d, double *Matrix_K_d);
  
  __global__ void AssemblyStiffness3DQ2(int numno, int RowSize, double *CoordQ2_d, double *MatPropQ2_d, double *Matrix_K_d);
  
  __global__ void AssemblyStiffness3DQ3(int numno, int RowSize, double *CoordQ3_d, double *MatPropQ3_d, double *Matrix_K_d);
  
  __global__ void AssemblyStiffness3DQ4(int numno, int RowSize, double *CoordQ4_d, double *MatPropQ4_d, double *Matrix_K_d);
  
  __global__ void AssemblyStiffness3DQ5(int numno, int RowSize, double *CoordQ5_d, double *MatPropQ5_d, double *Matrix_K_d);
  
  __global__ void AssemblyStiffness3DQ6(int numno, int RowSize, double *CoordQ6_d, double *MatPropQ6_d, double *Matrix_K_d);
  
  __global__ void AssemblyStiffness3DQ7(int numno, int RowSize, double *CoordQ7_d, double *MatPropQ7_d, double *Matrix_K_d);
  
  __global__ void AssemblyStiffness3DQ8(int numno, int RowSize, double *CoordQ8_d, double *MatPropQ8_d, double *Matrix_K_d);
  
  __global__ void TrasnsposeCoalesced(int NumThread, double *Matrix_K_Aux_d, double *Matrix_K_d);
  
  __global__ void AssemblyStiffnessTetrahedron(int RowSize, double *Matrix_K_d, double *abcParam_d, int MaxNodeAround, int MaxElemPerEdge);
  
  __global__ void EvaluateMatrixM(int numdof, int numno, int *iiPosition_d, double *Matrix_K_d, double *Matrix_M_d);
  
  __global__ void SUBT_V_V(int numdof, int numno, double *Vector_B_d, double *Vector_R_d);  // Gradient Conjugate Method
   
  __global__ void MULT_DM_V(int numdof, int numno, double *Matrix_M_d, double *Vector_R_d, double *Vector_D_d);
   
  __global__ void MULT_V_V_1024(int numdof, int numno, double *Vector_R_d, double *Vector_D_d, double *delta_new_d);
  
  __global__ void MULT_V_V_512(int numdof, int numno, double *Vector_R_d, double *Vector_D_d, double *delta_new_d);
   
  __global__ void SUM_V_4 (int numdof, int numno, double *delta_new_d, double *delta_new_V_d);
 
  __global__ void SUM_V_8 (int numdof, int numno, double *delta_new_d, double *delta_new_V_d);
 
  __global__ void SUM_V_16(int numdof, int numno, double *delta_new_d, double *delta_new_V_d);

  __global__ void SUM_V_32(int numdof, int numno, double *delta_new_d, double *delta_new_V_d);
  
  __global__ void MULT_SM_V_128(int numdof, int numno, int RowSize, double *Matrix_A_d, double *Vector_X_d, int *Vector_I_d, double *Vector_R_d);
  
  __global__ void MULT_SM_V_Text_128(int numdof, int numno, int RowSize, double *Matrix_A_d, double *Vector_X_d, int *Vector_I_d, double *Vector_R_d);
  
  __global__ void ADD_V_V_X(int numdof, int numno, double *Vector_X_d, double *Vector_D_d, double *delta_new_V_d, double *delta_aux_V_d, double *Vector_R_d, double *Vector_Q_d); 
  
  __global__ void UPDATE_DELTA(int numdof, int numno, double *delta_new_V_d, double *delta_old_V_d);
  
  __global__ void ADD_V_V_D(int numdof, int numno, double *Vector_R_d, double *Vector_D_d, double *delta_new_V_d, double *delta_old_V_d);
  
  __global__ void EvaluateStrainState(int numel, int RowSize, double *CoordElem_d, int *Connect_d, double *Vector_X_d, double *Strain_d);
  
  __global__ void EvaluateStressState(int numel, double *Material_Data_d, double *Strain_d, double *Stress_d);
  
  __global__ void SUM_GPU_4 (int numdof, int numno, double *delta_new_V_d);
  
  __global__ void Copy_Local_Global(int numdof, int numno, int position, double *Vector_D_d, double *Vector_D_Global_d);
  
  __global__ void Copy_for_Vector_4(double *delta_0_new_V_d, double *delta_0_GPU_d);
  
  __global__ void Clean_Global_Vector(int , int , double *);  

  // -------------------------------------------------------------------------------------------------------------------------

  __global__ void AssemblyStiffness3DQ1Mod(int numno, int RowSize, double *CoordQ1_d, double *MatPropQ1_d, double *Matrix_K_d);
  
  __global__ void AssemblyStiffness3DQ2Mod(int numno, int RowSize, double *CoordQ2_d, double *MatPropQ2_d, double *Matrix_K_d);
  
  __global__ void AssemblyStiffness3DQ3Mod(int numno, int RowSize, double *CoordQ3_d, double *MatPropQ3_d, double *Matrix_K_d);
  
  __global__ void AssemblyStiffness3DQ4Mod(int numno, int RowSize, double *CoordQ4_d, double *MatPropQ4_d, double *Matrix_K_d);
  
  __global__ void AssemblyStiffness3DQ5Mod(int numno, int RowSize, double *CoordQ5_d, double *MatPropQ5_d, double *Matrix_K_d);
  
  __global__ void AssemblyStiffness3DQ6Mod(int numno, int RowSize, double *CoordQ6_d, double *MatPropQ6_d, double *Matrix_K_d);
  
  __global__ void AssemblyStiffness3DQ7Mod(int numno, int RowSize, double *CoordQ7_d, double *MatPropQ7_d, double *Matrix_K_d);
  
  __global__ void AssemblyStiffness3DQ8Mod(int numno, int RowSize, double *CoordQ8_d, double *MatPropQ8_d, double *Matrix_K_d);

  // -------------------------------------------------------------------------------------------------------------------------
  
  extern "C" void Launch_SUBT_V_V(dim3 BlockSizeVector, dim3 ThreadsPerBlockXY, int numdof, int numno, double *Vector_B_d, double *Vector_R_d);
  extern "C" void Launch_CLEAR_V(dim3 BlockSizeVector, dim3 ThreadsPerBlockXY, int numdof, int numno, double *Vector_X_d);
  extern "C" void Launch_MULT_DM_V(dim3 , dim3 , int , int , double *, double *, double *);
  extern "C" void Launch_MULT_V_V(int , int , int , int , double *, double *, double *);
  extern "C" void Launch_SUM_V(dim3 , dim3 , int , int , int ,int, double *, double *, double *);
  extern "C" void Launch_MULT_SM_V_128(dim3 BlockSizeMatrix, dim3 ThreadsPerBlockXY, int NumMaxThreadsBlock, int numdof, int numno, int RowSize, double *Matrix_A_d, double *Vector_D_d, int *Vector_I_d, double *Vector_Q_d);
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
  extern "C" void Launch_Clean_Global_Vector(dim3 , dim3 , int , int , double *);
  
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

  extern "C" void Launch_EvaluateStrainStateAverage(dim3 numBlocksStrain, dim3 threadsPerBlockStrain, int Localnumel, int RowSize, 
  double *CoordElem_d, int *Connect_d, double *Vector_X_d, double *Strain_d);

  texture <double, 1, cudaReadModeElementType> texRef;
  
 //---------------------------------------------------------------------------------------------------------------------------
  
 void SolveGradientConjugate(int UseNumRows, int UseNumTerms, int NumMaxLines, int NumMaxTerms, int maxThreadsPerBlock, int multiProcessorCount, double *Matrix_M_d, 
 int *Vector_I_d, double *Matrix_K_d, double *Vector_B_h, double *Vector_B_d, double *Vector_X_h, double *Vector_X_d);
   
 //=============== x =============== x =============== x =============== x =============== x =============== x ===============

void GPU_AssemblyStiffnessHexahedron(int numno, int numdof, int RowSize, int NumThread, int BlockSizeX, int BlockSizeY, double *Matrix_K_d, 
double *Matrix_K_Aux_d, double *Matrix_M_d, double *MatPropQ1_d, double *MatPropQ2_d, double *MatPropQ3_d, double *MatPropQ4_d, double *MatPropQ5_d, 
double *MatPropQ6_d, double *MatPropQ7_d, double *MatPropQ8_d, double *CoordQ1_d, double *CoordQ2_d, double *CoordQ3_d, double *CoordQ4_d, double *CoordQ5_d, 
double *CoordQ6_d, double *CoordQ7_d, double *CoordQ8_d, int *iiPosition_d)
{
	//unsigned int hTimer1, hTimer2, hTimer3;
	//double Time1, Time2, Time3;
	
	using namespace std;
	
	// ==================================================================================================================================
	// Evaluating Stiffness Matrix (SM):
	
	printf("\n");
	printf("         Assembling Stiffness Matrix \n");
	printf("         ========================================= ");   
	
	//cutCreateTimer(&hTimer1);
	cudaThreadSynchronize();
	//cutResetTimer(hTimer1);
	//cutStartTimer(hTimer1);
		
	// Compute Capability 2.x and 3.0 => Maximum number of threads per block = 1024 (maxThreadsPerBlock)
	
	dim3 numBlocksSM(int(sqrt(double(numno))/BlockSizeX)+1, int(sqrt(double(numno))/BlockSizeY)+1);	
	dim3 threadsPerBlockSM(BlockSizeX, BlockSizeY); 
	
	// Assembly on GPU the Stiffness Matrix

	AssemblyStiffness3DQ1 <<<numBlocksSM, threadsPerBlockSM>>> (numno, RowSize, CoordQ1_d, MatPropQ1_d, Matrix_K_Aux_d);
	
	AssemblyStiffness3DQ2 <<<numBlocksSM, threadsPerBlockSM>>> (numno, RowSize, CoordQ2_d, MatPropQ2_d, Matrix_K_Aux_d);
	
	AssemblyStiffness3DQ3 <<<numBlocksSM, threadsPerBlockSM>>> (numno, RowSize, CoordQ3_d, MatPropQ3_d, Matrix_K_Aux_d);
	
	AssemblyStiffness3DQ4 <<<numBlocksSM, threadsPerBlockSM>>> (numno, RowSize, CoordQ4_d, MatPropQ4_d, Matrix_K_Aux_d);

	AssemblyStiffness3DQ5 <<<numBlocksSM, threadsPerBlockSM>>> (numno, RowSize, CoordQ5_d, MatPropQ5_d, Matrix_K_Aux_d);
	
	AssemblyStiffness3DQ6 <<<numBlocksSM, threadsPerBlockSM>>> (numno, RowSize, CoordQ6_d, MatPropQ6_d, Matrix_K_Aux_d);
	
	AssemblyStiffness3DQ7 <<<numBlocksSM, threadsPerBlockSM>>> (numno, RowSize, CoordQ7_d, MatPropQ7_d, Matrix_K_Aux_d);
	
	AssemblyStiffness3DQ8 <<<numBlocksSM, threadsPerBlockSM>>> (numno, RowSize, CoordQ8_d, MatPropQ8_d, Matrix_K_Aux_d);
	
	cudaThreadSynchronize();
	//cutStopTimer(hTimer1);
	//Time1 = cutGetTimerValue(hTimer1);
		
	printf("\n");
	//printf("         Assembly Stiffness Matrix Time: %0.3f s \n", Time1/CLOCKS_PER_SEC);
		
	// ==================================================================================================================================
	// Transpose (TR) Stiffness Matrix:
	
//	cutCreateTimer(&hTimer2);
	cudaThreadSynchronize();
//	cutResetTimer(hTimer2);
//	cutStartTimer(hTimer2);
		
	dim3 numBlocksTR(3*(RowSize/BlockSizeX), NumThread/BlockSizeY);	// X-Direction 3*96 (3 dof * RowSize)
	dim3 threadsPerBlockTR(BlockSizeX, BlockSizeY);
	
	// Transpose on GPU the Stiffness Matrix

	TrasnsposeCoalesced <<<numBlocksTR, threadsPerBlockTR>>> (NumThread, Matrix_K_Aux_d, Matrix_K_d);
	
	cudaThreadSynchronize();
	//cutStopTimer(hTimer2);
	//Time2 = cutGetTimerValue(hTimer2);
	
	//printf("         Transpose Stiffness Matrix Time: %0.3f s \n", Time2/CLOCKS_PER_SEC);
		
	// ==================================================================================================================================
	// Evaluating M matrix = 1/K:
	
	// Number of active terms (DOF x number of nodes)
	// numno = Number of nodes of the mesh  
		
	dim3 numBlocksM(int( sqrt(double(numdof*numno)) /BlockSizeX)+1, int( sqrt(double(numdof*numno)) /BlockSizeY)+1);
	dim3 threadsPerBlockM(BlockSizeX, BlockSizeY);
	
//	cutCreateTimer(&hTimer3);
	cudaThreadSynchronize();
//	cutResetTimer(hTimer3);
//	cutStartTimer(hTimer3);
	
	EvaluateMatrixM <<<numBlocksM, threadsPerBlockM>>> (numdof, numno, iiPosition_d, Matrix_K_d, Matrix_M_d);
	
	cudaThreadSynchronize();
//	cutStopTimer(hTimer3);
//	Time3 = cutGetTimerValue(hTimer3);
	
	//printf("         Evaluating M Matrix Time: %0.3f s \n", Time3/CLOCKS_PER_SEC);
			
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============

void EvaluateStressStateGPU(int numel, int BlockDimX, int BlockDimY, double *Material_Data_d, double *Strain_d, double *Stress_d)
{
	//unsigned int hTimer;
	//double Time;
	
	using namespace std;
	
	// ==================================================================================================================================
	// Evaluate Strain state
	
	printf("\n\n");
	printf("         Evaluating Stress State \n");
	printf("         ========================================= ");	
		
	// =================================================================
	
	dim3 numBlocksStrain(int( sqrt(double(numel)) /BlockDimX)+1, int( sqrt(double(numel)) /BlockDimY)+1);
	dim3 threadsPerBlockStrain(BlockDimX, BlockDimY);
	
	//cutCreateTimer(&hTimer);
	cudaThreadSynchronize();
	//cutResetTimer(hTimer);
	//cutStartTimer(hTimer);

	EvaluateStressState <<<numBlocksStrain, threadsPerBlockStrain>>> (numel, Material_Data_d, Strain_d, Stress_d);
	
	cudaThreadSynchronize();
	//cutStopTimer(hTimer);
//	Time = cutGetTimerValue(hTimer);
	
	printf("\n");
	//printf("         Evaluating Stress State Time: %0.3f s \n", Time/CLOCKS_PER_SEC);
	
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============

__global__ void EvaluateStressState(int numel, double *Material_Data_d, double *Strain_d, double *Stress_d) 
{

	int i, ig, jg, kg, cont;  // 3
	double E, p, sho, De[6][6];
	
	const int xIndex = blockIdx.x*blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y*blockDim.y + threadIdx.y;
	const int thread_id = (gridDim.x*blockDim.x)*yIndex + xIndex;
	
	__syncthreads();
	
	if(thread_id < numel) {
	
		E = Material_Data_d[thread_id        ];
		p = Material_Data_d[thread_id + numel];

		sho=(E*(1-p))/((1+p)*(1-2*p));

		De[0][0] = sho;
		De[0][1] = sho*(p/(1-p));
		De[0][2] = sho*(p/(1-p));
		De[0][3] = 0.;
		De[0][4] = 0.; 
		De[0][5] = 0.; 

		De[1][0] = sho*(p/(1-p));
		De[1][1] = sho;
		De[1][2] = sho*(p/(1-p));
		De[1][3] = 0.;
		De[1][4] = 0.; 
		De[1][5] = 0.; 

		De[2][0] = sho*(p/(1-p));
		De[2][1] = sho*(p/(1-p));
		De[2][2] = sho;
		De[2][3] = 0.;
		De[2][4] = 0.; 
		De[2][5] = 0.; 

		De[3][0] = 0.; 
		De[3][1] = 0.; 
		De[3][2] = 0.; 
		De[3][3] = sho*((1-2*p)/(2*(1-p)));
		De[3][4] = 0.; 
		De[3][5] = 0.; 

		De[4][0] = 0.; 
		De[4][1] = 0.; 
		De[4][2] = 0.; 
		De[4][3] = 0.; 
		De[4][4] = sho*((1-2*p)/(2*(1-p)));
		De[4][5] = 0.; 

		De[5][0] = 0.; 
		De[5][1] = 0.; 
		De[5][2] = 0.; 
		De[5][3] = 0.; 
		De[5][4] = 0.; 
		De[5][5] = sho*((1-2*p)/(2*(1-p)));
		
		cont = 0;
									
		for(ig=0; ig<2; ig++) {                                          // =================== Loop ig ===================

			for(jg=0; jg<2; jg++) {                                      // =================== Loop jg ===================
				
				for(kg=0; kg<2; kg++) {                                  // =================== Loop kg ===================			
											
					for(i=0; i<6; i++) {
						
						// Sxx
						Stress_d[thread_id+0*numel+cont*numel*6] += De[0][i]*Strain_d[thread_id+i*numel+cont*numel*6];
						
						// Syy
						Stress_d[thread_id+1*numel+cont*numel*6] += De[1][i]*Strain_d[thread_id+i*numel+cont*numel*6];
						
						// Szz
						Stress_d[thread_id+2*numel+cont*numel*6] += De[2][i]*Strain_d[thread_id+i*numel+cont*numel*6];
						
						// Sxy
						Stress_d[thread_id+3*numel+cont*numel*6] += De[3][i]*Strain_d[thread_id+i*numel+cont*numel*6];
						
						// Syz
						Stress_d[thread_id+4*numel+cont*numel*6] += De[4][i]*Strain_d[thread_id+i*numel+cont*numel*6];							  
						
						// Szx
						Stress_d[thread_id+5*numel+cont*numel*6] += De[5][i]*Strain_d[thread_id+i*numel+cont*numel*6];							                   
					}
														  
					cont++;
														  
				}                                                        // =================== Loop kg ===================
											
			}                                                            // =================== Loop jg ===================

		}                                                                // =================== Loop ig ===================
		
	}
	
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============

__global__ void EvaluateStrainState(int numel, int RowSize, double *CoordElem_d, int *Connect_d, double *Vector_X_d, double *Strain_d) 
{
	double r, s, t;  // 3
	double XEL0, YEL0, ZEL0, XEL1, YEL1, ZEL1, XEL2, YEL2, ZEL2, XEL3, YEL3, ZEL3;  // 12 
	double XEL4, YEL4, ZEL4, XEL5, YEL5, ZEL5, XEL6, YEL6, ZEL6, XEL7, YEL7, ZEL7;  // 12 
	double xgaus[2];  // 4
	double phi_r[8], phi_s[8], phi_t[8];  // 24
	double jac[3][3];  // 9
	double jacob;  // 1
	double gama[3][3];  // 9
	double auxX[8], auxY[8], auxZ[8];  // 24
	int i, ig, jg, kg, cont;  // 3
	
	const int xIndex = blockIdx.x*blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y*blockDim.y + threadIdx.y;
	const int thread_id = (gridDim.x*blockDim.x)*yIndex + xIndex;
	
	__syncthreads();
	
	if(thread_id < numel) {

		XEL0 = CoordElem_d[thread_id+ 0*numel];
		YEL0 = CoordElem_d[thread_id+ 1*numel];
		ZEL0 = CoordElem_d[thread_id+ 2*numel];   	
		XEL1 = CoordElem_d[thread_id+ 3*numel];
		YEL1 = CoordElem_d[thread_id+ 4*numel];  
		ZEL1 = CoordElem_d[thread_id+ 5*numel];  	
		XEL2 = CoordElem_d[thread_id+ 6*numel];
		YEL2 = CoordElem_d[thread_id+ 7*numel]; 
		ZEL2 = CoordElem_d[thread_id+ 8*numel];  		
		XEL3 = CoordElem_d[thread_id+ 9*numel];
		YEL3 = CoordElem_d[thread_id+10*numel];
		ZEL3 = CoordElem_d[thread_id+11*numel];
		
		XEL4 = CoordElem_d[thread_id+12*numel];
		YEL4 = CoordElem_d[thread_id+13*numel];
		ZEL4 = CoordElem_d[thread_id+14*numel];   	
		XEL5 = CoordElem_d[thread_id+15*numel];
		YEL5 = CoordElem_d[thread_id+16*numel];  
		ZEL5 = CoordElem_d[thread_id+17*numel];  	
		XEL6 = CoordElem_d[thread_id+18*numel];
		YEL6 = CoordElem_d[thread_id+19*numel]; 
		ZEL6 = CoordElem_d[thread_id+20*numel];  		
		XEL7 = CoordElem_d[thread_id+21*numel];
		YEL7 = CoordElem_d[thread_id+22*numel];
		ZEL7 = CoordElem_d[thread_id+23*numel];
		
		__syncthreads();

		xgaus[0]=-0.577350269189626;
		xgaus[1]= 0.577350269189626;
		
		cont = 0;
									
		for(ig=0; ig<2; ig++) {                                          // =================== Loop ig ===================
			t=xgaus[ig];
					
			for(jg=0; jg<2; jg++) {                                      // =================== Loop jg ===================
				s=xgaus[jg]; 
				
				for(kg=0; kg<2; kg++) {                                  // =================== Loop kg ===================
				r=xgaus[kg]; 
							
					// Shape function derivative:
					phi_r[0] = -0.125*(1.0-s)*(1.0-t);
					phi_r[1] =  0.125*(1.0-s)*(1.0-t);
					phi_r[2] =  0.125*(1.0+s)*(1.0-t);
					phi_r[3] = -0.125*(1.0+s)*(1.0-t);
					phi_r[4] = -0.125*(1.0-s)*(1.0+t);
					phi_r[5] =  0.125*(1.0-s)*(1.0+t);
					phi_r[6] =  0.125*(1.0+s)*(1.0+t);
					phi_r[7] = -0.125*(1.0+s)*(1.0+t);
	 
					phi_s[0] = -0.125*(1.0-r)*(1.0-t);
					phi_s[1] = -0.125*(1.0+r)*(1.0-t);
					phi_s[2] =  0.125*(1.0+r)*(1.0-t);
					phi_s[3] =  0.125*(1.0-r)*(1.0-t);
					phi_s[4] = -0.125*(1.0-r)*(1.0+t);
					phi_s[5] = -0.125*(1.0+r)*(1.0+t);
					phi_s[6] =  0.125*(1.0+r)*(1.0+t);
					phi_s[7] =  0.125*(1.0-r)*(1.0+t);
		 
					phi_t[0] = -0.125*(1.0-r)*(1.0-s);
					phi_t[1] = -0.125*(1.0+r)*(1.0-s);
					phi_t[2] = -0.125*(1.0+r)*(1.0+s);
					phi_t[3] = -0.125*(1.0-r)*(1.0+s);
					phi_t[4] =  0.125*(1.0-r)*(1.0-s);
					phi_t[5] =  0.125*(1.0+r)*(1.0-s);
					phi_t[6] =  0.125*(1.0+r)*(1.0+s);
					phi_t[7] =  0.125*(1.0-r)*(1.0+s);
					
					// Terms of the Jacobian matrix:
					jac[0][0]=phi_r[0]*XEL0+phi_r[1]*XEL1+phi_r[2]*XEL2+phi_r[3]*XEL3+phi_r[4]*XEL4+phi_r[5]*XEL5+phi_r[6]*XEL6+phi_r[7]*XEL7;
					jac[0][1]=phi_r[0]*YEL0+phi_r[1]*YEL1+phi_r[2]*YEL2+phi_r[3]*YEL3+phi_r[4]*YEL4+phi_r[5]*YEL5+phi_r[6]*YEL6+phi_r[7]*YEL7;
					jac[0][2]=phi_r[0]*ZEL0+phi_r[1]*ZEL1+phi_r[2]*ZEL2+phi_r[3]*ZEL3+phi_r[4]*ZEL4+phi_r[5]*ZEL5+phi_r[6]*ZEL6+phi_r[7]*ZEL7;

					jac[1][0]=phi_s[0]*XEL0+phi_s[1]*XEL1+phi_s[2]*XEL2+phi_s[3]*XEL3+phi_s[4]*XEL4+phi_s[5]*XEL5+phi_s[6]*XEL6+phi_s[7]*XEL7;
					jac[1][1]=phi_s[0]*YEL0+phi_s[1]*YEL1+phi_s[2]*YEL2+phi_s[3]*YEL3+phi_s[4]*YEL4+phi_s[5]*YEL5+phi_s[6]*YEL6+phi_s[7]*YEL7;
					jac[1][2]=phi_s[0]*ZEL0+phi_s[1]*ZEL1+phi_s[2]*ZEL2+phi_s[3]*ZEL3+phi_s[4]*ZEL4+phi_s[5]*ZEL5+phi_s[6]*ZEL6+phi_s[7]*ZEL7;
					
					jac[2][0]=phi_t[0]*XEL0+phi_t[1]*XEL1+phi_t[2]*XEL2+phi_t[3]*XEL3+phi_t[4]*XEL4+phi_t[5]*XEL5+phi_t[6]*XEL6+phi_t[7]*XEL7;
					jac[2][1]=phi_t[0]*YEL0+phi_t[1]*YEL1+phi_t[2]*YEL2+phi_t[3]*YEL3+phi_t[4]*YEL4+phi_t[5]*YEL5+phi_t[6]*YEL6+phi_t[7]*YEL7;
					jac[2][2]=phi_t[0]*ZEL0+phi_t[1]*ZEL1+phi_t[2]*ZEL2+phi_t[3]*ZEL3+phi_t[4]*ZEL4+phi_t[5]*ZEL5+phi_t[6]*ZEL6+phi_t[7]*ZEL7;
						
					// Jacobian matrix determinant:	
					jacob = jac[0][0]*(jac[1][1]*jac[2][2]-jac[1][2]*jac[2][1])-jac[0][1]*(jac[1][0]*jac[2][2]-jac[1][2]*jac[2][0])+
							jac[0][2]*(jac[1][0]*jac[2][1]-jac[1][1]*jac[2][0]);
					
					if(jacob > 0.) {
					
						gama[0][0] = (jac[1][1]*jac[2][2] - jac[1][2]*jac[2][1]) /jacob;
						gama[0][1] =-(jac[0][1]*jac[2][2] - jac[0][2]*jac[2][1]) /jacob;
						gama[0][2] = (jac[0][1]*jac[1][2] - jac[0][2]*jac[1][1]) /jacob;
						
						gama[1][0] =-(jac[1][0]*jac[2][2] - jac[1][2]*jac[2][0]) /jacob;
						gama[1][1] = (jac[0][0]*jac[2][2] - jac[0][2]*jac[2][0]) /jacob;
						gama[1][2] =-(jac[0][0]*jac[1][2] - jac[0][2]*jac[1][0]) /jacob;
						
						gama[2][0] = (jac[1][0]*jac[2][1] - jac[1][1]*jac[2][0]) /jacob;
						gama[2][1] =-(jac[0][0]*jac[2][1] - jac[0][1]*jac[2][0]) /jacob;
						gama[2][2] = (jac[0][0]*jac[1][1] - jac[0][1]*jac[1][0]) /jacob;
						
						// *******************************************************************************************************
						for(i=0; i<8; i++) {  // Number of element nodes
								
							auxX[i] = gama[0][0]*phi_r[i]+gama[0][1]*phi_s[i]+gama[0][2]*phi_t[i];
							auxY[i] = gama[1][0]*phi_r[i]+gama[1][1]*phi_s[i]+gama[1][2]*phi_t[i];
							auxZ[i] = gama[2][0]*phi_r[i]+gama[2][1]*phi_s[i]+gama[2][2]*phi_t[i];
							
						}
						
						// *******************************************************************************************************
						
						for(i=0; i<8; i++) {
						
							// Exx
							Strain_d[thread_id+0*numel+cont*numel*6] += auxX[i]*Vector_X_d[ 3*(Connect_d[thread_id + i*numel]-1)   ];
						
							// Eyy
							Strain_d[thread_id+1*numel+cont*numel*6] += auxY[i]*Vector_X_d[ 3*(Connect_d[thread_id + i*numel]-1)+1 ];
						
							// Ezz
							Strain_d[thread_id+2*numel+cont*numel*6] += auxZ[i]*Vector_X_d[ 3*(Connect_d[thread_id + i*numel]-1)+2 ];
						
							// Exy
							Strain_d[thread_id+3*numel+cont*numel*6] += auxY[i]*Vector_X_d[ 3*(Connect_d[thread_id + i*numel]-1)   ] +
																		auxX[i]*Vector_X_d[ 3*(Connect_d[thread_id + i*numel]-1)+1 ];
						
							// Eyz
							Strain_d[thread_id+4*numel+cont*numel*6] += auxZ[i]*Vector_X_d[ 3*(Connect_d[thread_id + i*numel]-1)+1 ] +
																		auxY[i]*Vector_X_d[ 3*(Connect_d[thread_id + i*numel]-1)+2 ];								  
						
							// Ezx
							Strain_d[thread_id+5*numel+cont*numel*6] += auxZ[i]*Vector_X_d[ 3*(Connect_d[thread_id + i*numel]-1)   ] +
																		auxX[i]*Vector_X_d[ 3*(Connect_d[thread_id + i*numel]-1)+2 ];							                   
						}
														  
						cont++;
														  
					}
					
				}                                                        // =================== Loop kg ===================
											
			}                                                            // =================== Loop jg ===================

		}                                                                // =================== Loop ig ===================
		
	}
	
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============

__global__ void EvaluateStrainStateAverage(int numel, int RowSize, double *CoordElem_d, int *Connect_d, double *Vector_X_d, double *Strain_d) 
{
	double r, s, t;  // 3
	double XEL0, YEL0, ZEL0, XEL1, YEL1, ZEL1, XEL2, YEL2, ZEL2, XEL3, YEL3, ZEL3;  // 12 
	double XEL4, YEL4, ZEL4, XEL5, YEL5, ZEL5, XEL6, YEL6, ZEL6, XEL7, YEL7, ZEL7;  // 12 
	double xgaus[2];  // 4
	double phi_r[8], phi_s[8], phi_t[8];  // 24
	double jac[3][3];  // 9
	double jacob;  // 1
	double gama[3][3];  // 9
	double auxX[8], auxY[8], auxZ[8];  // 24
	int i, ig, jg, kg;  // 3
	
	const int xIndex = blockIdx.x*blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y*blockDim.y + threadIdx.y;
	const int thread_id = (gridDim.x*blockDim.x)*yIndex + xIndex;
	
	__syncthreads();
	
	if(thread_id < numel) {

		XEL0 = CoordElem_d[thread_id+ 0*numel];
		YEL0 = CoordElem_d[thread_id+ 1*numel];
		ZEL0 = CoordElem_d[thread_id+ 2*numel];   	
		XEL1 = CoordElem_d[thread_id+ 3*numel];
		YEL1 = CoordElem_d[thread_id+ 4*numel];  
		ZEL1 = CoordElem_d[thread_id+ 5*numel];  	
		XEL2 = CoordElem_d[thread_id+ 6*numel];
		YEL2 = CoordElem_d[thread_id+ 7*numel]; 
		ZEL2 = CoordElem_d[thread_id+ 8*numel];  		
		XEL3 = CoordElem_d[thread_id+ 9*numel];
		YEL3 = CoordElem_d[thread_id+10*numel];
		ZEL3 = CoordElem_d[thread_id+11*numel];
		
		XEL4 = CoordElem_d[thread_id+12*numel];
		YEL4 = CoordElem_d[thread_id+13*numel];
		ZEL4 = CoordElem_d[thread_id+14*numel];   	
		XEL5 = CoordElem_d[thread_id+15*numel];
		YEL5 = CoordElem_d[thread_id+16*numel];  
		ZEL5 = CoordElem_d[thread_id+17*numel];  	
		XEL6 = CoordElem_d[thread_id+18*numel];
		YEL6 = CoordElem_d[thread_id+19*numel]; 
		ZEL6 = CoordElem_d[thread_id+20*numel];  		
		XEL7 = CoordElem_d[thread_id+21*numel];
		YEL7 = CoordElem_d[thread_id+22*numel];
		ZEL7 = CoordElem_d[thread_id+23*numel];
		
		__syncthreads();

		xgaus[0]=-0.577350269189626;
		xgaus[1]= 0.577350269189626;
									
		for(ig=0; ig<2; ig++) {                                          // =================== Loop ig ===================
			t=xgaus[ig];
					
			for(jg=0; jg<2; jg++) {                                      // =================== Loop jg ===================
				s=xgaus[jg]; 
				
				for(kg=0; kg<2; kg++) {                                  // =================== Loop kg ===================
				r=xgaus[kg]; 
							
					// Shape function derivative:
					phi_r[0] = -0.125*(1.0-s)*(1.0-t);
					phi_r[1] =  0.125*(1.0-s)*(1.0-t);
					phi_r[2] =  0.125*(1.0+s)*(1.0-t);
					phi_r[3] = -0.125*(1.0+s)*(1.0-t);
					phi_r[4] = -0.125*(1.0-s)*(1.0+t);
					phi_r[5] =  0.125*(1.0-s)*(1.0+t);
					phi_r[6] =  0.125*(1.0+s)*(1.0+t);
					phi_r[7] = -0.125*(1.0+s)*(1.0+t);
	 
					phi_s[0] = -0.125*(1.0-r)*(1.0-t);
					phi_s[1] = -0.125*(1.0+r)*(1.0-t);
					phi_s[2] =  0.125*(1.0+r)*(1.0-t);
					phi_s[3] =  0.125*(1.0-r)*(1.0-t);
					phi_s[4] = -0.125*(1.0-r)*(1.0+t);
					phi_s[5] = -0.125*(1.0+r)*(1.0+t);
					phi_s[6] =  0.125*(1.0+r)*(1.0+t);
					phi_s[7] =  0.125*(1.0-r)*(1.0+t);
		 
					phi_t[0] = -0.125*(1.0-r)*(1.0-s);
					phi_t[1] = -0.125*(1.0+r)*(1.0-s);
					phi_t[2] = -0.125*(1.0+r)*(1.0+s);
					phi_t[3] = -0.125*(1.0-r)*(1.0+s);
					phi_t[4] =  0.125*(1.0-r)*(1.0-s);
					phi_t[5] =  0.125*(1.0+r)*(1.0-s);
					phi_t[6] =  0.125*(1.0+r)*(1.0+s);
					phi_t[7] =  0.125*(1.0-r)*(1.0+s);
					
					// Terms of the Jacobian matrix:
					jac[0][0]=phi_r[0]*XEL0+phi_r[1]*XEL1+phi_r[2]*XEL2+phi_r[3]*XEL3+phi_r[4]*XEL4+phi_r[5]*XEL5+phi_r[6]*XEL6+phi_r[7]*XEL7;
					jac[0][1]=phi_r[0]*YEL0+phi_r[1]*YEL1+phi_r[2]*YEL2+phi_r[3]*YEL3+phi_r[4]*YEL4+phi_r[5]*YEL5+phi_r[6]*YEL6+phi_r[7]*YEL7;
					jac[0][2]=phi_r[0]*ZEL0+phi_r[1]*ZEL1+phi_r[2]*ZEL2+phi_r[3]*ZEL3+phi_r[4]*ZEL4+phi_r[5]*ZEL5+phi_r[6]*ZEL6+phi_r[7]*ZEL7;

					jac[1][0]=phi_s[0]*XEL0+phi_s[1]*XEL1+phi_s[2]*XEL2+phi_s[3]*XEL3+phi_s[4]*XEL4+phi_s[5]*XEL5+phi_s[6]*XEL6+phi_s[7]*XEL7;
					jac[1][1]=phi_s[0]*YEL0+phi_s[1]*YEL1+phi_s[2]*YEL2+phi_s[3]*YEL3+phi_s[4]*YEL4+phi_s[5]*YEL5+phi_s[6]*YEL6+phi_s[7]*YEL7;
					jac[1][2]=phi_s[0]*ZEL0+phi_s[1]*ZEL1+phi_s[2]*ZEL2+phi_s[3]*ZEL3+phi_s[4]*ZEL4+phi_s[5]*ZEL5+phi_s[6]*ZEL6+phi_s[7]*ZEL7;
					
					jac[2][0]=phi_t[0]*XEL0+phi_t[1]*XEL1+phi_t[2]*XEL2+phi_t[3]*XEL3+phi_t[4]*XEL4+phi_t[5]*XEL5+phi_t[6]*XEL6+phi_t[7]*XEL7;
					jac[2][1]=phi_t[0]*YEL0+phi_t[1]*YEL1+phi_t[2]*YEL2+phi_t[3]*YEL3+phi_t[4]*YEL4+phi_t[5]*YEL5+phi_t[6]*YEL6+phi_t[7]*YEL7;
					jac[2][2]=phi_t[0]*ZEL0+phi_t[1]*ZEL1+phi_t[2]*ZEL2+phi_t[3]*ZEL3+phi_t[4]*ZEL4+phi_t[5]*ZEL5+phi_t[6]*ZEL6+phi_t[7]*ZEL7;
						
					// Jacobian matrix determinant:	
					jacob = jac[0][0]*(jac[1][1]*jac[2][2]-jac[1][2]*jac[2][1])-jac[0][1]*(jac[1][0]*jac[2][2]-jac[1][2]*jac[2][0])+
							jac[0][2]*(jac[1][0]*jac[2][1]-jac[1][1]*jac[2][0]);
					
					if(jacob > 0.) {
					
						gama[0][0] = (jac[1][1]*jac[2][2] - jac[1][2]*jac[2][1]) /jacob;
						gama[0][1] =-(jac[0][1]*jac[2][2] - jac[0][2]*jac[2][1]) /jacob;
						gama[0][2] = (jac[0][1]*jac[1][2] - jac[0][2]*jac[1][1]) /jacob;
						
						gama[1][0] =-(jac[1][0]*jac[2][2] - jac[1][2]*jac[2][0]) /jacob;
						gama[1][1] = (jac[0][0]*jac[2][2] - jac[0][2]*jac[2][0]) /jacob;
						gama[1][2] =-(jac[0][0]*jac[1][2] - jac[0][2]*jac[1][0]) /jacob;
						
						gama[2][0] = (jac[1][0]*jac[2][1] - jac[1][1]*jac[2][0]) /jacob;
						gama[2][1] =-(jac[0][0]*jac[2][1] - jac[0][1]*jac[2][0]) /jacob;
						gama[2][2] = (jac[0][0]*jac[1][1] - jac[0][1]*jac[1][0]) /jacob;
						
						// *******************************************************************************************************
						for(i=0; i<8; i++) {  // Number of element nodes
								
							auxX[i] = gama[0][0]*phi_r[i]+gama[0][1]*phi_s[i]+gama[0][2]*phi_t[i];
							auxY[i] = gama[1][0]*phi_r[i]+gama[1][1]*phi_s[i]+gama[1][2]*phi_t[i];
							auxZ[i] = gama[2][0]*phi_r[i]+gama[2][1]*phi_s[i]+gama[2][2]*phi_t[i];
							
						}
						
						// *******************************************************************************************************
						
						for(i=0; i<8; i++) {
						
							// Exx
							Strain_d[thread_id+0*numel] += (auxX[i]*Vector_X_d[ 3*(Connect_d[thread_id + i*numel]-1)   ])/8;
						
							// Eyy
							Strain_d[thread_id+1*numel] += (auxY[i]*Vector_X_d[ 3*(Connect_d[thread_id + i*numel]-1)+1 ])/8;
						
							// Ezz
							Strain_d[thread_id+2*numel] += (auxZ[i]*Vector_X_d[ 3*(Connect_d[thread_id + i*numel]-1)+2 ])/8;
						
							// Exy
							Strain_d[thread_id+3*numel] += (auxY[i]*Vector_X_d[ 3*(Connect_d[thread_id + i*numel]-1)   ] +
														    auxX[i]*Vector_X_d[ 3*(Connect_d[thread_id + i*numel]-1)+1 ])/8;
						
							// Eyz
							Strain_d[thread_id+4*numel] += (auxZ[i]*Vector_X_d[ 3*(Connect_d[thread_id + i*numel]-1)+1 ] +
														    auxY[i]*Vector_X_d[ 3*(Connect_d[thread_id + i*numel]-1)+2 ])/8;								  
						
							// Ezx
							Strain_d[thread_id+5*numel] += (auxZ[i]*Vector_X_d[ 3*(Connect_d[thread_id + i*numel]-1)   ] +
														    auxX[i]*Vector_X_d[ 3*(Connect_d[thread_id + i*numel]-1)+2 ])/8;							                   
						}
														  
					}
					
				}                                                        // =================== Loop kg ===================
											
			}                                                            // =================== Loop jg ===================

		}                                                                // =================== Loop ig ===================
		
	}
	
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============

// This kernel adds up a vector by a alfa * vector of n terms. 
  
__global__ void ADD_V_V_D(int numdof, int numno, double *Vector_R_d, double *Vector_D_d, double *delta_new_V_d, double *delta_old_V_d)                             
{	
	const int xIndex = blockIdx.x*blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y*blockDim.y + threadIdx.y;
	const int thread_id = (gridDim.x*blockDim.x)*yIndex + xIndex;
		
	if(thread_id < numdof*numno)
		Vector_D_d[thread_id] = Vector_R_d[thread_id] + (delta_new_V_d[thread_id]/delta_old_V_d[thread_id]) * Vector_D_d[thread_id];
			
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============

// This kernel update the delta_old vector. 
__global__ void UPDATE_DELTA(int numdof, int numno, double *delta_new_V_d, double *delta_old_V_d)
{

	const int xIndex = blockIdx.x*blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y*blockDim.y + threadIdx.y;
	const int thread_id = (gridDim.x*blockDim.x)*yIndex + xIndex;
		
	if(thread_id < numdof*numno)
		delta_old_V_d[thread_id] = delta_new_V_d[thread_id];	

}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============

// This kernel adds up a vector by a alfa * vector of n terms. 
  
__global__ void ADD_V_V_X(int numdof, int numno, double *Vector_X_d, double *Vector_D_d, double *delta_new_V_d, double *delta_aux_V_d, double *Vector_R_d, double *Vector_Q_d)                             
{	
	const int xIndex = blockIdx.x*blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y*blockDim.y + threadIdx.y;
	const int thread_id = (gridDim.x*blockDim.x)*yIndex + xIndex;
		
	if(thread_id < numdof*numno) {
		Vector_X_d[thread_id] += (delta_new_V_d[thread_id]/delta_aux_V_d[thread_id]) * Vector_D_d[thread_id];
		Vector_R_d[thread_id] -= (delta_new_V_d[thread_id]/delta_aux_V_d[thread_id]) * Vector_Q_d[thread_id];
	}
				
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============

// This kernel multiplies a matrix (128 terms by line) by a vector. 

__global__ void MULT_SM_V_Text_128(int numdof, int numno, int RowSize, double *Matrix_A_d, double *Vector_X_d, int *Vector_I_d, double *Vector_R_d)
{

	int row, lane;
	
	__shared__ double vals[1024];
	//__shared__ double vals[512];
	
	const int thread_block_id = blockDim.x*threadIdx.y + threadIdx.x;
	
	const int thread_id = (gridDim.x*blockDim.x)*( (blockIdx.y)*blockDim.y) + (blockIdx.x)*blockDim.x*blockDim.y + thread_block_id;
	
	// ******************************************************************************************************* 

	if(thread_id < numdof*numno*RowSize) {
	
		row  = thread_id/128;
		lane = thread_id & (128-1);
			
		//vals[thread_block_id] = Matrix_A_d[thread_id] * tex1Dfetch(texRef, Vector_I_d[thread_id]-1);																			     
		
		__syncthreads();
		if(lane < 64) vals[thread_block_id] += vals[thread_block_id + 64]; __syncthreads();													                          
		if(lane < 32) vals[thread_block_id] += vals[thread_block_id + 32]; __syncthreads();		
		if(lane < 16) vals[thread_block_id] += vals[thread_block_id + 16]; __syncthreads();																				  
		if(lane <  8) vals[thread_block_id] += vals[thread_block_id +  8]; __syncthreads();	
		if(lane <  4) vals[thread_block_id] += vals[thread_block_id +  4]; __syncthreads();	
		if(lane <  2) vals[thread_block_id] += vals[thread_block_id +  2]; __syncthreads();	
		if(lane <  1) vals[thread_block_id] += vals[thread_block_id +  1]; __syncthreads();	
														                          	
		if(lane == 0)
			Vector_R_d[row] = vals[thread_block_id];
		
	}
		
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============

// This kernel multiplies a matrix (128 terms by line) by a vector. 

__global__ void MULT_SM_V_128_1024(int numdof, int numno, int RowSize, double *Matrix_A_d, double *Vector_X_d, int *Vector_I_d, double *Vector_R_d)
{

	int lane;
	
	__shared__ double vals[1024];
	
	const int thread_block_id = blockDim.x*threadIdx.y + threadIdx.x;
	
	const int thread_id = (gridDim.x*blockDim.x)*(blockIdx.y*blockDim.y) + blockIdx.x*blockDim.x*blockDim.y + thread_block_id;
	
	// ******************************************************************************************************* 

	if(thread_id < numdof*numno*RowSize) {
	
		
		lane = thread_id & (128-1);
		//lane = thread_id - int(thread_id/128)*128;
		
		/*if( fabs( Vector_X_d[Vector_I_d[thread_id]-1] ) > 0.001) 	
			vals[thread_block_id] = Matrix_A_d[thread_id] * Vector_X_d[Vector_I_d[thread_id]-1];	
		else	
			vals[thread_block_id] = 0.;	*/
			
	
		
		if( Vector_I_d[thread_id] > 0) 	
			vals[thread_block_id] = Matrix_A_d[thread_id] * Vector_X_d[Vector_I_d[thread_id]-1];	
		else	
			vals[thread_block_id] = 0.;
		
																		     
		
		__syncthreads();
		if(lane < 64) vals[thread_block_id] += vals[thread_block_id + 64]; __syncthreads();													                          
		if(lane < 32) vals[thread_block_id] += vals[thread_block_id + 32]; __syncthreads();		
		if(lane < 16) vals[thread_block_id] += vals[thread_block_id + 16]; __syncthreads();																				  
		if(lane <  8) vals[thread_block_id] += vals[thread_block_id +  8]; __syncthreads();	
		if(lane <  4) vals[thread_block_id] += vals[thread_block_id +  4]; __syncthreads();	
		if(lane <  2) vals[thread_block_id] += vals[thread_block_id +  2]; __syncthreads();	
		if(lane <  1) vals[thread_block_id] += vals[thread_block_id +  1]; __syncthreads();
		
		if(lane == 0) 
			Vector_R_d[thread_id/128] = vals[thread_block_id];
		
	}
		
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============

// This kernel multiplies a matrix (128 terms by line) by a vector. 

__global__ void MULT_SM_V_128_512(int numdof, int numno, int RowSize, double *Matrix_A_d, double *Vector_X_d, int *Vector_I_d, double *Vector_R_d)
{

	int row, lane;
	
	__shared__ double vals[512];
	
	const int thread_block_id = blockDim.x*threadIdx.y + threadIdx.x;
	
	const int thread_id = (gridDim.x*blockDim.x)*( (blockIdx.y)*blockDim.y) + (blockIdx.x)*blockDim.x*blockDim.y + thread_block_id;
	
	// ******************************************************************************************************* 

	if(thread_id < numdof*numno*RowSize) {
	
		row  = thread_id/128;
		lane = thread_id & (128-1);
			
		vals[thread_block_id] = Matrix_A_d[thread_id] * Vector_X_d[Vector_I_d[thread_id]-1];																				     
		
		__syncthreads();
		if(lane < 64) vals[thread_block_id] += vals[thread_block_id + 64]; __syncthreads();													                          
		if(lane < 32) vals[thread_block_id] += vals[thread_block_id + 32]; __syncthreads();		
		if(lane < 16) vals[thread_block_id] += vals[thread_block_id + 16]; __syncthreads();																				  
		if(lane <  8) vals[thread_block_id] += vals[thread_block_id +  8]; __syncthreads();	
		if(lane <  4) vals[thread_block_id] += vals[thread_block_id +  4]; __syncthreads();	
		if(lane <  2) vals[thread_block_id] += vals[thread_block_id +  2]; __syncthreads();	
		if(lane <  1) vals[thread_block_id] += vals[thread_block_id +  1]; __syncthreads();	
		
		if(lane == 0)
			Vector_R_d[row] = vals[thread_block_id];
		
	}
		
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============

// This kernel add up the four terms of the vector delta_new_d. 
__global__ void SUM_GPU_4(int numdof, int numno, double *delta_GPU_d, double *delta_new_V_d)
{

	__shared__ double sdata[4];

	const int thread_block_id = blockDim.x*threadIdx.y + threadIdx.x;
	const int thread_id = (gridDim.x*blockDim.x)*( (blockIdx.y)*blockDim.y) + (blockIdx.x)*blockDim.x*blockDim.y + thread_block_id;
	
	// ******************************************************************************************************* 
	
	if(thread_block_id < 4) sdata[thread_block_id] = delta_GPU_d[thread_block_id]; __syncthreads();
	if(thread_block_id < 2)	sdata[thread_block_id] += sdata[thread_block_id + 2]; __syncthreads();
	if(thread_block_id < 1)	sdata[thread_block_id] += sdata[thread_block_id + 1]; __syncthreads();
	
	if(thread_id < numdof*numno) {
		delta_new_V_d[thread_id] = sdata[0];
	}
		
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============

// This kernel add up the four terms of the vector delta_new_d. 
__global__ void SUM_V_4(int numdof, int numno, int position, double *delta_new_d, double *delta_GPU_d, double *Result_d)
{

	__shared__ double sdata[4];

	const int thread_block_id = blockDim.x*threadIdx.y + threadIdx.x;
	const int thread_id = (gridDim.x*blockDim.x)*( (blockIdx.y)*blockDim.y) + (blockIdx.x)*blockDim.x*blockDim.y + thread_block_id;
	
	// ******************************************************************************************************* 
	
	if(thread_block_id < 4) sdata[thread_block_id] = delta_new_d[thread_block_id]; __syncthreads();
	
	if(thread_block_id < 2)	sdata[thread_block_id] += sdata[thread_block_id + 2]; __syncthreads();
	if(thread_block_id < 1)	sdata[thread_block_id] += sdata[thread_block_id + 1]; __syncthreads();
	
	if(thread_id == 0) {
		delta_GPU_d[position] = sdata[0];
		Result_d[0] = sdata[0];
	}
		
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============

// This kernel add up the eight terms of the vector delta_new_d. 
__global__ void SUM_V_8(int numdof, int numno, int position, double *delta_new_d, double *delta_GPU_d, double *Result_d)
{
	
	__shared__ double sdata[8];

	const int thread_block_id = blockDim.x*threadIdx.y + threadIdx.x;
	const int thread_id = (gridDim.x*blockDim.x)*( (blockIdx.y)*blockDim.y) + (blockIdx.x)*blockDim.x*blockDim.y + thread_block_id;
	
	// ******************************************************************************************************* 
	
	if(thread_block_id < 8) sdata[thread_block_id] = delta_new_d[thread_block_id]; __syncthreads();

	if(thread_block_id < 4)	sdata[thread_block_id] += sdata[thread_block_id + 4]; __syncthreads();
	if(thread_block_id < 2)	sdata[thread_block_id] += sdata[thread_block_id + 2]; __syncthreads();
	if(thread_block_id < 1)	sdata[thread_block_id] += sdata[thread_block_id + 1]; __syncthreads();
	
	if(thread_id == 0) {
		delta_GPU_d[position] = sdata[0];
		Result_d[0] = sdata[0];
	}
			
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============

// This kernel add up the sixteen terms of the vector delta_new_d. 
__global__ void SUM_V_16(int numdof, int numno, int position, double *delta_new_d, double *delta_GPU_d, double *Result_d)
{

	__shared__ double sdata[16];

	const int thread_block_id = blockDim.x*threadIdx.y + threadIdx.x;
	const int thread_id = (gridDim.x*blockDim.x)*( (blockIdx.y)*blockDim.y) + (blockIdx.x)*blockDim.x*blockDim.y + thread_block_id;
	
	// ******************************************************************************************************* 
	
	if(thread_block_id < 16) sdata[thread_block_id] = delta_new_d[thread_block_id]; __syncthreads();
	
	if(thread_block_id < 8)	sdata[thread_block_id] += sdata[thread_block_id + 8]; __syncthreads();
	if(thread_block_id < 4)	sdata[thread_block_id] += sdata[thread_block_id + 4]; __syncthreads();
	if(thread_block_id < 2)	sdata[thread_block_id] += sdata[thread_block_id + 2]; __syncthreads();
	if(thread_block_id < 1)	sdata[thread_block_id] += sdata[thread_block_id + 1]; __syncthreads();
	
	if(thread_id == 0) {
		delta_GPU_d[position] = sdata[0];
		Result_d[0] = sdata[0];
	}
		
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============

// This kernel add up the sixteen terms of the vector delta_new_d. 
__global__ void SUM_V_32(int numdof, int numno, int position, double *delta_new_d, double *delta_GPU_d, double *Result_d)
{

	__shared__ double sdata[32];

	const int thread_block_id = blockDim.x*threadIdx.y + threadIdx.x;
	const int thread_id = (gridDim.x*blockDim.x)*( (blockIdx.y)*blockDim.y) + (blockIdx.x)*blockDim.x*blockDim.y + thread_block_id;
	
	// ******************************************************************************************************* 
	
	if(thread_block_id < 32) sdata[thread_block_id] = delta_new_d[thread_block_id]; __syncthreads();
	
	if(thread_block_id <16)	sdata[thread_block_id] += sdata[thread_block_id +16]; __syncthreads();
	if(thread_block_id < 8)	sdata[thread_block_id] += sdata[thread_block_id + 8]; __syncthreads();
	if(thread_block_id < 4)	sdata[thread_block_id] += sdata[thread_block_id + 4]; __syncthreads();
	if(thread_block_id < 2)	sdata[thread_block_id] += sdata[thread_block_id + 2]; __syncthreads();
	if(thread_block_id < 1)	sdata[thread_block_id] += sdata[thread_block_id + 1]; __syncthreads();
	
	if(thread_id == 0) {
		delta_GPU_d[position] = sdata[0];
		Result_d[0] = sdata[0];
	}
		
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============

// This kernel subtracts a vector by other vector of n terms. 
  
__global__ void SUBT_V_V(int numdof, int numno, double *Vector_B_d, double *Vector_R_d)                             
{	
	const int xIndex = blockIdx.x*blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y*blockDim.y + threadIdx.y;
	const int thread_id = (gridDim.x*blockDim.x)*yIndex + xIndex;
	
	if(thread_id < numdof*numno)
		Vector_R_d[thread_id] = Vector_B_d[thread_id];
		
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============

// This kernel clear vector terms. 

__global__ void CLEAR_V(int numdof, int numno, double *Vector_X_d)
{	
	const int xIndex = blockIdx.x*blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y*blockDim.y + threadIdx.y;
	const int thread_id = (gridDim.x*blockDim.x)*yIndex + xIndex;
	
	if(thread_id < numdof*numno)
		Vector_X_d[thread_id] = 0.;
		
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============

// This kernel multiplies a diagonal matrix by a vector of n terms. 

  __global__ void MULT_DM_V(int numdof, int numno, double *Matrix_M_d, double *Vector_R_d, double *Vector_D_d)
{	
	const int xIndex = blockIdx.x*blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y*blockDim.y + threadIdx.y;
	const int thread_id = (gridDim.x*blockDim.x)*yIndex + xIndex;
	
	if(thread_id < numdof*numno)
		Vector_D_d[thread_id] = Matrix_M_d[thread_id] * Vector_R_d[thread_id];
		
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============

// This kernel multiplies a vector by a other vector of n terms. 
__global__ void MULT_V_V_1024(int numdof, int numno, double *Vector_R_d, double *Vector_D_d, double *delta_new_d)
{

	__shared__ double sdata[1024];
	
	int thread_id = blockIdx.x*blockDim.x + threadIdx.x;    // Thread id on grid
	int gridSize  = gridDim.x*blockDim.x;                   // Total number of threads 
	
	sdata[threadIdx.x] = 0;
	
	while(thread_id < numdof*numno) {
	
		// Multiplying the vertor r by the vector d
		
		sdata[threadIdx.x] += Vector_R_d[thread_id]*Vector_D_d[thread_id];
		
		thread_id += gridSize;
		
	}
				
	__syncthreads();
	
	if(threadIdx.x < 512) sdata[threadIdx.x] += sdata[threadIdx.x + 512]; __syncthreads();
	if(threadIdx.x < 256) sdata[threadIdx.x] += sdata[threadIdx.x + 256]; __syncthreads();
	if(threadIdx.x < 128) sdata[threadIdx.x] += sdata[threadIdx.x + 128]; __syncthreads();
	if(threadIdx.x <  64) sdata[threadIdx.x] += sdata[threadIdx.x +  64]; __syncthreads();
	if(threadIdx.x <  32) sdata[threadIdx.x] += sdata[threadIdx.x +  32]; __syncthreads();
	if(threadIdx.x <  16) sdata[threadIdx.x] += sdata[threadIdx.x +  16]; __syncthreads();
	if(threadIdx.x <   8) sdata[threadIdx.x] += sdata[threadIdx.x +   8]; __syncthreads();
	if(threadIdx.x <   4) sdata[threadIdx.x] += sdata[threadIdx.x +   4]; __syncthreads();
	if(threadIdx.x <   2) sdata[threadIdx.x] += sdata[threadIdx.x +   2]; __syncthreads();
	if(threadIdx.x <   1) sdata[threadIdx.x] += sdata[threadIdx.x +   1]; __syncthreads();
	
	if(threadIdx.x == 0)
		delta_new_d[blockIdx.x] = sdata[0];
				
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============

// This kernel multiplies a vector by a other vector of n terms. 
__global__ void MULT_V_V_512(int numdof, int numno, double *Vector_R_d, double *Vector_D_d, double *delta_new_d)
{

	__shared__ double sdata[512];
	
	int thread_id = blockIdx.x*blockDim.x + threadIdx.x;    // Thread id on grid
	int gridSize  = gridDim.x*blockDim.x;                   // Total number of threads 
	
	sdata[threadIdx.x] = 0;
	
	while(thread_id < numdof*numno) {
	
		// Multiplying the vertor r by the vector d
		
		sdata[threadIdx.x] += Vector_R_d[thread_id]*Vector_D_d[thread_id];
		
		thread_id += gridSize;
		
	}
				
	__syncthreads();
	
	if(threadIdx.x < 256) sdata[threadIdx.x] += sdata[threadIdx.x + 256]; __syncthreads();
	if(threadIdx.x < 128) sdata[threadIdx.x] += sdata[threadIdx.x + 128]; __syncthreads();
	if(threadIdx.x <  64) sdata[threadIdx.x] += sdata[threadIdx.x +  64]; __syncthreads();
	if(threadIdx.x <  32) sdata[threadIdx.x] += sdata[threadIdx.x +  32]; __syncthreads();
	if(threadIdx.x <  16) sdata[threadIdx.x] += sdata[threadIdx.x +  16]; __syncthreads();
	if(threadIdx.x <   8) sdata[threadIdx.x] += sdata[threadIdx.x +   8]; __syncthreads();
	if(threadIdx.x <   4) sdata[threadIdx.x] += sdata[threadIdx.x +   4]; __syncthreads();
	if(threadIdx.x <   2) sdata[threadIdx.x] += sdata[threadIdx.x +   2]; __syncthreads();
	if(threadIdx.x <   1) sdata[threadIdx.x] += sdata[threadIdx.x +   1]; __syncthreads();
	
	if(threadIdx.x == 0)
		delta_new_d[blockIdx.x] = sdata[0];
				
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============

__global__ void EvaluateMatrixM(int numdof, int numno, int *iiPosition_d, double *Matrix_K_d, double *Matrix_M_d) 
{
	const int xIndex = blockIdx.x*blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y*blockDim.y + threadIdx.y;
	const int thread_id = (gridDim.x*blockDim.x)*yIndex + xIndex;
	
	if(thread_id < numdof*numno) {
	
		if(Matrix_K_d[iiPosition_d[thread_id]] == 0.) Matrix_M_d[thread_id] = 0.;
		else                                          Matrix_M_d[thread_id] = 1 / Matrix_K_d[iiPosition_d[thread_id]];
		
	}
	
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============
// This kernel update vector Vector_D_Global_d
__global__ void Copy_Local_Global(int numdof, int numno, int position, double *Vector_D_d, double *Vector_D_Global_d)
{
	const int thread_block_id = blockDim.x*threadIdx.y + threadIdx.x;
	const int thread_id = (gridDim.x*blockDim.x)*( (blockIdx.y)*blockDim.y) + (blockIdx.x)*blockDim.x*blockDim.y + thread_block_id;
	
	if(thread_id < numdof*numno)
		Vector_D_Global_d[thread_id + position] = Vector_D_d[thread_id];
	
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============
// This kernel update vector Vector_D_Global_d
__global__ void Copy_for_Vector_4(double *delta_0_new_V_d, double *delta_0_GPU_d)
{
	const int thread_id = threadIdx.x;

	delta_0_GPU_d[thread_id] = delta_0_new_V_d[thread_id];
	
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============
// This kernel cleansthe Vector_D_Global_d
__global__ void Clean_Global_Vector(int numdof, int numnoglobal, double *Vector_D_Global_d)
{
	const int thread_block_id = blockDim.x*threadIdx.y + threadIdx.x;
	const int thread_id = (gridDim.x*blockDim.x)*( (blockIdx.y)*blockDim.y) + (blockIdx.x)*blockDim.x*blockDim.y + thread_block_id;

	if(thread_id < numdof*numnoglobal)
		Vector_D_Global_d[thread_id] = 0.;
	
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============

__global__ void AssemblyStiffness3DQ1(int numno, int RowSize, double *CoordQ1_d, double *MatPropQ1_d, double *Matrix_K_Aux_d) 
{
	double r, s, t;  // 3
	double XEL0, YEL0, ZEL0, XEL1, YEL1, ZEL1, XEL2, YEL2, ZEL2, XEL3, YEL3, ZEL3;  // 12 
	double XEL4, YEL4, ZEL4, XEL5, YEL5, ZEL5, XEL6, YEL6, ZEL6, XEL7, YEL7, ZEL7;  // 12 
	double xgaus[2], wgaus[2];  // 4
	double p, E, C00, C01, C33;  // 5
	double phi_r[8], phi_s[8], phi_t[8];  // 24
	double jac[3][3];  // 9
	double jacob;  // 1
	double gama[3][3];  // 9
	double auxX[8], auxY[8], auxZ[8];  // 24
	int ig, jg, kg;  // 3
	
	const int xIndex = blockIdx.x*blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y*blockDim.y + threadIdx.y;
	const int thread_id = (gridDim.x*blockDim.x)*yIndex + xIndex;
	
	const int NumThread = (gridDim.x*blockDim.x)*(gridDim.y*blockDim.y);
	const int NumThreadSpace = RowSize*NumThread;
	
	__syncthreads();
	
	if(thread_id < numno) {
	
		XEL0 = CoordQ1_d[thread_id+ 0*numno];
		YEL0 = CoordQ1_d[thread_id+ 1*numno];
		ZEL0 = CoordQ1_d[thread_id+ 2*numno];   	
		XEL1 = CoordQ1_d[thread_id+ 3*numno];
		YEL1 = CoordQ1_d[thread_id+ 4*numno];  
		ZEL1 = CoordQ1_d[thread_id+ 5*numno];  	
		XEL2 = CoordQ1_d[thread_id+ 6*numno];
		YEL2 = CoordQ1_d[thread_id+ 7*numno]; 
		ZEL2 = CoordQ1_d[thread_id+ 8*numno];  		
		XEL3 = CoordQ1_d[thread_id+ 9*numno];
		YEL3 = CoordQ1_d[thread_id+10*numno];
		ZEL3 = CoordQ1_d[thread_id+11*numno];
		
		XEL4 = CoordQ1_d[thread_id+12*numno];
		YEL4 = CoordQ1_d[thread_id+13*numno];
		ZEL4 = CoordQ1_d[thread_id+14*numno];   	
		XEL5 = CoordQ1_d[thread_id+15*numno];
		YEL5 = CoordQ1_d[thread_id+16*numno];  
		ZEL5 = CoordQ1_d[thread_id+17*numno];  	
		XEL6 = CoordQ1_d[thread_id+18*numno];
		YEL6 = CoordQ1_d[thread_id+19*numno]; 
		ZEL6 = CoordQ1_d[thread_id+20*numno];  		
		XEL7 = CoordQ1_d[thread_id+21*numno];
		YEL7 = CoordQ1_d[thread_id+22*numno];
		ZEL7 = CoordQ1_d[thread_id+23*numno];
		
		__syncthreads();

		xgaus[0]=-0.577350269189626;
		xgaus[1]= 0.577350269189626;
		wgaus[0]= 1.;
		wgaus[1]= 1.;
							
		E = MatPropQ1_d[thread_id+ 0*numno];
		p = MatPropQ1_d[thread_id+ 1*numno];
					
		C00=(E*(1-p))/((1+p)*(1-2*p));  
		C01=(E*(1-p))/((1+p)*(1-2*p))*(p/(1-p)); 
		C33=(E*(1-p))/((1+p)*(1-2*p))*((1-2*p)/(2*(1-p)));
		
		for(ig=0; ig<2; ig++) {                                          // =================== Loop ig ===================
			t=xgaus[ig];
					
			for(jg=0; jg<2; jg++) {                                      // =================== Loop jg ===================
				s=xgaus[jg]; 
				
				for(kg=0; kg<2; kg++) {                                  // =================== Loop kg ===================
				r=xgaus[kg]; 
				
					// Shape function derivative:
					phi_r[0] = -0.125*(1.0-s)*(1.0-t);
					phi_r[1] =  0.125*(1.0-s)*(1.0-t);
					phi_r[2] =  0.125*(1.0+s)*(1.0-t);
					phi_r[3] = -0.125*(1.0+s)*(1.0-t);
					phi_r[4] = -0.125*(1.0-s)*(1.0+t);
					phi_r[5] =  0.125*(1.0-s)*(1.0+t);
					phi_r[6] =  0.125*(1.0+s)*(1.0+t);
					phi_r[7] = -0.125*(1.0+s)*(1.0+t);
	 
					phi_s[0] = -0.125*(1.0-r)*(1.0-t);
					phi_s[1] = -0.125*(1.0+r)*(1.0-t);
					phi_s[2] =  0.125*(1.0+r)*(1.0-t);
					phi_s[3] =  0.125*(1.0-r)*(1.0-t);
					phi_s[4] = -0.125*(1.0-r)*(1.0+t);
					phi_s[5] = -0.125*(1.0+r)*(1.0+t);
					phi_s[6] =  0.125*(1.0+r)*(1.0+t);
					phi_s[7] =  0.125*(1.0-r)*(1.0+t);
		 
					phi_t[0] = -0.125*(1.0-r)*(1.0-s);
					phi_t[1] = -0.125*(1.0+r)*(1.0-s);
					phi_t[2] = -0.125*(1.0+r)*(1.0+s);
					phi_t[3] = -0.125*(1.0-r)*(1.0+s);
					phi_t[4] =  0.125*(1.0-r)*(1.0-s);
					phi_t[5] =  0.125*(1.0+r)*(1.0-s);
					phi_t[6] =  0.125*(1.0+r)*(1.0+s);
					phi_t[7] =  0.125*(1.0-r)*(1.0+s);
					
					// Terms of the Jacobian matrix:
					jac[0][0]=phi_r[0]*XEL0+phi_r[1]*XEL1+phi_r[2]*XEL2+phi_r[3]*XEL3+phi_r[4]*XEL4+phi_r[5]*XEL5+phi_r[6]*XEL6+phi_r[7]*XEL7;
					jac[0][1]=phi_r[0]*YEL0+phi_r[1]*YEL1+phi_r[2]*YEL2+phi_r[3]*YEL3+phi_r[4]*YEL4+phi_r[5]*YEL5+phi_r[6]*YEL6+phi_r[7]*YEL7;
					jac[0][2]=phi_r[0]*ZEL0+phi_r[1]*ZEL1+phi_r[2]*ZEL2+phi_r[3]*ZEL3+phi_r[4]*ZEL4+phi_r[5]*ZEL5+phi_r[6]*ZEL6+phi_r[7]*ZEL7;

					jac[1][0]=phi_s[0]*XEL0+phi_s[1]*XEL1+phi_s[2]*XEL2+phi_s[3]*XEL3+phi_s[4]*XEL4+phi_s[5]*XEL5+phi_s[6]*XEL6+phi_s[7]*XEL7;
					jac[1][1]=phi_s[0]*YEL0+phi_s[1]*YEL1+phi_s[2]*YEL2+phi_s[3]*YEL3+phi_s[4]*YEL4+phi_s[5]*YEL5+phi_s[6]*YEL6+phi_s[7]*YEL7;
					jac[1][2]=phi_s[0]*ZEL0+phi_s[1]*ZEL1+phi_s[2]*ZEL2+phi_s[3]*ZEL3+phi_s[4]*ZEL4+phi_s[5]*ZEL5+phi_s[6]*ZEL6+phi_s[7]*ZEL7;
					
					jac[2][0]=phi_t[0]*XEL0+phi_t[1]*XEL1+phi_t[2]*XEL2+phi_t[3]*XEL3+phi_t[4]*XEL4+phi_t[5]*XEL5+phi_t[6]*XEL6+phi_t[7]*XEL7;
					jac[2][1]=phi_t[0]*YEL0+phi_t[1]*YEL1+phi_t[2]*YEL2+phi_t[3]*YEL3+phi_t[4]*YEL4+phi_t[5]*YEL5+phi_t[6]*YEL6+phi_t[7]*YEL7;
					jac[2][2]=phi_t[0]*ZEL0+phi_t[1]*ZEL1+phi_t[2]*ZEL2+phi_t[3]*ZEL3+phi_t[4]*ZEL4+phi_t[5]*ZEL5+phi_t[6]*ZEL6+phi_t[7]*ZEL7;
						
					// Jacobian matrix determinant:	
					jacob = jac[0][0]*(jac[1][1]*jac[2][2]-jac[1][2]*jac[2][1])-jac[0][1]*(jac[1][0]*jac[2][2]-jac[1][2]*jac[2][0])+
							jac[0][2]*(jac[1][0]*jac[2][1]-jac[1][1]*jac[2][0]);
					
					if(jacob > 0.) {
					
						gama[0][0] = (jac[1][1]*jac[2][2] - jac[1][2]*jac[2][1]) /jacob;
						gama[0][1] =-(jac[0][1]*jac[2][2] - jac[0][2]*jac[2][1]) /jacob;
						gama[0][2] = (jac[0][1]*jac[1][2] - jac[0][2]*jac[1][1]) /jacob;
						
						gama[1][0] =-(jac[1][0]*jac[2][2] - jac[1][2]*jac[2][0]) /jacob;
						gama[1][1] = (jac[0][0]*jac[2][2] - jac[0][2]*jac[2][0]) /jacob;
						gama[1][2] =-(jac[0][0]*jac[1][2] - jac[0][2]*jac[1][0]) /jacob;
						
						gama[2][0] = (jac[1][0]*jac[2][1] - jac[1][1]*jac[2][0]) /jacob;
						gama[2][1] =-(jac[0][0]*jac[2][1] - jac[0][1]*jac[2][0]) /jacob;
						gama[2][2] = (jac[0][0]*jac[1][1] - jac[0][1]*jac[1][0]) /jacob;
						
						// *******************************************************************************************************
										
						auxX[0] = gama[0][0]*phi_r[0]+gama[0][1]*phi_s[0]+gama[0][2]*phi_t[0];
						auxY[0] = gama[1][0]*phi_r[0]+gama[1][1]*phi_s[0]+gama[1][2]*phi_t[0];
						auxZ[0] = gama[2][0]*phi_r[0]+gama[2][1]*phi_s[0]+gama[2][2]*phi_t[0];
							
						auxX[1] = gama[0][0]*phi_r[1]+gama[0][1]*phi_s[1]+gama[0][2]*phi_t[1];
						auxY[1] = gama[1][0]*phi_r[1]+gama[1][1]*phi_s[1]+gama[1][2]*phi_t[1];
						auxZ[1] = gama[2][0]*phi_r[1]+gama[2][1]*phi_s[1]+gama[2][2]*phi_t[1];
						
						auxX[2] = gama[0][0]*phi_r[2]+gama[0][1]*phi_s[2]+gama[0][2]*phi_t[2];
						auxY[2] = gama[1][0]*phi_r[2]+gama[1][1]*phi_s[2]+gama[1][2]*phi_t[2];
						auxZ[2] = gama[2][0]*phi_r[2]+gama[2][1]*phi_s[2]+gama[2][2]*phi_t[2];
						
						auxX[3] = gama[0][0]*phi_r[3]+gama[0][1]*phi_s[3]+gama[0][2]*phi_t[3];
						auxY[3] = gama[1][0]*phi_r[3]+gama[1][1]*phi_s[3]+gama[1][2]*phi_t[3];
						auxZ[3] = gama[2][0]*phi_r[3]+gama[2][1]*phi_s[3]+gama[2][2]*phi_t[3];
						
						auxX[4] = gama[0][0]*phi_r[4]+gama[0][1]*phi_s[4]+gama[0][2]*phi_t[4];
						auxY[4] = gama[1][0]*phi_r[4]+gama[1][1]*phi_s[4]+gama[1][2]*phi_t[4];
						auxZ[4] = gama[2][0]*phi_r[4]+gama[2][1]*phi_s[4]+gama[2][2]*phi_t[4];
							
						auxX[5] = gama[0][0]*phi_r[5]+gama[0][1]*phi_s[5]+gama[0][2]*phi_t[5];
						auxY[5] = gama[1][0]*phi_r[5]+gama[1][1]*phi_s[5]+gama[1][2]*phi_t[5];
						auxZ[5] = gama[2][0]*phi_r[5]+gama[2][1]*phi_s[5]+gama[2][2]*phi_t[5];
						
						auxX[6] = gama[0][0]*phi_r[6]+gama[0][1]*phi_s[6]+gama[0][2]*phi_t[6];
						auxY[6] = gama[1][0]*phi_r[6]+gama[1][1]*phi_s[6]+gama[1][2]*phi_t[6];
						auxZ[6] = gama[2][0]*phi_r[6]+gama[2][1]*phi_s[6]+gama[2][2]*phi_t[6];
						
						auxX[7] = gama[0][0]*phi_r[7]+gama[0][1]*phi_s[7]+gama[0][2]*phi_t[7];
						auxY[7] = gama[1][0]*phi_r[7]+gama[1][1]*phi_s[7]+gama[1][2]*phi_t[7];
						auxZ[7] = gama[2][0]*phi_r[7]+gama[2][1]*phi_s[7]+gama[2][2]*phi_t[7];
						
						__syncthreads();  // Line 0:
							
						Matrix_K_Aux_d[thread_id+ 0*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[6]*C00*auxX[0] + auxY[6]*C33*auxY[0] + auxZ[6]*C33*auxZ[0]);
						Matrix_K_Aux_d[thread_id+ 1*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[6]*C01*auxY[0] + auxY[6]*C33*auxX[0]);
						Matrix_K_Aux_d[thread_id+ 2*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[6]*C01*auxZ[0] + auxZ[6]*C33*auxX[0]);
							
						Matrix_K_Aux_d[thread_id+ 3*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[6]*C00*auxX[1] + auxY[6]*C33*auxY[1] + auxZ[6]*C33*auxZ[1]);
						Matrix_K_Aux_d[thread_id+ 4*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[6]*C01*auxY[1] + auxY[6]*C33*auxX[1]);
						Matrix_K_Aux_d[thread_id+ 5*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[6]*C01*auxZ[1] + auxZ[6]*C33*auxX[1]);
							
						Matrix_K_Aux_d[thread_id+ 9*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[6]*C00*auxX[3] + auxY[6]*C33*auxY[3] + auxZ[6]*C33*auxZ[3]);
						Matrix_K_Aux_d[thread_id+10*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[6]*C01*auxY[3] + auxY[6]*C33*auxX[3]);
						Matrix_K_Aux_d[thread_id+11*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[6]*C01*auxZ[3] + auxZ[6]*C33*auxX[3]);
							
						Matrix_K_Aux_d[thread_id+12*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[6]*C00*auxX[2] + auxY[6]*C33*auxY[2] + auxZ[6]*C33*auxZ[2]);
						Matrix_K_Aux_d[thread_id+13*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[6]*C01*auxY[2] + auxY[6]*C33*auxX[2]);
						Matrix_K_Aux_d[thread_id+14*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[6]*C01*auxZ[2] + auxZ[6]*C33*auxX[2]);
							
						Matrix_K_Aux_d[thread_id+27*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[6]*C00*auxX[4] + auxY[6]*C33*auxY[4] + auxZ[6]*C33*auxZ[4]);
						Matrix_K_Aux_d[thread_id+28*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[6]*C01*auxY[4] + auxY[6]*C33*auxX[4]);
						Matrix_K_Aux_d[thread_id+29*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[6]*C01*auxZ[4] + auxZ[6]*C33*auxX[4]);
							
						Matrix_K_Aux_d[thread_id+30*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[6]*C00*auxX[5] + auxY[6]*C33*auxY[5] + auxZ[6]*C33*auxZ[5]);
						Matrix_K_Aux_d[thread_id+31*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[6]*C01*auxY[5] + auxY[6]*C33*auxX[5]);
						Matrix_K_Aux_d[thread_id+32*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[6]*C01*auxZ[5] + auxZ[6]*C33*auxX[5]);
							
						Matrix_K_Aux_d[thread_id+36*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[6]*C00*auxX[7] + auxY[6]*C33*auxY[7] + auxZ[6]*C33*auxZ[7]);
						Matrix_K_Aux_d[thread_id+37*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[6]*C01*auxY[7] + auxY[6]*C33*auxX[7]);
						Matrix_K_Aux_d[thread_id+38*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[6]*C01*auxZ[7] + auxZ[6]*C33*auxX[7]);
							
						Matrix_K_Aux_d[thread_id+39*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[6]*C00*auxX[6] + auxY[6]*C33*auxY[6] + auxZ[6]*C33*auxZ[6]);
						Matrix_K_Aux_d[thread_id+40*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[6]*C01*auxY[6] + auxY[6]*C33*auxX[6]);
						Matrix_K_Aux_d[thread_id+41*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[6]*C01*auxZ[6] + auxZ[6]*C33*auxX[6]);
								
						
						__syncthreads();  // Line 1:
						
						Matrix_K_Aux_d[thread_id+ 0*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[6]*C01*auxX[0] + auxX[6]*C33*auxY[0]);
						Matrix_K_Aux_d[thread_id+ 1*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[6]*C00*auxY[0] + auxX[6]*C33*auxX[0] + auxZ[6]*C33*auxZ[0]);
						Matrix_K_Aux_d[thread_id+ 2*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[6]*C01*auxZ[0] + auxZ[6]*C33*auxY[0]);
							
						Matrix_K_Aux_d[thread_id+ 3*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[6]*C01*auxX[1] + auxX[6]*C33*auxY[1]);
						Matrix_K_Aux_d[thread_id+ 4*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[6]*C00*auxY[1] + auxX[6]*C33*auxX[1] + auxZ[6]*C33*auxZ[1]);
						Matrix_K_Aux_d[thread_id+ 5*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[6]*C01*auxZ[1] + auxZ[6]*C33*auxY[1]);
							
						Matrix_K_Aux_d[thread_id+ 9*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[6]*C01*auxX[3] + auxX[6]*C33*auxY[3]);
						Matrix_K_Aux_d[thread_id+10*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[6]*C00*auxY[3] + auxX[6]*C33*auxX[3] + auxZ[6]*C33*auxZ[3]);
						Matrix_K_Aux_d[thread_id+11*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[6]*C01*auxZ[3] + auxZ[6]*C33*auxY[3]);
							
						Matrix_K_Aux_d[thread_id+12*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[6]*C01*auxX[2] + auxX[6]*C33*auxY[2]);
						Matrix_K_Aux_d[thread_id+13*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[6]*C00*auxY[2] + auxX[6]*C33*auxX[2] + auxZ[6]*C33*auxZ[2]);
						Matrix_K_Aux_d[thread_id+14*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[6]*C01*auxZ[2] + auxZ[6]*C33*auxY[2]);
							
						Matrix_K_Aux_d[thread_id+27*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[6]*C01*auxX[4] + auxX[6]*C33*auxY[4]);
						Matrix_K_Aux_d[thread_id+28*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[6]*C00*auxY[4] + auxX[6]*C33*auxX[4] + auxZ[6]*C33*auxZ[4]);
						Matrix_K_Aux_d[thread_id+29*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[6]*C01*auxZ[4] + auxZ[6]*C33*auxY[4]);
							
						Matrix_K_Aux_d[thread_id+30*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[6]*C01*auxX[5] + auxX[6]*C33*auxY[5]);
						Matrix_K_Aux_d[thread_id+31*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[6]*C00*auxY[5] + auxX[6]*C33*auxX[5] + auxZ[6]*C33*auxZ[5]);
						Matrix_K_Aux_d[thread_id+32*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[6]*C01*auxZ[5] + auxZ[6]*C33*auxY[5]);
							
						Matrix_K_Aux_d[thread_id+36*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[6]*C01*auxX[7] + auxX[6]*C33*auxY[7]);
						Matrix_K_Aux_d[thread_id+37*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[6]*C00*auxY[7] + auxX[6]*C33*auxX[7] + auxZ[6]*C33*auxZ[7]);
						Matrix_K_Aux_d[thread_id+38*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[6]*C01*auxZ[7] + auxZ[6]*C33*auxY[7]);
							
						Matrix_K_Aux_d[thread_id+39*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[6]*C01*auxX[6] + auxX[6]*C33*auxY[6]);
						Matrix_K_Aux_d[thread_id+40*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[6]*C00*auxY[6] + auxX[6]*C33*auxX[6] + auxZ[6]*C33*auxZ[6]);
						Matrix_K_Aux_d[thread_id+41*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[6]*C01*auxZ[6] + auxZ[6]*C33*auxY[6]);
									
						__syncthreads();  // Line 2:
						
						Matrix_K_Aux_d[thread_id+ 0*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[6]*C01*auxX[0] + auxX[6]*C33*auxZ[0]);
						Matrix_K_Aux_d[thread_id+ 1*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[6]*C01*auxY[0] + auxY[6]*C33*auxZ[0]);
						Matrix_K_Aux_d[thread_id+ 2*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[6]*C00*auxZ[0] + auxY[6]*C33*auxY[0] + auxX[6]*C33*auxX[0]);
							
						Matrix_K_Aux_d[thread_id+ 3*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[6]*C01*auxX[1] + auxX[6]*C33*auxZ[1]);
						Matrix_K_Aux_d[thread_id+ 4*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[6]*C01*auxY[1] + auxY[6]*C33*auxZ[1]);
						Matrix_K_Aux_d[thread_id+ 5*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[6]*C00*auxZ[1] + auxY[6]*C33*auxY[1] + auxX[6]*C33*auxX[1]);
							
						Matrix_K_Aux_d[thread_id+ 9*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[6]*C01*auxX[3] + auxX[6]*C33*auxZ[3]);
						Matrix_K_Aux_d[thread_id+10*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[6]*C01*auxY[3] + auxY[6]*C33*auxZ[3]);
						Matrix_K_Aux_d[thread_id+11*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[6]*C00*auxZ[3] + auxY[6]*C33*auxY[3] + auxX[6]*C33*auxX[3]);
							
						Matrix_K_Aux_d[thread_id+12*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[6]*C01*auxX[2] + auxX[6]*C33*auxZ[2]);
						Matrix_K_Aux_d[thread_id+13*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[6]*C01*auxY[2] + auxY[6]*C33*auxZ[2]);
						Matrix_K_Aux_d[thread_id+14*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[6]*C00*auxZ[2] + auxY[6]*C33*auxY[2] + auxX[6]*C33*auxX[2]);
							
						Matrix_K_Aux_d[thread_id+27*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[6]*C01*auxX[4] + auxX[6]*C33*auxZ[4]);
						Matrix_K_Aux_d[thread_id+28*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[6]*C01*auxY[4] + auxY[6]*C33*auxZ[4]);
						Matrix_K_Aux_d[thread_id+29*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[6]*C00*auxZ[4] + auxY[6]*C33*auxY[4] + auxX[6]*C33*auxX[4]);
							
						Matrix_K_Aux_d[thread_id+30*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[6]*C01*auxX[5] + auxX[6]*C33*auxZ[5]);
						Matrix_K_Aux_d[thread_id+31*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[6]*C01*auxY[5] + auxY[6]*C33*auxZ[5]);
						Matrix_K_Aux_d[thread_id+32*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[6]*C00*auxZ[5] + auxY[6]*C33*auxY[5] + auxX[6]*C33*auxX[5]);
							
						Matrix_K_Aux_d[thread_id+36*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[6]*C01*auxX[7] + auxX[6]*C33*auxZ[7]);
						Matrix_K_Aux_d[thread_id+37*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[6]*C01*auxY[7] + auxY[6]*C33*auxZ[7]);
						Matrix_K_Aux_d[thread_id+38*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[6]*C00*auxZ[7] + auxY[6]*C33*auxY[7] + auxX[6]*C33*auxX[7]);
							
						Matrix_K_Aux_d[thread_id+39*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[6]*C01*auxX[6] + auxX[6]*C33*auxZ[6]);
						Matrix_K_Aux_d[thread_id+40*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[6]*C01*auxY[6] + auxY[6]*C33*auxZ[6]);
						Matrix_K_Aux_d[thread_id+41*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[6]*C00*auxZ[6] + auxY[6]*C33*auxY[6] + auxX[6]*C33*auxX[6]);
		
					}
					
				}                                                        // =================== Loop kg ===================
											
			}                                                            // =================== Loop jg ===================

		}                                                                // =================== Loop ig ===================
		
	}
	
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============

__global__ void AssemblyStiffness3DQ2(int numno, int RowSize, double *CoordQ2_d, double *MatPropQ2_d, double *Matrix_K_Aux_d) 
{
	double r, s, t;  // 3
	double XEL0, YEL0, ZEL0, XEL1, YEL1, ZEL1, XEL2, YEL2, ZEL2, XEL3, YEL3, ZEL3;  // 12 
	double XEL4, YEL4, ZEL4, XEL5, YEL5, ZEL5, XEL6, YEL6, ZEL6, XEL7, YEL7, ZEL7;  // 12 
	double xgaus[2], wgaus[2];  // 4
	double p, E, C00, C01, C33;  // 5
	double phi_r[8], phi_s[8], phi_t[8];  // 24
	double jac[3][3];  // 9
	double jacob;  // 1
	double gama[3][3];  // 9
	double auxX[8], auxY[8], auxZ[8];  // 24
	int ig, jg, kg;  // 3
	
	const int xIndex = blockIdx.x*blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y*blockDim.y + threadIdx.y;
	const int thread_id = (gridDim.x*blockDim.x)*yIndex + xIndex;
	
	const int NumThread = (gridDim.x*blockDim.x)*(gridDim.y*blockDim.y);
	const int NumThreadSpace = RowSize*NumThread;
	
	__syncthreads();
	
	if(thread_id < numno) {
	
		XEL0 = CoordQ2_d[thread_id+ 0*numno];
		YEL0 = CoordQ2_d[thread_id+ 1*numno];
		ZEL0 = CoordQ2_d[thread_id+ 2*numno];   	
		XEL1 = CoordQ2_d[thread_id+ 3*numno];
		YEL1 = CoordQ2_d[thread_id+ 4*numno];  
		ZEL1 = CoordQ2_d[thread_id+ 5*numno];  	
		XEL2 = CoordQ2_d[thread_id+ 6*numno];
		YEL2 = CoordQ2_d[thread_id+ 7*numno]; 
		ZEL2 = CoordQ2_d[thread_id+ 8*numno];  		
		XEL3 = CoordQ2_d[thread_id+ 9*numno];
		YEL3 = CoordQ2_d[thread_id+10*numno];
		ZEL3 = CoordQ2_d[thread_id+11*numno];
		
		XEL4 = CoordQ2_d[thread_id+12*numno];
		YEL4 = CoordQ2_d[thread_id+13*numno];
		ZEL4 = CoordQ2_d[thread_id+14*numno];   	
		XEL5 = CoordQ2_d[thread_id+15*numno];
		YEL5 = CoordQ2_d[thread_id+16*numno];  
		ZEL5 = CoordQ2_d[thread_id+17*numno];  	
		XEL6 = CoordQ2_d[thread_id+18*numno];
		YEL6 = CoordQ2_d[thread_id+19*numno]; 
		ZEL6 = CoordQ2_d[thread_id+20*numno];  		
		XEL7 = CoordQ2_d[thread_id+21*numno];
		YEL7 = CoordQ2_d[thread_id+22*numno];
		ZEL7 = CoordQ2_d[thread_id+23*numno];
		
		__syncthreads();

		xgaus[0]=-0.577350269189626;
		xgaus[1]= 0.577350269189626;
		wgaus[0]= 1.;
		wgaus[1]= 1.;
							
		E = MatPropQ2_d[thread_id+ 0*numno];
		p = MatPropQ2_d[thread_id+ 1*numno];
					
		C00=(E*(1-p))/((1+p)*(1-2*p));  
		C01=(E*(1-p))/((1+p)*(1-2*p))*(p/(1-p)); 
		C33=(E*(1-p))/((1+p)*(1-2*p))*((1-2*p)/(2*(1-p)));
			
		for(ig=0; ig<2; ig++) {                                          // =================== Loop ig ===================
			t=xgaus[ig];
					
			for(jg=0; jg<2; jg++) {                                      // =================== Loop jg ===================
				s=xgaus[jg]; 
				
				for(kg=0; kg<2; kg++) {                                  // =================== Loop kg ===================
				r=xgaus[kg]; 
				
					phi_r[0] = -0.125*(1.0-s)*(1.0-t);
					phi_r[1] =  0.125*(1.0-s)*(1.0-t);
					phi_r[2] =  0.125*(1.0+s)*(1.0-t);
					phi_r[3] = -0.125*(1.0+s)*(1.0-t);
					phi_r[4] = -0.125*(1.0-s)*(1.0+t);
					phi_r[5] =  0.125*(1.0-s)*(1.0+t);
					phi_r[6] =  0.125*(1.0+s)*(1.0+t);
					phi_r[7] = -0.125*(1.0+s)*(1.0+t);
	 
					phi_s[0] = -0.125*(1.0-r)*(1.0-t);
					phi_s[1] = -0.125*(1.0+r)*(1.0-t);
					phi_s[2] =  0.125*(1.0+r)*(1.0-t);
					phi_s[3] =  0.125*(1.0-r)*(1.0-t);
					phi_s[4] = -0.125*(1.0-r)*(1.0+t);
					phi_s[5] = -0.125*(1.0+r)*(1.0+t);
					phi_s[6] =  0.125*(1.0+r)*(1.0+t);
					phi_s[7] =  0.125*(1.0-r)*(1.0+t);
		 
					phi_t[0] = -0.125*(1.0-r)*(1.0-s);
					phi_t[1] = -0.125*(1.0+r)*(1.0-s);
					phi_t[2] = -0.125*(1.0+r)*(1.0+s);
					phi_t[3] = -0.125*(1.0-r)*(1.0+s);
					phi_t[4] =  0.125*(1.0-r)*(1.0-s);
					phi_t[5] =  0.125*(1.0+r)*(1.0-s);
					phi_t[6] =  0.125*(1.0+r)*(1.0+s);
					phi_t[7] =  0.125*(1.0-r)*(1.0+s);
						
					jac[0][0]=phi_r[0]*XEL0+phi_r[1]*XEL1+phi_r[2]*XEL2+phi_r[3]*XEL3+phi_r[4]*XEL4+phi_r[5]*XEL5+phi_r[6]*XEL6+phi_r[7]*XEL7;
					jac[0][1]=phi_r[0]*YEL0+phi_r[1]*YEL1+phi_r[2]*YEL2+phi_r[3]*YEL3+phi_r[4]*YEL4+phi_r[5]*YEL5+phi_r[6]*YEL6+phi_r[7]*YEL7;
					jac[0][2]=phi_r[0]*ZEL0+phi_r[1]*ZEL1+phi_r[2]*ZEL2+phi_r[3]*ZEL3+phi_r[4]*ZEL4+phi_r[5]*ZEL5+phi_r[6]*ZEL6+phi_r[7]*ZEL7;

					jac[1][0]=phi_s[0]*XEL0+phi_s[1]*XEL1+phi_s[2]*XEL2+phi_s[3]*XEL3+phi_s[4]*XEL4+phi_s[5]*XEL5+phi_s[6]*XEL6+phi_s[7]*XEL7;
					jac[1][1]=phi_s[0]*YEL0+phi_s[1]*YEL1+phi_s[2]*YEL2+phi_s[3]*YEL3+phi_s[4]*YEL4+phi_s[5]*YEL5+phi_s[6]*YEL6+phi_s[7]*YEL7;
					jac[1][2]=phi_s[0]*ZEL0+phi_s[1]*ZEL1+phi_s[2]*ZEL2+phi_s[3]*ZEL3+phi_s[4]*ZEL4+phi_s[5]*ZEL5+phi_s[6]*ZEL6+phi_s[7]*ZEL7;
					
					jac[2][0]=phi_t[0]*XEL0+phi_t[1]*XEL1+phi_t[2]*XEL2+phi_t[3]*XEL3+phi_t[4]*XEL4+phi_t[5]*XEL5+phi_t[6]*XEL6+phi_t[7]*XEL7;
					jac[2][1]=phi_t[0]*YEL0+phi_t[1]*YEL1+phi_t[2]*YEL2+phi_t[3]*YEL3+phi_t[4]*YEL4+phi_t[5]*YEL5+phi_t[6]*YEL6+phi_t[7]*YEL7;
					jac[2][2]=phi_t[0]*ZEL0+phi_t[1]*ZEL1+phi_t[2]*ZEL2+phi_t[3]*ZEL3+phi_t[4]*ZEL4+phi_t[5]*ZEL5+phi_t[6]*ZEL6+phi_t[7]*ZEL7;
							
					jacob = jac[0][0]*(jac[1][1]*jac[2][2]-jac[1][2]*jac[2][1])-jac[0][1]*(jac[1][0]*jac[2][2]-jac[1][2]*jac[2][0])+
							jac[0][2]*(jac[1][0]*jac[2][1]-jac[1][1]*jac[2][0]);
					
					if(jacob > 0.) {
					
						gama[0][0] = (jac[1][1]*jac[2][2] - jac[1][2]*jac[2][1]) /jacob;
						gama[0][1] =-(jac[0][1]*jac[2][2] - jac[0][2]*jac[2][1]) /jacob;
						gama[0][2] = (jac[0][1]*jac[1][2] - jac[0][2]*jac[1][1]) /jacob;
						
						gama[1][0] =-(jac[1][0]*jac[2][2] - jac[1][2]*jac[2][0]) /jacob;
						gama[1][1] = (jac[0][0]*jac[2][2] - jac[0][2]*jac[2][0]) /jacob;
						gama[1][2] =-(jac[0][0]*jac[1][2] - jac[0][2]*jac[1][0]) /jacob;
						
						gama[2][0] = (jac[1][0]*jac[2][1] - jac[1][1]*jac[2][0]) /jacob;
						gama[2][1] =-(jac[0][0]*jac[2][1] - jac[0][1]*jac[2][0]) /jacob;
						gama[2][2] = (jac[0][0]*jac[1][1] - jac[0][1]*jac[1][0]) /jacob;
										
						auxX[0] = gama[0][0]*phi_r[0]+gama[0][1]*phi_s[0]+gama[0][2]*phi_t[0];
						auxY[0] = gama[1][0]*phi_r[0]+gama[1][1]*phi_s[0]+gama[1][2]*phi_t[0];
						auxZ[0] = gama[2][0]*phi_r[0]+gama[2][1]*phi_s[0]+gama[2][2]*phi_t[0];
							
						auxX[1] = gama[0][0]*phi_r[1]+gama[0][1]*phi_s[1]+gama[0][2]*phi_t[1];
						auxY[1] = gama[1][0]*phi_r[1]+gama[1][1]*phi_s[1]+gama[1][2]*phi_t[1];
						auxZ[1] = gama[2][0]*phi_r[1]+gama[2][1]*phi_s[1]+gama[2][2]*phi_t[1];
						
						auxX[2] = gama[0][0]*phi_r[2]+gama[0][1]*phi_s[2]+gama[0][2]*phi_t[2];
						auxY[2] = gama[1][0]*phi_r[2]+gama[1][1]*phi_s[2]+gama[1][2]*phi_t[2];
						auxZ[2] = gama[2][0]*phi_r[2]+gama[2][1]*phi_s[2]+gama[2][2]*phi_t[2];
						
						auxX[3] = gama[0][0]*phi_r[3]+gama[0][1]*phi_s[3]+gama[0][2]*phi_t[3];
						auxY[3] = gama[1][0]*phi_r[3]+gama[1][1]*phi_s[3]+gama[1][2]*phi_t[3];
						auxZ[3] = gama[2][0]*phi_r[3]+gama[2][1]*phi_s[3]+gama[2][2]*phi_t[3];
						
						auxX[4] = gama[0][0]*phi_r[4]+gama[0][1]*phi_s[4]+gama[0][2]*phi_t[4];
						auxY[4] = gama[1][0]*phi_r[4]+gama[1][1]*phi_s[4]+gama[1][2]*phi_t[4];
						auxZ[4] = gama[2][0]*phi_r[4]+gama[2][1]*phi_s[4]+gama[2][2]*phi_t[4];
							
						auxX[5] = gama[0][0]*phi_r[5]+gama[0][1]*phi_s[5]+gama[0][2]*phi_t[5];
						auxY[5] = gama[1][0]*phi_r[5]+gama[1][1]*phi_s[5]+gama[1][2]*phi_t[5];
						auxZ[5] = gama[2][0]*phi_r[5]+gama[2][1]*phi_s[5]+gama[2][2]*phi_t[5];
						
						auxX[6] = gama[0][0]*phi_r[6]+gama[0][1]*phi_s[6]+gama[0][2]*phi_t[6];
						auxY[6] = gama[1][0]*phi_r[6]+gama[1][1]*phi_s[6]+gama[1][2]*phi_t[6];
						auxZ[6] = gama[2][0]*phi_r[6]+gama[2][1]*phi_s[6]+gama[2][2]*phi_t[6];
						
						auxX[7] = gama[0][0]*phi_r[7]+gama[0][1]*phi_s[7]+gama[0][2]*phi_t[7];
						auxY[7] = gama[1][0]*phi_r[7]+gama[1][1]*phi_s[7]+gama[1][2]*phi_t[7];
						auxZ[7] = gama[2][0]*phi_r[7]+gama[2][1]*phi_s[7]+gama[2][2]*phi_t[7];
				
						__syncthreads();  // Line 0:
							
						Matrix_K_Aux_d[thread_id+ 3*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[7]*C00*auxX[0] + auxY[7]*C33*auxY[0] + auxZ[7]*C33*auxZ[0]);
						Matrix_K_Aux_d[thread_id+ 4*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[7]*C01*auxY[0] + auxY[7]*C33*auxX[0]);
						Matrix_K_Aux_d[thread_id+ 5*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[7]*C01*auxZ[0] + auxZ[7]*C33*auxX[0]);
							
						Matrix_K_Aux_d[thread_id+ 6*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[7]*C00*auxX[1] + auxY[7]*C33*auxY[1] + auxZ[7]*C33*auxZ[1]);
						Matrix_K_Aux_d[thread_id+ 7*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[7]*C01*auxY[1] + auxY[7]*C33*auxX[1]);
						Matrix_K_Aux_d[thread_id+ 8*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[7]*C01*auxZ[1] + auxZ[7]*C33*auxX[1]);
							
						Matrix_K_Aux_d[thread_id+12*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[7]*C00*auxX[3] + auxY[7]*C33*auxY[3] + auxZ[7]*C33*auxZ[3]);
						Matrix_K_Aux_d[thread_id+13*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[7]*C01*auxY[3] + auxY[7]*C33*auxX[3]);
						Matrix_K_Aux_d[thread_id+14*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[7]*C01*auxZ[3] + auxZ[7]*C33*auxX[3]);
							
						Matrix_K_Aux_d[thread_id+15*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[7]*C00*auxX[2] + auxY[7]*C33*auxY[2] + auxZ[7]*C33*auxZ[2]);
						Matrix_K_Aux_d[thread_id+16*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[7]*C01*auxY[2] + auxY[7]*C33*auxX[2]);
						Matrix_K_Aux_d[thread_id+17*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[7]*C01*auxZ[2] + auxZ[7]*C33*auxX[2]);
							
						Matrix_K_Aux_d[thread_id+30*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[7]*C00*auxX[4] + auxY[7]*C33*auxY[4] + auxZ[7]*C33*auxZ[4]);
						Matrix_K_Aux_d[thread_id+31*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[7]*C01*auxY[4] + auxY[7]*C33*auxX[4]);
						Matrix_K_Aux_d[thread_id+32*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[7]*C01*auxZ[4] + auxZ[7]*C33*auxX[4]);
							
						Matrix_K_Aux_d[thread_id+33*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[7]*C00*auxX[5] + auxY[7]*C33*auxY[5] + auxZ[7]*C33*auxZ[5]);
						Matrix_K_Aux_d[thread_id+34*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[7]*C01*auxY[5] + auxY[7]*C33*auxX[5]);
						Matrix_K_Aux_d[thread_id+35*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[7]*C01*auxZ[5] + auxZ[7]*C33*auxX[5]);
							
						Matrix_K_Aux_d[thread_id+39*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[7]*C00*auxX[7] + auxY[7]*C33*auxY[7] + auxZ[7]*C33*auxZ[7]);
						Matrix_K_Aux_d[thread_id+40*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[7]*C01*auxY[7] + auxY[7]*C33*auxX[7]);
						Matrix_K_Aux_d[thread_id+41*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[7]*C01*auxZ[7] + auxZ[7]*C33*auxX[7]);
							
						Matrix_K_Aux_d[thread_id+42*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[7]*C00*auxX[6] + auxY[7]*C33*auxY[6] + auxZ[7]*C33*auxZ[6]);
						Matrix_K_Aux_d[thread_id+43*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[7]*C01*auxY[6] + auxY[7]*C33*auxX[6]);
						Matrix_K_Aux_d[thread_id+44*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[7]*C01*auxZ[6] + auxZ[7]*C33*auxX[6]);
							
						__syncthreads();  // Line 1:
						
						Matrix_K_Aux_d[thread_id+ 3*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[7]*C01*auxX[0] + auxX[7]*C33*auxY[0]);
						Matrix_K_Aux_d[thread_id+ 4*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[7]*C00*auxY[0] + auxX[7]*C33*auxX[0] + auxZ[7]*C33*auxZ[0]);
						Matrix_K_Aux_d[thread_id+ 5*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[7]*C01*auxZ[0] + auxZ[7]*C33*auxY[0]);
							
						Matrix_K_Aux_d[thread_id+ 6*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[7]*C01*auxX[1] + auxX[7]*C33*auxY[1]);
						Matrix_K_Aux_d[thread_id+ 7*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[7]*C00*auxY[1] + auxX[7]*C33*auxX[1] + auxZ[7]*C33*auxZ[1]);
						Matrix_K_Aux_d[thread_id+ 8*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[7]*C01*auxZ[1] + auxZ[7]*C33*auxY[1]);
							
						Matrix_K_Aux_d[thread_id+12*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[7]*C01*auxX[3] + auxX[7]*C33*auxY[3]);
						Matrix_K_Aux_d[thread_id+13*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[7]*C00*auxY[3] + auxX[7]*C33*auxX[3] + auxZ[7]*C33*auxZ[3]);
						Matrix_K_Aux_d[thread_id+14*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[7]*C01*auxZ[3] + auxZ[7]*C33*auxY[3]);
							
						Matrix_K_Aux_d[thread_id+15*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[7]*C01*auxX[2] + auxX[7]*C33*auxY[2]);
						Matrix_K_Aux_d[thread_id+16*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[7]*C00*auxY[2] + auxX[7]*C33*auxX[2] + auxZ[7]*C33*auxZ[2]);
						Matrix_K_Aux_d[thread_id+17*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[7]*C01*auxZ[2] + auxZ[7]*C33*auxY[2]);
							
						Matrix_K_Aux_d[thread_id+30*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[7]*C01*auxX[4] + auxX[7]*C33*auxY[4]);
						Matrix_K_Aux_d[thread_id+31*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[7]*C00*auxY[4] + auxX[7]*C33*auxX[4] + auxZ[7]*C33*auxZ[4]);
						Matrix_K_Aux_d[thread_id+32*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[7]*C01*auxZ[4] + auxZ[7]*C33*auxY[4]);
							
						Matrix_K_Aux_d[thread_id+33*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[7]*C01*auxX[5] + auxX[7]*C33*auxY[5]);
						Matrix_K_Aux_d[thread_id+34*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[7]*C00*auxY[5] + auxX[7]*C33*auxX[5] + auxZ[7]*C33*auxZ[5]);
						Matrix_K_Aux_d[thread_id+35*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[7]*C01*auxZ[5] + auxZ[7]*C33*auxY[5]);
							
						Matrix_K_Aux_d[thread_id+39*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[7]*C01*auxX[7] + auxX[7]*C33*auxY[7]);
						Matrix_K_Aux_d[thread_id+40*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[7]*C00*auxY[7] + auxX[7]*C33*auxX[7] + auxZ[7]*C33*auxZ[7]);
						Matrix_K_Aux_d[thread_id+41*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[7]*C01*auxZ[7] + auxZ[7]*C33*auxY[7]);
							
						Matrix_K_Aux_d[thread_id+42*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[7]*C01*auxX[6] + auxX[7]*C33*auxY[6]);
						Matrix_K_Aux_d[thread_id+43*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[7]*C00*auxY[6] + auxX[7]*C33*auxX[6] + auxZ[7]*C33*auxZ[6]);
						Matrix_K_Aux_d[thread_id+44*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[7]*C01*auxZ[6] + auxZ[7]*C33*auxY[6]);
						
						__syncthreads();  // Line 2:
						Matrix_K_Aux_d[thread_id+ 3*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[7]*C01*auxX[0] + auxX[7]*C33*auxZ[0]);
						Matrix_K_Aux_d[thread_id+ 4*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[7]*C01*auxY[0] + auxY[7]*C33*auxZ[0]);
						Matrix_K_Aux_d[thread_id+ 5*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[7]*C00*auxZ[0] + auxY[7]*C33*auxY[0] + auxX[7]*C33*auxX[0]);
							
						Matrix_K_Aux_d[thread_id+ 6*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[7]*C01*auxX[1] + auxX[7]*C33*auxZ[1]);
						Matrix_K_Aux_d[thread_id+ 7*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[7]*C01*auxY[1] + auxY[7]*C33*auxZ[1]);
						Matrix_K_Aux_d[thread_id+ 8*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[7]*C00*auxZ[1] + auxY[7]*C33*auxY[1] + auxX[7]*C33*auxX[1]);
							
						Matrix_K_Aux_d[thread_id+12*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[7]*C01*auxX[3] + auxX[7]*C33*auxZ[3]);
						Matrix_K_Aux_d[thread_id+13*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[7]*C01*auxY[3] + auxY[7]*C33*auxZ[3]);
						Matrix_K_Aux_d[thread_id+14*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[7]*C00*auxZ[3] + auxY[7]*C33*auxY[3] + auxX[7]*C33*auxX[3]);
							
						Matrix_K_Aux_d[thread_id+15*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[7]*C01*auxX[2] + auxX[7]*C33*auxZ[2]);
						Matrix_K_Aux_d[thread_id+16*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[7]*C01*auxY[2] + auxY[7]*C33*auxZ[2]);
						Matrix_K_Aux_d[thread_id+17*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[7]*C00*auxZ[2] + auxY[7]*C33*auxY[2] + auxX[7]*C33*auxX[2]);
							
						Matrix_K_Aux_d[thread_id+30*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[7]*C01*auxX[4] + auxX[7]*C33*auxZ[4]);
						Matrix_K_Aux_d[thread_id+31*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[7]*C01*auxY[4] + auxY[7]*C33*auxZ[4]);
						Matrix_K_Aux_d[thread_id+32*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[7]*C00*auxZ[4] + auxY[7]*C33*auxY[4] + auxX[7]*C33*auxX[4]);
							
						Matrix_K_Aux_d[thread_id+33*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[7]*C01*auxX[5] + auxX[7]*C33*auxZ[5]);
						Matrix_K_Aux_d[thread_id+34*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[7]*C01*auxY[5] + auxY[7]*C33*auxZ[5]);
						Matrix_K_Aux_d[thread_id+35*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[7]*C00*auxZ[5] + auxY[7]*C33*auxY[5] + auxX[7]*C33*auxX[5]);
							
						Matrix_K_Aux_d[thread_id+39*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[7]*C01*auxX[7] + auxX[7]*C33*auxZ[7]);
						Matrix_K_Aux_d[thread_id+40*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[7]*C01*auxY[7] + auxY[7]*C33*auxZ[7]);
						Matrix_K_Aux_d[thread_id+41*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[7]*C00*auxZ[7] + auxY[7]*C33*auxY[7] + auxX[7]*C33*auxX[7]);
							
						Matrix_K_Aux_d[thread_id+42*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[7]*C01*auxX[6] + auxX[7]*C33*auxZ[6]);
						Matrix_K_Aux_d[thread_id+43*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[7]*C01*auxY[6] + auxY[7]*C33*auxZ[6]);
						Matrix_K_Aux_d[thread_id+44*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[7]*C00*auxZ[6] + auxY[7]*C33*auxY[6] + auxX[7]*C33*auxX[6]);

					}
					
				}                                                        // =================== Loop kg ===================
											
			}                                                            // =================== Loop jg ===================

		}                                                                // =================== Loop ig ===================
		
	}

}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============

__global__ void AssemblyStiffness3DQ3(int numno, int RowSize, double *CoordQ3_d, double *MatPropQ3_d, double *Matrix_K_Aux_d) 
{
	double r, s, t;  // 3
	double XEL0, YEL0, ZEL0, XEL1, YEL1, ZEL1, XEL2, YEL2, ZEL2, XEL3, YEL3, ZEL3;  // 12 
	double XEL4, YEL4, ZEL4, XEL5, YEL5, ZEL5, XEL6, YEL6, ZEL6, XEL7, YEL7, ZEL7;  // 12 
	double xgaus[2], wgaus[2];  // 4
	double p, E, C00, C01, C33;  // 5
	double phi_r[8], phi_s[8], phi_t[8];  // 24
	double jac[3][3];  // 9
	double jacob;  // 1
	double gama[3][3];  // 9
	double auxX[8], auxY[8], auxZ[8];  // 24
	int ig, jg, kg;  // 3
	
	const int xIndex = blockIdx.x*blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y*blockDim.y + threadIdx.y;
	const int thread_id = (gridDim.x*blockDim.x)*yIndex + xIndex;
	
	const int NumThread = (gridDim.x*blockDim.x)*(gridDim.y*blockDim.y);
	const int NumThreadSpace = RowSize*NumThread;
	
	__syncthreads();
	
	if(thread_id < numno) {
	
		XEL0 = CoordQ3_d[thread_id+ 0*numno];
		YEL0 = CoordQ3_d[thread_id+ 1*numno];
		ZEL0 = CoordQ3_d[thread_id+ 2*numno];   	
		XEL1 = CoordQ3_d[thread_id+ 3*numno];
		YEL1 = CoordQ3_d[thread_id+ 4*numno];  
		ZEL1 = CoordQ3_d[thread_id+ 5*numno];  	
		XEL2 = CoordQ3_d[thread_id+ 6*numno];
		YEL2 = CoordQ3_d[thread_id+ 7*numno]; 
		ZEL2 = CoordQ3_d[thread_id+ 8*numno];  		
		XEL3 = CoordQ3_d[thread_id+ 9*numno];
		YEL3 = CoordQ3_d[thread_id+10*numno];
		ZEL3 = CoordQ3_d[thread_id+11*numno];
		
		XEL4 = CoordQ3_d[thread_id+12*numno];
		YEL4 = CoordQ3_d[thread_id+13*numno];
		ZEL4 = CoordQ3_d[thread_id+14*numno];   	
		XEL5 = CoordQ3_d[thread_id+15*numno];
		YEL5 = CoordQ3_d[thread_id+16*numno];  
		ZEL5 = CoordQ3_d[thread_id+17*numno];  	
		XEL6 = CoordQ3_d[thread_id+18*numno];
		YEL6 = CoordQ3_d[thread_id+19*numno]; 
		ZEL6 = CoordQ3_d[thread_id+20*numno];  		
		XEL7 = CoordQ3_d[thread_id+21*numno];
		YEL7 = CoordQ3_d[thread_id+22*numno];
		ZEL7 = CoordQ3_d[thread_id+23*numno];
		
		__syncthreads();

		xgaus[0]=-0.577350269189626;
		xgaus[1]= 0.577350269189626;
		wgaus[0]= 1.;
		wgaus[1]= 1.;
							
		E = MatPropQ3_d[thread_id+ 0*numno];
		p = MatPropQ3_d[thread_id+ 1*numno];
					
		C00=(E*(1-p))/((1+p)*(1-2*p));  
		C01=(E*(1-p))/((1+p)*(1-2*p))*(p/(1-p)); 
		C33=(E*(1-p))/((1+p)*(1-2*p))*((1-2*p)/(2*(1-p)));
			
		for(ig=0; ig<2; ig++) {                                          // =================== Loop ig ===================
			t=xgaus[ig];
					
			for(jg=0; jg<2; jg++) {                                      // =================== Loop jg ===================
				s=xgaus[jg]; 
				
				for(kg=0; kg<2; kg++) {                                  // =================== Loop kg ===================
				r=xgaus[kg]; 
				
					phi_r[0] = -0.125*(1.0-s)*(1.0-t);
					phi_r[1] =  0.125*(1.0-s)*(1.0-t);
					phi_r[2] =  0.125*(1.0+s)*(1.0-t);
					phi_r[3] = -0.125*(1.0+s)*(1.0-t);
					phi_r[4] = -0.125*(1.0-s)*(1.0+t);
					phi_r[5] =  0.125*(1.0-s)*(1.0+t);
					phi_r[6] =  0.125*(1.0+s)*(1.0+t);
					phi_r[7] = -0.125*(1.0+s)*(1.0+t);
	 
					phi_s[0] = -0.125*(1.0-r)*(1.0-t);
					phi_s[1] = -0.125*(1.0+r)*(1.0-t);
					phi_s[2] =  0.125*(1.0+r)*(1.0-t);
					phi_s[3] =  0.125*(1.0-r)*(1.0-t);
					phi_s[4] = -0.125*(1.0-r)*(1.0+t);
					phi_s[5] = -0.125*(1.0+r)*(1.0+t);
					phi_s[6] =  0.125*(1.0+r)*(1.0+t);
					phi_s[7] =  0.125*(1.0-r)*(1.0+t);
		 
					phi_t[0] = -0.125*(1.0-r)*(1.0-s);
					phi_t[1] = -0.125*(1.0+r)*(1.0-s);
					phi_t[2] = -0.125*(1.0+r)*(1.0+s);
					phi_t[3] = -0.125*(1.0-r)*(1.0+s);
					phi_t[4] =  0.125*(1.0-r)*(1.0-s);
					phi_t[5] =  0.125*(1.0+r)*(1.0-s);
					phi_t[6] =  0.125*(1.0+r)*(1.0+s);
					phi_t[7] =  0.125*(1.0-r)*(1.0+s);
						
					jac[0][0]=phi_r[0]*XEL0+phi_r[1]*XEL1+phi_r[2]*XEL2+phi_r[3]*XEL3+phi_r[4]*XEL4+phi_r[5]*XEL5+phi_r[6]*XEL6+phi_r[7]*XEL7;
					jac[0][1]=phi_r[0]*YEL0+phi_r[1]*YEL1+phi_r[2]*YEL2+phi_r[3]*YEL3+phi_r[4]*YEL4+phi_r[5]*YEL5+phi_r[6]*YEL6+phi_r[7]*YEL7;
					jac[0][2]=phi_r[0]*ZEL0+phi_r[1]*ZEL1+phi_r[2]*ZEL2+phi_r[3]*ZEL3+phi_r[4]*ZEL4+phi_r[5]*ZEL5+phi_r[6]*ZEL6+phi_r[7]*ZEL7;

					jac[1][0]=phi_s[0]*XEL0+phi_s[1]*XEL1+phi_s[2]*XEL2+phi_s[3]*XEL3+phi_s[4]*XEL4+phi_s[5]*XEL5+phi_s[6]*XEL6+phi_s[7]*XEL7;
					jac[1][1]=phi_s[0]*YEL0+phi_s[1]*YEL1+phi_s[2]*YEL2+phi_s[3]*YEL3+phi_s[4]*YEL4+phi_s[5]*YEL5+phi_s[6]*YEL6+phi_s[7]*YEL7;
					jac[1][2]=phi_s[0]*ZEL0+phi_s[1]*ZEL1+phi_s[2]*ZEL2+phi_s[3]*ZEL3+phi_s[4]*ZEL4+phi_s[5]*ZEL5+phi_s[6]*ZEL6+phi_s[7]*ZEL7;
					
					jac[2][0]=phi_t[0]*XEL0+phi_t[1]*XEL1+phi_t[2]*XEL2+phi_t[3]*XEL3+phi_t[4]*XEL4+phi_t[5]*XEL5+phi_t[6]*XEL6+phi_t[7]*XEL7;
					jac[2][1]=phi_t[0]*YEL0+phi_t[1]*YEL1+phi_t[2]*YEL2+phi_t[3]*YEL3+phi_t[4]*YEL4+phi_t[5]*YEL5+phi_t[6]*YEL6+phi_t[7]*YEL7;
					jac[2][2]=phi_t[0]*ZEL0+phi_t[1]*ZEL1+phi_t[2]*ZEL2+phi_t[3]*ZEL3+phi_t[4]*ZEL4+phi_t[5]*ZEL5+phi_t[6]*ZEL6+phi_t[7]*ZEL7;
							
					jacob = jac[0][0]*(jac[1][1]*jac[2][2]-jac[1][2]*jac[2][1])-jac[0][1]*(jac[1][0]*jac[2][2]-jac[1][2]*jac[2][0])+
							jac[0][2]*(jac[1][0]*jac[2][1]-jac[1][1]*jac[2][0]);
					
					if(jacob > 0.) {
					
						gama[0][0] = (jac[1][1]*jac[2][2] - jac[1][2]*jac[2][1]) /jacob;
						gama[0][1] =-(jac[0][1]*jac[2][2] - jac[0][2]*jac[2][1]) /jacob;
						gama[0][2] = (jac[0][1]*jac[1][2] - jac[0][2]*jac[1][1]) /jacob;
						
						gama[1][0] =-(jac[1][0]*jac[2][2] - jac[1][2]*jac[2][0]) /jacob;
						gama[1][1] = (jac[0][0]*jac[2][2] - jac[0][2]*jac[2][0]) /jacob;
						gama[1][2] =-(jac[0][0]*jac[1][2] - jac[0][2]*jac[1][0]) /jacob;
						
						gama[2][0] = (jac[1][0]*jac[2][1] - jac[1][1]*jac[2][0]) /jacob;
						gama[2][1] =-(jac[0][0]*jac[2][1] - jac[0][1]*jac[2][0]) /jacob;
						gama[2][2] = (jac[0][0]*jac[1][1] - jac[0][1]*jac[1][0]) /jacob;
										
						auxX[0] = gama[0][0]*phi_r[0]+gama[0][1]*phi_s[0]+gama[0][2]*phi_t[0];
						auxY[0] = gama[1][0]*phi_r[0]+gama[1][1]*phi_s[0]+gama[1][2]*phi_t[0];
						auxZ[0] = gama[2][0]*phi_r[0]+gama[2][1]*phi_s[0]+gama[2][2]*phi_t[0];
							
						auxX[1] = gama[0][0]*phi_r[1]+gama[0][1]*phi_s[1]+gama[0][2]*phi_t[1];
						auxY[1] = gama[1][0]*phi_r[1]+gama[1][1]*phi_s[1]+gama[1][2]*phi_t[1];
						auxZ[1] = gama[2][0]*phi_r[1]+gama[2][1]*phi_s[1]+gama[2][2]*phi_t[1];
						
						auxX[2] = gama[0][0]*phi_r[2]+gama[0][1]*phi_s[2]+gama[0][2]*phi_t[2];
						auxY[2] = gama[1][0]*phi_r[2]+gama[1][1]*phi_s[2]+gama[1][2]*phi_t[2];
						auxZ[2] = gama[2][0]*phi_r[2]+gama[2][1]*phi_s[2]+gama[2][2]*phi_t[2];
						
						auxX[3] = gama[0][0]*phi_r[3]+gama[0][1]*phi_s[3]+gama[0][2]*phi_t[3];
						auxY[3] = gama[1][0]*phi_r[3]+gama[1][1]*phi_s[3]+gama[1][2]*phi_t[3];
						auxZ[3] = gama[2][0]*phi_r[3]+gama[2][1]*phi_s[3]+gama[2][2]*phi_t[3];
						
						auxX[4] = gama[0][0]*phi_r[4]+gama[0][1]*phi_s[4]+gama[0][2]*phi_t[4];
						auxY[4] = gama[1][0]*phi_r[4]+gama[1][1]*phi_s[4]+gama[1][2]*phi_t[4];
						auxZ[4] = gama[2][0]*phi_r[4]+gama[2][1]*phi_s[4]+gama[2][2]*phi_t[4];
							
						auxX[5] = gama[0][0]*phi_r[5]+gama[0][1]*phi_s[5]+gama[0][2]*phi_t[5];
						auxY[5] = gama[1][0]*phi_r[5]+gama[1][1]*phi_s[5]+gama[1][2]*phi_t[5];
						auxZ[5] = gama[2][0]*phi_r[5]+gama[2][1]*phi_s[5]+gama[2][2]*phi_t[5];
						
						auxX[6] = gama[0][0]*phi_r[6]+gama[0][1]*phi_s[6]+gama[0][2]*phi_t[6];
						auxY[6] = gama[1][0]*phi_r[6]+gama[1][1]*phi_s[6]+gama[1][2]*phi_t[6];
						auxZ[6] = gama[2][0]*phi_r[6]+gama[2][1]*phi_s[6]+gama[2][2]*phi_t[6];
						
						auxX[7] = gama[0][0]*phi_r[7]+gama[0][1]*phi_s[7]+gama[0][2]*phi_t[7];
						auxY[7] = gama[1][0]*phi_r[7]+gama[1][1]*phi_s[7]+gama[1][2]*phi_t[7];
						auxZ[7] = gama[2][0]*phi_r[7]+gama[2][1]*phi_s[7]+gama[2][2]*phi_t[7];
						
						__syncthreads();  // Line 0:
							
						Matrix_K_Aux_d[thread_id+ 9*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[5]*C00*auxX[0] + auxY[5]*C33*auxY[0] + auxZ[5]*C33*auxZ[0]);
						Matrix_K_Aux_d[thread_id+10*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[5]*C01*auxY[0] + auxY[5]*C33*auxX[0]);
						Matrix_K_Aux_d[thread_id+11*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[5]*C01*auxZ[0] + auxZ[5]*C33*auxX[0]);
							
						Matrix_K_Aux_d[thread_id+12*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[5]*C00*auxX[1] + auxY[5]*C33*auxY[1] + auxZ[5]*C33*auxZ[1]);
						Matrix_K_Aux_d[thread_id+13*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[5]*C01*auxY[1] + auxY[5]*C33*auxX[1]);
						Matrix_K_Aux_d[thread_id+14*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[5]*C01*auxZ[1] + auxZ[5]*C33*auxX[1]);
							
						Matrix_K_Aux_d[thread_id+18*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[5]*C00*auxX[3] + auxY[5]*C33*auxY[3] + auxZ[5]*C33*auxZ[3]);
						Matrix_K_Aux_d[thread_id+19*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[5]*C01*auxY[3] + auxY[5]*C33*auxX[3]);
						Matrix_K_Aux_d[thread_id+20*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[5]*C01*auxZ[3] + auxZ[5]*C33*auxX[3]);
							
						Matrix_K_Aux_d[thread_id+21*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[5]*C00*auxX[2] + auxY[5]*C33*auxY[2] + auxZ[5]*C33*auxZ[2]);
						Matrix_K_Aux_d[thread_id+22*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[5]*C01*auxY[2] + auxY[5]*C33*auxX[2]);
						Matrix_K_Aux_d[thread_id+23*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[5]*C01*auxZ[2] + auxZ[5]*C33*auxX[2]);
							
						Matrix_K_Aux_d[thread_id+36*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[5]*C00*auxX[4] + auxY[5]*C33*auxY[4] + auxZ[5]*C33*auxZ[4]);
						Matrix_K_Aux_d[thread_id+37*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[5]*C01*auxY[4] + auxY[5]*C33*auxX[4]);
						Matrix_K_Aux_d[thread_id+38*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[5]*C01*auxZ[4] + auxZ[5]*C33*auxX[4]);
							
						Matrix_K_Aux_d[thread_id+39*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[5]*C00*auxX[5] + auxY[5]*C33*auxY[5] + auxZ[5]*C33*auxZ[5]);
						Matrix_K_Aux_d[thread_id+40*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[5]*C01*auxY[5] + auxY[5]*C33*auxX[5]);
						Matrix_K_Aux_d[thread_id+41*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[5]*C01*auxZ[5] + auxZ[5]*C33*auxX[5]);
							
						Matrix_K_Aux_d[thread_id+45*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[5]*C00*auxX[7] + auxY[5]*C33*auxY[7] + auxZ[5]*C33*auxZ[7]);
						Matrix_K_Aux_d[thread_id+46*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[5]*C01*auxY[7] + auxY[5]*C33*auxX[7]);
						Matrix_K_Aux_d[thread_id+47*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[5]*C01*auxZ[7] + auxZ[5]*C33*auxX[7]);
							
						Matrix_K_Aux_d[thread_id+48*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[5]*C00*auxX[6] + auxY[5]*C33*auxY[6] + auxZ[5]*C33*auxZ[6]);
						Matrix_K_Aux_d[thread_id+49*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[5]*C01*auxY[6] + auxY[5]*C33*auxX[6]);
						Matrix_K_Aux_d[thread_id+50*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[5]*C01*auxZ[6] + auxZ[5]*C33*auxX[6]);
							
						__syncthreads();  // Line 1:
						
						Matrix_K_Aux_d[thread_id+ 9*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[5]*C01*auxX[0] + auxX[5]*C33*auxY[0]);
						Matrix_K_Aux_d[thread_id+10*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[5]*C00*auxY[0] + auxX[5]*C33*auxX[0] + auxZ[5]*C33*auxZ[0]);
						Matrix_K_Aux_d[thread_id+11*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[5]*C01*auxZ[0] + auxZ[5]*C33*auxY[0]);
							
						Matrix_K_Aux_d[thread_id+12*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[5]*C01*auxX[1] + auxX[5]*C33*auxY[1]);
						Matrix_K_Aux_d[thread_id+13*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[5]*C00*auxY[1] + auxX[5]*C33*auxX[1] + auxZ[5]*C33*auxZ[1]);
						Matrix_K_Aux_d[thread_id+14*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[5]*C01*auxZ[1] + auxZ[5]*C33*auxY[1]);
							
						Matrix_K_Aux_d[thread_id+18*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[5]*C01*auxX[3] + auxX[5]*C33*auxY[3]);
						Matrix_K_Aux_d[thread_id+19*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[5]*C00*auxY[3] + auxX[5]*C33*auxX[3] + auxZ[5]*C33*auxZ[3]);
						Matrix_K_Aux_d[thread_id+20*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[5]*C01*auxZ[3] + auxZ[5]*C33*auxY[3]);
							
						Matrix_K_Aux_d[thread_id+21*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[5]*C01*auxX[2] + auxX[5]*C33*auxY[2]);
						Matrix_K_Aux_d[thread_id+22*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[5]*C00*auxY[2] + auxX[5]*C33*auxX[2] + auxZ[5]*C33*auxZ[2]);
						Matrix_K_Aux_d[thread_id+23*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[5]*C01*auxZ[2] + auxZ[5]*C33*auxY[2]);
							
						Matrix_K_Aux_d[thread_id+36*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[5]*C01*auxX[4] + auxX[5]*C33*auxY[4]);
						Matrix_K_Aux_d[thread_id+37*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[5]*C00*auxY[4] + auxX[5]*C33*auxX[4] + auxZ[5]*C33*auxZ[4]);
						Matrix_K_Aux_d[thread_id+38*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[5]*C01*auxZ[4] + auxZ[5]*C33*auxY[4]);
							
						Matrix_K_Aux_d[thread_id+39*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[5]*C01*auxX[5] + auxX[5]*C33*auxY[5]);
						Matrix_K_Aux_d[thread_id+40*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[5]*C00*auxY[5] + auxX[5]*C33*auxX[5] + auxZ[5]*C33*auxZ[5]);
						Matrix_K_Aux_d[thread_id+41*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[5]*C01*auxZ[5] + auxZ[5]*C33*auxY[5]);
							
						Matrix_K_Aux_d[thread_id+45*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[5]*C01*auxX[7] + auxX[5]*C33*auxY[7]);
						Matrix_K_Aux_d[thread_id+46*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[5]*C00*auxY[7] + auxX[5]*C33*auxX[7] + auxZ[5]*C33*auxZ[7]);
						Matrix_K_Aux_d[thread_id+47*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[5]*C01*auxZ[7] + auxZ[5]*C33*auxY[7]);
							
						Matrix_K_Aux_d[thread_id+48*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[5]*C01*auxX[6] + auxX[5]*C33*auxY[6]);
						Matrix_K_Aux_d[thread_id+49*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[5]*C00*auxY[6] + auxX[5]*C33*auxX[6] + auxZ[5]*C33*auxZ[6]);
						Matrix_K_Aux_d[thread_id+50*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[5]*C01*auxZ[6] + auxZ[5]*C33*auxY[6]);
						
						__syncthreads();  // Line 2:
						Matrix_K_Aux_d[thread_id+ 9*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[5]*C01*auxX[0] + auxX[5]*C33*auxZ[0]);
						Matrix_K_Aux_d[thread_id+10*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[5]*C01*auxY[0] + auxY[5]*C33*auxZ[0]);
						Matrix_K_Aux_d[thread_id+11*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[5]*C00*auxZ[0] + auxY[5]*C33*auxY[0] + auxX[5]*C33*auxX[0]);
							
						Matrix_K_Aux_d[thread_id+12*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[5]*C01*auxX[1] + auxX[5]*C33*auxZ[1]);
						Matrix_K_Aux_d[thread_id+13*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[5]*C01*auxY[1] + auxY[5]*C33*auxZ[1]);
						Matrix_K_Aux_d[thread_id+14*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[5]*C00*auxZ[1] + auxY[5]*C33*auxY[1] + auxX[5]*C33*auxX[1]);
							
						Matrix_K_Aux_d[thread_id+18*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[5]*C01*auxX[3] + auxX[5]*C33*auxZ[3]);
						Matrix_K_Aux_d[thread_id+19*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[5]*C01*auxY[3] + auxY[5]*C33*auxZ[3]);
						Matrix_K_Aux_d[thread_id+20*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[5]*C00*auxZ[3] + auxY[5]*C33*auxY[3] + auxX[5]*C33*auxX[3]);
							
						Matrix_K_Aux_d[thread_id+21*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[5]*C01*auxX[2] + auxX[5]*C33*auxZ[2]);
						Matrix_K_Aux_d[thread_id+22*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[5]*C01*auxY[2] + auxY[5]*C33*auxZ[2]);
						Matrix_K_Aux_d[thread_id+23*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[5]*C00*auxZ[2] + auxY[5]*C33*auxY[2] + auxX[5]*C33*auxX[2]);
							
						Matrix_K_Aux_d[thread_id+36*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[5]*C01*auxX[4] + auxX[5]*C33*auxZ[4]);
						Matrix_K_Aux_d[thread_id+37*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[5]*C01*auxY[4] + auxY[5]*C33*auxZ[4]);
						Matrix_K_Aux_d[thread_id+38*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[5]*C00*auxZ[4] + auxY[5]*C33*auxY[4] + auxX[5]*C33*auxX[4]);
							
						Matrix_K_Aux_d[thread_id+39*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[5]*C01*auxX[5] + auxX[5]*C33*auxZ[5]);
						Matrix_K_Aux_d[thread_id+40*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[5]*C01*auxY[5] + auxY[5]*C33*auxZ[5]);
						Matrix_K_Aux_d[thread_id+41*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[5]*C00*auxZ[5] + auxY[5]*C33*auxY[5] + auxX[5]*C33*auxX[5]);
							
						Matrix_K_Aux_d[thread_id+45*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[5]*C01*auxX[7] + auxX[5]*C33*auxZ[7]);
						Matrix_K_Aux_d[thread_id+46*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[5]*C01*auxY[7] + auxY[5]*C33*auxZ[7]);
						Matrix_K_Aux_d[thread_id+47*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[5]*C00*auxZ[7] + auxY[5]*C33*auxY[7] + auxX[5]*C33*auxX[7]);
							
						Matrix_K_Aux_d[thread_id+48*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[5]*C01*auxX[6] + auxX[5]*C33*auxZ[6]);
						Matrix_K_Aux_d[thread_id+49*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[5]*C01*auxY[6] + auxY[5]*C33*auxZ[6]);
						Matrix_K_Aux_d[thread_id+50*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[5]*C00*auxZ[6] + auxY[5]*C33*auxY[6] + auxX[5]*C33*auxX[6]);

					}
					
				}                                                        // =================== Loop kg ===================
											
			}                                                            // =================== Loop jg ===================

		}                                                                // =================== Loop ig ===================
		
	}
	
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============

__global__ void AssemblyStiffness3DQ4(int numno, int RowSize, double *CoordQ4_d, double *MatPropQ4_d, double *Matrix_K_Aux_d) 
{
	double r, s, t;  // 3
	double XEL0, YEL0, ZEL0, XEL1, YEL1, ZEL1, XEL2, YEL2, ZEL2, XEL3, YEL3, ZEL3;  // 12 
	double XEL4, YEL4, ZEL4, XEL5, YEL5, ZEL5, XEL6, YEL6, ZEL6, XEL7, YEL7, ZEL7;  // 12 
	double xgaus[2], wgaus[2];  // 4
	double p, E, C00, C01, C33;  // 5
	double phi_r[8], phi_s[8], phi_t[8];  // 24
	double jac[3][3];  // 9
	double jacob;  // 1
	double gama[3][3];  // 9
	double auxX[8], auxY[8], auxZ[8];  // 24
	int ig, jg, kg;  // 3
	
	const int xIndex = blockIdx.x*blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y*blockDim.y + threadIdx.y;
	const int thread_id = (gridDim.x*blockDim.x)*yIndex + xIndex;
	
	const int NumThread = (gridDim.x*blockDim.x)*(gridDim.y*blockDim.y);
	const int NumThreadSpace = RowSize*NumThread;
	
	__syncthreads();
	
	if(thread_id < numno) {
	
		XEL0 = CoordQ4_d[thread_id+ 0*numno];
		YEL0 = CoordQ4_d[thread_id+ 1*numno];
		ZEL0 = CoordQ4_d[thread_id+ 2*numno];   	
		XEL1 = CoordQ4_d[thread_id+ 3*numno];
		YEL1 = CoordQ4_d[thread_id+ 4*numno];  
		ZEL1 = CoordQ4_d[thread_id+ 5*numno];  	
		XEL2 = CoordQ4_d[thread_id+ 6*numno];
		YEL2 = CoordQ4_d[thread_id+ 7*numno]; 
		ZEL2 = CoordQ4_d[thread_id+ 8*numno];  		
		XEL3 = CoordQ4_d[thread_id+ 9*numno];
		YEL3 = CoordQ4_d[thread_id+10*numno];
		ZEL3 = CoordQ4_d[thread_id+11*numno];
		
		XEL4 = CoordQ4_d[thread_id+12*numno];
		YEL4 = CoordQ4_d[thread_id+13*numno];
		ZEL4 = CoordQ4_d[thread_id+14*numno];   	
		XEL5 = CoordQ4_d[thread_id+15*numno];
		YEL5 = CoordQ4_d[thread_id+16*numno];  
		ZEL5 = CoordQ4_d[thread_id+17*numno];  	
		XEL6 = CoordQ4_d[thread_id+18*numno];
		YEL6 = CoordQ4_d[thread_id+19*numno]; 
		ZEL6 = CoordQ4_d[thread_id+20*numno];  		
		XEL7 = CoordQ4_d[thread_id+21*numno];
		YEL7 = CoordQ4_d[thread_id+22*numno];
		ZEL7 = CoordQ4_d[thread_id+23*numno];
		
		__syncthreads();

		xgaus[0]=-0.577350269189626;
		xgaus[1]= 0.577350269189626;
		wgaus[0]= 1.;
		wgaus[1]= 1.;
							
		E = MatPropQ4_d[thread_id+ 0*numno];
		p = MatPropQ4_d[thread_id+ 1*numno];
					
		C00=(E*(1-p))/((1+p)*(1-2*p));  
		C01=(E*(1-p))/((1+p)*(1-2*p))*(p/(1-p)); 
		C33=(E*(1-p))/((1+p)*(1-2*p))*((1-2*p)/(2*(1-p)));
			
		for(ig=0; ig<2; ig++) {                                          // =================== Loop ig ===================
			t=xgaus[ig];
					
			for(jg=0; jg<2; jg++) {                                      // =================== Loop jg ===================
				s=xgaus[jg]; 
				
				for(kg=0; kg<2; kg++) {                                  // =================== Loop kg ===================
				r=xgaus[kg]; 
				
					phi_r[0] = -0.125*(1.0-s)*(1.0-t);
					phi_r[1] =  0.125*(1.0-s)*(1.0-t);
					phi_r[2] =  0.125*(1.0+s)*(1.0-t);
					phi_r[3] = -0.125*(1.0+s)*(1.0-t);
					phi_r[4] = -0.125*(1.0-s)*(1.0+t);
					phi_r[5] =  0.125*(1.0-s)*(1.0+t);
					phi_r[6] =  0.125*(1.0+s)*(1.0+t);
					phi_r[7] = -0.125*(1.0+s)*(1.0+t);
	 
					phi_s[0] = -0.125*(1.0-r)*(1.0-t);
					phi_s[1] = -0.125*(1.0+r)*(1.0-t);
					phi_s[2] =  0.125*(1.0+r)*(1.0-t);
					phi_s[3] =  0.125*(1.0-r)*(1.0-t);
					phi_s[4] = -0.125*(1.0-r)*(1.0+t);
					phi_s[5] = -0.125*(1.0+r)*(1.0+t);
					phi_s[6] =  0.125*(1.0+r)*(1.0+t);
					phi_s[7] =  0.125*(1.0-r)*(1.0+t);
		 
					phi_t[0] = -0.125*(1.0-r)*(1.0-s);
					phi_t[1] = -0.125*(1.0+r)*(1.0-s);
					phi_t[2] = -0.125*(1.0+r)*(1.0+s);
					phi_t[3] = -0.125*(1.0-r)*(1.0+s);
					phi_t[4] =  0.125*(1.0-r)*(1.0-s);
					phi_t[5] =  0.125*(1.0+r)*(1.0-s);
					phi_t[6] =  0.125*(1.0+r)*(1.0+s);
					phi_t[7] =  0.125*(1.0-r)*(1.0+s);
						
					jac[0][0]=phi_r[0]*XEL0+phi_r[1]*XEL1+phi_r[2]*XEL2+phi_r[3]*XEL3+phi_r[4]*XEL4+phi_r[5]*XEL5+phi_r[6]*XEL6+phi_r[7]*XEL7;
					jac[0][1]=phi_r[0]*YEL0+phi_r[1]*YEL1+phi_r[2]*YEL2+phi_r[3]*YEL3+phi_r[4]*YEL4+phi_r[5]*YEL5+phi_r[6]*YEL6+phi_r[7]*YEL7;
					jac[0][2]=phi_r[0]*ZEL0+phi_r[1]*ZEL1+phi_r[2]*ZEL2+phi_r[3]*ZEL3+phi_r[4]*ZEL4+phi_r[5]*ZEL5+phi_r[6]*ZEL6+phi_r[7]*ZEL7;

					jac[1][0]=phi_s[0]*XEL0+phi_s[1]*XEL1+phi_s[2]*XEL2+phi_s[3]*XEL3+phi_s[4]*XEL4+phi_s[5]*XEL5+phi_s[6]*XEL6+phi_s[7]*XEL7;
					jac[1][1]=phi_s[0]*YEL0+phi_s[1]*YEL1+phi_s[2]*YEL2+phi_s[3]*YEL3+phi_s[4]*YEL4+phi_s[5]*YEL5+phi_s[6]*YEL6+phi_s[7]*YEL7;
					jac[1][2]=phi_s[0]*ZEL0+phi_s[1]*ZEL1+phi_s[2]*ZEL2+phi_s[3]*ZEL3+phi_s[4]*ZEL4+phi_s[5]*ZEL5+phi_s[6]*ZEL6+phi_s[7]*ZEL7;
					
					jac[2][0]=phi_t[0]*XEL0+phi_t[1]*XEL1+phi_t[2]*XEL2+phi_t[3]*XEL3+phi_t[4]*XEL4+phi_t[5]*XEL5+phi_t[6]*XEL6+phi_t[7]*XEL7;
					jac[2][1]=phi_t[0]*YEL0+phi_t[1]*YEL1+phi_t[2]*YEL2+phi_t[3]*YEL3+phi_t[4]*YEL4+phi_t[5]*YEL5+phi_t[6]*YEL6+phi_t[7]*YEL7;
					jac[2][2]=phi_t[0]*ZEL0+phi_t[1]*ZEL1+phi_t[2]*ZEL2+phi_t[3]*ZEL3+phi_t[4]*ZEL4+phi_t[5]*ZEL5+phi_t[6]*ZEL6+phi_t[7]*ZEL7;
							
					jacob = jac[0][0]*(jac[1][1]*jac[2][2]-jac[1][2]*jac[2][1])-jac[0][1]*(jac[1][0]*jac[2][2]-jac[1][2]*jac[2][0])+
							jac[0][2]*(jac[1][0]*jac[2][1]-jac[1][1]*jac[2][0]);
					
					if(jacob > 0.) {
					
						gama[0][0] = (jac[1][1]*jac[2][2] - jac[1][2]*jac[2][1]) /jacob;
						gama[0][1] =-(jac[0][1]*jac[2][2] - jac[0][2]*jac[2][1]) /jacob;
						gama[0][2] = (jac[0][1]*jac[1][2] - jac[0][2]*jac[1][1]) /jacob;
						
						gama[1][0] =-(jac[1][0]*jac[2][2] - jac[1][2]*jac[2][0]) /jacob;
						gama[1][1] = (jac[0][0]*jac[2][2] - jac[0][2]*jac[2][0]) /jacob;
						gama[1][2] =-(jac[0][0]*jac[1][2] - jac[0][2]*jac[1][0]) /jacob;
						
						gama[2][0] = (jac[1][0]*jac[2][1] - jac[1][1]*jac[2][0]) /jacob;
						gama[2][1] =-(jac[0][0]*jac[2][1] - jac[0][1]*jac[2][0]) /jacob;
						gama[2][2] = (jac[0][0]*jac[1][1] - jac[0][1]*jac[1][0]) /jacob;
										
						auxX[0] = gama[0][0]*phi_r[0]+gama[0][1]*phi_s[0]+gama[0][2]*phi_t[0];
						auxY[0] = gama[1][0]*phi_r[0]+gama[1][1]*phi_s[0]+gama[1][2]*phi_t[0];
						auxZ[0] = gama[2][0]*phi_r[0]+gama[2][1]*phi_s[0]+gama[2][2]*phi_t[0];
							
						auxX[1] = gama[0][0]*phi_r[1]+gama[0][1]*phi_s[1]+gama[0][2]*phi_t[1];
						auxY[1] = gama[1][0]*phi_r[1]+gama[1][1]*phi_s[1]+gama[1][2]*phi_t[1];
						auxZ[1] = gama[2][0]*phi_r[1]+gama[2][1]*phi_s[1]+gama[2][2]*phi_t[1];
						
						auxX[2] = gama[0][0]*phi_r[2]+gama[0][1]*phi_s[2]+gama[0][2]*phi_t[2];
						auxY[2] = gama[1][0]*phi_r[2]+gama[1][1]*phi_s[2]+gama[1][2]*phi_t[2];
						auxZ[2] = gama[2][0]*phi_r[2]+gama[2][1]*phi_s[2]+gama[2][2]*phi_t[2];
						
						auxX[3] = gama[0][0]*phi_r[3]+gama[0][1]*phi_s[3]+gama[0][2]*phi_t[3];
						auxY[3] = gama[1][0]*phi_r[3]+gama[1][1]*phi_s[3]+gama[1][2]*phi_t[3];
						auxZ[3] = gama[2][0]*phi_r[3]+gama[2][1]*phi_s[3]+gama[2][2]*phi_t[3];
						
						auxX[4] = gama[0][0]*phi_r[4]+gama[0][1]*phi_s[4]+gama[0][2]*phi_t[4];
						auxY[4] = gama[1][0]*phi_r[4]+gama[1][1]*phi_s[4]+gama[1][2]*phi_t[4];
						auxZ[4] = gama[2][0]*phi_r[4]+gama[2][1]*phi_s[4]+gama[2][2]*phi_t[4];
							
						auxX[5] = gama[0][0]*phi_r[5]+gama[0][1]*phi_s[5]+gama[0][2]*phi_t[5];
						auxY[5] = gama[1][0]*phi_r[5]+gama[1][1]*phi_s[5]+gama[1][2]*phi_t[5];
						auxZ[5] = gama[2][0]*phi_r[5]+gama[2][1]*phi_s[5]+gama[2][2]*phi_t[5];
						
						auxX[6] = gama[0][0]*phi_r[6]+gama[0][1]*phi_s[6]+gama[0][2]*phi_t[6];
						auxY[6] = gama[1][0]*phi_r[6]+gama[1][1]*phi_s[6]+gama[1][2]*phi_t[6];
						auxZ[6] = gama[2][0]*phi_r[6]+gama[2][1]*phi_s[6]+gama[2][2]*phi_t[6];
						
						auxX[7] = gama[0][0]*phi_r[7]+gama[0][1]*phi_s[7]+gama[0][2]*phi_t[7];
						auxY[7] = gama[1][0]*phi_r[7]+gama[1][1]*phi_s[7]+gama[1][2]*phi_t[7];
						auxZ[7] = gama[2][0]*phi_r[7]+gama[2][1]*phi_s[7]+gama[2][2]*phi_t[7];
											
						__syncthreads();  // Line 0:
							
						Matrix_K_Aux_d[thread_id+12*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[4]*C00*auxX[0] + auxY[4]*C33*auxY[0] + auxZ[4]*C33*auxZ[0]);
						Matrix_K_Aux_d[thread_id+13*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[4]*C01*auxY[0] + auxY[4]*C33*auxX[0]);
						Matrix_K_Aux_d[thread_id+14*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[4]*C01*auxZ[0] + auxZ[4]*C33*auxX[0]);
							
						Matrix_K_Aux_d[thread_id+15*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[4]*C00*auxX[1] + auxY[4]*C33*auxY[1] + auxZ[4]*C33*auxZ[1]);
						Matrix_K_Aux_d[thread_id+16*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[4]*C01*auxY[1] + auxY[4]*C33*auxX[1]);
						Matrix_K_Aux_d[thread_id+17*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[4]*C01*auxZ[1] + auxZ[4]*C33*auxX[1]);
							
						Matrix_K_Aux_d[thread_id+21*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[4]*C00*auxX[3] + auxY[4]*C33*auxY[3] + auxZ[4]*C33*auxZ[3]);
						Matrix_K_Aux_d[thread_id+22*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[4]*C01*auxY[3] + auxY[4]*C33*auxX[3]);
						Matrix_K_Aux_d[thread_id+23*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[4]*C01*auxZ[3] + auxZ[4]*C33*auxX[3]);
							
						Matrix_K_Aux_d[thread_id+24*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[4]*C00*auxX[2] + auxY[4]*C33*auxY[2] + auxZ[4]*C33*auxZ[2]);
						Matrix_K_Aux_d[thread_id+25*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[4]*C01*auxY[2] + auxY[4]*C33*auxX[2]);
						Matrix_K_Aux_d[thread_id+26*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[4]*C01*auxZ[2] + auxZ[4]*C33*auxX[2]);
							
						Matrix_K_Aux_d[thread_id+39*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[4]*C00*auxX[4] + auxY[4]*C33*auxY[4] + auxZ[4]*C33*auxZ[4]);
						Matrix_K_Aux_d[thread_id+40*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[4]*C01*auxY[4] + auxY[4]*C33*auxX[4]);
						Matrix_K_Aux_d[thread_id+41*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[4]*C01*auxZ[4] + auxZ[4]*C33*auxX[4]);
							
						Matrix_K_Aux_d[thread_id+42*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[4]*C00*auxX[5] + auxY[4]*C33*auxY[5] + auxZ[4]*C33*auxZ[5]);
						Matrix_K_Aux_d[thread_id+43*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[4]*C01*auxY[5] + auxY[4]*C33*auxX[5]);
						Matrix_K_Aux_d[thread_id+44*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[4]*C01*auxZ[5] + auxZ[4]*C33*auxX[5]);
							
						Matrix_K_Aux_d[thread_id+48*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[4]*C00*auxX[7] + auxY[4]*C33*auxY[7] + auxZ[4]*C33*auxZ[7]);
						Matrix_K_Aux_d[thread_id+49*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[4]*C01*auxY[7] + auxY[4]*C33*auxX[7]);
						Matrix_K_Aux_d[thread_id+50*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[4]*C01*auxZ[7] + auxZ[4]*C33*auxX[7]);
							
						Matrix_K_Aux_d[thread_id+51*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[4]*C00*auxX[6] + auxY[4]*C33*auxY[6] + auxZ[4]*C33*auxZ[6]);
						Matrix_K_Aux_d[thread_id+52*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[4]*C01*auxY[6] + auxY[4]*C33*auxX[6]);
						Matrix_K_Aux_d[thread_id+53*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[4]*C01*auxZ[6] + auxZ[4]*C33*auxX[6]);
						
						__syncthreads();  // Line 1:
							
						Matrix_K_Aux_d[thread_id+12*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[4]*C01*auxX[0] + auxX[4]*C33*auxY[0]);
						Matrix_K_Aux_d[thread_id+13*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[4]*C00*auxY[0] + auxX[4]*C33*auxX[0] + auxZ[4]*C33*auxZ[0]);
						Matrix_K_Aux_d[thread_id+14*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[4]*C01*auxZ[0] + auxZ[4]*C33*auxY[0]);
							
						Matrix_K_Aux_d[thread_id+15*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[4]*C01*auxX[1] + auxX[4]*C33*auxY[1]);
						Matrix_K_Aux_d[thread_id+16*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[4]*C00*auxY[1] + auxX[4]*C33*auxX[1] + auxZ[4]*C33*auxZ[1]);
						Matrix_K_Aux_d[thread_id+17*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[4]*C01*auxZ[1] + auxZ[4]*C33*auxY[1]);
							
						Matrix_K_Aux_d[thread_id+21*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[4]*C01*auxX[3] + auxX[4]*C33*auxY[3]);
						Matrix_K_Aux_d[thread_id+22*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[4]*C00*auxY[3] + auxX[4]*C33*auxX[3] + auxZ[4]*C33*auxZ[3]);
						Matrix_K_Aux_d[thread_id+23*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[4]*C01*auxZ[3] + auxZ[4]*C33*auxY[3]);
							
						Matrix_K_Aux_d[thread_id+24*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[4]*C01*auxX[2] + auxX[4]*C33*auxY[2]);
						Matrix_K_Aux_d[thread_id+25*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[4]*C00*auxY[2] + auxX[4]*C33*auxX[2] + auxZ[4]*C33*auxZ[2]);
						Matrix_K_Aux_d[thread_id+26*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[4]*C01*auxZ[2] + auxZ[4]*C33*auxY[2]);
							
						Matrix_K_Aux_d[thread_id+39*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[4]*C01*auxX[4] + auxX[4]*C33*auxY[4]);
						Matrix_K_Aux_d[thread_id+40*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[4]*C00*auxY[4] + auxX[4]*C33*auxX[4] + auxZ[4]*C33*auxZ[4]);
						Matrix_K_Aux_d[thread_id+41*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[4]*C01*auxZ[4] + auxZ[4]*C33*auxY[4]);
							
						Matrix_K_Aux_d[thread_id+42*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[4]*C01*auxX[5] + auxX[4]*C33*auxY[5]);
						Matrix_K_Aux_d[thread_id+43*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[4]*C00*auxY[5] + auxX[4]*C33*auxX[5] + auxZ[4]*C33*auxZ[5]);
						Matrix_K_Aux_d[thread_id+44*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[4]*C01*auxZ[5] + auxZ[4]*C33*auxY[5]);
							
						Matrix_K_Aux_d[thread_id+48*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[4]*C01*auxX[7] + auxX[4]*C33*auxY[7]);
						Matrix_K_Aux_d[thread_id+49*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[4]*C00*auxY[7] + auxX[4]*C33*auxX[7] + auxZ[4]*C33*auxZ[7]);
						Matrix_K_Aux_d[thread_id+50*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[4]*C01*auxZ[7] + auxZ[4]*C33*auxY[7]);
							
						Matrix_K_Aux_d[thread_id+51*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[4]*C01*auxX[6] + auxX[4]*C33*auxY[6]);
						Matrix_K_Aux_d[thread_id+52*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[4]*C00*auxY[6] + auxX[4]*C33*auxX[6] + auxZ[4]*C33*auxZ[6]);
						Matrix_K_Aux_d[thread_id+53*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[4]*C01*auxZ[6] + auxZ[4]*C33*auxY[6]);
						
						__syncthreads();  // Line 2:
						
						Matrix_K_Aux_d[thread_id+12*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[4]*C01*auxX[0] + auxX[4]*C33*auxZ[0]);
						Matrix_K_Aux_d[thread_id+13*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[4]*C01*auxY[0] + auxY[4]*C33*auxZ[0]);
						Matrix_K_Aux_d[thread_id+14*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[4]*C00*auxZ[0] + auxY[4]*C33*auxY[0] + auxX[4]*C33*auxX[0]);
							
						Matrix_K_Aux_d[thread_id+15*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[4]*C01*auxX[1] + auxX[4]*C33*auxZ[1]);
						Matrix_K_Aux_d[thread_id+16*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[4]*C01*auxY[1] + auxY[4]*C33*auxZ[1]);
						Matrix_K_Aux_d[thread_id+17*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[4]*C00*auxZ[1] + auxY[4]*C33*auxY[1] + auxX[4]*C33*auxX[1]);
							
						Matrix_K_Aux_d[thread_id+21*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[4]*C01*auxX[3] + auxX[4]*C33*auxZ[3]);
						Matrix_K_Aux_d[thread_id+22*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[4]*C01*auxY[3] + auxY[4]*C33*auxZ[3]);
						Matrix_K_Aux_d[thread_id+23*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[4]*C00*auxZ[3] + auxY[4]*C33*auxY[3] + auxX[4]*C33*auxX[3]);
							
						Matrix_K_Aux_d[thread_id+24*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[4]*C01*auxX[2] + auxX[4]*C33*auxZ[2]);
						Matrix_K_Aux_d[thread_id+25*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[4]*C01*auxY[2] + auxY[4]*C33*auxZ[2]);
						Matrix_K_Aux_d[thread_id+26*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[4]*C00*auxZ[2] + auxY[4]*C33*auxY[2] + auxX[4]*C33*auxX[2]);
							
						Matrix_K_Aux_d[thread_id+39*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[4]*C01*auxX[4] + auxX[4]*C33*auxZ[4]);
						Matrix_K_Aux_d[thread_id+40*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[4]*C01*auxY[4] + auxY[4]*C33*auxZ[4]);
						Matrix_K_Aux_d[thread_id+41*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[4]*C00*auxZ[4] + auxY[4]*C33*auxY[4] + auxX[4]*C33*auxX[4]);
							
						Matrix_K_Aux_d[thread_id+42*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[4]*C01*auxX[5] + auxX[4]*C33*auxZ[5]);
						Matrix_K_Aux_d[thread_id+43*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[4]*C01*auxY[5] + auxY[4]*C33*auxZ[5]);
						Matrix_K_Aux_d[thread_id+44*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[4]*C00*auxZ[5] + auxY[4]*C33*auxY[5] + auxX[4]*C33*auxX[5]);
							
						Matrix_K_Aux_d[thread_id+48*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[4]*C01*auxX[7] + auxX[4]*C33*auxZ[7]);
						Matrix_K_Aux_d[thread_id+49*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[4]*C01*auxY[7] + auxY[4]*C33*auxZ[7]);
						Matrix_K_Aux_d[thread_id+50*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[4]*C00*auxZ[7] + auxY[4]*C33*auxY[7] + auxX[4]*C33*auxX[7]);
							
						Matrix_K_Aux_d[thread_id+51*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[4]*C01*auxX[6] + auxX[4]*C33*auxZ[6]);
						Matrix_K_Aux_d[thread_id+52*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[4]*C01*auxY[6] + auxY[4]*C33*auxZ[6]);
						Matrix_K_Aux_d[thread_id+53*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[4]*C00*auxZ[6] + auxY[4]*C33*auxY[6] + auxX[4]*C33*auxX[6]);
						
					}
					
				}                                                        // =================== Loop kg ===================
											
			}                                                            // =================== Loop jg ===================

		}                                                                // =================== Loop ig ===================
		
	}

}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============

__global__ void AssemblyStiffness3DQ5(int numno, int RowSize, double *CoordQ5_d, double *MatPropQ5_d, double *Matrix_K_Aux_d) 
{
	double r, s, t;  // 3
	double XEL0, YEL0, ZEL0, XEL1, YEL1, ZEL1, XEL2, YEL2, ZEL2, XEL3, YEL3, ZEL3;  // 12 
	double XEL4, YEL4, ZEL4, XEL5, YEL5, ZEL5, XEL6, YEL6, ZEL6, XEL7, YEL7, ZEL7;  // 12 
	double xgaus[2], wgaus[2];  // 4
	double p, E, C00, C01, C33;  // 5
	double phi_r[8], phi_s[8], phi_t[8];  // 24
	double jac[3][3];  // 9
	double jacob;  // 1
	double gama[3][3];  // 9
	double auxX[8], auxY[8], auxZ[8];  // 24
	int ig, jg, kg;  // 3
	
	const int xIndex = blockIdx.x*blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y*blockDim.y + threadIdx.y;
	const int thread_id = (gridDim.x*blockDim.x)*yIndex + xIndex;
	
	const int NumThread = (gridDim.x*blockDim.x)*(gridDim.y*blockDim.y);
	const int NumThreadSpace = RowSize*NumThread;
	
	__syncthreads();
	
	if(thread_id < numno) {
	
		XEL0 = CoordQ5_d[thread_id+ 0*numno];
		YEL0 = CoordQ5_d[thread_id+ 1*numno];
		ZEL0 = CoordQ5_d[thread_id+ 2*numno];   	
		XEL1 = CoordQ5_d[thread_id+ 3*numno];
		YEL1 = CoordQ5_d[thread_id+ 4*numno];  
		ZEL1 = CoordQ5_d[thread_id+ 5*numno];  	
		XEL2 = CoordQ5_d[thread_id+ 6*numno];
		YEL2 = CoordQ5_d[thread_id+ 7*numno]; 
		ZEL2 = CoordQ5_d[thread_id+ 8*numno];  		
		XEL3 = CoordQ5_d[thread_id+ 9*numno];
		YEL3 = CoordQ5_d[thread_id+10*numno];
		ZEL3 = CoordQ5_d[thread_id+11*numno];
		
		XEL4 = CoordQ5_d[thread_id+12*numno];
		YEL4 = CoordQ5_d[thread_id+13*numno];
		ZEL4 = CoordQ5_d[thread_id+14*numno];   	
		XEL5 = CoordQ5_d[thread_id+15*numno];
		YEL5 = CoordQ5_d[thread_id+16*numno];  
		ZEL5 = CoordQ5_d[thread_id+17*numno];  	
		XEL6 = CoordQ5_d[thread_id+18*numno];
		YEL6 = CoordQ5_d[thread_id+19*numno]; 
		ZEL6 = CoordQ5_d[thread_id+20*numno];  		
		XEL7 = CoordQ5_d[thread_id+21*numno];
		YEL7 = CoordQ5_d[thread_id+22*numno];
		ZEL7 = CoordQ5_d[thread_id+23*numno];
		
		__syncthreads();

		xgaus[0]=-0.577350269189626;
		xgaus[1]= 0.577350269189626;
		wgaus[0]= 1.;
		wgaus[1]= 1.;
							
		E = MatPropQ5_d[thread_id+ 0*numno];
		p = MatPropQ5_d[thread_id+ 1*numno];
					
		C00=(E*(1-p))/((1+p)*(1-2*p));  
		C01=(E*(1-p))/((1+p)*(1-2*p))*(p/(1-p)); 
		C33=(E*(1-p))/((1+p)*(1-2*p))*((1-2*p)/(2*(1-p)));
			
		for(ig=0; ig<2; ig++) {                                          // =================== Loop ig ===================
			t=xgaus[ig];
					
			for(jg=0; jg<2; jg++) {                                      // =================== Loop jg ===================
				s=xgaus[jg]; 
				
				for(kg=0; kg<2; kg++) {                                  // =================== Loop kg ===================
				r=xgaus[kg]; 
				
					phi_r[0] = -0.125*(1.0-s)*(1.0-t);
					phi_r[1] =  0.125*(1.0-s)*(1.0-t);
					phi_r[2] =  0.125*(1.0+s)*(1.0-t);
					phi_r[3] = -0.125*(1.0+s)*(1.0-t);
					phi_r[4] = -0.125*(1.0-s)*(1.0+t);
					phi_r[5] =  0.125*(1.0-s)*(1.0+t);
					phi_r[6] =  0.125*(1.0+s)*(1.0+t);
					phi_r[7] = -0.125*(1.0+s)*(1.0+t);
	 
					phi_s[0] = -0.125*(1.0-r)*(1.0-t);
					phi_s[1] = -0.125*(1.0+r)*(1.0-t);
					phi_s[2] =  0.125*(1.0+r)*(1.0-t);
					phi_s[3] =  0.125*(1.0-r)*(1.0-t);
					phi_s[4] = -0.125*(1.0-r)*(1.0+t);
					phi_s[5] = -0.125*(1.0+r)*(1.0+t);
					phi_s[6] =  0.125*(1.0+r)*(1.0+t);
					phi_s[7] =  0.125*(1.0-r)*(1.0+t);
		 
					phi_t[0] = -0.125*(1.0-r)*(1.0-s);
					phi_t[1] = -0.125*(1.0+r)*(1.0-s);
					phi_t[2] = -0.125*(1.0+r)*(1.0+s);
					phi_t[3] = -0.125*(1.0-r)*(1.0+s);
					phi_t[4] =  0.125*(1.0-r)*(1.0-s);
					phi_t[5] =  0.125*(1.0+r)*(1.0-s);
					phi_t[6] =  0.125*(1.0+r)*(1.0+s);
					phi_t[7] =  0.125*(1.0-r)*(1.0+s);
						
					jac[0][0]=phi_r[0]*XEL0+phi_r[1]*XEL1+phi_r[2]*XEL2+phi_r[3]*XEL3+phi_r[4]*XEL4+phi_r[5]*XEL5+phi_r[6]*XEL6+phi_r[7]*XEL7;
					jac[0][1]=phi_r[0]*YEL0+phi_r[1]*YEL1+phi_r[2]*YEL2+phi_r[3]*YEL3+phi_r[4]*YEL4+phi_r[5]*YEL5+phi_r[6]*YEL6+phi_r[7]*YEL7;
					jac[0][2]=phi_r[0]*ZEL0+phi_r[1]*ZEL1+phi_r[2]*ZEL2+phi_r[3]*ZEL3+phi_r[4]*ZEL4+phi_r[5]*ZEL5+phi_r[6]*ZEL6+phi_r[7]*ZEL7;

					jac[1][0]=phi_s[0]*XEL0+phi_s[1]*XEL1+phi_s[2]*XEL2+phi_s[3]*XEL3+phi_s[4]*XEL4+phi_s[5]*XEL5+phi_s[6]*XEL6+phi_s[7]*XEL7;
					jac[1][1]=phi_s[0]*YEL0+phi_s[1]*YEL1+phi_s[2]*YEL2+phi_s[3]*YEL3+phi_s[4]*YEL4+phi_s[5]*YEL5+phi_s[6]*YEL6+phi_s[7]*YEL7;
					jac[1][2]=phi_s[0]*ZEL0+phi_s[1]*ZEL1+phi_s[2]*ZEL2+phi_s[3]*ZEL3+phi_s[4]*ZEL4+phi_s[5]*ZEL5+phi_s[6]*ZEL6+phi_s[7]*ZEL7;
					
					jac[2][0]=phi_t[0]*XEL0+phi_t[1]*XEL1+phi_t[2]*XEL2+phi_t[3]*XEL3+phi_t[4]*XEL4+phi_t[5]*XEL5+phi_t[6]*XEL6+phi_t[7]*XEL7;
					jac[2][1]=phi_t[0]*YEL0+phi_t[1]*YEL1+phi_t[2]*YEL2+phi_t[3]*YEL3+phi_t[4]*YEL4+phi_t[5]*YEL5+phi_t[6]*YEL6+phi_t[7]*YEL7;
					jac[2][2]=phi_t[0]*ZEL0+phi_t[1]*ZEL1+phi_t[2]*ZEL2+phi_t[3]*ZEL3+phi_t[4]*ZEL4+phi_t[5]*ZEL5+phi_t[6]*ZEL6+phi_t[7]*ZEL7;
							
					jacob = jac[0][0]*(jac[1][1]*jac[2][2]-jac[1][2]*jac[2][1])-jac[0][1]*(jac[1][0]*jac[2][2]-jac[1][2]*jac[2][0])+
							jac[0][2]*(jac[1][0]*jac[2][1]-jac[1][1]*jac[2][0]);
					
					if(jacob > 0.) {
					
						gama[0][0] = (jac[1][1]*jac[2][2] - jac[1][2]*jac[2][1]) /jacob;
						gama[0][1] =-(jac[0][1]*jac[2][2] - jac[0][2]*jac[2][1]) /jacob;
						gama[0][2] = (jac[0][1]*jac[1][2] - jac[0][2]*jac[1][1]) /jacob;
						
						gama[1][0] =-(jac[1][0]*jac[2][2] - jac[1][2]*jac[2][0]) /jacob;
						gama[1][1] = (jac[0][0]*jac[2][2] - jac[0][2]*jac[2][0]) /jacob;
						gama[1][2] =-(jac[0][0]*jac[1][2] - jac[0][2]*jac[1][0]) /jacob;
						
						gama[2][0] = (jac[1][0]*jac[2][1] - jac[1][1]*jac[2][0]) /jacob;
						gama[2][1] =-(jac[0][0]*jac[2][1] - jac[0][1]*jac[2][0]) /jacob;
						gama[2][2] = (jac[0][0]*jac[1][1] - jac[0][1]*jac[1][0]) /jacob;
										
						auxX[0] = gama[0][0]*phi_r[0]+gama[0][1]*phi_s[0]+gama[0][2]*phi_t[0];
						auxY[0] = gama[1][0]*phi_r[0]+gama[1][1]*phi_s[0]+gama[1][2]*phi_t[0];
						auxZ[0] = gama[2][0]*phi_r[0]+gama[2][1]*phi_s[0]+gama[2][2]*phi_t[0];
							
						auxX[1] = gama[0][0]*phi_r[1]+gama[0][1]*phi_s[1]+gama[0][2]*phi_t[1];
						auxY[1] = gama[1][0]*phi_r[1]+gama[1][1]*phi_s[1]+gama[1][2]*phi_t[1];
						auxZ[1] = gama[2][0]*phi_r[1]+gama[2][1]*phi_s[1]+gama[2][2]*phi_t[1];
						
						auxX[2] = gama[0][0]*phi_r[2]+gama[0][1]*phi_s[2]+gama[0][2]*phi_t[2];
						auxY[2] = gama[1][0]*phi_r[2]+gama[1][1]*phi_s[2]+gama[1][2]*phi_t[2];
						auxZ[2] = gama[2][0]*phi_r[2]+gama[2][1]*phi_s[2]+gama[2][2]*phi_t[2];
						
						auxX[3] = gama[0][0]*phi_r[3]+gama[0][1]*phi_s[3]+gama[0][2]*phi_t[3];
						auxY[3] = gama[1][0]*phi_r[3]+gama[1][1]*phi_s[3]+gama[1][2]*phi_t[3];
						auxZ[3] = gama[2][0]*phi_r[3]+gama[2][1]*phi_s[3]+gama[2][2]*phi_t[3];
						
						auxX[4] = gama[0][0]*phi_r[4]+gama[0][1]*phi_s[4]+gama[0][2]*phi_t[4];
						auxY[4] = gama[1][0]*phi_r[4]+gama[1][1]*phi_s[4]+gama[1][2]*phi_t[4];
						auxZ[4] = gama[2][0]*phi_r[4]+gama[2][1]*phi_s[4]+gama[2][2]*phi_t[4];
							
						auxX[5] = gama[0][0]*phi_r[5]+gama[0][1]*phi_s[5]+gama[0][2]*phi_t[5];
						auxY[5] = gama[1][0]*phi_r[5]+gama[1][1]*phi_s[5]+gama[1][2]*phi_t[5];
						auxZ[5] = gama[2][0]*phi_r[5]+gama[2][1]*phi_s[5]+gama[2][2]*phi_t[5];
						
						auxX[6] = gama[0][0]*phi_r[6]+gama[0][1]*phi_s[6]+gama[0][2]*phi_t[6];
						auxY[6] = gama[1][0]*phi_r[6]+gama[1][1]*phi_s[6]+gama[1][2]*phi_t[6];
						auxZ[6] = gama[2][0]*phi_r[6]+gama[2][1]*phi_s[6]+gama[2][2]*phi_t[6];
						
						auxX[7] = gama[0][0]*phi_r[7]+gama[0][1]*phi_s[7]+gama[0][2]*phi_t[7];
						auxY[7] = gama[1][0]*phi_r[7]+gama[1][1]*phi_s[7]+gama[1][2]*phi_t[7];
						auxZ[7] = gama[2][0]*phi_r[7]+gama[2][1]*phi_s[7]+gama[2][2]*phi_t[7];
						
						__syncthreads();  // Line 0:
							
						Matrix_K_Aux_d[thread_id+27*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[2]*C00*auxX[0] + auxY[2]*C33*auxY[0] + auxZ[2]*C33*auxZ[0]);
						Matrix_K_Aux_d[thread_id+28*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[2]*C01*auxY[0] + auxY[2]*C33*auxX[0]);
						Matrix_K_Aux_d[thread_id+29*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[2]*C01*auxZ[0] + auxZ[2]*C33*auxX[0]);
							
						Matrix_K_Aux_d[thread_id+30*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[2]*C00*auxX[1] + auxY[2]*C33*auxY[1] + auxZ[2]*C33*auxZ[1]);
						Matrix_K_Aux_d[thread_id+31*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[2]*C01*auxY[1] + auxY[2]*C33*auxX[1]);
						Matrix_K_Aux_d[thread_id+32*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[2]*C01*auxZ[1] + auxZ[2]*C33*auxX[1]);
							
						Matrix_K_Aux_d[thread_id+36*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[2]*C00*auxX[3] + auxY[2]*C33*auxY[3] + auxZ[2]*C33*auxZ[3]);
						Matrix_K_Aux_d[thread_id+37*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[2]*C01*auxY[3] + auxY[2]*C33*auxX[3]);
						Matrix_K_Aux_d[thread_id+38*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[2]*C01*auxZ[3] + auxZ[2]*C33*auxX[3]);
							
						Matrix_K_Aux_d[thread_id+39*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[2]*C00*auxX[2] + auxY[2]*C33*auxY[2] + auxZ[2]*C33*auxZ[2]);
						Matrix_K_Aux_d[thread_id+40*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[2]*C01*auxY[2] + auxY[2]*C33*auxX[2]);
						Matrix_K_Aux_d[thread_id+41*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[2]*C01*auxZ[2] + auxZ[2]*C33*auxX[2]);
							
						Matrix_K_Aux_d[thread_id+54*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[2]*C00*auxX[4] + auxY[2]*C33*auxY[4] + auxZ[2]*C33*auxZ[4]);
						Matrix_K_Aux_d[thread_id+55*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[2]*C01*auxY[4] + auxY[2]*C33*auxX[4]);
						Matrix_K_Aux_d[thread_id+56*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[2]*C01*auxZ[4] + auxZ[2]*C33*auxX[4]);
							
						Matrix_K_Aux_d[thread_id+57*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[2]*C00*auxX[5] + auxY[2]*C33*auxY[5] + auxZ[2]*C33*auxZ[5]);
						Matrix_K_Aux_d[thread_id+58*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[2]*C01*auxY[5] + auxY[2]*C33*auxX[5]);
						Matrix_K_Aux_d[thread_id+59*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[2]*C01*auxZ[5] + auxZ[2]*C33*auxX[5]);
							
						Matrix_K_Aux_d[thread_id+63*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[2]*C00*auxX[7] + auxY[2]*C33*auxY[7] + auxZ[2]*C33*auxZ[7]);
						Matrix_K_Aux_d[thread_id+64*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[2]*C01*auxY[7] + auxY[2]*C33*auxX[7]);
						Matrix_K_Aux_d[thread_id+65*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[2]*C01*auxZ[7] + auxZ[2]*C33*auxX[7]);
							
						Matrix_K_Aux_d[thread_id+66*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[2]*C00*auxX[6] + auxY[2]*C33*auxY[6] + auxZ[2]*C33*auxZ[6]);
						Matrix_K_Aux_d[thread_id+67*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[2]*C01*auxY[6] + auxY[2]*C33*auxX[6]);
						Matrix_K_Aux_d[thread_id+68*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[2]*C01*auxZ[6] + auxZ[2]*C33*auxX[6]);
							
						__syncthreads();  // Line 1:
						
						Matrix_K_Aux_d[thread_id+27*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[2]*C01*auxX[0] + auxX[2]*C33*auxY[0]);
						Matrix_K_Aux_d[thread_id+28*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[2]*C00*auxY[0] + auxX[2]*C33*auxX[0] + auxZ[2]*C33*auxZ[0]);
						Matrix_K_Aux_d[thread_id+29*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[2]*C01*auxZ[0] + auxZ[2]*C33*auxY[0]);
							
						Matrix_K_Aux_d[thread_id+30*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[2]*C01*auxX[1] + auxX[2]*C33*auxY[1]);
						Matrix_K_Aux_d[thread_id+31*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[2]*C00*auxY[1] + auxX[2]*C33*auxX[1] + auxZ[2]*C33*auxZ[1]);
						Matrix_K_Aux_d[thread_id+32*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[2]*C01*auxZ[1] + auxZ[2]*C33*auxY[1]);
							
						Matrix_K_Aux_d[thread_id+36*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[2]*C01*auxX[3] + auxX[2]*C33*auxY[3]);
						Matrix_K_Aux_d[thread_id+37*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[2]*C00*auxY[3] + auxX[2]*C33*auxX[3] + auxZ[2]*C33*auxZ[3]);
						Matrix_K_Aux_d[thread_id+38*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[2]*C01*auxZ[3] + auxZ[2]*C33*auxY[3]);
							
						Matrix_K_Aux_d[thread_id+39*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[2]*C01*auxX[2] + auxX[2]*C33*auxY[2]);
						Matrix_K_Aux_d[thread_id+40*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[2]*C00*auxY[2] + auxX[2]*C33*auxX[2] + auxZ[2]*C33*auxZ[2]);
						Matrix_K_Aux_d[thread_id+41*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[2]*C01*auxZ[2] + auxZ[2]*C33*auxY[2]);
							
						Matrix_K_Aux_d[thread_id+54*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[2]*C01*auxX[4] + auxX[2]*C33*auxY[4]);
						Matrix_K_Aux_d[thread_id+55*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[2]*C00*auxY[4] + auxX[2]*C33*auxX[4] + auxZ[2]*C33*auxZ[4]);
						Matrix_K_Aux_d[thread_id+56*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[2]*C01*auxZ[4] + auxZ[2]*C33*auxY[4]);
							
						Matrix_K_Aux_d[thread_id+57*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[2]*C01*auxX[5] + auxX[2]*C33*auxY[5]);
						Matrix_K_Aux_d[thread_id+58*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[2]*C00*auxY[5] + auxX[2]*C33*auxX[5] + auxZ[2]*C33*auxZ[5]);
						Matrix_K_Aux_d[thread_id+59*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[2]*C01*auxZ[5] + auxZ[2]*C33*auxY[5]);
							
						Matrix_K_Aux_d[thread_id+63*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[2]*C01*auxX[7] + auxX[2]*C33*auxY[7]);
						Matrix_K_Aux_d[thread_id+64*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[2]*C00*auxY[7] + auxX[2]*C33*auxX[7] + auxZ[2]*C33*auxZ[7]);
						Matrix_K_Aux_d[thread_id+65*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[2]*C01*auxZ[7] + auxZ[2]*C33*auxY[7]);
							
						Matrix_K_Aux_d[thread_id+66*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[2]*C01*auxX[6] + auxX[2]*C33*auxY[6]);
						Matrix_K_Aux_d[thread_id+67*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[2]*C00*auxY[6] + auxX[2]*C33*auxX[6] + auxZ[2]*C33*auxZ[6]);
						Matrix_K_Aux_d[thread_id+68*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[2]*C01*auxZ[6] + auxZ[2]*C33*auxY[6]);
						
						__syncthreads();  // Line 2:
							
						Matrix_K_Aux_d[thread_id+27*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[2]*C01*auxX[0] + auxX[2]*C33*auxZ[0]);
						Matrix_K_Aux_d[thread_id+28*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[2]*C01*auxY[0] + auxY[2]*C33*auxZ[0]);
						Matrix_K_Aux_d[thread_id+29*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[2]*C00*auxZ[0] + auxY[2]*C33*auxY[0] + auxX[2]*C33*auxX[0]);
							
						Matrix_K_Aux_d[thread_id+30*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[2]*C01*auxX[1] + auxX[2]*C33*auxZ[1]);
						Matrix_K_Aux_d[thread_id+31*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[2]*C01*auxY[1] + auxY[2]*C33*auxZ[1]);
						Matrix_K_Aux_d[thread_id+32*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[2]*C00*auxZ[1] + auxY[2]*C33*auxY[1] + auxX[2]*C33*auxX[1]);
							
						Matrix_K_Aux_d[thread_id+36*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[2]*C01*auxX[3] + auxX[2]*C33*auxZ[3]);
						Matrix_K_Aux_d[thread_id+37*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[2]*C01*auxY[3] + auxY[2]*C33*auxZ[3]);
						Matrix_K_Aux_d[thread_id+38*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[2]*C00*auxZ[3] + auxY[2]*C33*auxY[3] + auxX[2]*C33*auxX[3]);
							
						Matrix_K_Aux_d[thread_id+39*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[2]*C01*auxX[2] + auxX[2]*C33*auxZ[2]);
						Matrix_K_Aux_d[thread_id+40*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[2]*C01*auxY[2] + auxY[2]*C33*auxZ[2]);
						Matrix_K_Aux_d[thread_id+41*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[2]*C00*auxZ[2] + auxY[2]*C33*auxY[2] + auxX[2]*C33*auxX[2]);
							
						Matrix_K_Aux_d[thread_id+54*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[2]*C01*auxX[4] + auxX[2]*C33*auxZ[4]);
						Matrix_K_Aux_d[thread_id+55*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[2]*C01*auxY[4] + auxY[2]*C33*auxZ[4]);
						Matrix_K_Aux_d[thread_id+56*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[2]*C00*auxZ[4] + auxY[2]*C33*auxY[4] + auxX[2]*C33*auxX[4]);
							
						Matrix_K_Aux_d[thread_id+57*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[2]*C01*auxX[5] + auxX[2]*C33*auxZ[5]);
						Matrix_K_Aux_d[thread_id+58*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[2]*C01*auxY[5] + auxY[2]*C33*auxZ[5]);
						Matrix_K_Aux_d[thread_id+59*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[2]*C00*auxZ[5] + auxY[2]*C33*auxY[5] + auxX[2]*C33*auxX[5]);
							
						Matrix_K_Aux_d[thread_id+63*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[2]*C01*auxX[7] + auxX[2]*C33*auxZ[7]);
						Matrix_K_Aux_d[thread_id+64*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[2]*C01*auxY[7] + auxY[2]*C33*auxZ[7]);
						Matrix_K_Aux_d[thread_id+65*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[2]*C00*auxZ[7] + auxY[2]*C33*auxY[7] + auxX[2]*C33*auxX[7]);
							
						Matrix_K_Aux_d[thread_id+66*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[2]*C01*auxX[6] + auxX[2]*C33*auxZ[6]);
						Matrix_K_Aux_d[thread_id+67*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[2]*C01*auxY[6] + auxY[2]*C33*auxZ[6]);
						Matrix_K_Aux_d[thread_id+68*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[2]*C00*auxZ[6] + auxY[2]*C33*auxY[6] + auxX[2]*C33*auxX[6]);
			
					}
					
				}                                                        // =================== Loop kg ===================
											
			}                                                            // =================== Loop jg ===================

		}                                                                // =================== Loop ig ===================
		
	}
	
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============

__global__ void AssemblyStiffness3DQ6(int numno, int RowSize, double *CoordQ6_d, double *MatPropQ6_d, double *Matrix_K_Aux_d) 
{
	double r, s, t;  // 3
	double XEL0, YEL0, ZEL0, XEL1, YEL1, ZEL1, XEL2, YEL2, ZEL2, XEL3, YEL3, ZEL3;  // 12 
	double XEL4, YEL4, ZEL4, XEL5, YEL5, ZEL5, XEL6, YEL6, ZEL6, XEL7, YEL7, ZEL7;  // 12 
	double xgaus[2], wgaus[2];  // 4
	double p, E, C00, C01, C33;  // 5
	double phi_r[8], phi_s[8], phi_t[8];  // 24
	double jac[3][3];  // 9
	double jacob;  // 1
	double gama[3][3];  // 9
	double auxX[8], auxY[8], auxZ[8];  // 24
	int ig, jg, kg;  // 3
	
	const int xIndex = blockIdx.x*blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y*blockDim.y + threadIdx.y;
	const int thread_id = (gridDim.x*blockDim.x)*yIndex + xIndex;
	
	const int NumThread = (gridDim.x*blockDim.x)*(gridDim.y*blockDim.y);
	const int NumThreadSpace = RowSize*NumThread;
	
	__syncthreads();
	
	if(thread_id < numno) {
	
		XEL0 = CoordQ6_d[thread_id+ 0*numno];
		YEL0 = CoordQ6_d[thread_id+ 1*numno];
		ZEL0 = CoordQ6_d[thread_id+ 2*numno];   	
		XEL1 = CoordQ6_d[thread_id+ 3*numno];
		YEL1 = CoordQ6_d[thread_id+ 4*numno];  
		ZEL1 = CoordQ6_d[thread_id+ 5*numno];  	
		XEL2 = CoordQ6_d[thread_id+ 6*numno];
		YEL2 = CoordQ6_d[thread_id+ 7*numno]; 
		ZEL2 = CoordQ6_d[thread_id+ 8*numno];  		
		XEL3 = CoordQ6_d[thread_id+ 9*numno];
		YEL3 = CoordQ6_d[thread_id+10*numno];
		ZEL3 = CoordQ6_d[thread_id+11*numno];
		
		XEL4 = CoordQ6_d[thread_id+12*numno];
		YEL4 = CoordQ6_d[thread_id+13*numno];
		ZEL4 = CoordQ6_d[thread_id+14*numno];   	
		XEL5 = CoordQ6_d[thread_id+15*numno];
		YEL5 = CoordQ6_d[thread_id+16*numno];  
		ZEL5 = CoordQ6_d[thread_id+17*numno];  	
		XEL6 = CoordQ6_d[thread_id+18*numno];
		YEL6 = CoordQ6_d[thread_id+19*numno]; 
		ZEL6 = CoordQ6_d[thread_id+20*numno];  		
		XEL7 = CoordQ6_d[thread_id+21*numno];
		YEL7 = CoordQ6_d[thread_id+22*numno];
		ZEL7 = CoordQ6_d[thread_id+23*numno];
		
		__syncthreads();

		xgaus[0]=-0.577350269189626;
		xgaus[1]= 0.577350269189626;
		wgaus[0]= 1.;
		wgaus[1]= 1.;
							
		E = MatPropQ6_d[thread_id+ 0*numno];
		p = MatPropQ6_d[thread_id+ 1*numno];
					
		C00=(E*(1-p))/((1+p)*(1-2*p));  
		C01=(E*(1-p))/((1+p)*(1-2*p))*(p/(1-p)); 
		C33=(E*(1-p))/((1+p)*(1-2*p))*((1-2*p)/(2*(1-p)));
			
		for(ig=0; ig<2; ig++) {                                          // =================== Loop ig ===================
			t=xgaus[ig];
					
			for(jg=0; jg<2; jg++) {                                      // =================== Loop jg ===================
				s=xgaus[jg]; 
				
				for(kg=0; kg<2; kg++) {                                  // =================== Loop kg ===================
				r=xgaus[kg]; 
				
					phi_r[0] = -0.125*(1.0-s)*(1.0-t);
					phi_r[1] =  0.125*(1.0-s)*(1.0-t);
					phi_r[2] =  0.125*(1.0+s)*(1.0-t);
					phi_r[3] = -0.125*(1.0+s)*(1.0-t);
					phi_r[4] = -0.125*(1.0-s)*(1.0+t);
					phi_r[5] =  0.125*(1.0-s)*(1.0+t);
					phi_r[6] =  0.125*(1.0+s)*(1.0+t);
					phi_r[7] = -0.125*(1.0+s)*(1.0+t);
	 
					phi_s[0] = -0.125*(1.0-r)*(1.0-t);
					phi_s[1] = -0.125*(1.0+r)*(1.0-t);
					phi_s[2] =  0.125*(1.0+r)*(1.0-t);
					phi_s[3] =  0.125*(1.0-r)*(1.0-t);
					phi_s[4] = -0.125*(1.0-r)*(1.0+t);
					phi_s[5] = -0.125*(1.0+r)*(1.0+t);
					phi_s[6] =  0.125*(1.0+r)*(1.0+t);
					phi_s[7] =  0.125*(1.0-r)*(1.0+t);
		 
					phi_t[0] = -0.125*(1.0-r)*(1.0-s);
					phi_t[1] = -0.125*(1.0+r)*(1.0-s);
					phi_t[2] = -0.125*(1.0+r)*(1.0+s);
					phi_t[3] = -0.125*(1.0-r)*(1.0+s);
					phi_t[4] =  0.125*(1.0-r)*(1.0-s);
					phi_t[5] =  0.125*(1.0+r)*(1.0-s);
					phi_t[6] =  0.125*(1.0+r)*(1.0+s);
					phi_t[7] =  0.125*(1.0-r)*(1.0+s);
						
					jac[0][0]=phi_r[0]*XEL0+phi_r[1]*XEL1+phi_r[2]*XEL2+phi_r[3]*XEL3+phi_r[4]*XEL4+phi_r[5]*XEL5+phi_r[6]*XEL6+phi_r[7]*XEL7;
					jac[0][1]=phi_r[0]*YEL0+phi_r[1]*YEL1+phi_r[2]*YEL2+phi_r[3]*YEL3+phi_r[4]*YEL4+phi_r[5]*YEL5+phi_r[6]*YEL6+phi_r[7]*YEL7;
					jac[0][2]=phi_r[0]*ZEL0+phi_r[1]*ZEL1+phi_r[2]*ZEL2+phi_r[3]*ZEL3+phi_r[4]*ZEL4+phi_r[5]*ZEL5+phi_r[6]*ZEL6+phi_r[7]*ZEL7;

					jac[1][0]=phi_s[0]*XEL0+phi_s[1]*XEL1+phi_s[2]*XEL2+phi_s[3]*XEL3+phi_s[4]*XEL4+phi_s[5]*XEL5+phi_s[6]*XEL6+phi_s[7]*XEL7;
					jac[1][1]=phi_s[0]*YEL0+phi_s[1]*YEL1+phi_s[2]*YEL2+phi_s[3]*YEL3+phi_s[4]*YEL4+phi_s[5]*YEL5+phi_s[6]*YEL6+phi_s[7]*YEL7;
					jac[1][2]=phi_s[0]*ZEL0+phi_s[1]*ZEL1+phi_s[2]*ZEL2+phi_s[3]*ZEL3+phi_s[4]*ZEL4+phi_s[5]*ZEL5+phi_s[6]*ZEL6+phi_s[7]*ZEL7;
					
					jac[2][0]=phi_t[0]*XEL0+phi_t[1]*XEL1+phi_t[2]*XEL2+phi_t[3]*XEL3+phi_t[4]*XEL4+phi_t[5]*XEL5+phi_t[6]*XEL6+phi_t[7]*XEL7;
					jac[2][1]=phi_t[0]*YEL0+phi_t[1]*YEL1+phi_t[2]*YEL2+phi_t[3]*YEL3+phi_t[4]*YEL4+phi_t[5]*YEL5+phi_t[6]*YEL6+phi_t[7]*YEL7;
					jac[2][2]=phi_t[0]*ZEL0+phi_t[1]*ZEL1+phi_t[2]*ZEL2+phi_t[3]*ZEL3+phi_t[4]*ZEL4+phi_t[5]*ZEL5+phi_t[6]*ZEL6+phi_t[7]*ZEL7;
							
					jacob = jac[0][0]*(jac[1][1]*jac[2][2]-jac[1][2]*jac[2][1])-jac[0][1]*(jac[1][0]*jac[2][2]-jac[1][2]*jac[2][0])+
							jac[0][2]*(jac[1][0]*jac[2][1]-jac[1][1]*jac[2][0]);
					
					if(jacob > 0.) {
					
						gama[0][0] = (jac[1][1]*jac[2][2] - jac[1][2]*jac[2][1]) /jacob;
						gama[0][1] =-(jac[0][1]*jac[2][2] - jac[0][2]*jac[2][1]) /jacob;
						gama[0][2] = (jac[0][1]*jac[1][2] - jac[0][2]*jac[1][1]) /jacob;
						
						gama[1][0] =-(jac[1][0]*jac[2][2] - jac[1][2]*jac[2][0]) /jacob;
						gama[1][1] = (jac[0][0]*jac[2][2] - jac[0][2]*jac[2][0]) /jacob;
						gama[1][2] =-(jac[0][0]*jac[1][2] - jac[0][2]*jac[1][0]) /jacob;
						
						gama[2][0] = (jac[1][0]*jac[2][1] - jac[1][1]*jac[2][0]) /jacob;
						gama[2][1] =-(jac[0][0]*jac[2][1] - jac[0][1]*jac[2][0]) /jacob;
						gama[2][2] = (jac[0][0]*jac[1][1] - jac[0][1]*jac[1][0]) /jacob;
										
						auxX[0] = gama[0][0]*phi_r[0]+gama[0][1]*phi_s[0]+gama[0][2]*phi_t[0];
						auxY[0] = gama[1][0]*phi_r[0]+gama[1][1]*phi_s[0]+gama[1][2]*phi_t[0];
						auxZ[0] = gama[2][0]*phi_r[0]+gama[2][1]*phi_s[0]+gama[2][2]*phi_t[0];
							
						auxX[1] = gama[0][0]*phi_r[1]+gama[0][1]*phi_s[1]+gama[0][2]*phi_t[1];
						auxY[1] = gama[1][0]*phi_r[1]+gama[1][1]*phi_s[1]+gama[1][2]*phi_t[1];
						auxZ[1] = gama[2][0]*phi_r[1]+gama[2][1]*phi_s[1]+gama[2][2]*phi_t[1];
						
						auxX[2] = gama[0][0]*phi_r[2]+gama[0][1]*phi_s[2]+gama[0][2]*phi_t[2];
						auxY[2] = gama[1][0]*phi_r[2]+gama[1][1]*phi_s[2]+gama[1][2]*phi_t[2];
						auxZ[2] = gama[2][0]*phi_r[2]+gama[2][1]*phi_s[2]+gama[2][2]*phi_t[2];
						
						auxX[3] = gama[0][0]*phi_r[3]+gama[0][1]*phi_s[3]+gama[0][2]*phi_t[3];
						auxY[3] = gama[1][0]*phi_r[3]+gama[1][1]*phi_s[3]+gama[1][2]*phi_t[3];
						auxZ[3] = gama[2][0]*phi_r[3]+gama[2][1]*phi_s[3]+gama[2][2]*phi_t[3];
						
						auxX[4] = gama[0][0]*phi_r[4]+gama[0][1]*phi_s[4]+gama[0][2]*phi_t[4];
						auxY[4] = gama[1][0]*phi_r[4]+gama[1][1]*phi_s[4]+gama[1][2]*phi_t[4];
						auxZ[4] = gama[2][0]*phi_r[4]+gama[2][1]*phi_s[4]+gama[2][2]*phi_t[4];
							
						auxX[5] = gama[0][0]*phi_r[5]+gama[0][1]*phi_s[5]+gama[0][2]*phi_t[5];
						auxY[5] = gama[1][0]*phi_r[5]+gama[1][1]*phi_s[5]+gama[1][2]*phi_t[5];
						auxZ[5] = gama[2][0]*phi_r[5]+gama[2][1]*phi_s[5]+gama[2][2]*phi_t[5];
						
						auxX[6] = gama[0][0]*phi_r[6]+gama[0][1]*phi_s[6]+gama[0][2]*phi_t[6];
						auxY[6] = gama[1][0]*phi_r[6]+gama[1][1]*phi_s[6]+gama[1][2]*phi_t[6];
						auxZ[6] = gama[2][0]*phi_r[6]+gama[2][1]*phi_s[6]+gama[2][2]*phi_t[6];
						
						auxX[7] = gama[0][0]*phi_r[7]+gama[0][1]*phi_s[7]+gama[0][2]*phi_t[7];
						auxY[7] = gama[1][0]*phi_r[7]+gama[1][1]*phi_s[7]+gama[1][2]*phi_t[7];
						auxZ[7] = gama[2][0]*phi_r[7]+gama[2][1]*phi_s[7]+gama[2][2]*phi_t[7];

						__syncthreads();  // Line 0:
							
						Matrix_K_Aux_d[thread_id+30*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[3]*C00*auxX[0] + auxY[3]*C33*auxY[0] + auxZ[3]*C33*auxZ[0]);
						Matrix_K_Aux_d[thread_id+31*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[3]*C01*auxY[0] + auxY[3]*C33*auxX[0]);
						Matrix_K_Aux_d[thread_id+32*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[3]*C01*auxZ[0] + auxZ[3]*C33*auxX[0]);
							
						Matrix_K_Aux_d[thread_id+33*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[3]*C00*auxX[1] + auxY[3]*C33*auxY[1] + auxZ[3]*C33*auxZ[1]);
						Matrix_K_Aux_d[thread_id+34*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[3]*C01*auxY[1] + auxY[3]*C33*auxX[1]);
						Matrix_K_Aux_d[thread_id+35*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[3]*C01*auxZ[1] + auxZ[3]*C33*auxX[1]);
							
						Matrix_K_Aux_d[thread_id+39*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[3]*C00*auxX[3] + auxY[3]*C33*auxY[3] + auxZ[3]*C33*auxZ[3]);
						Matrix_K_Aux_d[thread_id+40*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[3]*C01*auxY[3] + auxY[3]*C33*auxX[3]);
						Matrix_K_Aux_d[thread_id+41*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[3]*C01*auxZ[3] + auxZ[3]*C33*auxX[3]);
							
						Matrix_K_Aux_d[thread_id+42*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[3]*C00*auxX[2] + auxY[3]*C33*auxY[2] + auxZ[3]*C33*auxZ[2]);
						Matrix_K_Aux_d[thread_id+43*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[3]*C01*auxY[2] + auxY[3]*C33*auxX[2]);
						Matrix_K_Aux_d[thread_id+44*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[3]*C01*auxZ[2] + auxZ[3]*C33*auxX[2]);
							
						Matrix_K_Aux_d[thread_id+57*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[3]*C00*auxX[4] + auxY[3]*C33*auxY[4] + auxZ[3]*C33*auxZ[4]);
						Matrix_K_Aux_d[thread_id+58*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[3]*C01*auxY[4] + auxY[3]*C33*auxX[4]);
						Matrix_K_Aux_d[thread_id+59*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[3]*C01*auxZ[4] + auxZ[3]*C33*auxX[4]);
							
						Matrix_K_Aux_d[thread_id+60*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[3]*C00*auxX[5] + auxY[3]*C33*auxY[5] + auxZ[3]*C33*auxZ[5]);
						Matrix_K_Aux_d[thread_id+61*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[3]*C01*auxY[5] + auxY[3]*C33*auxX[5]);
						Matrix_K_Aux_d[thread_id+62*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[3]*C01*auxZ[5] + auxZ[3]*C33*auxX[5]);
							
						Matrix_K_Aux_d[thread_id+66*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[3]*C00*auxX[7] + auxY[3]*C33*auxY[7] + auxZ[3]*C33*auxZ[7]);
						Matrix_K_Aux_d[thread_id+67*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[3]*C01*auxY[7] + auxY[3]*C33*auxX[7]);
						Matrix_K_Aux_d[thread_id+68*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[3]*C01*auxZ[7] + auxZ[3]*C33*auxX[7]);
							
						Matrix_K_Aux_d[thread_id+69*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[3]*C00*auxX[6] + auxY[3]*C33*auxY[6] + auxZ[3]*C33*auxZ[6]);
						Matrix_K_Aux_d[thread_id+70*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[3]*C01*auxY[6] + auxY[3]*C33*auxX[6]);
						Matrix_K_Aux_d[thread_id+71*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[3]*C01*auxZ[6] + auxZ[3]*C33*auxX[6]);
						
						__syncthreads();  // Line 1:
						
						Matrix_K_Aux_d[thread_id+30*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[3]*C01*auxX[0] + auxX[3]*C33*auxY[0]);
						Matrix_K_Aux_d[thread_id+31*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[3]*C00*auxY[0] + auxX[3]*C33*auxX[0] + auxZ[3]*C33*auxZ[0]);
						Matrix_K_Aux_d[thread_id+32*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[3]*C01*auxZ[0] + auxZ[3]*C33*auxY[0]);
							
						Matrix_K_Aux_d[thread_id+33*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[3]*C01*auxX[1] + auxX[3]*C33*auxY[1]);
						Matrix_K_Aux_d[thread_id+34*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[3]*C00*auxY[1] + auxX[3]*C33*auxX[1] + auxZ[3]*C33*auxZ[1]);
						Matrix_K_Aux_d[thread_id+35*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[3]*C01*auxZ[1] + auxZ[3]*C33*auxY[1]);
							
						Matrix_K_Aux_d[thread_id+39*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[3]*C01*auxX[3] + auxX[3]*C33*auxY[3]);
						Matrix_K_Aux_d[thread_id+40*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[3]*C00*auxY[3] + auxX[3]*C33*auxX[3] + auxZ[3]*C33*auxZ[3]);
						Matrix_K_Aux_d[thread_id+41*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[3]*C01*auxZ[3] + auxZ[3]*C33*auxY[3]);
							
						Matrix_K_Aux_d[thread_id+42*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[3]*C01*auxX[2] + auxX[3]*C33*auxY[2]);
						Matrix_K_Aux_d[thread_id+43*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[3]*C00*auxY[2] + auxX[3]*C33*auxX[2] + auxZ[3]*C33*auxZ[2]);
						Matrix_K_Aux_d[thread_id+44*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[3]*C01*auxZ[2] + auxZ[3]*C33*auxY[2]);
							
						Matrix_K_Aux_d[thread_id+57*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[3]*C01*auxX[4] + auxX[3]*C33*auxY[4]);
						Matrix_K_Aux_d[thread_id+58*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[3]*C00*auxY[4] + auxX[3]*C33*auxX[4] + auxZ[3]*C33*auxZ[4]);
						Matrix_K_Aux_d[thread_id+59*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[3]*C01*auxZ[4] + auxZ[3]*C33*auxY[4]);
							
						Matrix_K_Aux_d[thread_id+60*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[3]*C01*auxX[5] + auxX[3]*C33*auxY[5]);
						Matrix_K_Aux_d[thread_id+61*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[3]*C00*auxY[5] + auxX[3]*C33*auxX[5] + auxZ[3]*C33*auxZ[5]);
						Matrix_K_Aux_d[thread_id+62*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[3]*C01*auxZ[5] + auxZ[3]*C33*auxY[5]);
							
						Matrix_K_Aux_d[thread_id+66*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[3]*C01*auxX[7] + auxX[3]*C33*auxY[7]);
						Matrix_K_Aux_d[thread_id+67*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[3]*C00*auxY[7] + auxX[3]*C33*auxX[7] + auxZ[3]*C33*auxZ[7]);
						Matrix_K_Aux_d[thread_id+68*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[3]*C01*auxZ[7] + auxZ[3]*C33*auxY[7]);
							
						Matrix_K_Aux_d[thread_id+69*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[3]*C01*auxX[6] + auxX[3]*C33*auxY[6]);
						Matrix_K_Aux_d[thread_id+70*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[3]*C00*auxY[6] + auxX[3]*C33*auxX[6] + auxZ[3]*C33*auxZ[6]);
						Matrix_K_Aux_d[thread_id+71*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[3]*C01*auxZ[6] + auxZ[3]*C33*auxY[6]);
				
						__syncthreads();  // Line 2:
						
						Matrix_K_Aux_d[thread_id+30*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[3]*C01*auxX[0] + auxX[3]*C33*auxZ[0]);
						Matrix_K_Aux_d[thread_id+31*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[3]*C01*auxY[0] + auxY[3]*C33*auxZ[0]);
						Matrix_K_Aux_d[thread_id+32*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[3]*C00*auxZ[0] + auxY[3]*C33*auxY[0] + auxX[3]*C33*auxX[0]);
							
						Matrix_K_Aux_d[thread_id+33*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[3]*C01*auxX[1] + auxX[3]*C33*auxZ[1]);
						Matrix_K_Aux_d[thread_id+34*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[3]*C01*auxY[1] + auxY[3]*C33*auxZ[1]);
						Matrix_K_Aux_d[thread_id+35*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[3]*C00*auxZ[1] + auxY[3]*C33*auxY[1] + auxX[3]*C33*auxX[1]);
							
						Matrix_K_Aux_d[thread_id+39*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[3]*C01*auxX[3] + auxX[3]*C33*auxZ[3]);
						Matrix_K_Aux_d[thread_id+40*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[3]*C01*auxY[3] + auxY[3]*C33*auxZ[3]);
						Matrix_K_Aux_d[thread_id+41*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[3]*C00*auxZ[3] + auxY[3]*C33*auxY[3] + auxX[3]*C33*auxX[3]);
							
						Matrix_K_Aux_d[thread_id+42*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[3]*C01*auxX[2] + auxX[3]*C33*auxZ[2]);
						Matrix_K_Aux_d[thread_id+43*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[3]*C01*auxY[2] + auxY[3]*C33*auxZ[2]);
						Matrix_K_Aux_d[thread_id+44*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[3]*C00*auxZ[2] + auxY[3]*C33*auxY[2] + auxX[3]*C33*auxX[2]);
							
						Matrix_K_Aux_d[thread_id+57*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[3]*C01*auxX[4] + auxX[3]*C33*auxZ[4]);
						Matrix_K_Aux_d[thread_id+58*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[3]*C01*auxY[4] + auxY[3]*C33*auxZ[4]);
						Matrix_K_Aux_d[thread_id+59*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[3]*C00*auxZ[4] + auxY[3]*C33*auxY[4] + auxX[3]*C33*auxX[4]);
							
						Matrix_K_Aux_d[thread_id+60*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[3]*C01*auxX[5] + auxX[3]*C33*auxZ[5]);
						Matrix_K_Aux_d[thread_id+61*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[3]*C01*auxY[5] + auxY[3]*C33*auxZ[5]);
						Matrix_K_Aux_d[thread_id+62*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[3]*C00*auxZ[5] + auxY[3]*C33*auxY[5] + auxX[3]*C33*auxX[5]);
							
						Matrix_K_Aux_d[thread_id+66*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[3]*C01*auxX[7] + auxX[3]*C33*auxZ[7]);
						Matrix_K_Aux_d[thread_id+67*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[3]*C01*auxY[7] + auxY[3]*C33*auxZ[7]);
						Matrix_K_Aux_d[thread_id+68*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[3]*C00*auxZ[7] + auxY[3]*C33*auxY[7] + auxX[3]*C33*auxX[7]);
							
						Matrix_K_Aux_d[thread_id+69*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[3]*C01*auxX[6] + auxX[3]*C33*auxZ[6]);
						Matrix_K_Aux_d[thread_id+70*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[3]*C01*auxY[6] + auxY[3]*C33*auxZ[6]);
						Matrix_K_Aux_d[thread_id+71*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[3]*C00*auxZ[6] + auxY[3]*C33*auxY[6] + auxX[3]*C33*auxX[6]);
							
					}
					
				}                                                        // =================== Loop kg ===================
											
			}                                                            // =================== Loop jg ===================

		}                                                                // =================== Loop ig ===================
		
	}
	
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============

__global__ void AssemblyStiffness3DQ7(int numno, int RowSize, double *CoordQ7_d, double *MatPropQ7_d, double *Matrix_K_Aux_d) 
{
	double r, s, t;  // 3
	double XEL0, YEL0, ZEL0, XEL1, YEL1, ZEL1, XEL2, YEL2, ZEL2, XEL3, YEL3, ZEL3;  // 12 
	double XEL4, YEL4, ZEL4, XEL5, YEL5, ZEL5, XEL6, YEL6, ZEL6, XEL7, YEL7, ZEL7;  // 12 
	double xgaus[2], wgaus[2];  // 4
	double p, E, C00, C01, C33;  // 5
	double phi_r[8], phi_s[8], phi_t[8];  // 24
	double jac[3][3];  // 9
	double jacob;  // 1
	double gama[3][3];  // 9
	double auxX[8], auxY[8], auxZ[8];  // 24
	int ig, jg, kg;  // 3
	
	const int xIndex = blockIdx.x*blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y*blockDim.y + threadIdx.y;
	const int thread_id = (gridDim.x*blockDim.x)*yIndex + xIndex;
	
	const int NumThread = (gridDim.x*blockDim.x)*(gridDim.y*blockDim.y);
	const int NumThreadSpace = RowSize*NumThread;
	
	__syncthreads();
	
	if(thread_id < numno) {
	
		XEL0 = CoordQ7_d[thread_id+ 0*numno];
		YEL0 = CoordQ7_d[thread_id+ 1*numno];
		ZEL0 = CoordQ7_d[thread_id+ 2*numno];   	
		XEL1 = CoordQ7_d[thread_id+ 3*numno];
		YEL1 = CoordQ7_d[thread_id+ 4*numno];  
		ZEL1 = CoordQ7_d[thread_id+ 5*numno];  	
		XEL2 = CoordQ7_d[thread_id+ 6*numno];
		YEL2 = CoordQ7_d[thread_id+ 7*numno]; 
		ZEL2 = CoordQ7_d[thread_id+ 8*numno];  		
		XEL3 = CoordQ7_d[thread_id+ 9*numno];
		YEL3 = CoordQ7_d[thread_id+10*numno];
		ZEL3 = CoordQ7_d[thread_id+11*numno];
		
		XEL4 = CoordQ7_d[thread_id+12*numno];
		YEL4 = CoordQ7_d[thread_id+13*numno];
		ZEL4 = CoordQ7_d[thread_id+14*numno];   	
		XEL5 = CoordQ7_d[thread_id+15*numno];
		YEL5 = CoordQ7_d[thread_id+16*numno];  
		ZEL5 = CoordQ7_d[thread_id+17*numno];  	
		XEL6 = CoordQ7_d[thread_id+18*numno];
		YEL6 = CoordQ7_d[thread_id+19*numno]; 
		ZEL6 = CoordQ7_d[thread_id+20*numno];  		
		XEL7 = CoordQ7_d[thread_id+21*numno];
		YEL7 = CoordQ7_d[thread_id+22*numno];
		ZEL7 = CoordQ7_d[thread_id+23*numno];
		
		__syncthreads();

		xgaus[0]=-0.577350269189626;
		xgaus[1]= 0.577350269189626;
		wgaus[0]= 1.;
		wgaus[1]= 1.;
							
		E = MatPropQ7_d[thread_id+ 0*numno];
		p = MatPropQ7_d[thread_id+ 1*numno];
					
		C00=(E*(1-p))/((1+p)*(1-2*p));  
		C01=(E*(1-p))/((1+p)*(1-2*p))*(p/(1-p)); 
		C33=(E*(1-p))/((1+p)*(1-2*p))*((1-2*p)/(2*(1-p)));
			
		for(ig=0; ig<2; ig++) {                                          // =================== Loop ig ===================
			t=xgaus[ig];
					
			for(jg=0; jg<2; jg++) {                                      // =================== Loop jg ===================
				s=xgaus[jg]; 
				
				for(kg=0; kg<2; kg++) {                                  // =================== Loop kg ===================
				r=xgaus[kg]; 
				
					phi_r[0] = -0.125*(1.0-s)*(1.0-t);
					phi_r[1] =  0.125*(1.0-s)*(1.0-t);
					phi_r[2] =  0.125*(1.0+s)*(1.0-t);
					phi_r[3] = -0.125*(1.0+s)*(1.0-t);
					phi_r[4] = -0.125*(1.0-s)*(1.0+t);
					phi_r[5] =  0.125*(1.0-s)*(1.0+t);
					phi_r[6] =  0.125*(1.0+s)*(1.0+t);
					phi_r[7] = -0.125*(1.0+s)*(1.0+t);
	 
					phi_s[0] = -0.125*(1.0-r)*(1.0-t);
					phi_s[1] = -0.125*(1.0+r)*(1.0-t);
					phi_s[2] =  0.125*(1.0+r)*(1.0-t);
					phi_s[3] =  0.125*(1.0-r)*(1.0-t);
					phi_s[4] = -0.125*(1.0-r)*(1.0+t);
					phi_s[5] = -0.125*(1.0+r)*(1.0+t);
					phi_s[6] =  0.125*(1.0+r)*(1.0+t);
					phi_s[7] =  0.125*(1.0-r)*(1.0+t);
		 
					phi_t[0] = -0.125*(1.0-r)*(1.0-s);
					phi_t[1] = -0.125*(1.0+r)*(1.0-s);
					phi_t[2] = -0.125*(1.0+r)*(1.0+s);
					phi_t[3] = -0.125*(1.0-r)*(1.0+s);
					phi_t[4] =  0.125*(1.0-r)*(1.0-s);
					phi_t[5] =  0.125*(1.0+r)*(1.0-s);
					phi_t[6] =  0.125*(1.0+r)*(1.0+s);
					phi_t[7] =  0.125*(1.0-r)*(1.0+s);
						
					jac[0][0]=phi_r[0]*XEL0+phi_r[1]*XEL1+phi_r[2]*XEL2+phi_r[3]*XEL3+phi_r[4]*XEL4+phi_r[5]*XEL5+phi_r[6]*XEL6+phi_r[7]*XEL7;
					jac[0][1]=phi_r[0]*YEL0+phi_r[1]*YEL1+phi_r[2]*YEL2+phi_r[3]*YEL3+phi_r[4]*YEL4+phi_r[5]*YEL5+phi_r[6]*YEL6+phi_r[7]*YEL7;
					jac[0][2]=phi_r[0]*ZEL0+phi_r[1]*ZEL1+phi_r[2]*ZEL2+phi_r[3]*ZEL3+phi_r[4]*ZEL4+phi_r[5]*ZEL5+phi_r[6]*ZEL6+phi_r[7]*ZEL7;

					jac[1][0]=phi_s[0]*XEL0+phi_s[1]*XEL1+phi_s[2]*XEL2+phi_s[3]*XEL3+phi_s[4]*XEL4+phi_s[5]*XEL5+phi_s[6]*XEL6+phi_s[7]*XEL7;
					jac[1][1]=phi_s[0]*YEL0+phi_s[1]*YEL1+phi_s[2]*YEL2+phi_s[3]*YEL3+phi_s[4]*YEL4+phi_s[5]*YEL5+phi_s[6]*YEL6+phi_s[7]*YEL7;
					jac[1][2]=phi_s[0]*ZEL0+phi_s[1]*ZEL1+phi_s[2]*ZEL2+phi_s[3]*ZEL3+phi_s[4]*ZEL4+phi_s[5]*ZEL5+phi_s[6]*ZEL6+phi_s[7]*ZEL7;
					
					jac[2][0]=phi_t[0]*XEL0+phi_t[1]*XEL1+phi_t[2]*XEL2+phi_t[3]*XEL3+phi_t[4]*XEL4+phi_t[5]*XEL5+phi_t[6]*XEL6+phi_t[7]*XEL7;
					jac[2][1]=phi_t[0]*YEL0+phi_t[1]*YEL1+phi_t[2]*YEL2+phi_t[3]*YEL3+phi_t[4]*YEL4+phi_t[5]*YEL5+phi_t[6]*YEL6+phi_t[7]*YEL7;
					jac[2][2]=phi_t[0]*ZEL0+phi_t[1]*ZEL1+phi_t[2]*ZEL2+phi_t[3]*ZEL3+phi_t[4]*ZEL4+phi_t[5]*ZEL5+phi_t[6]*ZEL6+phi_t[7]*ZEL7;
							
					jacob = jac[0][0]*(jac[1][1]*jac[2][2]-jac[1][2]*jac[2][1])-jac[0][1]*(jac[1][0]*jac[2][2]-jac[1][2]*jac[2][0])+
							jac[0][2]*(jac[1][0]*jac[2][1]-jac[1][1]*jac[2][0]);
					
					if(jacob > 0.) {
					
						gama[0][0] = (jac[1][1]*jac[2][2] - jac[1][2]*jac[2][1]) /jacob;
						gama[0][1] =-(jac[0][1]*jac[2][2] - jac[0][2]*jac[2][1]) /jacob;
						gama[0][2] = (jac[0][1]*jac[1][2] - jac[0][2]*jac[1][1]) /jacob;
						
						gama[1][0] =-(jac[1][0]*jac[2][2] - jac[1][2]*jac[2][0]) /jacob;
						gama[1][1] = (jac[0][0]*jac[2][2] - jac[0][2]*jac[2][0]) /jacob;
						gama[1][2] =-(jac[0][0]*jac[1][2] - jac[0][2]*jac[1][0]) /jacob;
						
						gama[2][0] = (jac[1][0]*jac[2][1] - jac[1][1]*jac[2][0]) /jacob;
						gama[2][1] =-(jac[0][0]*jac[2][1] - jac[0][1]*jac[2][0]) /jacob;
						gama[2][2] = (jac[0][0]*jac[1][1] - jac[0][1]*jac[1][0]) /jacob;
										
						auxX[0] = gama[0][0]*phi_r[0]+gama[0][1]*phi_s[0]+gama[0][2]*phi_t[0];
						auxY[0] = gama[1][0]*phi_r[0]+gama[1][1]*phi_s[0]+gama[1][2]*phi_t[0];
						auxZ[0] = gama[2][0]*phi_r[0]+gama[2][1]*phi_s[0]+gama[2][2]*phi_t[0];
							
						auxX[1] = gama[0][0]*phi_r[1]+gama[0][1]*phi_s[1]+gama[0][2]*phi_t[1];
						auxY[1] = gama[1][0]*phi_r[1]+gama[1][1]*phi_s[1]+gama[1][2]*phi_t[1];
						auxZ[1] = gama[2][0]*phi_r[1]+gama[2][1]*phi_s[1]+gama[2][2]*phi_t[1];
						
						auxX[2] = gama[0][0]*phi_r[2]+gama[0][1]*phi_s[2]+gama[0][2]*phi_t[2];
						auxY[2] = gama[1][0]*phi_r[2]+gama[1][1]*phi_s[2]+gama[1][2]*phi_t[2];
						auxZ[2] = gama[2][0]*phi_r[2]+gama[2][1]*phi_s[2]+gama[2][2]*phi_t[2];
						
						auxX[3] = gama[0][0]*phi_r[3]+gama[0][1]*phi_s[3]+gama[0][2]*phi_t[3];
						auxY[3] = gama[1][0]*phi_r[3]+gama[1][1]*phi_s[3]+gama[1][2]*phi_t[3];
						auxZ[3] = gama[2][0]*phi_r[3]+gama[2][1]*phi_s[3]+gama[2][2]*phi_t[3];
						
						auxX[4] = gama[0][0]*phi_r[4]+gama[0][1]*phi_s[4]+gama[0][2]*phi_t[4];
						auxY[4] = gama[1][0]*phi_r[4]+gama[1][1]*phi_s[4]+gama[1][2]*phi_t[4];
						auxZ[4] = gama[2][0]*phi_r[4]+gama[2][1]*phi_s[4]+gama[2][2]*phi_t[4];
							
						auxX[5] = gama[0][0]*phi_r[5]+gama[0][1]*phi_s[5]+gama[0][2]*phi_t[5];
						auxY[5] = gama[1][0]*phi_r[5]+gama[1][1]*phi_s[5]+gama[1][2]*phi_t[5];
						auxZ[5] = gama[2][0]*phi_r[5]+gama[2][1]*phi_s[5]+gama[2][2]*phi_t[5];
						
						auxX[6] = gama[0][0]*phi_r[6]+gama[0][1]*phi_s[6]+gama[0][2]*phi_t[6];
						auxY[6] = gama[1][0]*phi_r[6]+gama[1][1]*phi_s[6]+gama[1][2]*phi_t[6];
						auxZ[6] = gama[2][0]*phi_r[6]+gama[2][1]*phi_s[6]+gama[2][2]*phi_t[6];
						
						auxX[7] = gama[0][0]*phi_r[7]+gama[0][1]*phi_s[7]+gama[0][2]*phi_t[7];
						auxY[7] = gama[1][0]*phi_r[7]+gama[1][1]*phi_s[7]+gama[1][2]*phi_t[7];
						auxZ[7] = gama[2][0]*phi_r[7]+gama[2][1]*phi_s[7]+gama[2][2]*phi_t[7];
						
						__syncthreads();  // Line 0:
							
						Matrix_K_Aux_d[thread_id+36*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[1]*C00*auxX[0] + auxY[1]*C33*auxY[0] + auxZ[1]*C33*auxZ[0]);
						Matrix_K_Aux_d[thread_id+37*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[1]*C01*auxY[0] + auxY[1]*C33*auxX[0]);
						Matrix_K_Aux_d[thread_id+38*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[1]*C01*auxZ[0] + auxZ[1]*C33*auxX[0]);
							
						Matrix_K_Aux_d[thread_id+39*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[1]*C00*auxX[1] + auxY[1]*C33*auxY[1] + auxZ[1]*C33*auxZ[1]);
						Matrix_K_Aux_d[thread_id+40*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[1]*C01*auxY[1] + auxY[1]*C33*auxX[1]);
						Matrix_K_Aux_d[thread_id+41*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[1]*C01*auxZ[1] + auxZ[1]*C33*auxX[1]);
							
						Matrix_K_Aux_d[thread_id+45*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[1]*C00*auxX[3] + auxY[1]*C33*auxY[3] + auxZ[1]*C33*auxZ[3]);
						Matrix_K_Aux_d[thread_id+46*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[1]*C01*auxY[3] + auxY[1]*C33*auxX[3]);
						Matrix_K_Aux_d[thread_id+47*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[1]*C01*auxZ[3] + auxZ[1]*C33*auxX[3]);
							
						Matrix_K_Aux_d[thread_id+48*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[1]*C00*auxX[2] + auxY[1]*C33*auxY[2] + auxZ[1]*C33*auxZ[2]);
						Matrix_K_Aux_d[thread_id+49*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[1]*C01*auxY[2] + auxY[1]*C33*auxX[2]);
						Matrix_K_Aux_d[thread_id+50*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[1]*C01*auxZ[2] + auxZ[1]*C33*auxX[2]);
							
						Matrix_K_Aux_d[thread_id+63*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[1]*C00*auxX[4] + auxY[1]*C33*auxY[4] + auxZ[1]*C33*auxZ[4]);
						Matrix_K_Aux_d[thread_id+64*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[1]*C01*auxY[4] + auxY[1]*C33*auxX[4]);
						Matrix_K_Aux_d[thread_id+65*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[1]*C01*auxZ[4] + auxZ[1]*C33*auxX[4]);
							
						Matrix_K_Aux_d[thread_id+66*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[1]*C00*auxX[5] + auxY[1]*C33*auxY[5] + auxZ[1]*C33*auxZ[5]);
						Matrix_K_Aux_d[thread_id+67*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[1]*C01*auxY[5] + auxY[1]*C33*auxX[5]);
						Matrix_K_Aux_d[thread_id+68*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[1]*C01*auxZ[5] + auxZ[1]*C33*auxX[5]);
							
						Matrix_K_Aux_d[thread_id+72*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[1]*C00*auxX[7] + auxY[1]*C33*auxY[7] + auxZ[1]*C33*auxZ[7]);
						Matrix_K_Aux_d[thread_id+73*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[1]*C01*auxY[7] + auxY[1]*C33*auxX[7]);
						Matrix_K_Aux_d[thread_id+74*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[1]*C01*auxZ[7] + auxZ[1]*C33*auxX[7]);
							
						Matrix_K_Aux_d[thread_id+75*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[1]*C00*auxX[6] + auxY[1]*C33*auxY[6] + auxZ[1]*C33*auxZ[6]);
						Matrix_K_Aux_d[thread_id+76*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[1]*C01*auxY[6] + auxY[1]*C33*auxX[6]);
						Matrix_K_Aux_d[thread_id+77*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[1]*C01*auxZ[6] + auxZ[1]*C33*auxX[6]);

						__syncthreads();  // Line 1:
							
						Matrix_K_Aux_d[thread_id+36*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[1]*C01*auxX[0] + auxX[1]*C33*auxY[0]);
						Matrix_K_Aux_d[thread_id+37*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[1]*C00*auxY[0] + auxX[1]*C33*auxX[0] + auxZ[1]*C33*auxZ[0]);
						Matrix_K_Aux_d[thread_id+38*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[1]*C01*auxZ[0] + auxZ[1]*C33*auxY[0]);
							
						Matrix_K_Aux_d[thread_id+39*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[1]*C01*auxX[1] + auxX[1]*C33*auxY[1]);
						Matrix_K_Aux_d[thread_id+40*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[1]*C00*auxY[1] + auxX[1]*C33*auxX[1] + auxZ[1]*C33*auxZ[1]);
						Matrix_K_Aux_d[thread_id+41*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[1]*C01*auxZ[1] + auxZ[1]*C33*auxY[1]);
							
						Matrix_K_Aux_d[thread_id+45*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[1]*C01*auxX[3] + auxX[1]*C33*auxY[3]);
						Matrix_K_Aux_d[thread_id+46*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[1]*C00*auxY[3] + auxX[1]*C33*auxX[3] + auxZ[1]*C33*auxZ[3]);
						Matrix_K_Aux_d[thread_id+47*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[1]*C01*auxZ[3] + auxZ[1]*C33*auxY[3]);
							
						Matrix_K_Aux_d[thread_id+48*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[1]*C01*auxX[2] + auxX[1]*C33*auxY[2]);
						Matrix_K_Aux_d[thread_id+49*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[1]*C00*auxY[2] + auxX[1]*C33*auxX[2] + auxZ[1]*C33*auxZ[2]);
						Matrix_K_Aux_d[thread_id+50*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[1]*C01*auxZ[2] + auxZ[1]*C33*auxY[2]);
							
						Matrix_K_Aux_d[thread_id+63*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[1]*C01*auxX[4] + auxX[1]*C33*auxY[4]);
						Matrix_K_Aux_d[thread_id+64*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[1]*C00*auxY[4] + auxX[1]*C33*auxX[4] + auxZ[1]*C33*auxZ[4]);
						Matrix_K_Aux_d[thread_id+65*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[1]*C01*auxZ[4] + auxZ[1]*C33*auxY[4]);
							
						Matrix_K_Aux_d[thread_id+66*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[1]*C01*auxX[5] + auxX[1]*C33*auxY[5]);
						Matrix_K_Aux_d[thread_id+67*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[1]*C00*auxY[5] + auxX[1]*C33*auxX[5] + auxZ[1]*C33*auxZ[5]);
						Matrix_K_Aux_d[thread_id+68*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[1]*C01*auxZ[5] + auxZ[1]*C33*auxY[5]);
							
						Matrix_K_Aux_d[thread_id+72*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[1]*C01*auxX[7] + auxX[1]*C33*auxY[7]);
						Matrix_K_Aux_d[thread_id+73*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[1]*C00*auxY[7] + auxX[1]*C33*auxX[7] + auxZ[1]*C33*auxZ[7]);
						Matrix_K_Aux_d[thread_id+74*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[1]*C01*auxZ[7] + auxZ[1]*C33*auxY[7]);
						
						Matrix_K_Aux_d[thread_id+75*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[1]*C01*auxX[6] + auxX[1]*C33*auxY[6]);
						Matrix_K_Aux_d[thread_id+76*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[1]*C00*auxY[6] + auxX[1]*C33*auxX[6] + auxZ[1]*C33*auxZ[6]);
						Matrix_K_Aux_d[thread_id+77*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[1]*C01*auxZ[6] + auxZ[1]*C33*auxY[6]);
				
						__syncthreads();  // Line 2:
						
						Matrix_K_Aux_d[thread_id+36*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[1]*C01*auxX[0] + auxX[1]*C33*auxZ[0]);
						Matrix_K_Aux_d[thread_id+37*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[1]*C01*auxY[0] + auxY[1]*C33*auxZ[0]);
						Matrix_K_Aux_d[thread_id+38*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[1]*C00*auxZ[0] + auxY[1]*C33*auxY[0] + auxX[1]*C33*auxX[0]);
							
						Matrix_K_Aux_d[thread_id+39*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[1]*C01*auxX[1] + auxX[1]*C33*auxZ[1]);
						Matrix_K_Aux_d[thread_id+40*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[1]*C01*auxY[1] + auxY[1]*C33*auxZ[1]);
						Matrix_K_Aux_d[thread_id+41*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[1]*C00*auxZ[1] + auxY[1]*C33*auxY[1] + auxX[1]*C33*auxX[1]);
							
						Matrix_K_Aux_d[thread_id+45*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[1]*C01*auxX[3] + auxX[1]*C33*auxZ[3]);
						Matrix_K_Aux_d[thread_id+46*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[1]*C01*auxY[3] + auxY[1]*C33*auxZ[3]);
						Matrix_K_Aux_d[thread_id+47*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[1]*C00*auxZ[3] + auxY[1]*C33*auxY[3] + auxX[1]*C33*auxX[3]);
							
						Matrix_K_Aux_d[thread_id+48*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[1]*C01*auxX[2] + auxX[1]*C33*auxZ[2]);
						Matrix_K_Aux_d[thread_id+49*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[1]*C01*auxY[2] + auxY[1]*C33*auxZ[2]);
						Matrix_K_Aux_d[thread_id+50*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[1]*C00*auxZ[2] + auxY[1]*C33*auxY[2] + auxX[1]*C33*auxX[2]);
							
						Matrix_K_Aux_d[thread_id+63*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[1]*C01*auxX[4] + auxX[1]*C33*auxZ[4]);
						Matrix_K_Aux_d[thread_id+64*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[1]*C01*auxY[4] + auxY[1]*C33*auxZ[4]);
						Matrix_K_Aux_d[thread_id+65*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[1]*C00*auxZ[4] + auxY[1]*C33*auxY[4] + auxX[1]*C33*auxX[4]);
							
						Matrix_K_Aux_d[thread_id+66*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[1]*C01*auxX[5] + auxX[1]*C33*auxZ[5]);
						Matrix_K_Aux_d[thread_id+67*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[1]*C01*auxY[5] + auxY[1]*C33*auxZ[5]);
						Matrix_K_Aux_d[thread_id+68*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[1]*C00*auxZ[5] + auxY[1]*C33*auxY[5] + auxX[1]*C33*auxX[5]);
							
						Matrix_K_Aux_d[thread_id+72*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[1]*C01*auxX[7] + auxX[1]*C33*auxZ[7]);
						Matrix_K_Aux_d[thread_id+73*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[1]*C01*auxY[7] + auxY[1]*C33*auxZ[7]);
						Matrix_K_Aux_d[thread_id+74*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[1]*C00*auxZ[7] + auxY[1]*C33*auxY[7] + auxX[1]*C33*auxX[7]);
							
						Matrix_K_Aux_d[thread_id+75*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[1]*C01*auxX[6] + auxX[1]*C33*auxZ[6]);
						Matrix_K_Aux_d[thread_id+76*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[1]*C01*auxY[6] + auxY[1]*C33*auxZ[6]);
						Matrix_K_Aux_d[thread_id+77*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[1]*C00*auxZ[6] + auxY[1]*C33*auxY[6] + auxX[1]*C33*auxX[6]);
						
					}
					
				}                                                        // =================== Loop kg ===================
											
			}                                                            // =================== Loop jg ===================

		}                                                                // =================== Loop ig ===================
		
	}
	
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============

__global__ void AssemblyStiffness3DQ8(int numno, int RowSize, double *CoordQ8_d, double *MatPropQ8_d, double *Matrix_K_Aux_d) 
{
	double r, s, t;  // 3
	double XEL0, YEL0, ZEL0, XEL1, YEL1, ZEL1, XEL2, YEL2, ZEL2, XEL3, YEL3, ZEL3;  // 12 
	double XEL4, YEL4, ZEL4, XEL5, YEL5, ZEL5, XEL6, YEL6, ZEL6, XEL7, YEL7, ZEL7;  // 12 
	double xgaus[2], wgaus[2];  // 4
	double p, E, C00, C01, C33;  // 5
	double phi_r[8], phi_s[8], phi_t[8];  // 24
	double jac[3][3];  // 9
	double jacob;  // 1
	double gama[3][3];  // 9
	double auxX[8], auxY[8], auxZ[8];  // 24
	int ig, jg, kg;  // 3
	
	const int xIndex = blockIdx.x*blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y*blockDim.y + threadIdx.y;
	const int thread_id = (gridDim.x*blockDim.x)*yIndex + xIndex;
	
	const int NumThread = (gridDim.x*blockDim.x)*(gridDim.y*blockDim.y);
	const int NumThreadSpace = RowSize*NumThread;
	
	__syncthreads();
	
	if(thread_id < numno) {
	
		XEL0 = CoordQ8_d[thread_id+ 0*numno];
		YEL0 = CoordQ8_d[thread_id+ 1*numno];
		ZEL0 = CoordQ8_d[thread_id+ 2*numno];   	
		XEL1 = CoordQ8_d[thread_id+ 3*numno];
		YEL1 = CoordQ8_d[thread_id+ 4*numno];  
		ZEL1 = CoordQ8_d[thread_id+ 5*numno];  	
		XEL2 = CoordQ8_d[thread_id+ 6*numno];
		YEL2 = CoordQ8_d[thread_id+ 7*numno]; 
		ZEL2 = CoordQ8_d[thread_id+ 8*numno];  		
		XEL3 = CoordQ8_d[thread_id+ 9*numno];
		YEL3 = CoordQ8_d[thread_id+10*numno];
		ZEL3 = CoordQ8_d[thread_id+11*numno];
		
		XEL4 = CoordQ8_d[thread_id+12*numno];
		YEL4 = CoordQ8_d[thread_id+13*numno];
		ZEL4 = CoordQ8_d[thread_id+14*numno];   	
		XEL5 = CoordQ8_d[thread_id+15*numno];
		YEL5 = CoordQ8_d[thread_id+16*numno];  
		ZEL5 = CoordQ8_d[thread_id+17*numno];  	
		XEL6 = CoordQ8_d[thread_id+18*numno];
		YEL6 = CoordQ8_d[thread_id+19*numno]; 
		ZEL6 = CoordQ8_d[thread_id+20*numno];  		
		XEL7 = CoordQ8_d[thread_id+21*numno];
		YEL7 = CoordQ8_d[thread_id+22*numno];
		ZEL7 = CoordQ8_d[thread_id+23*numno];
		
		__syncthreads();

		xgaus[0]=-0.577350269189626;
		xgaus[1]= 0.577350269189626;
		wgaus[0]= 1.;
		wgaus[1]= 1.;
							
		E = MatPropQ8_d[thread_id+ 0*numno];
		p = MatPropQ8_d[thread_id+ 1*numno];
					
		C00=(E*(1-p))/((1+p)*(1-2*p));  
		C01=(E*(1-p))/((1+p)*(1-2*p))*(p/(1-p)); 
		C33=(E*(1-p))/((1+p)*(1-2*p))*((1-2*p)/(2*(1-p)));
			
		for(ig=0; ig<2; ig++) {                                          // =================== Loop ig ===================
			t=xgaus[ig];
					
			for(jg=0; jg<2; jg++) {                                      // =================== Loop jg ===================
				s=xgaus[jg]; 
				
				for(kg=0; kg<2; kg++) {                                  // =================== Loop kg ===================
				r=xgaus[kg]; 
				
					phi_r[0] = -0.125*(1.0-s)*(1.0-t);
					phi_r[1] =  0.125*(1.0-s)*(1.0-t);
					phi_r[2] =  0.125*(1.0+s)*(1.0-t);
					phi_r[3] = -0.125*(1.0+s)*(1.0-t);
					phi_r[4] = -0.125*(1.0-s)*(1.0+t);
					phi_r[5] =  0.125*(1.0-s)*(1.0+t);
					phi_r[6] =  0.125*(1.0+s)*(1.0+t);
					phi_r[7] = -0.125*(1.0+s)*(1.0+t);
	 
					phi_s[0] = -0.125*(1.0-r)*(1.0-t);
					phi_s[1] = -0.125*(1.0+r)*(1.0-t);
					phi_s[2] =  0.125*(1.0+r)*(1.0-t);
					phi_s[3] =  0.125*(1.0-r)*(1.0-t);
					phi_s[4] = -0.125*(1.0-r)*(1.0+t);
					phi_s[5] = -0.125*(1.0+r)*(1.0+t);
					phi_s[6] =  0.125*(1.0+r)*(1.0+t);
					phi_s[7] =  0.125*(1.0-r)*(1.0+t);
		 
					phi_t[0] = -0.125*(1.0-r)*(1.0-s);
					phi_t[1] = -0.125*(1.0+r)*(1.0-s);
					phi_t[2] = -0.125*(1.0+r)*(1.0+s);
					phi_t[3] = -0.125*(1.0-r)*(1.0+s);
					phi_t[4] =  0.125*(1.0-r)*(1.0-s);
					phi_t[5] =  0.125*(1.0+r)*(1.0-s);
					phi_t[6] =  0.125*(1.0+r)*(1.0+s);
					phi_t[7] =  0.125*(1.0-r)*(1.0+s);
						
					jac[0][0]=phi_r[0]*XEL0+phi_r[1]*XEL1+phi_r[2]*XEL2+phi_r[3]*XEL3+phi_r[4]*XEL4+phi_r[5]*XEL5+phi_r[6]*XEL6+phi_r[7]*XEL7;
					jac[0][1]=phi_r[0]*YEL0+phi_r[1]*YEL1+phi_r[2]*YEL2+phi_r[3]*YEL3+phi_r[4]*YEL4+phi_r[5]*YEL5+phi_r[6]*YEL6+phi_r[7]*YEL7;
					jac[0][2]=phi_r[0]*ZEL0+phi_r[1]*ZEL1+phi_r[2]*ZEL2+phi_r[3]*ZEL3+phi_r[4]*ZEL4+phi_r[5]*ZEL5+phi_r[6]*ZEL6+phi_r[7]*ZEL7;

					jac[1][0]=phi_s[0]*XEL0+phi_s[1]*XEL1+phi_s[2]*XEL2+phi_s[3]*XEL3+phi_s[4]*XEL4+phi_s[5]*XEL5+phi_s[6]*XEL6+phi_s[7]*XEL7;
					jac[1][1]=phi_s[0]*YEL0+phi_s[1]*YEL1+phi_s[2]*YEL2+phi_s[3]*YEL3+phi_s[4]*YEL4+phi_s[5]*YEL5+phi_s[6]*YEL6+phi_s[7]*YEL7;
					jac[1][2]=phi_s[0]*ZEL0+phi_s[1]*ZEL1+phi_s[2]*ZEL2+phi_s[3]*ZEL3+phi_s[4]*ZEL4+phi_s[5]*ZEL5+phi_s[6]*ZEL6+phi_s[7]*ZEL7;
					
					jac[2][0]=phi_t[0]*XEL0+phi_t[1]*XEL1+phi_t[2]*XEL2+phi_t[3]*XEL3+phi_t[4]*XEL4+phi_t[5]*XEL5+phi_t[6]*XEL6+phi_t[7]*XEL7;
					jac[2][1]=phi_t[0]*YEL0+phi_t[1]*YEL1+phi_t[2]*YEL2+phi_t[3]*YEL3+phi_t[4]*YEL4+phi_t[5]*YEL5+phi_t[6]*YEL6+phi_t[7]*YEL7;
					jac[2][2]=phi_t[0]*ZEL0+phi_t[1]*ZEL1+phi_t[2]*ZEL2+phi_t[3]*ZEL3+phi_t[4]*ZEL4+phi_t[5]*ZEL5+phi_t[6]*ZEL6+phi_t[7]*ZEL7;
							
					jacob = jac[0][0]*(jac[1][1]*jac[2][2]-jac[1][2]*jac[2][1])-jac[0][1]*(jac[1][0]*jac[2][2]-jac[1][2]*jac[2][0])+
							jac[0][2]*(jac[1][0]*jac[2][1]-jac[1][1]*jac[2][0]);
					
					if(jacob > 0.) {
					
						gama[0][0] = (jac[1][1]*jac[2][2] - jac[1][2]*jac[2][1]) /jacob;
						gama[0][1] =-(jac[0][1]*jac[2][2] - jac[0][2]*jac[2][1]) /jacob;
						gama[0][2] = (jac[0][1]*jac[1][2] - jac[0][2]*jac[1][1]) /jacob;
						
						gama[1][0] =-(jac[1][0]*jac[2][2] - jac[1][2]*jac[2][0]) /jacob;
						gama[1][1] = (jac[0][0]*jac[2][2] - jac[0][2]*jac[2][0]) /jacob;
						gama[1][2] =-(jac[0][0]*jac[1][2] - jac[0][2]*jac[1][0]) /jacob;
						
						gama[2][0] = (jac[1][0]*jac[2][1] - jac[1][1]*jac[2][0]) /jacob;
						gama[2][1] =-(jac[0][0]*jac[2][1] - jac[0][1]*jac[2][0]) /jacob;
						gama[2][2] = (jac[0][0]*jac[1][1] - jac[0][1]*jac[1][0]) /jacob;
										
						auxX[0] = gama[0][0]*phi_r[0]+gama[0][1]*phi_s[0]+gama[0][2]*phi_t[0];
						auxY[0] = gama[1][0]*phi_r[0]+gama[1][1]*phi_s[0]+gama[1][2]*phi_t[0];
						auxZ[0] = gama[2][0]*phi_r[0]+gama[2][1]*phi_s[0]+gama[2][2]*phi_t[0];
							
						auxX[1] = gama[0][0]*phi_r[1]+gama[0][1]*phi_s[1]+gama[0][2]*phi_t[1];
						auxY[1] = gama[1][0]*phi_r[1]+gama[1][1]*phi_s[1]+gama[1][2]*phi_t[1];
						auxZ[1] = gama[2][0]*phi_r[1]+gama[2][1]*phi_s[1]+gama[2][2]*phi_t[1];
						
						auxX[2] = gama[0][0]*phi_r[2]+gama[0][1]*phi_s[2]+gama[0][2]*phi_t[2];
						auxY[2] = gama[1][0]*phi_r[2]+gama[1][1]*phi_s[2]+gama[1][2]*phi_t[2];
						auxZ[2] = gama[2][0]*phi_r[2]+gama[2][1]*phi_s[2]+gama[2][2]*phi_t[2];
						
						auxX[3] = gama[0][0]*phi_r[3]+gama[0][1]*phi_s[3]+gama[0][2]*phi_t[3];
						auxY[3] = gama[1][0]*phi_r[3]+gama[1][1]*phi_s[3]+gama[1][2]*phi_t[3];
						auxZ[3] = gama[2][0]*phi_r[3]+gama[2][1]*phi_s[3]+gama[2][2]*phi_t[3];
						
						auxX[4] = gama[0][0]*phi_r[4]+gama[0][1]*phi_s[4]+gama[0][2]*phi_t[4];
						auxY[4] = gama[1][0]*phi_r[4]+gama[1][1]*phi_s[4]+gama[1][2]*phi_t[4];
						auxZ[4] = gama[2][0]*phi_r[4]+gama[2][1]*phi_s[4]+gama[2][2]*phi_t[4];
							
						auxX[5] = gama[0][0]*phi_r[5]+gama[0][1]*phi_s[5]+gama[0][2]*phi_t[5];
						auxY[5] = gama[1][0]*phi_r[5]+gama[1][1]*phi_s[5]+gama[1][2]*phi_t[5];
						auxZ[5] = gama[2][0]*phi_r[5]+gama[2][1]*phi_s[5]+gama[2][2]*phi_t[5];
						
						auxX[6] = gama[0][0]*phi_r[6]+gama[0][1]*phi_s[6]+gama[0][2]*phi_t[6];
						auxY[6] = gama[1][0]*phi_r[6]+gama[1][1]*phi_s[6]+gama[1][2]*phi_t[6];
						auxZ[6] = gama[2][0]*phi_r[6]+gama[2][1]*phi_s[6]+gama[2][2]*phi_t[6];
						
						auxX[7] = gama[0][0]*phi_r[7]+gama[0][1]*phi_s[7]+gama[0][2]*phi_t[7];
						auxY[7] = gama[1][0]*phi_r[7]+gama[1][1]*phi_s[7]+gama[1][2]*phi_t[7];
						auxZ[7] = gama[2][0]*phi_r[7]+gama[2][1]*phi_s[7]+gama[2][2]*phi_t[7];
		
						__syncthreads();  // Line 0:
							
						Matrix_K_Aux_d[thread_id+39*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[0]*C00*auxX[0] + auxY[0]*C33*auxY[0] + auxZ[0]*C33*auxZ[0]);
						Matrix_K_Aux_d[thread_id+40*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[0]*C01*auxY[0] + auxY[0]*C33*auxX[0]);
						Matrix_K_Aux_d[thread_id+41*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[0]*C01*auxZ[0] + auxZ[0]*C33*auxX[0]);
							
						Matrix_K_Aux_d[thread_id+42*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[0]*C00*auxX[1] + auxY[0]*C33*auxY[1] + auxZ[0]*C33*auxZ[1]);
						Matrix_K_Aux_d[thread_id+43*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[0]*C01*auxY[1] + auxY[0]*C33*auxX[1]);
						Matrix_K_Aux_d[thread_id+44*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[0]*C01*auxZ[1] + auxZ[0]*C33*auxX[1]);
							
						Matrix_K_Aux_d[thread_id+48*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[0]*C00*auxX[3] + auxY[0]*C33*auxY[3] + auxZ[0]*C33*auxZ[3]);
						Matrix_K_Aux_d[thread_id+49*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[0]*C01*auxY[3] + auxY[0]*C33*auxX[3]);
						Matrix_K_Aux_d[thread_id+50*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[0]*C01*auxZ[3] + auxZ[0]*C33*auxX[3]);
							
						Matrix_K_Aux_d[thread_id+51*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[0]*C00*auxX[2] + auxY[0]*C33*auxY[2] + auxZ[0]*C33*auxZ[2]);
						Matrix_K_Aux_d[thread_id+52*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[0]*C01*auxY[2] + auxY[0]*C33*auxX[2]);
						Matrix_K_Aux_d[thread_id+53*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[0]*C01*auxZ[2] + auxZ[0]*C33*auxX[2]);
							
						Matrix_K_Aux_d[thread_id+66*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[0]*C00*auxX[4] + auxY[0]*C33*auxY[4] + auxZ[0]*C33*auxZ[4]);
						Matrix_K_Aux_d[thread_id+67*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[0]*C01*auxY[4] + auxY[0]*C33*auxX[4]);
						Matrix_K_Aux_d[thread_id+68*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[0]*C01*auxZ[4] + auxZ[0]*C33*auxX[4]);
							
						Matrix_K_Aux_d[thread_id+69*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[0]*C00*auxX[5] + auxY[0]*C33*auxY[5] + auxZ[0]*C33*auxZ[5]);
						Matrix_K_Aux_d[thread_id+70*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[0]*C01*auxY[5] + auxY[0]*C33*auxX[5]);
						Matrix_K_Aux_d[thread_id+71*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[0]*C01*auxZ[5] + auxZ[0]*C33*auxX[5]);
							
						Matrix_K_Aux_d[thread_id+75*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[0]*C00*auxX[7] + auxY[0]*C33*auxY[7] + auxZ[0]*C33*auxZ[7]);
						Matrix_K_Aux_d[thread_id+76*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[0]*C01*auxY[7] + auxY[0]*C33*auxX[7]);
						Matrix_K_Aux_d[thread_id+77*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[0]*C01*auxZ[7] + auxZ[0]*C33*auxX[7]);
							
						Matrix_K_Aux_d[thread_id+78*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[0]*C00*auxX[6] + auxY[0]*C33*auxY[6] + auxZ[0]*C33*auxZ[6]);
						Matrix_K_Aux_d[thread_id+79*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[0]*C01*auxY[6] + auxY[0]*C33*auxX[6]);
						Matrix_K_Aux_d[thread_id+80*NumThread] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxX[0]*C01*auxZ[6] + auxZ[0]*C33*auxX[6]);
							
						__syncthreads();  // Line 1:
						
						Matrix_K_Aux_d[thread_id+39*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[0]*C01*auxX[0] + auxX[0]*C33*auxY[0]);
						Matrix_K_Aux_d[thread_id+40*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[0]*C00*auxY[0] + auxX[0]*C33*auxX[0] + auxZ[0]*C33*auxZ[0]);
						Matrix_K_Aux_d[thread_id+41*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[0]*C01*auxZ[0] + auxZ[0]*C33*auxY[0]);
							
						Matrix_K_Aux_d[thread_id+42*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[0]*C01*auxX[1] + auxX[0]*C33*auxY[1]);
						Matrix_K_Aux_d[thread_id+43*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[0]*C00*auxY[1] + auxX[0]*C33*auxX[1] + auxZ[0]*C33*auxZ[1]);
						Matrix_K_Aux_d[thread_id+44*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[0]*C01*auxZ[1] + auxZ[0]*C33*auxY[1]);
							
						Matrix_K_Aux_d[thread_id+48*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[0]*C01*auxX[3] + auxX[0]*C33*auxY[3]);
						Matrix_K_Aux_d[thread_id+49*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[0]*C00*auxY[3] + auxX[0]*C33*auxX[3] + auxZ[0]*C33*auxZ[3]);
						Matrix_K_Aux_d[thread_id+50*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[0]*C01*auxZ[3] + auxZ[0]*C33*auxY[3]);
							
						Matrix_K_Aux_d[thread_id+51*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[0]*C01*auxX[2] + auxX[0]*C33*auxY[2]);
						Matrix_K_Aux_d[thread_id+52*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[0]*C00*auxY[2] + auxX[0]*C33*auxX[2] + auxZ[0]*C33*auxZ[2]);
						Matrix_K_Aux_d[thread_id+53*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[0]*C01*auxZ[2] + auxZ[0]*C33*auxY[2]);
							
						Matrix_K_Aux_d[thread_id+66*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[0]*C01*auxX[4] + auxX[0]*C33*auxY[4]);
						Matrix_K_Aux_d[thread_id+67*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[0]*C00*auxY[4] + auxX[0]*C33*auxX[4] + auxZ[0]*C33*auxZ[4]);
						Matrix_K_Aux_d[thread_id+68*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[0]*C01*auxZ[4] + auxZ[0]*C33*auxY[4]);
							
						Matrix_K_Aux_d[thread_id+69*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[0]*C01*auxX[5] + auxX[0]*C33*auxY[5]);
						Matrix_K_Aux_d[thread_id+70*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[0]*C00*auxY[5] + auxX[0]*C33*auxX[5] + auxZ[0]*C33*auxZ[5]);
						Matrix_K_Aux_d[thread_id+71*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[0]*C01*auxZ[5] + auxZ[0]*C33*auxY[5]);
						
						Matrix_K_Aux_d[thread_id+75*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[0]*C01*auxX[7] + auxX[0]*C33*auxY[7]);
						Matrix_K_Aux_d[thread_id+76*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[0]*C00*auxY[7] + auxX[0]*C33*auxX[7] + auxZ[0]*C33*auxZ[7]);
						Matrix_K_Aux_d[thread_id+77*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[0]*C01*auxZ[7] + auxZ[0]*C33*auxY[7]);
							
						Matrix_K_Aux_d[thread_id+78*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[0]*C01*auxX[6] + auxX[0]*C33*auxY[6]);
						Matrix_K_Aux_d[thread_id+79*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[0]*C00*auxY[6] + auxX[0]*C33*auxX[6] + auxZ[0]*C33*auxZ[6]);
						Matrix_K_Aux_d[thread_id+80*NumThread+NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxY[0]*C01*auxZ[6] + auxZ[0]*C33*auxY[6]);
						
						__syncthreads();  // Line 2:
						
						Matrix_K_Aux_d[thread_id+39*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[0]*C01*auxX[0] + auxX[0]*C33*auxZ[0]);
						Matrix_K_Aux_d[thread_id+40*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[0]*C01*auxY[0] + auxY[0]*C33*auxZ[0]);
						Matrix_K_Aux_d[thread_id+41*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[0]*C00*auxZ[0] + auxY[0]*C33*auxY[0] + auxX[0]*C33*auxX[0]);
							
						Matrix_K_Aux_d[thread_id+42*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[0]*C01*auxX[1] + auxX[0]*C33*auxZ[1]);
						Matrix_K_Aux_d[thread_id+43*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[0]*C01*auxY[1] + auxY[0]*C33*auxZ[1]);
						Matrix_K_Aux_d[thread_id+44*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[0]*C00*auxZ[1] + auxY[0]*C33*auxY[1] + auxX[0]*C33*auxX[1]);
							
						Matrix_K_Aux_d[thread_id+48*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[0]*C01*auxX[3] + auxX[0]*C33*auxZ[3]);
						Matrix_K_Aux_d[thread_id+49*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[0]*C01*auxY[3] + auxY[0]*C33*auxZ[3]);
						Matrix_K_Aux_d[thread_id+50*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[0]*C00*auxZ[3] + auxY[0]*C33*auxY[3] + auxX[0]*C33*auxX[3]);
							
						Matrix_K_Aux_d[thread_id+51*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[0]*C01*auxX[2] + auxX[0]*C33*auxZ[2]);
						Matrix_K_Aux_d[thread_id+52*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[0]*C01*auxY[2] + auxY[0]*C33*auxZ[2]);
						Matrix_K_Aux_d[thread_id+53*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[0]*C00*auxZ[2] + auxY[0]*C33*auxY[2] + auxX[0]*C33*auxX[2]);
							
						Matrix_K_Aux_d[thread_id+66*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[0]*C01*auxX[4] + auxX[0]*C33*auxZ[4]);
						Matrix_K_Aux_d[thread_id+67*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[0]*C01*auxY[4] + auxY[0]*C33*auxZ[4]);
						Matrix_K_Aux_d[thread_id+68*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[0]*C00*auxZ[4] + auxY[0]*C33*auxY[4] + auxX[0]*C33*auxX[4]);
							
						Matrix_K_Aux_d[thread_id+69*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[0]*C01*auxX[5] + auxX[0]*C33*auxZ[5]);
						Matrix_K_Aux_d[thread_id+70*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[0]*C01*auxY[5] + auxY[0]*C33*auxZ[5]);
						Matrix_K_Aux_d[thread_id+71*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[0]*C00*auxZ[5] + auxY[0]*C33*auxY[5] + auxX[0]*C33*auxX[5]);
							
						Matrix_K_Aux_d[thread_id+75*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[0]*C01*auxX[7] + auxX[0]*C33*auxZ[7]);
						Matrix_K_Aux_d[thread_id+76*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[0]*C01*auxY[7] + auxY[0]*C33*auxZ[7]);
						Matrix_K_Aux_d[thread_id+77*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[0]*C00*auxZ[7] + auxY[0]*C33*auxY[7] + auxX[0]*C33*auxX[7]);
							
						Matrix_K_Aux_d[thread_id+78*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[0]*C01*auxX[6] + auxX[0]*C33*auxZ[6]);
						Matrix_K_Aux_d[thread_id+79*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[0]*C01*auxY[6] + auxY[0]*C33*auxZ[6]);
						Matrix_K_Aux_d[thread_id+80*NumThread+2*NumThreadSpace] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg] * (auxZ[0]*C00*auxZ[6] + auxY[0]*C33*auxY[6] + auxX[0]*C33*auxX[6]);

					}
					
				}                                                        // =================== Loop kg ===================
											
			}                                                            // =================== Loop jg ===================

		}                                                                // =================== Loop ig ===================
		
	}

}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============

void AssemblyStiffnessGPUTetrahedron(int numdof, int numdofel, int numno, int maxThreadsPerBlock, int RowSize, int BlockDimX, int BlockDimY,
double *Matrix_K_h, double *Matrix_K_d, double *Matrix_K_Aux_d, int NumThread, double *abcParam_h, double *abcParam_d, int MaxNodeAround, int MaxElemPerEdge) 
{

	//cudaPrintfInit();
	int size;
	//unsigned int hTimer1, hTimer2;
	//double Time1, Time2;
	int NumTerms;
	
	using namespace std; 
	
	// numno = Number of nodes of the mesh
	// Compute Capability 2.x and 3.0 => Maximum number of threads per block = 1024 (maxThreadsPerBlock)
	
	dim3 numBlocksSM(BlockDimX, BlockDimY);	
	dim3 threadsPerBlockSM(32, 32); 
	
	size = NumThread*MaxNodeAround*MaxElemPerEdge*7*sizeof(double);  // 7 = as, bs, cs, at, bt, ct, and V
	
	cudaMemcpy(abcParam_d, abcParam_h, size, cudaMemcpyHostToDevice);    // To copy from host memory to device memory;
	
	// ==================================================================================================================================
	// Evaluating Stiffness Matrix (SM):
	
//	cutCreateTimer(&hTimer1);
	cudaThreadSynchronize();
//	cutResetTimer(hTimer1);
//	cutStartTimer(hTimer1);
			
	// Assembly on GPU the Stiffness Matrix

	AssemblyStiffnessTetrahedron <<<numBlocksSM, threadsPerBlockSM>>> (RowSize, Matrix_K_Aux_d, abcParam_d, MaxNodeAround, MaxElemPerEdge);
	
	cudaThreadSynchronize();
//	cutStopTimer(hTimer1);
//	Time1 = cutGetTimerValue(hTimer1);
		
	printf("\n");
	//printf("         Assembly Stiffness Matrix Time: %f ms \n", Time1);
	
	// ==================================================================================================================================
	// Transpose (TR) Stiffness Matrix:
	
	//cutCreateTimer(&hTimer2);
	cudaThreadSynchronize();
	//cutResetTimer(hTimer2);
	//cutStartTimer(hTimer2);
		
	dim3 numBlocksTR((3*RowSize)/32, NumThread/32);	// X-Direction 3*RowSize (3 dof * RowSize)
	dim3 threadsPerBlockTR(32, 32);
	
	// TRanspose on GPU the Stiffness Matrix

	TrasnsposeCoalesced <<<numBlocksTR, threadsPerBlockTR>>> (NumThread, Matrix_K_Aux_d, Matrix_K_d);
	
	cudaThreadSynchronize();
	//cutStopTimer(hTimer2);
	//Time2 = cutGetTimerValue(hTimer2);
	
	printf("\n");
	//printf("         Transpose Stiffness Matrix Time: %f ms \n", Time2);
		
	// ==================================================================================================================================
	
	NumTerms = numdof*NumThread*RowSize;  // Size of the stiffness matriz
	cudaMemcpy(Matrix_K_h, Matrix_K_d, sizeof(double)*NumTerms, cudaMemcpyDeviceToHost);
	
	//cudaPrintfDisplay(stdout, true);
	//cudaPrintfEnd();
	
	

}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============

__global__ void TrasnsposeCoalesced(int NumThread, double *Matrix_K_Aux_d, double *Matrix_K_d) 
{

	int xIndex, yIndex, index_in, index_out;

	// Shared Memory
	__shared__ double tile[32][32];
	
	// =================================================================
	// Copying from Matrix_K_d (GM) to tile (SM):

	xIndex = blockIdx.x*blockDim.x + threadIdx.x;
	yIndex = blockIdx.y*blockDim.y + threadIdx.y;
	index_in = (gridDim.y*blockDim.y)*xIndex + yIndex;
	
	tile[threadIdx.y][threadIdx.x] = Matrix_K_Aux_d[index_in];
	
	// =================================================================
	// Copying from tile (SM) to Matrix_K_d (GM):
	
	__syncthreads();
	
	xIndex = blockIdx.x*blockDim.x + threadIdx.x;
	yIndex = blockIdx.y*blockDim.y + threadIdx.y;
	index_out = (gridDim.x*blockDim.x)*yIndex + xIndex;
		
	Matrix_K_d[index_out] = tile[threadIdx.y][threadIdx.x];

}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============

__global__ void AssemblyStiffnessTetrahedron(int RowSize, double *Matrix_K_Aux_d, double *abcParam_d, int MaxNodeAround, int MaxElemPerEdge) 
{
	double as, bs, cs, at, bt, ct, V;
	double e11, e12, e13, e14, e15, e16, e21, e22, e23, e24, e25, e26, e31, e32, e33, e34, e35, e36;
	double e41, e42, e43, e44, e45, e46, e51, e52, e53, e54, e55, e56, e61, e62, e63, e64, e65, e66;
	double E, p, sho;
	
	int i, j;
	
	const int xIndex = blockIdx.x*blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y*blockDim.y + threadIdx.y;
	const int thread_id = (gridDim.x*blockDim.x)*yIndex + xIndex;
	
	const int NumThread = (gridDim.x*blockDim.x)*(gridDim.y*blockDim.y);
	const int NumThreadSpace = RowSize*NumThread;
	
	__syncthreads();
	
	E=3000e10;
	p=0.3;

	sho=(E*(1-p))/((1+p)*(1-2*p));

	e11 = sho;           e12 = sho*(p/(1-p)); e13 = sho*(p/(1-p)); e14 = 0.;                      e15 = 0.;                      e16 = 0.; 
	e21 = sho*(p/(1-p)); e22 = sho;           e23 = sho*(p/(1-p)); e24 = 0.;                      e25 = 0.;                      e26 = 0.; 
	e31 = sho*(p/(1-p)); e32 = sho*(p/(1-p)); e33 = sho;    	   e34 = 0.;                      e35 = 0.;                      e36 = 0.; 
	e41 = 0.;            e42 = 0.;            e43 = 0.;            e44 = sho*((1-2*p)/(2*(1-p))); e45 = 0.;                      e46 = 0.; 
	e51 = 0.;            e52 = 0.;            e53 = 0.;            e54 = 0.;                      e55 = sho*((1-2*p)/(2*(1-p))); e56 = 0.; 
	e61 = 0.;            e62 = 0.;            e63 = 0.;            e64 = 0.;                      e65 = 0.;                      e66 = sho*((1-2*p)/(2*(1-p)));
	
	for(i=0; i<MaxNodeAround; i++) {
	
		for(j=0; j<MaxElemPerEdge; j++) {
		
			as = bs = cs = at = bt = ct = V = 0.;
					
			as = abcParam_d[i*7*NumThread*MaxElemPerEdge + j*7*NumThread + 0*NumThread + thread_id];
			bs = abcParam_d[i*7*NumThread*MaxElemPerEdge + j*7*NumThread + 1*NumThread + thread_id];
			cs = abcParam_d[i*7*NumThread*MaxElemPerEdge + j*7*NumThread + 2*NumThread + thread_id];
			at = abcParam_d[i*7*NumThread*MaxElemPerEdge + j*7*NumThread + 3*NumThread + thread_id];
			bt = abcParam_d[i*7*NumThread*MaxElemPerEdge + j*7*NumThread + 4*NumThread + thread_id];
			ct = abcParam_d[i*7*NumThread*MaxElemPerEdge + j*7*NumThread + 5*NumThread + thread_id];
			V  = abcParam_d[i*7*NumThread*MaxElemPerEdge + j*7*NumThread + 6*NumThread + thread_id];
					
			Matrix_K_Aux_d[                   i*3*NumThread + 0*NumThread + thread_id] += ((as*e11+bs*e41+cs*e61)*at+(as*e14+bs*e44+cs*e64)*bt+(as*e16+bs*e46+cs*e66)*ct)*V;
			Matrix_K_Aux_d[                   i*3*NumThread + 1*NumThread + thread_id] += ((as*e12+bs*e42+cs*e62)*bt+(as*e14+bs*e44+cs*e64)*at+(as*e15+bs*e45+cs*e65)*ct)*V;
			Matrix_K_Aux_d[                   i*3*NumThread + 2*NumThread + thread_id] += ((as*e13+bs*e43+cs*e63)*ct+(as*e15+bs*e45+cs*e65)*bt+(as*e16+bs*e46+cs*e66)*at)*V;
			
			Matrix_K_Aux_d[  NumThreadSpace + i*3*NumThread + 0*NumThread + thread_id] += ((bs*e21+as*e41+cs*e51)*at+(bs*e24+as*e44+cs*e54)*bt+(bs*e26+as*e46+cs*e56)*ct)*V;
			Matrix_K_Aux_d[  NumThreadSpace + i*3*NumThread + 1*NumThread + thread_id] += ((bs*e22+as*e42+cs*e52)*bt+(bs*e24+as*e44+cs*e54)*at+(bs*e25+as*e45+cs*e55)*ct)*V;
			Matrix_K_Aux_d[  NumThreadSpace + i*3*NumThread + 2*NumThread + thread_id] += ((bs*e23+as*e43+cs*e53)*ct+(bs*e25+as*e45+cs*e55)*bt+(bs*e26+as*e46+cs*e56)*at)*V;
			
			Matrix_K_Aux_d[2*NumThreadSpace + i*3*NumThread + 0*NumThread + thread_id] += ((cs*e31+bs*e51+as*e61)*at+(cs*e34+bs*e54+as*e64)*bt+(cs*e36+bs*e56+as*e66)*ct)*V;
			Matrix_K_Aux_d[2*NumThreadSpace + i*3*NumThread + 1*NumThread + thread_id] += ((cs*e32+bs*e52+as*e62)*bt+(cs*e34+bs*e54+as*e64)*at+(cs*e35+bs*e55+as*e65)*ct)*V;
			Matrix_K_Aux_d[2*NumThreadSpace + i*3*NumThread + 2*NumThread + thread_id] += ((cs*e33+bs*e53+as*e63)*ct+(cs*e35+bs*e55+as*e65)*bt+(cs*e36+bs*e56+as*e66)*at)*V;
				
		}
			
	}
	
}

	/*K11  = (a1*e11+b1*e41+c1*e61)*a1+(a1*e14+b1*e44+c1*e64)*b1+(a1*e16+b1*e46+c1*e66)*c1;
	K12  = (a1*e12+b1*e42+c1*e62)*b1+(a1*e14+b1*e44+c1*e64)*a1+(a1*e15+b1*e45+c1*e65)*c1;
	K13  = (a1*e13+b1*e43+c1*e63)*c1+(a1*e15+b1*e45+c1*e65)*b1+(a1*e16+b1*e46+c1*e66)*a1;
	
	K14  = (a1*e11+b1*e41+c1*e61)*a2+(a1*e14+b1*e44+c1*e64)*b2+(a1*e16+b1*e46+c1*e66)*c2;
	K15  = (a1*e12+b1*e42+c1*e62)*b2+(a1*e14+b1*e44+c1*e64)*a2+(a1*e15+b1*e45+c1*e65)*c2;
	K16  = (a1*e13+b1*e43+c1*e63)*c2+(a1*e15+b1*e45+c1*e65)*b2+(a1*e16+b1*e46+c1*e66)*a2;
	
	K17  = (a1*e11+b1*e41+c1*e61)*a3+(a1*e14+b1*e44+c1*e64)*b3+(a1*e16+b1*e46+c1*e66)*c3;
	K18  = (a1*e12+b1*e42+c1*e62)*b3+(a1*e14+b1*e44+c1*e64)*a3+(a1*e15+b1*e45+c1*e65)*c3;
	K19  = (a1*e13+b1*e43+c1*e63)*c3+(a1*e15+b1*e45+c1*e65)*b3+(a1*e16+b1*e46+c1*e66)*a3;
	
	K110 = (a1*e11+b1*e41+c1*e61)*a4+(a1*e14+b1*e44+c1*e64)*b4+(a1*e16+b1*e46+c1*e66)*c4;
	K111 = (a1*e12+b1*e42+c1*e62)*b4+(a1*e14+b1*e44+c1*e64)*a4+(a1*e15+b1*e45+c1*e65)*c4;
	K112 = (a1*e13+b1*e43+c1*e63)*c4+(a1*e15+b1*e45+c1*e65)*b4+(a1*e16+b1*e46+c1*e66)*a4;
	
	// -------------------------------------------------------------------------------------------------------
	
	K21  = (b1*e21+a1*e41+c1*e51)*a1+(b1*e24+a1*e44+c1*e54)*b1+(b1*e26+a1*e46+c1*e56)*c1;
	K22  = (b1*e22+a1*e42+c1*e52)*b1+(b1*e24+a1*e44+c1*e54)*a1+(b1*e25+a1*e45+c1*e55)*c1;
	K23  = (b1*e23+a1*e43+c1*e53)*c1+(b1*e25+a1*e45+c1*e55)*b1+(b1*e26+a1*e46+c1*e56)*a1;
	
	K24  = (b1*e21+a1*e41+c1*e51)*a2+(b1*e24+a1*e44+c1*e54)*b2+(b1*e26+a1*e46+c1*e56)*c2;
	K25  = (b1*e22+a1*e42+c1*e52)*b2+(b1*e24+a1*e44+c1*e54)*a2+(b1*e25+a1*e45+c1*e55)*c2;
	K26  = (b1*e23+a1*e43+c1*e53)*c2+(b1*e25+a1*e45+c1*e55)*b2+(b1*e26+a1*e46+c1*e56)*a2;
	
	K27  = (b1*e21+a1*e41+c1*e51)*a3+(b1*e24+a1*e44+c1*e54)*b3+(b1*e26+a1*e46+c1*e56)*c3;
	K28  = (b1*e22+a1*e42+c1*e52)*b3+(b1*e24+a1*e44+c1*e54)*a3+(b1*e25+a1*e45+c1*e55)*c3;
	K29  = (b1*e23+a1*e43+c1*e53)*c3+(b1*e25+a1*e45+c1*e55)*b3+(b1*e26+a1*e46+c1*e56)*a3;
	
	K210 = (b1*e21+a1*e41+c1*e51)*a4+(b1*e24+a1*e44+c1*e54)*b4+(b1*e26+a1*e46+c1*e56)*c4;
	K211 = (b1*e22+a1*e42+c1*e52)*b4+(b1*e24+a1*e44+c1*e54)*a4+(b1*e25+a1*e45+c1*e55)*c4;
	K212 = (b1*e23+a1*e43+c1*e53)*c4+(b1*e25+a1*e45+c1*e55)*b4+(b1*e26+a1*e46+c1*e56)*a4;
	
	// -------------------------------------------------------------------------------------------------------
	
	K31  = (c1*e31+b1*e51+a1*e61)*a1+(c1*e34+b1*e54+a1*e64)*b1+(c1*e36+b1*e56+a1*e66)*c1;
	K32  = (c1*e32+b1*e52+a1*e62)*b1+(c1*e34+b1*e54+a1*e64)*a1+(c1*e35+b1*e55+a1*e65)*c1;
	K33  = (c1*e33+b1*e53+a1*e63)*c1+(c1*e35+b1*e55+a1*e65)*b1+(c1*e36+b1*e56+a1*e66)*a1;
	
	K34  = (c1*e31+b1*e51+a1*e61)*a2+(c1*e34+b1*e54+a1*e64)*b2+(c1*e36+b1*e56+a1*e66)*c1;
	K35  = (c1*e32+b1*e52+a1*e62)*b2+(c1*e34+b1*e54+a1*e64)*a2+(c1*e35+b1*e55+a1*e65)*c1;
	K36  = (c1*e33+b1*e53+a1*e63)*c2+(c1*e35+b1*e55+a1*e65)*b2+(c1*e36+b1*e56+a1*e66)*a1;
	
	K37  = (c1*e31+b1*e51+a1*e61)*a3+(c1*e34+b1*e54+a1*e64)*b3+(c1*e36+b1*e56+a1*e66)*c3;
	K38  = (c1*e32+b1*e52+a1*e62)*b3+(c1*e34+b1*e54+a1*e64)*a3+(c1*e35+b1*e55+a1*e65)*c3;
	K39  = (c1*e33+b1*e53+a1*e63)*c3+(c1*e35+b1*e55+a1*e65)*b3+(c1*e36+b1*e56+a1*e66)*a3;
	
	K310 = (c1*e31+b1*e51+a1*e61)*a4+(c1*e34+b1*e54+a1*e64)*b4+(c1*e36+b1*e56+a1*e66)*c4;
	K311 = (c1*e32+b1*e52+a1*e62)*b4+(c1*e34+b1*e54+a1*e64)*a4+(c1*e35+b1*e55+a1*e65)*c4;
	K312 = (c1*e33+b1*e53+a1*e63)*c4+(c1*e35+b1*e55+a1*e65)*b4+(c1*e36+b1*e56+a1*e66)*a4;*/
	
//=============== x =============== x =============== x =============== x =============== x =============== x ===============
void Launch_SUBT_V_V(dim3 BlockSizeVector, dim3 ThreadsPerBlockXY, int numdof, int numno, double *Vector_B_d, double *Vector_R_d) 
{
	SUBT_V_V   <<< BlockSizeVector, ThreadsPerBlockXY >>> (numdof, numno, Vector_B_d, Vector_R_d); // =======================> {r} = {b} - [A]{x}
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============
void Launch_CLEAR_V(dim3 BlockSizeVector, dim3 ThreadsPerBlockXY, int numdof, int numno, double *Vector_X_d)
{
	CLEAR_V <<< BlockSizeVector, ThreadsPerBlockXY >>> (numdof, numno, Vector_X_d); // =======================> {x} = 0
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============
void Launch_MULT_DM_V(dim3 BlockSizeVector, dim3 ThreadsPerBlockXY, int numdof, int numno, double *Matrix_M_d, double *Vector_R_d, double *Vector_D_d) 
{
	MULT_DM_V  <<< BlockSizeVector, ThreadsPerBlockXY >>> (numdof, numno, Matrix_M_d, Vector_R_d, Vector_D_d); // =======================> {d} = [M]-1{r}
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============
void Launch_MULT_V_V(int BlockSizeNumMultiProc, int NumMaxThreadsBlock, int numdof, int numno, double *Vector_R_d, double *Vector_D_d, double *delta_new_d)
{
	if(NumMaxThreadsBlock == 1024) 
		MULT_V_V_1024 <<< BlockSizeNumMultiProc, NumMaxThreadsBlock >>> (numdof, numno, Vector_R_d, Vector_D_d, delta_new_d); // =======================> delta_new = {r}T{d}
	else // == 512
		MULT_V_V_512  <<< BlockSizeNumMultiProc, NumMaxThreadsBlock >>> (numdof, numno, Vector_R_d, Vector_D_d, delta_new_d); // =======================> delta_new = {r}T{d}
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============
void Launch_SUM_V(dim3 BlockSizeVector, dim3 ThreadsPerBlockXY, int BlockSizeNumMultiProc, int numdof, int numno, int position, double *delta_new_d, double *delta_GPU_d, double *Result_d)
{
	if(BlockSizeNumMultiProc == 4)       SUM_V_4    <<< BlockSizeVector, ThreadsPerBlockXY >>> (numdof, numno, position, delta_new_d, delta_GPU_d, Result_d);  // =======================> {delta_new_V_d} = {r}T{d}
	else if(BlockSizeNumMultiProc == 8 ) SUM_V_8    <<< BlockSizeVector, ThreadsPerBlockXY >>> (numdof, numno, position, delta_new_d, delta_GPU_d, Result_d); 
	else if(BlockSizeNumMultiProc == 16) SUM_V_16   <<< BlockSizeVector, ThreadsPerBlockXY >>> (numdof, numno, position, delta_new_d, delta_GPU_d, Result_d); 
	else if(BlockSizeNumMultiProc == 32) SUM_V_32   <<< BlockSizeVector, ThreadsPerBlockXY >>> (numdof, numno, position, delta_new_d, delta_GPU_d, Result_d); 
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============
void Launch_SUM_GPU_4(dim3 BlockSizeVector, dim3 ThreadsPerBlockXY, int numdof, int numno, double *delta_GPU_d, double *delta_new_V_d)
{
	SUM_GPU_4 <<< BlockSizeVector, ThreadsPerBlockXY >>> (numdof, numno, delta_GPU_d, delta_new_V_d);  // =======================> delta_new_V_d[0]+[1]+[2]+[3] 
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============
void Launch_MULT_SM_V_128(dim3 BlockSizeMatrix, dim3 ThreadsPerBlockXY, int NumMaxThreadsBlock, int numdof, int numno, int RowSize, double *Matrix_A_d, double *Vector_D_d,
                          int *Vector_I_d, double *Vector_Q_d)
{
	if(NumMaxThreadsBlock == 1024)  
		MULT_SM_V_128_1024 <<< BlockSizeMatrix, ThreadsPerBlockXY >>> (numdof, numno, RowSize, Matrix_A_d, Vector_D_d, Vector_I_d, Vector_Q_d); // =======================> {q} = [A]{d}
	else // == 512
		MULT_SM_V_128_512  <<< BlockSizeMatrix, ThreadsPerBlockXY >>> (numdof, numno, RowSize, Matrix_A_d, Vector_D_d, Vector_I_d, Vector_Q_d); // =======================> {q} = [A]{d}
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============
void Launch_MULT_SM_V_Text_128(dim3 BlockSizeMatrix, dim3 ThreadsPerBlockXY, int numdof, int numno, int RowSize, double *Matrix_A_d, double *Vector_D_d, int *Vector_I_d, double *Vector_Q_d)
{
	MULT_SM_V_Text_128 <<< BlockSizeMatrix, ThreadsPerBlockXY >>> (numdof, numno, RowSize, Matrix_A_d, Vector_D_d, Vector_I_d, Vector_Q_d); // =======================> {q} = [A]{d}
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============
void Launch_ADD_V_V_X(dim3 BlockSizeVector, dim3 ThreadsPerBlockXY, int numdof, int numno, double *Vector_X_d, double *Vector_D_d, double *delta_new_V_d, double *delta_aux_V_d, 
							   double *Vector_R_d, double *Vector_Q_d)
{
	ADD_V_V_X <<< BlockSizeVector, ThreadsPerBlockXY >>> (numdof, numno, Vector_X_d, Vector_D_d, delta_new_V_d, delta_aux_V_d, Vector_R_d, Vector_Q_d); // =============> {x} = {x} + alfa{d}																																                  
	                                                                                                                                                    // =============> {r} = {r} - alfa{q}
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============
void Launch_UPDATE_DELTA(dim3 BlockSizeVector, dim3 ThreadsPerBlockXY, int numdof, int numno, double *delta_new_V_d, double *delta_old_V_d)
{

	UPDATE_DELTA <<< BlockSizeVector, ThreadsPerBlockXY >>> (numdof, numno, delta_new_V_d, delta_old_V_d);  // =======================> {delta_old} = {delta_new}
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============
void Launch_ADD_V_V_D(dim3 BlockSizeVector, dim3 ThreadsPerBlockXY, int numdof, int numno, double *Vector_S_d, double *Vector_D_d, double *delta_new_V_d, double *delta_old_V_d)
{
	ADD_V_V_D <<< BlockSizeVector, ThreadsPerBlockXY >>> (numdof, numno, Vector_S_d, Vector_D_d, delta_new_V_d, delta_old_V_d); // =======================> {d} = {s} + beta{d}
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============
void Launch_Copy_Local_Global(dim3 BlockSizeVector, dim3 ThreadsPerBlockXY, int numdof, int numno, int position, double *Vector_D_d, double *Vector_D_Global_d)
{
	Copy_Local_Global <<< BlockSizeVector, ThreadsPerBlockXY >>> (numdof, numno, position, Vector_D_d, Vector_D_Global_d); 
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============
void Launch_Copy_for_Vector_4(double *delta_0_new_V_d, double *delta_0_GPU_d)
{
	Copy_for_Vector_4 <<< 1, 1 >>> (delta_0_new_V_d, delta_0_GPU_d);  
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============
void Launch_Clean_Global_Vector(dim3 BlockSizeVectorGlobal, dim3 ThreadsPerBlockXY, int numdof, int numnoglobal, double *Vector_D_Global_d)
{
	Clean_Global_Vector <<< BlockSizeVectorGlobal, ThreadsPerBlockXY >>> (numdof, numnoglobal, Vector_D_Global_d);  
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============
void Launch_AssemblyStiffness3DQ1(dim3 numBlocksSM, dim3 threadsPerBlockSM, int numno, int RowSize, double *CoordQ1_d, double *MatPropQ1_d, double*Matrix_K_Aux_d)
{
	//AssemblyStiffness3DQ1 <<<numBlocksSM, threadsPerBlockSM>>> (numno, RowSize, CoordQ1_d, MatPropQ1_d, Matrix_K_Aux_d);
	AssemblyStiffness3DQ1Mod <<<numBlocksSM, threadsPerBlockSM>>> (numno, RowSize, CoordQ1_d, MatPropQ1_d, Matrix_K_Aux_d);
	cudaDeviceSynchronize();
}
	
//=============== x =============== x =============== x =============== x =============== x =============== x ===============
void Launch_AssemblyStiffness3DQ2(dim3 numBlocksSM, dim3 threadsPerBlockSM, int numno, int RowSize, double*CoordQ2_d, double*MatPropQ2_d, double*Matrix_K_Aux_d)
{
	//AssemblyStiffness3DQ2 <<<numBlocksSM, threadsPerBlockSM>>> (numno, RowSize, CoordQ2_d, MatPropQ2_d, Matrix_K_Aux_d);
	AssemblyStiffness3DQ2Mod <<<numBlocksSM, threadsPerBlockSM>>> (numno, RowSize, CoordQ2_d, MatPropQ2_d, Matrix_K_Aux_d);
	cudaDeviceSynchronize();
}
	
//=============== x =============== x =============== x =============== x =============== x =============== x ===============
void Launch_AssemblyStiffness3DQ3(dim3 numBlocksSM, dim3 threadsPerBlockSM, int numno, int RowSize, double*CoordQ3_d, double*MatPropQ3_d, double*Matrix_K_Aux_d)
{
	//AssemblyStiffness3DQ3 <<<numBlocksSM, threadsPerBlockSM>>> (numno, RowSize, CoordQ3_d, MatPropQ3_d, Matrix_K_Aux_d);  
	AssemblyStiffness3DQ3Mod <<<numBlocksSM, threadsPerBlockSM>>> (numno, RowSize, CoordQ3_d, MatPropQ3_d, Matrix_K_Aux_d);  
	cudaDeviceSynchronize();
}
	
//=============== x =============== x =============== x =============== x =============== x =============== x ===============
void Launch_AssemblyStiffness3DQ4(dim3 numBlocksSM, dim3 threadsPerBlockSM, int numno, int RowSize, double*CoordQ4_d, double*MatPropQ4_d, double*Matrix_K_Aux_d)
{
	//AssemblyStiffness3DQ4 <<<numBlocksSM, threadsPerBlockSM>>> (numno, RowSize, CoordQ4_d, MatPropQ4_d, Matrix_K_Aux_d);
	AssemblyStiffness3DQ4Mod <<<numBlocksSM, threadsPerBlockSM>>> (numno, RowSize, CoordQ4_d, MatPropQ4_d, Matrix_K_Aux_d);
	cudaDeviceSynchronize();
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============
void Launch_AssemblyStiffness3DQ5(dim3 numBlocksSM, dim3 threadsPerBlockSM, int numno, int RowSize, double*CoordQ5_d, double*MatPropQ5_d, double*Matrix_K_Aux_d)
{
	//AssemblyStiffness3DQ5 <<<numBlocksSM, threadsPerBlockSM>>> (numno, RowSize, CoordQ5_d, MatPropQ5_d, Matrix_K_Aux_d);
	AssemblyStiffness3DQ5Mod <<<numBlocksSM, threadsPerBlockSM>>> (numno, RowSize, CoordQ5_d, MatPropQ5_d, Matrix_K_Aux_d);
	cudaDeviceSynchronize();
}
	
//=============== x =============== x =============== x =============== x =============== x =============== x ===============
void Launch_AssemblyStiffness3DQ6(dim3 numBlocksSM, dim3 threadsPerBlockSM, int numno, int RowSize, double*CoordQ6_d, double*MatPropQ6_d, double*Matrix_K_Aux_d)
{
	//AssemblyStiffness3DQ6 <<<numBlocksSM, threadsPerBlockSM>>> (numno, RowSize, CoordQ6_d, MatPropQ6_d, Matrix_K_Aux_d);
	AssemblyStiffness3DQ6Mod <<<numBlocksSM, threadsPerBlockSM>>> (numno, RowSize, CoordQ6_d, MatPropQ6_d, Matrix_K_Aux_d);
	cudaDeviceSynchronize();
}
	
//=============== x =============== x =============== x =============== x =============== x =============== x ===============
void Launch_AssemblyStiffness3DQ7(dim3 numBlocksSM, dim3 threadsPerBlockSM, int numno, int RowSize, double*CoordQ7_d, double*MatPropQ7_d, double*Matrix_K_Aux_d)
{
	//AssemblyStiffness3DQ7 <<<numBlocksSM, threadsPerBlockSM>>> (numno, RowSize, CoordQ7_d, MatPropQ7_d, Matrix_K_Aux_d);
	AssemblyStiffness3DQ7Mod <<<numBlocksSM, threadsPerBlockSM>>> (numno, RowSize, CoordQ7_d, MatPropQ7_d, Matrix_K_Aux_d);
	cudaDeviceSynchronize();
}
	
//=============== x =============== x =============== x =============== x =============== x =============== x ===============
void Launch_AssemblyStiffness3DQ8(dim3 numBlocksSM, dim3 threadsPerBlockSM, int numno, int RowSize, double*CoordQ8_d, double*MatPropQ8_d, double*Matrix_K_Aux_d)
{
	//AssemblyStiffness3DQ8 <<<numBlocksSM, threadsPerBlockSM>>> (numno, RowSize, CoordQ8_d, MatPropQ8_d, Matrix_K_Aux_d); 
	AssemblyStiffness3DQ8Mod <<<numBlocksSM, threadsPerBlockSM>>> (numno, RowSize, CoordQ8_d, MatPropQ8_d, Matrix_K_Aux_d); 
	cudaDeviceSynchronize();
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============
void Launch_TrasnsposeCoalesced(dim3 numBlocksTR, dim3 threadsPerBlockTR, int NumThread, double *Matrix_K_Aux_d, double *Matrix_K_d)
{
	TrasnsposeCoalesced <<<numBlocksTR, threadsPerBlockTR>>> (NumThread, Matrix_K_Aux_d, Matrix_K_d);
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============
void Launch_EvaluateMatrixM(dim3 numBlocksM, dim3 threadsPerBlockM, int numdof, int numno, int *iiPosition_d, double *Matrix_K_d, double *Matrix_M_d)
{
	EvaluateMatrixM <<<numBlocksM, threadsPerBlockM>>> (numdof, numno, iiPosition_d, Matrix_K_d, Matrix_M_d);
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============
void Launch_EvaluateStrainState(dim3 numBlocksStrain, dim3 threadsPerBlockStrain, int Localnumel, int RowSize, 
double *CoordElem_d, int *Connect_d, double *Vector_X_d, double *Strain_d)
{
	EvaluateStrainStateAverage <<<numBlocksStrain, threadsPerBlockStrain>>> (Localnumel, RowSize, CoordElem_d, Connect_d, Vector_X_d, Strain_d);
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============

__global__ void AssemblyStiffness3DQ1Mod(int numno, int RowSize, double *CoordQ1_d, double *MatPropQ1_d, double *Matrix_K_Aux_d) 
{
	double r, s, t;  // 3
	double XEL0, YEL0, ZEL0, XEL1, YEL1, ZEL1, XEL2, YEL2, ZEL2, XEL3, YEL3, ZEL3;  // 12 
	double XEL4, YEL4, ZEL4, XEL5, YEL5, ZEL5, XEL6, YEL6, ZEL6, XEL7, YEL7, ZEL7;  // 12 
	double xgaus[2];  // 2
	double p, E, C00, C01, C33;  // 5
	double phi_r[8], phi_s[8], phi_t[8];  // 24
	double jac[3][3];  // 9
	double jacob;  // 1
	double gama[3][3];  // 9
	double auxX[8], auxY[8], auxZ[8];  // 24
	int ig, jg, kg;  // 3
	
	const int xIndex = blockIdx.x*blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y*blockDim.y + threadIdx.y;
	const int thread_id = (gridDim.x*blockDim.x)*yIndex + xIndex;
	
	__syncthreads();
	
	if(thread_id < numno) {
	
		XEL0 = CoordQ1_d[thread_id+ 0*numno];
		YEL0 = CoordQ1_d[thread_id+ 1*numno];
		ZEL0 = CoordQ1_d[thread_id+ 2*numno];   	
		XEL1 = CoordQ1_d[thread_id+ 3*numno];
		YEL1 = CoordQ1_d[thread_id+ 4*numno];  
		ZEL1 = CoordQ1_d[thread_id+ 5*numno];  	
		XEL2 = CoordQ1_d[thread_id+ 6*numno];
		YEL2 = CoordQ1_d[thread_id+ 7*numno]; 
		ZEL2 = CoordQ1_d[thread_id+ 8*numno];  		
		XEL3 = CoordQ1_d[thread_id+ 9*numno];
		YEL3 = CoordQ1_d[thread_id+10*numno];
		ZEL3 = CoordQ1_d[thread_id+11*numno];
		
		XEL4 = CoordQ1_d[thread_id+12*numno];
		YEL4 = CoordQ1_d[thread_id+13*numno];
		ZEL4 = CoordQ1_d[thread_id+14*numno];   	
		XEL5 = CoordQ1_d[thread_id+15*numno];
		YEL5 = CoordQ1_d[thread_id+16*numno];  
		ZEL5 = CoordQ1_d[thread_id+17*numno];  	
		XEL6 = CoordQ1_d[thread_id+18*numno];
		YEL6 = CoordQ1_d[thread_id+19*numno]; 
		ZEL6 = CoordQ1_d[thread_id+20*numno];  		
		XEL7 = CoordQ1_d[thread_id+21*numno];
		YEL7 = CoordQ1_d[thread_id+22*numno];
		ZEL7 = CoordQ1_d[thread_id+23*numno];
		
		__syncthreads();

		xgaus[0]=-0.577350269189626;
		xgaus[1]= 0.577350269189626;;
							
		E = MatPropQ1_d[thread_id+ 0*numno];
		p = MatPropQ1_d[thread_id+ 1*numno];
					
		C00=(E*(1-p))/((1+p)*(1-2*p));  
		C01=(E*(1-p))/((1+p)*(1-2*p))*(p/(1-p)); 
		C33=(E*(1-p))/((1+p)*(1-2*p))*((1-2*p)/(2*(1-p)));
		
		for(ig=0; ig<2; ig++) {                                          // =================== Loop ig ===================
			t=xgaus[ig];
					
			for(jg=0; jg<2; jg++) {                                      // =================== Loop jg ===================
				s=xgaus[jg]; 
				
				for(kg=0; kg<2; kg++) {                                  // =================== Loop kg ===================
				r=xgaus[kg]; 
				
					// Shape function derivative:
					phi_r[0] = -0.125*(1.0-s)*(1.0-t);
					phi_r[1] =  0.125*(1.0-s)*(1.0-t);
					phi_r[2] =  0.125*(1.0+s)*(1.0-t);
					phi_r[3] = -0.125*(1.0+s)*(1.0-t);
					phi_r[4] = -0.125*(1.0-s)*(1.0+t);
					phi_r[5] =  0.125*(1.0-s)*(1.0+t);
					phi_r[6] =  0.125*(1.0+s)*(1.0+t);
					phi_r[7] = -0.125*(1.0+s)*(1.0+t);
	 
					phi_s[0] = -0.125*(1.0-r)*(1.0-t);
					phi_s[1] = -0.125*(1.0+r)*(1.0-t);
					phi_s[2] =  0.125*(1.0+r)*(1.0-t);
					phi_s[3] =  0.125*(1.0-r)*(1.0-t);
					phi_s[4] = -0.125*(1.0-r)*(1.0+t);
					phi_s[5] = -0.125*(1.0+r)*(1.0+t);
					phi_s[6] =  0.125*(1.0+r)*(1.0+t);
					phi_s[7] =  0.125*(1.0-r)*(1.0+t);
		 
					phi_t[0] = -0.125*(1.0-r)*(1.0-s);
					phi_t[1] = -0.125*(1.0+r)*(1.0-s);
					phi_t[2] = -0.125*(1.0+r)*(1.0+s);
					phi_t[3] = -0.125*(1.0-r)*(1.0+s);
					phi_t[4] =  0.125*(1.0-r)*(1.0-s);
					phi_t[5] =  0.125*(1.0+r)*(1.0-s);
					phi_t[6] =  0.125*(1.0+r)*(1.0+s);
					phi_t[7] =  0.125*(1.0-r)*(1.0+s);
					
					// Terms of the Jacobian matrix:
					jac[0][0]=phi_r[0]*XEL0+phi_r[1]*XEL1+phi_r[2]*XEL2+phi_r[3]*XEL3+phi_r[4]*XEL4+phi_r[5]*XEL5+phi_r[6]*XEL6+phi_r[7]*XEL7;
					jac[0][1]=phi_r[0]*YEL0+phi_r[1]*YEL1+phi_r[2]*YEL2+phi_r[3]*YEL3+phi_r[4]*YEL4+phi_r[5]*YEL5+phi_r[6]*YEL6+phi_r[7]*YEL7;
					jac[0][2]=phi_r[0]*ZEL0+phi_r[1]*ZEL1+phi_r[2]*ZEL2+phi_r[3]*ZEL3+phi_r[4]*ZEL4+phi_r[5]*ZEL5+phi_r[6]*ZEL6+phi_r[7]*ZEL7;

					jac[1][0]=phi_s[0]*XEL0+phi_s[1]*XEL1+phi_s[2]*XEL2+phi_s[3]*XEL3+phi_s[4]*XEL4+phi_s[5]*XEL5+phi_s[6]*XEL6+phi_s[7]*XEL7;
					jac[1][1]=phi_s[0]*YEL0+phi_s[1]*YEL1+phi_s[2]*YEL2+phi_s[3]*YEL3+phi_s[4]*YEL4+phi_s[5]*YEL5+phi_s[6]*YEL6+phi_s[7]*YEL7;
					jac[1][2]=phi_s[0]*ZEL0+phi_s[1]*ZEL1+phi_s[2]*ZEL2+phi_s[3]*ZEL3+phi_s[4]*ZEL4+phi_s[5]*ZEL5+phi_s[6]*ZEL6+phi_s[7]*ZEL7;
					
					jac[2][0]=phi_t[0]*XEL0+phi_t[1]*XEL1+phi_t[2]*XEL2+phi_t[3]*XEL3+phi_t[4]*XEL4+phi_t[5]*XEL5+phi_t[6]*XEL6+phi_t[7]*XEL7;
					jac[2][1]=phi_t[0]*YEL0+phi_t[1]*YEL1+phi_t[2]*YEL2+phi_t[3]*YEL3+phi_t[4]*YEL4+phi_t[5]*YEL5+phi_t[6]*YEL6+phi_t[7]*YEL7;
					jac[2][2]=phi_t[0]*ZEL0+phi_t[1]*ZEL1+phi_t[2]*ZEL2+phi_t[3]*ZEL3+phi_t[4]*ZEL4+phi_t[5]*ZEL5+phi_t[6]*ZEL6+phi_t[7]*ZEL7;
						
					// Jacobian matrix determinant:	
					jacob = jac[0][0]*(jac[1][1]*jac[2][2]-jac[1][2]*jac[2][1])-jac[0][1]*(jac[1][0]*jac[2][2]-jac[1][2]*jac[2][0])+
							jac[0][2]*(jac[1][0]*jac[2][1]-jac[1][1]*jac[2][0]);
					
					if(jacob > 0.) {
					
						gama[0][0] = (jac[1][1]*jac[2][2] - jac[1][2]*jac[2][1]) /jacob;
						gama[0][1] =-(jac[0][1]*jac[2][2] - jac[0][2]*jac[2][1]) /jacob;
						gama[0][2] = (jac[0][1]*jac[1][2] - jac[0][2]*jac[1][1]) /jacob;
						
						gama[1][0] =-(jac[1][0]*jac[2][2] - jac[1][2]*jac[2][0]) /jacob;
						gama[1][1] = (jac[0][0]*jac[2][2] - jac[0][2]*jac[2][0]) /jacob;
						gama[1][2] =-(jac[0][0]*jac[1][2] - jac[0][2]*jac[1][0]) /jacob;
						
						gama[2][0] = (jac[1][0]*jac[2][1] - jac[1][1]*jac[2][0]) /jacob;
						gama[2][1] =-(jac[0][0]*jac[2][1] - jac[0][1]*jac[2][0]) /jacob;
						gama[2][2] = (jac[0][0]*jac[1][1] - jac[0][1]*jac[1][0]) /jacob;
						
						// *******************************************************************************************************
										
						auxX[0] = gama[0][0]*phi_r[0]+gama[0][1]*phi_s[0]+gama[0][2]*phi_t[0];
						auxY[0] = gama[1][0]*phi_r[0]+gama[1][1]*phi_s[0]+gama[1][2]*phi_t[0];
						auxZ[0] = gama[2][0]*phi_r[0]+gama[2][1]*phi_s[0]+gama[2][2]*phi_t[0];
							
						auxX[1] = gama[0][0]*phi_r[1]+gama[0][1]*phi_s[1]+gama[0][2]*phi_t[1];
						auxY[1] = gama[1][0]*phi_r[1]+gama[1][1]*phi_s[1]+gama[1][2]*phi_t[1];
						auxZ[1] = gama[2][0]*phi_r[1]+gama[2][1]*phi_s[1]+gama[2][2]*phi_t[1];
						
						auxX[2] = gama[0][0]*phi_r[2]+gama[0][1]*phi_s[2]+gama[0][2]*phi_t[2];
						auxY[2] = gama[1][0]*phi_r[2]+gama[1][1]*phi_s[2]+gama[1][2]*phi_t[2];
						auxZ[2] = gama[2][0]*phi_r[2]+gama[2][1]*phi_s[2]+gama[2][2]*phi_t[2];
						
						auxX[3] = gama[0][0]*phi_r[3]+gama[0][1]*phi_s[3]+gama[0][2]*phi_t[3];
						auxY[3] = gama[1][0]*phi_r[3]+gama[1][1]*phi_s[3]+gama[1][2]*phi_t[3];
						auxZ[3] = gama[2][0]*phi_r[3]+gama[2][1]*phi_s[3]+gama[2][2]*phi_t[3];
						
						auxX[4] = gama[0][0]*phi_r[4]+gama[0][1]*phi_s[4]+gama[0][2]*phi_t[4];
						auxY[4] = gama[1][0]*phi_r[4]+gama[1][1]*phi_s[4]+gama[1][2]*phi_t[4];
						auxZ[4] = gama[2][0]*phi_r[4]+gama[2][1]*phi_s[4]+gama[2][2]*phi_t[4];
							
						auxX[5] = gama[0][0]*phi_r[5]+gama[0][1]*phi_s[5]+gama[0][2]*phi_t[5];
						auxY[5] = gama[1][0]*phi_r[5]+gama[1][1]*phi_s[5]+gama[1][2]*phi_t[5];
						auxZ[5] = gama[2][0]*phi_r[5]+gama[2][1]*phi_s[5]+gama[2][2]*phi_t[5];
						
						auxX[6] = gama[0][0]*phi_r[6]+gama[0][1]*phi_s[6]+gama[0][2]*phi_t[6];
						auxY[6] = gama[1][0]*phi_r[6]+gama[1][1]*phi_s[6]+gama[1][2]*phi_t[6];
						auxZ[6] = gama[2][0]*phi_r[6]+gama[2][1]*phi_s[6]+gama[2][2]*phi_t[6];
						
						auxX[7] = gama[0][0]*phi_r[7]+gama[0][1]*phi_s[7]+gama[0][2]*phi_t[7];
						auxY[7] = gama[1][0]*phi_r[7]+gama[1][1]*phi_s[7]+gama[1][2]*phi_t[7];
						auxZ[7] = gama[2][0]*phi_r[7]+gama[2][1]*phi_s[7]+gama[2][2]*phi_t[7];
						
						__syncthreads();  // Line 0:
							
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+ 0] += jacob*(auxX[6]*C00*auxX[0] + auxY[6]*C33*auxY[0] + auxZ[6]*C33*auxZ[0]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+ 1] += jacob*(auxX[6]*C01*auxY[0] + auxY[6]*C33*auxX[0]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+ 2] += jacob*(auxX[6]*C01*auxZ[0] + auxZ[6]*C33*auxX[0]);
							
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+ 3] += jacob*(auxX[6]*C00*auxX[1] + auxY[6]*C33*auxY[1] + auxZ[6]*C33*auxZ[1]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+ 4] += jacob*(auxX[6]*C01*auxY[1] + auxY[6]*C33*auxX[1]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+ 5] += jacob*(auxX[6]*C01*auxZ[1] + auxZ[6]*C33*auxX[1]);
							
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+ 9] += jacob*(auxX[6]*C00*auxX[3] + auxY[6]*C33*auxY[3] + auxZ[6]*C33*auxZ[3]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+10] += jacob*(auxX[6]*C01*auxY[3] + auxY[6]*C33*auxX[3]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+11] += jacob*(auxX[6]*C01*auxZ[3] + auxZ[6]*C33*auxX[3]);
							
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+12] += jacob*(auxX[6]*C00*auxX[2] + auxY[6]*C33*auxY[2] + auxZ[6]*C33*auxZ[2]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+13] += jacob*(auxX[6]*C01*auxY[2] + auxY[6]*C33*auxX[2]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+14] += jacob*(auxX[6]*C01*auxZ[2] + auxZ[6]*C33*auxX[2]);
							
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+27] += jacob*(auxX[6]*C00*auxX[4] + auxY[6]*C33*auxY[4] + auxZ[6]*C33*auxZ[4]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+28] += jacob*(auxX[6]*C01*auxY[4] + auxY[6]*C33*auxX[4]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+29] += jacob*(auxX[6]*C01*auxZ[4] + auxZ[6]*C33*auxX[4]);
							
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+30] += jacob*(auxX[6]*C00*auxX[5] + auxY[6]*C33*auxY[5] + auxZ[6]*C33*auxZ[5]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+31] += jacob*(auxX[6]*C01*auxY[5] + auxY[6]*C33*auxX[5]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+32] += jacob*(auxX[6]*C01*auxZ[5] + auxZ[6]*C33*auxX[5]);
							
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+36] += jacob*(auxX[6]*C00*auxX[7] + auxY[6]*C33*auxY[7] + auxZ[6]*C33*auxZ[7]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+37] += jacob*(auxX[6]*C01*auxY[7] + auxY[6]*C33*auxX[7]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+38] += jacob*(auxX[6]*C01*auxZ[7] + auxZ[6]*C33*auxX[7]);
							
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+39] += jacob*(auxX[6]*C00*auxX[6] + auxY[6]*C33*auxY[6] + auxZ[6]*C33*auxZ[6]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+40] += jacob*(auxX[6]*C01*auxY[6] + auxY[6]*C33*auxX[6]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+41] += jacob*(auxX[6]*C01*auxZ[6] + auxZ[6]*C33*auxX[6]);
								
						
						__syncthreads();  // Line 1:
						
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+ 0] += jacob*(auxY[6]*C01*auxX[0] + auxX[6]*C33*auxY[0]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+ 1] += jacob*(auxY[6]*C00*auxY[0] + auxX[6]*C33*auxX[0] + auxZ[6]*C33*auxZ[0]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+ 2] += jacob*(auxY[6]*C01*auxZ[0] + auxZ[6]*C33*auxY[0]);
							
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+ 3] += jacob*(auxY[6]*C01*auxX[1] + auxX[6]*C33*auxY[1]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+ 4] += jacob*(auxY[6]*C00*auxY[1] + auxX[6]*C33*auxX[1] + auxZ[6]*C33*auxZ[1]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+ 5] += jacob*(auxY[6]*C01*auxZ[1] + auxZ[6]*C33*auxY[1]);
							
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+ 9] += jacob*(auxY[6]*C01*auxX[3] + auxX[6]*C33*auxY[3]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+10] += jacob*(auxY[6]*C00*auxY[3] + auxX[6]*C33*auxX[3] + auxZ[6]*C33*auxZ[3]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+11] += jacob*(auxY[6]*C01*auxZ[3] + auxZ[6]*C33*auxY[3]);
							
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+12] += jacob*(auxY[6]*C01*auxX[2] + auxX[6]*C33*auxY[2]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+13] += jacob*(auxY[6]*C00*auxY[2] + auxX[6]*C33*auxX[2] + auxZ[6]*C33*auxZ[2]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+14] += jacob*(auxY[6]*C01*auxZ[2] + auxZ[6]*C33*auxY[2]);
							
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+27] += jacob*(auxY[6]*C01*auxX[4] + auxX[6]*C33*auxY[4]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+28] += jacob*(auxY[6]*C00*auxY[4] + auxX[6]*C33*auxX[4] + auxZ[6]*C33*auxZ[4]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+29] += jacob*(auxY[6]*C01*auxZ[4] + auxZ[6]*C33*auxY[4]);
							
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+30] += jacob*(auxY[6]*C01*auxX[5] + auxX[6]*C33*auxY[5]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+31] += jacob*(auxY[6]*C00*auxY[5] + auxX[6]*C33*auxX[5] + auxZ[6]*C33*auxZ[5]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+32] += jacob*(auxY[6]*C01*auxZ[5] + auxZ[6]*C33*auxY[5]);
							
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+36] += jacob*(auxY[6]*C01*auxX[7] + auxX[6]*C33*auxY[7]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+37] += jacob*(auxY[6]*C00*auxY[7] + auxX[6]*C33*auxX[7] + auxZ[6]*C33*auxZ[7]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+38] += jacob*(auxY[6]*C01*auxZ[7] + auxZ[6]*C33*auxY[7]);
							
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+39] += jacob*(auxY[6]*C01*auxX[6] + auxX[6]*C33*auxY[6]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+40] += jacob*(auxY[6]*C00*auxY[6] + auxX[6]*C33*auxX[6] + auxZ[6]*C33*auxZ[6]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+41] += jacob*(auxY[6]*C01*auxZ[6] + auxZ[6]*C33*auxY[6]);
									
						__syncthreads();  // Line 2:
						
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+ 0] += jacob*(auxZ[6]*C01*auxX[0] + auxX[6]*C33*auxZ[0]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+ 1] += jacob*(auxZ[6]*C01*auxY[0] + auxY[6]*C33*auxZ[0]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+ 2] += jacob*(auxZ[6]*C00*auxZ[0] + auxY[6]*C33*auxY[0] + auxX[6]*C33*auxX[0]);
							
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+ 3] += jacob*(auxZ[6]*C01*auxX[1] + auxX[6]*C33*auxZ[1]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+ 4] += jacob*(auxZ[6]*C01*auxY[1] + auxY[6]*C33*auxZ[1]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+ 5] += jacob*(auxZ[6]*C00*auxZ[1] + auxY[6]*C33*auxY[1] + auxX[6]*C33*auxX[1]);
							
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+ 9] += jacob*(auxZ[6]*C01*auxX[3] + auxX[6]*C33*auxZ[3]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+10] += jacob*(auxZ[6]*C01*auxY[3] + auxY[6]*C33*auxZ[3]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+11] += jacob*(auxZ[6]*C00*auxZ[3] + auxY[6]*C33*auxY[3] + auxX[6]*C33*auxX[3]);
							
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+12] += jacob*(auxZ[6]*C01*auxX[2] + auxX[6]*C33*auxZ[2]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+13] += jacob*(auxZ[6]*C01*auxY[2] + auxY[6]*C33*auxZ[2]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+14] += jacob*(auxZ[6]*C00*auxZ[2] + auxY[6]*C33*auxY[2] + auxX[6]*C33*auxX[2]);
							
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+27] += jacob*(auxZ[6]*C01*auxX[4] + auxX[6]*C33*auxZ[4]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+28] += jacob*(auxZ[6]*C01*auxY[4] + auxY[6]*C33*auxZ[4]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+29] += jacob*(auxZ[6]*C00*auxZ[4] + auxY[6]*C33*auxY[4] + auxX[6]*C33*auxX[4]);
							
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+30] += jacob*(auxZ[6]*C01*auxX[5] + auxX[6]*C33*auxZ[5]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+31] += jacob*(auxZ[6]*C01*auxY[5] + auxY[6]*C33*auxZ[5]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+32] += jacob*(auxZ[6]*C00*auxZ[5] + auxY[6]*C33*auxY[5] + auxX[6]*C33*auxX[5]);
							
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+36] += jacob*(auxZ[6]*C01*auxX[7] + auxX[6]*C33*auxZ[7]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+37] += jacob*(auxZ[6]*C01*auxY[7] + auxY[6]*C33*auxZ[7]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+38] += jacob*(auxZ[6]*C00*auxZ[7] + auxY[6]*C33*auxY[7] + auxX[6]*C33*auxX[7]);
							
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+39] += jacob*(auxZ[6]*C01*auxX[6] + auxX[6]*C33*auxZ[6]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+40] += jacob*(auxZ[6]*C01*auxY[6] + auxY[6]*C33*auxZ[6]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+41] += jacob*(auxZ[6]*C00*auxZ[6] + auxY[6]*C33*auxY[6] + auxX[6]*C33*auxX[6]);
		
					}
					
				}                                                        // =================== Loop kg ===================
											
			}                                                            // =================== Loop jg ===================

		}                                                                // =================== Loop ig ===================
		
	}
	
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============

__global__ void AssemblyStiffness3DQ2Mod(int numno, int RowSize, double *CoordQ2_d, double *MatPropQ2_d, double *Matrix_K_Aux_d) 
{
	double r, s, t;  // 3
	double XEL0, YEL0, ZEL0, XEL1, YEL1, ZEL1, XEL2, YEL2, ZEL2, XEL3, YEL3, ZEL3;  // 12 
	double XEL4, YEL4, ZEL4, XEL5, YEL5, ZEL5, XEL6, YEL6, ZEL6, XEL7, YEL7, ZEL7;  // 12 
	double xgaus[2];  // 2
	double p, E, C00, C01, C33;  // 5
	double phi_r[8], phi_s[8], phi_t[8];  // 24
	double jac[3][3];  // 9
	double jacob;  // 1
	double gama[3][3];  // 9
	double auxX[8], auxY[8], auxZ[8];  // 24
	int ig, jg, kg;  // 3
	
	const int xIndex = blockIdx.x*blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y*blockDim.y + threadIdx.y;
	const int thread_id = (gridDim.x*blockDim.x)*yIndex + xIndex;

	__syncthreads();
	
	if(thread_id < numno) {
	
		XEL0 = CoordQ2_d[thread_id+ 0*numno];
		YEL0 = CoordQ2_d[thread_id+ 1*numno];
		ZEL0 = CoordQ2_d[thread_id+ 2*numno];   	
		XEL1 = CoordQ2_d[thread_id+ 3*numno];
		YEL1 = CoordQ2_d[thread_id+ 4*numno];  
		ZEL1 = CoordQ2_d[thread_id+ 5*numno];  	
		XEL2 = CoordQ2_d[thread_id+ 6*numno];
		YEL2 = CoordQ2_d[thread_id+ 7*numno]; 
		ZEL2 = CoordQ2_d[thread_id+ 8*numno];  		
		XEL3 = CoordQ2_d[thread_id+ 9*numno];
		YEL3 = CoordQ2_d[thread_id+10*numno];
		ZEL3 = CoordQ2_d[thread_id+11*numno];
		
		XEL4 = CoordQ2_d[thread_id+12*numno];
		YEL4 = CoordQ2_d[thread_id+13*numno];
		ZEL4 = CoordQ2_d[thread_id+14*numno];   	
		XEL5 = CoordQ2_d[thread_id+15*numno];
		YEL5 = CoordQ2_d[thread_id+16*numno];  
		ZEL5 = CoordQ2_d[thread_id+17*numno];  	
		XEL6 = CoordQ2_d[thread_id+18*numno];
		YEL6 = CoordQ2_d[thread_id+19*numno]; 
		ZEL6 = CoordQ2_d[thread_id+20*numno];  		
		XEL7 = CoordQ2_d[thread_id+21*numno];
		YEL7 = CoordQ2_d[thread_id+22*numno];
		ZEL7 = CoordQ2_d[thread_id+23*numno];
		
		__syncthreads();

		xgaus[0]=-0.577350269189626;
		xgaus[1]= 0.577350269189626;
							
		E = MatPropQ2_d[thread_id+ 0*numno];
		p = MatPropQ2_d[thread_id+ 1*numno];
					
		C00=(E*(1-p))/((1+p)*(1-2*p));  
		C01=(E*(1-p))/((1+p)*(1-2*p))*(p/(1-p)); 
		C33=(E*(1-p))/((1+p)*(1-2*p))*((1-2*p)/(2*(1-p)));
			
		for(ig=0; ig<2; ig++) {                                          // =================== Loop ig ===================
			t=xgaus[ig];
					
			for(jg=0; jg<2; jg++) {                                      // =================== Loop jg ===================
				s=xgaus[jg]; 
				
				for(kg=0; kg<2; kg++) {                                  // =================== Loop kg ===================
				r=xgaus[kg]; 
				
					phi_r[0] = -0.125*(1.0-s)*(1.0-t);
					phi_r[1] =  0.125*(1.0-s)*(1.0-t);
					phi_r[2] =  0.125*(1.0+s)*(1.0-t);
					phi_r[3] = -0.125*(1.0+s)*(1.0-t);
					phi_r[4] = -0.125*(1.0-s)*(1.0+t);
					phi_r[5] =  0.125*(1.0-s)*(1.0+t);
					phi_r[6] =  0.125*(1.0+s)*(1.0+t);
					phi_r[7] = -0.125*(1.0+s)*(1.0+t);
	 
					phi_s[0] = -0.125*(1.0-r)*(1.0-t);
					phi_s[1] = -0.125*(1.0+r)*(1.0-t);
					phi_s[2] =  0.125*(1.0+r)*(1.0-t);
					phi_s[3] =  0.125*(1.0-r)*(1.0-t);
					phi_s[4] = -0.125*(1.0-r)*(1.0+t);
					phi_s[5] = -0.125*(1.0+r)*(1.0+t);
					phi_s[6] =  0.125*(1.0+r)*(1.0+t);
					phi_s[7] =  0.125*(1.0-r)*(1.0+t);
		 
					phi_t[0] = -0.125*(1.0-r)*(1.0-s);
					phi_t[1] = -0.125*(1.0+r)*(1.0-s);
					phi_t[2] = -0.125*(1.0+r)*(1.0+s);
					phi_t[3] = -0.125*(1.0-r)*(1.0+s);
					phi_t[4] =  0.125*(1.0-r)*(1.0-s);
					phi_t[5] =  0.125*(1.0+r)*(1.0-s);
					phi_t[6] =  0.125*(1.0+r)*(1.0+s);
					phi_t[7] =  0.125*(1.0-r)*(1.0+s);
						
					jac[0][0]=phi_r[0]*XEL0+phi_r[1]*XEL1+phi_r[2]*XEL2+phi_r[3]*XEL3+phi_r[4]*XEL4+phi_r[5]*XEL5+phi_r[6]*XEL6+phi_r[7]*XEL7;
					jac[0][1]=phi_r[0]*YEL0+phi_r[1]*YEL1+phi_r[2]*YEL2+phi_r[3]*YEL3+phi_r[4]*YEL4+phi_r[5]*YEL5+phi_r[6]*YEL6+phi_r[7]*YEL7;
					jac[0][2]=phi_r[0]*ZEL0+phi_r[1]*ZEL1+phi_r[2]*ZEL2+phi_r[3]*ZEL3+phi_r[4]*ZEL4+phi_r[5]*ZEL5+phi_r[6]*ZEL6+phi_r[7]*ZEL7;

					jac[1][0]=phi_s[0]*XEL0+phi_s[1]*XEL1+phi_s[2]*XEL2+phi_s[3]*XEL3+phi_s[4]*XEL4+phi_s[5]*XEL5+phi_s[6]*XEL6+phi_s[7]*XEL7;
					jac[1][1]=phi_s[0]*YEL0+phi_s[1]*YEL1+phi_s[2]*YEL2+phi_s[3]*YEL3+phi_s[4]*YEL4+phi_s[5]*YEL5+phi_s[6]*YEL6+phi_s[7]*YEL7;
					jac[1][2]=phi_s[0]*ZEL0+phi_s[1]*ZEL1+phi_s[2]*ZEL2+phi_s[3]*ZEL3+phi_s[4]*ZEL4+phi_s[5]*ZEL5+phi_s[6]*ZEL6+phi_s[7]*ZEL7;
					
					jac[2][0]=phi_t[0]*XEL0+phi_t[1]*XEL1+phi_t[2]*XEL2+phi_t[3]*XEL3+phi_t[4]*XEL4+phi_t[5]*XEL5+phi_t[6]*XEL6+phi_t[7]*XEL7;
					jac[2][1]=phi_t[0]*YEL0+phi_t[1]*YEL1+phi_t[2]*YEL2+phi_t[3]*YEL3+phi_t[4]*YEL4+phi_t[5]*YEL5+phi_t[6]*YEL6+phi_t[7]*YEL7;
					jac[2][2]=phi_t[0]*ZEL0+phi_t[1]*ZEL1+phi_t[2]*ZEL2+phi_t[3]*ZEL3+phi_t[4]*ZEL4+phi_t[5]*ZEL5+phi_t[6]*ZEL6+phi_t[7]*ZEL7;
							
					jacob = jac[0][0]*(jac[1][1]*jac[2][2]-jac[1][2]*jac[2][1])-jac[0][1]*(jac[1][0]*jac[2][2]-jac[1][2]*jac[2][0])+
							jac[0][2]*(jac[1][0]*jac[2][1]-jac[1][1]*jac[2][0]);
					
					if(jacob > 0.) {
					
						gama[0][0] = (jac[1][1]*jac[2][2] - jac[1][2]*jac[2][1]) /jacob;
						gama[0][1] =-(jac[0][1]*jac[2][2] - jac[0][2]*jac[2][1]) /jacob;
						gama[0][2] = (jac[0][1]*jac[1][2] - jac[0][2]*jac[1][1]) /jacob;
						
						gama[1][0] =-(jac[1][0]*jac[2][2] - jac[1][2]*jac[2][0]) /jacob;
						gama[1][1] = (jac[0][0]*jac[2][2] - jac[0][2]*jac[2][0]) /jacob;
						gama[1][2] =-(jac[0][0]*jac[1][2] - jac[0][2]*jac[1][0]) /jacob;
						
						gama[2][0] = (jac[1][0]*jac[2][1] - jac[1][1]*jac[2][0]) /jacob;
						gama[2][1] =-(jac[0][0]*jac[2][1] - jac[0][1]*jac[2][0]) /jacob;
						gama[2][2] = (jac[0][0]*jac[1][1] - jac[0][1]*jac[1][0]) /jacob;
										
						auxX[0] = gama[0][0]*phi_r[0]+gama[0][1]*phi_s[0]+gama[0][2]*phi_t[0];
						auxY[0] = gama[1][0]*phi_r[0]+gama[1][1]*phi_s[0]+gama[1][2]*phi_t[0];
						auxZ[0] = gama[2][0]*phi_r[0]+gama[2][1]*phi_s[0]+gama[2][2]*phi_t[0];
							
						auxX[1] = gama[0][0]*phi_r[1]+gama[0][1]*phi_s[1]+gama[0][2]*phi_t[1];
						auxY[1] = gama[1][0]*phi_r[1]+gama[1][1]*phi_s[1]+gama[1][2]*phi_t[1];
						auxZ[1] = gama[2][0]*phi_r[1]+gama[2][1]*phi_s[1]+gama[2][2]*phi_t[1];
						
						auxX[2] = gama[0][0]*phi_r[2]+gama[0][1]*phi_s[2]+gama[0][2]*phi_t[2];
						auxY[2] = gama[1][0]*phi_r[2]+gama[1][1]*phi_s[2]+gama[1][2]*phi_t[2];
						auxZ[2] = gama[2][0]*phi_r[2]+gama[2][1]*phi_s[2]+gama[2][2]*phi_t[2];
						
						auxX[3] = gama[0][0]*phi_r[3]+gama[0][1]*phi_s[3]+gama[0][2]*phi_t[3];
						auxY[3] = gama[1][0]*phi_r[3]+gama[1][1]*phi_s[3]+gama[1][2]*phi_t[3];
						auxZ[3] = gama[2][0]*phi_r[3]+gama[2][1]*phi_s[3]+gama[2][2]*phi_t[3];
						
						auxX[4] = gama[0][0]*phi_r[4]+gama[0][1]*phi_s[4]+gama[0][2]*phi_t[4];
						auxY[4] = gama[1][0]*phi_r[4]+gama[1][1]*phi_s[4]+gama[1][2]*phi_t[4];
						auxZ[4] = gama[2][0]*phi_r[4]+gama[2][1]*phi_s[4]+gama[2][2]*phi_t[4];
							
						auxX[5] = gama[0][0]*phi_r[5]+gama[0][1]*phi_s[5]+gama[0][2]*phi_t[5];
						auxY[5] = gama[1][0]*phi_r[5]+gama[1][1]*phi_s[5]+gama[1][2]*phi_t[5];
						auxZ[5] = gama[2][0]*phi_r[5]+gama[2][1]*phi_s[5]+gama[2][2]*phi_t[5];
						
						auxX[6] = gama[0][0]*phi_r[6]+gama[0][1]*phi_s[6]+gama[0][2]*phi_t[6];
						auxY[6] = gama[1][0]*phi_r[6]+gama[1][1]*phi_s[6]+gama[1][2]*phi_t[6];
						auxZ[6] = gama[2][0]*phi_r[6]+gama[2][1]*phi_s[6]+gama[2][2]*phi_t[6];
						
						auxX[7] = gama[0][0]*phi_r[7]+gama[0][1]*phi_s[7]+gama[0][2]*phi_t[7];
						auxY[7] = gama[1][0]*phi_r[7]+gama[1][1]*phi_s[7]+gama[1][2]*phi_t[7];
						auxZ[7] = gama[2][0]*phi_r[7]+gama[2][1]*phi_s[7]+gama[2][2]*phi_t[7];
				
						__syncthreads();  // Line 0:
							
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+ 3] += jacob*(auxX[7]*C00*auxX[0] + auxY[7]*C33*auxY[0] + auxZ[7]*C33*auxZ[0]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+ 4] += jacob*(auxX[7]*C01*auxY[0] + auxY[7]*C33*auxX[0]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+ 5] += jacob*(auxX[7]*C01*auxZ[0] + auxZ[7]*C33*auxX[0]);
							
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+ 6] += jacob*(auxX[7]*C00*auxX[1] + auxY[7]*C33*auxY[1] + auxZ[7]*C33*auxZ[1]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+ 7] += jacob*(auxX[7]*C01*auxY[1] + auxY[7]*C33*auxX[1]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+ 8] += jacob*(auxX[7]*C01*auxZ[1] + auxZ[7]*C33*auxX[1]);
							
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+12] += jacob*(auxX[7]*C00*auxX[3] + auxY[7]*C33*auxY[3] + auxZ[7]*C33*auxZ[3]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+13] += jacob*(auxX[7]*C01*auxY[3] + auxY[7]*C33*auxX[3]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+14] += jacob*(auxX[7]*C01*auxZ[3] + auxZ[7]*C33*auxX[3]);
							
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+15] += jacob*(auxX[7]*C00*auxX[2] + auxY[7]*C33*auxY[2] + auxZ[7]*C33*auxZ[2]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+16] += jacob*(auxX[7]*C01*auxY[2] + auxY[7]*C33*auxX[2]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+17] += jacob*(auxX[7]*C01*auxZ[2] + auxZ[7]*C33*auxX[2]);
							
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+30] += jacob*(auxX[7]*C00*auxX[4] + auxY[7]*C33*auxY[4] + auxZ[7]*C33*auxZ[4]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+31] += jacob*(auxX[7]*C01*auxY[4] + auxY[7]*C33*auxX[4]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+32] += jacob*(auxX[7]*C01*auxZ[4] + auxZ[7]*C33*auxX[4]);
							
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+33] += jacob*(auxX[7]*C00*auxX[5] + auxY[7]*C33*auxY[5] + auxZ[7]*C33*auxZ[5]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+34] += jacob*(auxX[7]*C01*auxY[5] + auxY[7]*C33*auxX[5]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+35] += jacob*(auxX[7]*C01*auxZ[5] + auxZ[7]*C33*auxX[5]);
							
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+39] += jacob*(auxX[7]*C00*auxX[7] + auxY[7]*C33*auxY[7] + auxZ[7]*C33*auxZ[7]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+40] += jacob*(auxX[7]*C01*auxY[7] + auxY[7]*C33*auxX[7]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+41] += jacob*(auxX[7]*C01*auxZ[7] + auxZ[7]*C33*auxX[7]);
							
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+42] += jacob*(auxX[7]*C00*auxX[6] + auxY[7]*C33*auxY[6] + auxZ[7]*C33*auxZ[6]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+43] += jacob*(auxX[7]*C01*auxY[6] + auxY[7]*C33*auxX[6]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+44] += jacob*(auxX[7]*C01*auxZ[6] + auxZ[7]*C33*auxX[6]);
							
						__syncthreads();  // Line 1:
						
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+ 3] += jacob*(auxY[7]*C01*auxX[0] + auxX[7]*C33*auxY[0]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+ 4] += jacob*(auxY[7]*C00*auxY[0] + auxX[7]*C33*auxX[0] + auxZ[7]*C33*auxZ[0]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+ 5] += jacob*(auxY[7]*C01*auxZ[0] + auxZ[7]*C33*auxY[0]);
							
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+ 6] += jacob*(auxY[7]*C01*auxX[1] + auxX[7]*C33*auxY[1]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+ 7] += jacob*(auxY[7]*C00*auxY[1] + auxX[7]*C33*auxX[1] + auxZ[7]*C33*auxZ[1]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+ 8] += jacob*(auxY[7]*C01*auxZ[1] + auxZ[7]*C33*auxY[1]);
							
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+12] += jacob*(auxY[7]*C01*auxX[3] + auxX[7]*C33*auxY[3]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+13] += jacob*(auxY[7]*C00*auxY[3] + auxX[7]*C33*auxX[3] + auxZ[7]*C33*auxZ[3]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+14] += jacob*(auxY[7]*C01*auxZ[3] + auxZ[7]*C33*auxY[3]);
							
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+15] += jacob*(auxY[7]*C01*auxX[2] + auxX[7]*C33*auxY[2]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+16] += jacob*(auxY[7]*C00*auxY[2] + auxX[7]*C33*auxX[2] + auxZ[7]*C33*auxZ[2]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+17] += jacob*(auxY[7]*C01*auxZ[2] + auxZ[7]*C33*auxY[2]);
							
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+30] += jacob*(auxY[7]*C01*auxX[4] + auxX[7]*C33*auxY[4]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+31] += jacob*(auxY[7]*C00*auxY[4] + auxX[7]*C33*auxX[4] + auxZ[7]*C33*auxZ[4]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+32] += jacob*(auxY[7]*C01*auxZ[4] + auxZ[7]*C33*auxY[4]);
							
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+33] += jacob*(auxY[7]*C01*auxX[5] + auxX[7]*C33*auxY[5]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+34] += jacob*(auxY[7]*C00*auxY[5] + auxX[7]*C33*auxX[5] + auxZ[7]*C33*auxZ[5]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+35] += jacob*(auxY[7]*C01*auxZ[5] + auxZ[7]*C33*auxY[5]);
							
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+39] += jacob*(auxY[7]*C01*auxX[7] + auxX[7]*C33*auxY[7]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+40] += jacob*(auxY[7]*C00*auxY[7] + auxX[7]*C33*auxX[7] + auxZ[7]*C33*auxZ[7]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+41] += jacob*(auxY[7]*C01*auxZ[7] + auxZ[7]*C33*auxY[7]);
							
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+42] += jacob*(auxY[7]*C01*auxX[6] + auxX[7]*C33*auxY[6]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+43] += jacob*(auxY[7]*C00*auxY[6] + auxX[7]*C33*auxX[6] + auxZ[7]*C33*auxZ[6]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+44] += jacob*(auxY[7]*C01*auxZ[6] + auxZ[7]*C33*auxY[6]);
						
						__syncthreads();  // Line 2:
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+ 3] += jacob*(auxZ[7]*C01*auxX[0] + auxX[7]*C33*auxZ[0]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+ 4] += jacob*(auxZ[7]*C01*auxY[0] + auxY[7]*C33*auxZ[0]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+ 5] += jacob*(auxZ[7]*C00*auxZ[0] + auxY[7]*C33*auxY[0] + auxX[7]*C33*auxX[0]);
							
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+ 6] += jacob*(auxZ[7]*C01*auxX[1] + auxX[7]*C33*auxZ[1]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+ 7] += jacob*(auxZ[7]*C01*auxY[1] + auxY[7]*C33*auxZ[1]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+ 8] += jacob*(auxZ[7]*C00*auxZ[1] + auxY[7]*C33*auxY[1] + auxX[7]*C33*auxX[1]);
							
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+12] += jacob*(auxZ[7]*C01*auxX[3] + auxX[7]*C33*auxZ[3]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+13] += jacob*(auxZ[7]*C01*auxY[3] + auxY[7]*C33*auxZ[3]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+14] += jacob*(auxZ[7]*C00*auxZ[3] + auxY[7]*C33*auxY[3] + auxX[7]*C33*auxX[3]);
							
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+15] += jacob*(auxZ[7]*C01*auxX[2] + auxX[7]*C33*auxZ[2]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+16] += jacob*(auxZ[7]*C01*auxY[2] + auxY[7]*C33*auxZ[2]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+17] += jacob*(auxZ[7]*C00*auxZ[2] + auxY[7]*C33*auxY[2] + auxX[7]*C33*auxX[2]);
							
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+30] += jacob*(auxZ[7]*C01*auxX[4] + auxX[7]*C33*auxZ[4]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+31] += jacob*(auxZ[7]*C01*auxY[4] + auxY[7]*C33*auxZ[4]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+32] += jacob*(auxZ[7]*C00*auxZ[4] + auxY[7]*C33*auxY[4] + auxX[7]*C33*auxX[4]);
							
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+33] += jacob*(auxZ[7]*C01*auxX[5] + auxX[7]*C33*auxZ[5]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+34] += jacob*(auxZ[7]*C01*auxY[5] + auxY[7]*C33*auxZ[5]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+35] += jacob*(auxZ[7]*C00*auxZ[5] + auxY[7]*C33*auxY[5] + auxX[7]*C33*auxX[5]);
							
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+39] += jacob*(auxZ[7]*C01*auxX[7] + auxX[7]*C33*auxZ[7]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+40] += jacob*(auxZ[7]*C01*auxY[7] + auxY[7]*C33*auxZ[7]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+41] += jacob*(auxZ[7]*C00*auxZ[7] + auxY[7]*C33*auxY[7] + auxX[7]*C33*auxX[7]);
							
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+42] += jacob*(auxZ[7]*C01*auxX[6] + auxX[7]*C33*auxZ[6]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+43] += jacob*(auxZ[7]*C01*auxY[6] + auxY[7]*C33*auxZ[6]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+44] += jacob*(auxZ[7]*C00*auxZ[6] + auxY[7]*C33*auxY[6] + auxX[7]*C33*auxX[6]);

					}
					
				}                                                        // =================== Loop kg ===================
											
			}                                                            // =================== Loop jg ===================

		}                                                                // =================== Loop ig ===================
		
	}

}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============

__global__ void AssemblyStiffness3DQ3Mod(int numno, int RowSize, double *CoordQ3_d, double *MatPropQ3_d, double *Matrix_K_Aux_d) 
{
	double r, s, t;  // 3
	double XEL0, YEL0, ZEL0, XEL1, YEL1, ZEL1, XEL2, YEL2, ZEL2, XEL3, YEL3, ZEL3;  // 12 
	double XEL4, YEL4, ZEL4, XEL5, YEL5, ZEL5, XEL6, YEL6, ZEL6, XEL7, YEL7, ZEL7;  // 12 
	double xgaus[2];  // 2
	double p, E, C00, C01, C33;  // 5
	double phi_r[8], phi_s[8], phi_t[8];  // 24
	double jac[3][3];  // 9
	double jacob;  // 1
	double gama[3][3];  // 9
	double auxX[8], auxY[8], auxZ[8];  // 24
	int ig, jg, kg;  // 3
	
	const int xIndex = blockIdx.x*blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y*blockDim.y + threadIdx.y;
	const int thread_id = (gridDim.x*blockDim.x)*yIndex + xIndex;
	
	__syncthreads();
	
	if(thread_id < numno) {
	
		XEL0 = CoordQ3_d[thread_id+ 0*numno];
		YEL0 = CoordQ3_d[thread_id+ 1*numno];
		ZEL0 = CoordQ3_d[thread_id+ 2*numno];   	
		XEL1 = CoordQ3_d[thread_id+ 3*numno];
		YEL1 = CoordQ3_d[thread_id+ 4*numno];  
		ZEL1 = CoordQ3_d[thread_id+ 5*numno];  	
		XEL2 = CoordQ3_d[thread_id+ 6*numno];
		YEL2 = CoordQ3_d[thread_id+ 7*numno]; 
		ZEL2 = CoordQ3_d[thread_id+ 8*numno];  		
		XEL3 = CoordQ3_d[thread_id+ 9*numno];
		YEL3 = CoordQ3_d[thread_id+10*numno];
		ZEL3 = CoordQ3_d[thread_id+11*numno];
		
		XEL4 = CoordQ3_d[thread_id+12*numno];
		YEL4 = CoordQ3_d[thread_id+13*numno];
		ZEL4 = CoordQ3_d[thread_id+14*numno];   	
		XEL5 = CoordQ3_d[thread_id+15*numno];
		YEL5 = CoordQ3_d[thread_id+16*numno];  
		ZEL5 = CoordQ3_d[thread_id+17*numno];  	
		XEL6 = CoordQ3_d[thread_id+18*numno];
		YEL6 = CoordQ3_d[thread_id+19*numno]; 
		ZEL6 = CoordQ3_d[thread_id+20*numno];  		
		XEL7 = CoordQ3_d[thread_id+21*numno];
		YEL7 = CoordQ3_d[thread_id+22*numno];
		ZEL7 = CoordQ3_d[thread_id+23*numno];
		
		__syncthreads();

		xgaus[0]=-0.577350269189626;
		xgaus[1]= 0.577350269189626;
							
		E = MatPropQ3_d[thread_id+ 0*numno];
		p = MatPropQ3_d[thread_id+ 1*numno];
					
		C00=(E*(1-p))/((1+p)*(1-2*p));  
		C01=(E*(1-p))/((1+p)*(1-2*p))*(p/(1-p)); 
		C33=(E*(1-p))/((1+p)*(1-2*p))*((1-2*p)/(2*(1-p)));
			
		for(ig=0; ig<2; ig++) {                                          // =================== Loop ig ===================
			t=xgaus[ig];
					
			for(jg=0; jg<2; jg++) {                                      // =================== Loop jg ===================
				s=xgaus[jg]; 
				
				for(kg=0; kg<2; kg++) {                                  // =================== Loop kg ===================
				r=xgaus[kg]; 
				
					phi_r[0] = -0.125*(1.0-s)*(1.0-t);
					phi_r[1] =  0.125*(1.0-s)*(1.0-t);
					phi_r[2] =  0.125*(1.0+s)*(1.0-t);
					phi_r[3] = -0.125*(1.0+s)*(1.0-t);
					phi_r[4] = -0.125*(1.0-s)*(1.0+t);
					phi_r[5] =  0.125*(1.0-s)*(1.0+t);
					phi_r[6] =  0.125*(1.0+s)*(1.0+t);
					phi_r[7] = -0.125*(1.0+s)*(1.0+t);
	 
					phi_s[0] = -0.125*(1.0-r)*(1.0-t);
					phi_s[1] = -0.125*(1.0+r)*(1.0-t);
					phi_s[2] =  0.125*(1.0+r)*(1.0-t);
					phi_s[3] =  0.125*(1.0-r)*(1.0-t);
					phi_s[4] = -0.125*(1.0-r)*(1.0+t);
					phi_s[5] = -0.125*(1.0+r)*(1.0+t);
					phi_s[6] =  0.125*(1.0+r)*(1.0+t);
					phi_s[7] =  0.125*(1.0-r)*(1.0+t);
		 
					phi_t[0] = -0.125*(1.0-r)*(1.0-s);
					phi_t[1] = -0.125*(1.0+r)*(1.0-s);
					phi_t[2] = -0.125*(1.0+r)*(1.0+s);
					phi_t[3] = -0.125*(1.0-r)*(1.0+s);
					phi_t[4] =  0.125*(1.0-r)*(1.0-s);
					phi_t[5] =  0.125*(1.0+r)*(1.0-s);
					phi_t[6] =  0.125*(1.0+r)*(1.0+s);
					phi_t[7] =  0.125*(1.0-r)*(1.0+s);
						
					jac[0][0]=phi_r[0]*XEL0+phi_r[1]*XEL1+phi_r[2]*XEL2+phi_r[3]*XEL3+phi_r[4]*XEL4+phi_r[5]*XEL5+phi_r[6]*XEL6+phi_r[7]*XEL7;
					jac[0][1]=phi_r[0]*YEL0+phi_r[1]*YEL1+phi_r[2]*YEL2+phi_r[3]*YEL3+phi_r[4]*YEL4+phi_r[5]*YEL5+phi_r[6]*YEL6+phi_r[7]*YEL7;
					jac[0][2]=phi_r[0]*ZEL0+phi_r[1]*ZEL1+phi_r[2]*ZEL2+phi_r[3]*ZEL3+phi_r[4]*ZEL4+phi_r[5]*ZEL5+phi_r[6]*ZEL6+phi_r[7]*ZEL7;

					jac[1][0]=phi_s[0]*XEL0+phi_s[1]*XEL1+phi_s[2]*XEL2+phi_s[3]*XEL3+phi_s[4]*XEL4+phi_s[5]*XEL5+phi_s[6]*XEL6+phi_s[7]*XEL7;
					jac[1][1]=phi_s[0]*YEL0+phi_s[1]*YEL1+phi_s[2]*YEL2+phi_s[3]*YEL3+phi_s[4]*YEL4+phi_s[5]*YEL5+phi_s[6]*YEL6+phi_s[7]*YEL7;
					jac[1][2]=phi_s[0]*ZEL0+phi_s[1]*ZEL1+phi_s[2]*ZEL2+phi_s[3]*ZEL3+phi_s[4]*ZEL4+phi_s[5]*ZEL5+phi_s[6]*ZEL6+phi_s[7]*ZEL7;
					
					jac[2][0]=phi_t[0]*XEL0+phi_t[1]*XEL1+phi_t[2]*XEL2+phi_t[3]*XEL3+phi_t[4]*XEL4+phi_t[5]*XEL5+phi_t[6]*XEL6+phi_t[7]*XEL7;
					jac[2][1]=phi_t[0]*YEL0+phi_t[1]*YEL1+phi_t[2]*YEL2+phi_t[3]*YEL3+phi_t[4]*YEL4+phi_t[5]*YEL5+phi_t[6]*YEL6+phi_t[7]*YEL7;
					jac[2][2]=phi_t[0]*ZEL0+phi_t[1]*ZEL1+phi_t[2]*ZEL2+phi_t[3]*ZEL3+phi_t[4]*ZEL4+phi_t[5]*ZEL5+phi_t[6]*ZEL6+phi_t[7]*ZEL7;
							
					jacob = jac[0][0]*(jac[1][1]*jac[2][2]-jac[1][2]*jac[2][1])-jac[0][1]*(jac[1][0]*jac[2][2]-jac[1][2]*jac[2][0])+
							jac[0][2]*(jac[1][0]*jac[2][1]-jac[1][1]*jac[2][0]);
					
					if(jacob > 0.) {
					
						gama[0][0] = (jac[1][1]*jac[2][2] - jac[1][2]*jac[2][1]) /jacob;
						gama[0][1] =-(jac[0][1]*jac[2][2] - jac[0][2]*jac[2][1]) /jacob;
						gama[0][2] = (jac[0][1]*jac[1][2] - jac[0][2]*jac[1][1]) /jacob;
						
						gama[1][0] =-(jac[1][0]*jac[2][2] - jac[1][2]*jac[2][0]) /jacob;
						gama[1][1] = (jac[0][0]*jac[2][2] - jac[0][2]*jac[2][0]) /jacob;
						gama[1][2] =-(jac[0][0]*jac[1][2] - jac[0][2]*jac[1][0]) /jacob;
						
						gama[2][0] = (jac[1][0]*jac[2][1] - jac[1][1]*jac[2][0]) /jacob;
						gama[2][1] =-(jac[0][0]*jac[2][1] - jac[0][1]*jac[2][0]) /jacob;
						gama[2][2] = (jac[0][0]*jac[1][1] - jac[0][1]*jac[1][0]) /jacob;
										
						auxX[0] = gama[0][0]*phi_r[0]+gama[0][1]*phi_s[0]+gama[0][2]*phi_t[0];
						auxY[0] = gama[1][0]*phi_r[0]+gama[1][1]*phi_s[0]+gama[1][2]*phi_t[0];
						auxZ[0] = gama[2][0]*phi_r[0]+gama[2][1]*phi_s[0]+gama[2][2]*phi_t[0];
							
						auxX[1] = gama[0][0]*phi_r[1]+gama[0][1]*phi_s[1]+gama[0][2]*phi_t[1];
						auxY[1] = gama[1][0]*phi_r[1]+gama[1][1]*phi_s[1]+gama[1][2]*phi_t[1];
						auxZ[1] = gama[2][0]*phi_r[1]+gama[2][1]*phi_s[1]+gama[2][2]*phi_t[1];
						
						auxX[2] = gama[0][0]*phi_r[2]+gama[0][1]*phi_s[2]+gama[0][2]*phi_t[2];
						auxY[2] = gama[1][0]*phi_r[2]+gama[1][1]*phi_s[2]+gama[1][2]*phi_t[2];
						auxZ[2] = gama[2][0]*phi_r[2]+gama[2][1]*phi_s[2]+gama[2][2]*phi_t[2];
						
						auxX[3] = gama[0][0]*phi_r[3]+gama[0][1]*phi_s[3]+gama[0][2]*phi_t[3];
						auxY[3] = gama[1][0]*phi_r[3]+gama[1][1]*phi_s[3]+gama[1][2]*phi_t[3];
						auxZ[3] = gama[2][0]*phi_r[3]+gama[2][1]*phi_s[3]+gama[2][2]*phi_t[3];
						
						auxX[4] = gama[0][0]*phi_r[4]+gama[0][1]*phi_s[4]+gama[0][2]*phi_t[4];
						auxY[4] = gama[1][0]*phi_r[4]+gama[1][1]*phi_s[4]+gama[1][2]*phi_t[4];
						auxZ[4] = gama[2][0]*phi_r[4]+gama[2][1]*phi_s[4]+gama[2][2]*phi_t[4];
							
						auxX[5] = gama[0][0]*phi_r[5]+gama[0][1]*phi_s[5]+gama[0][2]*phi_t[5];
						auxY[5] = gama[1][0]*phi_r[5]+gama[1][1]*phi_s[5]+gama[1][2]*phi_t[5];
						auxZ[5] = gama[2][0]*phi_r[5]+gama[2][1]*phi_s[5]+gama[2][2]*phi_t[5];
						
						auxX[6] = gama[0][0]*phi_r[6]+gama[0][1]*phi_s[6]+gama[0][2]*phi_t[6];
						auxY[6] = gama[1][0]*phi_r[6]+gama[1][1]*phi_s[6]+gama[1][2]*phi_t[6];
						auxZ[6] = gama[2][0]*phi_r[6]+gama[2][1]*phi_s[6]+gama[2][2]*phi_t[6];
						
						auxX[7] = gama[0][0]*phi_r[7]+gama[0][1]*phi_s[7]+gama[0][2]*phi_t[7];
						auxY[7] = gama[1][0]*phi_r[7]+gama[1][1]*phi_s[7]+gama[1][2]*phi_t[7];
						auxZ[7] = gama[2][0]*phi_r[7]+gama[2][1]*phi_s[7]+gama[2][2]*phi_t[7];
						
						__syncthreads();  // Line 0:
							
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+ 9] += jacob*(auxX[5]*C00*auxX[0] + auxY[5]*C33*auxY[0] + auxZ[5]*C33*auxZ[0]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+10] += jacob*(auxX[5]*C01*auxY[0] + auxY[5]*C33*auxX[0]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+11] += jacob*(auxX[5]*C01*auxZ[0] + auxZ[5]*C33*auxX[0]);
							
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+12] += jacob*(auxX[5]*C00*auxX[1] + auxY[5]*C33*auxY[1] + auxZ[5]*C33*auxZ[1]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+13] += jacob*(auxX[5]*C01*auxY[1] + auxY[5]*C33*auxX[1]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+14] += jacob*(auxX[5]*C01*auxZ[1] + auxZ[5]*C33*auxX[1]);
							
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+18] += jacob*(auxX[5]*C00*auxX[3] + auxY[5]*C33*auxY[3] + auxZ[5]*C33*auxZ[3]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+19] += jacob*(auxX[5]*C01*auxY[3] + auxY[5]*C33*auxX[3]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+20] += jacob*(auxX[5]*C01*auxZ[3] + auxZ[5]*C33*auxX[3]);
							
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+21] += jacob*(auxX[5]*C00*auxX[2] + auxY[5]*C33*auxY[2] + auxZ[5]*C33*auxZ[2]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+22] += jacob*(auxX[5]*C01*auxY[2] + auxY[5]*C33*auxX[2]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+23] += jacob*(auxX[5]*C01*auxZ[2] + auxZ[5]*C33*auxX[2]);
							
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+36] += jacob*(auxX[5]*C00*auxX[4] + auxY[5]*C33*auxY[4] + auxZ[5]*C33*auxZ[4]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+37] += jacob*(auxX[5]*C01*auxY[4] + auxY[5]*C33*auxX[4]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+38] += jacob*(auxX[5]*C01*auxZ[4] + auxZ[5]*C33*auxX[4]);
							
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+39] += jacob*(auxX[5]*C00*auxX[5] + auxY[5]*C33*auxY[5] + auxZ[5]*C33*auxZ[5]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+40] += jacob*(auxX[5]*C01*auxY[5] + auxY[5]*C33*auxX[5]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+41] += jacob*(auxX[5]*C01*auxZ[5] + auxZ[5]*C33*auxX[5]);
							
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+45] += jacob*(auxX[5]*C00*auxX[7] + auxY[5]*C33*auxY[7] + auxZ[5]*C33*auxZ[7]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+46] += jacob*(auxX[5]*C01*auxY[7] + auxY[5]*C33*auxX[7]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+47] += jacob*(auxX[5]*C01*auxZ[7] + auxZ[5]*C33*auxX[7]);
							
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+48] += jacob*(auxX[5]*C00*auxX[6] + auxY[5]*C33*auxY[6] + auxZ[5]*C33*auxZ[6]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+49] += jacob*(auxX[5]*C01*auxY[6] + auxY[5]*C33*auxX[6]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+50] += jacob*(auxX[5]*C01*auxZ[6] + auxZ[5]*C33*auxX[6]);
							
						__syncthreads();  // Line 1:
						
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+ 9] += jacob*(auxY[5]*C01*auxX[0] + auxX[5]*C33*auxY[0]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+10] += jacob*(auxY[5]*C00*auxY[0] + auxX[5]*C33*auxX[0] + auxZ[5]*C33*auxZ[0]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+11] += jacob*(auxY[5]*C01*auxZ[0] + auxZ[5]*C33*auxY[0]);
							
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+12] += jacob*(auxY[5]*C01*auxX[1] + auxX[5]*C33*auxY[1]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+13] += jacob*(auxY[5]*C00*auxY[1] + auxX[5]*C33*auxX[1] + auxZ[5]*C33*auxZ[1]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+14] += jacob*(auxY[5]*C01*auxZ[1] + auxZ[5]*C33*auxY[1]);
							
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+18] += jacob*(auxY[5]*C01*auxX[3] + auxX[5]*C33*auxY[3]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+19] += jacob*(auxY[5]*C00*auxY[3] + auxX[5]*C33*auxX[3] + auxZ[5]*C33*auxZ[3]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+20] += jacob*(auxY[5]*C01*auxZ[3] + auxZ[5]*C33*auxY[3]);
							
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+21] += jacob*(auxY[5]*C01*auxX[2] + auxX[5]*C33*auxY[2]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+22] += jacob*(auxY[5]*C00*auxY[2] + auxX[5]*C33*auxX[2] + auxZ[5]*C33*auxZ[2]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+23] += jacob*(auxY[5]*C01*auxZ[2] + auxZ[5]*C33*auxY[2]);
							
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+36] += jacob*(auxY[5]*C01*auxX[4] + auxX[5]*C33*auxY[4]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+37] += jacob*(auxY[5]*C00*auxY[4] + auxX[5]*C33*auxX[4] + auxZ[5]*C33*auxZ[4]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+38] += jacob*(auxY[5]*C01*auxZ[4] + auxZ[5]*C33*auxY[4]);
							
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+39] += jacob*(auxY[5]*C01*auxX[5] + auxX[5]*C33*auxY[5]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+40] += jacob*(auxY[5]*C00*auxY[5] + auxX[5]*C33*auxX[5] + auxZ[5]*C33*auxZ[5]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+41] += jacob*(auxY[5]*C01*auxZ[5] + auxZ[5]*C33*auxY[5]);
							
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+45] += jacob*(auxY[5]*C01*auxX[7] + auxX[5]*C33*auxY[7]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+46] += jacob*(auxY[5]*C00*auxY[7] + auxX[5]*C33*auxX[7] + auxZ[5]*C33*auxZ[7]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+47] += jacob*(auxY[5]*C01*auxZ[7] + auxZ[5]*C33*auxY[7]);
							
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+48] += jacob*(auxY[5]*C01*auxX[6] + auxX[5]*C33*auxY[6]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+49] += jacob*(auxY[5]*C00*auxY[6] + auxX[5]*C33*auxX[6] + auxZ[5]*C33*auxZ[6]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+50] += jacob*(auxY[5]*C01*auxZ[6] + auxZ[5]*C33*auxY[6]);
						
						__syncthreads();  // Line 2:
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+ 9] += jacob*(auxZ[5]*C01*auxX[0] + auxX[5]*C33*auxZ[0]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+10] += jacob*(auxZ[5]*C01*auxY[0] + auxY[5]*C33*auxZ[0]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+11] += jacob*(auxZ[5]*C00*auxZ[0] + auxY[5]*C33*auxY[0] + auxX[5]*C33*auxX[0]);
							
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+12] += jacob*(auxZ[5]*C01*auxX[1] + auxX[5]*C33*auxZ[1]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+13] += jacob*(auxZ[5]*C01*auxY[1] + auxY[5]*C33*auxZ[1]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+14] += jacob*(auxZ[5]*C00*auxZ[1] + auxY[5]*C33*auxY[1] + auxX[5]*C33*auxX[1]);
							
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+18] += jacob*(auxZ[5]*C01*auxX[3] + auxX[5]*C33*auxZ[3]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+19] += jacob*(auxZ[5]*C01*auxY[3] + auxY[5]*C33*auxZ[3]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+20] += jacob*(auxZ[5]*C00*auxZ[3] + auxY[5]*C33*auxY[3] + auxX[5]*C33*auxX[3]);
							
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+21] += jacob*(auxZ[5]*C01*auxX[2] + auxX[5]*C33*auxZ[2]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+22] += jacob*(auxZ[5]*C01*auxY[2] + auxY[5]*C33*auxZ[2]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+23] += jacob*(auxZ[5]*C00*auxZ[2] + auxY[5]*C33*auxY[2] + auxX[5]*C33*auxX[2]);
							
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+36] += jacob*(auxZ[5]*C01*auxX[4] + auxX[5]*C33*auxZ[4]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+37] += jacob*(auxZ[5]*C01*auxY[4] + auxY[5]*C33*auxZ[4]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+38] += jacob*(auxZ[5]*C00*auxZ[4] + auxY[5]*C33*auxY[4] + auxX[5]*C33*auxX[4]);
							
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+39] += jacob*(auxZ[5]*C01*auxX[5] + auxX[5]*C33*auxZ[5]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+40] += jacob*(auxZ[5]*C01*auxY[5] + auxY[5]*C33*auxZ[5]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+41] += jacob*(auxZ[5]*C00*auxZ[5] + auxY[5]*C33*auxY[5] + auxX[5]*C33*auxX[5]);
							
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+45] += jacob*(auxZ[5]*C01*auxX[7] + auxX[5]*C33*auxZ[7]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+46] += jacob*(auxZ[5]*C01*auxY[7] + auxY[5]*C33*auxZ[7]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+47] += jacob*(auxZ[5]*C00*auxZ[7] + auxY[5]*C33*auxY[7] + auxX[5]*C33*auxX[7]);
							
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+48] += jacob*(auxZ[5]*C01*auxX[6] + auxX[5]*C33*auxZ[6]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+49] += jacob*(auxZ[5]*C01*auxY[6] + auxY[5]*C33*auxZ[6]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+50] += jacob*(auxZ[5]*C00*auxZ[6] + auxY[5]*C33*auxY[6] + auxX[5]*C33*auxX[6]);

					}
					
				}                                                        // =================== Loop kg ===================
											
			}                                                            // =================== Loop jg ===================

		}                                                                // =================== Loop ig ===================
		
	}
	
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============

__global__ void AssemblyStiffness3DQ4Mod(int numno, int RowSize, double *CoordQ4_d, double *MatPropQ4_d, double *Matrix_K_Aux_d) 
{
	double r, s, t;  // 3
	double XEL0, YEL0, ZEL0, XEL1, YEL1, ZEL1, XEL2, YEL2, ZEL2, XEL3, YEL3, ZEL3;  // 12 
	double XEL4, YEL4, ZEL4, XEL5, YEL5, ZEL5, XEL6, YEL6, ZEL6, XEL7, YEL7, ZEL7;  // 12 
	double xgaus[2];  // 2
	double p, E, C00, C01, C33;  // 5
	double phi_r[8], phi_s[8], phi_t[8];  // 24
	double jac[3][3];  // 9
	double jacob;  // 1
	double gama[3][3];  // 9
	double auxX[8], auxY[8], auxZ[8];  // 24
	int ig, jg, kg;  // 3
	
	const int xIndex = blockIdx.x*blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y*blockDim.y + threadIdx.y;
	const int thread_id = (gridDim.x*blockDim.x)*yIndex + xIndex;
	
	__syncthreads();
	
	if(thread_id < numno) {
	
		XEL0 = CoordQ4_d[thread_id+ 0*numno];
		YEL0 = CoordQ4_d[thread_id+ 1*numno];
		ZEL0 = CoordQ4_d[thread_id+ 2*numno];   	
		XEL1 = CoordQ4_d[thread_id+ 3*numno];
		YEL1 = CoordQ4_d[thread_id+ 4*numno];  
		ZEL1 = CoordQ4_d[thread_id+ 5*numno];  	
		XEL2 = CoordQ4_d[thread_id+ 6*numno];
		YEL2 = CoordQ4_d[thread_id+ 7*numno]; 
		ZEL2 = CoordQ4_d[thread_id+ 8*numno];  		
		XEL3 = CoordQ4_d[thread_id+ 9*numno];
		YEL3 = CoordQ4_d[thread_id+10*numno];
		ZEL3 = CoordQ4_d[thread_id+11*numno];
		
		XEL4 = CoordQ4_d[thread_id+12*numno];
		YEL4 = CoordQ4_d[thread_id+13*numno];
		ZEL4 = CoordQ4_d[thread_id+14*numno];   	
		XEL5 = CoordQ4_d[thread_id+15*numno];
		YEL5 = CoordQ4_d[thread_id+16*numno];  
		ZEL5 = CoordQ4_d[thread_id+17*numno];  	
		XEL6 = CoordQ4_d[thread_id+18*numno];
		YEL6 = CoordQ4_d[thread_id+19*numno]; 
		ZEL6 = CoordQ4_d[thread_id+20*numno];  		
		XEL7 = CoordQ4_d[thread_id+21*numno];
		YEL7 = CoordQ4_d[thread_id+22*numno];
		ZEL7 = CoordQ4_d[thread_id+23*numno];
		
		__syncthreads();

		xgaus[0]=-0.577350269189626;
		xgaus[1]= 0.577350269189626;
							
		E = MatPropQ4_d[thread_id+ 0*numno];
		p = MatPropQ4_d[thread_id+ 1*numno];
					
		C00=(E*(1-p))/((1+p)*(1-2*p));  
		C01=(E*(1-p))/((1+p)*(1-2*p))*(p/(1-p)); 
		C33=(E*(1-p))/((1+p)*(1-2*p))*((1-2*p)/(2*(1-p)));
			
		for(ig=0; ig<2; ig++) {                                          // =================== Loop ig ===================
			t=xgaus[ig];
					
			for(jg=0; jg<2; jg++) {                                      // =================== Loop jg ===================
				s=xgaus[jg]; 
				
				for(kg=0; kg<2; kg++) {                                  // =================== Loop kg ===================
				r=xgaus[kg]; 
				
					phi_r[0] = -0.125*(1.0-s)*(1.0-t);
					phi_r[1] =  0.125*(1.0-s)*(1.0-t);
					phi_r[2] =  0.125*(1.0+s)*(1.0-t);
					phi_r[3] = -0.125*(1.0+s)*(1.0-t);
					phi_r[4] = -0.125*(1.0-s)*(1.0+t);
					phi_r[5] =  0.125*(1.0-s)*(1.0+t);
					phi_r[6] =  0.125*(1.0+s)*(1.0+t);
					phi_r[7] = -0.125*(1.0+s)*(1.0+t);
	 
					phi_s[0] = -0.125*(1.0-r)*(1.0-t);
					phi_s[1] = -0.125*(1.0+r)*(1.0-t);
					phi_s[2] =  0.125*(1.0+r)*(1.0-t);
					phi_s[3] =  0.125*(1.0-r)*(1.0-t);
					phi_s[4] = -0.125*(1.0-r)*(1.0+t);
					phi_s[5] = -0.125*(1.0+r)*(1.0+t);
					phi_s[6] =  0.125*(1.0+r)*(1.0+t);
					phi_s[7] =  0.125*(1.0-r)*(1.0+t);
		 
					phi_t[0] = -0.125*(1.0-r)*(1.0-s);
					phi_t[1] = -0.125*(1.0+r)*(1.0-s);
					phi_t[2] = -0.125*(1.0+r)*(1.0+s);
					phi_t[3] = -0.125*(1.0-r)*(1.0+s);
					phi_t[4] =  0.125*(1.0-r)*(1.0-s);
					phi_t[5] =  0.125*(1.0+r)*(1.0-s);
					phi_t[6] =  0.125*(1.0+r)*(1.0+s);
					phi_t[7] =  0.125*(1.0-r)*(1.0+s);
						
					jac[0][0]=phi_r[0]*XEL0+phi_r[1]*XEL1+phi_r[2]*XEL2+phi_r[3]*XEL3+phi_r[4]*XEL4+phi_r[5]*XEL5+phi_r[6]*XEL6+phi_r[7]*XEL7;
					jac[0][1]=phi_r[0]*YEL0+phi_r[1]*YEL1+phi_r[2]*YEL2+phi_r[3]*YEL3+phi_r[4]*YEL4+phi_r[5]*YEL5+phi_r[6]*YEL6+phi_r[7]*YEL7;
					jac[0][2]=phi_r[0]*ZEL0+phi_r[1]*ZEL1+phi_r[2]*ZEL2+phi_r[3]*ZEL3+phi_r[4]*ZEL4+phi_r[5]*ZEL5+phi_r[6]*ZEL6+phi_r[7]*ZEL7;

					jac[1][0]=phi_s[0]*XEL0+phi_s[1]*XEL1+phi_s[2]*XEL2+phi_s[3]*XEL3+phi_s[4]*XEL4+phi_s[5]*XEL5+phi_s[6]*XEL6+phi_s[7]*XEL7;
					jac[1][1]=phi_s[0]*YEL0+phi_s[1]*YEL1+phi_s[2]*YEL2+phi_s[3]*YEL3+phi_s[4]*YEL4+phi_s[5]*YEL5+phi_s[6]*YEL6+phi_s[7]*YEL7;
					jac[1][2]=phi_s[0]*ZEL0+phi_s[1]*ZEL1+phi_s[2]*ZEL2+phi_s[3]*ZEL3+phi_s[4]*ZEL4+phi_s[5]*ZEL5+phi_s[6]*ZEL6+phi_s[7]*ZEL7;
					
					jac[2][0]=phi_t[0]*XEL0+phi_t[1]*XEL1+phi_t[2]*XEL2+phi_t[3]*XEL3+phi_t[4]*XEL4+phi_t[5]*XEL5+phi_t[6]*XEL6+phi_t[7]*XEL7;
					jac[2][1]=phi_t[0]*YEL0+phi_t[1]*YEL1+phi_t[2]*YEL2+phi_t[3]*YEL3+phi_t[4]*YEL4+phi_t[5]*YEL5+phi_t[6]*YEL6+phi_t[7]*YEL7;
					jac[2][2]=phi_t[0]*ZEL0+phi_t[1]*ZEL1+phi_t[2]*ZEL2+phi_t[3]*ZEL3+phi_t[4]*ZEL4+phi_t[5]*ZEL5+phi_t[6]*ZEL6+phi_t[7]*ZEL7;
							
					jacob = jac[0][0]*(jac[1][1]*jac[2][2]-jac[1][2]*jac[2][1])-jac[0][1]*(jac[1][0]*jac[2][2]-jac[1][2]*jac[2][0])+
							jac[0][2]*(jac[1][0]*jac[2][1]-jac[1][1]*jac[2][0]);
					
					if(jacob > 0.) {
					
						gama[0][0] = (jac[1][1]*jac[2][2] - jac[1][2]*jac[2][1]) /jacob;
						gama[0][1] =-(jac[0][1]*jac[2][2] - jac[0][2]*jac[2][1]) /jacob;
						gama[0][2] = (jac[0][1]*jac[1][2] - jac[0][2]*jac[1][1]) /jacob;
						
						gama[1][0] =-(jac[1][0]*jac[2][2] - jac[1][2]*jac[2][0]) /jacob;
						gama[1][1] = (jac[0][0]*jac[2][2] - jac[0][2]*jac[2][0]) /jacob;
						gama[1][2] =-(jac[0][0]*jac[1][2] - jac[0][2]*jac[1][0]) /jacob;
						
						gama[2][0] = (jac[1][0]*jac[2][1] - jac[1][1]*jac[2][0]) /jacob;
						gama[2][1] =-(jac[0][0]*jac[2][1] - jac[0][1]*jac[2][0]) /jacob;
						gama[2][2] = (jac[0][0]*jac[1][1] - jac[0][1]*jac[1][0]) /jacob;
										
						auxX[0] = gama[0][0]*phi_r[0]+gama[0][1]*phi_s[0]+gama[0][2]*phi_t[0];
						auxY[0] = gama[1][0]*phi_r[0]+gama[1][1]*phi_s[0]+gama[1][2]*phi_t[0];
						auxZ[0] = gama[2][0]*phi_r[0]+gama[2][1]*phi_s[0]+gama[2][2]*phi_t[0];
							
						auxX[1] = gama[0][0]*phi_r[1]+gama[0][1]*phi_s[1]+gama[0][2]*phi_t[1];
						auxY[1] = gama[1][0]*phi_r[1]+gama[1][1]*phi_s[1]+gama[1][2]*phi_t[1];
						auxZ[1] = gama[2][0]*phi_r[1]+gama[2][1]*phi_s[1]+gama[2][2]*phi_t[1];
						
						auxX[2] = gama[0][0]*phi_r[2]+gama[0][1]*phi_s[2]+gama[0][2]*phi_t[2];
						auxY[2] = gama[1][0]*phi_r[2]+gama[1][1]*phi_s[2]+gama[1][2]*phi_t[2];
						auxZ[2] = gama[2][0]*phi_r[2]+gama[2][1]*phi_s[2]+gama[2][2]*phi_t[2];
						
						auxX[3] = gama[0][0]*phi_r[3]+gama[0][1]*phi_s[3]+gama[0][2]*phi_t[3];
						auxY[3] = gama[1][0]*phi_r[3]+gama[1][1]*phi_s[3]+gama[1][2]*phi_t[3];
						auxZ[3] = gama[2][0]*phi_r[3]+gama[2][1]*phi_s[3]+gama[2][2]*phi_t[3];
						
						auxX[4] = gama[0][0]*phi_r[4]+gama[0][1]*phi_s[4]+gama[0][2]*phi_t[4];
						auxY[4] = gama[1][0]*phi_r[4]+gama[1][1]*phi_s[4]+gama[1][2]*phi_t[4];
						auxZ[4] = gama[2][0]*phi_r[4]+gama[2][1]*phi_s[4]+gama[2][2]*phi_t[4];
							
						auxX[5] = gama[0][0]*phi_r[5]+gama[0][1]*phi_s[5]+gama[0][2]*phi_t[5];
						auxY[5] = gama[1][0]*phi_r[5]+gama[1][1]*phi_s[5]+gama[1][2]*phi_t[5];
						auxZ[5] = gama[2][0]*phi_r[5]+gama[2][1]*phi_s[5]+gama[2][2]*phi_t[5];
						
						auxX[6] = gama[0][0]*phi_r[6]+gama[0][1]*phi_s[6]+gama[0][2]*phi_t[6];
						auxY[6] = gama[1][0]*phi_r[6]+gama[1][1]*phi_s[6]+gama[1][2]*phi_t[6];
						auxZ[6] = gama[2][0]*phi_r[6]+gama[2][1]*phi_s[6]+gama[2][2]*phi_t[6];
						
						auxX[7] = gama[0][0]*phi_r[7]+gama[0][1]*phi_s[7]+gama[0][2]*phi_t[7];
						auxY[7] = gama[1][0]*phi_r[7]+gama[1][1]*phi_s[7]+gama[1][2]*phi_t[7];
						auxZ[7] = gama[2][0]*phi_r[7]+gama[2][1]*phi_s[7]+gama[2][2]*phi_t[7];
											
						__syncthreads();  // Line 0:
							
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+12] += jacob*(auxX[4]*C00*auxX[0] + auxY[4]*C33*auxY[0] + auxZ[4]*C33*auxZ[0]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+13] += jacob*(auxX[4]*C01*auxY[0] + auxY[4]*C33*auxX[0]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+14] += jacob*(auxX[4]*C01*auxZ[0] + auxZ[4]*C33*auxX[0]);
							
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+15] += jacob*(auxX[4]*C00*auxX[1] + auxY[4]*C33*auxY[1] + auxZ[4]*C33*auxZ[1]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+16] += jacob*(auxX[4]*C01*auxY[1] + auxY[4]*C33*auxX[1]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+17] += jacob*(auxX[4]*C01*auxZ[1] + auxZ[4]*C33*auxX[1]);
							
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+21] += jacob*(auxX[4]*C00*auxX[3] + auxY[4]*C33*auxY[3] + auxZ[4]*C33*auxZ[3]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+22] += jacob*(auxX[4]*C01*auxY[3] + auxY[4]*C33*auxX[3]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+23] += jacob*(auxX[4]*C01*auxZ[3] + auxZ[4]*C33*auxX[3]);
							
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+24] += jacob*(auxX[4]*C00*auxX[2] + auxY[4]*C33*auxY[2] + auxZ[4]*C33*auxZ[2]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+25] += jacob*(auxX[4]*C01*auxY[2] + auxY[4]*C33*auxX[2]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+26] += jacob*(auxX[4]*C01*auxZ[2] + auxZ[4]*C33*auxX[2]);
							
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+39] += jacob*(auxX[4]*C00*auxX[4] + auxY[4]*C33*auxY[4] + auxZ[4]*C33*auxZ[4]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+40] += jacob*(auxX[4]*C01*auxY[4] + auxY[4]*C33*auxX[4]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+41] += jacob*(auxX[4]*C01*auxZ[4] + auxZ[4]*C33*auxX[4]);
							
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+42] += jacob*(auxX[4]*C00*auxX[5] + auxY[4]*C33*auxY[5] + auxZ[4]*C33*auxZ[5]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+43] += jacob*(auxX[4]*C01*auxY[5] + auxY[4]*C33*auxX[5]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+44] += jacob*(auxX[4]*C01*auxZ[5] + auxZ[4]*C33*auxX[5]);
							
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+48] += jacob*(auxX[4]*C00*auxX[7] + auxY[4]*C33*auxY[7] + auxZ[4]*C33*auxZ[7]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+49] += jacob*(auxX[4]*C01*auxY[7] + auxY[4]*C33*auxX[7]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+50] += jacob*(auxX[4]*C01*auxZ[7] + auxZ[4]*C33*auxX[7]);
							
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+51] += jacob*(auxX[4]*C00*auxX[6] + auxY[4]*C33*auxY[6] + auxZ[4]*C33*auxZ[6]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+52] += jacob*(auxX[4]*C01*auxY[6] + auxY[4]*C33*auxX[6]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+53] += jacob*(auxX[4]*C01*auxZ[6] + auxZ[4]*C33*auxX[6]);
						
						__syncthreads();  // Line 1:
							
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+12] += jacob*(auxY[4]*C01*auxX[0] + auxX[4]*C33*auxY[0]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+13] += jacob*(auxY[4]*C00*auxY[0] + auxX[4]*C33*auxX[0] + auxZ[4]*C33*auxZ[0]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+14] += jacob*(auxY[4]*C01*auxZ[0] + auxZ[4]*C33*auxY[0]);
							
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+15] += jacob*(auxY[4]*C01*auxX[1] + auxX[4]*C33*auxY[1]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+16] += jacob*(auxY[4]*C00*auxY[1] + auxX[4]*C33*auxX[1] + auxZ[4]*C33*auxZ[1]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+17] += jacob*(auxY[4]*C01*auxZ[1] + auxZ[4]*C33*auxY[1]);
							
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+21] += jacob*(auxY[4]*C01*auxX[3] + auxX[4]*C33*auxY[3]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+22] += jacob*(auxY[4]*C00*auxY[3] + auxX[4]*C33*auxX[3] + auxZ[4]*C33*auxZ[3]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+23] += jacob*(auxY[4]*C01*auxZ[3] + auxZ[4]*C33*auxY[3]);
							
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+24] += jacob*(auxY[4]*C01*auxX[2] + auxX[4]*C33*auxY[2]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+25] += jacob*(auxY[4]*C00*auxY[2] + auxX[4]*C33*auxX[2] + auxZ[4]*C33*auxZ[2]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+26] += jacob*(auxY[4]*C01*auxZ[2] + auxZ[4]*C33*auxY[2]);
							
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+39] += jacob*(auxY[4]*C01*auxX[4] + auxX[4]*C33*auxY[4]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+40] += jacob*(auxY[4]*C00*auxY[4] + auxX[4]*C33*auxX[4] + auxZ[4]*C33*auxZ[4]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+41] += jacob*(auxY[4]*C01*auxZ[4] + auxZ[4]*C33*auxY[4]);
							
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+42] += jacob*(auxY[4]*C01*auxX[5] + auxX[4]*C33*auxY[5]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+43] += jacob*(auxY[4]*C00*auxY[5] + auxX[4]*C33*auxX[5] + auxZ[4]*C33*auxZ[5]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+44] += jacob*(auxY[4]*C01*auxZ[5] + auxZ[4]*C33*auxY[5]);
							
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+48] += jacob*(auxY[4]*C01*auxX[7] + auxX[4]*C33*auxY[7]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+49] += jacob*(auxY[4]*C00*auxY[7] + auxX[4]*C33*auxX[7] + auxZ[4]*C33*auxZ[7]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+50] += jacob*(auxY[4]*C01*auxZ[7] + auxZ[4]*C33*auxY[7]);
							
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+51] += jacob*(auxY[4]*C01*auxX[6] + auxX[4]*C33*auxY[6]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+52] += jacob*(auxY[4]*C00*auxY[6] + auxX[4]*C33*auxX[6] + auxZ[4]*C33*auxZ[6]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+53] += jacob*(auxY[4]*C01*auxZ[6] + auxZ[4]*C33*auxY[6]);
						
						__syncthreads();  // Line 2:
						
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+12] += jacob*(auxZ[4]*C01*auxX[0] + auxX[4]*C33*auxZ[0]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+13] += jacob*(auxZ[4]*C01*auxY[0] + auxY[4]*C33*auxZ[0]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+14] += jacob*(auxZ[4]*C00*auxZ[0] + auxY[4]*C33*auxY[0] + auxX[4]*C33*auxX[0]);
							
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+15] += jacob*(auxZ[4]*C01*auxX[1] + auxX[4]*C33*auxZ[1]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+16] += jacob*(auxZ[4]*C01*auxY[1] + auxY[4]*C33*auxZ[1]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+17] += jacob*(auxZ[4]*C00*auxZ[1] + auxY[4]*C33*auxY[1] + auxX[4]*C33*auxX[1]);
							
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+21] += jacob*(auxZ[4]*C01*auxX[3] + auxX[4]*C33*auxZ[3]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+22] += jacob*(auxZ[4]*C01*auxY[3] + auxY[4]*C33*auxZ[3]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+23] += jacob*(auxZ[4]*C00*auxZ[3] + auxY[4]*C33*auxY[3] + auxX[4]*C33*auxX[3]);
							
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+24] += jacob*(auxZ[4]*C01*auxX[2] + auxX[4]*C33*auxZ[2]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+25] += jacob*(auxZ[4]*C01*auxY[2] + auxY[4]*C33*auxZ[2]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+26] += jacob*(auxZ[4]*C00*auxZ[2] + auxY[4]*C33*auxY[2] + auxX[4]*C33*auxX[2]);
							
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+39] += jacob*(auxZ[4]*C01*auxX[4] + auxX[4]*C33*auxZ[4]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+40] += jacob*(auxZ[4]*C01*auxY[4] + auxY[4]*C33*auxZ[4]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+41] += jacob*(auxZ[4]*C00*auxZ[4] + auxY[4]*C33*auxY[4] + auxX[4]*C33*auxX[4]);
							
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+42] += jacob*(auxZ[4]*C01*auxX[5] + auxX[4]*C33*auxZ[5]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+43] += jacob*(auxZ[4]*C01*auxY[5] + auxY[4]*C33*auxZ[5]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+44] += jacob*(auxZ[4]*C00*auxZ[5] + auxY[4]*C33*auxY[5] + auxX[4]*C33*auxX[5]);
							
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+48] += jacob*(auxZ[4]*C01*auxX[7] + auxX[4]*C33*auxZ[7]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+49] += jacob*(auxZ[4]*C01*auxY[7] + auxY[4]*C33*auxZ[7]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+50] += jacob*(auxZ[4]*C00*auxZ[7] + auxY[4]*C33*auxY[7] + auxX[4]*C33*auxX[7]);
							
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+51] += jacob*(auxZ[4]*C01*auxX[6] + auxX[4]*C33*auxZ[6]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+52] += jacob*(auxZ[4]*C01*auxY[6] + auxY[4]*C33*auxZ[6]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+53] += jacob*(auxZ[4]*C00*auxZ[6] + auxY[4]*C33*auxY[6] + auxX[4]*C33*auxX[6]);
						
					}
					
				}                                                        // =================== Loop kg ===================
											
			}                                                            // =================== Loop jg ===================

		}                                                                // =================== Loop ig ===================
		
	}

}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============

__global__ void AssemblyStiffness3DQ5Mod(int numno, int RowSize, double *CoordQ5_d, double *MatPropQ5_d, double *Matrix_K_Aux_d) 
{
	double r, s, t;  // 3
	double XEL0, YEL0, ZEL0, XEL1, YEL1, ZEL1, XEL2, YEL2, ZEL2, XEL3, YEL3, ZEL3;  // 12 
	double XEL4, YEL4, ZEL4, XEL5, YEL5, ZEL5, XEL6, YEL6, ZEL6, XEL7, YEL7, ZEL7;  // 12 
	double xgaus[2];  // 2
	double p, E, C00, C01, C33;  // 5
	double phi_r[8], phi_s[8], phi_t[8];  // 24
	double jac[3][3];  // 9
	double jacob;  // 1
	double gama[3][3];  // 9
	double auxX[8], auxY[8], auxZ[8];  // 24
	int ig, jg, kg;  // 3
	
	const int xIndex = blockIdx.x*blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y*blockDim.y + threadIdx.y;
	const int thread_id = (gridDim.x*blockDim.x)*yIndex + xIndex;
	
	__syncthreads();
	
	if(thread_id < numno) {
	
		XEL0 = CoordQ5_d[thread_id+ 0*numno];
		YEL0 = CoordQ5_d[thread_id+ 1*numno];
		ZEL0 = CoordQ5_d[thread_id+ 2*numno];   	
		XEL1 = CoordQ5_d[thread_id+ 3*numno];
		YEL1 = CoordQ5_d[thread_id+ 4*numno];  
		ZEL1 = CoordQ5_d[thread_id+ 5*numno];  	
		XEL2 = CoordQ5_d[thread_id+ 6*numno];
		YEL2 = CoordQ5_d[thread_id+ 7*numno]; 
		ZEL2 = CoordQ5_d[thread_id+ 8*numno];  		
		XEL3 = CoordQ5_d[thread_id+ 9*numno];
		YEL3 = CoordQ5_d[thread_id+10*numno];
		ZEL3 = CoordQ5_d[thread_id+11*numno];
		
		XEL4 = CoordQ5_d[thread_id+12*numno];
		YEL4 = CoordQ5_d[thread_id+13*numno];
		ZEL4 = CoordQ5_d[thread_id+14*numno];   	
		XEL5 = CoordQ5_d[thread_id+15*numno];
		YEL5 = CoordQ5_d[thread_id+16*numno];  
		ZEL5 = CoordQ5_d[thread_id+17*numno];  	
		XEL6 = CoordQ5_d[thread_id+18*numno];
		YEL6 = CoordQ5_d[thread_id+19*numno]; 
		ZEL6 = CoordQ5_d[thread_id+20*numno];  		
		XEL7 = CoordQ5_d[thread_id+21*numno];
		YEL7 = CoordQ5_d[thread_id+22*numno];
		ZEL7 = CoordQ5_d[thread_id+23*numno];
		
		__syncthreads();

		xgaus[0]=-0.577350269189626;
		xgaus[1]= 0.577350269189626;
							
		E = MatPropQ5_d[thread_id+ 0*numno];
		p = MatPropQ5_d[thread_id+ 1*numno];
					
		C00=(E*(1-p))/((1+p)*(1-2*p));  
		C01=(E*(1-p))/((1+p)*(1-2*p))*(p/(1-p)); 
		C33=(E*(1-p))/((1+p)*(1-2*p))*((1-2*p)/(2*(1-p)));
			
		for(ig=0; ig<2; ig++) {                                          // =================== Loop ig ===================
			t=xgaus[ig];
					
			for(jg=0; jg<2; jg++) {                                      // =================== Loop jg ===================
				s=xgaus[jg]; 
				
				for(kg=0; kg<2; kg++) {                                  // =================== Loop kg ===================
				r=xgaus[kg]; 
				
					phi_r[0] = -0.125*(1.0-s)*(1.0-t);
					phi_r[1] =  0.125*(1.0-s)*(1.0-t);
					phi_r[2] =  0.125*(1.0+s)*(1.0-t);
					phi_r[3] = -0.125*(1.0+s)*(1.0-t);
					phi_r[4] = -0.125*(1.0-s)*(1.0+t);
					phi_r[5] =  0.125*(1.0-s)*(1.0+t);
					phi_r[6] =  0.125*(1.0+s)*(1.0+t);
					phi_r[7] = -0.125*(1.0+s)*(1.0+t);
	 
					phi_s[0] = -0.125*(1.0-r)*(1.0-t);
					phi_s[1] = -0.125*(1.0+r)*(1.0-t);
					phi_s[2] =  0.125*(1.0+r)*(1.0-t);
					phi_s[3] =  0.125*(1.0-r)*(1.0-t);
					phi_s[4] = -0.125*(1.0-r)*(1.0+t);
					phi_s[5] = -0.125*(1.0+r)*(1.0+t);
					phi_s[6] =  0.125*(1.0+r)*(1.0+t);
					phi_s[7] =  0.125*(1.0-r)*(1.0+t);
		 
					phi_t[0] = -0.125*(1.0-r)*(1.0-s);
					phi_t[1] = -0.125*(1.0+r)*(1.0-s);
					phi_t[2] = -0.125*(1.0+r)*(1.0+s);
					phi_t[3] = -0.125*(1.0-r)*(1.0+s);
					phi_t[4] =  0.125*(1.0-r)*(1.0-s);
					phi_t[5] =  0.125*(1.0+r)*(1.0-s);
					phi_t[6] =  0.125*(1.0+r)*(1.0+s);
					phi_t[7] =  0.125*(1.0-r)*(1.0+s);
						
					jac[0][0]=phi_r[0]*XEL0+phi_r[1]*XEL1+phi_r[2]*XEL2+phi_r[3]*XEL3+phi_r[4]*XEL4+phi_r[5]*XEL5+phi_r[6]*XEL6+phi_r[7]*XEL7;
					jac[0][1]=phi_r[0]*YEL0+phi_r[1]*YEL1+phi_r[2]*YEL2+phi_r[3]*YEL3+phi_r[4]*YEL4+phi_r[5]*YEL5+phi_r[6]*YEL6+phi_r[7]*YEL7;
					jac[0][2]=phi_r[0]*ZEL0+phi_r[1]*ZEL1+phi_r[2]*ZEL2+phi_r[3]*ZEL3+phi_r[4]*ZEL4+phi_r[5]*ZEL5+phi_r[6]*ZEL6+phi_r[7]*ZEL7;

					jac[1][0]=phi_s[0]*XEL0+phi_s[1]*XEL1+phi_s[2]*XEL2+phi_s[3]*XEL3+phi_s[4]*XEL4+phi_s[5]*XEL5+phi_s[6]*XEL6+phi_s[7]*XEL7;
					jac[1][1]=phi_s[0]*YEL0+phi_s[1]*YEL1+phi_s[2]*YEL2+phi_s[3]*YEL3+phi_s[4]*YEL4+phi_s[5]*YEL5+phi_s[6]*YEL6+phi_s[7]*YEL7;
					jac[1][2]=phi_s[0]*ZEL0+phi_s[1]*ZEL1+phi_s[2]*ZEL2+phi_s[3]*ZEL3+phi_s[4]*ZEL4+phi_s[5]*ZEL5+phi_s[6]*ZEL6+phi_s[7]*ZEL7;
					
					jac[2][0]=phi_t[0]*XEL0+phi_t[1]*XEL1+phi_t[2]*XEL2+phi_t[3]*XEL3+phi_t[4]*XEL4+phi_t[5]*XEL5+phi_t[6]*XEL6+phi_t[7]*XEL7;
					jac[2][1]=phi_t[0]*YEL0+phi_t[1]*YEL1+phi_t[2]*YEL2+phi_t[3]*YEL3+phi_t[4]*YEL4+phi_t[5]*YEL5+phi_t[6]*YEL6+phi_t[7]*YEL7;
					jac[2][2]=phi_t[0]*ZEL0+phi_t[1]*ZEL1+phi_t[2]*ZEL2+phi_t[3]*ZEL3+phi_t[4]*ZEL4+phi_t[5]*ZEL5+phi_t[6]*ZEL6+phi_t[7]*ZEL7;
							
					jacob = jac[0][0]*(jac[1][1]*jac[2][2]-jac[1][2]*jac[2][1])-jac[0][1]*(jac[1][0]*jac[2][2]-jac[1][2]*jac[2][0])+
							jac[0][2]*(jac[1][0]*jac[2][1]-jac[1][1]*jac[2][0]);
					
					if(jacob > 0.) {
					
						gama[0][0] = (jac[1][1]*jac[2][2] - jac[1][2]*jac[2][1]) /jacob;
						gama[0][1] =-(jac[0][1]*jac[2][2] - jac[0][2]*jac[2][1]) /jacob;
						gama[0][2] = (jac[0][1]*jac[1][2] - jac[0][2]*jac[1][1]) /jacob;
						
						gama[1][0] =-(jac[1][0]*jac[2][2] - jac[1][2]*jac[2][0]) /jacob;
						gama[1][1] = (jac[0][0]*jac[2][2] - jac[0][2]*jac[2][0]) /jacob;
						gama[1][2] =-(jac[0][0]*jac[1][2] - jac[0][2]*jac[1][0]) /jacob;
						
						gama[2][0] = (jac[1][0]*jac[2][1] - jac[1][1]*jac[2][0]) /jacob;
						gama[2][1] =-(jac[0][0]*jac[2][1] - jac[0][1]*jac[2][0]) /jacob;
						gama[2][2] = (jac[0][0]*jac[1][1] - jac[0][1]*jac[1][0]) /jacob;
										
						auxX[0] = gama[0][0]*phi_r[0]+gama[0][1]*phi_s[0]+gama[0][2]*phi_t[0];
						auxY[0] = gama[1][0]*phi_r[0]+gama[1][1]*phi_s[0]+gama[1][2]*phi_t[0];
						auxZ[0] = gama[2][0]*phi_r[0]+gama[2][1]*phi_s[0]+gama[2][2]*phi_t[0];
							
						auxX[1] = gama[0][0]*phi_r[1]+gama[0][1]*phi_s[1]+gama[0][2]*phi_t[1];
						auxY[1] = gama[1][0]*phi_r[1]+gama[1][1]*phi_s[1]+gama[1][2]*phi_t[1];
						auxZ[1] = gama[2][0]*phi_r[1]+gama[2][1]*phi_s[1]+gama[2][2]*phi_t[1];
						
						auxX[2] = gama[0][0]*phi_r[2]+gama[0][1]*phi_s[2]+gama[0][2]*phi_t[2];
						auxY[2] = gama[1][0]*phi_r[2]+gama[1][1]*phi_s[2]+gama[1][2]*phi_t[2];
						auxZ[2] = gama[2][0]*phi_r[2]+gama[2][1]*phi_s[2]+gama[2][2]*phi_t[2];
						
						auxX[3] = gama[0][0]*phi_r[3]+gama[0][1]*phi_s[3]+gama[0][2]*phi_t[3];
						auxY[3] = gama[1][0]*phi_r[3]+gama[1][1]*phi_s[3]+gama[1][2]*phi_t[3];
						auxZ[3] = gama[2][0]*phi_r[3]+gama[2][1]*phi_s[3]+gama[2][2]*phi_t[3];
						
						auxX[4] = gama[0][0]*phi_r[4]+gama[0][1]*phi_s[4]+gama[0][2]*phi_t[4];
						auxY[4] = gama[1][0]*phi_r[4]+gama[1][1]*phi_s[4]+gama[1][2]*phi_t[4];
						auxZ[4] = gama[2][0]*phi_r[4]+gama[2][1]*phi_s[4]+gama[2][2]*phi_t[4];
							
						auxX[5] = gama[0][0]*phi_r[5]+gama[0][1]*phi_s[5]+gama[0][2]*phi_t[5];
						auxY[5] = gama[1][0]*phi_r[5]+gama[1][1]*phi_s[5]+gama[1][2]*phi_t[5];
						auxZ[5] = gama[2][0]*phi_r[5]+gama[2][1]*phi_s[5]+gama[2][2]*phi_t[5];
						
						auxX[6] = gama[0][0]*phi_r[6]+gama[0][1]*phi_s[6]+gama[0][2]*phi_t[6];
						auxY[6] = gama[1][0]*phi_r[6]+gama[1][1]*phi_s[6]+gama[1][2]*phi_t[6];
						auxZ[6] = gama[2][0]*phi_r[6]+gama[2][1]*phi_s[6]+gama[2][2]*phi_t[6];
						
						auxX[7] = gama[0][0]*phi_r[7]+gama[0][1]*phi_s[7]+gama[0][2]*phi_t[7];
						auxY[7] = gama[1][0]*phi_r[7]+gama[1][1]*phi_s[7]+gama[1][2]*phi_t[7];
						auxZ[7] = gama[2][0]*phi_r[7]+gama[2][1]*phi_s[7]+gama[2][2]*phi_t[7];
						
						__syncthreads();  // Line 0:
							
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+27] += jacob*(auxX[2]*C00*auxX[0] + auxY[2]*C33*auxY[0] + auxZ[2]*C33*auxZ[0]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+28] += jacob*(auxX[2]*C01*auxY[0] + auxY[2]*C33*auxX[0]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+29] += jacob*(auxX[2]*C01*auxZ[0] + auxZ[2]*C33*auxX[0]);
							
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+30] += jacob*(auxX[2]*C00*auxX[1] + auxY[2]*C33*auxY[1] + auxZ[2]*C33*auxZ[1]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+31] += jacob*(auxX[2]*C01*auxY[1] + auxY[2]*C33*auxX[1]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+32] += jacob*(auxX[2]*C01*auxZ[1] + auxZ[2]*C33*auxX[1]);
							
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+36] += jacob*(auxX[2]*C00*auxX[3] + auxY[2]*C33*auxY[3] + auxZ[2]*C33*auxZ[3]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+37] += jacob*(auxX[2]*C01*auxY[3] + auxY[2]*C33*auxX[3]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+38] += jacob*(auxX[2]*C01*auxZ[3] + auxZ[2]*C33*auxX[3]);
							
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+39] += jacob*(auxX[2]*C00*auxX[2] + auxY[2]*C33*auxY[2] + auxZ[2]*C33*auxZ[2]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+40] += jacob*(auxX[2]*C01*auxY[2] + auxY[2]*C33*auxX[2]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+41] += jacob*(auxX[2]*C01*auxZ[2] + auxZ[2]*C33*auxX[2]);
							
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+54] += jacob*(auxX[2]*C00*auxX[4] + auxY[2]*C33*auxY[4] + auxZ[2]*C33*auxZ[4]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+55] += jacob*(auxX[2]*C01*auxY[4] + auxY[2]*C33*auxX[4]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+56] += jacob*(auxX[2]*C01*auxZ[4] + auxZ[2]*C33*auxX[4]);
							
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+57] += jacob*(auxX[2]*C00*auxX[5] + auxY[2]*C33*auxY[5] + auxZ[2]*C33*auxZ[5]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+58] += jacob*(auxX[2]*C01*auxY[5] + auxY[2]*C33*auxX[5]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+59] += jacob*(auxX[2]*C01*auxZ[5] + auxZ[2]*C33*auxX[5]);
							
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+63] += jacob*(auxX[2]*C00*auxX[7] + auxY[2]*C33*auxY[7] + auxZ[2]*C33*auxZ[7]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+64] += jacob*(auxX[2]*C01*auxY[7] + auxY[2]*C33*auxX[7]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+65] += jacob*(auxX[2]*C01*auxZ[7] + auxZ[2]*C33*auxX[7]);
							
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+66] += jacob*(auxX[2]*C00*auxX[6] + auxY[2]*C33*auxY[6] + auxZ[2]*C33*auxZ[6]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+67] += jacob*(auxX[2]*C01*auxY[6] + auxY[2]*C33*auxX[6]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+68] += jacob*(auxX[2]*C01*auxZ[6] + auxZ[2]*C33*auxX[6]);
							
						__syncthreads();  // Line 1:
						
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+27] += jacob*(auxY[2]*C01*auxX[0] + auxX[2]*C33*auxY[0]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+28] += jacob*(auxY[2]*C00*auxY[0] + auxX[2]*C33*auxX[0] + auxZ[2]*C33*auxZ[0]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+29] += jacob*(auxY[2]*C01*auxZ[0] + auxZ[2]*C33*auxY[0]);
							
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+30] += jacob*(auxY[2]*C01*auxX[1] + auxX[2]*C33*auxY[1]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+31] += jacob*(auxY[2]*C00*auxY[1] + auxX[2]*C33*auxX[1] + auxZ[2]*C33*auxZ[1]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+32] += jacob*(auxY[2]*C01*auxZ[1] + auxZ[2]*C33*auxY[1]);
							
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+36] += jacob*(auxY[2]*C01*auxX[3] + auxX[2]*C33*auxY[3]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+37] += jacob*(auxY[2]*C00*auxY[3] + auxX[2]*C33*auxX[3] + auxZ[2]*C33*auxZ[3]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+38] += jacob*(auxY[2]*C01*auxZ[3] + auxZ[2]*C33*auxY[3]);
							
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+39] += jacob*(auxY[2]*C01*auxX[2] + auxX[2]*C33*auxY[2]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+40] += jacob*(auxY[2]*C00*auxY[2] + auxX[2]*C33*auxX[2] + auxZ[2]*C33*auxZ[2]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+41] += jacob*(auxY[2]*C01*auxZ[2] + auxZ[2]*C33*auxY[2]);
							
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+54] += jacob*(auxY[2]*C01*auxX[4] + auxX[2]*C33*auxY[4]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+55] += jacob*(auxY[2]*C00*auxY[4] + auxX[2]*C33*auxX[4] + auxZ[2]*C33*auxZ[4]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+56] += jacob*(auxY[2]*C01*auxZ[4] + auxZ[2]*C33*auxY[4]);
							
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+57] += jacob*(auxY[2]*C01*auxX[5] + auxX[2]*C33*auxY[5]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+58] += jacob*(auxY[2]*C00*auxY[5] + auxX[2]*C33*auxX[5] + auxZ[2]*C33*auxZ[5]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+59] += jacob*(auxY[2]*C01*auxZ[5] + auxZ[2]*C33*auxY[5]);
							
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+63] += jacob*(auxY[2]*C01*auxX[7] + auxX[2]*C33*auxY[7]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+64] += jacob*(auxY[2]*C00*auxY[7] + auxX[2]*C33*auxX[7] + auxZ[2]*C33*auxZ[7]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+65] += jacob*(auxY[2]*C01*auxZ[7] + auxZ[2]*C33*auxY[7]);
							
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+66] += jacob*(auxY[2]*C01*auxX[6] + auxX[2]*C33*auxY[6]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+67] += jacob*(auxY[2]*C00*auxY[6] + auxX[2]*C33*auxX[6] + auxZ[2]*C33*auxZ[6]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+68] += jacob*(auxY[2]*C01*auxZ[6] + auxZ[2]*C33*auxY[6]);
						
						__syncthreads();  // Line 2:
							
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+27] += jacob*(auxZ[2]*C01*auxX[0] + auxX[2]*C33*auxZ[0]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+28] += jacob*(auxZ[2]*C01*auxY[0] + auxY[2]*C33*auxZ[0]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+29] += jacob*(auxZ[2]*C00*auxZ[0] + auxY[2]*C33*auxY[0] + auxX[2]*C33*auxX[0]);
							
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+30] += jacob*(auxZ[2]*C01*auxX[1] + auxX[2]*C33*auxZ[1]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+31] += jacob*(auxZ[2]*C01*auxY[1] + auxY[2]*C33*auxZ[1]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+32] += jacob*(auxZ[2]*C00*auxZ[1] + auxY[2]*C33*auxY[1] + auxX[2]*C33*auxX[1]);
							
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+36] += jacob*(auxZ[2]*C01*auxX[3] + auxX[2]*C33*auxZ[3]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+37] += jacob*(auxZ[2]*C01*auxY[3] + auxY[2]*C33*auxZ[3]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+38] += jacob*(auxZ[2]*C00*auxZ[3] + auxY[2]*C33*auxY[3] + auxX[2]*C33*auxX[3]);
							
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+39] += jacob*(auxZ[2]*C01*auxX[2] + auxX[2]*C33*auxZ[2]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+40] += jacob*(auxZ[2]*C01*auxY[2] + auxY[2]*C33*auxZ[2]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+41] += jacob*(auxZ[2]*C00*auxZ[2] + auxY[2]*C33*auxY[2] + auxX[2]*C33*auxX[2]);
							
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+54] += jacob*(auxZ[2]*C01*auxX[4] + auxX[2]*C33*auxZ[4]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+55] += jacob*(auxZ[2]*C01*auxY[4] + auxY[2]*C33*auxZ[4]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+56] += jacob*(auxZ[2]*C00*auxZ[4] + auxY[2]*C33*auxY[4] + auxX[2]*C33*auxX[4]);
							
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+57] += jacob*(auxZ[2]*C01*auxX[5] + auxX[2]*C33*auxZ[5]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+58] += jacob*(auxZ[2]*C01*auxY[5] + auxY[2]*C33*auxZ[5]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+59] += jacob*(auxZ[2]*C00*auxZ[5] + auxY[2]*C33*auxY[5] + auxX[2]*C33*auxX[5]);
							
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+63] += jacob*(auxZ[2]*C01*auxX[7] + auxX[2]*C33*auxZ[7]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+64] += jacob*(auxZ[2]*C01*auxY[7] + auxY[2]*C33*auxZ[7]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+65] += jacob*(auxZ[2]*C00*auxZ[7] + auxY[2]*C33*auxY[7] + auxX[2]*C33*auxX[7]);
							
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+66] += jacob*(auxZ[2]*C01*auxX[6] + auxX[2]*C33*auxZ[6]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+67] += jacob*(auxZ[2]*C01*auxY[6] + auxY[2]*C33*auxZ[6]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+68] += jacob*(auxZ[2]*C00*auxZ[6] + auxY[2]*C33*auxY[6] + auxX[2]*C33*auxX[6]);
			
					}
					
				}                                                        // =================== Loop kg ===================
											
			}                                                            // =================== Loop jg ===================

		}                                                                // =================== Loop ig ===================
		
	}
	
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============

__global__ void AssemblyStiffness3DQ6Mod(int numno, int RowSize, double *CoordQ6_d, double *MatPropQ6_d, double *Matrix_K_Aux_d) 
{
	double r, s, t;  // 3
	double XEL0, YEL0, ZEL0, XEL1, YEL1, ZEL1, XEL2, YEL2, ZEL2, XEL3, YEL3, ZEL3;  // 12 
	double XEL4, YEL4, ZEL4, XEL5, YEL5, ZEL5, XEL6, YEL6, ZEL6, XEL7, YEL7, ZEL7;  // 12 
	double xgaus[2];  // 2
	double p, E, C00, C01, C33;  // 5
	double phi_r[8], phi_s[8], phi_t[8];  // 24
	double jac[3][3];  // 9
	double jacob;  // 1
	double gama[3][3];  // 9
	double auxX[8], auxY[8], auxZ[8];  // 24
	int ig, jg, kg;  // 3
	
	const int xIndex = blockIdx.x*blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y*blockDim.y + threadIdx.y;
	const int thread_id = (gridDim.x*blockDim.x)*yIndex + xIndex;
	
	__syncthreads();
	
	if(thread_id < numno) {
	
		XEL0 = CoordQ6_d[thread_id+ 0*numno];
		YEL0 = CoordQ6_d[thread_id+ 1*numno];
		ZEL0 = CoordQ6_d[thread_id+ 2*numno];   	
		XEL1 = CoordQ6_d[thread_id+ 3*numno];
		YEL1 = CoordQ6_d[thread_id+ 4*numno];  
		ZEL1 = CoordQ6_d[thread_id+ 5*numno];  	
		XEL2 = CoordQ6_d[thread_id+ 6*numno];
		YEL2 = CoordQ6_d[thread_id+ 7*numno]; 
		ZEL2 = CoordQ6_d[thread_id+ 8*numno];  		
		XEL3 = CoordQ6_d[thread_id+ 9*numno];
		YEL3 = CoordQ6_d[thread_id+10*numno];
		ZEL3 = CoordQ6_d[thread_id+11*numno];
		
		XEL4 = CoordQ6_d[thread_id+12*numno];
		YEL4 = CoordQ6_d[thread_id+13*numno];
		ZEL4 = CoordQ6_d[thread_id+14*numno];   	
		XEL5 = CoordQ6_d[thread_id+15*numno];
		YEL5 = CoordQ6_d[thread_id+16*numno];  
		ZEL5 = CoordQ6_d[thread_id+17*numno];  	
		XEL6 = CoordQ6_d[thread_id+18*numno];
		YEL6 = CoordQ6_d[thread_id+19*numno]; 
		ZEL6 = CoordQ6_d[thread_id+20*numno];  		
		XEL7 = CoordQ6_d[thread_id+21*numno];
		YEL7 = CoordQ6_d[thread_id+22*numno];
		ZEL7 = CoordQ6_d[thread_id+23*numno];
		
		__syncthreads();

		xgaus[0]=-0.577350269189626;
		xgaus[1]= 0.577350269189626;
							
		E = MatPropQ6_d[thread_id+ 0*numno];
		p = MatPropQ6_d[thread_id+ 1*numno];
					
		C00=(E*(1-p))/((1+p)*(1-2*p));  
		C01=(E*(1-p))/((1+p)*(1-2*p))*(p/(1-p)); 
		C33=(E*(1-p))/((1+p)*(1-2*p))*((1-2*p)/(2*(1-p)));
			
		for(ig=0; ig<2; ig++) {                                          // =================== Loop ig ===================
			t=xgaus[ig];
					
			for(jg=0; jg<2; jg++) {                                      // =================== Loop jg ===================
				s=xgaus[jg]; 
				
				for(kg=0; kg<2; kg++) {                                  // =================== Loop kg ===================
				r=xgaus[kg]; 
				
					phi_r[0] = -0.125*(1.0-s)*(1.0-t);
					phi_r[1] =  0.125*(1.0-s)*(1.0-t);
					phi_r[2] =  0.125*(1.0+s)*(1.0-t);
					phi_r[3] = -0.125*(1.0+s)*(1.0-t);
					phi_r[4] = -0.125*(1.0-s)*(1.0+t);
					phi_r[5] =  0.125*(1.0-s)*(1.0+t);
					phi_r[6] =  0.125*(1.0+s)*(1.0+t);
					phi_r[7] = -0.125*(1.0+s)*(1.0+t);
	 
					phi_s[0] = -0.125*(1.0-r)*(1.0-t);
					phi_s[1] = -0.125*(1.0+r)*(1.0-t);
					phi_s[2] =  0.125*(1.0+r)*(1.0-t);
					phi_s[3] =  0.125*(1.0-r)*(1.0-t);
					phi_s[4] = -0.125*(1.0-r)*(1.0+t);
					phi_s[5] = -0.125*(1.0+r)*(1.0+t);
					phi_s[6] =  0.125*(1.0+r)*(1.0+t);
					phi_s[7] =  0.125*(1.0-r)*(1.0+t);
		 
					phi_t[0] = -0.125*(1.0-r)*(1.0-s);
					phi_t[1] = -0.125*(1.0+r)*(1.0-s);
					phi_t[2] = -0.125*(1.0+r)*(1.0+s);
					phi_t[3] = -0.125*(1.0-r)*(1.0+s);
					phi_t[4] =  0.125*(1.0-r)*(1.0-s);
					phi_t[5] =  0.125*(1.0+r)*(1.0-s);
					phi_t[6] =  0.125*(1.0+r)*(1.0+s);
					phi_t[7] =  0.125*(1.0-r)*(1.0+s);
						
					jac[0][0]=phi_r[0]*XEL0+phi_r[1]*XEL1+phi_r[2]*XEL2+phi_r[3]*XEL3+phi_r[4]*XEL4+phi_r[5]*XEL5+phi_r[6]*XEL6+phi_r[7]*XEL7;
					jac[0][1]=phi_r[0]*YEL0+phi_r[1]*YEL1+phi_r[2]*YEL2+phi_r[3]*YEL3+phi_r[4]*YEL4+phi_r[5]*YEL5+phi_r[6]*YEL6+phi_r[7]*YEL7;
					jac[0][2]=phi_r[0]*ZEL0+phi_r[1]*ZEL1+phi_r[2]*ZEL2+phi_r[3]*ZEL3+phi_r[4]*ZEL4+phi_r[5]*ZEL5+phi_r[6]*ZEL6+phi_r[7]*ZEL7;

					jac[1][0]=phi_s[0]*XEL0+phi_s[1]*XEL1+phi_s[2]*XEL2+phi_s[3]*XEL3+phi_s[4]*XEL4+phi_s[5]*XEL5+phi_s[6]*XEL6+phi_s[7]*XEL7;
					jac[1][1]=phi_s[0]*YEL0+phi_s[1]*YEL1+phi_s[2]*YEL2+phi_s[3]*YEL3+phi_s[4]*YEL4+phi_s[5]*YEL5+phi_s[6]*YEL6+phi_s[7]*YEL7;
					jac[1][2]=phi_s[0]*ZEL0+phi_s[1]*ZEL1+phi_s[2]*ZEL2+phi_s[3]*ZEL3+phi_s[4]*ZEL4+phi_s[5]*ZEL5+phi_s[6]*ZEL6+phi_s[7]*ZEL7;
					
					jac[2][0]=phi_t[0]*XEL0+phi_t[1]*XEL1+phi_t[2]*XEL2+phi_t[3]*XEL3+phi_t[4]*XEL4+phi_t[5]*XEL5+phi_t[6]*XEL6+phi_t[7]*XEL7;
					jac[2][1]=phi_t[0]*YEL0+phi_t[1]*YEL1+phi_t[2]*YEL2+phi_t[3]*YEL3+phi_t[4]*YEL4+phi_t[5]*YEL5+phi_t[6]*YEL6+phi_t[7]*YEL7;
					jac[2][2]=phi_t[0]*ZEL0+phi_t[1]*ZEL1+phi_t[2]*ZEL2+phi_t[3]*ZEL3+phi_t[4]*ZEL4+phi_t[5]*ZEL5+phi_t[6]*ZEL6+phi_t[7]*ZEL7;
							
					jacob = jac[0][0]*(jac[1][1]*jac[2][2]-jac[1][2]*jac[2][1])-jac[0][1]*(jac[1][0]*jac[2][2]-jac[1][2]*jac[2][0])+
							jac[0][2]*(jac[1][0]*jac[2][1]-jac[1][1]*jac[2][0]);
					
					if(jacob > 0.) {
					
						gama[0][0] = (jac[1][1]*jac[2][2] - jac[1][2]*jac[2][1]) /jacob;
						gama[0][1] =-(jac[0][1]*jac[2][2] - jac[0][2]*jac[2][1]) /jacob;
						gama[0][2] = (jac[0][1]*jac[1][2] - jac[0][2]*jac[1][1]) /jacob;
						
						gama[1][0] =-(jac[1][0]*jac[2][2] - jac[1][2]*jac[2][0]) /jacob;
						gama[1][1] = (jac[0][0]*jac[2][2] - jac[0][2]*jac[2][0]) /jacob;
						gama[1][2] =-(jac[0][0]*jac[1][2] - jac[0][2]*jac[1][0]) /jacob;
						
						gama[2][0] = (jac[1][0]*jac[2][1] - jac[1][1]*jac[2][0]) /jacob;
						gama[2][1] =-(jac[0][0]*jac[2][1] - jac[0][1]*jac[2][0]) /jacob;
						gama[2][2] = (jac[0][0]*jac[1][1] - jac[0][1]*jac[1][0]) /jacob;
										
						auxX[0] = gama[0][0]*phi_r[0]+gama[0][1]*phi_s[0]+gama[0][2]*phi_t[0];
						auxY[0] = gama[1][0]*phi_r[0]+gama[1][1]*phi_s[0]+gama[1][2]*phi_t[0];
						auxZ[0] = gama[2][0]*phi_r[0]+gama[2][1]*phi_s[0]+gama[2][2]*phi_t[0];
							
						auxX[1] = gama[0][0]*phi_r[1]+gama[0][1]*phi_s[1]+gama[0][2]*phi_t[1];
						auxY[1] = gama[1][0]*phi_r[1]+gama[1][1]*phi_s[1]+gama[1][2]*phi_t[1];
						auxZ[1] = gama[2][0]*phi_r[1]+gama[2][1]*phi_s[1]+gama[2][2]*phi_t[1];
						
						auxX[2] = gama[0][0]*phi_r[2]+gama[0][1]*phi_s[2]+gama[0][2]*phi_t[2];
						auxY[2] = gama[1][0]*phi_r[2]+gama[1][1]*phi_s[2]+gama[1][2]*phi_t[2];
						auxZ[2] = gama[2][0]*phi_r[2]+gama[2][1]*phi_s[2]+gama[2][2]*phi_t[2];
						
						auxX[3] = gama[0][0]*phi_r[3]+gama[0][1]*phi_s[3]+gama[0][2]*phi_t[3];
						auxY[3] = gama[1][0]*phi_r[3]+gama[1][1]*phi_s[3]+gama[1][2]*phi_t[3];
						auxZ[3] = gama[2][0]*phi_r[3]+gama[2][1]*phi_s[3]+gama[2][2]*phi_t[3];
						
						auxX[4] = gama[0][0]*phi_r[4]+gama[0][1]*phi_s[4]+gama[0][2]*phi_t[4];
						auxY[4] = gama[1][0]*phi_r[4]+gama[1][1]*phi_s[4]+gama[1][2]*phi_t[4];
						auxZ[4] = gama[2][0]*phi_r[4]+gama[2][1]*phi_s[4]+gama[2][2]*phi_t[4];
							
						auxX[5] = gama[0][0]*phi_r[5]+gama[0][1]*phi_s[5]+gama[0][2]*phi_t[5];
						auxY[5] = gama[1][0]*phi_r[5]+gama[1][1]*phi_s[5]+gama[1][2]*phi_t[5];
						auxZ[5] = gama[2][0]*phi_r[5]+gama[2][1]*phi_s[5]+gama[2][2]*phi_t[5];
						
						auxX[6] = gama[0][0]*phi_r[6]+gama[0][1]*phi_s[6]+gama[0][2]*phi_t[6];
						auxY[6] = gama[1][0]*phi_r[6]+gama[1][1]*phi_s[6]+gama[1][2]*phi_t[6];
						auxZ[6] = gama[2][0]*phi_r[6]+gama[2][1]*phi_s[6]+gama[2][2]*phi_t[6];
						
						auxX[7] = gama[0][0]*phi_r[7]+gama[0][1]*phi_s[7]+gama[0][2]*phi_t[7];
						auxY[7] = gama[1][0]*phi_r[7]+gama[1][1]*phi_s[7]+gama[1][2]*phi_t[7];
						auxZ[7] = gama[2][0]*phi_r[7]+gama[2][1]*phi_s[7]+gama[2][2]*phi_t[7];

						__syncthreads();  // Line 0:
							
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+30] += jacob*(auxX[3]*C00*auxX[0] + auxY[3]*C33*auxY[0] + auxZ[3]*C33*auxZ[0]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+31] += jacob*(auxX[3]*C01*auxY[0] + auxY[3]*C33*auxX[0]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+32] += jacob*(auxX[3]*C01*auxZ[0] + auxZ[3]*C33*auxX[0]);
							
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+33] += jacob*(auxX[3]*C00*auxX[1] + auxY[3]*C33*auxY[1] + auxZ[3]*C33*auxZ[1]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+34] += jacob*(auxX[3]*C01*auxY[1] + auxY[3]*C33*auxX[1]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+35] += jacob*(auxX[3]*C01*auxZ[1] + auxZ[3]*C33*auxX[1]);
							
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+39] += jacob*(auxX[3]*C00*auxX[3] + auxY[3]*C33*auxY[3] + auxZ[3]*C33*auxZ[3]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+40] += jacob*(auxX[3]*C01*auxY[3] + auxY[3]*C33*auxX[3]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+41] += jacob*(auxX[3]*C01*auxZ[3] + auxZ[3]*C33*auxX[3]);
							
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+42] += jacob*(auxX[3]*C00*auxX[2] + auxY[3]*C33*auxY[2] + auxZ[3]*C33*auxZ[2]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+43] += jacob*(auxX[3]*C01*auxY[2] + auxY[3]*C33*auxX[2]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+44] += jacob*(auxX[3]*C01*auxZ[2] + auxZ[3]*C33*auxX[2]);
							
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+57] += jacob*(auxX[3]*C00*auxX[4] + auxY[3]*C33*auxY[4] + auxZ[3]*C33*auxZ[4]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+58] += jacob*(auxX[3]*C01*auxY[4] + auxY[3]*C33*auxX[4]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+59] += jacob*(auxX[3]*C01*auxZ[4] + auxZ[3]*C33*auxX[4]);
							
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+60] += jacob*(auxX[3]*C00*auxX[5] + auxY[3]*C33*auxY[5] + auxZ[3]*C33*auxZ[5]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+61] += jacob*(auxX[3]*C01*auxY[5] + auxY[3]*C33*auxX[5]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+62] += jacob*(auxX[3]*C01*auxZ[5] + auxZ[3]*C33*auxX[5]);
							
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+66] += jacob*(auxX[3]*C00*auxX[7] + auxY[3]*C33*auxY[7] + auxZ[3]*C33*auxZ[7]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+67] += jacob*(auxX[3]*C01*auxY[7] + auxY[3]*C33*auxX[7]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+68] += jacob*(auxX[3]*C01*auxZ[7] + auxZ[3]*C33*auxX[7]);
							
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+69] += jacob*(auxX[3]*C00*auxX[6] + auxY[3]*C33*auxY[6] + auxZ[3]*C33*auxZ[6]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+70] += jacob*(auxX[3]*C01*auxY[6] + auxY[3]*C33*auxX[6]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+71] += jacob*(auxX[3]*C01*auxZ[6] + auxZ[3]*C33*auxX[6]);
						
						__syncthreads();  // Line 1:
						
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+30] += jacob*(auxY[3]*C01*auxX[0] + auxX[3]*C33*auxY[0]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+31] += jacob*(auxY[3]*C00*auxY[0] + auxX[3]*C33*auxX[0] + auxZ[3]*C33*auxZ[0]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+32] += jacob*(auxY[3]*C01*auxZ[0] + auxZ[3]*C33*auxY[0]);
							
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+33] += jacob*(auxY[3]*C01*auxX[1] + auxX[3]*C33*auxY[1]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+34] += jacob*(auxY[3]*C00*auxY[1] + auxX[3]*C33*auxX[1] + auxZ[3]*C33*auxZ[1]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+35] += jacob*(auxY[3]*C01*auxZ[1] + auxZ[3]*C33*auxY[1]);
							
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+39] += jacob*(auxY[3]*C01*auxX[3] + auxX[3]*C33*auxY[3]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+40] += jacob*(auxY[3]*C00*auxY[3] + auxX[3]*C33*auxX[3] + auxZ[3]*C33*auxZ[3]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+41] += jacob*(auxY[3]*C01*auxZ[3] + auxZ[3]*C33*auxY[3]);
							
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+42] += jacob*(auxY[3]*C01*auxX[2] + auxX[3]*C33*auxY[2]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+43] += jacob*(auxY[3]*C00*auxY[2] + auxX[3]*C33*auxX[2] + auxZ[3]*C33*auxZ[2]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+44] += jacob*(auxY[3]*C01*auxZ[2] + auxZ[3]*C33*auxY[2]);
							
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+57] += jacob*(auxY[3]*C01*auxX[4] + auxX[3]*C33*auxY[4]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+58] += jacob*(auxY[3]*C00*auxY[4] + auxX[3]*C33*auxX[4] + auxZ[3]*C33*auxZ[4]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+59] += jacob*(auxY[3]*C01*auxZ[4] + auxZ[3]*C33*auxY[4]);
							
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+60] += jacob*(auxY[3]*C01*auxX[5] + auxX[3]*C33*auxY[5]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+61] += jacob*(auxY[3]*C00*auxY[5] + auxX[3]*C33*auxX[5] + auxZ[3]*C33*auxZ[5]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+62] += jacob*(auxY[3]*C01*auxZ[5] + auxZ[3]*C33*auxY[5]);
							
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+66] += jacob*(auxY[3]*C01*auxX[7] + auxX[3]*C33*auxY[7]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+67] += jacob*(auxY[3]*C00*auxY[7] + auxX[3]*C33*auxX[7] + auxZ[3]*C33*auxZ[7]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+68] += jacob*(auxY[3]*C01*auxZ[7] + auxZ[3]*C33*auxY[7]);
							
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+69] += jacob*(auxY[3]*C01*auxX[6] + auxX[3]*C33*auxY[6]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+70] += jacob*(auxY[3]*C00*auxY[6] + auxX[3]*C33*auxX[6] + auxZ[3]*C33*auxZ[6]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+71] += jacob*(auxY[3]*C01*auxZ[6] + auxZ[3]*C33*auxY[6]);
				
						__syncthreads();  // Line 2:
						
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+30] += jacob*(auxZ[3]*C01*auxX[0] + auxX[3]*C33*auxZ[0]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+31] += jacob*(auxZ[3]*C01*auxY[0] + auxY[3]*C33*auxZ[0]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+32] += jacob*(auxZ[3]*C00*auxZ[0] + auxY[3]*C33*auxY[0] + auxX[3]*C33*auxX[0]);
							
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+33] += jacob*(auxZ[3]*C01*auxX[1] + auxX[3]*C33*auxZ[1]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+34] += jacob*(auxZ[3]*C01*auxY[1] + auxY[3]*C33*auxZ[1]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+35] += jacob*(auxZ[3]*C00*auxZ[1] + auxY[3]*C33*auxY[1] + auxX[3]*C33*auxX[1]);
							
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+39] += jacob*(auxZ[3]*C01*auxX[3] + auxX[3]*C33*auxZ[3]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+40] += jacob*(auxZ[3]*C01*auxY[3] + auxY[3]*C33*auxZ[3]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+41] += jacob*(auxZ[3]*C00*auxZ[3] + auxY[3]*C33*auxY[3] + auxX[3]*C33*auxX[3]);
							
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+42] += jacob*(auxZ[3]*C01*auxX[2] + auxX[3]*C33*auxZ[2]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+43] += jacob*(auxZ[3]*C01*auxY[2] + auxY[3]*C33*auxZ[2]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+44] += jacob*(auxZ[3]*C00*auxZ[2] + auxY[3]*C33*auxY[2] + auxX[3]*C33*auxX[2]);
							
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+57] += jacob*(auxZ[3]*C01*auxX[4] + auxX[3]*C33*auxZ[4]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+58] += jacob*(auxZ[3]*C01*auxY[4] + auxY[3]*C33*auxZ[4]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+59] += jacob*(auxZ[3]*C00*auxZ[4] + auxY[3]*C33*auxY[4] + auxX[3]*C33*auxX[4]);
							
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+60] += jacob*(auxZ[3]*C01*auxX[5] + auxX[3]*C33*auxZ[5]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+61] += jacob*(auxZ[3]*C01*auxY[5] + auxY[3]*C33*auxZ[5]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+62] += jacob*(auxZ[3]*C00*auxZ[5] + auxY[3]*C33*auxY[5] + auxX[3]*C33*auxX[5]);
							
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+66] += jacob*(auxZ[3]*C01*auxX[7] + auxX[3]*C33*auxZ[7]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+67] += jacob*(auxZ[3]*C01*auxY[7] + auxY[3]*C33*auxZ[7]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+68] += jacob*(auxZ[3]*C00*auxZ[7] + auxY[3]*C33*auxY[7] + auxX[3]*C33*auxX[7]);
							
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+69] += jacob*(auxZ[3]*C01*auxX[6] + auxX[3]*C33*auxZ[6]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+70] += jacob*(auxZ[3]*C01*auxY[6] + auxY[3]*C33*auxZ[6]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+71] += jacob*(auxZ[3]*C00*auxZ[6] + auxY[3]*C33*auxY[6] + auxX[3]*C33*auxX[6]);
							
					}
					
				}                                                        // =================== Loop kg ===================
											
			}                                                            // =================== Loop jg ===================

		}                                                                // =================== Loop ig ===================
		
	}
	
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============

__global__ void AssemblyStiffness3DQ7Mod(int numno, int RowSize, double *CoordQ7_d, double *MatPropQ7_d, double *Matrix_K_Aux_d) 
{
	double r, s, t;  // 3
	double XEL0, YEL0, ZEL0, XEL1, YEL1, ZEL1, XEL2, YEL2, ZEL2, XEL3, YEL3, ZEL3;  // 12 
	double XEL4, YEL4, ZEL4, XEL5, YEL5, ZEL5, XEL6, YEL6, ZEL6, XEL7, YEL7, ZEL7;  // 12 
	double xgaus[2];  // 2
	double p, E, C00, C01, C33;  // 5
	double phi_r[8], phi_s[8], phi_t[8];  // 24
	double jac[3][3];  // 9
	double jacob;  // 1
	double gama[3][3];  // 9
	double auxX[8], auxY[8], auxZ[8];  // 24
	int ig, jg, kg;  // 3
	
	const int xIndex = blockIdx.x*blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y*blockDim.y + threadIdx.y;
	const int thread_id = (gridDim.x*blockDim.x)*yIndex + xIndex;
	
	__syncthreads();
	
	if(thread_id < numno) {
	
		XEL0 = CoordQ7_d[thread_id+ 0*numno];
		YEL0 = CoordQ7_d[thread_id+ 1*numno];
		ZEL0 = CoordQ7_d[thread_id+ 2*numno];   	
		XEL1 = CoordQ7_d[thread_id+ 3*numno];
		YEL1 = CoordQ7_d[thread_id+ 4*numno];  
		ZEL1 = CoordQ7_d[thread_id+ 5*numno];  	
		XEL2 = CoordQ7_d[thread_id+ 6*numno];
		YEL2 = CoordQ7_d[thread_id+ 7*numno]; 
		ZEL2 = CoordQ7_d[thread_id+ 8*numno];  		
		XEL3 = CoordQ7_d[thread_id+ 9*numno];
		YEL3 = CoordQ7_d[thread_id+10*numno];
		ZEL3 = CoordQ7_d[thread_id+11*numno];
		
		XEL4 = CoordQ7_d[thread_id+12*numno];
		YEL4 = CoordQ7_d[thread_id+13*numno];
		ZEL4 = CoordQ7_d[thread_id+14*numno];   	
		XEL5 = CoordQ7_d[thread_id+15*numno];
		YEL5 = CoordQ7_d[thread_id+16*numno];  
		ZEL5 = CoordQ7_d[thread_id+17*numno];  	
		XEL6 = CoordQ7_d[thread_id+18*numno];
		YEL6 = CoordQ7_d[thread_id+19*numno]; 
		ZEL6 = CoordQ7_d[thread_id+20*numno];  		
		XEL7 = CoordQ7_d[thread_id+21*numno];
		YEL7 = CoordQ7_d[thread_id+22*numno];
		ZEL7 = CoordQ7_d[thread_id+23*numno];
		
		__syncthreads();

		xgaus[0]=-0.577350269189626;
		xgaus[1]= 0.577350269189626;
							
		E = MatPropQ7_d[thread_id+ 0*numno];
		p = MatPropQ7_d[thread_id+ 1*numno];
					
		C00=(E*(1-p))/((1+p)*(1-2*p));  
		C01=(E*(1-p))/((1+p)*(1-2*p))*(p/(1-p)); 
		C33=(E*(1-p))/((1+p)*(1-2*p))*((1-2*p)/(2*(1-p)));
			
		for(ig=0; ig<2; ig++) {                                          // =================== Loop ig ===================
			t=xgaus[ig];
					
			for(jg=0; jg<2; jg++) {                                      // =================== Loop jg ===================
				s=xgaus[jg]; 
				
				for(kg=0; kg<2; kg++) {                                  // =================== Loop kg ===================
				r=xgaus[kg]; 
				
					phi_r[0] = -0.125*(1.0-s)*(1.0-t);
					phi_r[1] =  0.125*(1.0-s)*(1.0-t);
					phi_r[2] =  0.125*(1.0+s)*(1.0-t);
					phi_r[3] = -0.125*(1.0+s)*(1.0-t);
					phi_r[4] = -0.125*(1.0-s)*(1.0+t);
					phi_r[5] =  0.125*(1.0-s)*(1.0+t);
					phi_r[6] =  0.125*(1.0+s)*(1.0+t);
					phi_r[7] = -0.125*(1.0+s)*(1.0+t);
	 
					phi_s[0] = -0.125*(1.0-r)*(1.0-t);
					phi_s[1] = -0.125*(1.0+r)*(1.0-t);
					phi_s[2] =  0.125*(1.0+r)*(1.0-t);
					phi_s[3] =  0.125*(1.0-r)*(1.0-t);
					phi_s[4] = -0.125*(1.0-r)*(1.0+t);
					phi_s[5] = -0.125*(1.0+r)*(1.0+t);
					phi_s[6] =  0.125*(1.0+r)*(1.0+t);
					phi_s[7] =  0.125*(1.0-r)*(1.0+t);
		 
					phi_t[0] = -0.125*(1.0-r)*(1.0-s);
					phi_t[1] = -0.125*(1.0+r)*(1.0-s);
					phi_t[2] = -0.125*(1.0+r)*(1.0+s);
					phi_t[3] = -0.125*(1.0-r)*(1.0+s);
					phi_t[4] =  0.125*(1.0-r)*(1.0-s);
					phi_t[5] =  0.125*(1.0+r)*(1.0-s);
					phi_t[6] =  0.125*(1.0+r)*(1.0+s);
					phi_t[7] =  0.125*(1.0-r)*(1.0+s);
						
					jac[0][0]=phi_r[0]*XEL0+phi_r[1]*XEL1+phi_r[2]*XEL2+phi_r[3]*XEL3+phi_r[4]*XEL4+phi_r[5]*XEL5+phi_r[6]*XEL6+phi_r[7]*XEL7;
					jac[0][1]=phi_r[0]*YEL0+phi_r[1]*YEL1+phi_r[2]*YEL2+phi_r[3]*YEL3+phi_r[4]*YEL4+phi_r[5]*YEL5+phi_r[6]*YEL6+phi_r[7]*YEL7;
					jac[0][2]=phi_r[0]*ZEL0+phi_r[1]*ZEL1+phi_r[2]*ZEL2+phi_r[3]*ZEL3+phi_r[4]*ZEL4+phi_r[5]*ZEL5+phi_r[6]*ZEL6+phi_r[7]*ZEL7;

					jac[1][0]=phi_s[0]*XEL0+phi_s[1]*XEL1+phi_s[2]*XEL2+phi_s[3]*XEL3+phi_s[4]*XEL4+phi_s[5]*XEL5+phi_s[6]*XEL6+phi_s[7]*XEL7;
					jac[1][1]=phi_s[0]*YEL0+phi_s[1]*YEL1+phi_s[2]*YEL2+phi_s[3]*YEL3+phi_s[4]*YEL4+phi_s[5]*YEL5+phi_s[6]*YEL6+phi_s[7]*YEL7;
					jac[1][2]=phi_s[0]*ZEL0+phi_s[1]*ZEL1+phi_s[2]*ZEL2+phi_s[3]*ZEL3+phi_s[4]*ZEL4+phi_s[5]*ZEL5+phi_s[6]*ZEL6+phi_s[7]*ZEL7;
					
					jac[2][0]=phi_t[0]*XEL0+phi_t[1]*XEL1+phi_t[2]*XEL2+phi_t[3]*XEL3+phi_t[4]*XEL4+phi_t[5]*XEL5+phi_t[6]*XEL6+phi_t[7]*XEL7;
					jac[2][1]=phi_t[0]*YEL0+phi_t[1]*YEL1+phi_t[2]*YEL2+phi_t[3]*YEL3+phi_t[4]*YEL4+phi_t[5]*YEL5+phi_t[6]*YEL6+phi_t[7]*YEL7;
					jac[2][2]=phi_t[0]*ZEL0+phi_t[1]*ZEL1+phi_t[2]*ZEL2+phi_t[3]*ZEL3+phi_t[4]*ZEL4+phi_t[5]*ZEL5+phi_t[6]*ZEL6+phi_t[7]*ZEL7;
							
					jacob = jac[0][0]*(jac[1][1]*jac[2][2]-jac[1][2]*jac[2][1])-jac[0][1]*(jac[1][0]*jac[2][2]-jac[1][2]*jac[2][0])+
							jac[0][2]*(jac[1][0]*jac[2][1]-jac[1][1]*jac[2][0]);
					
					if(jacob > 0.) {
					
						gama[0][0] = (jac[1][1]*jac[2][2] - jac[1][2]*jac[2][1]) /jacob;
						gama[0][1] =-(jac[0][1]*jac[2][2] - jac[0][2]*jac[2][1]) /jacob;
						gama[0][2] = (jac[0][1]*jac[1][2] - jac[0][2]*jac[1][1]) /jacob;
						
						gama[1][0] =-(jac[1][0]*jac[2][2] - jac[1][2]*jac[2][0]) /jacob;
						gama[1][1] = (jac[0][0]*jac[2][2] - jac[0][2]*jac[2][0]) /jacob;
						gama[1][2] =-(jac[0][0]*jac[1][2] - jac[0][2]*jac[1][0]) /jacob;
						
						gama[2][0] = (jac[1][0]*jac[2][1] - jac[1][1]*jac[2][0]) /jacob;
						gama[2][1] =-(jac[0][0]*jac[2][1] - jac[0][1]*jac[2][0]) /jacob;
						gama[2][2] = (jac[0][0]*jac[1][1] - jac[0][1]*jac[1][0]) /jacob;
										
						auxX[0] = gama[0][0]*phi_r[0]+gama[0][1]*phi_s[0]+gama[0][2]*phi_t[0];
						auxY[0] = gama[1][0]*phi_r[0]+gama[1][1]*phi_s[0]+gama[1][2]*phi_t[0];
						auxZ[0] = gama[2][0]*phi_r[0]+gama[2][1]*phi_s[0]+gama[2][2]*phi_t[0];
							
						auxX[1] = gama[0][0]*phi_r[1]+gama[0][1]*phi_s[1]+gama[0][2]*phi_t[1];
						auxY[1] = gama[1][0]*phi_r[1]+gama[1][1]*phi_s[1]+gama[1][2]*phi_t[1];
						auxZ[1] = gama[2][0]*phi_r[1]+gama[2][1]*phi_s[1]+gama[2][2]*phi_t[1];
						
						auxX[2] = gama[0][0]*phi_r[2]+gama[0][1]*phi_s[2]+gama[0][2]*phi_t[2];
						auxY[2] = gama[1][0]*phi_r[2]+gama[1][1]*phi_s[2]+gama[1][2]*phi_t[2];
						auxZ[2] = gama[2][0]*phi_r[2]+gama[2][1]*phi_s[2]+gama[2][2]*phi_t[2];
						
						auxX[3] = gama[0][0]*phi_r[3]+gama[0][1]*phi_s[3]+gama[0][2]*phi_t[3];
						auxY[3] = gama[1][0]*phi_r[3]+gama[1][1]*phi_s[3]+gama[1][2]*phi_t[3];
						auxZ[3] = gama[2][0]*phi_r[3]+gama[2][1]*phi_s[3]+gama[2][2]*phi_t[3];
						
						auxX[4] = gama[0][0]*phi_r[4]+gama[0][1]*phi_s[4]+gama[0][2]*phi_t[4];
						auxY[4] = gama[1][0]*phi_r[4]+gama[1][1]*phi_s[4]+gama[1][2]*phi_t[4];
						auxZ[4] = gama[2][0]*phi_r[4]+gama[2][1]*phi_s[4]+gama[2][2]*phi_t[4];
							
						auxX[5] = gama[0][0]*phi_r[5]+gama[0][1]*phi_s[5]+gama[0][2]*phi_t[5];
						auxY[5] = gama[1][0]*phi_r[5]+gama[1][1]*phi_s[5]+gama[1][2]*phi_t[5];
						auxZ[5] = gama[2][0]*phi_r[5]+gama[2][1]*phi_s[5]+gama[2][2]*phi_t[5];
						
						auxX[6] = gama[0][0]*phi_r[6]+gama[0][1]*phi_s[6]+gama[0][2]*phi_t[6];
						auxY[6] = gama[1][0]*phi_r[6]+gama[1][1]*phi_s[6]+gama[1][2]*phi_t[6];
						auxZ[6] = gama[2][0]*phi_r[6]+gama[2][1]*phi_s[6]+gama[2][2]*phi_t[6];
						
						auxX[7] = gama[0][0]*phi_r[7]+gama[0][1]*phi_s[7]+gama[0][2]*phi_t[7];
						auxY[7] = gama[1][0]*phi_r[7]+gama[1][1]*phi_s[7]+gama[1][2]*phi_t[7];
						auxZ[7] = gama[2][0]*phi_r[7]+gama[2][1]*phi_s[7]+gama[2][2]*phi_t[7];
						
						__syncthreads();  // Line 0:
							
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+36] += jacob*(auxX[1]*C00*auxX[0] + auxY[1]*C33*auxY[0] + auxZ[1]*C33*auxZ[0]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+37] += jacob*(auxX[1]*C01*auxY[0] + auxY[1]*C33*auxX[0]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+38] += jacob*(auxX[1]*C01*auxZ[0] + auxZ[1]*C33*auxX[0]);
							
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+39] += jacob*(auxX[1]*C00*auxX[1] + auxY[1]*C33*auxY[1] + auxZ[1]*C33*auxZ[1]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+40] += jacob*(auxX[1]*C01*auxY[1] + auxY[1]*C33*auxX[1]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+41] += jacob*(auxX[1]*C01*auxZ[1] + auxZ[1]*C33*auxX[1]);
							
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+45] += jacob*(auxX[1]*C00*auxX[3] + auxY[1]*C33*auxY[3] + auxZ[1]*C33*auxZ[3]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+46] += jacob*(auxX[1]*C01*auxY[3] + auxY[1]*C33*auxX[3]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+47] += jacob*(auxX[1]*C01*auxZ[3] + auxZ[1]*C33*auxX[3]);
							
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+48] += jacob*(auxX[1]*C00*auxX[2] + auxY[1]*C33*auxY[2] + auxZ[1]*C33*auxZ[2]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+49] += jacob*(auxX[1]*C01*auxY[2] + auxY[1]*C33*auxX[2]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+50] += jacob*(auxX[1]*C01*auxZ[2] + auxZ[1]*C33*auxX[2]);
							
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+63] += jacob*(auxX[1]*C00*auxX[4] + auxY[1]*C33*auxY[4] + auxZ[1]*C33*auxZ[4]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+64] += jacob*(auxX[1]*C01*auxY[4] + auxY[1]*C33*auxX[4]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+65] += jacob*(auxX[1]*C01*auxZ[4] + auxZ[1]*C33*auxX[4]);
							
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+66] += jacob*(auxX[1]*C00*auxX[5] + auxY[1]*C33*auxY[5] + auxZ[1]*C33*auxZ[5]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+67] += jacob*(auxX[1]*C01*auxY[5] + auxY[1]*C33*auxX[5]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+68] += jacob*(auxX[1]*C01*auxZ[5] + auxZ[1]*C33*auxX[5]);
							
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+72] += jacob*(auxX[1]*C00*auxX[7] + auxY[1]*C33*auxY[7] + auxZ[1]*C33*auxZ[7]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+73] += jacob*(auxX[1]*C01*auxY[7] + auxY[1]*C33*auxX[7]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+74] += jacob*(auxX[1]*C01*auxZ[7] + auxZ[1]*C33*auxX[7]);
							
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+75] += jacob*(auxX[1]*C00*auxX[6] + auxY[1]*C33*auxY[6] + auxZ[1]*C33*auxZ[6]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+76] += jacob*(auxX[1]*C01*auxY[6] + auxY[1]*C33*auxX[6]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+77] += jacob*(auxX[1]*C01*auxZ[6] + auxZ[1]*C33*auxX[6]);

						__syncthreads();  // Line 1:
							
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+36] += jacob*(auxY[1]*C01*auxX[0] + auxX[1]*C33*auxY[0]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+37] += jacob*(auxY[1]*C00*auxY[0] + auxX[1]*C33*auxX[0] + auxZ[1]*C33*auxZ[0]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+38] += jacob*(auxY[1]*C01*auxZ[0] + auxZ[1]*C33*auxY[0]);
							
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+39] += jacob*(auxY[1]*C01*auxX[1] + auxX[1]*C33*auxY[1]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+40] += jacob*(auxY[1]*C00*auxY[1] + auxX[1]*C33*auxX[1] + auxZ[1]*C33*auxZ[1]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+41] += jacob*(auxY[1]*C01*auxZ[1] + auxZ[1]*C33*auxY[1]);
							
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+45] += jacob*(auxY[1]*C01*auxX[3] + auxX[1]*C33*auxY[3]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+46] += jacob*(auxY[1]*C00*auxY[3] + auxX[1]*C33*auxX[3] + auxZ[1]*C33*auxZ[3]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+47] += jacob*(auxY[1]*C01*auxZ[3] + auxZ[1]*C33*auxY[3]);
							
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+48] += jacob*(auxY[1]*C01*auxX[2] + auxX[1]*C33*auxY[2]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+49] += jacob*(auxY[1]*C00*auxY[2] + auxX[1]*C33*auxX[2] + auxZ[1]*C33*auxZ[2]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+50] += jacob*(auxY[1]*C01*auxZ[2] + auxZ[1]*C33*auxY[2]);
							
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+63] += jacob*(auxY[1]*C01*auxX[4] + auxX[1]*C33*auxY[4]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+64] += jacob*(auxY[1]*C00*auxY[4] + auxX[1]*C33*auxX[4] + auxZ[1]*C33*auxZ[4]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+65] += jacob*(auxY[1]*C01*auxZ[4] + auxZ[1]*C33*auxY[4]);
							
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+66] += jacob*(auxY[1]*C01*auxX[5] + auxX[1]*C33*auxY[5]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+67] += jacob*(auxY[1]*C00*auxY[5] + auxX[1]*C33*auxX[5] + auxZ[1]*C33*auxZ[5]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+68] += jacob*(auxY[1]*C01*auxZ[5] + auxZ[1]*C33*auxY[5]);
							
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+72] += jacob*(auxY[1]*C01*auxX[7] + auxX[1]*C33*auxY[7]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+73] += jacob*(auxY[1]*C00*auxY[7] + auxX[1]*C33*auxX[7] + auxZ[1]*C33*auxZ[7]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+74] += jacob*(auxY[1]*C01*auxZ[7] + auxZ[1]*C33*auxY[7]);
						
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+75] += jacob*(auxY[1]*C01*auxX[6] + auxX[1]*C33*auxY[6]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+76] += jacob*(auxY[1]*C00*auxY[6] + auxX[1]*C33*auxX[6] + auxZ[1]*C33*auxZ[6]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+77] += jacob*(auxY[1]*C01*auxZ[6] + auxZ[1]*C33*auxY[6]);
				
						__syncthreads();  // Line 2:
						
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+36] += jacob*(auxZ[1]*C01*auxX[0] + auxX[1]*C33*auxZ[0]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+37] += jacob*(auxZ[1]*C01*auxY[0] + auxY[1]*C33*auxZ[0]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+38] += jacob*(auxZ[1]*C00*auxZ[0] + auxY[1]*C33*auxY[0] + auxX[1]*C33*auxX[0]);
							
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+39] += jacob*(auxZ[1]*C01*auxX[1] + auxX[1]*C33*auxZ[1]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+40] += jacob*(auxZ[1]*C01*auxY[1] + auxY[1]*C33*auxZ[1]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+41] += jacob*(auxZ[1]*C00*auxZ[1] + auxY[1]*C33*auxY[1] + auxX[1]*C33*auxX[1]);
							
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+45] += jacob*(auxZ[1]*C01*auxX[3] + auxX[1]*C33*auxZ[3]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+46] += jacob*(auxZ[1]*C01*auxY[3] + auxY[1]*C33*auxZ[3]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+47] += jacob*(auxZ[1]*C00*auxZ[3] + auxY[1]*C33*auxY[3] + auxX[1]*C33*auxX[3]);
							
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+48] += jacob*(auxZ[1]*C01*auxX[2] + auxX[1]*C33*auxZ[2]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+49] += jacob*(auxZ[1]*C01*auxY[2] + auxY[1]*C33*auxZ[2]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+50] += jacob*(auxZ[1]*C00*auxZ[2] + auxY[1]*C33*auxY[2] + auxX[1]*C33*auxX[2]);
							
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+63] += jacob*(auxZ[1]*C01*auxX[4] + auxX[1]*C33*auxZ[4]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+64] += jacob*(auxZ[1]*C01*auxY[4] + auxY[1]*C33*auxZ[4]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+65] += jacob*(auxZ[1]*C00*auxZ[4] + auxY[1]*C33*auxY[4] + auxX[1]*C33*auxX[4]);
							
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+66] += jacob*(auxZ[1]*C01*auxX[5] + auxX[1]*C33*auxZ[5]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+67] += jacob*(auxZ[1]*C01*auxY[5] + auxY[1]*C33*auxZ[5]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+68] += jacob*(auxZ[1]*C00*auxZ[5] + auxY[1]*C33*auxY[5] + auxX[1]*C33*auxX[5]);
							
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+72] += jacob*(auxZ[1]*C01*auxX[7] + auxX[1]*C33*auxZ[7]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+73] += jacob*(auxZ[1]*C01*auxY[7] + auxY[1]*C33*auxZ[7]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+74] += jacob*(auxZ[1]*C00*auxZ[7] + auxY[1]*C33*auxY[7] + auxX[1]*C33*auxX[7]);
							
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+75] += jacob*(auxZ[1]*C01*auxX[6] + auxX[1]*C33*auxZ[6]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+76] += jacob*(auxZ[1]*C01*auxY[6] + auxY[1]*C33*auxZ[6]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+77] += jacob*(auxZ[1]*C00*auxZ[6] + auxY[1]*C33*auxY[6] + auxX[1]*C33*auxX[6]);
						
					}
					
				}                                                        // =================== Loop kg ===================
											
			}                                                            // =================== Loop jg ===================

		}                                                                // =================== Loop ig ===================
		
	}
	
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============

__global__ void AssemblyStiffness3DQ8Mod(int numno, int RowSize, double *CoordQ8_d, double *MatPropQ8_d, double *Matrix_K_Aux_d) 
{
	double r, s, t;  // 3
	double XEL0, YEL0, ZEL0, XEL1, YEL1, ZEL1, XEL2, YEL2, ZEL2, XEL3, YEL3, ZEL3;  // 12 
	double XEL4, YEL4, ZEL4, XEL5, YEL5, ZEL5, XEL6, YEL6, ZEL6, XEL7, YEL7, ZEL7;  // 12 
	double xgaus[2];  // 2
	double p, E, C00, C01, C33;  // 5
	double phi_r[8], phi_s[8], phi_t[8];  // 24
	double jac[3][3];  // 9
	double jacob;  // 1
	double gama[3][3];  // 9
	double auxX[8], auxY[8], auxZ[8];  // 24
	int ig, jg, kg;  // 3
	
	const int xIndex = blockIdx.x*blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y*blockDim.y + threadIdx.y;
	const int thread_id = (gridDim.x*blockDim.x)*yIndex + xIndex;
	
	__syncthreads();
	
	if(thread_id < numno) {
	
		XEL0 = CoordQ8_d[thread_id+ 0*numno];
		YEL0 = CoordQ8_d[thread_id+ 1*numno];
		ZEL0 = CoordQ8_d[thread_id+ 2*numno];   	
		XEL1 = CoordQ8_d[thread_id+ 3*numno];
		YEL1 = CoordQ8_d[thread_id+ 4*numno];  
		ZEL1 = CoordQ8_d[thread_id+ 5*numno];  	
		XEL2 = CoordQ8_d[thread_id+ 6*numno];
		YEL2 = CoordQ8_d[thread_id+ 7*numno]; 
		ZEL2 = CoordQ8_d[thread_id+ 8*numno];  		
		XEL3 = CoordQ8_d[thread_id+ 9*numno];
		YEL3 = CoordQ8_d[thread_id+10*numno];
		ZEL3 = CoordQ8_d[thread_id+11*numno];
		
		XEL4 = CoordQ8_d[thread_id+12*numno];
		YEL4 = CoordQ8_d[thread_id+13*numno];
		ZEL4 = CoordQ8_d[thread_id+14*numno];   	
		XEL5 = CoordQ8_d[thread_id+15*numno];
		YEL5 = CoordQ8_d[thread_id+16*numno];  
		ZEL5 = CoordQ8_d[thread_id+17*numno];  	
		XEL6 = CoordQ8_d[thread_id+18*numno];
		YEL6 = CoordQ8_d[thread_id+19*numno]; 
		ZEL6 = CoordQ8_d[thread_id+20*numno];  		
		XEL7 = CoordQ8_d[thread_id+21*numno];
		YEL7 = CoordQ8_d[thread_id+22*numno];
		ZEL7 = CoordQ8_d[thread_id+23*numno];
		
		__syncthreads();

		xgaus[0]=-0.577350269189626;
		xgaus[1]= 0.577350269189626;
							
		E = MatPropQ8_d[thread_id+ 0*numno];
		p = MatPropQ8_d[thread_id+ 1*numno];
					
		C00=(E*(1-p))/((1+p)*(1-2*p));  
		C01=(E*(1-p))/((1+p)*(1-2*p))*(p/(1-p)); 
		C33=(E*(1-p))/((1+p)*(1-2*p))*((1-2*p)/(2*(1-p)));
			
		for(ig=0; ig<2; ig++) {                                          // =================== Loop ig ===================
			t=xgaus[ig];
					
			for(jg=0; jg<2; jg++) {                                      // =================== Loop jg ===================
				s=xgaus[jg]; 
				
				for(kg=0; kg<2; kg++) {                                  // =================== Loop kg ===================
				r=xgaus[kg]; 
				
					phi_r[0] = -0.125*(1.0-s)*(1.0-t);
					phi_r[1] =  0.125*(1.0-s)*(1.0-t);
					phi_r[2] =  0.125*(1.0+s)*(1.0-t);
					phi_r[3] = -0.125*(1.0+s)*(1.0-t);
					phi_r[4] = -0.125*(1.0-s)*(1.0+t);
					phi_r[5] =  0.125*(1.0-s)*(1.0+t);
					phi_r[6] =  0.125*(1.0+s)*(1.0+t);
					phi_r[7] = -0.125*(1.0+s)*(1.0+t);
	 
					phi_s[0] = -0.125*(1.0-r)*(1.0-t);
					phi_s[1] = -0.125*(1.0+r)*(1.0-t);
					phi_s[2] =  0.125*(1.0+r)*(1.0-t);
					phi_s[3] =  0.125*(1.0-r)*(1.0-t);
					phi_s[4] = -0.125*(1.0-r)*(1.0+t);
					phi_s[5] = -0.125*(1.0+r)*(1.0+t);
					phi_s[6] =  0.125*(1.0+r)*(1.0+t);
					phi_s[7] =  0.125*(1.0-r)*(1.0+t);
		 
					phi_t[0] = -0.125*(1.0-r)*(1.0-s);
					phi_t[1] = -0.125*(1.0+r)*(1.0-s);
					phi_t[2] = -0.125*(1.0+r)*(1.0+s);
					phi_t[3] = -0.125*(1.0-r)*(1.0+s);
					phi_t[4] =  0.125*(1.0-r)*(1.0-s);
					phi_t[5] =  0.125*(1.0+r)*(1.0-s);
					phi_t[6] =  0.125*(1.0+r)*(1.0+s);
					phi_t[7] =  0.125*(1.0-r)*(1.0+s);
						
					jac[0][0]=phi_r[0]*XEL0+phi_r[1]*XEL1+phi_r[2]*XEL2+phi_r[3]*XEL3+phi_r[4]*XEL4+phi_r[5]*XEL5+phi_r[6]*XEL6+phi_r[7]*XEL7;
					jac[0][1]=phi_r[0]*YEL0+phi_r[1]*YEL1+phi_r[2]*YEL2+phi_r[3]*YEL3+phi_r[4]*YEL4+phi_r[5]*YEL5+phi_r[6]*YEL6+phi_r[7]*YEL7;
					jac[0][2]=phi_r[0]*ZEL0+phi_r[1]*ZEL1+phi_r[2]*ZEL2+phi_r[3]*ZEL3+phi_r[4]*ZEL4+phi_r[5]*ZEL5+phi_r[6]*ZEL6+phi_r[7]*ZEL7;

					jac[1][0]=phi_s[0]*XEL0+phi_s[1]*XEL1+phi_s[2]*XEL2+phi_s[3]*XEL3+phi_s[4]*XEL4+phi_s[5]*XEL5+phi_s[6]*XEL6+phi_s[7]*XEL7;
					jac[1][1]=phi_s[0]*YEL0+phi_s[1]*YEL1+phi_s[2]*YEL2+phi_s[3]*YEL3+phi_s[4]*YEL4+phi_s[5]*YEL5+phi_s[6]*YEL6+phi_s[7]*YEL7;
					jac[1][2]=phi_s[0]*ZEL0+phi_s[1]*ZEL1+phi_s[2]*ZEL2+phi_s[3]*ZEL3+phi_s[4]*ZEL4+phi_s[5]*ZEL5+phi_s[6]*ZEL6+phi_s[7]*ZEL7;
					
					jac[2][0]=phi_t[0]*XEL0+phi_t[1]*XEL1+phi_t[2]*XEL2+phi_t[3]*XEL3+phi_t[4]*XEL4+phi_t[5]*XEL5+phi_t[6]*XEL6+phi_t[7]*XEL7;
					jac[2][1]=phi_t[0]*YEL0+phi_t[1]*YEL1+phi_t[2]*YEL2+phi_t[3]*YEL3+phi_t[4]*YEL4+phi_t[5]*YEL5+phi_t[6]*YEL6+phi_t[7]*YEL7;
					jac[2][2]=phi_t[0]*ZEL0+phi_t[1]*ZEL1+phi_t[2]*ZEL2+phi_t[3]*ZEL3+phi_t[4]*ZEL4+phi_t[5]*ZEL5+phi_t[6]*ZEL6+phi_t[7]*ZEL7;
							
					jacob = jac[0][0]*(jac[1][1]*jac[2][2]-jac[1][2]*jac[2][1])-jac[0][1]*(jac[1][0]*jac[2][2]-jac[1][2]*jac[2][0])+
							jac[0][2]*(jac[1][0]*jac[2][1]-jac[1][1]*jac[2][0]);
					
					if(jacob > 0.) {
					
						gama[0][0] = (jac[1][1]*jac[2][2] - jac[1][2]*jac[2][1]) /jacob;
						gama[0][1] =-(jac[0][1]*jac[2][2] - jac[0][2]*jac[2][1]) /jacob;
						gama[0][2] = (jac[0][1]*jac[1][2] - jac[0][2]*jac[1][1]) /jacob;
						
						gama[1][0] =-(jac[1][0]*jac[2][2] - jac[1][2]*jac[2][0]) /jacob;
						gama[1][1] = (jac[0][0]*jac[2][2] - jac[0][2]*jac[2][0]) /jacob;
						gama[1][2] =-(jac[0][0]*jac[1][2] - jac[0][2]*jac[1][0]) /jacob;
						
						gama[2][0] = (jac[1][0]*jac[2][1] - jac[1][1]*jac[2][0]) /jacob;
						gama[2][1] =-(jac[0][0]*jac[2][1] - jac[0][1]*jac[2][0]) /jacob;
						gama[2][2] = (jac[0][0]*jac[1][1] - jac[0][1]*jac[1][0]) /jacob;
										
						auxX[0] = gama[0][0]*phi_r[0]+gama[0][1]*phi_s[0]+gama[0][2]*phi_t[0];
						auxY[0] = gama[1][0]*phi_r[0]+gama[1][1]*phi_s[0]+gama[1][2]*phi_t[0];
						auxZ[0] = gama[2][0]*phi_r[0]+gama[2][1]*phi_s[0]+gama[2][2]*phi_t[0];
							
						auxX[1] = gama[0][0]*phi_r[1]+gama[0][1]*phi_s[1]+gama[0][2]*phi_t[1];
						auxY[1] = gama[1][0]*phi_r[1]+gama[1][1]*phi_s[1]+gama[1][2]*phi_t[1];
						auxZ[1] = gama[2][0]*phi_r[1]+gama[2][1]*phi_s[1]+gama[2][2]*phi_t[1];
						
						auxX[2] = gama[0][0]*phi_r[2]+gama[0][1]*phi_s[2]+gama[0][2]*phi_t[2];
						auxY[2] = gama[1][0]*phi_r[2]+gama[1][1]*phi_s[2]+gama[1][2]*phi_t[2];
						auxZ[2] = gama[2][0]*phi_r[2]+gama[2][1]*phi_s[2]+gama[2][2]*phi_t[2];
						
						auxX[3] = gama[0][0]*phi_r[3]+gama[0][1]*phi_s[3]+gama[0][2]*phi_t[3];
						auxY[3] = gama[1][0]*phi_r[3]+gama[1][1]*phi_s[3]+gama[1][2]*phi_t[3];
						auxZ[3] = gama[2][0]*phi_r[3]+gama[2][1]*phi_s[3]+gama[2][2]*phi_t[3];
						
						auxX[4] = gama[0][0]*phi_r[4]+gama[0][1]*phi_s[4]+gama[0][2]*phi_t[4];
						auxY[4] = gama[1][0]*phi_r[4]+gama[1][1]*phi_s[4]+gama[1][2]*phi_t[4];
						auxZ[4] = gama[2][0]*phi_r[4]+gama[2][1]*phi_s[4]+gama[2][2]*phi_t[4];
							
						auxX[5] = gama[0][0]*phi_r[5]+gama[0][1]*phi_s[5]+gama[0][2]*phi_t[5];
						auxY[5] = gama[1][0]*phi_r[5]+gama[1][1]*phi_s[5]+gama[1][2]*phi_t[5];
						auxZ[5] = gama[2][0]*phi_r[5]+gama[2][1]*phi_s[5]+gama[2][2]*phi_t[5];
						
						auxX[6] = gama[0][0]*phi_r[6]+gama[0][1]*phi_s[6]+gama[0][2]*phi_t[6];
						auxY[6] = gama[1][0]*phi_r[6]+gama[1][1]*phi_s[6]+gama[1][2]*phi_t[6];
						auxZ[6] = gama[2][0]*phi_r[6]+gama[2][1]*phi_s[6]+gama[2][2]*phi_t[6];
						
						auxX[7] = gama[0][0]*phi_r[7]+gama[0][1]*phi_s[7]+gama[0][2]*phi_t[7];
						auxY[7] = gama[1][0]*phi_r[7]+gama[1][1]*phi_s[7]+gama[1][2]*phi_t[7];
						auxZ[7] = gama[2][0]*phi_r[7]+gama[2][1]*phi_s[7]+gama[2][2]*phi_t[7];
		
						__syncthreads();  // Line 0:
							
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+39] += jacob*(auxX[0]*C00*auxX[0] + auxY[0]*C33*auxY[0] + auxZ[0]*C33*auxZ[0]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+40] += jacob*(auxX[0]*C01*auxY[0] + auxY[0]*C33*auxX[0]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+41] += jacob*(auxX[0]*C01*auxZ[0] + auxZ[0]*C33*auxX[0]);
							
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+42] += jacob*(auxX[0]*C00*auxX[1] + auxY[0]*C33*auxY[1] + auxZ[0]*C33*auxZ[1]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+43] += jacob*(auxX[0]*C01*auxY[1] + auxY[0]*C33*auxX[1]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+44] += jacob*(auxX[0]*C01*auxZ[1] + auxZ[0]*C33*auxX[1]);
							
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+48] += jacob*(auxX[0]*C00*auxX[3] + auxY[0]*C33*auxY[3] + auxZ[0]*C33*auxZ[3]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+49] += jacob*(auxX[0]*C01*auxY[3] + auxY[0]*C33*auxX[3]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+50] += jacob*(auxX[0]*C01*auxZ[3] + auxZ[0]*C33*auxX[3]);
							
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+51] += jacob*(auxX[0]*C00*auxX[2] + auxY[0]*C33*auxY[2] + auxZ[0]*C33*auxZ[2]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+52] += jacob*(auxX[0]*C01*auxY[2] + auxY[0]*C33*auxX[2]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+53] += jacob*(auxX[0]*C01*auxZ[2] + auxZ[0]*C33*auxX[2]);
							
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+66] += jacob*(auxX[0]*C00*auxX[4] + auxY[0]*C33*auxY[4] + auxZ[0]*C33*auxZ[4]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+67] += jacob*(auxX[0]*C01*auxY[4] + auxY[0]*C33*auxX[4]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+68] += jacob*(auxX[0]*C01*auxZ[4] + auxZ[0]*C33*auxX[4]);
							
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+69] += jacob*(auxX[0]*C00*auxX[5] + auxY[0]*C33*auxY[5] + auxZ[0]*C33*auxZ[5]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+70] += jacob*(auxX[0]*C01*auxY[5] + auxY[0]*C33*auxX[5]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+71] += jacob*(auxX[0]*C01*auxZ[5] + auxZ[0]*C33*auxX[5]);
							
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+75] += jacob*(auxX[0]*C00*auxX[7] + auxY[0]*C33*auxY[7] + auxZ[0]*C33*auxZ[7]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+76] += jacob*(auxX[0]*C01*auxY[7] + auxY[0]*C33*auxX[7]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+77] += jacob*(auxX[0]*C01*auxZ[7] + auxZ[0]*C33*auxX[7]);
							
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+78] += jacob*(auxX[0]*C00*auxX[6] + auxY[0]*C33*auxY[6] + auxZ[0]*C33*auxZ[6]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+79] += jacob*(auxX[0]*C01*auxY[6] + auxY[0]*C33*auxX[6]);
						Matrix_K_Aux_d[(3*thread_id  )*RowSize+80] += jacob*(auxX[0]*C01*auxZ[6] + auxZ[0]*C33*auxX[6]);
							
						__syncthreads();  // Line 1:
						
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+39] += jacob*(auxY[0]*C01*auxX[0] + auxX[0]*C33*auxY[0]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+40] += jacob*(auxY[0]*C00*auxY[0] + auxX[0]*C33*auxX[0] + auxZ[0]*C33*auxZ[0]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+41] += jacob*(auxY[0]*C01*auxZ[0] + auxZ[0]*C33*auxY[0]);
							
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+42] += jacob*(auxY[0]*C01*auxX[1] + auxX[0]*C33*auxY[1]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+43] += jacob*(auxY[0]*C00*auxY[1] + auxX[0]*C33*auxX[1] + auxZ[0]*C33*auxZ[1]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+44] += jacob*(auxY[0]*C01*auxZ[1] + auxZ[0]*C33*auxY[1]);
							
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+48] += jacob*(auxY[0]*C01*auxX[3] + auxX[0]*C33*auxY[3]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+49] += jacob*(auxY[0]*C00*auxY[3] + auxX[0]*C33*auxX[3] + auxZ[0]*C33*auxZ[3]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+50] += jacob*(auxY[0]*C01*auxZ[3] + auxZ[0]*C33*auxY[3]);
							
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+51] += jacob*(auxY[0]*C01*auxX[2] + auxX[0]*C33*auxY[2]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+52] += jacob*(auxY[0]*C00*auxY[2] + auxX[0]*C33*auxX[2] + auxZ[0]*C33*auxZ[2]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+53] += jacob*(auxY[0]*C01*auxZ[2] + auxZ[0]*C33*auxY[2]);
							
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+66] += jacob*(auxY[0]*C01*auxX[4] + auxX[0]*C33*auxY[4]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+67] += jacob*(auxY[0]*C00*auxY[4] + auxX[0]*C33*auxX[4] + auxZ[0]*C33*auxZ[4]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+68] += jacob*(auxY[0]*C01*auxZ[4] + auxZ[0]*C33*auxY[4]);
							
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+69] += jacob*(auxY[0]*C01*auxX[5] + auxX[0]*C33*auxY[5]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+70] += jacob*(auxY[0]*C00*auxY[5] + auxX[0]*C33*auxX[5] + auxZ[0]*C33*auxZ[5]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+71] += jacob*(auxY[0]*C01*auxZ[5] + auxZ[0]*C33*auxY[5]);
						
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+75] += jacob*(auxY[0]*C01*auxX[7] + auxX[0]*C33*auxY[7]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+76] += jacob*(auxY[0]*C00*auxY[7] + auxX[0]*C33*auxX[7] + auxZ[0]*C33*auxZ[7]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+77] += jacob*(auxY[0]*C01*auxZ[7] + auxZ[0]*C33*auxY[7]);
							
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+78] += jacob*(auxY[0]*C01*auxX[6] + auxX[0]*C33*auxY[6]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+79] += jacob*(auxY[0]*C00*auxY[6] + auxX[0]*C33*auxX[6] + auxZ[0]*C33*auxZ[6]);
						Matrix_K_Aux_d[(3*thread_id+1)*RowSize+80] += jacob*(auxY[0]*C01*auxZ[6] + auxZ[0]*C33*auxY[6]);
						
						__syncthreads();  // Line 2:
						
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+39] += jacob*(auxZ[0]*C01*auxX[0] + auxX[0]*C33*auxZ[0]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+40] += jacob*(auxZ[0]*C01*auxY[0] + auxY[0]*C33*auxZ[0]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+41] += jacob*(auxZ[0]*C00*auxZ[0] + auxY[0]*C33*auxY[0] + auxX[0]*C33*auxX[0]);
							
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+42] += jacob*(auxZ[0]*C01*auxX[1] + auxX[0]*C33*auxZ[1]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+43] += jacob*(auxZ[0]*C01*auxY[1] + auxY[0]*C33*auxZ[1]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+44] += jacob*(auxZ[0]*C00*auxZ[1] + auxY[0]*C33*auxY[1] + auxX[0]*C33*auxX[1]);
							
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+48] += jacob*(auxZ[0]*C01*auxX[3] + auxX[0]*C33*auxZ[3]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+49] += jacob*(auxZ[0]*C01*auxY[3] + auxY[0]*C33*auxZ[3]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+50] += jacob*(auxZ[0]*C00*auxZ[3] + auxY[0]*C33*auxY[3] + auxX[0]*C33*auxX[3]);
							
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+51] += jacob*(auxZ[0]*C01*auxX[2] + auxX[0]*C33*auxZ[2]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+52] += jacob*(auxZ[0]*C01*auxY[2] + auxY[0]*C33*auxZ[2]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+53] += jacob*(auxZ[0]*C00*auxZ[2] + auxY[0]*C33*auxY[2] + auxX[0]*C33*auxX[2]);
							
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+66] += jacob*(auxZ[0]*C01*auxX[4] + auxX[0]*C33*auxZ[4]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+67] += jacob*(auxZ[0]*C01*auxY[4] + auxY[0]*C33*auxZ[4]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+68] += jacob*(auxZ[0]*C00*auxZ[4] + auxY[0]*C33*auxY[4] + auxX[0]*C33*auxX[4]);
							
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+69] += jacob*(auxZ[0]*C01*auxX[5] + auxX[0]*C33*auxZ[5]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+70] += jacob*(auxZ[0]*C01*auxY[5] + auxY[0]*C33*auxZ[5]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+71] += jacob*(auxZ[0]*C00*auxZ[5] + auxY[0]*C33*auxY[5] + auxX[0]*C33*auxX[5]);
							
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+75] += jacob*(auxZ[0]*C01*auxX[7] + auxX[0]*C33*auxZ[7]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+76] += jacob*(auxZ[0]*C01*auxY[7] + auxY[0]*C33*auxZ[7]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+77] += jacob*(auxZ[0]*C00*auxZ[7] + auxY[0]*C33*auxY[7] + auxX[0]*C33*auxX[7]);
							
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+78] += jacob*(auxZ[0]*C01*auxX[6] + auxX[0]*C33*auxZ[6]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+79] += jacob*(auxZ[0]*C01*auxY[6] + auxY[0]*C33*auxZ[6]);
						Matrix_K_Aux_d[(3*thread_id+2)*RowSize+80] += jacob*(auxZ[0]*C00*auxZ[6] + auxY[0]*C33*auxY[6] + auxX[0]*C33*auxX[6]);

					}
					
				}                                                        // =================== Loop kg ===================
											
			}                                                            // =================== Loop jg ===================

		}                                                                // =================== Loop ig ===================
		
	}

}