/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\
*  Copyright (c) 2019 <GTEP> - All Rights Reserved                            *
*  This file is part of HERMES Project.                                       *
*  Unauthorized copying of this file, via any medium is strictly prohibited.  *
*  Proprietary and confidential.                                              *
*                                                                             *
*  Developers:                                                                *
*     - Nelson Inoue <inoue@puc-rio.br>                                       *
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */


#ifndef	_GPU_H
#define	_GPU_H
#include <cuda_runtime.h>

typedef struct {

	int LocalNumTerm, DeviceNumber, DeviceId, GPUmultiProcessorCount, GPUmaxThreadsPerBlock;
	int Localnumel, Localnumno, LocalnumnoAcum;
	double delta_new;
	double *Matrix_K_h_Pt, *Vector_B_h_Pt, *Vector_X_h_Pt, *Matrix_M_h_Pt, *Delta_New_GPU_h_Pt, *Delta_Aux_GPU_h_Pt;
	double *Vector_D_h_A_Pt, *Vector_D_h_P_Pt, *DeltaVolStrain_h_Pt;
	int   *Vector_I_h_Pt, *eventId_1_h_Pt, *eventId_2_h_Pt;
	double *CoordElem_h_Pt, *DeltaStrain_h_Pt, *Material_Data_h_Pt, *DeltaStress_h_Pt;
	int   GlobalNumRow, GlobalNumTerm, BlockSize, RowSize, NumMultiProcessor, NumMaxThreadsBlock;
	double *MatPropQ1_h_Pt, *MatPropQ2_h_Pt, *MatPropQ3_h_Pt, *MatPropQ4_h_Pt, *MatPropQ5_h_Pt, *MatPropQ6_h_Pt, *MatPropQ7_h_Pt, *MatPropQ8_h_Pt;
	double *CoordQ1_h_Pt, *CoordQ2_h_Pt, *CoordQ3_h_Pt, *CoordQ4_h_Pt, *CoordQ5_h_Pt, *CoordQ6_h_Pt, *CoordQ7_h_Pt, *CoordQ8_h_Pt;
	int   *iiPosition_h_Pt, *Connect_h_Pt;
	double *Vector_B_d, *Vector_R_d, *Matrix_M_d, *delta_new_d, *delta_new_V_d, *Matrix_A_d, *Vector_Q_d, *delta_aux_d, *delta_aux_V_d;
	double *Vector_S_d, *delta_old_V_d;
	double *delta_new_V_h, *Matrix_A_h;
	int *BC_h_Pt;
	int   *Vector_I_d, *iiPosition_d, *Connect_d;
	int BlockSizeX, BlockSizeY, GPUNumThread;
	double *Vector_D_d, *delta_GPU_d, *Result_d,*Vector_D_Global_d, *Vector_X_d, *Vector_X_Global_d;
	double *CoordQ1_d, *CoordQ2_d, *CoordQ3_d, *CoordQ4_d, *CoordQ5_d, *CoordQ6_d, *CoordQ7_d, *CoordQ8_d;
	double *MatPropQ1_d, *MatPropQ2_d, *MatPropQ3_d, *MatPropQ4_d, *MatPropQ5_d, *MatPropQ6_d, *MatPropQ7_d, *MatPropQ8_d;
	double *CoordElem_d, *Material_Data_d, *DeltaStrain_d, *DeltaVolStrain_d, *DeltaStress_d;
	int *BC_d;
	int *Aux_1_h, *LocalnumnoAcumVar;


	//Partial sum for this GPU
	double *h_Sum;

} GPU_Struct;

//==============================================================================
// cGPU
//==============================================================================
typedef class cGPU *pcGPU, &rcGPU;

class cGPU
{
public:

	cGPU();
	~cGPU();
	int PrepareDataMultiGPU(const char* inputPath);
	void GPU_ReadInPutFile(int numno, int numel, int nummat, int numprop, int numdof, int numnoel, int numstr, int numbc, int numnl, char GPU_arqaux[80],
		double *X_h, double *Y_h, double *Z_h, int *BC_h, double *Material_Param_h, int *Connect_h, int *Material_Id_h);
	int AssemblyStiffnessMatrixOnGPU();
	int SolveConjugateGradientOnGPU(double *, double * );
	int EvaluateDeltaStrainStateOnGPU(double *DeltaStrainChr_h);
	int EvaluateDeltaStressStateOnGPU(double *StressChr_h);
	int ReleaseMemoryChronos();
	int EvaluateInitialStressState(int * , double * , double * );

	int nummat, numprop, numnoel, numbc, numnl, imax, jmax, kmax, numdofel;

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

	static GPU_Struct plan_FEM[4];
	static int numno, numdof, RowSize, numel, numstr;

	int i, j;
	int NumThread, NumMultProc;
	double time, SumNo, SumEl;
	int NumTerms, BlockDimX, BlockDimY;
	char GPU_arqaux[80];

	int nsi1, nsi2, nsj1, nsj2, nov, nun;

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

	// **************  Declaring Matrix Stiffness Variables:  **************

	int   *Vector_I_h, *BC_h, *Connect_h, *Material_Id_h, *Reservoir_Id_h;
	double *Vector_B_h, *Vector_X_h, *DeltaVolStrain_h;
	double *Material_Data_h, *Material_Param_h, *CoordElem_h, *Strain_h, *DeltaStrain_h, *Stress_h, *DeltaStress_h;
	double *Material_Density_h;
	double dE, dNu;
	double *X_h, *Y_h, *Z_h;

	// Host Allocation:
	double *MatPropQ1_h, *MatPropQ2_h, *MatPropQ3_h, *MatPropQ4_h, *MatPropQ5_h, *MatPropQ6_h, *MatPropQ7_h, *MatPropQ8_h;
	double *CoordQ1_h, *CoordQ2_h, *CoordQ3_h, *CoordQ4_h, *CoordQ5_h, *CoordQ6_h, *CoordQ7_h, *CoordQ8_h;
	int   *iiPosition_h;  // Evaluating M matrix = 1/K:

};


#endif

