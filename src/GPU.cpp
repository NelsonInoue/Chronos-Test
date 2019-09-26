/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\
*  Copyright (c) 2019 <GTEP> - All Rights Reserved                            *
*  This file is part of HERMES Project.                                       *
*  Unauthorized copying of this file, via any medium is strictly prohibited.  *
*  Proprietary and confidential.                                              *
*                                                                             *
*  Developers:                                                                *
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
#include "GPU.h"
#include <omp.h>
#include "defs.h"

//----------------------------------------------------
// External Functions for Multi GPUs Implementation
//----------------------------------------------------

extern "C" void AssemblyStiffnessGPUTetrahedron(int numdof, int numdofel, int numno, int RowSize,
int maxThreadsPerBlock, int Block33X, int BlockDimY, double *Matrix_K_h, double *Matrix_K_d, double *Matrix_K_Aux_d, int NumThread,
double *abcParam_h, double *abcParam_d, int MaxNodeAround, int MaxElemPerEdge);

extern "C" int GPU_ReadSize(int &numno, int &numel, int &nummat, int &numprop, int &numdof, int &numnoel, int &numstr, int &numbc, int &numnl,
int &imax, int &jmax, int &kmax, char GPU_arqaux[80]);

void GPU_Information(int & , double * );

extern "C" void ReadConcLoad(int numno, double *Vector_B_h, int *BC_h);

extern "C" void ReadConcLoadCouple(int numno, double *Vector_B_h, int *BC_h, double *NodeForceChr);

void EvaluateGPUMaterialProperty(int imax, int jmax, int kmax, int numno, double *MatPropQ1_h, double *MatPropQ2_h, double *MatPropQ3_h, double *MatPropQ4_h,
double *MatPropQ5_h, double *MatPropQ6_h, double *MatPropQ7_h, double *MatPropQ8_h, double *Material_Data_h, int *BC_h);

void EvaluateGPUCoordinate(int imax, int jmax, int kmax, double *X_h, double *Y_h, double *Z_h, int numno, double *CoordQ1_h, double *CoordQ2_h,
double *CoordQ3_h, double *CoordQ4_h, double *CoordQ5_h, double *CoordQ6_h, double *CoordQ7_h, double *CoordQ8_h);

void EvaluateGPURowNumbering(int imax, int jmax, int kmax, int RowSize, int *Vector_I_h);

void EvaluateiiPositionMultiGPU(int numdof, int numno, int , int , int , int, int RowSize, int *Vector_I_h, int *iiPosition_h);

void WriteDisplacement(int numno, double *Vector_X_h);

void WriteDisplacementCouple(int numno, double *Vector_X_h, double *VectorChr_X_h);

void WriteStrainState(int numel, double *Strain_h);

void WriteStrainStateCouple(int numel, double *Strain_h, double *StrainChr);

void CopyDeltaStrainFromGlobalToLocal(int numel, double *Strain_h, double *StrainChr);

void WriteStressState(int numel, double *Stress_h);

void EvaluateCoordStrainState(int numel, int numdof, int numnoel, int *Connect_h,  double *X_h,
double *Y_h,  double *Z_h,  double *CoordElem_h);

void EvaluateGPUFreeMemory();

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

extern "C" void Launch_AssemblyStiffness3DQ1(dim3 , dim3 , int , int , double * , double * , int * , double * );
extern "C" void Launch_AssemblyStiffness3DQ2(dim3 , dim3 , int , int , double * , double * , int * , double * );
extern "C" void Launch_AssemblyStiffness3DQ3(dim3 , dim3 , int , int , double * , double * , int * , double * );
extern "C" void Launch_AssemblyStiffness3DQ4(dim3 , dim3 , int , int , double * , double * , int * , double * );
extern "C" void Launch_AssemblyStiffness3DQ5(dim3 , dim3 , int , int , double * , double * , int * , double * );
extern "C" void Launch_AssemblyStiffness3DQ6(dim3 , dim3 , int , int , double * , double * , int * , double * );
extern "C" void Launch_AssemblyStiffness3DQ7(dim3 , dim3 , int , int , double * , double * , int * , double * );
extern "C" void Launch_AssemblyStiffness3DQ8(dim3 , dim3 , int , int , double * , double * , int * , double * );

extern "C" void Launch_TrasnsposeCoalesced(dim3 , dim3 , int , double * , double * );
extern "C" void Launch_EvaluateMatrixM(dim3 , dim3 , int , int , int * , double * , double * );

extern "C" void Launch_EvaluateDeltaStrainState(dim3 numBlocksStrain, dim3 threadsPerBlockStrain, int Localnumel, int RowSize,
double *CoordElem_d, int *Connect_d, double *Vector_X_d, double *DeltaStrain_d);

extern "C" void Launch_EvaluateDeltaVolStrain(dim3 numBlocksStrain, dim3 threadsPerBlockStrain, int Localnumel,
double *DeltaStrain_d, double *DeltaVolStrain_d);

extern "C" void Launch_EvaluateDeltaStressState(dim3 numBlocksStrain, dim3 threadsPerBlockStrain, int Localnumel,
double *Material_Data_d, double *DeltaStrain_d, double *DeltaStress_d);

using namespace std;
//pcGPU cGPU::GPU = NULL;

// -------------------------------------------------------------------------------------------------------------------------

GPU_Struct cGPU::plan_FEM[4];
int cGPU::numno, cGPU::numdof, cGPU::RowSize, cGPU::numel, cGPU::numstr;







// -------------------------------------------------------------------------------------------------------------------------




//==============================================================================
cGPU::cGPU()
{





}

//==============================================================================
cGPU::~cGPU()
{





}

//========================================================================================================
int cGPU::PrepareDataMultiGPU(const char* inputPath)
{
	if(inputPath)
		strcpy(GPU_arqaux, inputPath);

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

	// ========= Reset Device =========

	// Explicitly destroys and cleans up all resources associated with the current device in the current process.

	/*for(j=0; j<deviceCount; j++) {
		// Reset device j:
		cudaSetDevice(j); cudaDeviceReset();

	}*/

	// ================================

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
	BC_h = (int *)malloc(numno*numdof*(sizeof(int))); for(i=0; i<numno*numdof; i++) BC_h[i] = 0;

	// Allocating memory space for host material parameters vector
	Material_Param_h = (double *)malloc(nummat*numprop*(sizeof(double)));
	Material_Density_h = (double *)malloc(nummat*3*(sizeof(double)));

	// Allocating memory space for host element conectivity vector

	Material_Id_h = (int *)malloc(numel*(sizeof(int)));
	Reservoir_Id_h = (int *)malloc(nummat*(sizeof(int)));
	Connect_h  = (int *)malloc(numel*numnoel*(sizeof(int)));  // numnoel = Number of nodes per element

	// ========= Reading input file =========

	printf("\n");
	printf("         Reading input data file \n");
	printf("         ========================================= ");

	time = clock();

	GPU_ReadInPutFile(numno, numel, nummat, numprop, numdof, numnoel, numstr, numbc, numnl, GPU_arqaux, X_h, Y_h, Z_h,
		BC_h, Material_Param_h, Connect_h, Material_Id_h);

	time = clock()-time;
	printf("\n");
	printf("         Input File Reading Time: %0.3f s \n", time/1000);

	// ========= Allocating material property per element =========

	Material_Data_h = (double *)malloc(numprop*numel*sizeof(double));            // Allocating menoty space for host
	//cudaMalloc((void **) &Material_Data_d, numprop*numel*sizeof(double));       // Allocating menoty space for device

	for(i=0; i<numel; i++) {

		dE  = Material_Param_h[ Material_Id_h[i]-1         ];
		dNu = Material_Param_h[ Material_Id_h[i]-1 + nummat];

		Material_Data_h[i  ]     = dE;
		Material_Data_h[i+numel] = dNu;

	}

	// ------------------------------------------------------------------
	// Allocating menory space for host

	Vector_I_h   = (int *)malloc(numdof*numno*RowSize*(sizeof(int)));
	iiPosition_h = (int *)malloc(numdof*numno*(sizeof(int)));

	Vector_B_h  = (double *)malloc(numdof*numno*sizeof(double));
	Vector_X_h  = (double *)malloc(numdof*numno*sizeof(double));
	DeltaVolStrain_h = (double *)malloc(numel*sizeof(double));

	for(i=0; i<numdof*numno*RowSize; i++)
		Vector_I_h[i]=0;

	for(i=0; i<numdof*numno; i++)
		Vector_B_h[i]=0.;

	// ------------------------------------------------------------------



	// Allocating vector to storage coord x, y, and z for each element
	CoordElem_h  = (double *)malloc(numel*numdof*numnoel*sizeof(double));

	for(i=0; i<numel*numdof*numnoel; i++) CoordElem_h[i] = 0.;

	// ------------------------------------------------------------------

	// Strain state vector memory allocation
	Strain_h       = (double *)malloc(numel*numstr*sizeof(double));      // numstr = 6 = number of components of strain
	DeltaStrain_h  = (double *)malloc(numel*numstr*sizeof(double));      // numstr = 6 = number of components of strain

	// Stress state vector memory allocation
	Stress_h  = (double *)malloc(numel*numstr*sizeof(double));      // numstr = 6 = number of components of strain
	DeltaStress_h  = (double *)malloc(numel*numstr*sizeof(double));      // numstr = 6 = number of components of strain

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
			plan_FEM[i].Localnumno  = numno-SumNo;
			plan_FEM[i].Localnumel  = numel-SumEl;
			plan_FEM[i].LocalnumnoAcum  = plan_FEM[i-1].LocalnumnoAcum + plan_FEM[i].Localnumno;

		}
		else {
			plan_FEM[i].Localnumno  = int(numno/deviceCount);  // number of nodes for each GPU
			SumNo += plan_FEM[i].Localnumno;

			if(i == 0) plan_FEM[i].LocalnumnoAcum  = 0;  // number of nodes accumulated for each GPU
			else       plan_FEM[i].LocalnumnoAcum  = plan_FEM[i-1].LocalnumnoAcum + plan_FEM[i].Localnumno;

			plan_FEM[i].Localnumel  = int(numel/deviceCount);  // number of elements for each GPU
			SumEl += plan_FEM[i].Localnumel;
		}

	}

	// =================================================================

	for(i=0; i<deviceCount; i++) {
		plan_FEM[i].Aux_1_h           = (int *)malloc((deviceCount-1)*sizeof(int));
		plan_FEM[i].LocalnumnoAcumVar = (int *)malloc((deviceCount-1)*sizeof(int));
	}

	if(deviceCount == 4) {

		plan_FEM[0].Aux_1_h[0] = 1;
		plan_FEM[0].Aux_1_h[1] = 2;
		plan_FEM[0].Aux_1_h[2] = 3;

		plan_FEM[1].Aux_1_h[0] = 0;
		plan_FEM[1].Aux_1_h[1] = 2;
		plan_FEM[1].Aux_1_h[2] = 3;

		plan_FEM[2].Aux_1_h[0] = 0;
		plan_FEM[2].Aux_1_h[1] = 1;
		plan_FEM[2].Aux_1_h[2] = 3;

		plan_FEM[3].Aux_1_h[0] = 0;
		plan_FEM[3].Aux_1_h[1] = 1;
		plan_FEM[3].Aux_1_h[2] = 2;

		// -----------------------------------------------------------------

		plan_FEM[0].LocalnumnoAcumVar[0] = plan_FEM[0].Localnumno;
		plan_FEM[0].LocalnumnoAcumVar[1] = plan_FEM[0].Localnumno + plan_FEM[1].Localnumno;
		plan_FEM[0].LocalnumnoAcumVar[2] = plan_FEM[0].Localnumno + plan_FEM[1].Localnumno + plan_FEM[2].Localnumno;

		plan_FEM[1].LocalnumnoAcumVar[0] = 0;
		plan_FEM[1].LocalnumnoAcumVar[1] = plan_FEM[0].Localnumno + plan_FEM[1].Localnumno;
		plan_FEM[1].LocalnumnoAcumVar[2] = plan_FEM[0].Localnumno + plan_FEM[1].Localnumno + plan_FEM[2].Localnumno;

		plan_FEM[2].LocalnumnoAcumVar[0] = 0;
		plan_FEM[2].LocalnumnoAcumVar[1] = plan_FEM[0].Localnumno;
		plan_FEM[2].LocalnumnoAcumVar[2] = plan_FEM[0].Localnumno + plan_FEM[1].Localnumno + plan_FEM[2].Localnumno;

		plan_FEM[3].LocalnumnoAcumVar[0] = 0;
		plan_FEM[3].LocalnumnoAcumVar[1] = plan_FEM[0].Localnumno;
		plan_FEM[3].LocalnumnoAcumVar[2] = plan_FEM[0].Localnumno + plan_FEM[1].Localnumno;

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

	EvaluateiiPositionMultiGPU(numdof, numno, plan_FEM[0].Localnumno, plan_FEM[1].Localnumno, plan_FEM[2].Localnumno, plan_FEM[3].Localnumno,
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

		plan_FEM[i].MatPropQ1_h_Pt = MatPropQ1_h + gpuBaseMatProp;
		plan_FEM[i].MatPropQ2_h_Pt = MatPropQ2_h + gpuBaseMatProp;
		plan_FEM[i].MatPropQ3_h_Pt = MatPropQ3_h + gpuBaseMatProp;
		plan_FEM[i].MatPropQ4_h_Pt = MatPropQ4_h + gpuBaseMatProp;
		plan_FEM[i].MatPropQ5_h_Pt = MatPropQ5_h + gpuBaseMatProp;
		plan_FEM[i].MatPropQ6_h_Pt = MatPropQ6_h + gpuBaseMatProp;
		plan_FEM[i].MatPropQ7_h_Pt = MatPropQ7_h + gpuBaseMatProp;
		plan_FEM[i].MatPropQ8_h_Pt = MatPropQ8_h + gpuBaseMatProp;

		plan_FEM[i].CoordQ1_h_Pt = CoordQ1_h + gpuBaseCoord;
		plan_FEM[i].CoordQ2_h_Pt = CoordQ2_h + gpuBaseCoord;
		plan_FEM[i].CoordQ3_h_Pt = CoordQ3_h + gpuBaseCoord;
		plan_FEM[i].CoordQ4_h_Pt = CoordQ4_h + gpuBaseCoord;
		plan_FEM[i].CoordQ5_h_Pt = CoordQ5_h + gpuBaseCoord;
		plan_FEM[i].CoordQ6_h_Pt = CoordQ6_h + gpuBaseCoord;
		plan_FEM[i].CoordQ7_h_Pt = CoordQ7_h + gpuBaseCoord;
		plan_FEM[i].CoordQ8_h_Pt = CoordQ8_h + gpuBaseCoord;

		plan_FEM[i].iiPosition_h_Pt = iiPosition_h + gpuBaseiiPos;

		plan_FEM[i].BC_h_Pt = BC_h + gpuBaseCoord;

		gpuBaseMatProp += plan_FEM[i].Localnumno;
		gpuBaseCoord   += plan_FEM[i].Localnumno;
		gpuBaseiiPos   += plan_FEM[i].Localnumno*numdof;

	}

	// =================================================================

	// Allocating Device memory:

	for(j=0; j<deviceCount; j++) {

		// Device 0:
		cudaSetDevice(j);              // Sets device j as the current device

		// Allocating Material Properties data:
		cudaMalloc((void **) &plan_FEM[j].MatPropQ1_d, plan_FEM[j].Localnumno*numprop*sizeof(double));
		cudaMalloc((void **) &plan_FEM[j].MatPropQ2_d, plan_FEM[j].Localnumno*numprop*sizeof(double));
		cudaMalloc((void **) &plan_FEM[j].MatPropQ3_d, plan_FEM[j].Localnumno*numprop*sizeof(double));
		cudaMalloc((void **) &plan_FEM[j].MatPropQ4_d, plan_FEM[j].Localnumno*numprop*sizeof(double));
		cudaMalloc((void **) &plan_FEM[j].MatPropQ5_d, plan_FEM[j].Localnumno*numprop*sizeof(double));
		cudaMalloc((void **) &plan_FEM[j].MatPropQ6_d, plan_FEM[j].Localnumno*numprop*sizeof(double));
		cudaMalloc((void **) &plan_FEM[j].MatPropQ7_d, plan_FEM[j].Localnumno*numprop*sizeof(double));
		cudaMalloc((void **) &plan_FEM[j].MatPropQ8_d, plan_FEM[j].Localnumno*numprop*sizeof(double));

		// Allocating Coodinate data:
		cudaMalloc((void **) &plan_FEM[j].CoordQ1_d, plan_FEM[j].Localnumno*numdof*numnoel*sizeof(double));
		cudaMalloc((void **) &plan_FEM[j].CoordQ2_d, plan_FEM[j].Localnumno*numdof*numnoel*sizeof(double));
		cudaMalloc((void **) &plan_FEM[j].CoordQ3_d, plan_FEM[j].Localnumno*numdof*numnoel*sizeof(double));
		cudaMalloc((void **) &plan_FEM[j].CoordQ4_d, plan_FEM[j].Localnumno*numdof*numnoel*sizeof(double));
		cudaMalloc((void **) &plan_FEM[j].CoordQ5_d, plan_FEM[j].Localnumno*numdof*numnoel*sizeof(double));
		cudaMalloc((void **) &plan_FEM[j].CoordQ6_d, plan_FEM[j].Localnumno*numdof*numnoel*sizeof(double));
		cudaMalloc((void **) &plan_FEM[j].CoordQ7_d, plan_FEM[j].Localnumno*numdof*numnoel*sizeof(double));
		cudaMalloc((void **) &plan_FEM[j].CoordQ8_d, plan_FEM[j].Localnumno*numdof*numnoel*sizeof(double));

		// Allocating ii position:
		cudaMalloc((void **) &plan_FEM[j].iiPosition_d, numdof*plan_FEM[j].Localnumno*sizeof(int));

		// Allocating BC:
		cudaMalloc((void **) &plan_FEM[j].BC_d, plan_FEM[j].Localnumno*numdof*sizeof(int));

		for(i=0; i<numprop; i++) {

			// Copying Material Properties data from host to device:
			cudaMemcpy(plan_FEM[j].MatPropQ1_d + i*plan_FEM[j].Localnumno, plan_FEM[j].MatPropQ1_h_Pt + i*numno, plan_FEM[j].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  // To copy from host memory to device memory;
			cudaMemcpy(plan_FEM[j].MatPropQ2_d + i*plan_FEM[j].Localnumno, plan_FEM[j].MatPropQ2_h_Pt + i*numno, plan_FEM[j].Localnumno*sizeof(double), cudaMemcpyHostToDevice);
			cudaMemcpy(plan_FEM[j].MatPropQ3_d + i*plan_FEM[j].Localnumno, plan_FEM[j].MatPropQ3_h_Pt + i*numno, plan_FEM[j].Localnumno*sizeof(double), cudaMemcpyHostToDevice);
			cudaMemcpy(plan_FEM[j].MatPropQ4_d + i*plan_FEM[j].Localnumno, plan_FEM[j].MatPropQ4_h_Pt + i*numno, plan_FEM[j].Localnumno*sizeof(double), cudaMemcpyHostToDevice);
			cudaMemcpy(plan_FEM[j].MatPropQ5_d + i*plan_FEM[j].Localnumno, plan_FEM[j].MatPropQ5_h_Pt + i*numno, plan_FEM[j].Localnumno*sizeof(double), cudaMemcpyHostToDevice);
			cudaMemcpy(plan_FEM[j].MatPropQ6_d + i*plan_FEM[j].Localnumno, plan_FEM[j].MatPropQ6_h_Pt + i*numno, plan_FEM[j].Localnumno*sizeof(double), cudaMemcpyHostToDevice);
			cudaMemcpy(plan_FEM[j].MatPropQ7_d + i*plan_FEM[j].Localnumno, plan_FEM[j].MatPropQ7_h_Pt + i*numno, plan_FEM[j].Localnumno*sizeof(double), cudaMemcpyHostToDevice);
			cudaMemcpy(plan_FEM[j].MatPropQ8_d + i*plan_FEM[j].Localnumno, plan_FEM[j].MatPropQ8_h_Pt + i*numno, plan_FEM[j].Localnumno*sizeof(double), cudaMemcpyHostToDevice);

		}

		for(i=0; i<numdof*numnoel; i++) {

			// Copying Coordinate data from host to device:
			cudaMemcpy(plan_FEM[j].CoordQ1_d + i*plan_FEM[j].Localnumno, plan_FEM[j].CoordQ1_h_Pt + i*numno, plan_FEM[j].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  // To copy from host memory to device memory;
			cudaMemcpy(plan_FEM[j].CoordQ2_d + i*plan_FEM[j].Localnumno, plan_FEM[j].CoordQ2_h_Pt + i*numno, plan_FEM[j].Localnumno*sizeof(double), cudaMemcpyHostToDevice);
			cudaMemcpy(plan_FEM[j].CoordQ3_d + i*plan_FEM[j].Localnumno, plan_FEM[j].CoordQ3_h_Pt + i*numno, plan_FEM[j].Localnumno*sizeof(double), cudaMemcpyHostToDevice);
			cudaMemcpy(plan_FEM[j].CoordQ4_d + i*plan_FEM[j].Localnumno, plan_FEM[j].CoordQ4_h_Pt + i*numno, plan_FEM[j].Localnumno*sizeof(double), cudaMemcpyHostToDevice);
			cudaMemcpy(plan_FEM[j].CoordQ5_d + i*plan_FEM[j].Localnumno, plan_FEM[j].CoordQ5_h_Pt + i*numno, plan_FEM[j].Localnumno*sizeof(double), cudaMemcpyHostToDevice);
			cudaMemcpy(plan_FEM[j].CoordQ6_d + i*plan_FEM[j].Localnumno, plan_FEM[j].CoordQ6_h_Pt + i*numno, plan_FEM[j].Localnumno*sizeof(double), cudaMemcpyHostToDevice);
			cudaMemcpy(plan_FEM[j].CoordQ7_d + i*plan_FEM[j].Localnumno, plan_FEM[j].CoordQ7_h_Pt + i*numno, plan_FEM[j].Localnumno*sizeof(double), cudaMemcpyHostToDevice);
			cudaMemcpy(plan_FEM[j].CoordQ8_d + i*plan_FEM[j].Localnumno, plan_FEM[j].CoordQ8_h_Pt + i*numno, plan_FEM[j].Localnumno*sizeof(double), cudaMemcpyHostToDevice);

		}

		for(i=0; i<numdof; i++) {

			// Copying Material Properties data from host to device:
			cudaMemcpy(plan_FEM[j].BC_d + i*plan_FEM[j].Localnumno, plan_FEM[j].BC_h_Pt + i*numno, plan_FEM[j].Localnumno*sizeof(int), cudaMemcpyHostToDevice);  // To copy from host memory to device memory;

		}

		// Copying ii position data from host to device:
		cudaMemcpy(plan_FEM[j].iiPosition_d, plan_FEM[j].iiPosition_h_Pt, numdof*plan_FEM[j].Localnumno*sizeof(int), cudaMemcpyHostToDevice);    // To copy from host memory to device memory;

	}

	/*// Allocating Device memory:

	// Device 0:
	cudaSetDevice(0);              // Sets device 0 as the current device

	// Allocating Material Properties data:
	cudaMalloc((void **) &plan_FEM[0].MatPropQ1_d, plan_FEM[0].Localnumno*numprop*sizeof(double));
	cudaMalloc((void **) &plan_FEM[0].MatPropQ2_d, plan_FEM[0].Localnumno*numprop*sizeof(double));
	cudaMalloc((void **) &plan_FEM[0].MatPropQ3_d, plan_FEM[0].Localnumno*numprop*sizeof(double));
	cudaMalloc((void **) &plan_FEM[0].MatPropQ4_d, plan_FEM[0].Localnumno*numprop*sizeof(double));
	cudaMalloc((void **) &plan_FEM[0].MatPropQ5_d, plan_FEM[0].Localnumno*numprop*sizeof(double));
	cudaMalloc((void **) &plan_FEM[0].MatPropQ6_d, plan_FEM[0].Localnumno*numprop*sizeof(double));
	cudaMalloc((void **) &plan_FEM[0].MatPropQ7_d, plan_FEM[0].Localnumno*numprop*sizeof(double));
	cudaMalloc((void **) &plan_FEM[0].MatPropQ8_d, plan_FEM[0].Localnumno*numprop*sizeof(double));

	// Allocating Coodinate data:
	cudaMalloc((void **) &plan_FEM[0].CoordQ1_d, plan_FEM[0].Localnumno*numdof*numnoel*sizeof(double));
	cudaMalloc((void **) &plan_FEM[0].CoordQ2_d, plan_FEM[0].Localnumno*numdof*numnoel*sizeof(double));
	cudaMalloc((void **) &plan_FEM[0].CoordQ3_d, plan_FEM[0].Localnumno*numdof*numnoel*sizeof(double));
	cudaMalloc((void **) &plan_FEM[0].CoordQ4_d, plan_FEM[0].Localnumno*numdof*numnoel*sizeof(double));
	cudaMalloc((void **) &plan_FEM[0].CoordQ5_d, plan_FEM[0].Localnumno*numdof*numnoel*sizeof(double));
	cudaMalloc((void **) &plan_FEM[0].CoordQ6_d, plan_FEM[0].Localnumno*numdof*numnoel*sizeof(double));
	cudaMalloc((void **) &plan_FEM[0].CoordQ7_d, plan_FEM[0].Localnumno*numdof*numnoel*sizeof(double));
	cudaMalloc((void **) &plan_FEM[0].CoordQ8_d, plan_FEM[0].Localnumno*numdof*numnoel*sizeof(double));

	// Allocating ii position:
	cudaMalloc((void **) &plan_FEM[0].iiPosition_d, numdof*plan_FEM[0].Localnumno*sizeof(int));

	for(i=0; i<numprop; i++) {

		// Copying Material Properties data from host to device:
		cudaMemcpy(plan_FEM[0].MatPropQ1_d + i*plan_FEM[0].Localnumno, plan_FEM[0].MatPropQ1_h_Pt + i*numno, plan_FEM[0].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  // To copy from host memory to device memory;
		cudaMemcpy(plan_FEM[0].MatPropQ2_d + i*plan_FEM[0].Localnumno, plan_FEM[0].MatPropQ2_h_Pt + i*numno, plan_FEM[0].Localnumno*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(plan_FEM[0].MatPropQ3_d + i*plan_FEM[0].Localnumno, plan_FEM[0].MatPropQ3_h_Pt + i*numno, plan_FEM[0].Localnumno*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(plan_FEM[0].MatPropQ4_d + i*plan_FEM[0].Localnumno, plan_FEM[0].MatPropQ4_h_Pt + i*numno, plan_FEM[0].Localnumno*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(plan_FEM[0].MatPropQ5_d + i*plan_FEM[0].Localnumno, plan_FEM[0].MatPropQ5_h_Pt + i*numno, plan_FEM[0].Localnumno*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(plan_FEM[0].MatPropQ6_d + i*plan_FEM[0].Localnumno, plan_FEM[0].MatPropQ6_h_Pt + i*numno, plan_FEM[0].Localnumno*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(plan_FEM[0].MatPropQ7_d + i*plan_FEM[0].Localnumno, plan_FEM[0].MatPropQ7_h_Pt + i*numno, plan_FEM[0].Localnumno*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(plan_FEM[0].MatPropQ8_d + i*plan_FEM[0].Localnumno, plan_FEM[0].MatPropQ8_h_Pt + i*numno, plan_FEM[0].Localnumno*sizeof(double), cudaMemcpyHostToDevice);

	}

	for(i=0; i<numdof*numnoel; i++) {

		// Copying Coordinate data from host to device:
		cudaMemcpy(plan_FEM[0].CoordQ1_d + i*plan_FEM[0].Localnumno, plan_FEM[0].CoordQ1_h_Pt + i*numno, plan_FEM[0].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  // To copy from host memory to device memory;
		cudaMemcpy(plan_FEM[0].CoordQ2_d + i*plan_FEM[0].Localnumno, plan_FEM[0].CoordQ2_h_Pt + i*numno, plan_FEM[0].Localnumno*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(plan_FEM[0].CoordQ3_d + i*plan_FEM[0].Localnumno, plan_FEM[0].CoordQ3_h_Pt + i*numno, plan_FEM[0].Localnumno*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(plan_FEM[0].CoordQ4_d + i*plan_FEM[0].Localnumno, plan_FEM[0].CoordQ4_h_Pt + i*numno, plan_FEM[0].Localnumno*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(plan_FEM[0].CoordQ5_d + i*plan_FEM[0].Localnumno, plan_FEM[0].CoordQ5_h_Pt + i*numno, plan_FEM[0].Localnumno*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(plan_FEM[0].CoordQ6_d + i*plan_FEM[0].Localnumno, plan_FEM[0].CoordQ6_h_Pt + i*numno, plan_FEM[0].Localnumno*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(plan_FEM[0].CoordQ7_d + i*plan_FEM[0].Localnumno, plan_FEM[0].CoordQ7_h_Pt + i*numno, plan_FEM[0].Localnumno*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(plan_FEM[0].CoordQ8_d + i*plan_FEM[0].Localnumno, plan_FEM[0].CoordQ8_h_Pt + i*numno, plan_FEM[0].Localnumno*sizeof(double), cudaMemcpyHostToDevice);

	}

	// Copying ii position data from host to device:
	cudaMemcpy(plan_FEM[0].iiPosition_d, plan_FEM[0].iiPosition_h_Pt, numdof*plan_FEM[0].Localnumno*sizeof(int), cudaMemcpyHostToDevice);    // To copy from host memory to device memory;

	// -----------------------------------------------------------------

	// Device 1:
	cudaSetDevice(1);              // Sets device 1 as the current device

	// Allocating Material Properties data:
	cudaMalloc((void **) &plan_FEM[1].MatPropQ1_d, plan_FEM[1].Localnumno*numprop*sizeof(double));
	cudaMalloc((void **) &plan_FEM[1].MatPropQ2_d, plan_FEM[1].Localnumno*numprop*sizeof(double));
	cudaMalloc((void **) &plan_FEM[1].MatPropQ3_d, plan_FEM[1].Localnumno*numprop*sizeof(double));
	cudaMalloc((void **) &plan_FEM[1].MatPropQ4_d, plan_FEM[1].Localnumno*numprop*sizeof(double));
	cudaMalloc((void **) &plan_FEM[1].MatPropQ5_d, plan_FEM[1].Localnumno*numprop*sizeof(double));
	cudaMalloc((void **) &plan_FEM[1].MatPropQ6_d, plan_FEM[1].Localnumno*numprop*sizeof(double));
	cudaMalloc((void **) &plan_FEM[1].MatPropQ7_d, plan_FEM[1].Localnumno*numprop*sizeof(double));
	cudaMalloc((void **) &plan_FEM[1].MatPropQ8_d, plan_FEM[1].Localnumno*numprop*sizeof(double));

	// Allocating Coordinate data:
	cudaMalloc((void **) &plan_FEM[1].CoordQ1_d, plan_FEM[1].Localnumno*numdof*numnoel*sizeof(double));
	cudaMalloc((void **) &plan_FEM[1].CoordQ2_d, plan_FEM[1].Localnumno*numdof*numnoel*sizeof(double));
	cudaMalloc((void **) &plan_FEM[1].CoordQ3_d, plan_FEM[1].Localnumno*numdof*numnoel*sizeof(double));
	cudaMalloc((void **) &plan_FEM[1].CoordQ4_d, plan_FEM[1].Localnumno*numdof*numnoel*sizeof(double));
	cudaMalloc((void **) &plan_FEM[1].CoordQ5_d, plan_FEM[1].Localnumno*numdof*numnoel*sizeof(double));
	cudaMalloc((void **) &plan_FEM[1].CoordQ6_d, plan_FEM[1].Localnumno*numdof*numnoel*sizeof(double));
	cudaMalloc((void **) &plan_FEM[1].CoordQ7_d, plan_FEM[1].Localnumno*numdof*numnoel*sizeof(double));
	cudaMalloc((void **) &plan_FEM[1].CoordQ8_d, plan_FEM[1].Localnumno*numdof*numnoel*sizeof(double));

	// Allocating ii position:
	cudaMalloc((void **) &plan_FEM[1].iiPosition_d, numdof*plan_FEM[1].Localnumno*sizeof(int));

	for(i=0; i<numprop; i++) {

		// Copying Material Properties data from host to device:
		cudaMemcpy(plan_FEM[1].MatPropQ1_d + i*plan_FEM[1].Localnumno, plan_FEM[1].MatPropQ1_h_Pt + i*numno, plan_FEM[1].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  // To copy from host memory to device memory;
		cudaMemcpy(plan_FEM[1].MatPropQ2_d + i*plan_FEM[1].Localnumno, plan_FEM[1].MatPropQ2_h_Pt + i*numno, plan_FEM[1].Localnumno*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(plan_FEM[1].MatPropQ3_d + i*plan_FEM[1].Localnumno, plan_FEM[1].MatPropQ3_h_Pt + i*numno, plan_FEM[1].Localnumno*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(plan_FEM[1].MatPropQ4_d + i*plan_FEM[1].Localnumno, plan_FEM[1].MatPropQ4_h_Pt + i*numno, plan_FEM[1].Localnumno*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(plan_FEM[1].MatPropQ5_d + i*plan_FEM[1].Localnumno, plan_FEM[1].MatPropQ5_h_Pt + i*numno, plan_FEM[1].Localnumno*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(plan_FEM[1].MatPropQ6_d + i*plan_FEM[1].Localnumno, plan_FEM[1].MatPropQ6_h_Pt + i*numno, plan_FEM[1].Localnumno*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(plan_FEM[1].MatPropQ7_d + i*plan_FEM[1].Localnumno, plan_FEM[1].MatPropQ7_h_Pt + i*numno, plan_FEM[1].Localnumno*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(plan_FEM[1].MatPropQ8_d + i*plan_FEM[1].Localnumno, plan_FEM[1].MatPropQ8_h_Pt + i*numno, plan_FEM[1].Localnumno*sizeof(double), cudaMemcpyHostToDevice);

	}

	for(i=0; i<numdof*numnoel; i++) {

		// Copying Coordinate data from host to device:
		cudaMemcpy(plan_FEM[1].CoordQ1_d + i*plan_FEM[1].Localnumno, plan_FEM[1].CoordQ1_h_Pt + i*numno, plan_FEM[1].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  // To copy from host memory to device memory;
		cudaMemcpy(plan_FEM[1].CoordQ2_d + i*plan_FEM[1].Localnumno, plan_FEM[1].CoordQ2_h_Pt + i*numno, plan_FEM[1].Localnumno*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(plan_FEM[1].CoordQ3_d + i*plan_FEM[1].Localnumno, plan_FEM[1].CoordQ3_h_Pt + i*numno, plan_FEM[1].Localnumno*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(plan_FEM[1].CoordQ4_d + i*plan_FEM[1].Localnumno, plan_FEM[1].CoordQ4_h_Pt + i*numno, plan_FEM[1].Localnumno*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(plan_FEM[1].CoordQ5_d + i*plan_FEM[1].Localnumno, plan_FEM[1].CoordQ5_h_Pt + i*numno, plan_FEM[1].Localnumno*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(plan_FEM[1].CoordQ6_d + i*plan_FEM[1].Localnumno, plan_FEM[1].CoordQ6_h_Pt + i*numno, plan_FEM[1].Localnumno*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(plan_FEM[1].CoordQ7_d + i*plan_FEM[1].Localnumno, plan_FEM[1].CoordQ7_h_Pt + i*numno, plan_FEM[1].Localnumno*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(plan_FEM[1].CoordQ8_d + i*plan_FEM[1].Localnumno, plan_FEM[1].CoordQ8_h_Pt + i*numno, plan_FEM[1].Localnumno*sizeof(double), cudaMemcpyHostToDevice);

	}

	// Copying ii position data from host to device:
	cudaMemcpy(plan_FEM[1].iiPosition_d, plan_FEM[1].iiPosition_h_Pt, numdof*plan_FEM[1].Localnumno*sizeof(int), cudaMemcpyHostToDevice);    // To copy from host memory to device memory;

	// -----------------------------------------------------------------

	// Device 2:
	cudaSetDevice(2);              // Sets device 2 as the current device

	// Allocating Material Properties data:
	cudaMalloc((void **) &plan_FEM[2].MatPropQ1_d, plan_FEM[2].Localnumno*numprop*sizeof(double));
	cudaMalloc((void **) &plan_FEM[2].MatPropQ2_d, plan_FEM[2].Localnumno*numprop*sizeof(double));
	cudaMalloc((void **) &plan_FEM[2].MatPropQ3_d, plan_FEM[2].Localnumno*numprop*sizeof(double));
	cudaMalloc((void **) &plan_FEM[2].MatPropQ4_d, plan_FEM[2].Localnumno*numprop*sizeof(double));
	cudaMalloc((void **) &plan_FEM[2].MatPropQ5_d, plan_FEM[2].Localnumno*numprop*sizeof(double));
	cudaMalloc((void **) &plan_FEM[2].MatPropQ6_d, plan_FEM[2].Localnumno*numprop*sizeof(double));
	cudaMalloc((void **) &plan_FEM[2].MatPropQ7_d, plan_FEM[2].Localnumno*numprop*sizeof(double));
	cudaMalloc((void **) &plan_FEM[2].MatPropQ8_d, plan_FEM[2].Localnumno*numprop*sizeof(double));

	// Allocating Coodinate data:
	cudaMalloc((void **) &plan_FEM[2].CoordQ1_d, plan_FEM[2].Localnumno*numdof*numnoel*sizeof(double));
	cudaMalloc((void **) &plan_FEM[2].CoordQ2_d, plan_FEM[2].Localnumno*numdof*numnoel*sizeof(double));
	cudaMalloc((void **) &plan_FEM[2].CoordQ3_d, plan_FEM[2].Localnumno*numdof*numnoel*sizeof(double));
	cudaMalloc((void **) &plan_FEM[2].CoordQ4_d, plan_FEM[2].Localnumno*numdof*numnoel*sizeof(double));
	cudaMalloc((void **) &plan_FEM[2].CoordQ5_d, plan_FEM[2].Localnumno*numdof*numnoel*sizeof(double));
	cudaMalloc((void **) &plan_FEM[2].CoordQ6_d, plan_FEM[2].Localnumno*numdof*numnoel*sizeof(double));
	cudaMalloc((void **) &plan_FEM[2].CoordQ7_d, plan_FEM[2].Localnumno*numdof*numnoel*sizeof(double));
	cudaMalloc((void **) &plan_FEM[2].CoordQ8_d, plan_FEM[2].Localnumno*numdof*numnoel*sizeof(double));

	// Allocating ii position:
	cudaMalloc((void **) &plan_FEM[2].iiPosition_d, numdof*plan_FEM[2].Localnumno*sizeof(int));

	// Copying Material Properties data from host to device:
	for(i=0; i<numprop; i++) {

		// Copying Material Properties data from host to device:
		cudaMemcpy(plan_FEM[2].MatPropQ1_d + i*plan_FEM[2].Localnumno, plan_FEM[2].MatPropQ1_h_Pt + i*numno, plan_FEM[2].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  // To copy from host memory to device memory;
		cudaMemcpy(plan_FEM[2].MatPropQ2_d + i*plan_FEM[2].Localnumno, plan_FEM[2].MatPropQ2_h_Pt + i*numno, plan_FEM[2].Localnumno*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(plan_FEM[2].MatPropQ3_d + i*plan_FEM[2].Localnumno, plan_FEM[2].MatPropQ3_h_Pt + i*numno, plan_FEM[2].Localnumno*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(plan_FEM[2].MatPropQ4_d + i*plan_FEM[2].Localnumno, plan_FEM[2].MatPropQ4_h_Pt + i*numno, plan_FEM[2].Localnumno*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(plan_FEM[2].MatPropQ5_d + i*plan_FEM[2].Localnumno, plan_FEM[2].MatPropQ5_h_Pt + i*numno, plan_FEM[2].Localnumno*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(plan_FEM[2].MatPropQ6_d + i*plan_FEM[2].Localnumno, plan_FEM[2].MatPropQ6_h_Pt + i*numno, plan_FEM[2].Localnumno*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(plan_FEM[2].MatPropQ7_d + i*plan_FEM[2].Localnumno, plan_FEM[2].MatPropQ7_h_Pt + i*numno, plan_FEM[2].Localnumno*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(plan_FEM[2].MatPropQ8_d + i*plan_FEM[2].Localnumno, plan_FEM[2].MatPropQ8_h_Pt + i*numno, plan_FEM[2].Localnumno*sizeof(double), cudaMemcpyHostToDevice);

	}

	for(i=0; i<numdof*numnoel; i++) {

		// Copying Coordinate data from host to device:
		cudaMemcpy(plan_FEM[2].CoordQ1_d + i*plan_FEM[2].Localnumno, plan_FEM[2].CoordQ1_h_Pt + i*numno, plan_FEM[2].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  // To copy from host memory to device memory;
		cudaMemcpy(plan_FEM[2].CoordQ2_d + i*plan_FEM[2].Localnumno, plan_FEM[2].CoordQ2_h_Pt + i*numno, plan_FEM[2].Localnumno*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(plan_FEM[2].CoordQ3_d + i*plan_FEM[2].Localnumno, plan_FEM[2].CoordQ3_h_Pt + i*numno, plan_FEM[2].Localnumno*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(plan_FEM[2].CoordQ4_d + i*plan_FEM[2].Localnumno, plan_FEM[2].CoordQ4_h_Pt + i*numno, plan_FEM[2].Localnumno*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(plan_FEM[2].CoordQ5_d + i*plan_FEM[2].Localnumno, plan_FEM[2].CoordQ5_h_Pt + i*numno, plan_FEM[2].Localnumno*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(plan_FEM[2].CoordQ6_d + i*plan_FEM[2].Localnumno, plan_FEM[2].CoordQ6_h_Pt + i*numno, plan_FEM[2].Localnumno*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(plan_FEM[2].CoordQ7_d + i*plan_FEM[2].Localnumno, plan_FEM[2].CoordQ7_h_Pt + i*numno, plan_FEM[2].Localnumno*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(plan_FEM[2].CoordQ8_d + i*plan_FEM[2].Localnumno, plan_FEM[2].CoordQ8_h_Pt + i*numno, plan_FEM[2].Localnumno*sizeof(double), cudaMemcpyHostToDevice);

	}

	// Copying ii position data from host to device:
	cudaMemcpy(plan_FEM[2].iiPosition_d, plan_FEM[2].iiPosition_h_Pt, numdof*plan_FEM[2].Localnumno*sizeof(int), cudaMemcpyHostToDevice);    // To copy from host memory to device memory;

	// -----------------------------------------------------------------

	// Device 3:
	cudaSetDevice(3);              // Sets device 3 as the current device

	// Allocating Material Properties data:
	cudaMalloc((void **) &plan_FEM[3].MatPropQ1_d, plan_FEM[3].Localnumno*numprop*sizeof(double));
	cudaMalloc((void **) &plan_FEM[3].MatPropQ2_d, plan_FEM[3].Localnumno*numprop*sizeof(double));
	cudaMalloc((void **) &plan_FEM[3].MatPropQ3_d, plan_FEM[3].Localnumno*numprop*sizeof(double));
	cudaMalloc((void **) &plan_FEM[3].MatPropQ4_d, plan_FEM[3].Localnumno*numprop*sizeof(double));
	cudaMalloc((void **) &plan_FEM[3].MatPropQ5_d, plan_FEM[3].Localnumno*numprop*sizeof(double));
	cudaMalloc((void **) &plan_FEM[3].MatPropQ6_d, plan_FEM[3].Localnumno*numprop*sizeof(double));
	cudaMalloc((void **) &plan_FEM[3].MatPropQ7_d, plan_FEM[3].Localnumno*numprop*sizeof(double));
	cudaMalloc((void **) &plan_FEM[3].MatPropQ8_d, plan_FEM[3].Localnumno*numprop*sizeof(double));

	// Allocating Coodinate data:
	cudaMalloc((void **) &plan_FEM[3].CoordQ1_d, plan_FEM[3].Localnumno*numdof*numnoel*sizeof(double));
	cudaMalloc((void **) &plan_FEM[3].CoordQ2_d, plan_FEM[3].Localnumno*numdof*numnoel*sizeof(double));
	cudaMalloc((void **) &plan_FEM[3].CoordQ3_d, plan_FEM[3].Localnumno*numdof*numnoel*sizeof(double));
	cudaMalloc((void **) &plan_FEM[3].CoordQ4_d, plan_FEM[3].Localnumno*numdof*numnoel*sizeof(double));
	cudaMalloc((void **) &plan_FEM[3].CoordQ5_d, plan_FEM[3].Localnumno*numdof*numnoel*sizeof(double));
	cudaMalloc((void **) &plan_FEM[3].CoordQ6_d, plan_FEM[3].Localnumno*numdof*numnoel*sizeof(double));
	cudaMalloc((void **) &plan_FEM[3].CoordQ7_d, plan_FEM[3].Localnumno*numdof*numnoel*sizeof(double));
	cudaMalloc((void **) &plan_FEM[3].CoordQ8_d, plan_FEM[3].Localnumno*numdof*numnoel*sizeof(double));

	// Allocating ii position:
	cudaMalloc((void **) &plan_FEM[3].iiPosition_d, numdof*plan_FEM[3].Localnumno*sizeof(int));

	for(i=0; i<numprop; i++) {

		// Copying Material Properties data from host to device:
		cudaMemcpy(plan_FEM[3].MatPropQ1_d + i*plan_FEM[3].Localnumno, plan_FEM[3].MatPropQ1_h_Pt + i*numno, plan_FEM[3].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  // To copy from host memory to device memory;
		cudaMemcpy(plan_FEM[3].MatPropQ2_d + i*plan_FEM[3].Localnumno, plan_FEM[3].MatPropQ2_h_Pt + i*numno, plan_FEM[3].Localnumno*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(plan_FEM[3].MatPropQ3_d + i*plan_FEM[3].Localnumno, plan_FEM[3].MatPropQ3_h_Pt + i*numno, plan_FEM[3].Localnumno*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(plan_FEM[3].MatPropQ4_d + i*plan_FEM[3].Localnumno, plan_FEM[3].MatPropQ4_h_Pt + i*numno, plan_FEM[3].Localnumno*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(plan_FEM[3].MatPropQ5_d + i*plan_FEM[3].Localnumno, plan_FEM[3].MatPropQ5_h_Pt + i*numno, plan_FEM[3].Localnumno*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(plan_FEM[3].MatPropQ6_d + i*plan_FEM[3].Localnumno, plan_FEM[3].MatPropQ6_h_Pt + i*numno, plan_FEM[3].Localnumno*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(plan_FEM[3].MatPropQ7_d + i*plan_FEM[3].Localnumno, plan_FEM[3].MatPropQ7_h_Pt + i*numno, plan_FEM[3].Localnumno*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(plan_FEM[3].MatPropQ8_d + i*plan_FEM[3].Localnumno, plan_FEM[3].MatPropQ8_h_Pt + i*numno, plan_FEM[3].Localnumno*sizeof(double), cudaMemcpyHostToDevice);

	}

	for(i=0; i<numdof*numnoel; i++) {

		// Copying Coordinate data from host to device:
		cudaMemcpy(plan_FEM[3].CoordQ1_d + i*plan_FEM[3].Localnumno, plan_FEM[3].CoordQ1_h_Pt + i*numno, plan_FEM[3].Localnumno*sizeof(double), cudaMemcpyHostToDevice);  // To copy from host memory to device memory;
		cudaMemcpy(plan_FEM[3].CoordQ2_d + i*plan_FEM[3].Localnumno, plan_FEM[3].CoordQ2_h_Pt + i*numno, plan_FEM[3].Localnumno*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(plan_FEM[3].CoordQ3_d + i*plan_FEM[3].Localnumno, plan_FEM[3].CoordQ3_h_Pt + i*numno, plan_FEM[3].Localnumno*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(plan_FEM[3].CoordQ4_d + i*plan_FEM[3].Localnumno, plan_FEM[3].CoordQ4_h_Pt + i*numno, plan_FEM[3].Localnumno*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(plan_FEM[3].CoordQ5_d + i*plan_FEM[3].Localnumno, plan_FEM[3].CoordQ5_h_Pt + i*numno, plan_FEM[3].Localnumno*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(plan_FEM[3].CoordQ6_d + i*plan_FEM[3].Localnumno, plan_FEM[3].CoordQ6_h_Pt + i*numno, plan_FEM[3].Localnumno*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(plan_FEM[3].CoordQ7_d + i*plan_FEM[3].Localnumno, plan_FEM[3].CoordQ7_h_Pt + i*numno, plan_FEM[3].Localnumno*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(plan_FEM[3].CoordQ8_d + i*plan_FEM[3].Localnumno, plan_FEM[3].CoordQ8_h_Pt + i*numno, plan_FEM[3].Localnumno*sizeof(double), cudaMemcpyHostToDevice);

	}

	// Copying ii position data from host to device:
	cudaMemcpy(plan_FEM[3].iiPosition_d, plan_FEM[3].iiPosition_h_Pt, numdof*plan_FEM[3].Localnumno*sizeof(int), cudaMemcpyHostToDevice);    // To copy from host memory to device memory;
	*/
	//=============== x =============== x =============== x =============== x =============== x =============== x ===============

	//Get the number of terms of the matrix for each GPU:

	//Assign data ranges to GPUs
    gpuBaseTerm = gpuBaseRow = gpuBaseElem = 0;

	for(i=0; i<deviceCount; i++) {

		plan_FEM[i].Vector_I_h_Pt = Vector_I_h + gpuBaseTerm;
		plan_FEM[i].Vector_B_h_Pt = Vector_B_h + gpuBaseRow;
		plan_FEM[i].Vector_X_h_Pt = Vector_X_h + gpuBaseRow;

		plan_FEM[i].CoordElem_h_Pt      = CoordElem_h      + gpuBaseElem;
		plan_FEM[i].Connect_h_Pt        = Connect_h        + gpuBaseElem;
		plan_FEM[i].Material_Data_h_Pt  = Material_Data_h  + gpuBaseElem;
		plan_FEM[i].DeltaStrain_h_Pt    = DeltaStrain_h    + gpuBaseElem;
		plan_FEM[i].DeltaStress_h_Pt    = DeltaStress_h    + gpuBaseElem;
		plan_FEM[i].DeltaVolStrain_h_Pt = DeltaVolStrain_h + gpuBaseElem;

		plan_FEM[i].DeviceId = i;
		plan_FEM[i].DeviceNumber      = deviceCount;
		plan_FEM[i].GlobalNumRow      = numno;             //  Total number of rows;
		plan_FEM[i].NumMultiProcessor = GPUData[i*5+0];
		plan_FEM[i].BlockSize         = GPUData[i*5+4];
		plan_FEM[i].RowSize           = RowSize;

		gpuBaseTerm += numdof*plan_FEM[i].Localnumno*RowSize;
		gpuBaseRow  += numdof*plan_FEM[i].Localnumno;
		gpuBaseElem += plan_FEM[i].Localnumel;

	}

	//=============== x =============== x =============== x =============== x =============== x =============== x ===============

	// Gradient Conjugate Method Memory Allocation:

	for(j=0; j<deviceCount; j++) {

		// Allocating memory space on device "j"

		cudaSetDevice(j);  // Sets device "j" as the current device

		plan_FEM[j].GPUmultiProcessorCount = GPUData[j*5+0];
		plan_FEM[j].GPUmaxThreadsPerBlock  = GPUData[j*5+4];

		plan_FEM[j].BlockSizeX = 32; plan_FEM[j].BlockSizeY = 32;  // 32 x 32 = 1024

		BlockDimX = int(sqrt(double(plan_FEM[j].Localnumno))/plan_FEM[j].BlockSizeX)+1;
		BlockDimY = int(sqrt(double(plan_FEM[j].Localnumno))/plan_FEM[j].BlockSizeY)+1;
		plan_FEM[j].GPUNumThread = BlockDimX*BlockDimY*plan_FEM[j].GPUmaxThreadsPerBlock;

		// Number of blocks needs to sum the terms of vector:
		if(plan_FEM[j].GPUmultiProcessorCount <= 4)       NumMultProc =  4;
		else if(plan_FEM[j].GPUmultiProcessorCount <=  8) NumMultProc =  8;
		else if(plan_FEM[j].GPUmultiProcessorCount <= 16) NumMultProc = 16;
		else if(plan_FEM[j].GPUmultiProcessorCount <= 32) NumMultProc = 32;

		// ------------------------------------------------------------------------------------

		// Doing the minimum NumRowLocal = GPUBlockSizeX*GPUBlockSizeY

		NumRowLocal  = numdof*plan_FEM[j].Localnumno;
		NumTermLocal = NumRowLocal*RowSize;
		NumRowGlobal = numdof*numno;

		// ------------------------------------------------------------------------------------

		cudaMalloc((void **) &plan_FEM[j].Vector_B_d, NumRowLocal*sizeof(double));
		cudaMalloc((void **) &plan_FEM[j].Vector_R_d, NumRowLocal*sizeof(double));
		cudaMalloc((void **) &plan_FEM[j].Matrix_M_d, NumRowLocal*sizeof(double));
		cudaMalloc((void **) &plan_FEM[j].delta_new_d, NumMultProc*sizeof(double));
		cudaMalloc((void **) &plan_FEM[j].delta_new_V_d, NumRowLocal*sizeof(double));
		plan_FEM[j].delta_new_V_h = (double *)malloc(NumRowLocal*sizeof(double));
		cudaMalloc((void **) &plan_FEM[j].Matrix_A_d, numdof*plan_FEM[j].GPUNumThread*RowSize*sizeof(double));
		plan_FEM[j].Matrix_A_h = (double *)malloc(numdof*plan_FEM[j].GPUNumThread*RowSize*sizeof(double));
		cudaMalloc((void **) &plan_FEM[j].Vector_I_d, NumTermLocal*(sizeof(int)));
		cudaMalloc((void **) &plan_FEM[j].Vector_Q_d, NumRowLocal*sizeof(double));
		cudaMalloc((void **) &plan_FEM[j].delta_aux_d, NumMultProc*sizeof(double));
		cudaMalloc((void **) &plan_FEM[j].delta_aux_V_d, NumRowLocal*sizeof(double));
		cudaMalloc((void **) &plan_FEM[j].Vector_S_d, NumRowLocal*sizeof(double));
		cudaMalloc((void **) &plan_FEM[j].delta_old_V_d, NumRowLocal*sizeof(double));

		cudaMalloc((void **) &plan_FEM[j].Vector_X_d, NumRowLocal*sizeof(double));
		cudaMalloc((void **) &plan_FEM[j].Connect_d, plan_FEM[j].Localnumel*numnoel*sizeof(int));
		cudaMalloc((void **) &plan_FEM[j].CoordElem_d, plan_FEM[j].Localnumel*numdof*numnoel*sizeof(double));
		cudaMalloc((void **) &plan_FEM[j].DeltaStrain_d, plan_FEM[j].Localnumel*numstr*sizeof(double));
		cudaMalloc((void **) &plan_FEM[j].DeltaVolStrain_d, plan_FEM[j].Localnumel*sizeof(double));
		cudaMalloc((void **) &plan_FEM[j].Material_Data_d, plan_FEM[j].Localnumel*numprop*sizeof(double));
		cudaMalloc((void **) &plan_FEM[j].DeltaStress_d, plan_FEM[j].Localnumel*numstr*sizeof(double));
		cudaMalloc((void **) &plan_FEM[j].Vector_X_Global_d, NumRowGlobal*sizeof(double));

		cudaMalloc((void **) &plan_FEM[j].delta_GPU_d, plan_FEM[j].DeviceNumber*sizeof(double));
		cudaMalloc((void **) &plan_FEM[j].Result_d, sizeof(double));
		cudaMalloc((void **) &plan_FEM[j].Vector_D_d, numdof*plan_FEM[j].Localnumno*sizeof(double));
		cudaMalloc((void **) &plan_FEM[j].Vector_D_Global_d, numdof*numno*sizeof(double));

		// Copying from Host to Device for device 0:
		cudaMemcpy(plan_FEM[j].Vector_I_d, plan_FEM[j].Vector_I_h_Pt, numdof*plan_FEM[j].Localnumno*RowSize*sizeof(int), cudaMemcpyHostToDevice);

		// Copying element node coordinate data from host to device:
		for(i=0; i<numdof*numnoel; i++)
			cudaMemcpy(plan_FEM[j].CoordElem_d + i*plan_FEM[j].Localnumel, plan_FEM[j].CoordElem_h_Pt + i*numel, plan_FEM[j].Localnumel*sizeof(double), cudaMemcpyHostToDevice);

		// Copying element conectivity from host to device:
		for(i=0; i<numnoel; i++)
			cudaMemcpy(plan_FEM[j].Connect_d + i*plan_FEM[j].Localnumel, plan_FEM[j].Connect_h_Pt + i*numel, plan_FEM[j].Localnumel*sizeof(int), cudaMemcpyHostToDevice);

		// Copying material data from host to device:
		for(i=0; i<numprop; i++)
			cudaMemcpy(plan_FEM[j].Material_Data_d + i*plan_FEM[j].Localnumel, plan_FEM[j].Material_Data_h_Pt + i*numel, plan_FEM[j].Localnumel*sizeof(double), cudaMemcpyHostToDevice);

		for(i=0; i<numdof*plan_FEM[j].GPUNumThread*RowSize; i++) {
			plan_FEM[j].Matrix_A_h[i] = 0;
		}

	}

	/*// Allocating memory space on device "0"

	cudaSetDevice(0);  // Sets device "0" as the current device

	plan_FEM[0].GPUmultiProcessorCount = GPUData[0*5+0];
	plan_FEM[0].GPUmaxThreadsPerBlock  = GPUData[0*5+4];

	plan_FEM[0].BlockSizeX = 32; plan_FEM[0].BlockSizeY = 32;  // 32 x 32 = 1024

	BlockDimX = int(sqrt(double(plan_FEM[0].Localnumno))/plan_FEM[0].BlockSizeX)+1;
	BlockDimY = int(sqrt(double(plan_FEM[0].Localnumno))/plan_FEM[0].BlockSizeY)+1;
	plan_FEM[0].GPUNumThread = BlockDimX*BlockDimY*plan_FEM[0].GPUmaxThreadsPerBlock;

	// Number of blocks needs to sum the terms of vector:
	if(plan_FEM[0].GPUmultiProcessorCount <= 4)       NumMultProc =  4;
	else if(plan_FEM[0].GPUmultiProcessorCount <=  8) NumMultProc =  8;
	else if(plan_FEM[0].GPUmultiProcessorCount <= 16) NumMultProc = 16;
	else if(plan_FEM[0].GPUmultiProcessorCount <= 32) NumMultProc = 32;

	// ------------------------------------------------------------------------------------

	// Doing the minimum NumRowLocal = GPUBlockSizeX*GPUBlockSizeY

	NumRowLocal  = numdof*plan_FEM[0].Localnumno;
	NumTermLocal = NumRowLocal*RowSize;
	NumRowGlobal = numdof*numno;

	// ------------------------------------------------------------------------------------

	cudaMalloc((void **) &plan_FEM[0].Vector_B_d, NumRowLocal*sizeof(double));
	cudaMalloc((void **) &plan_FEM[0].Vector_R_d, NumRowLocal*sizeof(double));
	cudaMalloc((void **) &plan_FEM[0].Matrix_M_d, NumRowLocal*sizeof(double));
	cudaMalloc((void **) &plan_FEM[0].delta_new_d, NumMultProc*sizeof(double));
	cudaMalloc((void **) &plan_FEM[0].delta_new_V_d, NumRowLocal*sizeof(double));
	plan_FEM[0].delta_new_V_h = (double *)malloc(NumRowLocal*sizeof(double));
	cudaMalloc((void **) &plan_FEM[0].Matrix_A_d, numdof*plan_FEM[0].GPUNumThread*RowSize*sizeof(double));
	plan_FEM[0].Matrix_A_h = (double *)malloc(numdof*plan_FEM[0].GPUNumThread*RowSize*sizeof(double));
	cudaMalloc((void **) &plan_FEM[0].Vector_I_d, NumTermLocal*(sizeof(int)));
	cudaMalloc((void **) &plan_FEM[0].Vector_Q_d, NumRowLocal*sizeof(double));
	cudaMalloc((void **) &plan_FEM[0].delta_aux_d, NumMultProc*sizeof(double));
	cudaMalloc((void **) &plan_FEM[0].delta_aux_V_d, NumRowLocal*sizeof(double));
	cudaMalloc((void **) &plan_FEM[0].Vector_S_d, NumRowLocal*sizeof(double));
	cudaMalloc((void **) &plan_FEM[0].delta_old_V_d, NumRowLocal*sizeof(double));

	cudaMalloc((void **) &plan_FEM[0].Vector_X_d, NumRowLocal*sizeof(double));
	cudaMalloc((void **) &plan_FEM[0].Connect_d, plan_FEM[0].Localnumel*numnoel*sizeof(int));
	cudaMalloc((void **) &plan_FEM[0].CoordElem_d, plan_FEM[0].Localnumel*numdof*numnoel*sizeof(double));
	cudaMalloc((void **) &plan_FEM[0].Strain_d, plan_FEM[0].Localnumel*numstr*8*sizeof(double));
	cudaMalloc((void **) &plan_FEM[0].Material_Data_d, plan_FEM[0].Localnumel*numprop*sizeof(double));
	cudaMalloc((void **) &plan_FEM[0].Stress_d, plan_FEM[0].Localnumel*numstr*8*sizeof(double));
	cudaMalloc((void **) &plan_FEM[0].Vector_X_Global_d, NumRowGlobal*sizeof(double));

	cudaMalloc((void **) &plan_FEM[0].delta_GPU_d, plan_FEM[0].DeviceNumber*sizeof(double));
	cudaMalloc((void **) &plan_FEM[0].Result_d, sizeof(double));
	cudaMalloc((void **) &plan_FEM[0].Vector_D_d, numdof*plan_FEM[0].Localnumno*sizeof(double));
	cudaMalloc((void **) &plan_FEM[0].Vector_D_Global_d, numdof*numno*sizeof(double));

	// Copying from Host to Device for device 0:
	cudaMemcpy(plan_FEM[0].Vector_I_d, plan_FEM[0].Vector_I_h_Pt, numdof*plan_FEM[0].Localnumno*RowSize*sizeof(int), cudaMemcpyHostToDevice);

	// Copying element node coordinate data from host to device:
	for(i=0; i<numdof*numnoel; i++)
		cudaMemcpy(plan_FEM[0].CoordElem_d + i*plan_FEM[0].Localnumel, plan_FEM[0].CoordElem_h_Pt + i*numel, plan_FEM[0].Localnumel*sizeof(double), cudaMemcpyHostToDevice);

	// Copying element conectivity from host to device:
	for(i=0; i<numnoel; i++)
		cudaMemcpy(plan_FEM[0].Connect_d + i*plan_FEM[0].Localnumel, plan_FEM[0].Connect_h_Pt + i*numel, plan_FEM[0].Localnumel*sizeof(int), cudaMemcpyHostToDevice);

	// Copying material data from host to device:
	for(i=0; i<numprop; i++)
		cudaMemcpy(plan_FEM[0].Material_Data_d + i*plan_FEM[0].Localnumel, plan_FEM[0].Material_Data_h_Pt + i*numel, plan_FEM[0].Localnumel*sizeof(int), cudaMemcpyHostToDevice);

	for(i=0; i<numdof*plan_FEM[0].GPUNumThread*RowSize; i++) {
		plan_FEM[0].Matrix_A_h[i] = 0;
	}

	//--------------------------------------------------------------------------------------------------------------------------------------

	// Allocating memory space on device "1"

	cudaSetDevice(1);  // Sets device "1" as the current device

	plan_FEM[1].GPUmultiProcessorCount = GPUData[1*5+0];
	plan_FEM[1].GPUmaxThreadsPerBlock  = GPUData[1*5+4];

	plan_FEM[1].BlockSizeX = 32; plan_FEM[1].BlockSizeY = 32;  // 32 x 32 = 1024

	BlockDimX = int(sqrt(double(plan_FEM[1].Localnumno))/plan_FEM[1].BlockSizeX)+1;
	BlockDimY = int(sqrt(double(plan_FEM[1].Localnumno))/plan_FEM[1].BlockSizeY)+1;
	plan_FEM[1].GPUNumThread = BlockDimX*BlockDimY*plan_FEM[1].GPUmaxThreadsPerBlock;

	// Number of blocks needs to sum the terms of vector:
	if(plan_FEM[1].GPUmultiProcessorCount <= 4)       NumMultProc =  4;
	else if(plan_FEM[1].GPUmultiProcessorCount <=  8) NumMultProc =  8;
	else if(plan_FEM[1].GPUmultiProcessorCount <= 16) NumMultProc = 16;
	else if(plan_FEM[1].GPUmultiProcessorCount <= 32) NumMultProc = 32;

	// ------------------------------------------------------------------------------------

	// Doing the minimum NumRowLocal = GPUBlockSizeX*GPUBlockSizeY

	NumRowLocal  = numdof*plan_FEM[1].Localnumno;
	NumTermLocal = NumRowLocal*RowSize;
	NumRowGlobal = numdof*numno;

	// ------------------------------------------------------------------------------------

	cudaMalloc((void **) &plan_FEM[1].Vector_B_d, NumRowLocal*sizeof(double));
	cudaMalloc((void **) &plan_FEM[1].Vector_R_d, NumRowLocal*sizeof(double));
	cudaMalloc((void **) &plan_FEM[1].Matrix_M_d, NumRowLocal*sizeof(double));
	cudaMalloc((void **) &plan_FEM[1].delta_new_d, NumMultProc*sizeof(double));
	cudaMalloc((void **) &plan_FEM[1].delta_new_V_d, NumRowLocal*sizeof(double));
	plan_FEM[1].delta_new_V_h = (double *)malloc(NumRowLocal*sizeof(double));
	cudaMalloc((void **) &plan_FEM[1].Matrix_A_d, numdof*plan_FEM[1].GPUNumThread*RowSize*sizeof(double));
	plan_FEM[1].Matrix_A_h = (double *)malloc(numdof*plan_FEM[1].GPUNumThread*RowSize*sizeof(double));
	cudaMalloc((void **) &plan_FEM[1].Vector_I_d, NumTermLocal*(sizeof(int)));
	cudaMalloc((void **) &plan_FEM[1].Vector_Q_d, NumRowLocal*sizeof(double));
	cudaMalloc((void **) &plan_FEM[1].delta_aux_d, NumMultProc*sizeof(double));
	cudaMalloc((void **) &plan_FEM[1].delta_aux_V_d, NumRowLocal*sizeof(double));
	cudaMalloc((void **) &plan_FEM[1].Vector_S_d, NumRowLocal*sizeof(double));
	cudaMalloc((void **) &plan_FEM[1].delta_old_V_d, NumRowLocal*sizeof(double));

	cudaMalloc((void **) &plan_FEM[1].Vector_X_d, NumRowLocal*sizeof(double));
	cudaMalloc((void **) &plan_FEM[1].Connect_d, plan_FEM[1].Localnumel*numnoel*sizeof(int));
	cudaMalloc((void **) &plan_FEM[1].CoordElem_d, plan_FEM[1].Localnumel*numdof*numnoel*sizeof(double));
	cudaMalloc((void **) &plan_FEM[1].Strain_d, plan_FEM[1].Localnumel*numstr*8*sizeof(double));
	cudaMalloc((void **) &plan_FEM[1].Material_Data_d, plan_FEM[1].Localnumel*numprop*sizeof(double));
	cudaMalloc((void **) &plan_FEM[1].Stress_d, plan_FEM[1].Localnumel*numstr*8*sizeof(double));
	cudaMalloc((void **) &plan_FEM[1].Vector_X_Global_d, NumRowGlobal*sizeof(double));

	cudaMalloc((void **) &plan_FEM[1].delta_GPU_d, plan_FEM[1].DeviceNumber*sizeof(double));
	cudaMalloc((void **) &plan_FEM[1].Result_d, sizeof(double));
	cudaMalloc((void **) &plan_FEM[1].Vector_D_d, numdof*plan_FEM[1].Localnumno*sizeof(double));
	cudaMalloc((void **) &plan_FEM[1].Vector_D_Global_d, numdof*numno*sizeof(double));

	// Copying from Host to Device for device 1:
	cudaMemcpy(plan_FEM[1].Vector_I_d, plan_FEM[1].Vector_I_h_Pt, numdof*plan_FEM[1].Localnumno*RowSize*sizeof(int), cudaMemcpyHostToDevice);

	// Copying element node coordinate data from host to device:
	for(i=0; i<numdof*numnoel; i++)
		cudaMemcpy(plan_FEM[1].CoordElem_d + i*plan_FEM[1].Localnumel, plan_FEM[1].CoordElem_h_Pt + i*numel, plan_FEM[1].Localnumel*sizeof(double), cudaMemcpyHostToDevice);  // To copy from host memory to device memory;

	// Copying element conectivity from host to device:
	for(i=0; i<numnoel; i++)
		cudaMemcpy(plan_FEM[1].Connect_d + i*plan_FEM[1].Localnumel, plan_FEM[1].Connect_h_Pt + i*numel, plan_FEM[1].Localnumel*sizeof(int), cudaMemcpyHostToDevice);

	// Copying material data from host to device:
	for(i=0; i<numprop; i++)
		cudaMemcpy(plan_FEM[1].Material_Data_d + i*plan_FEM[1].Localnumel, plan_FEM[1].Material_Data_h_Pt + i*numel, plan_FEM[1].Localnumel*sizeof(int), cudaMemcpyHostToDevice);

	for(i=0; i<numdof*plan_FEM[1].GPUNumThread*RowSize; i++) {
		plan_FEM[1].Matrix_A_h[i] = 0;
	}

	//--------------------------------------------------------------------------------------------------------------------------------------

	// Allocating memory space on device "2"

	cudaSetDevice(2);  // Sets device "2" as the current device

	plan_FEM[2].GPUmultiProcessorCount = GPUData[2*5+0];
	plan_FEM[2].GPUmaxThreadsPerBlock  = GPUData[2*5+4];

	plan_FEM[2].BlockSizeX = 32; plan_FEM[2].BlockSizeY = 32;  // 32 x 32 = 1024

	BlockDimX = int(sqrt(double(plan_FEM[2].Localnumno))/plan_FEM[2].BlockSizeX)+1;
	BlockDimY = int(sqrt(double(plan_FEM[2].Localnumno))/plan_FEM[2].BlockSizeY)+1;
	plan_FEM[2].GPUNumThread = BlockDimX*BlockDimY*plan_FEM[2].GPUmaxThreadsPerBlock;

	// Number of blocks needs to sum the terms of vector:
	if(plan_FEM[2].GPUmultiProcessorCount <= 4)       NumMultProc =  4;
	else if(plan_FEM[2].GPUmultiProcessorCount <=  8) NumMultProc =  8;
	else if(plan_FEM[2].GPUmultiProcessorCount <= 16) NumMultProc = 16;
	else if(plan_FEM[2].GPUmultiProcessorCount <= 32) NumMultProc = 32;

	// ------------------------------------------------------------------------------------

	// Doing the minimum NumRowLocal = GPUBlockSizeX*GPUBlockSizeY

	NumRowLocal  = numdof*plan_FEM[2].Localnumno;
	NumTermLocal = NumRowLocal*RowSize;
	NumRowGlobal = numdof*numno;

	// ------------------------------------------------------------------------------------

	cudaMalloc((void **) &plan_FEM[2].Vector_B_d, NumRowLocal*sizeof(double));
	cudaMalloc((void **) &plan_FEM[2].Vector_R_d, NumRowLocal*sizeof(double));
	cudaMalloc((void **) &plan_FEM[2].Matrix_M_d, NumRowLocal*sizeof(double));
	cudaMalloc((void **) &plan_FEM[2].delta_new_d, NumMultProc*sizeof(double));
	cudaMalloc((void **) &plan_FEM[2].delta_new_V_d, NumRowLocal*sizeof(double));
	plan_FEM[2].delta_new_V_h = (double *)malloc(NumRowLocal*sizeof(double));
	cudaMalloc((void **) &plan_FEM[2].Matrix_A_d, numdof*plan_FEM[2].GPUNumThread*RowSize*sizeof(double));
	plan_FEM[2].Matrix_A_h = (double *)malloc(numdof*plan_FEM[2].GPUNumThread*RowSize*sizeof(double));
	cudaMalloc((void **) &plan_FEM[2].Vector_I_d, NumTermLocal*(sizeof(int)));
	cudaMalloc((void **) &plan_FEM[2].Vector_Q_d, NumRowLocal*sizeof(double));
	cudaMalloc((void **) &plan_FEM[2].delta_aux_d, NumMultProc*sizeof(double));
	cudaMalloc((void **) &plan_FEM[2].delta_aux_V_d, NumRowLocal*sizeof(double));
	cudaMalloc((void **) &plan_FEM[2].Vector_S_d, NumRowLocal*sizeof(double));
	cudaMalloc((void **) &plan_FEM[2].delta_old_V_d, NumRowLocal*sizeof(double));

	cudaMalloc((void **) &plan_FEM[2].Vector_X_d, NumRowLocal*sizeof(double));
	cudaMalloc((void **) &plan_FEM[2].Connect_d, plan_FEM[2].Localnumel*numnoel*sizeof(int));
	cudaMalloc((void **) &plan_FEM[2].CoordElem_d, plan_FEM[2].Localnumel*numdof*numnoel*sizeof(double));
	cudaMalloc((void **) &plan_FEM[2].Strain_d, plan_FEM[2].Localnumel*numstr*8*sizeof(double));
	cudaMalloc((void **) &plan_FEM[2].Material_Data_d, plan_FEM[2].Localnumel*numprop*sizeof(double));
	cudaMalloc((void **) &plan_FEM[2].Stress_d, plan_FEM[2].Localnumel*numstr*8*sizeof(double));
	cudaMalloc((void **) &plan_FEM[2].Vector_X_Global_d, NumRowGlobal*sizeof(double));

	cudaMalloc((void **) &plan_FEM[2].delta_GPU_d, plan_FEM[2].DeviceNumber*sizeof(double));
	cudaMalloc((void **) &plan_FEM[2].Result_d, sizeof(double));
	cudaMalloc((void **) &plan_FEM[2].Vector_D_d, numdof*plan_FEM[2].Localnumno*sizeof(double));
	cudaMalloc((void **) &plan_FEM[2].Vector_D_Global_d, numdof*numno*sizeof(double));

	// Copying from Host to Device for device 2:
	cudaMemcpy(plan_FEM[2].Vector_I_d, plan_FEM[2].Vector_I_h_Pt, numdof*plan_FEM[2].Localnumno*RowSize*sizeof(int), cudaMemcpyHostToDevice);

	// Copying element node coordinate data from host to device:
	for(i=0; i<numdof*numnoel; i++)
		cudaMemcpy(plan_FEM[2].CoordElem_d + i*plan_FEM[2].Localnumel, plan_FEM[2].CoordElem_h_Pt + i*numel, plan_FEM[2].Localnumel*sizeof(double), cudaMemcpyHostToDevice);  // To copy from host memory to device memory;

	// Copying element conectivity from host to device:
	for(i=0; i<numnoel; i++)
		cudaMemcpy(plan_FEM[2].Connect_d + i*plan_FEM[2].Localnumel, plan_FEM[2].Connect_h_Pt + i*numel, plan_FEM[2].Localnumel*sizeof(int), cudaMemcpyHostToDevice);

	// Copying material data from host to device:
	for(i=0; i<numprop; i++)
		cudaMemcpy(plan_FEM[2].Material_Data_d + i*plan_FEM[2].Localnumel, plan_FEM[2].Material_Data_h_Pt + i*numel, plan_FEM[2].Localnumel*sizeof(int), cudaMemcpyHostToDevice);

	for(i=0; i<numdof*plan_FEM[2].GPUNumThread*RowSize; i++) {
		plan_FEM[2].Matrix_A_h[i] = 0;
	}

	//--------------------------------------------------------------------------------------------------------------------------------------

	// Allocating memory space on device "3"

	cudaSetDevice(3);  // Sets device "3" as the current device

	plan_FEM[3].GPUmultiProcessorCount = GPUData[3*5+0];
	plan_FEM[3].GPUmaxThreadsPerBlock  = GPUData[3*5+4];

	plan_FEM[3].BlockSizeX = 32; plan_FEM[3].BlockSizeY = 32;  // 32 x 32 = 1024

	BlockDimX = int(sqrt(double(plan_FEM[3].Localnumno))/plan_FEM[3].BlockSizeX)+1;
	BlockDimY = int(sqrt(double(plan_FEM[3].Localnumno))/plan_FEM[3].BlockSizeY)+1;
	plan_FEM[3].GPUNumThread = BlockDimX*BlockDimY*plan_FEM[3].GPUmaxThreadsPerBlock;

	// Number of blocks needs to sum the terms of vector:
	if(plan_FEM[3].GPUmultiProcessorCount <= 4)       NumMultProc =  4;
	else if(plan_FEM[3].GPUmultiProcessorCount <=  8) NumMultProc =  8;
	else if(plan_FEM[3].GPUmultiProcessorCount <= 16) NumMultProc = 16;
	else if(plan_FEM[3].GPUmultiProcessorCount <= 32) NumMultProc = 32;

	// ------------------------------------------------------------------------------------

	// Doing the minimum NumRowLocal = GPUBlockSizeX*GPUBlockSizeY

	NumRowLocal  = numdof*plan_FEM[3].Localnumno;
	NumTermLocal = NumRowLocal*RowSize;
	NumRowGlobal = numdof*numno;

	// ------------------------------------------------------------------------------------

	cudaMalloc((void **) &plan_FEM[3].Vector_B_d, NumRowLocal*sizeof(double));
	cudaMalloc((void **) &plan_FEM[3].Vector_R_d, NumRowLocal*sizeof(double));
	cudaMalloc((void **) &plan_FEM[3].Matrix_M_d, NumRowLocal*sizeof(double));
	cudaMalloc((void **) &plan_FEM[3].delta_new_d, NumMultProc*sizeof(double));
	cudaMalloc((void **) &plan_FEM[3].delta_new_V_d, NumRowLocal*sizeof(double));
	plan_FEM[3].delta_new_V_h = (double *)malloc(NumRowLocal*sizeof(double));
	cudaMalloc((void **) &plan_FEM[3].Matrix_A_d, numdof*plan_FEM[3].GPUNumThread*RowSize*sizeof(double));
	plan_FEM[3].Matrix_A_h = (double *)malloc(numdof*plan_FEM[3].GPUNumThread*RowSize*sizeof(double));
	cudaMalloc((void **) &plan_FEM[3].Vector_I_d, NumTermLocal*(sizeof(int)));
	cudaMalloc((void **) &plan_FEM[3].Vector_Q_d, NumRowLocal*sizeof(double));
	cudaMalloc((void **) &plan_FEM[3].delta_aux_d, NumMultProc*sizeof(double));
	cudaMalloc((void **) &plan_FEM[3].delta_aux_V_d, NumRowLocal*sizeof(double));
	cudaMalloc((void **) &plan_FEM[3].Vector_S_d, NumRowLocal*sizeof(double));
	cudaMalloc((void **) &plan_FEM[3].delta_old_V_d, NumRowLocal*sizeof(double));

	cudaMalloc((void **) &plan_FEM[3].Vector_X_d, NumRowLocal*sizeof(double));
	cudaMalloc((void **) &plan_FEM[3].Connect_d, plan_FEM[3].Localnumel*numnoel*sizeof(int));
	cudaMalloc((void **) &plan_FEM[3].CoordElem_d, plan_FEM[3].Localnumel*numdof*numnoel*sizeof(double));
	cudaMalloc((void **) &plan_FEM[3].Strain_d, plan_FEM[3].Localnumel*numstr*8*sizeof(double));
	cudaMalloc((void **) &plan_FEM[3].Material_Data_d, plan_FEM[3].Localnumel*numprop*sizeof(double));
	cudaMalloc((void **) &plan_FEM[3].Stress_d, plan_FEM[3].Localnumel*numstr*8*sizeof(double));
	cudaMalloc((void **) &plan_FEM[3].Vector_X_Global_d, NumRowGlobal*sizeof(double));

	cudaMalloc((void **) &plan_FEM[3].delta_GPU_d, plan_FEM[3].DeviceNumber*sizeof(double));
	cudaMalloc((void **) &plan_FEM[3].Result_d, sizeof(double));
	cudaMalloc((void **) &plan_FEM[3].Vector_D_d, numdof*plan_FEM[3].Localnumno*sizeof(double));
	cudaMalloc((void **) &plan_FEM[3].Vector_D_Global_d, numdof*numno*sizeof(double));

	// Copying from Host to Device for device 3:
	cudaMemcpy(plan_FEM[3].Vector_I_d, plan_FEM[3].Vector_I_h_Pt, numdof*plan_FEM[3].Localnumno*RowSize*sizeof(int), cudaMemcpyHostToDevice);

	// Copying element node coordinate data from host to device:
	for(i=0; i<numdof*numnoel; i++)
		cudaMemcpy(plan_FEM[3].CoordElem_d + i*plan_FEM[3].Localnumel, plan_FEM[3].CoordElem_h_Pt + i*numel, plan_FEM[3].Localnumel*sizeof(double), cudaMemcpyHostToDevice);  // To copy from host memory to device memory;

	// Copying element conectivity from host to device:
	for(i=0; i<numnoel; i++)
		cudaMemcpy(plan_FEM[3].Connect_d + i*plan_FEM[3].Localnumel, plan_FEM[3].Connect_h_Pt + i*numel, plan_FEM[3].Localnumel*sizeof(int), cudaMemcpyHostToDevice);

	// Copying material data from host to device:
	for(i=0; i<numprop; i++)
		cudaMemcpy(plan_FEM[3].Material_Data_d + i*plan_FEM[3].Localnumel, plan_FEM[3].Material_Data_h_Pt + i*numel, plan_FEM[3].Localnumel*sizeof(int), cudaMemcpyHostToDevice);

	for(i=0; i<numdof*plan_FEM[3].GPUNumThread*RowSize; i++) {
		plan_FEM[3].Matrix_A_h[i] = 0;

	}*/

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

	//--------------------------------------------------------------------------------------------------------------------------------------

	return 1;

}

//========================================================================================================

int cGPU::AssemblyStiffnessMatrixOnGPU()
{

	//=============== x =============== x =============== x =============== x =============== x =============== x ===============

	// Assembly the stiffness matrix for Hexahedron Element on Multi GPUs:

	// To define one CPU thread for each GPU:

	omp_set_num_threads(deviceCount);

#pragma omp parallel
	{

		int DeviceId;
		double time;

		DeviceId = omp_get_thread_num();

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

		// Barrier Synchronization
#pragma omp barrier

		// To start time count to assembly stiffness matrix:
		time = clock();

		cudaMemcpy(cGPU::plan_FEM[DeviceId].Matrix_A_d, cGPU::plan_FEM[DeviceId].Matrix_A_h, cGPU::numdof*cGPU::plan_FEM[DeviceId].GPUNumThread*cGPU::RowSize*sizeof(double), cudaMemcpyHostToDevice);

		// Compute Capability 2.x and 3.0 => Maximum number of threads per block = 1024 (maxThreadsPerBlock)

		dim3 numBlocksSM(int(sqrt(double(cGPU::plan_FEM[DeviceId].Localnumno))/cGPU::plan_FEM[DeviceId].BlockSizeX)+1, int(sqrt(double(cGPU::plan_FEM[DeviceId].Localnumno))/cGPU::plan_FEM[DeviceId].BlockSizeY)+1);
		dim3 threadsPerBlockSM(cGPU::plan_FEM[DeviceId].BlockSizeX, cGPU::plan_FEM[DeviceId].BlockSizeY);

		// Assembly on GPU the Stiffness Matrix

		Launch_AssemblyStiffness3DQ1(numBlocksSM, threadsPerBlockSM, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::RowSize, cGPU::plan_FEM[DeviceId].CoordQ1_d, cGPU::plan_FEM[DeviceId].MatPropQ1_d, cGPU::plan_FEM[DeviceId].BC_d,
			cGPU::plan_FEM[DeviceId].Matrix_A_d);

		Launch_AssemblyStiffness3DQ2(numBlocksSM, threadsPerBlockSM, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::RowSize, cGPU::plan_FEM[DeviceId].CoordQ2_d, cGPU::plan_FEM[DeviceId].MatPropQ2_d, cGPU::plan_FEM[DeviceId].BC_d,
			cGPU::plan_FEM[DeviceId].Matrix_A_d);

		Launch_AssemblyStiffness3DQ3(numBlocksSM, threadsPerBlockSM, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::RowSize, cGPU::plan_FEM[DeviceId].CoordQ3_d, cGPU::plan_FEM[DeviceId].MatPropQ3_d, cGPU::plan_FEM[DeviceId].BC_d,
			cGPU::plan_FEM[DeviceId].Matrix_A_d);

		Launch_AssemblyStiffness3DQ4(numBlocksSM, threadsPerBlockSM, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::RowSize, cGPU::plan_FEM[DeviceId].CoordQ4_d, cGPU::plan_FEM[DeviceId].MatPropQ4_d, cGPU::plan_FEM[DeviceId].BC_d,
			cGPU::plan_FEM[DeviceId].Matrix_A_d);

		Launch_AssemblyStiffness3DQ5(numBlocksSM, threadsPerBlockSM, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::RowSize, cGPU::plan_FEM[DeviceId].CoordQ5_d, cGPU::plan_FEM[DeviceId].MatPropQ5_d, cGPU::plan_FEM[DeviceId].BC_d,
			cGPU::plan_FEM[DeviceId].Matrix_A_d);

		Launch_AssemblyStiffness3DQ6(numBlocksSM, threadsPerBlockSM, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::RowSize, cGPU::plan_FEM[DeviceId].CoordQ6_d, cGPU::plan_FEM[DeviceId].MatPropQ6_d, cGPU::plan_FEM[DeviceId].BC_d,
			cGPU::plan_FEM[DeviceId].Matrix_A_d);

		Launch_AssemblyStiffness3DQ7(numBlocksSM, threadsPerBlockSM, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::RowSize, cGPU::plan_FEM[DeviceId].CoordQ7_d, cGPU::plan_FEM[DeviceId].MatPropQ7_d, cGPU::plan_FEM[DeviceId].BC_d,
			cGPU::plan_FEM[DeviceId].Matrix_A_d);

		Launch_AssemblyStiffness3DQ8(numBlocksSM, threadsPerBlockSM, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::RowSize, cGPU::plan_FEM[DeviceId].CoordQ8_d, cGPU::plan_FEM[DeviceId].MatPropQ8_d, cGPU::plan_FEM[DeviceId].BC_d,
			cGPU::plan_FEM[DeviceId].Matrix_A_d);


		// Barrier Synchronization
#pragma omp barrier

		if(DeviceId == 0) printf("\n         Assembly Stiffness Matrix Time: %0.3f s \n", (clock()-time)/1000);

		// Barrier Synchronization
#pragma omp barrier

		// ==================================================================================================================================
		// Evaluating M matrix = 1/K:

		// To start time count to transpose stiffness matrix:
		time = clock();

		// Number of active terms (DOF x number of nodes)
		// numno = Number of nodes of the mesh

		dim3 numBlocksM(int(sqrt(double(cGPU::numdof*cGPU::plan_FEM[DeviceId].Localnumno)) /cGPU::plan_FEM[DeviceId].BlockSizeX)+1,
			            int(sqrt(double(cGPU::numdof*cGPU::plan_FEM[DeviceId].Localnumno)) /cGPU::plan_FEM[DeviceId].BlockSizeY)+1);

		dim3 threadsPerBlockM(cGPU::plan_FEM[DeviceId].BlockSizeX, cGPU::plan_FEM[DeviceId].BlockSizeY);

		Launch_EvaluateMatrixM(numBlocksM, threadsPerBlockM, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::plan_FEM[DeviceId].iiPosition_d, cGPU::plan_FEM[DeviceId].Matrix_A_d, cGPU::plan_FEM[DeviceId].Matrix_M_d);

		// Barrier Synchronization
#pragma omp barrier

		if(DeviceId == 0) printf("         Evaluating M Matrix Time: %0.3f s \n", (clock()-time)/1000);

	}

	//=============== x =============== x =============== x =============== x =============== x =============== x ===============

	/*cudaSetDevice(0);

	using namespace std;
	ofstream outfile;
	std::string fileName2 = "Matrix_M_d_GPU";
	fileName2 += ".dat";
	outfile.open(fileName2.c_str());


	double *NelAux = (double *)malloc(numdof*plan_FEM[0].Localnumno*sizeof(double));
	cudaMemcpy(NelAux, plan_FEM[0].Matrix_M_d, numdof*plan_FEM[0].Localnumno*sizeof(double), cudaMemcpyDeviceToHost);
	for(int j=0; j<numdof*plan_FEM[0].Localnumno; j++) {
		outfile << endl;  outfile << NelAux[j];
		//if(NelAux[j] != 0) printf("Matrix_M_d : %f \n", NelAux[j]);
	}

	outfile.close();*/

	/*cudaSetDevice(2);  // Sets device "0" as the current device

	using namespace std;
	ofstream outfile;
	std::string fileName2 = "StiffnessMatrixGPU";
	fileName2 += ".dat";
	outfile.open(fileName2.c_str());

	double *Matrix_K_h;
	Matrix_K_h  = (double *)malloc(numdof*plan_FEM[2].GPUNumThread*RowSize*sizeof(double));           // Allocating menoty space for host

	cudaMemcpy(Matrix_K_h, plan_FEM[2].Matrix_A_d, numdof*plan_FEM[2].GPUNumThread*RowSize*sizeof(double), cudaMemcpyDeviceToHost);

	for(int j=0; j<numdof*plan_FEM[2].GPUNumThread*RowSize; j++) {
	//for(int j=0; j<RowSize; j++) {
		if(Matrix_K_h[j] != 0) outfile << endl;  outfile << Matrix_K_h[j];
	}

	outfile.close();*/

	return 1;

}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============

int cGPU::SolveConjugateGradientOnGPU(double *NodeForceChr, double *VectorChr_X_h)
{

	//=============== x =============== x =============== x =============== x =============== x =============== x ===============

	// Solve linear system of equation with conjugate gradient on Multi GPUs:

	// To define one CPU thread for each GPU:

	omp_set_num_threads(deviceCount);

#pragma omp parallel  // It begins linear system solver /////////////////////////////////////////////////////////////////////////////////////
	{

		int DeviceId = omp_get_thread_num();

		int cont, BlockSizeNumMultiProc, numnoglobal;
		double delta_new;
		double time;

		// ==================================================================================================================================
		// Sets device as the current device and allocate memory

		cudaSetDevice(DeviceId);

		// ==================================================================================================================================
		// Cleaning Vectors

		cudaMemset(cGPU::plan_FEM[DeviceId].delta_GPU_d,       0, cGPU::plan_FEM[DeviceId].DeviceNumber*                                  sizeof(double));
		cudaMemset(cGPU::plan_FEM[DeviceId].Vector_D_Global_d, 0, cGPU::numdof*                       cGPU::numno*                        sizeof(double));
		cudaMemset(cGPU::plan_FEM[DeviceId].Vector_X_d,        0, cGPU::numdof*                       cGPU::plan_FEM[DeviceId].Localnumno*sizeof(double));
		cudaMemset(cGPU::plan_FEM[DeviceId].Vector_Q_d,        0, cGPU::numdof*                       cGPU::plan_FEM[DeviceId].Localnumno*sizeof(double));
		cudaMemset(cGPU::plan_FEM[DeviceId].Vector_R_d,        0, cGPU::numdof*                       cGPU::plan_FEM[DeviceId].Localnumno*sizeof(double));

		// Copying 1/4 load Vector to each GPU

		cudaMemcpy(cGPU::plan_FEM[DeviceId].Vector_B_d, NodeForceChr+numdof*cGPU::plan_FEM[DeviceId].LocalnumnoAcum,
			       cGPU::numdof*cGPU::plan_FEM[DeviceId].Localnumno*sizeof(double), cudaMemcpyHostToDevice);

		// ==================================================================================================================================
		// Solver on GPU of Linear System of Equations - Conjugate Gradient Method

		// Number of blocks needs to sum the terms of vector:
		if(plan_FEM[DeviceId].NumMultiProcessor <= 4)       BlockSizeNumMultiProc =  4;
		else if(cGPU::plan_FEM[DeviceId].NumMultiProcessor <=  8) BlockSizeNumMultiProc =  8;
		else if(cGPU::plan_FEM[DeviceId].NumMultiProcessor <= 16) BlockSizeNumMultiProc = 16;
		else if(cGPU::plan_FEM[DeviceId].NumMultiProcessor <= 32) BlockSizeNumMultiProc = 32;

		BlockSizeNumMultiProc = 16;

		// **************  Evaluating Block and Thread dimesion:  **************

		// Block size for local vector operation
		dim3 BlockSizeVector(int(sqrt(double(cGPU::numdof*cGPU::plan_FEM[DeviceId].Localnumno))/cGPU::plan_FEM[DeviceId].BlockSizeX)+1,
			                 int(sqrt(double(cGPU::numdof*cGPU::plan_FEM[DeviceId].Localnumno))/cGPU::plan_FEM[DeviceId].BlockSizeY)+1);

		dim3 ThreadsPerBlockXY(cGPU::plan_FEM[DeviceId].BlockSizeX, cGPU::plan_FEM[DeviceId].BlockSizeY);

		// Block size for global vector operation
		numnoglobal = cGPU::plan_FEM[0].Localnumno + cGPU::plan_FEM[1].Localnumno + cGPU::plan_FEM[2].Localnumno + cGPU::plan_FEM[3].Localnumno;

		dim3 BlockSizeVectorGlobal(int(sqrt(double(cGPU::numdof*numnoglobal))/cGPU::plan_FEM[DeviceId].BlockSizeX)+1,
			                       int(sqrt(double(cGPU::numdof*numnoglobal))/cGPU::plan_FEM[DeviceId].BlockSizeY)+1);

		// Block size for local matrix operation
		dim3 BlockSizeMatrix(int(sqrt(double(cGPU::numdof*cGPU::plan_FEM[DeviceId].Localnumno*cGPU::RowSize))/cGPU::plan_FEM[DeviceId].BlockSizeX)+1,
			                 int(sqrt(double(cGPU::numdof*cGPU::plan_FEM[DeviceId].Localnumno*cGPU::RowSize))/cGPU::plan_FEM[DeviceId].BlockSizeY)+1);

		// **************  This Kernel Subtracts a Vector by a Vector:  **************

		Launch_SUBT_V_V(BlockSizeVector, ThreadsPerBlockXY, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::plan_FEM[DeviceId].Vector_B_d,
			cGPU::plan_FEM[DeviceId].Vector_R_d); // =======================> {r} = {b} - [A]{x}


		// **************  This Kernel Multiplies a Vector by a other Vector:  **************

		// Minimum vector size is: NumMultiProcessor * 2 * blocksize

		/*switch(DeviceId) {

		case 0:  // Run device 0
			// **************  This Kernel Multiplies a Diagonal Matrix by a Vector:  **************
			Launch_MULT_DM_V(BlockSizeVector, ThreadsPerBlockXY, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::plan_FEM[DeviceId].Matrix_M_d, cGPU::plan_FEM[DeviceId].Vector_R_d, cGPU::plan_FEM[DeviceId].Vector_D_d); // =======================> {d} = [M]-1{r}

			Launch_MULT_V_V(BlockSizeNumMultiProc, cGPU::plan_FEM[DeviceId].BlockSize, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::plan_FEM[DeviceId].Vector_R_d, cGPU::plan_FEM[DeviceId].Vector_D_d, cGPU::plan_FEM[DeviceId].delta_new_d); // =======================> delta_new = {r}T{d}
			// **************  This Kernel add up the tems of the delta_new_d vector:  **************
			Launch_SUM_V(BlockSizeVector, ThreadsPerBlockXY, BlockSizeNumMultiProc, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, 0, cGPU::plan_FEM[DeviceId].delta_new_d, cGPU::plan_FEM[DeviceId].delta_GPU_d, cGPU::plan_FEM[DeviceId].Result_d);  // =======================> {delta_new_V_d} = {r}T{d}

			break;

		case 1:  // Run device 1
			// **************  This Kernel Multiplies a Diagonal Matrix by a Vector:  **************
			Launch_MULT_DM_V(BlockSizeVector, ThreadsPerBlockXY, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::plan_FEM[DeviceId].Matrix_M_d, cGPU::plan_FEM[DeviceId].Vector_R_d, cGPU::plan_FEM[DeviceId].Vector_D_d); // =======================> {d} = [M]-1{r}

			Launch_MULT_V_V(BlockSizeNumMultiProc, cGPU::plan_FEM[DeviceId].BlockSize, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::plan_FEM[DeviceId].Vector_R_d, cGPU::plan_FEM[DeviceId].Vector_D_d, cGPU::plan_FEM[DeviceId].delta_new_d); // =======================> delta_new = {r}T{d}
			// **************  This Kernel add up the tems of the delta_new_d vector:  **************
			Launch_SUM_V(BlockSizeVector, ThreadsPerBlockXY, BlockSizeNumMultiProc, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, 1, cGPU::plan_FEM[DeviceId].delta_new_d, cGPU::plan_FEM[DeviceId].delta_GPU_d, cGPU::plan_FEM[DeviceId].Result_d);  // =======================> {delta_new_V_d} = {r}T{d}

			break;

		case 2:  // Run device 2
			// **************  This Kernel Multiplies a Diagonal Matrix by a Vector:  **************
			Launch_MULT_DM_V(BlockSizeVector, ThreadsPerBlockXY, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::plan_FEM[DeviceId].Matrix_M_d, cGPU::plan_FEM[DeviceId].Vector_R_d, cGPU::plan_FEM[DeviceId].Vector_D_d); // =======================> {d} = [M]-1{r}

			Launch_MULT_V_V(BlockSizeNumMultiProc, cGPU::plan_FEM[DeviceId].BlockSize, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::plan_FEM[DeviceId].Vector_R_d, cGPU::plan_FEM[DeviceId].Vector_D_d, cGPU::plan_FEM[DeviceId].delta_new_d); // =======================> delta_new = {r}T{d}
			// **************  This Kernel add up the tems of the delta_new_d vector:  **************
			Launch_SUM_V(BlockSizeVector, ThreadsPerBlockXY, BlockSizeNumMultiProc, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, 2, cGPU::plan_FEM[DeviceId].delta_new_d, cGPU::plan_FEM[DeviceId].delta_GPU_d, cGPU::plan_FEM[DeviceId].Result_d);  // =======================> {delta_new_V_d} = {r}T{d}

			break;

		case 3:  // Run device 3
			// **************  This Kernel Multiplies a Diagonal Matrix by a Vector:  **************
			Launch_MULT_DM_V(BlockSizeVector, ThreadsPerBlockXY, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::plan_FEM[DeviceId].Matrix_M_d, cGPU::plan_FEM[DeviceId].Vector_R_d, cGPU::plan_FEM[DeviceId].Vector_D_d); // =======================> {d} = [M]-1{r}

			Launch_MULT_V_V(BlockSizeNumMultiProc, cGPU::plan_FEM[DeviceId].BlockSize, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::plan_FEM[DeviceId].Vector_R_d, cGPU::plan_FEM[DeviceId].Vector_D_d, cGPU::plan_FEM[DeviceId].delta_new_d); // =======================> delta_new = {r}T{d}
			// **************  This Kernel add up the tems of the delta_new_d vector:  **************
			Launch_SUM_V(BlockSizeVector, ThreadsPerBlockXY, BlockSizeNumMultiProc, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, 3, cGPU::plan_FEM[DeviceId].delta_new_d, cGPU::plan_FEM[DeviceId].delta_GPU_d, cGPU::plan_FEM[DeviceId].Result_d);  // =======================> {delta_new_V_d} = {r}T{d}

			break;

		}*/

		// **************  This Kernel Multiplies a Diagonal Matrix by a Vector:  **************
		Launch_MULT_DM_V(BlockSizeVector, ThreadsPerBlockXY, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::plan_FEM[DeviceId].Matrix_M_d,
			cGPU::plan_FEM[DeviceId].Vector_R_d, cGPU::plan_FEM[DeviceId].Vector_D_d); // =======================> {d} = [M]-1{r}

		Launch_MULT_V_V(BlockSizeNumMultiProc, cGPU::plan_FEM[DeviceId].BlockSize, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno,
			cGPU::plan_FEM[DeviceId].Vector_R_d, cGPU::plan_FEM[DeviceId].Vector_D_d, cGPU::plan_FEM[DeviceId].delta_new_d); // =======================> delta_new = {r}T{d}

		// **************  This Kernel add up the tems of the delta_new_d vector:  **************
		Launch_SUM_V(BlockSizeVector, ThreadsPerBlockXY, BlockSizeNumMultiProc, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, DeviceId,
			cGPU::plan_FEM[DeviceId].delta_new_d, cGPU::plan_FEM[DeviceId].delta_GPU_d, cGPU::plan_FEM[DeviceId].Result_d);  // =======================> {delta_new_V_d} = {r}T{d}

		// **************  Copies memory from one device to memory on another device:  **************



#pragma omp barrier
		/*if(DeviceId == 0)  // Sets device "0" as the current device
			cudaMemcpy(cGPU::plan_FEM[DeviceId].delta_GPU_d,   cGPU::plan_FEM[DeviceId].Result_d, sizeof(double), cudaMemcpyDeviceToDevice);

		if(DeviceId == 1)  // Sets device "1" as the current device
			cudaMemcpy(cGPU::plan_FEM[DeviceId].delta_GPU_d+1, cGPU::plan_FEM[DeviceId].Result_d, sizeof(double), cudaMemcpyDeviceToDevice);

		if(DeviceId == 2)  // Sets device "2" as the current device
			cudaMemcpy(cGPU::plan_FEM[DeviceId].delta_GPU_d+2, cGPU::plan_FEM[DeviceId].Result_d, sizeof(double), cudaMemcpyDeviceToDevice);

		if(DeviceId == 3)  // Sets device "3" as the current device
			cudaMemcpy(cGPU::plan_FEM[DeviceId].delta_GPU_d+3, cGPU::plan_FEM[DeviceId].Result_d, sizeof(double), cudaMemcpyDeviceToDevice);*/

		cudaMemcpy(cGPU::plan_FEM[DeviceId].delta_GPU_d+DeviceId, cGPU::plan_FEM[DeviceId].Result_d, sizeof(double), cudaMemcpyDeviceToDevice);

/*#pragma omp barrier  // Synchronizes all threads in a team; all threads pause at the barrier, until all threads execute the barrier
		if(DeviceId == 0) {
			cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].delta_GPU_d+1, 0, cGPU::plan_FEM[1].Result_d, 1, sizeof(double));
			cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].delta_GPU_d+2, 0, cGPU::plan_FEM[2].Result_d, 2, sizeof(double));
			cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].delta_GPU_d+3, 0, cGPU::plan_FEM[3].Result_d, 3, sizeof(double));
		}

#pragma omp barrier
		if(DeviceId == 1) {
			cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].delta_GPU_d  , 1, cGPU::plan_FEM[0].Result_d, 0, sizeof(double));
			cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].delta_GPU_d+2, 1, cGPU::plan_FEM[2].Result_d, 2, sizeof(double));
			cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].delta_GPU_d+3, 1, cGPU::plan_FEM[3].Result_d, 3, sizeof(double));
		}

#pragma omp barrier
		if(DeviceId == 2) {
			cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].delta_GPU_d  , 2, cGPU::plan_FEM[0].Result_d, 0, sizeof(double));
			cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].delta_GPU_d+1, 2, cGPU::plan_FEM[1].Result_d, 1, sizeof(double));
			cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].delta_GPU_d+3, 2, cGPU::plan_FEM[3].Result_d, 3, sizeof(double));
		}

#pragma omp barrier
		if(DeviceId == 3) {
			cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].delta_GPU_d  , 3, cGPU::plan_FEM[0].Result_d, 0, sizeof(double));
			cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].delta_GPU_d+1, 3, cGPU::plan_FEM[1].Result_d, 1, sizeof(double));
			cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].delta_GPU_d+2, 3, cGPU::plan_FEM[2].Result_d, 2, sizeof(double));

		}*/

#pragma omp barrier
		cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].delta_GPU_d+cGPU::plan_FEM[DeviceId].Aux_1_h[0], DeviceId, cGPU::plan_FEM[cGPU::plan_FEM[DeviceId].Aux_1_h[0]].Result_d,
			cGPU::plan_FEM[DeviceId].Aux_1_h[0], sizeof(double));
		cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].delta_GPU_d+cGPU::plan_FEM[DeviceId].Aux_1_h[1], DeviceId, cGPU::plan_FEM[cGPU::plan_FEM[DeviceId].Aux_1_h[1]].Result_d,
			cGPU::plan_FEM[DeviceId].Aux_1_h[1], sizeof(double));
		cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].delta_GPU_d+cGPU::plan_FEM[DeviceId].Aux_1_h[2], DeviceId, cGPU::plan_FEM[cGPU::plan_FEM[DeviceId].Aux_1_h[2]].Result_d,
			cGPU::plan_FEM[DeviceId].Aux_1_h[2], sizeof(double));

		// Barrier Synchronization
#pragma omp barrier

		// **************  Adding delta_3_new_V_d [0]+[1]+[2]+[3] terms:  **************

		/*switch(DeviceId) {

		case 0:  // Run device 0
			Launch_SUM_GPU_4(BlockSizeVector, ThreadsPerBlockXY, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::plan_FEM[DeviceId].delta_GPU_d, cGPU::plan_FEM[DeviceId].delta_new_V_d);
			break;

		case 1:  // Run device 1
			Launch_SUM_GPU_4(BlockSizeVector, ThreadsPerBlockXY, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::plan_FEM[DeviceId].delta_GPU_d, cGPU::plan_FEM[DeviceId].delta_new_V_d);
			break;

		case 2:  // Run device 2
			Launch_SUM_GPU_4(BlockSizeVector, ThreadsPerBlockXY, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::plan_FEM[DeviceId].delta_GPU_d, cGPU::plan_FEM[DeviceId].delta_new_V_d);
			break;

		case 3:  // Run device 3
			Launch_SUM_GPU_4(BlockSizeVector, ThreadsPerBlockXY, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::plan_FEM[DeviceId].delta_GPU_d, cGPU::plan_FEM[DeviceId].delta_new_V_d);
			break;

		}*/

		Launch_SUM_GPU_4(BlockSizeVector, ThreadsPerBlockXY, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::plan_FEM[DeviceId].delta_GPU_d, cGPU::plan_FEM[DeviceId].delta_new_V_d);

		// Barrier Synchronization
#pragma omp barrier

		cudaMemcpy(plan_FEM[DeviceId].delta_new_V_h, cGPU::plan_FEM[DeviceId].delta_new_V_d, sizeof(double)*cGPU::numdof*cGPU::plan_FEM[DeviceId].Localnumno, cudaMemcpyDeviceToHost);

		delta_new = plan_FEM[DeviceId].delta_new_V_h[0];

		// err = cur_err
		double epsilon  = 0.00001;
		//double epsilon  = 0.01;
		//double epsilon  = 0.075;
		double error = (double)(delta_new * epsilon * epsilon) ;

		// ************************************************************************************

		cont = 0;

		if(DeviceId == 0) {  // Only device 0 print message

			printf("\n");
			printf("         Solving Linear Equation System on MultiGPU algorithm \n");
			printf("         ======================================================= \n");
			printf("         * Conjugate Gradient Method * \n\n");

		}

		// *************************************************************************************************************************************************************************************************

		// Barrier Synchronization
#pragma omp barrier

		time = clock();  // Time count

		while(delta_new > error && cont < 4000) {

			// **************  This Kernel copy data from Vector_D_d to Vector_D_Global_d:  **************

			/*switch(DeviceId) {

			case 0:  // Run device 0
				Launch_Copy_Local_Global(BlockSizeVector, ThreadsPerBlockXY, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, 0,                                                                                                     cGPU::plan_FEM[DeviceId].Vector_D_d, cGPU::plan_FEM[DeviceId].Vector_D_Global_d);
				break;

			case 1:  // Run device 1
				Launch_Copy_Local_Global(BlockSizeVector, ThreadsPerBlockXY, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::numdof*cGPU::plan_FEM[0].Localnumno,                                                             cGPU::plan_FEM[DeviceId].Vector_D_d, cGPU::plan_FEM[DeviceId].Vector_D_Global_d);
				break;

			case 2:  // Run device 2
				Launch_Copy_Local_Global(BlockSizeVector, ThreadsPerBlockXY, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::numdof*(cGPU::plan_FEM[0].Localnumno+cGPU::plan_FEM[1].Localnumno),                              cGPU::plan_FEM[DeviceId].Vector_D_d, cGPU::plan_FEM[DeviceId].Vector_D_Global_d);
				break;

			case 3:  // Run device 3
				Launch_Copy_Local_Global(BlockSizeVector, ThreadsPerBlockXY, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::numdof*(cGPU::plan_FEM[0].Localnumno+cGPU::plan_FEM[1].Localnumno+cGPU::plan_FEM[2].Localnumno), cGPU::plan_FEM[DeviceId].Vector_D_d, cGPU::plan_FEM[DeviceId].Vector_D_Global_d);
				break;

			}*/

			Launch_Copy_Local_Global(BlockSizeVector, ThreadsPerBlockXY, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::numdof*cGPU::plan_FEM[DeviceId].LocalnumnoAcum,
				cGPU::plan_FEM[DeviceId].Vector_D_d, cGPU::plan_FEM[DeviceId].Vector_D_Global_d);

			// **************   Assembling global vector Vector_D_Global_d from each GPU information:  **************

#pragma omp barrier
			/*if(DeviceId == 0)  // Sets device "0" as the current device
				cudaMemcpy(cGPU::plan_FEM[DeviceId].Vector_D_Global_d,                                                                                                       cGPU::plan_FEM[DeviceId].Vector_D_d, cGPU::numdof*cGPU::plan_FEM[DeviceId].Localnumno*sizeof(double), cudaMemcpyDeviceToDevice);

			if(DeviceId == 1)  // Sets device "1" as the current device
				cudaMemcpy(cGPU::plan_FEM[DeviceId].Vector_D_Global_d+cGPU::numdof*(cGPU::plan_FEM[0].Localnumno),                                                           cGPU::plan_FEM[DeviceId].Vector_D_d, cGPU::numdof*cGPU::plan_FEM[DeviceId].Localnumno*sizeof(double), cudaMemcpyDeviceToDevice);

			if(DeviceId == 2)  // Sets device "2" as the current device
				cudaMemcpy(cGPU::plan_FEM[DeviceId].Vector_D_Global_d+cGPU::numdof*(cGPU::plan_FEM[0].Localnumno+cGPU::plan_FEM[1].Localnumno),                              cGPU::plan_FEM[DeviceId].Vector_D_d, cGPU::numdof*cGPU::plan_FEM[DeviceId].Localnumno*sizeof(double), cudaMemcpyDeviceToDevice);

			if(DeviceId == 3)  // Sets device "3" as the current device
				cudaMemcpy(cGPU::plan_FEM[DeviceId].Vector_D_Global_d+cGPU::numdof*(cGPU::plan_FEM[0].Localnumno+cGPU::plan_FEM[1].Localnumno+cGPU::plan_FEM[2].Localnumno), cGPU::plan_FEM[DeviceId].Vector_D_d, cGPU::numdof*cGPU::plan_FEM[DeviceId].Localnumno*sizeof(double), cudaMemcpyDeviceToDevice);*/

			cudaMemcpy(cGPU::plan_FEM[DeviceId].Vector_D_Global_d+cGPU::numdof*cGPU::plan_FEM[DeviceId].LocalnumnoAcum,
				cGPU::plan_FEM[DeviceId].Vector_D_d, cGPU::numdof*cGPU::plan_FEM[DeviceId].Localnumno*sizeof(double), cudaMemcpyDeviceToDevice);

#pragma omp barrier  // Synchronizes all threads in a team; all threads pause at the barrier, until all threads execute the barrier
			/*if(DeviceId == 0) {
				cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].Vector_D_Global_d+cGPU::numdof*(cGPU::plan_FEM[0].Localnumno),                                                           0, cGPU::plan_FEM[1].Vector_D_d, 1, cGPU::numdof*cGPU::plan_FEM[1].Localnumno*sizeof(double));
				cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].Vector_D_Global_d+cGPU::numdof*(cGPU::plan_FEM[0].Localnumno+cGPU::plan_FEM[1].Localnumno),                              0, cGPU::plan_FEM[2].Vector_D_d, 2, cGPU::numdof*cGPU::plan_FEM[2].Localnumno*sizeof(double));
				cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].Vector_D_Global_d+cGPU::numdof*(cGPU::plan_FEM[0].Localnumno+cGPU::plan_FEM[1].Localnumno+cGPU::plan_FEM[2].Localnumno), 0, cGPU::plan_FEM[3].Vector_D_d, 3, cGPU::numdof*cGPU::plan_FEM[3].Localnumno*sizeof(double));
			}

#pragma omp barrier
			if(DeviceId == 1) {
				cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].Vector_D_Global_d,                                                                                                       1, cGPU::plan_FEM[0].Vector_D_d, 0, cGPU::numdof*cGPU::plan_FEM[0].Localnumno*sizeof(double));
				cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].Vector_D_Global_d+cGPU::numdof*(cGPU::plan_FEM[0].Localnumno+cGPU::plan_FEM[1].Localnumno),                              1, cGPU::plan_FEM[2].Vector_D_d, 2, cGPU::numdof*cGPU::plan_FEM[2].Localnumno*sizeof(double));
				cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].Vector_D_Global_d+cGPU::numdof*(cGPU::plan_FEM[0].Localnumno+cGPU::plan_FEM[1].Localnumno+cGPU::plan_FEM[2].Localnumno), 1, cGPU::plan_FEM[3].Vector_D_d, 3, cGPU::numdof*cGPU::plan_FEM[3].Localnumno*sizeof(double));
			}

#pragma omp barrier
			if(DeviceId == 2) {
				cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].Vector_D_Global_d,                                                                                                       2, cGPU::plan_FEM[0].Vector_D_d, 0, cGPU::numdof*cGPU::plan_FEM[0].Localnumno*sizeof(double));
				cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].Vector_D_Global_d+cGPU::numdof*(cGPU::plan_FEM[0].Localnumno),                                                           2, cGPU::plan_FEM[1].Vector_D_d, 1, cGPU::numdof*cGPU::plan_FEM[1].Localnumno*sizeof(double));
				cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].Vector_D_Global_d+cGPU::numdof*(cGPU::plan_FEM[0].Localnumno+cGPU::plan_FEM[1].Localnumno+cGPU::plan_FEM[2].Localnumno), 2, cGPU::plan_FEM[3].Vector_D_d, 3, cGPU::numdof*cGPU::plan_FEM[3].Localnumno*sizeof(double));
			}

#pragma omp barrier
			if(DeviceId == 3) {
				cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].Vector_D_Global_d,                                                                          3, cGPU::plan_FEM[0].Vector_D_d, 0, cGPU::numdof*cGPU::plan_FEM[0].Localnumno*sizeof(double));
				cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].Vector_D_Global_d+cGPU::numdof*(cGPU::plan_FEM[0].Localnumno),                              3, cGPU::plan_FEM[1].Vector_D_d, 1, cGPU::numdof*cGPU::plan_FEM[1].Localnumno*sizeof(double));
				cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].Vector_D_Global_d+cGPU::numdof*(cGPU::plan_FEM[0].Localnumno+cGPU::plan_FEM[1].Localnumno), 3, cGPU::plan_FEM[2].Vector_D_d, 2, cGPU::numdof*cGPU::plan_FEM[2].Localnumno*sizeof(double));
			}*/

			cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].Vector_D_Global_d+cGPU::numdof*(cGPU::plan_FEM[DeviceId].LocalnumnoAcumVar[0]), DeviceId,
				cGPU::plan_FEM[cGPU::plan_FEM[DeviceId].Aux_1_h[0]].Vector_D_d, cGPU::plan_FEM[DeviceId].Aux_1_h[0],
				cGPU::numdof*cGPU::plan_FEM[cGPU::plan_FEM[DeviceId].Aux_1_h[0]].Localnumno*sizeof(double));

			cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].Vector_D_Global_d+cGPU::numdof*(cGPU::plan_FEM[DeviceId].LocalnumnoAcumVar[1]), DeviceId,
				cGPU::plan_FEM[cGPU::plan_FEM[DeviceId].Aux_1_h[1]].Vector_D_d, cGPU::plan_FEM[DeviceId].Aux_1_h[1],
				cGPU::numdof*cGPU::plan_FEM[cGPU::plan_FEM[DeviceId].Aux_1_h[1]].Localnumno*sizeof(double));

			cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].Vector_D_Global_d+cGPU::numdof*(cGPU::plan_FEM[DeviceId].LocalnumnoAcumVar[2]), DeviceId,
				cGPU::plan_FEM[cGPU::plan_FEM[DeviceId].Aux_1_h[2]].Vector_D_d, cGPU::plan_FEM[DeviceId].Aux_1_h[2],
				cGPU::numdof*cGPU::plan_FEM[cGPU::plan_FEM[DeviceId].Aux_1_h[2]].Localnumno*sizeof(double));

			// Barrier Synchronization
#pragma omp barrier

			// **************   Multiplying matrix A by global vector Vector_D_Global_d:  **************

			/*switch(DeviceId) {

			case 0:  // Run device 0
				Launch_MULT_SM_V_128(BlockSizeMatrix, ThreadsPerBlockXY, cGPU::plan_FEM[DeviceId].BlockSize, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::RowSize, cGPU::plan_FEM[DeviceId].Matrix_A_d, cGPU::plan_FEM[DeviceId].Vector_D_Global_d, cGPU::plan_FEM[DeviceId].Vector_I_d, cGPU::plan_FEM[DeviceId].Vector_Q_d); // ======> {q} = [A]{d}
				break;

			case 1:  // Run device 1
				Launch_MULT_SM_V_128(BlockSizeMatrix, ThreadsPerBlockXY, cGPU::plan_FEM[DeviceId].BlockSize, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::RowSize, cGPU::plan_FEM[DeviceId].Matrix_A_d, cGPU::plan_FEM[DeviceId].Vector_D_Global_d, cGPU::plan_FEM[DeviceId].Vector_I_d, cGPU::plan_FEM[DeviceId].Vector_Q_d); // ======> {q} = [A]{d}
				break;

			case 2:  // Run device 2
				Launch_MULT_SM_V_128(BlockSizeMatrix, ThreadsPerBlockXY, cGPU::plan_FEM[DeviceId].BlockSize, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::RowSize, cGPU::plan_FEM[DeviceId].Matrix_A_d, cGPU::plan_FEM[DeviceId].Vector_D_Global_d, cGPU::plan_FEM[DeviceId].Vector_I_d, cGPU::plan_FEM[DeviceId].Vector_Q_d); // ======> {q} = [A]{d}
				break;

			case 3:  // Run device 3
				Launch_MULT_SM_V_128(BlockSizeMatrix, ThreadsPerBlockXY, cGPU::plan_FEM[DeviceId].BlockSize, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::RowSize, cGPU::plan_FEM[DeviceId].Matrix_A_d, cGPU::plan_FEM[DeviceId].Vector_D_Global_d, cGPU::plan_FEM[DeviceId].Vector_I_d, cGPU::plan_FEM[DeviceId].Vector_Q_d); // ======> {q} = [A]{d}
				break;

			}*/

			Launch_MULT_SM_V_128(BlockSizeMatrix, ThreadsPerBlockXY, cGPU::plan_FEM[DeviceId].BlockSize, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::RowSize,
				cGPU::plan_FEM[DeviceId].Matrix_A_d, cGPU::plan_FEM[DeviceId].Vector_D_Global_d, cGPU::plan_FEM[DeviceId].Vector_I_d, cGPU::plan_FEM[DeviceId].Vector_Q_d); // ======> {q} = [A]{d}

			// Barrier Synchronization
#pragma omp barrier

			// **************   This Kernel Multiplies a Vector by a other Vector:  **************

			/*switch(DeviceId) {

			case 0:  // Run device 0
				Launch_MULT_V_V(BlockSizeNumMultiProc, cGPU::plan_FEM[DeviceId].BlockSize, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::plan_FEM[DeviceId].Vector_D_d, cGPU::plan_FEM[DeviceId].Vector_Q_d, cGPU::plan_FEM[DeviceId].delta_aux_d); // =======================> {delta_aux_V_d} = {d}T{q}
				// **************  This Kernel add up the tems of the delta_new_d vector:  **************
				Launch_SUM_V(BlockSizeVector, ThreadsPerBlockXY, BlockSizeNumMultiProc, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, 0, cGPU::plan_FEM[DeviceId].delta_aux_d, cGPU::plan_FEM[DeviceId].delta_GPU_d, cGPU::plan_FEM[DeviceId].Result_d);  // =======================> {delta_new_V_d} = {r}T{d}
				break;

			case 1:  // Run device 1
				Launch_MULT_V_V(BlockSizeNumMultiProc, cGPU::plan_FEM[DeviceId].BlockSize, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::plan_FEM[DeviceId].Vector_D_d, cGPU::plan_FEM[DeviceId].Vector_Q_d, cGPU::plan_FEM[DeviceId].delta_aux_d); // =======================> {delta_aux_V_d} = {d}T{q}
				// **************  This Kernel add up the tems of the delta_new_d vector:  **************
				Launch_SUM_V(BlockSizeVector, ThreadsPerBlockXY, BlockSizeNumMultiProc, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, 1, cGPU::plan_FEM[DeviceId].delta_aux_d, cGPU::plan_FEM[DeviceId].delta_GPU_d, cGPU::plan_FEM[DeviceId].Result_d);  // =======================> {delta_new_V_d} = {r}T{d}
				break;

			case 2:  // Run device 2
				Launch_MULT_V_V(BlockSizeNumMultiProc, cGPU::plan_FEM[DeviceId].BlockSize, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::plan_FEM[DeviceId].Vector_D_d, cGPU::plan_FEM[DeviceId].Vector_Q_d, cGPU::plan_FEM[DeviceId].delta_aux_d); // =======================> {delta_aux_V_d} = {d}T{q}
				// **************  This Kernel add up the tems of the delta_new_d vector:  **************
				Launch_SUM_V(BlockSizeVector, ThreadsPerBlockXY, BlockSizeNumMultiProc, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, 2, cGPU::plan_FEM[DeviceId].delta_aux_d, cGPU::plan_FEM[DeviceId].delta_GPU_d, cGPU::plan_FEM[DeviceId].Result_d);  // =======================> {delta_new_V_d} = {r}T{d}
				break;

			case 3:  // Run device 3
				Launch_MULT_V_V(BlockSizeNumMultiProc, cGPU::plan_FEM[DeviceId].BlockSize, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::plan_FEM[DeviceId].Vector_D_d, cGPU::plan_FEM[DeviceId].Vector_Q_d, cGPU::plan_FEM[DeviceId].delta_aux_d); // =======================> {delta_aux_V_d} = {d}T{q}
				// **************  This Kernel add up the tems of the delta_new_d vector:  **************
				Launch_SUM_V(BlockSizeVector, ThreadsPerBlockXY, BlockSizeNumMultiProc, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, 3, cGPU::plan_FEM[DeviceId].delta_aux_d, cGPU::plan_FEM[DeviceId].delta_GPU_d, cGPU::plan_FEM[DeviceId].Result_d);  // =======================> {delta_new_V_d} = {r}T{d}
				break;

			}*/

			Launch_MULT_V_V(BlockSizeNumMultiProc, cGPU::plan_FEM[DeviceId].BlockSize, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::plan_FEM[DeviceId].Vector_D_d,
				cGPU::plan_FEM[DeviceId].Vector_Q_d, cGPU::plan_FEM[DeviceId].delta_aux_d); // =======================> {delta_aux_V_d} = {d}T{q}

			// **************  This Kernel add up the tems of the delta_new_d vector:  **************
			Launch_SUM_V(BlockSizeVector, ThreadsPerBlockXY, BlockSizeNumMultiProc, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, DeviceId, cGPU::plan_FEM[DeviceId].delta_aux_d,
				cGPU::plan_FEM[DeviceId].delta_GPU_d, cGPU::plan_FEM[DeviceId].Result_d);  // =======================> {delta_new_V_d} = {r}T{d}

			// **************  Copies memory from one device to memory on another device:  **************

#pragma omp barrier
			/*if(DeviceId == 0)  // Sets device "0" as the current device
				cudaMemcpy(cGPU::plan_FEM[DeviceId].delta_GPU_d,   cGPU::plan_FEM[DeviceId].Result_d, sizeof(double), cudaMemcpyDeviceToDevice);

			if(DeviceId == 1)  // Sets device "1" as the current device
				cudaMemcpy(cGPU::plan_FEM[DeviceId].delta_GPU_d+1, cGPU::plan_FEM[DeviceId].Result_d, sizeof(double), cudaMemcpyDeviceToDevice);

			if(DeviceId == 2)  // Sets device "2" as the current device
				cudaMemcpy(cGPU::plan_FEM[DeviceId].delta_GPU_d+2, cGPU::plan_FEM[DeviceId].Result_d, sizeof(double), cudaMemcpyDeviceToDevice);

			if(DeviceId == 3)  // Sets device "3" as the current device
				cudaMemcpy(cGPU::plan_FEM[DeviceId].delta_GPU_d+3, cGPU::plan_FEM[DeviceId].Result_d, sizeof(double), cudaMemcpyDeviceToDevice);*/

			cudaMemcpy(cGPU::plan_FEM[DeviceId].delta_GPU_d+DeviceId, cGPU::plan_FEM[DeviceId].Result_d, sizeof(double), cudaMemcpyDeviceToDevice);

/*#pragma omp barrier  // Synchronizes all threads in a team; all threads pause at the barrier, until all threads execute the barrier
			if(DeviceId == 0) {
				cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].delta_GPU_d+1, 0, cGPU::plan_FEM[1].Result_d, 1, sizeof(double));
				cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].delta_GPU_d+2, 0, cGPU::plan_FEM[2].Result_d, 2, sizeof(double));
				cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].delta_GPU_d+3, 0, cGPU::plan_FEM[3].Result_d, 3, sizeof(double));
			}

#pragma omp barrier
			if(DeviceId == 1) {
				cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].delta_GPU_d  , 1, cGPU::plan_FEM[0].Result_d, 0, sizeof(double));
				cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].delta_GPU_d+2, 1, cGPU::plan_FEM[2].Result_d, 2, sizeof(double));
				cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].delta_GPU_d+3, 1, cGPU::plan_FEM[3].Result_d, 3, sizeof(double));
			}

#pragma omp barrier
			if(DeviceId == 2) {
				cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].delta_GPU_d  , 2, cGPU::plan_FEM[0].Result_d, 0, sizeof(double));
				cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].delta_GPU_d+1, 2, cGPU::plan_FEM[1].Result_d, 1, sizeof(double));
				cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].delta_GPU_d+3, 2, cGPU::plan_FEM[3].Result_d, 3, sizeof(double));
			}

#pragma omp barrier
			if(DeviceId == 3) {
				cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].delta_GPU_d  , 3, cGPU::plan_FEM[0].Result_d, 0, sizeof(double));
				cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].delta_GPU_d+1, 3, cGPU::plan_FEM[1].Result_d, 1, sizeof(double));
				cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].delta_GPU_d+2, 3, cGPU::plan_FEM[2].Result_d, 2, sizeof(double));
			}*/

#pragma omp barrier
			cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].delta_GPU_d+cGPU::plan_FEM[DeviceId].Aux_1_h[0], DeviceId, cGPU::plan_FEM[cGPU::plan_FEM[DeviceId].Aux_1_h[0]].Result_d,
				cGPU::plan_FEM[DeviceId].Aux_1_h[0], sizeof(double));

			cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].delta_GPU_d+cGPU::plan_FEM[DeviceId].Aux_1_h[1], DeviceId, cGPU::plan_FEM[cGPU::plan_FEM[DeviceId].Aux_1_h[1]].Result_d,
				cGPU::plan_FEM[DeviceId].Aux_1_h[1], sizeof(double));

			cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].delta_GPU_d+cGPU::plan_FEM[DeviceId].Aux_1_h[2], DeviceId, cGPU::plan_FEM[cGPU::plan_FEM[DeviceId].Aux_1_h[2]].Result_d,
				cGPU::plan_FEM[DeviceId].Aux_1_h[2], sizeof(double));

			// Barrier Synchronization
#pragma omp barrier

			// **************  Adding delta_3_new_V_d [0]+[1]+[2]+[3] terms:  **************

			/*switch(DeviceId) {

			case 0:  // Run device 0
				Launch_SUM_GPU_4(BlockSizeVector, ThreadsPerBlockXY, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::plan_FEM[DeviceId].delta_GPU_d, cGPU::plan_FEM[DeviceId].delta_aux_V_d);
				break;

			case 1:  // Run device 1
				Launch_SUM_GPU_4(BlockSizeVector, ThreadsPerBlockXY, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::plan_FEM[DeviceId].delta_GPU_d, cGPU::plan_FEM[DeviceId].delta_aux_V_d);
				break;

			case 2:  // Run device 2
				Launch_SUM_GPU_4(BlockSizeVector, ThreadsPerBlockXY, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::plan_FEM[DeviceId].delta_GPU_d, cGPU::plan_FEM[DeviceId].delta_aux_V_d);
				break;

			case 3:  // Run device 3
				Launch_SUM_GPU_4(BlockSizeVector, ThreadsPerBlockXY, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::plan_FEM[DeviceId].delta_GPU_d, cGPU::plan_FEM[DeviceId].delta_aux_V_d);
				break;

			}*/

			Launch_SUM_GPU_4(BlockSizeVector, ThreadsPerBlockXY, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::plan_FEM[DeviceId].delta_GPU_d, cGPU::plan_FEM[DeviceId].delta_aux_V_d);

			// Barrier Synchronization
#pragma omp barrier

			// **************  This Kernel adds up a Vector by a alfa*Vector:  **************

			/*switch(DeviceId) {

			case 0:  // Run device 0
				Launch_ADD_V_V_X (BlockSizeVector, ThreadsPerBlockXY, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::plan_FEM[DeviceId].Vector_X_d, cGPU::plan_FEM[DeviceId].Vector_D_d, cGPU::plan_FEM[DeviceId].delta_new_V_d, cGPU::plan_FEM[DeviceId].delta_aux_V_d, cGPU::plan_FEM[DeviceId].Vector_R_d, cGPU::plan_FEM[DeviceId].Vector_Q_d); // =============> {x} = {x} + alfa{d}                                                                                                                                                    // =============> {r} = {r} - alfa{q}
				break;

			case 1:  // Run device 1
				Launch_ADD_V_V_X (BlockSizeVector, ThreadsPerBlockXY, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::plan_FEM[DeviceId].Vector_X_d, cGPU::plan_FEM[DeviceId].Vector_D_d, cGPU::plan_FEM[DeviceId].delta_new_V_d, cGPU::plan_FEM[DeviceId].delta_aux_V_d, cGPU::plan_FEM[DeviceId].Vector_R_d, cGPU::plan_FEM[DeviceId].Vector_Q_d); // =============> {x} = {x} + alfa{d}                                                                                                                                                              // =============> {r} = {r} - alfa{q}
				break;

			case 2:  // Run device 2
				Launch_ADD_V_V_X (BlockSizeVector, ThreadsPerBlockXY, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::plan_FEM[DeviceId].Vector_X_d, cGPU::plan_FEM[DeviceId].Vector_D_d, cGPU::plan_FEM[DeviceId].delta_new_V_d, cGPU::plan_FEM[DeviceId].delta_aux_V_d, cGPU::plan_FEM[DeviceId].Vector_R_d, cGPU::plan_FEM[DeviceId].Vector_Q_d); // =============> {x} = {x} + alfa{d}	                                                                                                                                                              // =============> {r} = {r} - alfa{q}
				break;

			case 3:  // Run device 3
				Launch_ADD_V_V_X (BlockSizeVector, ThreadsPerBlockXY, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::plan_FEM[DeviceId].Vector_X_d, cGPU::plan_FEM[DeviceId].Vector_D_d, cGPU::plan_FEM[DeviceId].delta_new_V_d, cGPU::plan_FEM[DeviceId].delta_aux_V_d, cGPU::plan_FEM[DeviceId].Vector_R_d, cGPU::plan_FEM[DeviceId].Vector_Q_d); // =============> {x} = {x} + alfa{d}                                                                                                                                                            // =============> {r} = {r} - alfa{q}
				break;

			}*/

			Launch_ADD_V_V_X (BlockSizeVector, ThreadsPerBlockXY, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::plan_FEM[DeviceId].Vector_X_d, cGPU::plan_FEM[DeviceId].Vector_D_d,
				cGPU::plan_FEM[DeviceId].delta_new_V_d, cGPU::plan_FEM[DeviceId].delta_aux_V_d, cGPU::plan_FEM[DeviceId].Vector_R_d, cGPU::plan_FEM[DeviceId].Vector_Q_d); // =============> {x} = {x} + alfa{d}

			// **************  This Kernel Multiplies a Diagonal Matrix by a Vector:  **************

			Launch_MULT_DM_V(BlockSizeVector, ThreadsPerBlockXY, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::plan_FEM[DeviceId].Matrix_M_d, cGPU::plan_FEM[DeviceId].Vector_R_d,
				cGPU::plan_FEM[DeviceId].Vector_S_d); // =======================> {s} = [M]-1{r}

			// **************  Storing delta_new in delta_old:  **************

			Launch_UPDATE_DELTA(BlockSizeVector, ThreadsPerBlockXY, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::plan_FEM[DeviceId].delta_new_V_d,
				cGPU::plan_FEM[DeviceId].delta_old_V_d);  // =======================> {delta_old} = {delta_new}

			// **************  This Kernel Multiplies a Vector by a other Vector:  **************

			Launch_MULT_V_V(BlockSizeNumMultiProc, cGPU::plan_FEM[DeviceId].BlockSize, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::plan_FEM[DeviceId].Vector_R_d,
				cGPU::plan_FEM[DeviceId].Vector_S_d, cGPU::plan_FEM[DeviceId].delta_new_d); // =======================> {delta_new} = {r}T{s}

			// **************  This Kernel add up the tems of the delta_new_d vector:  **************

			/*switch(DeviceId) {

			case 0:  // Run device 0
				Launch_SUM_V(BlockSizeVector, ThreadsPerBlockXY, BlockSizeNumMultiProc, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, 0, cGPU::plan_FEM[DeviceId].delta_new_d, cGPU::plan_FEM[0].delta_GPU_d, cGPU::plan_FEM[DeviceId].Result_d);  // =======================> {delta_new_V_d} = {r}T{d}
				break;

			case 1:  // Run device 1
				Launch_SUM_V(BlockSizeVector, ThreadsPerBlockXY, BlockSizeNumMultiProc, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, 1, cGPU::plan_FEM[DeviceId].delta_new_d, cGPU::plan_FEM[1].delta_GPU_d, cGPU::plan_FEM[DeviceId].Result_d);  // =======================> {delta_new_V_d} = {r}T{d}
				break;

			case 2:  // Run device 2
				Launch_SUM_V(BlockSizeVector, ThreadsPerBlockXY, BlockSizeNumMultiProc, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, 2, cGPU::plan_FEM[DeviceId].delta_new_d, cGPU::plan_FEM[2].delta_GPU_d, cGPU::plan_FEM[DeviceId].Result_d);  // =======================> {delta_new_V_d} = {r}T{d}
				break;

			case 3:  // Run device 3
				Launch_SUM_V(BlockSizeVector, ThreadsPerBlockXY, BlockSizeNumMultiProc, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, 3, cGPU::plan_FEM[DeviceId].delta_new_d, cGPU::plan_FEM[3].delta_GPU_d, cGPU::plan_FEM[DeviceId].Result_d);  // =======================> {delta_new_V_d} = {r}T{d}
				break;

			}*/

			Launch_SUM_V(BlockSizeVector, ThreadsPerBlockXY, BlockSizeNumMultiProc, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, DeviceId, cGPU::plan_FEM[DeviceId].delta_new_d,
				cGPU::plan_FEM[DeviceId].delta_GPU_d, cGPU::plan_FEM[DeviceId].Result_d);  // =======================> {delta_new_V_d} = {r}T{d}

			// **************  Copies memory from one device to memory on another device:  **************

#pragma omp barrier
			/*if(DeviceId == 0)  // Sets device "0" as the current device
				cudaMemcpy(cGPU::plan_FEM[DeviceId].delta_GPU_d,   cGPU::plan_FEM[DeviceId].Result_d, sizeof(double), cudaMemcpyDeviceToDevice);

			if(DeviceId == 1)  // Sets device "1" as the current device
				cudaMemcpy(cGPU::plan_FEM[DeviceId].delta_GPU_d+1, cGPU::plan_FEM[DeviceId].Result_d, sizeof(double), cudaMemcpyDeviceToDevice);

			if(DeviceId == 2)  // Sets device "2" as the current device
				cudaMemcpy(cGPU::plan_FEM[DeviceId].delta_GPU_d+2, cGPU::plan_FEM[DeviceId].Result_d, sizeof(double), cudaMemcpyDeviceToDevice);

			if(DeviceId == 3)  // Sets device "3" as the current device
				cudaMemcpy(cGPU::plan_FEM[DeviceId].delta_GPU_d+3, cGPU::plan_FEM[DeviceId].Result_d, sizeof(double), cudaMemcpyDeviceToDevice);*/

			cudaMemcpy(cGPU::plan_FEM[DeviceId].delta_GPU_d+DeviceId, cGPU::plan_FEM[DeviceId].Result_d, sizeof(double), cudaMemcpyDeviceToDevice);

/*#pragma omp barrier  // Synchronizes all threads in a team; all threads pause at the barrier, until all threads execute the barrier
			if(DeviceId == 0) {
				cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].delta_GPU_d+1, 0, cGPU::plan_FEM[1].Result_d, 1, sizeof(double));
				cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].delta_GPU_d+2, 0, cGPU::plan_FEM[2].Result_d, 2, sizeof(double));
				cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].delta_GPU_d+3, 0, cGPU::plan_FEM[3].Result_d, 3, sizeof(double));
			}

#pragma omp barrier
			if(DeviceId == 1) {
				cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].delta_GPU_d  , 1, cGPU::plan_FEM[0].Result_d, 0, sizeof(double));
				cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].delta_GPU_d+2, 1, cGPU::plan_FEM[2].Result_d, 2, sizeof(double));
				cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].delta_GPU_d+3, 1, cGPU::plan_FEM[3].Result_d, 3, sizeof(double));
			}

#pragma omp barrier
			if(DeviceId == 2) {
				cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].delta_GPU_d  , 2, cGPU::plan_FEM[0].Result_d, 0, sizeof(double));
				cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].delta_GPU_d+1, 2, cGPU::plan_FEM[1].Result_d, 1, sizeof(double));
				cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].delta_GPU_d+3, 2, cGPU::plan_FEM[3].Result_d, 3, sizeof(double));
			}

#pragma omp barrier
			if(DeviceId == 3) {
				cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].delta_GPU_d  , 3, cGPU::plan_FEM[0].Result_d, 0, sizeof(double));
				cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].delta_GPU_d+1, 3, cGPU::plan_FEM[1].Result_d, 1, sizeof(double));
				cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].delta_GPU_d+2, 3, cGPU::plan_FEM[2].Result_d, 2, sizeof(double));
			}*/

#pragma omp barrier
			cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].delta_GPU_d+cGPU::plan_FEM[DeviceId].Aux_1_h[0], DeviceId, cGPU::plan_FEM[cGPU::plan_FEM[DeviceId].Aux_1_h[0]].Result_d,
				cGPU::plan_FEM[DeviceId].Aux_1_h[0], sizeof(double));
			cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].delta_GPU_d+cGPU::plan_FEM[DeviceId].Aux_1_h[1], DeviceId, cGPU::plan_FEM[cGPU::plan_FEM[DeviceId].Aux_1_h[1]].Result_d,
				cGPU::plan_FEM[DeviceId].Aux_1_h[1], sizeof(double));
			cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].delta_GPU_d+cGPU::plan_FEM[DeviceId].Aux_1_h[2], DeviceId, cGPU::plan_FEM[cGPU::plan_FEM[DeviceId].Aux_1_h[2]].Result_d,
				cGPU::plan_FEM[DeviceId].Aux_1_h[2], sizeof(double));

			// Barrier Synchronization
#pragma omp barrier

			// **************  Adding delta_new_V_d [0]+[1]+[2]+[3] terms:  **************

			/*switch(DeviceId) {

			case 0:  // Run device 0
				Launch_SUM_GPU_4(BlockSizeVector, ThreadsPerBlockXY, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::plan_FEM[DeviceId].delta_GPU_d, cGPU::plan_FEM[DeviceId].delta_new_V_d);
				break;

			case 1:  // Run device 1
				Launch_SUM_GPU_4(BlockSizeVector, ThreadsPerBlockXY, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::plan_FEM[DeviceId].delta_GPU_d, cGPU::plan_FEM[DeviceId].delta_new_V_d);
				break;

			case 2:  // Run device 2
				Launch_SUM_GPU_4(BlockSizeVector, ThreadsPerBlockXY, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::plan_FEM[DeviceId].delta_GPU_d, cGPU::plan_FEM[DeviceId].delta_new_V_d);
				break;

			case 3:  // Run device 3
				Launch_SUM_GPU_4(BlockSizeVector, ThreadsPerBlockXY, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::plan_FEM[DeviceId].delta_GPU_d, cGPU::plan_FEM[DeviceId].delta_new_V_d);
				break;

			}*/

			Launch_SUM_GPU_4(BlockSizeVector, ThreadsPerBlockXY, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::plan_FEM[DeviceId].delta_GPU_d, cGPU::plan_FEM[DeviceId].delta_new_V_d);

			// Barrier Synchronization
#pragma omp barrier

			int inc = 1;

			if((cont - int(cont/inc)*inc) == 0) {
				cudaMemcpy(plan_FEM[DeviceId].delta_new_V_h, cGPU::plan_FEM[DeviceId].delta_new_V_d, sizeof(double)*1, cudaMemcpyDeviceToHost);
				delta_new = plan_FEM[DeviceId].delta_new_V_h[0];
			}

			// Barrier Synchronization
#pragma omp barrier

			// **************  This Kernel adds up a Vector by a beta*Vector:  **************

			/*switch(DeviceId) {

			case 0:  // Run device 0
				Launch_ADD_V_V_D(BlockSizeVector, ThreadsPerBlockXY, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::plan_FEM[DeviceId].Vector_S_d, cGPU::plan_FEM[DeviceId].Vector_D_d, cGPU::plan_FEM[DeviceId].delta_new_V_d, cGPU::plan_FEM[DeviceId].delta_old_V_d); // =======================> {d} = {s} + beta{d}
				break;

			case 1:  // Run device 1
				Launch_ADD_V_V_D(BlockSizeVector, ThreadsPerBlockXY, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::plan_FEM[DeviceId].Vector_S_d, cGPU::plan_FEM[DeviceId].Vector_D_d, cGPU::plan_FEM[DeviceId].delta_new_V_d, cGPU::plan_FEM[DeviceId].delta_old_V_d); // =======================> {d} = {s} + beta{d}
				break;

			case 2:  // Run device 2
				Launch_ADD_V_V_D(BlockSizeVector, ThreadsPerBlockXY, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::plan_FEM[DeviceId].Vector_S_d, cGPU::plan_FEM[DeviceId].Vector_D_d, cGPU::plan_FEM[DeviceId].delta_new_V_d, cGPU::plan_FEM[DeviceId].delta_old_V_d); // =======================> {d} = {s} + beta{d}
				break;

			case 3:  // Run device 3
				Launch_ADD_V_V_D(BlockSizeVector, ThreadsPerBlockXY, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::plan_FEM[DeviceId].Vector_S_d, cGPU::plan_FEM[DeviceId].Vector_D_d, cGPU::plan_FEM[DeviceId].delta_new_V_d, cGPU::plan_FEM[DeviceId].delta_old_V_d); // =======================> {d} = {s} + beta{d}
				break;

			}*/

			Launch_ADD_V_V_D(BlockSizeVector, ThreadsPerBlockXY, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::plan_FEM[DeviceId].Vector_S_d,
				cGPU::plan_FEM[DeviceId].Vector_D_d, cGPU::plan_FEM[DeviceId].delta_new_V_d, cGPU::plan_FEM[DeviceId].delta_old_V_d); // =======================> {d} = {s} + beta{d}

			// Barrier Synchronization
#pragma omp barrier

			cont++;

			// ******************************************************************************************************************************************************

		}

		// *************************************************************************************************************************************************************************************************

		if(DeviceId == 0) {

			time = (clock()-time)/1000;
			printf("\n");
			printf("         Time Execution : %0.3f s \n", time);

			printf("\n");
			printf("         Iteration Number = %i  \n", cont);
			printf("         Error = %0.7f  \n", delta_new);

		}

		// **************  Assembling global displacement vector from each GPU:  **************

		// Barrier Synchronization
#pragma omp barrier

		/*switch(DeviceId) {

		case 0:  // Run device 0
			Launch_Copy_Local_Global(BlockSizeVector, ThreadsPerBlockXY, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, 0,                                                                                                     cGPU::plan_FEM[DeviceId].Vector_X_d, cGPU::plan_FEM[DeviceId].Vector_X_Global_d);
			break;

		case 1:  // Run device 1
			Launch_Copy_Local_Global(BlockSizeVector, ThreadsPerBlockXY, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::numdof*cGPU::plan_FEM[0].Localnumno,                                                             cGPU::plan_FEM[DeviceId].Vector_X_d, cGPU::plan_FEM[DeviceId].Vector_X_Global_d);
			break;

		case 2:  // Run device 2
			Launch_Copy_Local_Global(BlockSizeVector, ThreadsPerBlockXY, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::numdof*(cGPU::plan_FEM[0].Localnumno+cGPU::plan_FEM[1].Localnumno),                              cGPU::plan_FEM[DeviceId].Vector_X_d, cGPU::plan_FEM[DeviceId].Vector_X_Global_d);
			break;

		case 3:  // Run device 3
			Launch_Copy_Local_Global(BlockSizeVector, ThreadsPerBlockXY, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::numdof*(cGPU::plan_FEM[0].Localnumno+cGPU::plan_FEM[1].Localnumno+cGPU::plan_FEM[2].Localnumno), cGPU::plan_FEM[DeviceId].Vector_X_d, cGPU::plan_FEM[DeviceId].Vector_X_Global_d);
			break;

		}*/

		Launch_Copy_Local_Global(BlockSizeVector, ThreadsPerBlockXY, cGPU::numdof, cGPU::plan_FEM[DeviceId].Localnumno, cGPU::numdof*cGPU::plan_FEM[DeviceId].LocalnumnoAcum, cGPU::plan_FEM[DeviceId].Vector_X_d, cGPU::plan_FEM[DeviceId].Vector_X_Global_d);


		// **************   Assembling global vector Vector_X_Global_d from each GPU information:  **************

#pragma omp barrier
		/*if(DeviceId == 0)  // Sets device "0" as the current device
			cudaMemcpy(cGPU::plan_FEM[DeviceId].Vector_X_Global_d,                                                                                                 cGPU::plan_FEM[DeviceId].Vector_X_d, numdof*cGPU::plan_FEM[DeviceId].Localnumno*sizeof(double), cudaMemcpyDeviceToDevice);

		if(DeviceId == 1)  // Sets device "1" as the current device
			cudaMemcpy(cGPU::plan_FEM[DeviceId].Vector_X_Global_d+numdof*(cGPU::plan_FEM[0].Localnumno),                                                           cGPU::plan_FEM[DeviceId].Vector_X_d, numdof*cGPU::plan_FEM[DeviceId].Localnumno*sizeof(double), cudaMemcpyDeviceToDevice);

		if(DeviceId == 2)  // Sets device "2" as the current device
			cudaMemcpy(cGPU::plan_FEM[DeviceId].Vector_X_Global_d+numdof*(cGPU::plan_FEM[0].Localnumno+cGPU::plan_FEM[1].Localnumno),                              cGPU::plan_FEM[DeviceId].Vector_X_d, numdof*cGPU::plan_FEM[DeviceId].Localnumno*sizeof(double), cudaMemcpyDeviceToDevice);

		if(DeviceId == 3)  // Sets device "3" as the current device
			cudaMemcpy(cGPU::plan_FEM[DeviceId].Vector_X_Global_d+numdof*(cGPU::plan_FEM[0].Localnumno+cGPU::plan_FEM[1].Localnumno+cGPU::plan_FEM[2].Localnumno), cGPU::plan_FEM[DeviceId].Vector_X_d, numdof*cGPU::plan_FEM[DeviceId].Localnumno*sizeof(double), cudaMemcpyDeviceToDevice);*/

		cudaMemcpy(cGPU::plan_FEM[DeviceId].Vector_X_Global_d+numdof*cGPU::plan_FEM[DeviceId].LocalnumnoAcum, cGPU::plan_FEM[DeviceId].Vector_X_d,
			numdof*cGPU::plan_FEM[DeviceId].Localnumno*sizeof(double), cudaMemcpyDeviceToDevice);

#pragma omp barrier  // Synchronizes all threads in a team; all threads pause at the barrier, until all threads execute the barrier
		/*if(DeviceId == 0) {
			cudaMemcpyPeer(cGPU::plan_FEM[0].Vector_X_Global_d+cGPU::numdof*(cGPU::plan_FEM[DeviceId].Localnumno),                                                           0, cGPU::plan_FEM[1].Vector_X_d, 1, cGPU::numdof*cGPU::plan_FEM[1].Localnumno*sizeof(double));
			cudaMemcpyPeer(cGPU::plan_FEM[0].Vector_X_Global_d+cGPU::numdof*(cGPU::plan_FEM[DeviceId].Localnumno+cGPU::plan_FEM[1].Localnumno),                              0, cGPU::plan_FEM[2].Vector_X_d, 2, cGPU::numdof*cGPU::plan_FEM[2].Localnumno*sizeof(double));
			cudaMemcpyPeer(cGPU::plan_FEM[0].Vector_X_Global_d+cGPU::numdof*(cGPU::plan_FEM[DeviceId].Localnumno+cGPU::plan_FEM[1].Localnumno+cGPU::plan_FEM[2].Localnumno), 0, cGPU::plan_FEM[3].Vector_X_d, 3, cGPU::numdof*cGPU::plan_FEM[3].Localnumno*sizeof(double));
		}

#pragma omp barrier
		if(DeviceId == 1) {
			cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].Vector_X_Global_d,                                                                                                       1, cGPU::plan_FEM[0].Vector_X_d, 0, cGPU::numdof*cGPU::plan_FEM[0].Localnumno*sizeof(double));
			cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].Vector_X_Global_d+cGPU::numdof*(cGPU::plan_FEM[0].Localnumno+cGPU::plan_FEM[1].Localnumno),                              1, cGPU::plan_FEM[2].Vector_X_d, 2, cGPU::numdof*cGPU::plan_FEM[2].Localnumno*sizeof(double));
			cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].Vector_X_Global_d+cGPU::numdof*(cGPU::plan_FEM[0].Localnumno+cGPU::plan_FEM[1].Localnumno+cGPU::plan_FEM[2].Localnumno), 1, cGPU::plan_FEM[3].Vector_X_d, 3, cGPU::numdof*cGPU::plan_FEM[3].Localnumno*sizeof(double));
		}

#pragma omp barrier
		if(DeviceId == 2) {
			cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].Vector_X_Global_d,                                                                                                       2, cGPU::plan_FEM[0].Vector_X_d, 0, cGPU::numdof*cGPU::plan_FEM[0].Localnumno*sizeof(double));
			cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].Vector_X_Global_d+cGPU::numdof*(cGPU::plan_FEM[0].Localnumno),                                                           2, cGPU::plan_FEM[1].Vector_X_d, 1, cGPU::numdof*cGPU::plan_FEM[1].Localnumno*sizeof(double));
			cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].Vector_X_Global_d+cGPU::numdof*(cGPU::plan_FEM[0].Localnumno+cGPU::plan_FEM[1].Localnumno+cGPU::plan_FEM[2].Localnumno), 2, cGPU::plan_FEM[3].Vector_X_d, 3, cGPU::numdof*cGPU::plan_FEM[2].Localnumno*sizeof(double));
		}

#pragma omp barrier
		if(DeviceId == 3) {
			cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].Vector_X_Global_d,                                                                          3, cGPU::plan_FEM[0].Vector_X_d, 0, sizeof(double));
			cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].Vector_X_Global_d+cGPU::numdof*(cGPU::plan_FEM[0].Localnumno),                              3, cGPU::plan_FEM[1].Vector_X_d, 1, sizeof(double));
			cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].Vector_X_Global_d+cGPU::numdof*(cGPU::plan_FEM[0].Localnumno+cGPU::plan_FEM[1].Localnumno), 3, cGPU::plan_FEM[2].Vector_X_d, 2, sizeof(double));
		} */

		cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].Vector_X_Global_d+cGPU::numdof*cGPU::plan_FEM[DeviceId].LocalnumnoAcumVar[0], DeviceId,
			cGPU::plan_FEM[cGPU::plan_FEM[DeviceId].Aux_1_h[0]].Vector_X_d, cGPU::plan_FEM[DeviceId].Aux_1_h[0],
			cGPU::numdof*cGPU::plan_FEM[cGPU::plan_FEM[DeviceId].Aux_1_h[0]].Localnumno*sizeof(double));

		cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].Vector_X_Global_d+cGPU::numdof*cGPU::plan_FEM[DeviceId].LocalnumnoAcumVar[1], DeviceId,
			cGPU::plan_FEM[cGPU::plan_FEM[DeviceId].Aux_1_h[1]].Vector_X_d, cGPU::plan_FEM[DeviceId].Aux_1_h[1],
			cGPU::numdof*cGPU::plan_FEM[cGPU::plan_FEM[DeviceId].Aux_1_h[1]].Localnumno*sizeof(double));

		cudaMemcpyPeer(cGPU::plan_FEM[DeviceId].Vector_X_Global_d+cGPU::numdof*cGPU::plan_FEM[DeviceId].LocalnumnoAcumVar[2], DeviceId,
			cGPU::plan_FEM[cGPU::plan_FEM[DeviceId].Aux_1_h[2]].Vector_X_d, cGPU::plan_FEM[DeviceId].Aux_1_h[2],
			cGPU::numdof*cGPU::plan_FEM[cGPU::plan_FEM[DeviceId].Aux_1_h[2]].Localnumno*sizeof(double));

		// Barrier Synchronization
#pragma omp barrier

	}                   // It ends linear system solver /////////////////////////////////////////////////////////////////////////////////////

	//=============== x =============== x =============== x =============== x =============== x =============== x ===============

	// Write Displacement Field

	// -----------------------------------------------------------------

	// Device 0:
	cudaSetDevice(0);              // Sets device 0 as the current device
	cudaMemcpy(Vector_X_h, plan_FEM[0].Vector_X_Global_d, sizeof(double)*numdof*numno, cudaMemcpyDeviceToHost);

	// Write displacement field in output file GPUDisplacement.pos
	//WriteDisplacement(numno, Vector_X_h);

	// Write displacement field in vector VectorChr_X_h
	WriteDisplacementCouple(numno, Vector_X_h, VectorChr_X_h);

	//=============== x =============== x =============== x =============== x =============== x =============== x ===============

	return 1;

}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============

int cGPU::EvaluateDeltaStrainStateOnGPU(double *DeltaStrainChr_h)
{

	//=============== x =============== x =============== x =============== x =============== x =============== x ===============

	// Evaluate strain state on Multi GPUs:

	// To define one CPU thread for each GPU:

	int Id;

	omp_set_num_threads(deviceCount);

#pragma omp parallel
	{

		int DeviceId;
		double time;

		DeviceId = omp_get_thread_num();

		// ==================================================================================================================================
		// Sets device as the current device and allocate memory

		cudaSetDevice(DeviceId);

		cudaMemset(cGPU::plan_FEM[DeviceId].DeltaStrain_d, 0, cGPU::plan_FEM[DeviceId].Localnumel*cGPU::numstr*sizeof(double));

		// Barrier Synchronization
#pragma omp barrier

		if(DeviceId == 0) {
			printf("\n");
			printf("         Evaluating Delta Strain State \n");
			printf("         ========================================= ");
		}

		// Barrier Synchronization
#pragma omp barrier

		dim3 numBlocksStrain(int( sqrt(double(cGPU::plan_FEM[DeviceId].Localnumel)) /cGPU::plan_FEM[DeviceId].BlockSizeX)+1,
			                 int( sqrt(double(cGPU::plan_FEM[DeviceId].Localnumel)) /cGPU::plan_FEM[DeviceId].BlockSizeY)+1);

		dim3 threadsPerBlockStrain(cGPU::plan_FEM[DeviceId].BlockSizeX, cGPU::plan_FEM[DeviceId].BlockSizeY);

		// To start time count to evaluate strain state:
		time = clock();

		Launch_EvaluateDeltaStrainState(numBlocksStrain, threadsPerBlockStrain, cGPU::plan_FEM[DeviceId].Localnumel,
			cGPU::RowSize, cGPU::plan_FEM[DeviceId].CoordElem_d, cGPU::plan_FEM[DeviceId].Connect_d,
			cGPU::plan_FEM[DeviceId].Vector_X_Global_d, cGPU::plan_FEM[DeviceId].DeltaStrain_d);

		// Barrier Synchronization
#pragma omp barrier

		if(DeviceId == 0) {
			printf("\n");
			printf("         Evaluating Strain State Time: %0.3f s \n", (clock()-time)/1000);
		}

	// Barrier Synchronization
#pragma omp barrier

	}

	//=============== x =============== x =============== x =============== x =============== x =============== x ===============

	// Copying Strain State data from device (1/4) to host (global):

	for(Id=0; Id<deviceCount; Id++) {
		// Device Id:
		cudaSetDevice(Id);              // Sets device 0 as the current device
		for(i=0; i<numstr; i++)  // => Strain_h_Pt is a pointer to vector Strain_h
			cudaMemcpy(plan_FEM[Id].DeltaStrain_h_Pt + i*numel, plan_FEM[Id].DeltaStrain_d + i*plan_FEM[Id].Localnumel,
			plan_FEM[Id].Localnumel*sizeof(double), cudaMemcpyDeviceToHost);
	}

	/*// Device 0:
	cudaSetDevice(0);              // Sets device 0 as the current device
	for(i=0; i<numstr; i++)  // => Strain_h_Pt is a pointer to vector Strain_h
		cudaMemcpy(plan_FEM[0].DeltaStrain_h_Pt + i*numel, plan_FEM[0].DeltaStrain_d + i*plan_FEM[0].Localnumel, plan_FEM[0].Localnumel*sizeof(double), cudaMemcpyDeviceToHost);

	// Device 1:
	cudaSetDevice(1);              // Sets device 1 as the current device
	for(i=0; i<numstr; i++)
		cudaMemcpy(plan_FEM[1].DeltaStrain_h_Pt + i*numel, plan_FEM[1].DeltaStrain_d + i*plan_FEM[1].Localnumel, plan_FEM[1].Localnumel*sizeof(double), cudaMemcpyDeviceToHost);

	// Device 2:
	cudaSetDevice(2);              // Sets device 2 as the current device
	for(i=0; i<numstr; i++)
		cudaMemcpy(plan_FEM[2].DeltaStrain_h_Pt + i*numel, plan_FEM[2].DeltaStrain_d + i*plan_FEM[2].Localnumel, plan_FEM[2].Localnumel*sizeof(double), cudaMemcpyDeviceToHost);

	// Device 3:
	cudaSetDevice(3);              // Sets device 3 as the current device
	for(i=0; i<numstr; i++)
		cudaMemcpy(plan_FEM[3].DeltaStrain_h_Pt + i*numel, plan_FEM[3].DeltaStrain_d + i*plan_FEM[3].Localnumel, plan_FEM[3].Localnumel*sizeof(double), cudaMemcpyDeviceToHost);
*/


	//=============== x =============== x =============== x =============== x =============== x =============== x ===============

	// Write strain state in output file GPUStrainState.pos
	//WriteStrainState(numel, DeltaStrain_h);

	// Write strain state in vector DeltaStrainChr_h
	CopyDeltaStrainFromGlobalToLocal(numel, DeltaStrain_h, DeltaStrainChr_h);

	return 1;

}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============

int cGPU::EvaluateDeltaStressStateOnGPU(double *DeltaStressChr_h)
{

	//=============== x =============== x =============== x =============== x =============== x =============== x ===============

	// Evaluate stress state on Multi GPUs:

	// To define one CPU thread for each GPU:

	int Id;

	omp_set_num_threads(deviceCount);

#pragma omp parallel
	{

		int DeviceId = omp_get_thread_num();

		double time;

		// ==================================================================================================================================
		// Sets device as the current device and allocate memory

		cudaSetDevice(DeviceId);

		cudaMemset(cGPU::plan_FEM[DeviceId].DeltaStress_d, 0, cGPU::plan_FEM[DeviceId].Localnumel*cGPU::numstr*sizeof(double));

		// ==================================================================================================================================
		// Evaluate Strain state

		// Barrier Synchronization
#pragma omp barrier

		if(DeviceId == 0) {
			printf("\n");
			printf("         Evaluating Stress State \n");
			printf("         ========================================= ");
		}

		// =================================================================

		dim3 numBlocksStress(int( sqrt(double(cGPU::plan_FEM[DeviceId].Localnumel)) /cGPU::plan_FEM[DeviceId].BlockSizeX)+1,
			                 int( sqrt(double(cGPU::plan_FEM[DeviceId].Localnumel)) /cGPU::plan_FEM[DeviceId].BlockSizeY)+1);

		dim3 threadsPerBlockStress(cGPU::plan_FEM[DeviceId].BlockSizeX, cGPU::plan_FEM[DeviceId].BlockSizeY);

		// To start time count to evaluate strain state:
		time = clock();

		Launch_EvaluateDeltaStressState(numBlocksStress, threadsPerBlockStress, cGPU::plan_FEM[DeviceId].Localnumel, cGPU::plan_FEM[DeviceId].Material_Data_d,
			                            cGPU::plan_FEM[DeviceId].DeltaStrain_d, cGPU::plan_FEM[DeviceId].DeltaStress_d);

		// Barrier Synchronization
#pragma omp barrier

		if(DeviceId == 0) {
			printf("\n");
			printf("         Evaluating Stress State Time: %0.3f s \n", (clock()-time)/1000);
		}

		// Barrier Synchronization
#pragma omp barrier

	}

	//=============== x =============== x =============== x =============== x =============== x =============== x ===============

	// Copying Stress State data from device (1/4) to host (global):

	for(Id=0; Id<deviceCount; Id++) {
		// Device Id:
		cudaSetDevice(Id);              // Sets device 0 as the current device
		for(i=0; i<numstr; i++)  // => Strain_h_Pt is a pointer to vector Strain_h
			cudaMemcpy(plan_FEM[Id].DeltaStress_h_Pt + i*numel, plan_FEM[Id].DeltaStress_d + i*plan_FEM[Id].Localnumel,
			plan_FEM[Id].Localnumel*sizeof(double), cudaMemcpyDeviceToHost);
	}

	// Device 0:
	/*cudaSetDevice(0);              // Sets device 0 as the current device
	for(i=0; i<numstr*8; i++)
		cudaMemcpy(plan_FEM[0].DeltaStress_h_Pt + i*numel, plan_FEM[0].DeltaStress_d + i*plan_FEM[0].Localnumel, plan_FEM[0].Localnumel*sizeof(double), cudaMemcpyDeviceToHost);

	// Device 1:
	cudaSetDevice(1);              // Sets device 1 as the current device
	for(i=0; i<numstr*8; i++)
		cudaMemcpy(plan_FEM[1].DeltaStress_h_Pt + i*numel, plan_FEM[1].DeltaStress_d + i*plan_FEM[1].Localnumel, plan_FEM[1].Localnumel*sizeof(double), cudaMemcpyDeviceToHost);

	// Device 2:
	cudaSetDevice(2);              // Sets device 2 as the current device
	for(i=0; i<numstr*8; i++)
		cudaMemcpy(plan_FEM[2].DeltaStress_h_Pt + i*numel, plan_FEM[2].DeltaStress_d + i*plan_FEM[2].Localnumel, plan_FEM[2].Localnumel*sizeof(double), cudaMemcpyDeviceToHost);

	// Device 3:
	cudaSetDevice(3);              // Sets device 3 as the current device
	for(i=0; i<numstr*8; i++)
		cudaMemcpy(plan_FEM[3].DeltaStress_h_Pt + i*numel, plan_FEM[3].DeltaStress_d + i*plan_FEM[3].Localnumel, plan_FEM[3].Localnumel*sizeof(double), cudaMemcpyDeviceToHost);*/

	//=============== x =============== x =============== x =============== x =============== x =============== x ===============

	// Write stress state in output file GPUStressState.pos
	//WriteStressState(numel, Stress_h);

	// Write stress state in vector StressChr_h
	CopyDeltaStrainFromGlobalToLocal(numel, DeltaStress_h, DeltaStressChr_h);  // Same function used to copy the strain state

	return 1;

}

//========================================================================================================

int cGPU::ReleaseMemoryChronos()
{
	// Releasing CPU memory:
	free(GPUData);

	free(Vector_I_h); free(BC_h); free(Connect_h); free(Material_Id_h);
	free(Vector_B_h); free(Vector_X_h);
	free(Material_Data_h); free(Material_Param_h); free(CoordElem_h); free(Strain_h); free(Stress_h);
	free(X_h); free(Y_h); free(Z_h);
	free(MatPropQ1_h); free(MatPropQ2_h); free(MatPropQ3_h); free(MatPropQ4_h); free(MatPropQ5_h); free(MatPropQ6_h); free(MatPropQ7_h); free(MatPropQ8_h);
	free(CoordQ1_h); free(CoordQ2_h); free(CoordQ3_h); free(CoordQ4_h); free(CoordQ5_h); free(CoordQ6_h); free(CoordQ7_h); free(CoordQ8_h);
	free(iiPosition_h);

	for(i=0; i<deviceCount; i++) {

		free(plan_FEM[i].delta_new_V_h);

		// Releasing GPU memory:

		cudaFree(plan_FEM[i].delta_GPU_d); cudaFree(plan_FEM[i].Result_d); cudaFree(plan_FEM[i].Vector_D_Global_d); cudaFree(plan_FEM[i].Vector_D_d);
		cudaFree(plan_FEM[i].Vector_I_d); cudaFree(plan_FEM[i].Connect_d);
		cudaFree(plan_FEM[i].Matrix_A_d); cudaFree(plan_FEM[i].Vector_X_d); cudaFree(plan_FEM[i].Matrix_M_d); cudaFree(plan_FEM[i].Vector_B_d); cudaFree(plan_FEM[i].Result_d);
		cudaFree(plan_FEM[i].Vector_R_d); cudaFree(plan_FEM[i].Vector_D_d); cudaFree(plan_FEM[i].Vector_D_Global_d); cudaFree(plan_FEM[i].Vector_Q_d); cudaFree(plan_FEM[i].Vector_S_d);
		cudaFree(plan_FEM[i].delta_new_d); cudaFree(plan_FEM[i].delta_aux_d); cudaFree(plan_FEM[i].delta_new_V_d); cudaFree(plan_FEM[i].delta_aux_V_d); cudaFree(plan_FEM[i].delta_old_V_d);
		cudaFree(plan_FEM[i].delta_GPU_d); 	cudaFree(plan_FEM[i].CoordElem_d); cudaFree(plan_FEM[0].Vector_X_Global_d);
		cudaFree(plan_FEM[i].DeltaStrain_d);

		free(plan_FEM[i].Matrix_K_h_Pt); //free(plan_FEM[i].Vector_B_h_Pt); free(plan_FEM[i].Vector_X_h_Pt);
		free(plan_FEM[i].Matrix_M_h_Pt); free(plan_FEM[i].Delta_New_GPU_h_Pt); free(plan_FEM[i].Delta_Aux_GPU_h_Pt);
		free(plan_FEM[i].Vector_D_h_A_Pt); free(plan_FEM[i].Vector_D_h_P_Pt);
		//free(plan_FEM[i].Vector_I_h_Pt);
		free(plan_FEM[i].eventId_1_h_Pt); free(plan_FEM[i].eventId_2_h_Pt);
		//free(plan_FEM[i].CoordElem_h_Pt); free(plan_FEM[i].Material_Data_h_Pt); free(plan_FEM[i].DeltaStress_h_Pt);
		//free(plan_FEM[i].MatPropQ1_h_Pt); free(plan_FEM[i].MatPropQ2_h_Pt); free(plan_FEM[i].MatPropQ3_h_Pt); free(plan_FEM[i].MatPropQ4_h_Pt); free(plan_FEM[i].MatPropQ5_h_Pt); free(plan_FEM[i].MatPropQ6_h_Pt); free(plan_FEM[i].MatPropQ7_h_Pt); free(plan_FEM[i].MatPropQ8_h_Pt);
		//free(plan_FEM[i].CoordQ1_h_Pt); free(plan_FEM[i].CoordQ2_h_Pt); free(plan_FEM[i].CoordQ3_h_Pt); free(plan_FEM[i].CoordQ4_h_Pt); free(plan_FEM[i].CoordQ5_h_Pt); free(plan_FEM[i].CoordQ6_h_Pt); free(plan_FEM[i].CoordQ7_h_Pt); free(plan_FEM[i].CoordQ8_h_Pt);
		//free(plan_FEM[i].iiPosition_h_Pt); free(plan_FEM[i].Connect_h_Pt);

		cudaFree(plan_FEM[i].Vector_B_d); cudaFree(plan_FEM[i].Vector_R_d); cudaFree(plan_FEM[i].Matrix_M_d); cudaFree(plan_FEM[i].delta_new_d); cudaFree(plan_FEM[i].delta_new_V_d); cudaFree(plan_FEM[i].Matrix_A_d); cudaFree(plan_FEM[i].Vector_Q_d); cudaFree(plan_FEM[i].delta_aux_d); cudaFree(plan_FEM[i].delta_aux_V_d);
		cudaFree(plan_FEM[i].Vector_S_d); cudaFree(plan_FEM[i].delta_old_V_d);
		cudaFree(plan_FEM[i].delta_new_V_h); cudaFree(plan_FEM[i].Matrix_A_h);
		cudaFree(plan_FEM[i].Vector_I_d); cudaFree(plan_FEM[i].iiPosition_d); cudaFree(plan_FEM[i].Connect_d);

		cudaFree(plan_FEM[i].Vector_D_d); cudaFree(plan_FEM[i].delta_GPU_d); cudaFree(plan_FEM[i].Result_d);cudaFree(plan_FEM[i].Vector_D_Global_d); cudaFree(plan_FEM[i].Vector_X_d); cudaFree(plan_FEM[i].Vector_X_Global_d);
		cudaFree(plan_FEM[i].CoordQ1_d); cudaFree(plan_FEM[i].CoordQ2_d); cudaFree(plan_FEM[i].CoordQ3_d); cudaFree(plan_FEM[i].CoordQ4_d); cudaFree(plan_FEM[i].CoordQ5_d); cudaFree(plan_FEM[i].CoordQ6_d); cudaFree(plan_FEM[i].CoordQ7_d); cudaFree(plan_FEM[i].CoordQ8_d);
		cudaFree(plan_FEM[i].MatPropQ1_d); cudaFree(plan_FEM[i].MatPropQ2_d); cudaFree(plan_FEM[i].MatPropQ3_d); cudaFree(plan_FEM[i].MatPropQ4_d); cudaFree(plan_FEM[i].MatPropQ5_d); cudaFree(plan_FEM[i].MatPropQ6_d); cudaFree(plan_FEM[i].MatPropQ7_d); cudaFree(plan_FEM[i].MatPropQ8_d);
		cudaFree(plan_FEM[i].CoordElem_d); cudaFree(plan_FEM[i].Material_Data_d);

	}

	return 1;

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

	imaxhalos=imax+2;  // number of nodes in X direction imax = nx+1 (nx, ny and nz = number of cells/elements)
	jmaxhalos=jmax+2;  // number of nodes in X direction imay = ny+1
	kmaxhalos=kmax+2;  // number of nodes in X direction imaz = nz+1

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

				//if(BC_h[cont]==0) {            // if BC_h = 1 Displacement is constrained at the node

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

				//}

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

    inFile = fopen("LoadFile.inc", "r");

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

	}

	for(i=0; i<numno; i++) {

		if(BC_h[i+0*numno]==1)            // BC in x direction
			Vector_B_h[3*i  ] = 0.;
		if(BC_h[i+1*numno]==1)            // BC in y direction
			Vector_B_h[3*i+1] = 0.;
		if(BC_h[i+2*numno]==1)            // BC in z direction
			Vector_B_h[3*i+2] = 0.;

		//printf("%f %f %f \n", Vector_B_h[3*i  ], Vector_B_h[3*i+1], Vector_B_h[3*i+2]);

	}

	fclose(inFile);

}

//==============================================================================
// Reading nodal concentrate load from external file:

void ReadConcLoadCouple(int numno, double *Vector_B_h, int *BC_h, double *NodeForceChr)
{
	int i;

	for(i=0; i<numno; i++) {

		Vector_B_h[3*i  ] = NodeForceChr[3*i  ];
		Vector_B_h[3*i+1] = NodeForceChr[3*i+1];
		Vector_B_h[3*i+2] = NodeForceChr[3*i+2];

		if(BC_h[i+0*numno]==1)            // BC in x direction
			Vector_B_h[3*i  ] = 0.;
		if(BC_h[i+1*numno]==1)            // BC in y direction
			Vector_B_h[3*i+1] = 0.;
		if(BC_h[i+2*numno]==1)            // BC in z direction
			Vector_B_h[3*i+2] = 0.;

	}

}

//==============================================================================
void GPU_Information(int &deviceCount, double *GPUData)
{
	int dev;
	size_t free_byte, total_byte;
	int driverVersion = 0, runtimeVersion = 0;
	double auxN;

	// Printing on output file:  ===============================================
	// Create a new file
	ofstream outfile;

	IF_DEBUGGING {
		outfile.open("GPUsInformation.out",ios::app);
		outfile << "======================================================== " << endl;
		outfile << endl;
		outfile << "Number of devices " << endl;
		outfile << deviceCount << endl;
		outfile << endl;
	}

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

		IF_DEBUGGING {
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
	}

	IF_DEBUGGING
		outfile.close();

}

//==============================================================================
void InitialInformation()
{
	printf("\n\n");
	printf("\t --------------------------------------------------   \n");
	printf("\t  PONTIFICAL CATHOLIC UNIVERSITY OF RIO DE JANEIRO    \n");
	printf("\t **       DEPARTMENT OF CIVIL ENGINEERING        **   \n");
	printf("\t GTEP - Group of Technology in Petroleum Engineering  \n");
	printf("\t               F A S T F E M  -  3  D                 \n");
	printf("\t            FINITE ELEMENT METHOD ON GPU              \n");
	printf("\t                           Version: 1-000 - Jul/11    \n");
	printf("\t -------------------------------------------------- \n\n");

}

//==============================================================================
int GPU_ReadSize(int &numno, int &numel, int &nummat, int &numprop, int &numdof, int &numnoel, int &numstr, int &numbc, int &numnl,
int &imax, int &jmax, int &kmax, char GPU_arqaux[80])
{
	char label[80];
	std::string s;
	FILE *inFile;
	int c;

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
void cGPU::GPU_ReadInPutFile(int numno, int numel, int nummat, int numprop, int numdof, int numnoel, int numstr, int numbc, int numnl, char GPU_arqaux[80],
double *X_h, double *Y_h, double *Z_h, int *BC_h, double *Material_Param_h, int *Connect_h, int *Material_Id_h)
{
	char label[80];
	std::string s;
	FILE *inFile;
	int c;
	int i, number, matId, aux, el1, el2, el3, el4, el5, el6, el7, el8, resId;

	int _numno, _numel, _nummat,  _numbc;
	int dx, dy, dz, rx, ry, rz;
	double prop1, prop2, prop3, prop4;
	bool skip;

	inFile  = fopen(GPU_arqaux, "r");

	if((inFile == NULL)) {
		printf("\n\n\n\t *** Error on reading Input file !!! *** \n");
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
				printf("\n Error on reading number of nodes !!!\n\n" );;
			}


			for(i=0; i<numno; i++) {

				if(fscanf(inFile, "%d %lf %lf %lf", &number, &X_h[i], &Y_h[i], &Z_h[i]) != 4){
					printf("\nError on reading nodes coordinates, node Id = %d", number);
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
			}


			for(i=0; i<numbc; i++) {

				if(fscanf(inFile, "%d %d %d %d %d %d %d", &number, &dx, &dy, &dz, &rx, &ry, &rz) != 7){
					printf("\nError on reading nodal boundary conditions, node Id = %d", number);
				}

				if(dx == 1)
					BC_h[number-1 + 0*numno] = 1;
				if(dy == 1)
					BC_h[number-1 + 1*numno] = 1;
				if(dz == 1)
					BC_h[number-1 + 2*numno] = 1;

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
			}

			for(i=0; i<nummat; i++) {

				if(fscanf(inFile, "%d %lf %lf", &number, &prop1, &prop2) != 3){
					printf("\nError on reading material property, material Id = %d", number);
				}

				Material_Param_h[i       ] = prop1;
				Material_Param_h[i+nummat] = prop2;

			}

			break;

		}

	}

	// ========= Reading material densinty =========

	while(1) {

		while((c = fgetc(inFile))!='%') {
			if(c == EOF) {
				break;
				printf("\nError on reading property: MATERIAL.INITIALSTRESS");
			}
		}

		fscanf(inFile, "%s", label);          // scan label string

		s = label;

		if(s == "MATERIAL.INITIALSTRESS" || s == "MATERIAL.DENSITY") {

			if(fscanf(inFile, "%d", &_nummat) != 1) {
				printf("\n Error on reading number of nodes !!!\n\n" );
			}

			for(i=0; i<nummat; i++) {

				if(fscanf(inFile, "%d %lf %lf %lf %d", &number, &prop1, &prop2, &prop3, &resId) != 5){
					printf("\nError on reading material property, material Id = %d", number);
				}

				Material_Density_h[i+0*nummat] = prop1;
				Material_Density_h[i+1*nummat] = prop2;
				Material_Density_h[i+2*nummat] = prop3;

				Reservoir_Id_h[i] = resId;

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
			}

			for(i=0; i<numel; i++) {

				if(fscanf(inFile, "%d %d %d %d %d %d %d %d %d %d %d", &number, &matId, &aux, &el1, &el2, &el3, &el4, &el5, &el6, &el7, &el8) != 11){
					printf("\nError on reading element connectivity, element Id = %d", number);
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

	// ========= Reading material densinty =========

	while(1) {

		while((c = fgetc(inFile))!='%') {
			if(c == EOF) break;
		}

		fscanf(inFile, "%s", label);          // scan label string

		s = label;

		if(s == "ADJROCK") {

			if(fscanf(inFile, "%d %d %d %d %d %d", &nsi1, &nsi2, &nsj1, &nsj2, &nov, &nun) != 6){
				printf("\nError on reading material property, material Id = %d", number);
			}

			break;

		}

	}

	// =============================================

	fclose(inFile);

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
							CoordQ1_h[cont+ 0*numno] = X_h[pos_ijk-(imax*jmax)-imax-1]-X_h[pos_ijk-(imax*jmax)-imax-1];
							CoordQ1_h[cont+ 1*numno] = Y_h[pos_ijk-(imax*jmax)-imax-1]-Y_h[pos_ijk-(imax*jmax)-imax-1];
							CoordQ1_h[cont+ 2*numno] = Z_h[pos_ijk-(imax*jmax)-imax-1]-Z_h[pos_ijk-(imax*jmax)-imax-1];

							// Node 2:
							CoordQ1_h[cont+ 3*numno] = X_h[pos_ijk-(imax*jmax)-imax]-X_h[pos_ijk-(imax*jmax)-imax-1];
							CoordQ1_h[cont+ 4*numno] = Y_h[pos_ijk-(imax*jmax)-imax]-Y_h[pos_ijk-(imax*jmax)-imax-1];
							CoordQ1_h[cont+ 5*numno] = Z_h[pos_ijk-(imax*jmax)-imax]-Z_h[pos_ijk-(imax*jmax)-imax-1];

							// Node 3:
							CoordQ1_h[cont+ 6*numno] = X_h[pos_ijk-(imax*jmax)]-X_h[pos_ijk-(imax*jmax)-imax-1];
							CoordQ1_h[cont+ 7*numno] = Y_h[pos_ijk-(imax*jmax)]-Y_h[pos_ijk-(imax*jmax)-imax-1];
							CoordQ1_h[cont+ 8*numno] = Z_h[pos_ijk-(imax*jmax)]-Z_h[pos_ijk-(imax*jmax)-imax-1];

							// Node 4:
							CoordQ1_h[cont+ 9*numno] = X_h[pos_ijk-(imax*jmax)-1]-X_h[pos_ijk-(imax*jmax)-imax-1];
							CoordQ1_h[cont+10*numno] = Y_h[pos_ijk-(imax*jmax)-1]-Y_h[pos_ijk-(imax*jmax)-imax-1];
							CoordQ1_h[cont+11*numno] = Z_h[pos_ijk-(imax*jmax)-1]-Z_h[pos_ijk-(imax*jmax)-imax-1];

							// Node 5:
							CoordQ1_h[cont+12*numno] = X_h[pos_ijk-imax-1]-X_h[pos_ijk-(imax*jmax)-imax-1];
							CoordQ1_h[cont+13*numno] = Y_h[pos_ijk-imax-1]-Y_h[pos_ijk-(imax*jmax)-imax-1];
							CoordQ1_h[cont+14*numno] = Z_h[pos_ijk-imax-1]-Z_h[pos_ijk-(imax*jmax)-imax-1];

							// Node 6:
							CoordQ1_h[cont+15*numno] = X_h[pos_ijk-imax]-X_h[pos_ijk-(imax*jmax)-imax-1];
							CoordQ1_h[cont+16*numno] = Y_h[pos_ijk-imax]-Y_h[pos_ijk-(imax*jmax)-imax-1];
							CoordQ1_h[cont+17*numno] = Z_h[pos_ijk-imax]-Z_h[pos_ijk-(imax*jmax)-imax-1];

							// Node 7:
							CoordQ1_h[cont+18*numno] = X_h[pos_ijk]-X_h[pos_ijk-(imax*jmax)-imax-1];
							CoordQ1_h[cont+19*numno] = Y_h[pos_ijk]-Y_h[pos_ijk-(imax*jmax)-imax-1];
							CoordQ1_h[cont+20*numno] = Z_h[pos_ijk]-Z_h[pos_ijk-(imax*jmax)-imax-1];

							// Node 8:
							CoordQ1_h[cont+21*numno] = X_h[pos_ijk-1]-X_h[pos_ijk-(imax*jmax)-imax-1];
							CoordQ1_h[cont+22*numno] = Y_h[pos_ijk-1]-Y_h[pos_ijk-(imax*jmax)-imax-1];
							CoordQ1_h[cont+23*numno] = Z_h[pos_ijk-1]-Z_h[pos_ijk-(imax*jmax)-imax-1];

						}

					}

				}

				// ****************************** Evaluating CoordQ2_h ******************************

				if(i != imax-1) {

					if(j != 0) {

						if(k != 0) {

							// Node 1:
							CoordQ2_h[cont+ 0*numno] = X_h[pos_ijk-(imax*jmax)-imax]-X_h[pos_ijk-(imax*jmax)-imax];
							CoordQ2_h[cont+ 1*numno] = Y_h[pos_ijk-(imax*jmax)-imax]-Y_h[pos_ijk-(imax*jmax)-imax];
							CoordQ2_h[cont+ 2*numno] = Z_h[pos_ijk-(imax*jmax)-imax]-Z_h[pos_ijk-(imax*jmax)-imax];

							// Node 2:
							CoordQ2_h[cont+ 3*numno] = X_h[pos_ijk-(imax*jmax)-imax+1]-X_h[pos_ijk-(imax*jmax)-imax];
							CoordQ2_h[cont+ 4*numno] = Y_h[pos_ijk-(imax*jmax)-imax+1]-Y_h[pos_ijk-(imax*jmax)-imax];
							CoordQ2_h[cont+ 5*numno] = Z_h[pos_ijk-(imax*jmax)-imax+1]-Z_h[pos_ijk-(imax*jmax)-imax];

							// Node 3:
							CoordQ2_h[cont+ 6*numno] = X_h[pos_ijk-(imax*jmax)+1]-X_h[pos_ijk-(imax*jmax)-imax];
							CoordQ2_h[cont+ 7*numno] = Y_h[pos_ijk-(imax*jmax)+1]-Y_h[pos_ijk-(imax*jmax)-imax];
							CoordQ2_h[cont+ 8*numno] = Z_h[pos_ijk-(imax*jmax)+1]-Z_h[pos_ijk-(imax*jmax)-imax];

							// Node 4:
							CoordQ2_h[cont+ 9*numno] = X_h[pos_ijk-(imax*jmax)]-X_h[pos_ijk-(imax*jmax)-imax];
							CoordQ2_h[cont+10*numno] = Y_h[pos_ijk-(imax*jmax)]-Y_h[pos_ijk-(imax*jmax)-imax];
							CoordQ2_h[cont+11*numno] = Z_h[pos_ijk-(imax*jmax)]-Z_h[pos_ijk-(imax*jmax)-imax];

							// Node 5:
							CoordQ2_h[cont+12*numno] = X_h[pos_ijk-imax]-X_h[pos_ijk-(imax*jmax)-imax];
							CoordQ2_h[cont+13*numno] = Y_h[pos_ijk-imax]-Y_h[pos_ijk-(imax*jmax)-imax];
							CoordQ2_h[cont+14*numno] = Z_h[pos_ijk-imax]-Z_h[pos_ijk-(imax*jmax)-imax];

							// Node 6:
							CoordQ2_h[cont+15*numno] = X_h[pos_ijk-imax+1]-X_h[pos_ijk-(imax*jmax)-imax];
							CoordQ2_h[cont+16*numno] = Y_h[pos_ijk-imax+1]-Y_h[pos_ijk-(imax*jmax)-imax];
							CoordQ2_h[cont+17*numno] = Z_h[pos_ijk-imax+1]-Z_h[pos_ijk-(imax*jmax)-imax];

							// Node 7:
							CoordQ2_h[cont+18*numno] = X_h[pos_ijk+1]-X_h[pos_ijk-(imax*jmax)-imax];
							CoordQ2_h[cont+19*numno] = Y_h[pos_ijk+1]-Y_h[pos_ijk-(imax*jmax)-imax];
							CoordQ2_h[cont+20*numno] = Z_h[pos_ijk+1]-Z_h[pos_ijk-(imax*jmax)-imax];

							// Node 8:
							CoordQ2_h[cont+21*numno] = X_h[pos_ijk]-X_h[pos_ijk-(imax*jmax)-imax];
							CoordQ2_h[cont+22*numno] = Y_h[pos_ijk]-Y_h[pos_ijk-(imax*jmax)-imax];
							CoordQ2_h[cont+23*numno] = Z_h[pos_ijk]-Z_h[pos_ijk-(imax*jmax)-imax];

						}

					}

				}

				// ****************************** Evaluating CoordQ3_h ******************************

				if(i != 0) {

					if(j != jmax-1) {

						if(k != 0) {

							// Node 1:
							CoordQ3_h[cont+ 0*numno] = X_h[pos_ijk-(imax*jmax)-1]-X_h[pos_ijk-(imax*jmax)-1];
							CoordQ3_h[cont+ 1*numno] = Y_h[pos_ijk-(imax*jmax)-1]-Y_h[pos_ijk-(imax*jmax)-1];
							CoordQ3_h[cont+ 2*numno] = Z_h[pos_ijk-(imax*jmax)-1]-Z_h[pos_ijk-(imax*jmax)-1];

							// Node 2:
							CoordQ3_h[cont+ 3*numno] = X_h[pos_ijk-(imax*jmax)]-X_h[pos_ijk-(imax*jmax)-1];
							CoordQ3_h[cont+ 4*numno] = Y_h[pos_ijk-(imax*jmax)]-Y_h[pos_ijk-(imax*jmax)-1];
							CoordQ3_h[cont+ 5*numno] = Z_h[pos_ijk-(imax*jmax)]-Z_h[pos_ijk-(imax*jmax)-1];

							// Node 3:
							CoordQ3_h[cont+ 6*numno] = X_h[pos_ijk-(imax*jmax)+imax]-X_h[pos_ijk-(imax*jmax)-1];
							CoordQ3_h[cont+ 7*numno] = Y_h[pos_ijk-(imax*jmax)+imax]-Y_h[pos_ijk-(imax*jmax)-1];
							CoordQ3_h[cont+ 8*numno] = Z_h[pos_ijk-(imax*jmax)+imax]-Z_h[pos_ijk-(imax*jmax)-1];

							// Node 4:
							CoordQ3_h[cont+ 9*numno] = X_h[pos_ijk-(imax*jmax)+imax-1]-X_h[pos_ijk-(imax*jmax)-1];
							CoordQ3_h[cont+10*numno] = Y_h[pos_ijk-(imax*jmax)+imax-1]-Y_h[pos_ijk-(imax*jmax)-1];
							CoordQ3_h[cont+11*numno] = Z_h[pos_ijk-(imax*jmax)+imax-1]-Z_h[pos_ijk-(imax*jmax)-1];

							// Node 5:
							CoordQ3_h[cont+12*numno] = X_h[pos_ijk-1]-X_h[pos_ijk-(imax*jmax)-1];
							CoordQ3_h[cont+13*numno] = Y_h[pos_ijk-1]-Y_h[pos_ijk-(imax*jmax)-1];
							CoordQ3_h[cont+14*numno] = Z_h[pos_ijk-1]-Z_h[pos_ijk-(imax*jmax)-1];

							// Node 6:
							CoordQ3_h[cont+15*numno] = X_h[pos_ijk]-X_h[pos_ijk-(imax*jmax)-1];
							CoordQ3_h[cont+16*numno] = Y_h[pos_ijk]-Y_h[pos_ijk-(imax*jmax)-1];
							CoordQ3_h[cont+17*numno] = Z_h[pos_ijk]-Z_h[pos_ijk-(imax*jmax)-1];

							// Node 7:
							CoordQ3_h[cont+18*numno] = X_h[pos_ijk+imax]-X_h[pos_ijk-(imax*jmax)-1];
							CoordQ3_h[cont+19*numno] = Y_h[pos_ijk+imax]-Y_h[pos_ijk-(imax*jmax)-1];
							CoordQ3_h[cont+20*numno] = Z_h[pos_ijk+imax]-Z_h[pos_ijk-(imax*jmax)-1];

							// Node 8:
							CoordQ3_h[cont+21*numno] = X_h[pos_ijk+imax-1]-X_h[pos_ijk-(imax*jmax)-1];
							CoordQ3_h[cont+22*numno] = Y_h[pos_ijk+imax-1]-Y_h[pos_ijk-(imax*jmax)-1];
							CoordQ3_h[cont+23*numno] = Z_h[pos_ijk+imax-1]-Z_h[pos_ijk-(imax*jmax)-1];

						}

					}

				}

				// ****************************** Evaluating CoordQ4_h ******************************

				if(i != imax-1) {

					if(j != jmax-1) {

						if(k != 0) {

							// Node 1:
							CoordQ4_h[cont+ 0*numno] = X_h[pos_ijk-(imax*jmax)]-X_h[pos_ijk-(imax*jmax)];
							CoordQ4_h[cont+ 1*numno] = Y_h[pos_ijk-(imax*jmax)]-Y_h[pos_ijk-(imax*jmax)];
							CoordQ4_h[cont+ 2*numno] = Z_h[pos_ijk-(imax*jmax)]-Z_h[pos_ijk-(imax*jmax)];

							// Node 2:
							CoordQ4_h[cont+ 3*numno] = X_h[pos_ijk-(imax*jmax)+1]-X_h[pos_ijk-(imax*jmax)];
							CoordQ4_h[cont+ 4*numno] = Y_h[pos_ijk-(imax*jmax)+1]-Y_h[pos_ijk-(imax*jmax)];
							CoordQ4_h[cont+ 5*numno] = Z_h[pos_ijk-(imax*jmax)+1]-Z_h[pos_ijk-(imax*jmax)];

							// Node 3:
							CoordQ4_h[cont+ 6*numno] = X_h[pos_ijk-(imax*jmax)+imax+1]-X_h[pos_ijk-(imax*jmax)];
							CoordQ4_h[cont+ 7*numno] = Y_h[pos_ijk-(imax*jmax)+imax+1]-Y_h[pos_ijk-(imax*jmax)];
							CoordQ4_h[cont+ 8*numno] = Z_h[pos_ijk-(imax*jmax)+imax+1]-Z_h[pos_ijk-(imax*jmax)];

							// Node 4:
							CoordQ4_h[cont+ 9*numno] = X_h[pos_ijk-(imax*jmax)+imax]-X_h[pos_ijk-(imax*jmax)];
							CoordQ4_h[cont+10*numno] = Y_h[pos_ijk-(imax*jmax)+imax]-Y_h[pos_ijk-(imax*jmax)];
							CoordQ4_h[cont+11*numno] = Z_h[pos_ijk-(imax*jmax)+imax]-Z_h[pos_ijk-(imax*jmax)];

							// Node 5:
							CoordQ4_h[cont+12*numno] = X_h[pos_ijk]-X_h[pos_ijk-(imax*jmax)];
							CoordQ4_h[cont+13*numno] = Y_h[pos_ijk]-Y_h[pos_ijk-(imax*jmax)];
							CoordQ4_h[cont+14*numno] = Z_h[pos_ijk]-Z_h[pos_ijk-(imax*jmax)];

							// Node 6:
							CoordQ4_h[cont+15*numno] = X_h[pos_ijk+1]-X_h[pos_ijk-(imax*jmax)];
							CoordQ4_h[cont+16*numno] = Y_h[pos_ijk+1]-Y_h[pos_ijk-(imax*jmax)];
							CoordQ4_h[cont+17*numno] = Z_h[pos_ijk+1]-Z_h[pos_ijk-(imax*jmax)];

							// Node 7:
							CoordQ4_h[cont+18*numno] = X_h[pos_ijk+imax+1]-X_h[pos_ijk-(imax*jmax)];
							CoordQ4_h[cont+19*numno] = Y_h[pos_ijk+imax+1]-Y_h[pos_ijk-(imax*jmax)];
							CoordQ4_h[cont+20*numno] = Z_h[pos_ijk+imax+1]-Z_h[pos_ijk-(imax*jmax)];

							// Node 8:
							CoordQ4_h[cont+21*numno] = X_h[pos_ijk+imax]-X_h[pos_ijk-(imax*jmax)];
							CoordQ4_h[cont+22*numno] = Y_h[pos_ijk+imax]-Y_h[pos_ijk-(imax*jmax)];
							CoordQ4_h[cont+23*numno] = Z_h[pos_ijk+imax]-Z_h[pos_ijk-(imax*jmax)];

						}

					}

				}

				// ****************************** Evaluating CoordQ5_h ******************************

				if(i != 0) {

					if(j != 0) {

						if(k != kmax-1) {

							// Node 1:
							CoordQ5_h[cont+ 0*numno] = X_h[pos_ijk-imax-1]-X_h[pos_ijk-imax-1];
							CoordQ5_h[cont+ 1*numno] = Y_h[pos_ijk-imax-1]-Y_h[pos_ijk-imax-1];
							CoordQ5_h[cont+ 2*numno] = Z_h[pos_ijk-imax-1]-Z_h[pos_ijk-imax-1];

							// Node 2:
							CoordQ5_h[cont+ 3*numno] = X_h[pos_ijk-imax]-X_h[pos_ijk-imax-1];
							CoordQ5_h[cont+ 4*numno] = Y_h[pos_ijk-imax]-Y_h[pos_ijk-imax-1];
							CoordQ5_h[cont+ 5*numno] = Z_h[pos_ijk-imax]-Z_h[pos_ijk-imax-1];

							// Node 3:
							CoordQ5_h[cont+ 6*numno] = X_h[pos_ijk]-X_h[pos_ijk-imax-1];
							CoordQ5_h[cont+ 7*numno] = Y_h[pos_ijk]-Y_h[pos_ijk-imax-1];
							CoordQ5_h[cont+ 8*numno] = Z_h[pos_ijk]-Z_h[pos_ijk-imax-1];

							// Node 4:
							CoordQ5_h[cont+ 9*numno] = X_h[pos_ijk-1]-X_h[pos_ijk-imax-1];
							CoordQ5_h[cont+10*numno] = Y_h[pos_ijk-1]-Y_h[pos_ijk-imax-1];
							CoordQ5_h[cont+11*numno] = Z_h[pos_ijk-1]-Z_h[pos_ijk-imax-1];

							// Node 5:
							CoordQ5_h[cont+12*numno] = X_h[pos_ijk+(imax*jmax)-imax-1]-X_h[pos_ijk-imax-1];
							CoordQ5_h[cont+13*numno] = Y_h[pos_ijk+(imax*jmax)-imax-1]-Y_h[pos_ijk-imax-1];
							CoordQ5_h[cont+14*numno] = Z_h[pos_ijk+(imax*jmax)-imax-1]-Z_h[pos_ijk-imax-1];

							// Node 6:
							CoordQ5_h[cont+15*numno] = X_h[pos_ijk+(imax*jmax)-imax]-X_h[pos_ijk-imax-1];
							CoordQ5_h[cont+16*numno] = Y_h[pos_ijk+(imax*jmax)-imax]-Y_h[pos_ijk-imax-1];
							CoordQ5_h[cont+17*numno] = Z_h[pos_ijk+(imax*jmax)-imax]-Z_h[pos_ijk-imax-1];

							// Node 7:
							CoordQ5_h[cont+18*numno] = X_h[pos_ijk+(imax*jmax)]-X_h[pos_ijk-imax-1];
							CoordQ5_h[cont+19*numno] = Y_h[pos_ijk+(imax*jmax)]-Y_h[pos_ijk-imax-1];
							CoordQ5_h[cont+20*numno] = Z_h[pos_ijk+(imax*jmax)]-Z_h[pos_ijk-imax-1];

							// Node 8:
							CoordQ5_h[cont+21*numno] = X_h[pos_ijk+(imax*jmax)-1]-X_h[pos_ijk-imax-1];
							CoordQ5_h[cont+22*numno] = Y_h[pos_ijk+(imax*jmax)-1]-Y_h[pos_ijk-imax-1];
							CoordQ5_h[cont+23*numno] = Z_h[pos_ijk+(imax*jmax)-1]-Z_h[pos_ijk-imax-1];

						}

					}

				}

				// ****************************** Evaluating CoordQ6_h ******************************

				if(i != imax-1) {

					if(j != 0) {

						if(k != kmax-1) {

							// Node 1:
							CoordQ6_h[cont+ 0*numno] = X_h[pos_ijk-imax]-X_h[pos_ijk-imax];
							CoordQ6_h[cont+ 1*numno] = Y_h[pos_ijk-imax]-Y_h[pos_ijk-imax];
							CoordQ6_h[cont+ 2*numno] = Z_h[pos_ijk-imax]-Z_h[pos_ijk-imax];

							// Node 2:
							CoordQ6_h[cont+ 3*numno] = X_h[pos_ijk-imax+1]-X_h[pos_ijk-imax];
							CoordQ6_h[cont+ 4*numno] = Y_h[pos_ijk-imax+1]-Y_h[pos_ijk-imax];
							CoordQ6_h[cont+ 5*numno] = Z_h[pos_ijk-imax+1]-Z_h[pos_ijk-imax];

							// Node 3:
							CoordQ6_h[cont+ 6*numno] = X_h[pos_ijk+1]-X_h[pos_ijk-imax];
							CoordQ6_h[cont+ 7*numno] = Y_h[pos_ijk+1]-Y_h[pos_ijk-imax];
							CoordQ6_h[cont+ 8*numno] = Z_h[pos_ijk+1]-Z_h[pos_ijk-imax];

							// Node 4:
							CoordQ6_h[cont+ 9*numno] = X_h[pos_ijk]-X_h[pos_ijk-imax];
							CoordQ6_h[cont+10*numno] = Y_h[pos_ijk]-Y_h[pos_ijk-imax];
							CoordQ6_h[cont+11*numno] = Z_h[pos_ijk]-Z_h[pos_ijk-imax];

							// Node 5:
							CoordQ6_h[cont+12*numno] = X_h[pos_ijk+(imax*jmax)-imax]-X_h[pos_ijk-imax];
							CoordQ6_h[cont+13*numno] = Y_h[pos_ijk+(imax*jmax)-imax]-Y_h[pos_ijk-imax];
							CoordQ6_h[cont+14*numno] = Z_h[pos_ijk+(imax*jmax)-imax]-Z_h[pos_ijk-imax];

							// Node 6:
							CoordQ6_h[cont+15*numno] = X_h[pos_ijk+(imax*jmax)-imax+1]-X_h[pos_ijk-imax];
							CoordQ6_h[cont+16*numno] = Y_h[pos_ijk+(imax*jmax)-imax+1]-Y_h[pos_ijk-imax];
							CoordQ6_h[cont+17*numno] = Z_h[pos_ijk+(imax*jmax)-imax+1]-Z_h[pos_ijk-imax];

							// Node 7:
							CoordQ6_h[cont+18*numno] = X_h[pos_ijk+(imax*jmax)+1]-X_h[pos_ijk-imax];
							CoordQ6_h[cont+19*numno] = Y_h[pos_ijk+(imax*jmax)+1]-Y_h[pos_ijk-imax];
							CoordQ6_h[cont+20*numno] = Z_h[pos_ijk+(imax*jmax)+1]-Z_h[pos_ijk-imax];

							// Node 8:
							CoordQ6_h[cont+21*numno] = X_h[pos_ijk+(imax*jmax)]-X_h[pos_ijk-imax];
							CoordQ6_h[cont+22*numno] = Y_h[pos_ijk+(imax*jmax)]-Y_h[pos_ijk-imax];
							CoordQ6_h[cont+23*numno] = Z_h[pos_ijk+(imax*jmax)]-Z_h[pos_ijk-imax];

						}

					}

				}

				// ****************************** Evaluating CoordQ7_h ******************************

				if(i != 0) {

					if(j != jmax-1) {

						if(k != kmax-1) {

							// Node 1:
							CoordQ7_h[cont+ 0*numno] = X_h[pos_ijk-1]-X_h[pos_ijk-1];
							CoordQ7_h[cont+ 1*numno] = Y_h[pos_ijk-1]-Y_h[pos_ijk-1];
							CoordQ7_h[cont+ 2*numno] = Z_h[pos_ijk-1]-Z_h[pos_ijk-1];

							// Node 2:
							CoordQ7_h[cont+ 3*numno] = X_h[pos_ijk]-X_h[pos_ijk-1];
							CoordQ7_h[cont+ 4*numno] = Y_h[pos_ijk]-Y_h[pos_ijk-1];
							CoordQ7_h[cont+ 5*numno] = Z_h[pos_ijk]-Z_h[pos_ijk-1];

							// Node 3:
							CoordQ7_h[cont+ 6*numno] = X_h[pos_ijk+imax]-X_h[pos_ijk-1];
							CoordQ7_h[cont+ 7*numno] = Y_h[pos_ijk+imax]-Y_h[pos_ijk-1];
							CoordQ7_h[cont+ 8*numno] = Z_h[pos_ijk+imax]-Z_h[pos_ijk-1];

							// Node 4:
							CoordQ7_h[cont+ 9*numno] = X_h[pos_ijk+imax-1]-X_h[pos_ijk-1];
							CoordQ7_h[cont+10*numno] = Y_h[pos_ijk+imax-1]-Y_h[pos_ijk-1];
							CoordQ7_h[cont+11*numno] = Z_h[pos_ijk+imax-1]-Z_h[pos_ijk-1];

							// Node 5:
							CoordQ7_h[cont+12*numno] = X_h[pos_ijk+(imax*jmax)-1]-X_h[pos_ijk-1];
							CoordQ7_h[cont+13*numno] = Y_h[pos_ijk+(imax*jmax)-1]-Y_h[pos_ijk-1];
							CoordQ7_h[cont+14*numno] = Z_h[pos_ijk+(imax*jmax)-1]-Z_h[pos_ijk-1];

							// Node 6:
							CoordQ7_h[cont+15*numno] = X_h[pos_ijk+(imax*jmax)]-X_h[pos_ijk-1];
							CoordQ7_h[cont+16*numno] = Y_h[pos_ijk+(imax*jmax)]-Y_h[pos_ijk-1];
							CoordQ7_h[cont+17*numno] = Z_h[pos_ijk+(imax*jmax)]-Z_h[pos_ijk-1];

							// Node 7:
							CoordQ7_h[cont+18*numno] = X_h[pos_ijk+(imax*jmax)+imax]-X_h[pos_ijk-1];
							CoordQ7_h[cont+19*numno] = Y_h[pos_ijk+(imax*jmax)+imax]-Y_h[pos_ijk-1];
							CoordQ7_h[cont+20*numno] = Z_h[pos_ijk+(imax*jmax)+imax]-Z_h[pos_ijk-1];

							// Node 8:
							CoordQ7_h[cont+21*numno] = X_h[pos_ijk+(imax*jmax)+imax-1]-X_h[pos_ijk-1];
							CoordQ7_h[cont+22*numno] = Y_h[pos_ijk+(imax*jmax)+imax-1]-Y_h[pos_ijk-1];
							CoordQ7_h[cont+23*numno] = Z_h[pos_ijk+(imax*jmax)+imax-1]-Z_h[pos_ijk-1];

						}

					}

				}

				// ****************************** Evaluating CoordQ8_h ******************************

				if(i != imax-1) {

					if(j != jmax-1) {

						if(k != kmax-1) {

							// Node 1:
							CoordQ8_h[cont+ 0*numno] = X_h[pos_ijk]-X_h[pos_ijk];
							CoordQ8_h[cont+ 1*numno] = Y_h[pos_ijk]-Y_h[pos_ijk];
							CoordQ8_h[cont+ 2*numno] = Z_h[pos_ijk]-Z_h[pos_ijk];

							// Node 2:
							CoordQ8_h[cont+ 3*numno] = X_h[pos_ijk+1]-X_h[pos_ijk];
							CoordQ8_h[cont+ 4*numno] = Y_h[pos_ijk+1]-Y_h[pos_ijk];
							CoordQ8_h[cont+ 5*numno] = Z_h[pos_ijk+1]-Z_h[pos_ijk];

							// Node 3:
							CoordQ8_h[cont+ 6*numno] = X_h[pos_ijk+imax+1]-X_h[pos_ijk];
							CoordQ8_h[cont+ 7*numno] = Y_h[pos_ijk+imax+1]-Y_h[pos_ijk];
							CoordQ8_h[cont+ 8*numno] = Z_h[pos_ijk+imax+1]-Z_h[pos_ijk];

							// Node 4:
							CoordQ8_h[cont+ 9*numno] = X_h[pos_ijk+imax]-X_h[pos_ijk];
							CoordQ8_h[cont+10*numno] = Y_h[pos_ijk+imax]-Y_h[pos_ijk];
							CoordQ8_h[cont+11*numno] = Z_h[pos_ijk+imax]-Z_h[pos_ijk];

							// Node 5:
							CoordQ8_h[cont+12*numno] = X_h[pos_ijk+(imax*jmax)]-X_h[pos_ijk];
							CoordQ8_h[cont+13*numno] = Y_h[pos_ijk+(imax*jmax)]-Y_h[pos_ijk];
							CoordQ8_h[cont+14*numno] = Z_h[pos_ijk+(imax*jmax)]-Z_h[pos_ijk];

							// Node 6:
							CoordQ8_h[cont+15*numno] = X_h[pos_ijk+(imax*jmax)+1]-X_h[pos_ijk];
							CoordQ8_h[cont+16*numno] = Y_h[pos_ijk+(imax*jmax)+1]-Y_h[pos_ijk];
							CoordQ8_h[cont+17*numno] = Z_h[pos_ijk+(imax*jmax)+1]-Z_h[pos_ijk];

							// Node 7:
							CoordQ8_h[cont+18*numno] = X_h[pos_ijk+(imax*jmax)+imax+1]-X_h[pos_ijk];
							CoordQ8_h[cont+19*numno] = Y_h[pos_ijk+(imax*jmax)+imax+1]-Y_h[pos_ijk];
							CoordQ8_h[cont+20*numno] = Z_h[pos_ijk+(imax*jmax)+imax+1]-Z_h[pos_ijk];

							// Node 8:
							CoordQ8_h[cont+21*numno] = X_h[pos_ijk+(imax*jmax)+imax]-X_h[pos_ijk];
							CoordQ8_h[cont+22*numno] = Y_h[pos_ijk+(imax*jmax)+imax]-Y_h[pos_ijk];
							CoordQ8_h[cont+23*numno] = Z_h[pos_ijk+(imax*jmax)+imax]-Z_h[pos_ijk];

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

		for(j=0; j<6; j++) {  // 6 = stress components

			//fprintf(outFile, "  IP %d  %+0.8e %+0.8e %+0.8e %+0.8e %+0.8e %+0.8e", j+1, Strain_h[i+0*numel+j*numel*6], Strain_h[i+1*numel+j*numel*6],
																		           //Strain_h[i+2*numel+j*numel*6], Strain_h[i+3*numel+j*numel*6],
																		           //Strain_h[i+4*numel+j*numel*6], Strain_h[i+5*numel+j*numel*6]);

			fprintf(outFile, "  IP %d  %+0.8e %+0.8e %+0.8e %+0.8e %+0.8e %+0.8e", j+1, Strain_h[6*i+0], Strain_h[6*i+1],
																		           Strain_h[6*i+2], Strain_h[6*i+3],
																		           Strain_h[6*i+4], Strain_h[6*i+5]);

			fprintf(outFile, "\n");

		}

		fprintf(outFile, "\n");

	}

	fclose(outFile);

}

//==============================================================================
// Writing strain state in output file - GPUStrainState:

void WriteStrainStateCouple(int numel, double *DeltaStrain_h, double *DeltaStrainChr_h)
{
	int i, j;

	// Write results in StrainChr vector

	for(i=0; i<numel; i++) {

		for(j=0; j<8; j++) {  // 8 = number of integration points

			DeltaStrainChr_h[i+0*numel+j*numel*6] = DeltaStrain_h[i+0*numel+j*numel*6];
			DeltaStrainChr_h[i+1*numel+j*numel*6] = DeltaStrain_h[i+1*numel+j*numel*6];
			DeltaStrainChr_h[i+2*numel+j*numel*6] = DeltaStrain_h[i+2*numel+j*numel*6];
			DeltaStrainChr_h[i+3*numel+j*numel*6] = DeltaStrain_h[i+3*numel+j*numel*6];
			DeltaStrainChr_h[i+4*numel+j*numel*6] = DeltaStrain_h[i+4*numel+j*numel*6];
			DeltaStrainChr_h[i+5*numel+j*numel*6] = DeltaStrain_h[i+5*numel+j*numel*6];

		}

	}

}

//==============================================================================
// Writing strain state in output file - GPUStrainState:

void CopyDeltaStrainFromGlobalToLocal(int numel, double *DeltaStrain_h, double *DeltaStrainChr_h)
{
	int i;

	// Write results in StrainChr vector

	for(i=0; i<numel; i++) {

		DeltaStrainChr_h[i+0*numel] = DeltaStrain_h[i+0*numel];
		DeltaStrainChr_h[i+1*numel] = DeltaStrain_h[i+1*numel];
		DeltaStrainChr_h[i+2*numel] = DeltaStrain_h[i+2*numel];
		DeltaStrainChr_h[i+3*numel] = DeltaStrain_h[i+3*numel];
		DeltaStrainChr_h[i+4*numel] = DeltaStrain_h[i+4*numel];
		DeltaStrainChr_h[i+5*numel] = DeltaStrain_h[i+5*numel];

	}

	//for(i=0; i<100; i++)
		//printf("%e %e %e \n", DeltaStrainChr_h[i+0*numel], DeltaStrainChr_h[i+1*numel], DeltaStrainChr_h[i+2*numel]);

}

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
// Writing strain state in output file - GPUStrainState:

void WriteDisplacementCouple(int numno, double *Vector_X_h, double *VectorChr_X_h)
{
	int i;

	// Write results in StrainChr vector

	for(i=0; i<numno; i++) {

		VectorChr_X_h[3*i  ] = Vector_X_h[3*i  ];
		VectorChr_X_h[3*i+1] = Vector_X_h[3*i+1];
		VectorChr_X_h[3*i+2] = Vector_X_h[3*i+2];

	}

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

//==============================================================================
// GPU global memory information:

void EvaluateGPUFreeMemory()
{
	size_t free_byte, total_byte;
	double FreeMem[4], UsedMem[4], TotalMem[4];

	printf("\n");
	printf("         GPU global memory report \n");
	printf("         ========================================= ");
	printf("\n");

	for(int i=0; i<4; i++) {
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

}

//==============================================================================
// Evalauting initial stress state:

int cGPU::EvaluateInitialStressState(int *VecLinkCellMesh_h, double *PrevPorosity_h, double *StressTotalChr_h)
{

	int i, j, k, kaux, nel, num_nodes, cont;
	double Zwd, Z;  // Zwd = water depth
	double StressInitialAux;

	int Ti, Tj, Tk, ElemIndex;
	int nx, ny, nz;

	double densCurr, densPrev, K0x, K0y;  //  dg = grain density, ds = saturated density,K0x = relation between Sx and Sz, K0y = relation between Sy and Sz
	double dw = 9810., por;
	int pos;

	double ZcenterCurr, ZcenterPrev, Ztop;

	// imax             Number of nodes in x direction
	// jmax             Number of nodes in y direction
	// kmax             Number of nodes in z direction

	Ti = imax - 1;  // Total number of elements in x direction
	Tj = jmax - 1;  // Total number of elements in y direction
	Tk = kmax - 1;  // Total number of elements in z direction

	num_nodes = 8;    // Number of node on element

	nx = Ti - nsi1 - nsi2;     // Total number of elements in reservoir in i direction
	ny = Tj - nsj1 - nsj2;     // Total number of elements in reservoir in j direction
	nz = Tk - nov -  nun;      // Total number of elements in reservoir in k direction

	nel = Ti*Tj*Tk;  // Total number of elements in the model

	// ------------------------------------------------------------------

	for(j=0; j<Tj; j++) {                        // Loop elements in x direcion

		for(i=0; i<Ti; i++) {                    // Loop elements in y direcion

			StressInitialAux = 0.;

			for(kaux=0; kaux<Tk; kaux++) {       // Vertical Direction

				k = Tk - 1 - kaux;

				ElemIndex = k*(Ti*Tj) + j*Ti + i;

				// Material density

				//if(Reservoir_Id_h[Material_Id_h[ElemIndex]-1] == 0)          // Outside of the reservoir
					densCurr = Material_Density_h[Material_Id_h[ElemIndex]-1+0*nummat];
				//else {                                                       // Inside of the reservoir
					//pos = VecLinkCellMesh_h[ElemIndex]-1;
					//por = PrevPorosity_h[pos];
					//densCurr = por*Material_Density_h[Material_Id_h[ElemIndex]-1+0*nummat];
				//}

				K0x = Material_Density_h[Material_Id_h[ElemIndex]-1+1*nummat];
				K0y = Material_Density_h[Material_Id_h[ElemIndex]-1+2*nummat];

				// --------------------------------------------------------------------------------

				Ztop = 0.;

				for(int j2=4; j2<num_nodes; j2++) {

					Z = Z_h[Connect_h[ElemIndex + j2*numel]-1];

					Ztop += Z/4;

				}

				// --------------------------------------------------------------------------------

				ZcenterCurr = 0.;

				for(int j2=0; j2<num_nodes; j2++) {

					Z = Z_h[Connect_h[ElemIndex + j2*numel]-1];

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
				StressTotalChr_h[ElemIndex+0*nel] = StressInitialAux;
				StressTotalChr_h[ElemIndex+1*nel] = K0x*StressInitialAux;
				StressTotalChr_h[ElemIndex+2*nel] = K0y*StressInitialAux;

				// --------------------------------------------------------------------------------

			}                                    // Vertical Direction

		}

	}

	return 1;

}