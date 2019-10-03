/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\
*  Copyright (c) 2019 <GTEP> - All Rights Reserved                            *
*  This file is part of HERMES Project.                                       *
*  Unauthorized copying of this file, via any medium is strictly prohibited.  *
*  Proprietary and confidential.                                              *
*                                                                             *
*  Developers:                                                                *
*     - Nelson Inoue <inoue@puc-rio.br>                                       *
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#ifndef	_FemGPU_H
#define	_FemGPU_H

#include "Input.h"
#include "ReadFile_CHR.h"
#include <vector>

typedef class cInput *pcInput, &rcInput;

//==============================================================================
// cGPU
//==============================================================================
typedef class cFemOneGPU *pcFemOneGPU, &rcFemOneGPU;

using namespace std;

#define OUTPUT_STRESS   OUTPUT_DIR "StressState.pos"
#define OUTPUT_STRAIN   OUTPUT_DIR "StrainState.pos"
#define OUTPUT_EXAMPLES "../examples/"

class cFemOneGPU
{
public:

	cInput *in;

	double delta_new_;

	double _dE;            // Young's modulus
	double _dNu;           // Poisson's ratio

	int Id;
	int multiProcessorCount, maxThreadsPerBlock, maxThreadsPerMultiProc, ThreadsMultiProcPerGpu, BlockMultiProcPerGpu;
	int BlockSizeX, BlockSizeY, NumBlockX, NumBlockY, VecNumBlockX, VecNumBlockY;

	double *delta_1_aux, *delta_new, *dTq, *delta_old;
	double *delta_new_h;

	double *U, *R, *D, *D_Full, *Q, *M, *S, *B, *X_Full;
	int *off;

	int numel_Halo, numelprv_Halo, numno_Halo, numnoprv_Halo;
	int *numelprv_Halo_Color, *numel_Halo_Color;

	double *coord, *prop, *K;
	int *connect, *offfull, *supp;

	double *vector;

	double *stress, *strain, *con;
	double *stress_h, *strain_h, *X_h;

	int numcolor;
	int *numelcolor, *numelcolorprv;

	char _anmType[30];

	cFemOneGPU() {};
	cFemOneGPU(string inputFile);
	~cFemOneGPU() { delete chrData; };
	void AnalyzeFemOneGPU(int ii, int jj, int GridCoord, double *dP_h, double *DeltaStrainChr_h, double *DeltaStressChr_h);
	void PrepareInputData();
	void AllocateAndCopyVectors(); 
	void AssemblyStiffnessMatrix();
	void EvaluateStrainState(double * );
	void WriteAverageStrainState(int , double *);
	void WriteAverageStressState(int , double *);
	void ImpositionBoundaryConditionClass();
	void EvaluateMmatrixClass();
	void ImpositionBoundaryConditionNeumannClass();
	void SolveLinearSystemClass(double * );
	void EvaluateStressStateClass(double * );
	void EvaluateInitialStressState(double * );

	void CheckK();
	void FreeMemory();
	void PrintKDia();
	void PrintDispl();
	void PrintRate(double *);

	double CGError, CGTime, StffTime, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, sumK;
	void PrintTimeCG();
	int CGIter;
	double *t;

	// --------------------------------------------------------------------------------------------------------------------------------------------------------

	// For coupling

	int *LinkMeshMesh_h, *LinkMeshCell_h;
	int *LinkMeshMeshColor_h, *LinkMeshCellColor_h, *LinkMeshColor_h;
	int *numelcolornodalforce, *numelcolornodalforceprv;
	double *dP, *F_h;
	int *LinkMeshMeshColor, *LinkMeshCellColor, *LinkMeshColor;

	typedef std::vector<int> ivector;
	typedef std::vector<ivector> ijvector;
	ijvector ElemColorByGPU;
	
	void AllocateAndCopyVectorsPartialCoupling();
	void LinkMeshGridMapping(int GridCoord);
	void EvalColoringMeshBrick8Struct();

	void EvaluateNodalForce(double *dP_h);
	void LinkColorMapping();

	// --------------------------------------------------------------------------------------------------------------------------------------------------------

private:
	int nx, ny, nz;
	int nNodes;
	int nOffsets;
	int nDofNode;
	int nSupports;
	int nElements;
	int nMaterials;
	ReadFile_CHR *chrData;

	void ReadCHR(string);

};


#endif

