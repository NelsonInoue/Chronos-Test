#ifndef	_FemGPU_H
#define	_FemGPU_H

#include "input.h"

typedef class cInput *pcInput, &rcInput;

//==============================================================================
// cGPU
//==============================================================================
typedef class cFemOneGPU *pcFemOneGPU, &rcFemOneGPU;

using namespace std;

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

	double *X, *R, *D, *D_Full, *Q, *M, *S, *B, *X_Full;
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

	cFemOneGPU();
	void AnalyzeFemOneGPU();
	void PrepareInputData();
	void AllocateAndCopyVectors(); 
	void AssemblyStiffnessMatrix();
	void EvaluateStrainState();
	void WriteAverageStrainState(int , double *);
	void WriteAverageStressState(int , double *);

	void CheckK();
	void FreeMemory();
	void PrintKDia();

	double CGError, CGTime, StffTime, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, sumK;
	void PrintTimeCG();
	int CGIter;
	double *t;

};


#endif

