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

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <time.h>
#include <cuda_runtime.h>

//----------------------------------------------------
// External Functions for one GPU Implementation						 
//----------------------------------------------------


extern "C" void ImpositionBoundaryCondition(int Id, int BlockSizeX, int _iNumMeshNodes, int _iNumSuppNodes, int _inumDiaPart, int *supp, double *K, double *B);

extern "C" void ImpositionBoundaryConditionNeumann(int Id, int BlockSizeX, int _iNumMeshNodes, int _iNumSuppNodes, int *supp, double *B);


//==============================================================================
__global__ void ImpositionBoundaryConditionKernel(int _iNumMeshNodes, int _iNumSuppNodes, int _inumDiaPart, int *supp, double *K, double *B)
{

	// _iNumSuppNodes = total number of supports

	int i;
	const int xIndex = blockIdx.x*blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y*blockDim.y + threadIdx.y;
	const int thread_id = (gridDim.x*blockDim.x)*yIndex + xIndex;

	if(thread_id < _iNumSuppNodes) {  // -----------------------

		for(i=0; i<_inumDiaPart; i++) {

			if(supp[thread_id+1*_iNumSuppNodes] == 1) K[3*(supp[thread_id]-1)   + i*3*_iNumMeshNodes] = 0.;
			if(supp[thread_id+2*_iNumSuppNodes] == 1) K[3*(supp[thread_id]-1)+1 + i*3*_iNumMeshNodes] = 0.;
			if(supp[thread_id+3*_iNumSuppNodes] == 1) K[3*(supp[thread_id]-1)+2 + i*3*_iNumMeshNodes] = 0.;

		}

		if(supp[thread_id+1*_iNumSuppNodes] == 1) B[3*(supp[thread_id]-1)  ] = 0.;
		if(supp[thread_id+2*_iNumSuppNodes] == 1) B[3*(supp[thread_id]-1)+1] = 0.;
		if(supp[thread_id+3*_iNumSuppNodes] == 1) B[3*(supp[thread_id]-1)+2] = 0.;

	}

}

//=====================================================================================================================
void ImpositionBoundaryCondition(int Id, int BlockSizeX, int _iNumMeshNodes, int _iNumSuppNodes, int _inumDiaPart, int *supp, double *K, double *B)
{
	double time;

	cudaSetDevice(Id);

	dim3 threadsPerBlock(BlockSizeX, BlockSizeX);
	dim3 blocksPerGrid(int(sqrt(double(_iNumSuppNodes))/BlockSizeX)+1, int(sqrt(double(_iNumSuppNodes))/BlockSizeX)+1);

	time = clock();

	ImpositionBoundaryConditionKernel<<<blocksPerGrid, threadsPerBlock>>>(_iNumMeshNodes, _iNumSuppNodes, _inumDiaPart, supp, K, B);

	cudaDeviceSynchronize();

	printf("         Time Execution : %0.3f s \n", (clock()-time)/CLOCKS_PER_SEC);

}

//==============================================================================
__global__ void ImpositionBoundaryConditionNeumannKernel(int _iNumMeshNodes, int _iNumSuppNodes, int *supp, double *B)
{

	// _iNumSuppNodes = total number of supports

	int i;
	const int xIndex = blockIdx.x*blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y*blockDim.y + threadIdx.y;
	const int thread_id = (gridDim.x*blockDim.x)*yIndex + xIndex;

	if(thread_id < _iNumSuppNodes) {  // -----------------------

		if(supp[thread_id+1*_iNumSuppNodes] == 1) B[3*(supp[thread_id]-1)  ] = 0.;
		if(supp[thread_id+2*_iNumSuppNodes] == 1) B[3*(supp[thread_id]-1)+1] = 0.;
		if(supp[thread_id+3*_iNumSuppNodes] == 1) B[3*(supp[thread_id]-1)+2] = 0.;

	}

}

//=====================================================================================================================
void ImpositionBoundaryConditionNeumann(int Id, int BlockSizeX, int _iNumMeshNodes, int _iNumSuppNodes, int *supp, double *B)
{
	double time;

	cudaSetDevice(Id);

	dim3 threadsPerBlock(BlockSizeX, BlockSizeX);
	dim3 blocksPerGrid(int(sqrt(double(_iNumSuppNodes))/BlockSizeX)+1, int(sqrt(double(_iNumSuppNodes))/BlockSizeX)+1);

	time = clock();

	ImpositionBoundaryConditionNeumannKernel<<<blocksPerGrid, threadsPerBlock>>>(_iNumMeshNodes, _iNumSuppNodes, supp, B);

	cudaDeviceSynchronize();

	printf("         Time Execution : %0.3f s \n", (clock()-time)/CLOCKS_PER_SEC);

}

//==============================================================================
