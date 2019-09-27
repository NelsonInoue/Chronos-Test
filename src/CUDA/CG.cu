/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\
*  Copyright (c) 2019 <GTEP> - All Rights Reserved                            *
*  This file is part of HERMES Project.                                       *
*  Unauthorized copying of this file, via any medium is strictly prohibited.  *
*  Proprietary and confidential.                                              *
*                                                                             *
*  Developers:                                                                *
*     - Nelson Inoue <inoue@puc-rio.br>                                       *
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

extern "C" void SolveLinearSystem(int Id, int _inumDiaPart, int _iNumDofNode, int BlockSizeX, int BlockMultiProcPerGpu, int _iNumMeshNodes, double *B, double *M, double *K, int *off, double *t, double *CGTime, int *CGIter, double *CGError, double *X, double *R, double *D, double *Q, double *S,
		double *delta_1_aux, double *delta_new, double *dTq, double *delta_old, double *delta_new_h);



//=============== x =============== x =============== x =============== x =============== x =============== x ===============

// This kernel subtracts a vector by other vector. 
  
__global__ void _FILL_K_M(int _iNumDofNode, int _iNumMeshNodes, int _iNumMeshNodesgpu, int _iNumMeshNodesprv, int _inumDiaPart, double *KDia, double *K, double *M_full, double *M)                             
{	
	const int xIndex = blockIdx.x*blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y*blockDim.y + threadIdx.y;
	const int thread_id = (gridDim.x*blockDim.x)*yIndex + xIndex;

	int n;

	if(thread_id < _iNumDofNode*_iNumMeshNodesgpu) {
		for(n=0; n<_inumDiaPart; n++)
			K[thread_id + n*_iNumDofNode*_iNumMeshNodesgpu] = KDia[thread_id + _iNumDofNode*_iNumMeshNodesprv + n*_iNumDofNode*_iNumMeshNodes];

		M[thread_id] = M_full[thread_id + _iNumDofNode*_iNumMeshNodesprv];

	}
	
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============

// This kernel subtracts a vector by other vector. 
  
__global__ void _CLEAN_V(int _iNumDofNode, int _iNumMeshNodesgpu, double *X)                             
{	
	const int xIndex = blockIdx.x*blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y*blockDim.y + threadIdx.y;
	const int thread_id = (gridDim.x*blockDim.x)*yIndex + xIndex;

	if(thread_id < _iNumDofNode*_iNumMeshNodesgpu)
		X[thread_id] = 0.;
		
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============

// This kernel subtracts a vector by other vector. 
  
__global__ void _SUBT_V_V(int _iNumDofNode, int _iNumMeshNodesgpu, int _iNumMeshNodesprv, double *B, double *R)                             
{	
	const int xIndex = blockIdx.x*blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y*blockDim.y + threadIdx.y;
	const int thread_id = (gridDim.x*blockDim.x)*yIndex + xIndex;

	if(thread_id < _iNumDofNode*_iNumMeshNodesgpu)
		R[thread_id] = B[thread_id];
		
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============

// This kernel multiplies a diagonal matrix by a vector.

  __global__ void _MULT_M_V(int _iNumDofNode, int _iNumMeshNodesgpu, int _iNumMeshNodesprv, double *M, double *R, double *D)
{	
	const int xIndex = blockIdx.x*blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y*blockDim.y + threadIdx.y;
	const int thread_id = (gridDim.x*blockDim.x)*yIndex + xIndex;

	if(thread_id < _iNumDofNode*_iNumMeshNodesgpu)
		D[thread_id] = M[thread_id] * R[thread_id];
		
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============

// This kernel multiplies a vector by a other vector of n terms. 
__global__ void _SET_ZERO(double *delta_new)
{
	delta_new[0] = 0.;
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============

// This kernel multiplies a vector by a other vector of n terms. 
__global__ void _DOT_V_V(int _iNumDofNode, int _iNumMeshNodesgpu, double *R, double *D, double *delta_1_aux)
{
	__shared__ double sdata[1024];
	
	int thread_id = blockIdx.x*blockDim.x + threadIdx.x;    // Thread id on grid
	int gridSize  = gridDim.x*blockDim.x;                   // Total number of threads 

	sdata[threadIdx.x] = 0;

	while(thread_id < _iNumDofNode*_iNumMeshNodesgpu) {
	
		// Multiplying the vertor r by the vector d
		
		sdata[threadIdx.x] += R[thread_id]*D[thread_id];
		
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
		delta_1_aux[blockIdx.x] = sdata[0];
					
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============

// This kernel multiplies a vector by a other vector of n terms. 
__global__ void _SUM_16(double *delta_1_aux, double *delta_new)
{
	__shared__ double sdata[16];

	const int thread_id = blockDim.x*threadIdx.y + threadIdx.x;
	
	// ******************************************************************************************************* 
	
	if(thread_id < 16) sdata[thread_id] = delta_1_aux[thread_id]; __syncthreads();
	
	if(thread_id < 8) sdata[thread_id] += sdata[thread_id + 8]; __syncthreads();
	if(thread_id < 4) sdata[thread_id] += sdata[thread_id + 4]; __syncthreads();
	if(thread_id < 2) sdata[thread_id] += sdata[thread_id + 2]; __syncthreads();
	if(thread_id < 1) sdata[thread_id] += sdata[thread_id + 1]; __syncthreads();
	
	if(thread_id == 0)
		delta_new[0] = sdata[0];

}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============

// This kernel add up the sixteen terms of the vector delta_new_d. 
__global__ void _SUM_32(double *delta_1_aux, double *delta_new)
{

	__shared__ double sdata[32];

	const int thread_id = blockDim.x*threadIdx.y + threadIdx.x;
	
	// ******************************************************************************************************* 
	
	if(thread_id < 32) sdata[thread_id] = delta_1_aux[thread_id]; __syncthreads();
	
	if(thread_id <16) sdata[thread_id] += sdata[thread_id +16]; __syncthreads();
	if(thread_id < 8) sdata[thread_id] += sdata[thread_id + 8]; __syncthreads();
	if(thread_id < 4) sdata[thread_id] += sdata[thread_id + 4]; __syncthreads();
	if(thread_id < 2) sdata[thread_id] += sdata[thread_id + 2]; __syncthreads();
	if(thread_id < 1) sdata[thread_id] += sdata[thread_id + 1]; __syncthreads();
	
	if(thread_id == 0)
		delta_new[0] = sdata[0];
		
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============

// This kernel add up the sixteen terms of the vector delta_new_d. 
__global__ void _SUM_64(double *delta_1_aux, double *delta_new)
{

	__shared__ double sdata[64];

	const int thread_id = blockDim.x*threadIdx.y + threadIdx.x;
	
	// ******************************************************************************************************* 
	
	if(thread_id < 64) sdata[thread_id] = delta_1_aux[thread_id]; __syncthreads();
	
	if(thread_id <32) sdata[thread_id] += sdata[thread_id +32]; __syncthreads();
	if(thread_id <16) sdata[thread_id] += sdata[thread_id +16]; __syncthreads();
	if(thread_id < 8) sdata[thread_id] += sdata[thread_id + 8]; __syncthreads();
	if(thread_id < 4) sdata[thread_id] += sdata[thread_id + 4]; __syncthreads();
	if(thread_id < 2) sdata[thread_id] += sdata[thread_id + 2]; __syncthreads();
	if(thread_id < 1) sdata[thread_id] += sdata[thread_id + 1]; __syncthreads();
	
	if(thread_id == 0)
		delta_new[0] = sdata[0];
		
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============

// This kernel add up the sixteen terms of the vector delta_new_d. 
__global__ void _SUM_128(double *delta_1_aux, double *delta_new)
{

	__shared__ double sdata[128];

	const int thread_id = blockDim.x*threadIdx.y + threadIdx.x;
	
	// ******************************************************************************************************* 
	
	if(thread_id < 128) sdata[thread_id] = delta_1_aux[thread_id]; __syncthreads();
	
	if(thread_id <64) sdata[thread_id] += sdata[thread_id +64]; __syncthreads();
	if(thread_id <32) sdata[thread_id] += sdata[thread_id +32]; __syncthreads();
	if(thread_id <16) sdata[thread_id] += sdata[thread_id +16]; __syncthreads();
	if(thread_id < 8) sdata[thread_id] += sdata[thread_id + 8]; __syncthreads();
	if(thread_id < 4) sdata[thread_id] += sdata[thread_id + 4]; __syncthreads();
	if(thread_id < 2) sdata[thread_id] += sdata[thread_id + 2]; __syncthreads();
	if(thread_id < 1) sdata[thread_id] += sdata[thread_id + 1]; __syncthreads();
	
	if(thread_id == 0)
		delta_new[0] = sdata[0];
		
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============

// This kernel add up the sixteen terms of the vector delta_new_d. 
__global__ void _SUM_256(double *delta_1_aux, double *delta_new)
{

	__shared__ double sdata[256];

	const int thread_id = blockDim.x*threadIdx.y + threadIdx.x;
	
	// ******************************************************************************************************* 
	
	if(thread_id < 256) sdata[thread_id] = delta_1_aux[thread_id]; __syncthreads();
	
	if(thread_id <128) sdata[thread_id] += sdata[thread_id +128]; __syncthreads();
	if(thread_id <64)  sdata[thread_id] += sdata[thread_id +64]; __syncthreads();
	if(thread_id <32)  sdata[thread_id] += sdata[thread_id +32]; __syncthreads();
	if(thread_id <16)  sdata[thread_id] += sdata[thread_id +16]; __syncthreads();
	if(thread_id < 8)  sdata[thread_id] += sdata[thread_id + 8]; __syncthreads();
	if(thread_id < 4)  sdata[thread_id] += sdata[thread_id + 4]; __syncthreads();
	if(thread_id < 2)  sdata[thread_id] += sdata[thread_id + 2]; __syncthreads();
	if(thread_id < 1)  sdata[thread_id] += sdata[thread_id + 1]; __syncthreads();
	
	if(thread_id == 0)
		delta_new[0] = sdata[0];
		
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============
// This kernel update vector Vector_D_Global_d
__global__ void _Copy_Local_Global_(int _iNumDofNode, int _iNumMeshNodesgpu, int _iNumMeshNodesprv, double *D, double *D_Full)
{
	const int xIndex = blockIdx.x*blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y*blockDim.y + threadIdx.y;
	const int thread_id = (gridDim.x*blockDim.x)*yIndex + xIndex;
	
	if(thread_id < _iNumDofNode*_iNumMeshNodesgpu)
		D_Full[thread_id + _iNumDofNode*_iNumMeshNodesprv] = D[thread_id];
	
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============

// This kernel multiplies a diagonal matrix by a vector.

  __global__ void _ADD_V_V_X_(int _iNumDofNode, int _iNumMeshNodesgpu, double *D, double *Q, double *delta_new, double *dTq, double *X, double *R)
{	
	const int xIndex = blockIdx.x*blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y*blockDim.y + threadIdx.y;
	const int thread_id = (gridDim.x*blockDim.x)*yIndex + xIndex;

	if(thread_id < _iNumDofNode*_iNumMeshNodesgpu) {
		X[thread_id] += (delta_new[0]/dTq[0]) * D[thread_id];
		R[thread_id] -= (delta_new[0]/dTq[0]) * Q[thread_id];
	}
		
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============

// This kernel multiplies a diagonal matrix by a vector.

  __global__ void _UPDATE_DELTA_(double *delta_new, double *delta_old)
{	
	delta_old[0] = delta_new[0];		
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============

// This kernel multiplies a diagonal matrix by a vector.

  __global__ void _ADD_V_V_D_(int _iNumDofNode, int _iNumMeshNodesgpu, double *S, double *D, double *delta_new, double *delta_old)
{	
	const int xIndex = blockIdx.x*blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y*blockDim.y + threadIdx.y;
	const int thread_id = (gridDim.x*blockDim.x)*yIndex + xIndex;
	
	if(thread_id < _iNumDofNode*_iNumMeshNodesgpu)
		D[thread_id] = S[thread_id] + (delta_new[0]/delta_old[0]) * D[thread_id];
		
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============
void _Sum_DOT_V_V(dim3 blocksPerGridOne, dim3 threadsPerBlockVary, double *delta_1_aux, double *delta_new)
{
	if(threadsPerBlockVary.x == 16)  _SUM_16  <<< blocksPerGridOne, threadsPerBlockVary >>> (delta_1_aux, delta_new);
	if(threadsPerBlockVary.x == 32)  _SUM_32  <<< blocksPerGridOne, threadsPerBlockVary >>> (delta_1_aux, delta_new);
	if(threadsPerBlockVary.x == 64)  _SUM_64  <<< blocksPerGridOne, threadsPerBlockVary >>> (delta_1_aux, delta_new);
	if(threadsPerBlockVary.x == 128) _SUM_128 <<< blocksPerGridOne, threadsPerBlockVary >>> (delta_1_aux, delta_new);
	if(threadsPerBlockVary.x == 256) _SUM_256 <<< blocksPerGridOne, threadsPerBlockVary >>> (delta_1_aux, delta_new);
	
}

//=============== x =============== x =============== x =============== x =============== x =============== x ===============

// This kernel multiplies a vector by a other vector of n terms. 
__global__ void _MULT_K_V(int _iNumDofNode, int _iNumMeshNodes, int _iNumMeshNodesgpu, int _iNumMeshNodesprv, int _inumDiaPart, int *off, double *K, double *D, double *Q)
{
	const int xIndex = blockIdx.x*blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y*blockDim.y + threadIdx.y;
	const int thread_id = (gridDim.x*blockDim.x)*yIndex + xIndex;

	int n, row, col, num_cols;
	double dot=0, val;

	row      =  thread_id + _iNumDofNode*_iNumMeshNodesprv;
	num_cols = _iNumDofNode*_iNumMeshNodes;

	if(thread_id < _iNumDofNode*_iNumMeshNodesgpu) {

		for(n=0; n<_inumDiaPart; n++) {

			col = row + off[n];
			//val = K[thread_id + n*_iNumDofNode*_iNumMeshNodesgpu];

			if(col >=0 && col < num_cols)
				dot += K[thread_id + n*num_cols]*D[col];

		}

		Q[thread_id] = dot;

	}

}

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------

void SolveLinearSystem(int Id, int _inumDiaPart, int _iNumDofNode, int BlockSizeX, int BlockMultiProcPerGpu, int _iNumMeshNodes, double *B, double *M, double *K, int *off, double *t, double *CGTime, int *CGIter, double *CGError, double *X, double *R, double *D, double *Q, double *S,
		double *delta_1_aux, double *delta_new, double *dTq, double *delta_old, double *delta_new_h)
{
	int i, _iNumMeshNodesprv=0;
	int GPU_N;

	double TotalTime, time0, time1, time2, time3, time4, time5, time6, time7, time8, time9, time10, time11, time12, time13, time14, time15, time_;
	time0=0.; time1=0.; time2=0.; time3=0.; time4=0.; time5=0.; time6=0.; time7=0.; time8=0.; time9=0.; time10=0.; time11=0.; time12=0.; time13=0.; time14=0.; time15=0.;

	size_t free_byte, total_byte;
	double FreeMem[4], UsedMem[4], TotalMem[4];

    cudaSetDevice(Id);
	int cont=0;
	double epsilon, err;
	double time=0., delta_new_, inc=1;

	// **************  Block and Thread dimesion:  ********************************************************************
	
	// Block size
	dim3 threadsPerBlock(BlockSizeX, BlockSizeX);

	// Block number for local vector operation
	dim3 blocksPerGridVectorLocal(int(sqrt(double(_iNumDofNode*_iNumMeshNodes))/BlockSizeX)+1, int(sqrt(double(_iNumDofNode*_iNumMeshNodes))/BlockSizeX)+1);	
	
	// Thread number => 1D
	dim3 threadsPerBlockMultiProcPerGpu(BlockSizeX*BlockSizeX, 1);
	// Block number: multiple of 32 for sum reduction => 1D
	dim3 blocksPerGridMultiProcPerGpu(BlockMultiProcPerGpu, 1);

	// One thread
	dim3 threadsPerBlockVary(BlockMultiProcPerGpu, 1);
	// One block
	dim3 blocksPerGridOne(1, 1);

	// ------------------------------------------------------------------------------------------------------------------------------------

	// This Kernel clean vector:
	_CLEAN_V <<< blocksPerGridVectorLocal, threadsPerBlock >>> (_iNumDofNode, _iNumMeshNodes, X); // =======================> {x} = {0}
	_CLEAN_V <<< blocksPerGridVectorLocal, threadsPerBlock >>> (_iNumDofNode, _iNumMeshNodes, R); // =======================> {x} = {0}

	// This Kernel Subtracts a Vector by a Vector:
	_SUBT_V_V <<< blocksPerGridVectorLocal, threadsPerBlock >>> (_iNumDofNode, _iNumMeshNodes, _iNumMeshNodesprv, B, R); // =======================> {r} = {b} - [A]{x}

	 // This Kernel Multiplies a Diagonal Matrix by a Vector:
	_MULT_M_V <<< blocksPerGridVectorLocal, threadsPerBlock >>> (_iNumDofNode, _iNumMeshNodes, _iNumMeshNodesprv, M, R, D); // =======================> {d} = [M]-1{r}

	//This Kernel Performs a Dot Product:
	_DOT_V_V <<< blocksPerGridMultiProcPerGpu, threadsPerBlockMultiProcPerGpu >>> (_iNumDofNode, _iNumMeshNodes, R, D, delta_1_aux); // =======================> delta_new = {r}T{d}
	_Sum_DOT_V_V(blocksPerGridOne, threadsPerBlockVary, delta_1_aux, delta_new);

	cudaMemcpy(delta_new_h, delta_new, sizeof(double), cudaMemcpyDeviceToHost);
	delta_new_ = delta_new_h[0];
	epsilon  = 0.0000001;
	err = (double)(delta_new_ * epsilon * epsilon) ;

	// ----------------------------------------------------------------------------------------------------------------

	cont = 0;
	TotalTime = clock();

	while(delta_new_ > err && cont < 4000) {

		//time = clock();
		// This Kernel Multiplies a Sparse Diagonal Format Matrix by a Vector:
		_MULT_K_V <<< blocksPerGridVectorLocal, threadsPerBlock >>> (_iNumDofNode, _iNumMeshNodes, _iNumMeshNodes, _iNumMeshNodesprv, _inumDiaPart, off, K, D, Q);
		//cudaDeviceSynchronize();
		//time0 += (clock()-time)/CLOCKS_PER_SEC;
		//This Kernel Performs a Dot Product:
		_DOT_V_V <<< blocksPerGridMultiProcPerGpu, threadsPerBlockMultiProcPerGpu >>> (_iNumDofNode, _iNumMeshNodes, D, Q, delta_1_aux);
		_Sum_DOT_V_V(blocksPerGridOne, threadsPerBlockVary, delta_1_aux, dTq);
				
		//time = clock();
		_ADD_V_V_X_ <<< blocksPerGridVectorLocal, threadsPerBlock >>> (_iNumDofNode, _iNumMeshNodes, D, Q, delta_new, dTq, X, R); 
		//cudaDeviceSynchronize();
		//time1 += (clock()-time)/CLOCKS_PER_SEC;

		_MULT_M_V <<< blocksPerGridVectorLocal, threadsPerBlock >>> (_iNumDofNode, _iNumMeshNodes, _iNumMeshNodesprv, M, R, S); 
		
		_UPDATE_DELTA_ <<< 1, 1 >>> (delta_new, delta_old);
		
		_DOT_V_V <<< blocksPerGridMultiProcPerGpu, threadsPerBlockMultiProcPerGpu >>> (_iNumDofNode, _iNumMeshNodes, R, S, delta_1_aux);
		_Sum_DOT_V_V(blocksPerGridOne, threadsPerBlockVary, delta_1_aux, delta_new);
				
		cudaMemcpy(delta_new_h, delta_new, sizeof(double), cudaMemcpyDeviceToHost);
		delta_new_ = delta_new_h[0];
		
		//time = clock();
		_ADD_V_V_D_ <<< blocksPerGridVectorLocal, threadsPerBlock >>> (_iNumDofNode, _iNumMeshNodes, S, D, delta_new, delta_old); 
		//cudaDeviceSynchronize();
		//time2 += (clock()-time)/CLOCKS_PER_SEC;
					
		cont++;

	}  // ---------------------------------------------------------------------------------------------------------------------------------


	printf("         Time = %0.3f s / error = %f / iteration = %d\n", (clock()-TotalTime)/CLOCKS_PER_SEC, delta_new_, cont);

	t[0]=time0; t[1]=time1; t[2]=time2; t[3]=time3; t[4]=time4; t[5]=time5; t[6]=time6; t[7]=time7; t[8]=time8; t[9]=time9; t[10]=time10; t[11]=time11;
	*CGTime = (clock()-TotalTime)/CLOCKS_PER_SEC;
	*CGIter = cont;
	*CGError = delta_new_;

}


