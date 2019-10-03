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

#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>

//----------------------------------------------------
// External Functions for one GPU Implementation						 
//----------------------------------------------------

extern "C" void EvaluateMmatrix(int Id, int BlockSizeX, int _iNumMeshNodes, int _iNumDofNode, int _inumDiaPart, double *K, double *M);

extern "C" void AssemblyStiffnessMatrixColor(dim3 blocksPerGrid, dim3 threadsPerBlock, int Id, int _iNumMeshNodes, int _iNumMeshElem, int _iNumDofNode, int _iNumElasMat, int numelcolor, int numelcolorprv,
	int *connect, double *coord, double *prop, double *K, int *offsets, int offsets_size);

extern "C" void EvaluateStrainStateColor(dim3 blocksPerGrid, dim3 threadsPerBlock, int Id, int _iNumMeshNodes, int _iNumMeshElem, int _iNumDofNode, int _iNumElasMat, int numelcolor, int numelcolorprv,
	int *connect, double *coord, double *U, double *strain);

extern "C" void EvaluateStressState(int Id, int BlockSizeX, int _iNumMeshElem, int _iNumElasMat, int *connect, int *LinkMeshColor, double *prop, double *strain, double *stress);

extern "C" void EvaluateNodalForceColor(dim3 blocksPerGrid, dim3 threadsPerBlock, int Id, int _iNumMeshNodes, int _iNumMeshElem, int _iNumDofNode, int _iNumElasMat, int numelcolornodalforce, int numelcolornodalforceprv,
			int *connect, double *coord, int *LinkMeshMeshColor, int *LinkMeshCellColor, double *dP, double *B);

//=============================================================================

__device__ void SolidMatPropMatrix(double E, double p, double C[6][6])
{
	double sho;
	// Solid material matrix

	sho=(E*(1-p))/((1+p)*(1-2*p));

	C[0][0] = sho;
	C[0][1] = sho*(p/(1-p));
	C[0][2] = sho*(p/(1-p));
	C[0][3] = 0.;
	C[0][4] = 0.; 
	C[0][5] = 0.; 

	C[1][0] = sho*(p/(1-p));
	C[1][1] = sho;
	C[1][2] = sho*(p/(1-p));
	C[1][3] = 0.;
	C[1][4] = 0.; 
	C[1][5] = 0.; 

	C[2][0] = sho*(p/(1-p));
	C[2][1] = sho*(p/(1-p));
	C[2][2] = sho;
	C[2][3] = 0.;
	C[2][4] = 0.; 
	C[2][5] = 0.; 

	C[3][0] = 0.; 
	C[3][1] = 0.; 
	C[3][2] = 0.; 
	C[3][3] = sho*((1-2*p)/(2*(1-p)));
	C[3][4] = 0.; 
	C[3][5] = 0.; 

	C[4][0] = 0.; 
	C[4][1] = 0.; 
	C[4][2] = 0.; 
	C[4][3] = 0.; 
	C[4][4] = sho*((1-2*p)/(2*(1-p)));
	C[4][5] = 0.; 

	C[5][0] = 0.; 
	C[5][1] = 0.; 
	C[5][2] = 0.; 
	C[5][3] = 0.; 
	C[5][4] = 0.; 
	C[5][5] = sho*((1-2*p)/(2*(1-p)));

}

//=============================================================================
__device__ void DerivPhiRST(double r, double s, double t, double phi_r[8], double phi_s[8], double phi_t[8])
{
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
}

//=============================================================================
__device__ void Jacobian(double phi_r[8], double phi_s[8], double phi_t[8], double jac[3][3],
	double X[8], double Y[8], double Z[8], double &detjac, double invjac[3][3])
{
	int i, j;

	for(i=0; i<3; i++)
		for(j=0; j<3; j++)
			jac[i][j] = 0.;

	// Calculate the jacobian matrix by appropriately multiplying local
	// derivatives by nodal coords  

	for(i=0; i<8; i++)
	{
		jac[0][0] += phi_r[i] * X[i];
		jac[0][1] += phi_r[i] * Y[i];
		jac[0][2] += phi_r[i] * Z[i];

		jac[1][0] += phi_s[i] * X[i];
		jac[1][1] += phi_s[i] * Y[i];
		jac[1][2] += phi_s[i] * Z[i];

		jac[2][0] += phi_t[i] * X[i];
		jac[2][1] += phi_t[i] * Y[i];
		jac[2][2] += phi_t[i] * Z[i];
	}

	// Jacob determinant
	detjac = jac[0][0]*(jac[1][1]*jac[2][2]-jac[1][2]*jac[2][1])-jac[0][1]*(jac[1][0]*jac[2][2]-jac[1][2]*jac[2][0])+jac[0][2]*(jac[1][0]*jac[2][1]-jac[1][1]*jac[2][0]);	

	// Inverse Jacob
	invjac[0][0] = (jac[1][1]*jac[2][2] - jac[1][2]*jac[2][1]) /detjac;
	invjac[0][1] =-(jac[0][1]*jac[2][2] - jac[0][2]*jac[2][1]) /detjac;
	invjac[0][2] = (jac[0][1]*jac[1][2] - jac[0][2]*jac[1][1]) /detjac;

	invjac[1][0] =-(jac[1][0]*jac[2][2] - jac[1][2]*jac[2][0]) /detjac;
	invjac[1][1] = (jac[0][0]*jac[2][2] - jac[0][2]*jac[2][0]) /detjac;
	invjac[1][2] =-(jac[0][0]*jac[1][2] - jac[0][2]*jac[1][0]) /detjac;

	invjac[2][0] = (jac[1][0]*jac[2][1] - jac[1][1]*jac[2][0]) /detjac;
	invjac[2][1] =-(jac[0][0]*jac[2][1] - jac[0][1]*jac[2][0]) /detjac;
	invjac[2][2] = (jac[0][0]*jac[1][1] - jac[0][1]*jac[1][0]) /detjac;

}

//=============================================================================
__device__ void DerivXYZ(double invjac[3][3], double phi_r[8], double phi_s[8], double phi_t[8], 
	double deriv_x[8], double deriv_y[8], double deriv_z[8])
{
	int i;

	for(i=0; i<8; i++) { 

		deriv_x[i] = invjac[0][0]*phi_r[i]+invjac[0][1]*phi_s[i]+invjac[0][2]*phi_t[i];
		deriv_y[i] = invjac[1][0]*phi_r[i]+invjac[1][1]*phi_s[i]+invjac[1][2]*phi_t[i];
		deriv_z[i] = invjac[2][0]*phi_r[i]+invjac[2][1]*phi_s[i]+invjac[2][2]*phi_t[i];

	}

}

//=============================================================================
__device__ void Bmatrix(double deriv_x[8], double deriv_y[8], double deriv_z[8], double B[6][24])
{
	int i;

	for(i=0; i<8; i++) { 

		B[0][3*i  ] = deriv_x[i];
		B[3][3*i  ] = deriv_y[i];
		B[5][3*i  ] = deriv_z[i];

		B[1][3*i+1] = deriv_y[i];
		B[3][3*i+1] = deriv_x[i];
		B[4][3*i+1] = deriv_z[i];

		B[2][3*i+2] = deriv_z[i];
		B[4][3*i+2] = deriv_y[i];
		B[5][3*i+2] = deriv_x[i];

	}

}

//=============================================================================


//=============================================================================
__device__ void AssemblyK(double coeff, double C[6][6], double B[6][24], double _k[24][24])
{
	int i, j, k;
	double soma, aux[6][24];

	for(i=0; i<6; i++) { 

		for(j=0; j<24; j++) {  

			aux[i][j] = 0.;

			for(k=0; k<6; k++) {

				aux[i][j] += C[i][k]*B[k][j];

			}

		}

	}

	// ------------------------------------------------------------------------

	for(i=0; i<24; i++) {  

		for(j=0; j<24; j++) {  

			soma=0.;

			for(k=0; k<6; k++)
				soma += B[k][i]*aux[k][j];

			_k[i][j] += coeff*soma;

		}

	}

}

//==============================================================================
__global__ void EvaluateMmatrixKernel(int _iNumMeshNodes, int _iNumDofNode, int _inumDiaPart, double *K, double *M)
{
	int off;
	const int xIndex = blockIdx.x*blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y*blockDim.y + threadIdx.y;
	const int thread_id = (gridDim.x*blockDim.x)*yIndex + xIndex;

	off = (_inumDiaPart-1)/2;

	if(thread_id < _iNumDofNode*_iNumMeshNodes) {  // -----------------------

		if(K[thread_id + off*_iNumDofNode*_iNumMeshNodes] != 0.)
			M[thread_id] = 1/K[thread_id + off*_iNumDofNode*_iNumMeshNodes];
		else
			M[thread_id] = 0.;

	}

}

//=====================================================================================================================
void EvaluateMmatrix(int Id, int BlockSizeX, int _iNumMeshNodes, int _iNumDofNode, int _inumDiaPart, double *K, double *M)
{
	double time;

	cudaSetDevice(Id);

	dim3 threadsPerBlock(BlockSizeX, BlockSizeX);
	dim3 blocksPerGrid(int(sqrt(double(_iNumDofNode*_iNumMeshNodes))/BlockSizeX)+1, int(sqrt(double(_iNumDofNode*_iNumMeshNodes))/BlockSizeX)+1);	

	time = clock();

	EvaluateMmatrixKernel<<<blocksPerGrid, threadsPerBlock>>>(_iNumMeshNodes, _iNumDofNode, _inumDiaPart, K, M);

	cudaDeviceSynchronize();

	printf("         Time Execution : %0.3f s \n", (clock()-time)/CLOCKS_PER_SEC);

}



//==============================================================================
__global__ void AssemblyStiffnessMatrixKernel(int numno, int numel, int numelcolor, int numelcolorprv, int numdof, int nummat, int *connect, double *coord, double *prop, double *K, int *offsets, int offsets_size)
{
	double r, s, t;  
	double xgaus[2], wgaus[2], E, p; 
	int no, row, col, off_full, off_part; 
	int ig, jg, kg;  
	int i, j, LM[24];
	double X[8], Y[8], Z[8], C[6][6], phi_r[8], phi_s[8], phi_t[8], jac[3][3], invjac[3][3];
	double detjac, deriv_x[8], deriv_y[8], deriv_z[8], B[6][24], k[24][24];

	offsets_size /= 2;
	const int xIndex = blockIdx.x*blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y*blockDim.y + threadIdx.y;
	const int thread_id = (gridDim.x*blockDim.x)*yIndex + xIndex;

	if(thread_id < numelcolor) {  // ------------------------------------------------------------------------------------

		for(i=0; i<8; i++) {

			no = connect[thread_id + numelcolorprv + (i+3)*numel]-1;
			//if(thread_id==0) printf("%d\n", no);

			X[i] = coord[no];
			Y[i] = coord[no+numno]; 
			Z[i] = coord[no+2*numno];

			//printf("%f %f %f\n", X[i], Y[i], Z[i]);

			LM[3*i]   = 3*no;
			LM[3*i+1] = 3*no+1;
			LM[3*i+2] = 3*no+2;

		}

		// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

		for(i=0; i<24; i++)
			for(j=0; j<24; j++) 
				k[i][j] = 0.;

		for(i=0; i<6; i++)
			for(j=0; j<24; j++)
				B[i][j] = 0.;


		// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


		E = prop[(connect[thread_id + numelcolorprv + numel])-1];
		p = prop[(connect[thread_id + numelcolorprv + numel])-1+nummat];

		SolidMatPropMatrix(E, p, C);

		//printf("%f %f %f %f %f %f %f %f %f %f %f %f\n", C[0][0], C[0][1], C[0][2], C[1][0], C[1][1], C[1][2], C[2][0], C[2][1], C[2][2], C[3][3], C[4][4], C[5][5]);

		// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

		// Points of integration:
		xgaus[0]=-0.577350269189626;
		xgaus[1]= 0.577350269189626;
		wgaus[0]= 1.;
		wgaus[1]= 1.;

		// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

		for(ig=0; ig<2; ig++) {                                      // =================== Loop ig ===================
			s=xgaus[ig];

			for(jg=0; jg<2; jg++) {                                  // =================== Loop jg ===================
				r=xgaus[jg]; 

				for(kg=0; kg<2; kg++) {                              // =================== Loop kg ===================
					t=xgaus[kg]; 

					// Shape function derivative:
					DerivPhiRST(r, s, t, phi_r, phi_s, phi_t);

					// Evaluate the Jacobian determinant and inverse Jacobian matrix:
					Jacobian(phi_r, phi_s, phi_t, jac, X, Y, Z, detjac, invjac);

					//if(Id == 0) printf("%f %f %f %f %f %f %f %f %f\n", jac[0][0], jac[0][1], jac[0][2], jac[1][0], jac[1][1], jac[1][2], jac[2][0], jac[2][1], jac[2][2]);
					//if(Id == 0)  printf("%f\n", detjac);

					// Evaluate the global derivatives of the shape functions:
					DerivXYZ(invjac, phi_r, phi_s, phi_t, deriv_x, deriv_y, deriv_z);

					//printf("%f %f %f %f %f %f %f %f\n", deriv_z[0], deriv_z[1], deriv_z[2], deriv_z[3], deriv_z[4], deriv_z[5], deriv_z[6], deriv_z[7]);

					// Evaluate the B matrix:
					Bmatrix(deriv_x, deriv_y, deriv_z, B);

					// Assembly the ement K matrix:
					AssemblyK(detjac, C, B, k);

				}                                                    // =================== Loop kg ===================

			}                                                        // =================== Loop jg ===================

		}                                                            // =================== Loop ig ===================

		// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

		/*for(i=0; i<24; i++)  
		for(j=0; j<24; j++)
		printf("%f\n", k[i][j]);*/

		// ------------------------------------------------------------------------------------------------------------

		for(i=0; i<24; i++) {  

			row = LM[i];

			off_part = offsets_size;

			for(j=0; j<24; j++) {  

				col = LM[j]; 

				off_full = col - row;

				while (true) {
					if (offsets[off_part] == off_full)
						break;

					if (offsets[off_part] < off_full)
						off_part++;
					else 
						off_part--;
				}

				K[row + off_part*3*numno] += k[i][j];

			}

		}

	}  // -------------------------------------------------------------------------------------------------------------

}

//=====================================================================================================================
void AssemblyStiffnessMatrixColor(dim3 blocksPerGrid, dim3 threadsPerBlock, int Id, int _iNumMeshNodes, int _iNumMeshElem, int _iNumDofNode, int _iNumElasMat, int numelcolor, int numelcolorprv,
	int *connect, double *coord, double *prop, double *K, int *offsets, int offsets_size)
{

	cudaSetDevice(Id);

	AssemblyStiffnessMatrixKernel<<< blocksPerGrid, threadsPerBlock >>> (_iNumMeshNodes, _iNumMeshElem, numelcolor, numelcolorprv, _iNumDofNode, _iNumElasMat, connect, coord, prop, K, offsets, offsets_size);

	cudaDeviceSynchronize();

}

//==============================================================================
__global__ void EvaluateStrainStateKernel(int numno, int numel, int numelcolor, int numelcolorprv, int numdof, int nummat, int *connect, double *coord, double *D, double *strain)
{
	double r, s, t;  
	double xgaus[2], wgaus[2]; 
	int ig, jg, kg;  
	int i, no, cont;
	double X[8], Y[8], Z[8], phi_r[8], phi_s[8], phi_t[8], jac[3][3], invjac[3][3];
	double detjac, deriv_x[8], deriv_y[8], deriv_z[8];

	const int xIndex = blockIdx.x*blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y*blockDim.y + threadIdx.y;
	const int thread_id = (gridDim.x*blockDim.x)*yIndex + xIndex;

	if(thread_id < numelcolor) {  // ------------------------------------------------------------------------------------

		for(i=0; i<8; i++) {

			no = connect[thread_id + numelcolorprv + (i+3)*numel]-1;

			X[i] = coord[no];
			Y[i] = coord[no+numno]; 
			Z[i] = coord[no+2*numno];

		}

		// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

		// Points of integration:
		xgaus[0]=-0.577350269189626;
		xgaus[1]= 0.577350269189626;
		wgaus[0]= 1.;
		wgaus[1]= 1.;

		// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

		cont = 0;

		for(ig=0; ig<2; ig++) {                                      // =================== Loop ig ===================
			s=xgaus[ig];

			for(jg=0; jg<2; jg++) {                                  // =================== Loop jg ===================
				r=xgaus[jg]; 

				for(kg=0; kg<2; kg++) {                              // =================== Loop kg ===================
					t=xgaus[kg]; 

					// Shape function derivative:
					DerivPhiRST(r, s, t, phi_r, phi_s, phi_t);

					// Evaluate the Jacobian determinant and inverse Jacobian matrix:
					Jacobian(phi_r, phi_s, phi_t, jac, X, Y, Z, detjac, invjac);

					// Evaluate the global derivatives of the shape functions:
					DerivXYZ(invjac, phi_r, phi_s, phi_t, deriv_x, deriv_y, deriv_z);

					// *******************************************************************************************************

					for(i=0; i<8; i++) {

						// Exx
						strain[connect[thread_id + numelcolorprv]-1+0*numel+cont*numel*6] += deriv_x[i]*D[3*(connect[thread_id + numelcolorprv + (i+3)*numel]-1)  ];

						// Eyy
						strain[connect[thread_id + numelcolorprv]-1+1*numel+cont*numel*6] += deriv_y[i]*D[3*(connect[thread_id + numelcolorprv + (i+3)*numel]-1)+1];

						// Ezz
						strain[connect[thread_id + numelcolorprv]-1+2*numel+cont*numel*6] += deriv_z[i]*D[3*(connect[thread_id + numelcolorprv + (i+3)*numel]-1)+2];

						// Exy
						strain[connect[thread_id + numelcolorprv]-1+3*numel+cont*numel*6] += deriv_y[i]*D[3*(connect[thread_id + numelcolorprv + (i+3)*numel]-1)  ] + 
							deriv_x[i]*D[3*(connect[thread_id + numelcolorprv + (i+3)*numel]-1)+1];

						// Eyz
						strain[connect[thread_id + numelcolorprv]-1+4*numel+cont*numel*6] += deriv_z[i]*D[3*(connect[thread_id + numelcolorprv + (i+3)*numel]-1)+1] + 
							deriv_y[i]*D[3*(connect[thread_id + numelcolorprv + (i+3)*numel]-1)+2];								  

						// Ezx
						strain[connect[thread_id + numelcolorprv]-1+5*numel+cont*numel*6] += deriv_z[i]*D[3*(connect[thread_id + numelcolorprv + (i+3)*numel]-1)  ] + 
							deriv_x[i]*D[3*(connect[thread_id + numelcolorprv + (i+3)*numel]-1)+2];							                   
					}

					cont++;

				}                                                    // =================== Loop kg ===================

			}                                                        // =================== Loop jg ===================

		}                                                            // =================== Loop ig ===================

	}  // -------------------------------------------------------------------------------------------------------------

}

//==============================================================================
__global__ void EvaluateAverageStrainStateKernel(int numno, int numel, int numelcolor, int numelcolorprv, int numdof, int nummat, int *connect, double *coord, double *D, double *strain)
{
	double r, s, t;  
	double xgaus[2], wgaus[2]; 
	int ig, jg, kg;  
	int i, no;
	double X[8], Y[8], Z[8], phi_r[8], phi_s[8], phi_t[8], jac[3][3], invjac[3][3];
	double detjac, deriv_x[8], deriv_y[8], deriv_z[8];

	const int xIndex = blockIdx.x*blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y*blockDim.y + threadIdx.y;
	const int thread_id = (gridDim.x*blockDim.x)*yIndex + xIndex;

	if(thread_id < numelcolor) {  // ------------------------------------------------------------------------------------

		for(i=0; i<8; i++) {

			no = connect[thread_id + numelcolorprv + (i+3)*numel]-1;

			X[i] = coord[no];
			Y[i] = coord[no+numno]; 
			Z[i] = coord[no+2*numno];

		}

		// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

		// Points of integration:
		xgaus[0]=-0.577350269189626;
		xgaus[1]= 0.577350269189626;
		wgaus[0]= 1.;
		wgaus[1]= 1.;

		// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

		for(ig=0; ig<2; ig++) {                                      // =================== Loop ig ===================
			s=xgaus[ig];

			for(jg=0; jg<2; jg++) {                                  // =================== Loop jg ===================
				r=xgaus[jg]; 

				for(kg=0; kg<2; kg++) {                              // =================== Loop kg ===================
					t=xgaus[kg]; 

					// Shape function derivative:
					DerivPhiRST(r, s, t, phi_r, phi_s, phi_t);

					// Evaluate the Jacobian determinant and inverse Jacobian matrix:
					Jacobian(phi_r, phi_s, phi_t, jac, X, Y, Z, detjac, invjac);

					// Evaluate the global derivatives of the shape functions:
					DerivXYZ(invjac, phi_r, phi_s, phi_t, deriv_x, deriv_y, deriv_z);

					// *******************************************************************************************************

					for(i=0; i<8; i++) {

						// Exx
						strain[connect[thread_id + numelcolorprv]-1+0*numel] += (deriv_x[i]*D[3*(connect[thread_id + numelcolorprv + (i+3)*numel]-1)  ])/8;

						// Eyy
						strain[connect[thread_id + numelcolorprv]-1+1*numel] += (deriv_y[i]*D[3*(connect[thread_id + numelcolorprv + (i+3)*numel]-1)+1])/8;

						// Ezz
						strain[connect[thread_id + numelcolorprv]-1+2*numel] += (deriv_z[i]*D[3*(connect[thread_id + numelcolorprv + (i+3)*numel]-1)+2])/8;

						// Exy
						strain[connect[thread_id + numelcolorprv]-1+3*numel] += (deriv_y[i]*D[3*(connect[thread_id + numelcolorprv + (i+3)*numel]-1)  ] + 
							deriv_x[i]*D[3*(connect[thread_id + numelcolorprv + (i+3)*numel]-1)+1])/8;

						// Eyz
						strain[connect[thread_id + numelcolorprv]-1+4*numel] += (deriv_z[i]*D[3*(connect[thread_id + numelcolorprv + (i+3)*numel]-1)+1] + 
							deriv_y[i]*D[3*(connect[thread_id + numelcolorprv + (i+3)*numel]-1)+2])/8;								  

						// Ezx
						strain[connect[thread_id + numelcolorprv]-1+5*numel] += (deriv_z[i]*D[3*(connect[thread_id + numelcolorprv + (i+3)*numel]-1)  ] + 
							deriv_x[i]*D[3*(connect[thread_id + numelcolorprv + (i+3)*numel]-1)+2])/8;	
													                   
					}

				}                                                    // =================== Loop kg ===================

			}                                                        // =================== Loop jg ===================

		}                                                            // =================== Loop ig ===================

	}  // -------------------------------------------------------------------------------------------------------------

}

//=====================================================================================================================
void EvaluateStrainStateColor(dim3 blocksPerGrid, dim3 threadsPerBlock, int Id, int _iNumMeshNodes, int _iNumMeshElem, int _iNumDofNode, int _iNumElasMat, int numelcolor, int numelcolorprv,
	int *connect, double *coord, double *U, double *strain)
{

	cudaSetDevice(Id);

	//EvaluateStrainStateKernel <<< blocksPerGrid, threadsPerBlock >>> (_iNumMeshNodes, _iNumMeshElem, numelcolor, numelcolorprv, _iNumDofNode, _iNumElasMat, connect, coord, U, strain);
	EvaluateAverageStrainStateKernel <<< blocksPerGrid, threadsPerBlock >>> (_iNumMeshNodes, _iNumMeshElem, numelcolor, numelcolorprv, _iNumDofNode, _iNumElasMat, connect, coord, U, strain);

	cudaDeviceSynchronize();

}

//==============================================================================
__global__ void EvaluateStressStateKernel(int numel, int nummat, int *connect, double *prop, double *strain, double *stress)
{ 

	int i, cont, ig, jg, kg;
	double E, p, C[6][6];
	
	const int xIndex = blockIdx.x*blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y*blockDim.y + threadIdx.y;
	const int thread_id = (gridDim.x*blockDim.x)*yIndex + xIndex;

	if(thread_id < numel) {  // ------------------------------------------------------------------------------------

		E = prop[(connect[thread_id + numel])-1];
		p = prop[(connect[thread_id + numel])-1+nummat];

		SolidMatPropMatrix(E, p, C);

		// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

		cont = 0;

		for(ig=0; ig<2; ig++) {                                      // =================== Loop ig ===================

			for(jg=0; jg<2; jg++) {                                  // =================== Loop jg ===================

				for(kg=0; kg<2; kg++) {                              // =================== Loop kg ===================

					for(i=0; i<6; i++) {

						// Sxx
						stress[thread_id+0*numel+cont*numel*6] += C[0][i]*strain[thread_id+i*numel+cont*numel*6];

						// Syy
						stress[thread_id+1*numel+cont*numel*6] += C[1][i]*strain[thread_id+i*numel+cont*numel*6];

						// Szz
						stress[thread_id+2*numel+cont*numel*6] += C[2][i]*strain[thread_id+i*numel+cont*numel*6];

						// Sxy
						stress[thread_id+3*numel+cont*numel*6] += C[3][i]*strain[thread_id+i*numel+cont*numel*6];

						// Syz
						stress[thread_id+4*numel+cont*numel*6] += C[4][i]*strain[thread_id+i*numel+cont*numel*6];							  

						// Szx
						stress[thread_id+5*numel+cont*numel*6] += C[5][i]*strain[thread_id+i*numel+cont*numel*6];

					}

					cont++;

				}                                                    // =================== Loop kg ===================

			}                                                        // =================== Loop jg ===================

		}                                                            // =================== Loop ig ===================

	}  // -------------------------------------------------------------------------------------------------------------

}

//==============================================================================
__global__ void EvaluateAverageStressStateKernel(int numel, int nummat, int *connect, int *LinkMeshColor, double *prop, double *strain, double *stress)
{ 

	int i;
	double E, p, C[6][6];
	
	const int xIndex = blockIdx.x*blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y*blockDim.y + threadIdx.y;
	const int thread_id = (gridDim.x*blockDim.x)*yIndex + xIndex;

	if(thread_id < numel) {  // ------------------------------------------------------------------------------------

		E = prop[(connect[LinkMeshColor[thread_id] + numel])-1];
		p = prop[(connect[LinkMeshColor[thread_id] + numel])-1+nummat];

		SolidMatPropMatrix(E, p, C);

		// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

		for(i=0; i<6; i++) {

			// Sxx
			stress[thread_id+0*numel] += C[0][i]*strain[thread_id+i*numel];

			// Syy
			stress[thread_id+1*numel] += C[1][i]*strain[thread_id+i*numel];

			// Szz
			stress[thread_id+2*numel] += C[2][i]*strain[thread_id+i*numel];

			// Sxy
			stress[thread_id+3*numel] += C[3][i]*strain[thread_id+i*numel];

			// Syz
			stress[thread_id+4*numel] += C[4][i]*strain[thread_id+i*numel];							  

			// Szx
			stress[thread_id+5*numel] += C[5][i]*strain[thread_id+i*numel];	

		}

	}  // -------------------------------------------------------------------------------------------------------------

}

//=====================================================================================================================
void EvaluateStressState(int Id, int BlockSizeX, int numel, int nummat, int *connect, int *LinkMeshColor, double *prop, double *strain, double *stress)
{
	double time;

	cudaSetDevice(Id);
	cudaMemset(stress, 0, sizeof(double)*numel*6);

	dim3 threadsPerBlock(BlockSizeX, BlockSizeX);
	dim3 blocksPerGrid(int(sqrt(double(numel))/BlockSizeX)+1, int(sqrt(double(numel))/BlockSizeX)+1);

	time = clock();

	EvaluateAverageStressStateKernel <<<blocksPerGrid, threadsPerBlock>>>(numel, nummat, connect, LinkMeshColor, prop, strain, stress);
	//EvaluateStressStateKernel <<<blocksPerGrid, threadsPerBlock>>>(numel, nummat, connect, prop, strain, stress);

	cudaDeviceSynchronize();

	printf("         Time Execution : %0.3f s \n", (clock()-time)/CLOCKS_PER_SEC);

}

//==============================================================================
__global__ void EvaluateNodalForceKernel(int numno, int numel, int numelcolor, int numelcolorprv, int numdof, int nummat, int *connect, double *coord, int *LinkMeshMeshColor, int *LinkMeshCellColor, double *dP, double *B)
{
	double r, s, t;  
	double xgaus[2], wgaus[2]; 
	int ig, jg, kg;  
	int i, no;
	double X[8], Y[8], Z[8], phi_r[8], phi_s[8], phi_t[8], jac[3][3], invjac[3][3];
	double detjac, deriv_x[8], deriv_y[8], deriv_z[8];

	const int xIndex = blockIdx.x*blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y*blockDim.y + threadIdx.y;
	const int thread_id = (gridDim.x*blockDim.x)*yIndex + xIndex;

	if(thread_id < numelcolor) {  // ------------------------------------------------------------------------------------

		for(i=0; i<8; i++) {

			no = connect[LinkMeshMeshColor[thread_id + numelcolorprv] + (i+3)*numel]-1;

			X[i] = coord[no];
			Y[i] = coord[no+numno]; 
			Z[i] = coord[no+2*numno];

		}

		// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

		// Points of integration:
		xgaus[0]=-0.577350269189626;
		xgaus[1]= 0.577350269189626;
		wgaus[0]= 1.;
		wgaus[1]= 1.;

		// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

		for(ig=0; ig<2; ig++) {                                      // =================== Loop ig ===================
			s=xgaus[ig];

			for(jg=0; jg<2; jg++) {                                  // =================== Loop jg ===================
				r=xgaus[jg]; 

				for(kg=0; kg<2; kg++) {                              // =================== Loop kg ===================
					t=xgaus[kg]; 

					// Shape function derivative:
					DerivPhiRST(r, s, t, phi_r, phi_s, phi_t);

					// Evaluate the Jacobian determinant and inverse Jacobian matrix:
					Jacobian(phi_r, phi_s, phi_t, jac, X, Y, Z, detjac, invjac);

					// Evaluate the global derivatives of the shape functions:
					DerivXYZ(invjac, phi_r, phi_s, phi_t, deriv_x, deriv_y, deriv_z);

					// *******************************************************************************************************

					for(i=0; i<8; i++) {

						no = connect[LinkMeshMeshColor[thread_id + numelcolorprv] + (i+3)*numel]-1;

						// Fx
						B[3*no  ] += deriv_x[i]*detjac*dP[LinkMeshCellColor[thread_id + numelcolorprv]];

						// Fy
						B[3*no+1] += deriv_y[i]*detjac*dP[LinkMeshCellColor[thread_id + numelcolorprv]];

						// Fz
						B[3*no+2] += deriv_z[i]*detjac*dP[LinkMeshCellColor[thread_id + numelcolorprv]];
					                   
					}

				}                                                    // =================== Loop kg ===================

			}                                                        // =================== Loop jg ===================

		}                                                            // =================== Loop ig ===================

	}  // -------------------------------------------------------------------------------------------------------------

}

//=====================================================================================================================
void EvaluateNodalForceColor(dim3 blocksPerGrid, dim3 threadsPerBlock, int Id, int _iNumMeshNodes, int _iNumMeshElem, int _iNumDofNode, int _iNumElasMat, int numelcolor, int numelcolorprv,
			int *connect, double *coord, int *LinkMeshMeshColor, int *LinkMeshCellColor, double *dP, double *B)
{

	cudaSetDevice(Id);

	EvaluateNodalForceKernel <<< blocksPerGrid, threadsPerBlock >>> (_iNumMeshNodes, _iNumMeshElem, numelcolor, numelcolorprv, _iNumDofNode, _iNumElasMat, connect, coord, LinkMeshMeshColor, LinkMeshCellColor, dP, B);

	cudaDeviceSynchronize();

}