#include <fstream>
#include <iostream>
#include <string>
#include <strstream>
#include <cmath>
#include <iomanip>
#include <time.h>
#include <stdio.h>
#include <vector>
#include <stdlib.h>
#include "cpu.h"
//#include <cutil_inline.h>

//----------------------------------------------------
// External Functions, forward declarations						 
//----------------------------------------------------

extern "C" int GPU_ReadSize(int &numno, int &numel, int &nummat, int &numprop, int &numdof, int &numnoel, int &numstr, int &numbc, int &numnl,
int &imax, int &jmax, int &kmax, char GPU_arqaux[80]);

int CPU_ReadInPutFile(int numno, int numel, int nummat, int numprop, int numdof, int numnoel, int numstr, int numbc, int numnl, char GPU_arqaux[80],
double *X_h, double *Y_h, double *Z_h, int *BC_h, double *Material_Param_h, int *Connect, int *Material_Id_h);

void CPU_PreProcesssing(int numel, int *Connect, int numnoel, int numdof, int RowSize, int *Vector_I_h);

void CPU_EvaluateNumbering(int ii, int jj, int RowSize, int *Vector_I_h);

void CPU_AssemblyStiffnessHexahedron(int ii, int jj, double v, int RowSize, int *Vector_I_h, double *Matrix_K_h);

void CPU_ElementStiffnessHexahedron(int k, int numnoel, int numel, int numstr, int numdof, double *X_h, double *Y_h, 
double *Z_h, int *Connect, double dE, double dNu, double *Matrix_k_h);

void CPU_ElementStiffnessHexahedronConventional(int k, int numnoel, int numel, int numstr, int numdof, double *X_h, double *Y_h, 
double *Z_h, int *Connect, double dE, double dNu, std::vector< std::vector<double> > &KCPULocalMatrix);

void Grad_Conjugate(int numdof, int numno, int RowSize, double *Matrix_K_h, double *Vector_X_h, double *Vector_B_h, int *Vector_I_h);

void ReadConcLoad(int numno, double *Vector_B_h, int *BC_h);

void WriteDisplacementCPU(int numno, double *Vector_X_h);

void CPU_EvaluateStrainState(int numnoel, int numel, int numstr, int numdof, double *X_h, double *Y_h, double *Z_h, int *Connect,
							 double *Vector_X_h, double *Strain_h);

void EvaluateStressStateCPU(int numel, int nummat, int *Material_Id_h, double *Material_Param_h, double *Strain_h, double *Stress_h);

void WriteStrainStateCPU(int numel, double *Strain_h);

void WriteStressStateCPU(int numel, double *Stress_h);

pcCpu cCpu::Cpu = NULL;

//==============================================================================
cCpu::cCpu()
{





}

//====================================================================================================
int cCpu::AnalyzeCPUNaive()
{
	unsigned int hTimer;
	int i, j, k, j0, j1, j2, RowSize, NumTerms;
	int i1, i2, i3, ii, jj;
	double men_size;
	double Time;
	double dE, dNu;

	double *X_h, *Y_h, *Z_h, *Material_Param_h;
	int *Connect, *Material_Id_h, *BC_h, *Vector_I_h;
	double *Matrix_K_h, *Matrix_k_h, *Vector_X_h, *Vector_B_h, *Strain_h, *Stress_h;

	unsigned int hTimer1, hTimer2;
	double Time1, Time2;

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

	
	char GPU_arqaux[80];

	// ========= Reading vector size =========

	if(GPU_ReadSize(numno, numel, nummat, numprop, numdof, numnoel, numstr, numbc, numnl, imax, jmax, kmax, GPU_arqaux) == 0) {
		printf("Invalid neutral file or label END doesn't exist" );
	}

	// ========= Allocating Vectors =========         

	// Allocating memory space for host global node coordinate
	X_h  = (double *)malloc(numno*(sizeof(double))); 
	Y_h  = (double *)malloc(numno*(sizeof(double))); 
	Z_h  = (double *)malloc(numno*(sizeof(double))); 

	// Allocating memory space for host boundary condition vector
	BC_h = (int *)malloc(numno*(sizeof(int))); 
	for(i=0; i<numno; i++) 
		BC_h[i] = 0;

	// Allocating memory space for host material parameters vector
	Material_Param_h = (double *)malloc(nummat*numprop*(sizeof(double))); 

	// Allocating memory space for host element conectivity vector

	Material_Id_h = (int *)malloc(numel*(sizeof(int)));  
	Connect  = (int *)malloc(numel*numnoel*(sizeof(int)));  // numnoel = Number of nodes per element 

	// ========= Reading input file =========

	printf("\n\n");
	printf("         Reading input data file \n");
	printf("         ========================================= ");

	//cutCreateTimer(&hTimer1);
	//cudaThreadSynchronize();
	//cutResetTimer(hTimer1);
	//cutStartTimer(hTimer1);

	CPU_ReadInPutFile(numno, numel, nummat, numprop, numdof, numnoel, numstr, numbc, numnl, GPU_arqaux, X_h, Y_h, Z_h, BC_h, Material_Param_h, Connect, Material_Id_h);

	//cudaThreadSynchronize();
	//cutStopTimer(hTimer1);
//	Time1 = cutGetTimerValue(hTimer1);

	printf("\n");
	printf("         Input File Reading Time: %0.3f s \n", Time1/CLOCKS_PER_SEC);

	numdofel = numdof*numnoel;  // Number of DOF per element

	// Maximum row size for hexahedral element:
	RowSize = 81;   // 3D Problem = Total number of terms per line is 81 (hexahedrol element)

	// Stiffness Matrix - Allocating Memory Space:
	NumTerms = numdof*numno*RowSize;      // Size of the stiffness matriz  
										   // numdof = Number of freedom of degree (DOF) per node
	Matrix_K_h = (double *)malloc(NumTerms*(sizeof(double)));
	Matrix_k_h = (double *)malloc(numnoel*numdof*numnoel*numdof*(sizeof(double)));
	Vector_I_h = (int *)malloc(NumTerms*(sizeof(int)));
	Vector_X_h = (double *)malloc(numdof*numno*(sizeof(double)));
	Vector_B_h = (double *)malloc(numdof*numno*(sizeof(double)));

	Strain_h = (double *)malloc(numel*numstr*8*(sizeof(double))); // 8 number of integration points 
	Stress_h = (double *)malloc(numel*numstr*8*(sizeof(double))); // 8 number of integration points 

	// Cleaning vectors
	for(i=0; i<NumTerms; i++) {
		Matrix_K_h[i] = 0.;
		Vector_I_h[i] = 0;
	}

	for(i=0; i<numdof*numno; i++) {
		Vector_X_h[i] = 0.;
		Vector_B_h[i] = 0.;
	}

	for(i=0; i<numel*numstr*8; i++) {
		Strain_h[i] = 0.;
		Stress_h[i] = 0.;
	}

	// ========= CPU Preprocessing =========

	CPU_PreProcesssing(numel, Connect, numnoel, numdof, RowSize, Vector_I_h);        // numnoel = Number of nodes per element

	// ========= Assembling CPU Stiffness Matrix =========

	std::vector<int> LM;            // Local - Global relation

	LM.resize(numdof*numnoel);

	typedef std::vector<double>  ijId;
	typedef std::vector<ijId>  ijMatrix;
	ijMatrix  KCPUGlobalMatrix, KCPULocalMatrix;

	

	KCPUGlobalMatrix.resize(numdof*numno);
	for(i=0; i<numdof*numno; i++) 
		KCPUGlobalMatrix[i].resize(numdof*numno);

	KCPULocalMatrix.resize(numdof*numnoel);
	for(i=0; i<numdof*numnoel; i++) 
		KCPULocalMatrix[i].resize(numdof*numnoel);
			
	int compact = 0;

	if(compact == 0) {

		printf("\n");
		printf("         Assembling Stiffness Matrix \n");
		printf("         ========================================= ");   

//		cutCreateTimer(&hTimer2);
//		cudaThreadSynchronize();
//		cutResetTimer(hTimer2);
//		cutStartTimer(hTimer2);

		for(k=0; k<numel; k++) {

			dE  = Material_Param_h[ (Material_Id_h[k]-1) ];
			dNu = Material_Param_h[ (Material_Id_h[k]-1) + nummat ];

			// ========= Evaluate Element Stiffness Matrix ========= 

			CPU_ElementStiffnessHexahedron(k, numnoel, numel, numstr, numdof, X_h, Y_h, Z_h, Connect, dE, dNu, Matrix_k_h);

			for(i=0; i<numnoel; i++) {
				i1 = 3*i;
				i2 = 3*i+1;
				i3 = 3*i+2;
				LM[i1] = 3*(Connect[k + i*numel]-1)  ;
				LM[i2] = 3*(Connect[k + i*numel]-1)+1;
				LM[i3] = 3*(Connect[k + i*numel]-1)+2;
			}

			// ========= Assembly Global Stiffness Matrix =========

			for(i=0; i<numdof*numnoel; i++) {

				ii = LM[i];

				for(j=0; j<numdof*numnoel; j++) {  // numnoel = Number of nodes per element
					// numdof  = Number of freedom of degree (DOF) per node
					jj = LM[j];    

					CPU_AssemblyStiffnessHexahedron(ii, jj, Matrix_k_h[i*numdof*numnoel + j], RowSize, Vector_I_h, Matrix_K_h);

				}

			}

		}

		// ========= Apply Boundary Condition =========

		for(j=0; j<numno; j++) {

			if(BC_h[j] == 1) {

				j0 = 3*j;
				j1 = 3*j + 1;
				j2 = 3*j + 2;

				for(i=0; i<RowSize; i++) {

					Matrix_K_h[j0*RowSize + i] = 0.;
					Matrix_K_h[j1*RowSize + i] = 0.;
					Matrix_K_h[j2*RowSize + i] = 0.;

				}

			}

		}

//		cudaThreadSynchronize();
//		cutStopTimer(hTimer2);
//		Time2 = cutGetTimerValue(hTimer2);

		printf("\n");
		printf("         Assembly Stiffness Matrix Time: %0.3f s \n", Time2/CLOCKS_PER_SEC);

	}
	else {

		//int auxnumel = 510;

		for(k=0; k<numel; k++) {

			dE  = Material_Param_h[ (Material_Id_h[k]-1) ];
			dNu = Material_Param_h[ (Material_Id_h[k]-1) + nummat ];

			// ========= Evaluate Element Stiffness Matrix ========= 

			CPU_ElementStiffnessHexahedronConventional(k, numnoel, numel, numstr, numdof, X_h, Y_h, Z_h, Connect, dE, dNu, KCPULocalMatrix);

			for(i=0; i<numnoel; i++) {
				i1 = 3*i;
				i2 = 3*i+1;
				i3 = 3*i+2;
				LM[i1] = 3*(Connect[k + i*numel]-1)  ;
				LM[i2] = 3*(Connect[k + i*numel]-1)+1;
				LM[i3] = 3*(Connect[k + i*numel]-1)+2;
			}

			// ========= Assembly Global Stiffness Matrix =========

			for(j=0; j<numdof*numnoel; j++) {  // numnoel = Number of nodes per element
				// numdof  = Number of freedom of degree (DOF) per node

				jj = LM[j];

				for(i=0; i<numdof*numnoel; i++) {  

					ii = LM[i];

					KCPUGlobalMatrix[ii][jj] += KCPULocalMatrix[i][j];

				}

			}

		}

		// ========= Apply Boundary Condition =========

		for(j=0; j<numno; j++) {

			if(BC_h[j] == 1) {

				j0 = 3*j;
				j1 = 3*j + 1;
				j2 = 3*j + 2;

				for(i=0; i<3*numno; i++) {

					KCPUGlobalMatrix[j0][i] = 0.;
					KCPUGlobalMatrix[j1][i] = 0.;
					KCPUGlobalMatrix[j2][i] = 0.;

				}

			}

		}

	}

	

	// Assembly the stiffness matrix for Hexahedron Element on GPU:

	ReadConcLoad(numno, Vector_B_h, BC_h);

	// ========= Solve Linear System Equation - Gradient Conjugate Method =========

	Grad_Conjugate(numdof, numno, RowSize, Matrix_K_h, Vector_X_h, Vector_B_h, Vector_I_h);

	// ========= Write Dispalcement Field  in output file CPUDisplacement.pos =========

	WriteDisplacementCPU(numno, Vector_X_h);

	// ========= Evaluate Strain State =========

	CPU_EvaluateStrainState(numnoel, numel, numstr, numdof, X_h, Y_h, Z_h, Connect, Vector_X_h, Strain_h);

	// ========= Write Strain state in output file CPUStrainState.pos =========

	WriteStrainStateCPU(numel, Strain_h);

	// ========= Evaluate Stress State =========

	EvaluateStressStateCPU(numel, nummat, Material_Id_h, Material_Param_h, Strain_h, Stress_h);

	// ========= Write Stress state in output file CPUStressState.pos =========

	WriteStressStateCPU(numel, Stress_h);

	//=============== x =============== x =============== x =============== x =============== x =============== x ===============


	//std::vector<sNodeCoord> coord;  // Current element Nodes coordinates
	//std::vector<int> connect;       // Element conectivity
	//std::vector<int> LM;            // Local - Global relation 

 	/*numdof = Model->NumDofNode();          // Number of freedom of degree (DOF) per node
	numnoel = Elms[0]->Shp->MapNodeNo;    // Number of nodes per element
	numno = Node->NodeNo;               // Number of nodes of the mesh

	numdofel = numdof*numnoel;  // Number of DOF per element
	num_dof_mesh = numdof*numno;  // Number of DOF of the mesh
	num_stre_comp = Model->NumStressComp();      // Number of stress components

	coord.resize(numnoel);
	connect.resize(numnoel);
	LM.resize(numdofel);

	numel = Elms.size();  // Number of elements on mesh

	men_size = (num_dof_mesh*128*sizeof(double))/1073741824;  // in GB

	if(men_size > 2) {
		printf("Matrix Stiffness memory size overcomes 2 GB."); 
		//break;
	}
	//else  
		//continue;


	// Get memory for global stiffness matrix
	/*GlobMatrix.resize(num_dof_mesh);
	for(i=0; i<num_dof_mesh; i++)
		GlobMatrix[i].resize(num_dof_mesh);

 	for(i=0; i<num_dof_mesh; i++)  
		for(j=0; j<num_dof_mesh; j++)  	
			GlobMatrix[i][j] = 0.;

	// Get memory for element stiffness matrix
	ElemMatrix.resize(numdofel);
	for(i=0; i<numdofel; i++)
		ElemMatrix[i].resize(numdofel);

	ElemMatrix = new (double *[numdofel]);
	for ( i=0; i<numdofel; i++ )
		ElemMatrix[i] = new double [numdofel];

	GblMatrix.resize(num_dof_mesh);
	for(i=0; i<num_dof_mesh; i++)
		GblMatrix[i].resize(num_dof_mesh);

	cutCreateTimer(&hTimer);
	cutResetTimer(hTimer);
	cutStartTimer(hTimer);

	int compact = 0;

	if(compact == 1) {

		for(k=0; k<numel; k++) {

			Elms[k]->Shp->NodalCoord(coord);

			Mats[(Elms[k]->Mat->Id)-1]->GetIsotropicMat(dE, dNu);

			AssemblyStiffnessCPUHexahedron(coord, ElemMatrix, dE, dNu);
			//AssemblyStiffnessCPUTetrahedron(coord, ElemMatrix);

			// Get element connectivity from element class
			Elms[k]->Shp->Connectivity(connect);

			for(i=0; i<numnoel; i++) {
				i1 = 3*i+2;
				i2 = 3*i+1;
				i3 = 3*i;
				LM[i1] = 3*(connect[i]-1)+2;
				LM[i2] = 3*(connect[i]-1)+1;
				LM[i3] = 3*(connect[i]-1);
			}

			for(i=0; i<numdofel; i++) {
				ii = LM[i];
				for(j=0; j<numdofel; j++) {
					jj = LM[j];
					//GlobMatrix[ii][jj] += ElemMatrix[i][j];
					Add(ii, jj, ElemMatrix[i][j]);
				}
			}
		}

	}
	else {

		for(k=0; k<numel; k++) {

			Elms[k]->Shp->NodalCoord(coord);

			Mats[(Elms[k]->Mat->Id)-1]->GetIsotropicMat(dE, dNu);

			AssemblyStiffnessCPUHexahedron(coord, ElemMatrix, dE, dNu);
			//AssemblyStiffnessCPUTetrahedron(coord, ElemMatrix);

			// Get element connectivity from element class
			Elms[k]->Shp->Connectivity(connect);

			for(i=0; i<numnoel; i++) {
				i1 = 3*i+2; 
				i2 = 3*i+1;
				i3 = 3*i;
				LM[i1] = 3*(connect[i]-1)+2;
				LM[i2] = 3*(connect[i]-1)+1;
				LM[i3] = 3*(connect[i]-1);
			}

			for(i=0; i<numdofel; i++) {
				ii = LM[i];

				for(j=0; j<numdofel; j++) {
					jj = LM[j];

					GblMatrix[ii][jj] += ElemMatrix[i][j];

				}
			}
		}

	}*/

	//cutStopTimer(&hTimer);
	//Time = cutGetTimerValue(hTimer);
	
	//printf("\n");
	//printf("         CPU processing time: %f ms \n", Time);

	/*
	using namespace std;
	ofstream outfile;
	std::string fileName2 = "StiffnessMatrixCPU";
	fileName2 += ".dat";
	outfile.open(fileName2.c_str());
	using std::ios;
	using std::setw;
	outfile.setf(ios::fixed,ios::floatfield);
	outfile.precision(2);

	if(compact == 0) {

		for(i=0; i<numdof*numno; i++) {

			outfile << "line = " << i << endl;
			for(j=0; j<RowSize; j++) {
				if(Matrix_K_h[i*RowSize + j] != 0.) {
					if(fabs(Matrix_K_h[i*RowSize + j]) > 0.1) outfile << Matrix_K_h[i*RowSize + j] << " ";
				}
			}

			outfile << endl; outfile << endl;

		}

	}
	else {

		for(i=0; i<numdof*numno; i++) {

			outfile << "line = " << i << endl;

			for(j=0; j<numdof*numno; j++) {

				if(fabs(KCPUGlobalMatrix[i][j])>0.1) outfile << KCPUGlobalMatrix[i][j] << " ";

			}

			outfile << endl; outfile << endl;

		}

	}

	outfile.close();
	*/

	return 1;

}

//====================================================================================================
void EvaluateStressStateCPU(int numel, int nummat, int *Material_Id_h, double *Material_Param_h, double *Strain_h, double *Stress_h)
{
	int i, l, ig, jg, kg, cont;
	double E, p, sho, De[6][6];
	unsigned int hTimer;
	double Time;

	printf("\n\n");
	printf("         Evaluating Stress State \n");
	printf("         ========================================= ");

//	cutCreateTimer(&hTimer);
//	cudaThreadSynchronize();
//	cutResetTimer(hTimer);
//	cutStartTimer(hTimer);

	for(l=0; l<numel; l++) {        // numel = Number of elements on mesh

		E = Material_Param_h[ (Material_Id_h[l]-1) ];
		p = Material_Param_h[ (Material_Id_h[l]-1) + nummat ];

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
						Stress_h[l+0*numel+cont*numel*6] += De[0][i]*Strain_h[l+i*numel+cont*numel*6];

						// Syy
						Stress_h[l+1*numel+cont*numel*6] += De[1][i]*Strain_h[l+i*numel+cont*numel*6];

						// Szz
						Stress_h[l+2*numel+cont*numel*6] += De[2][i]*Strain_h[l+i*numel+cont*numel*6];

						// Sxy
						Stress_h[l+3*numel+cont*numel*6] += De[3][i]*Strain_h[l+i*numel+cont*numel*6];

						// Syz
						Stress_h[l+4*numel+cont*numel*6] += De[4][i]*Strain_h[l+i*numel+cont*numel*6];							  

						// Szx
						Stress_h[l+5*numel+cont*numel*6] += De[5][i]*Strain_h[l+i*numel+cont*numel*6];

					}

					cont++;

				}

			}                                                    // =================== Loop kg ===================

		}                                                        // =================== Loop jg ===================

	}                                                            // =================== Loop ig ===================

//	cudaThreadSynchronize();
//	cutStopTimer(hTimer);
//	Time = cutGetTimerValue(hTimer);
	
	printf("\n");
	printf("         Evaluating Stress State Time: %0.3f s \n", Time/CLOCKS_PER_SEC);

}      

//====================================================================================================
void CPU_EvaluateStrainState(int numnoel, int numel, int numstr, int numdof, double *X_h, double *Y_h, double *Z_h, int *Connect,
							 double *Vector_X_h, double *Strain_h)
{
	int i, j, k, l, m, n, ig, jg, kg, cont;
	double jacob, soma;
	double r, s, t;
	double xgaus[2], wgaus[2], De[6][6], gama[3][3];
	double phi_r[8], phi_s[8], phi_t[8];
	double jac[3][3], B[6][24];
	double auxX[8], auxY[8], auxZ[8], aux[6][24];	
	double E, p, sho;
	unsigned int hTimer;
	double Time;

	double *XEL, *YEL, *ZEL;

	XEL = (double *)malloc(numnoel*(sizeof(double)));
	YEL = (double *)malloc(numnoel*(sizeof(double)));
	ZEL = (double *)malloc(numnoel*(sizeof(double)));

	printf("\n");
	printf("         Evaluating Strain State \n");
	printf("         ========================================= ");

//	cutCreateTimer(&hTimer);
	//cudaThreadSynchronize();
	//cutResetTimer(hTimer);
	//cutStartTimer(hTimer);

	for(l=0; l<numel; l++) {        // numel = Number of elements on mesh

		cont = 0;

		for(i=0; i<numnoel; i++) {  // numnoel = Number of nodes per element

			XEL[i] = X_h[(Connect[l + i*numel]-1)];
			YEL[i] = Y_h[(Connect[l + i*numel]-1)];
			ZEL[i] = Z_h[(Connect[l + i*numel]-1)];

		}

		xgaus[0]=-0.577350269189626;
		xgaus[1]= 0.577350269189626;
		wgaus[0]= 1.;
		wgaus[1]= 1.;

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

					jac[0][0]=phi_r[0]*XEL[0]+phi_r[1]*XEL[1]+phi_r[2]*XEL[2]+phi_r[3]*XEL[3]+phi_r[4]*XEL[4]+phi_r[5]*XEL[5]+phi_r[6]*XEL[6]+phi_r[7]*XEL[7];
					jac[0][1]=phi_r[0]*YEL[0]+phi_r[1]*YEL[1]+phi_r[2]*YEL[2]+phi_r[3]*YEL[3]+phi_r[4]*YEL[4]+phi_r[5]*YEL[5]+phi_r[6]*YEL[6]+phi_r[7]*YEL[7];
					jac[0][2]=phi_r[0]*ZEL[0]+phi_r[1]*ZEL[1]+phi_r[2]*ZEL[2]+phi_r[3]*ZEL[3]+phi_r[4]*ZEL[4]+phi_r[5]*ZEL[5]+phi_r[6]*ZEL[6]+phi_r[7]*ZEL[7];

					jac[1][0]=phi_s[0]*XEL[0]+phi_s[1]*XEL[1]+phi_s[2]*XEL[2]+phi_s[3]*XEL[3]+phi_s[4]*XEL[4]+phi_s[5]*XEL[5]+phi_s[6]*XEL[6]+phi_s[7]*XEL[7];
					jac[1][1]=phi_s[0]*YEL[0]+phi_s[1]*YEL[1]+phi_s[2]*YEL[2]+phi_s[3]*YEL[3]+phi_s[4]*YEL[4]+phi_s[5]*YEL[5]+phi_s[6]*YEL[6]+phi_s[7]*YEL[7];
					jac[1][2]=phi_s[0]*ZEL[0]+phi_s[1]*ZEL[1]+phi_s[2]*ZEL[2]+phi_s[3]*ZEL[3]+phi_s[4]*ZEL[4]+phi_s[5]*ZEL[5]+phi_s[6]*ZEL[6]+phi_s[7]*ZEL[7];

					jac[2][0]=phi_t[0]*XEL[0]+phi_t[1]*XEL[1]+phi_t[2]*XEL[2]+phi_t[3]*XEL[3]+phi_t[4]*XEL[4]+phi_t[5]*XEL[5]+phi_t[6]*XEL[6]+phi_t[7]*XEL[7];
					jac[2][1]=phi_t[0]*YEL[0]+phi_t[1]*YEL[1]+phi_t[2]*YEL[2]+phi_t[3]*YEL[3]+phi_t[4]*YEL[4]+phi_t[5]*YEL[5]+phi_t[6]*YEL[6]+phi_t[7]*YEL[7];
					jac[2][2]=phi_t[0]*ZEL[0]+phi_t[1]*ZEL[1]+phi_t[2]*ZEL[2]+phi_t[3]*ZEL[3]+phi_t[4]*ZEL[4]+phi_t[5]*ZEL[5]+phi_t[6]*ZEL[6]+phi_t[7]*ZEL[7];

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

						for(i=0; i<numnoel; i++) { 

							auxX[i] = gama[0][0]*phi_r[i]+gama[0][1]*phi_s[i]+gama[0][2]*phi_t[i];
							auxY[i] = gama[1][0]*phi_r[i]+gama[1][1]*phi_s[i]+gama[1][2]*phi_t[i];
							auxZ[i] = gama[2][0]*phi_r[i]+gama[2][1]*phi_s[i]+gama[2][2]*phi_t[i];

						}

						for(i=0; i<numnoel; i++) {

							// Exx:
							Strain_h[l + 0*numel + cont*numstr*numel] += auxX[i]*Vector_X_h[3*(Connect[l + i*numel]-1)  ];

							// Eyy
							Strain_h[l + 1*numel + cont*numstr*numel] += auxY[i]*Vector_X_h[3*(Connect[l + i*numel]-1)+1];

							// Ezz
							Strain_h[l + 2*numel + cont*numstr*numel] += auxZ[i]*Vector_X_h[3*(Connect[l + i*numel]-1)+2];

							// Exy:
							Strain_h[l + 3*numel + cont*numstr*numel] += auxY[i]*Vector_X_h[3*(Connect[l + i*numel]-1)  ] + auxX[i]*Vector_X_h[3*(Connect[l + i*numel]-1)+1];

							// Eyz:
							Strain_h[l + 4*numel + cont*numstr*numel] += auxZ[i]*Vector_X_h[3*(Connect[l + i*numel]-1)+1] + auxY[i]*Vector_X_h[3*(Connect[l + i*numel]-1)+2];

							// Ezx:
							Strain_h[l + 5*numel + cont*numstr*numel] += auxX[i]*Vector_X_h[3*(Connect[l + i*numel]-1)+2] + auxZ[i]*Vector_X_h[3*(Connect[l + i*numel]-1)  ];

						}

						cont++;

					}

				}                                                    // =================== Loop kg ===================

			}                                                        // =================== Loop jg ===================

		}                                                            // =================== Loop ig ===================

	}

//	cudaThreadSynchronize();
//	cutStopTimer(hTimer);
//	Time = cutGetTimerValue(hTimer);
	
	printf("\n");
	printf("         Evaluating Strain State Time: %0.3f s \n", Time/CLOCKS_PER_SEC);

	free(XEL); free(YEL); free(ZEL);

}      

//==============================================================================
// Writing strain state in output file - CPUStrainState:

void WriteStrainStateCPU(int numel, double *Strain_h)
{
	int i, j;
	FILE *outFile;
    
    outFile = fopen("CPUStrainState.pos", "w");
    
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
		
}

//==============================================================================
// Writing stress state in output file - CPUStressState:

void WriteStressStateCPU(int numel, double *Stress_h)
{
	int i, j;
	FILE *outFile;
    
    outFile = fopen("CPUStressState.pos", "w");
    
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
// Writing nodal displacement in output file - GPUDisplacement:

void WriteDisplacementCPU(int numno, double *Vector_X_h)
{
	int i;
	FILE *outFile;
    
    outFile = fopen("CPUDisplacement.pos", "w");
    
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
void Grad_Conjugate(int numdof, int numno, int RowSize, double *Matrix_K_h, double *Vector_X_h, double *Vector_B_h, int *Vector_I_h)
{
	int i, j, cont, LineStep, aux1, aux2;
	double alfa, dnew, dold, epsilon, err, DEN, beta;
	unsigned int hTimer1;
	double time;
	std::vector<double> r, MI, d, q, s;
	std::vector<double> vx(numdof*numno);
	std::vector<std::vector<double>> M (numdof*numno, std::vector<double>(RowSize, 0));

	r.resize(numdof*numno);
	MI.resize(numdof*numno);
	d.resize(numdof*numno);
	q.resize(numdof*numno);
	s.resize(numdof*numno);

	for(j=0; j<numdof*numno; j++) {

		for(i=0; i<RowSize; i++) {
			M[j][i] = Matrix_K_h[j*RowSize + i];

			if(Vector_I_h[j*RowSize + i] == j+1 && Matrix_K_h[j*RowSize + i] != 0.) 
				MI[j] = 1/Matrix_K_h[j*RowSize + i];

		}

	}

	for(i=0; i<numdof*numno; i++) {
		r[i] = Vector_B_h[i]; 
		d[i] = MI[i]*r[i];
		Vector_X_h[i] = 0.;
	}

	dnew = 0.;
	for(i=0; i<numdof*numno; i++)
		dnew += r[i]*d[i];

	cont = 0;
	DEN = 0;
	epsilon  = 0.001;
	err = dnew * epsilon * epsilon;

	printf("\n");
	printf("         Solving Linear Equation System \n");
	printf("         ========================================= \n");
	printf("         * Conjugate Gradient Method * \n\n");
		
	//cutCreateTimer(&hTimer1);
	//cudaThreadSynchronize();
	//cutResetTimer(hTimer1);
	//cutStartTimer(hTimer1);

	while(dnew > err && cont < 2000) {

		for(j=0; j<numdof*numno; j++) {

			q[j] = 0.;

			for(i=0; i<RowSize; i++) {

				if(Vector_I_h[j*RowSize + i]-1 < 0) break;

				q[j] += Matrix_K_h[j*RowSize + i]*d[Vector_I_h[j*RowSize + i]-1];

			}

		}

		DEN = 0.;
		for(i=0; i<numdof*numno; i++)
			DEN += d[i]*q[i];

		alfa = dnew/DEN;

		for(i=0; i<numdof*numno; i++) {
			Vector_X_h[i] += alfa*d[i];
		    vx[i] = Vector_X_h[i]; 
			r[i] -= alfa*q[i];
			s[i] = MI[i]*r[i];
		}

		dold = dnew;

		dnew = 0.;
		for(i=0; i<numdof*numno; i++)
			dnew += r[i]*s[i];

		beta = dnew/dold;

		for(i=0; i<numdof*numno; i++)
			d[i] = s[i] + beta*d[i];

		cont++;

	}

	//cudaThreadSynchronize();
	//cutStopTimer(hTimer1);
	//time = cutGetTimerValue(hTimer1);
	printf("         Time Execution = %0.3f s \n", time/CLOCKS_PER_SEC);
	
	printf("\n");	
	printf("         Iteration Number = %i  \n", cont);
	printf("         Error = %0.7f  \n", dnew);


	

	


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
		
	}
	
	for(i=0; i<numno; i++) {
	
		if(BC_h[i]==1) {            // if BC_h = 1 Displacement is constrained at the node 
		
			Vector_B_h[3*i  ] = 0.;
			Vector_B_h[3*i+1] = 0.;
			Vector_B_h[3*i+2] = 0.;
		
		}
		
		//printf("%f %f %f \n", Vector_B_h[3*i  ], Vector_B_h[3*i+1], Vector_B_h[3*i+2]);
		
	}
	
	fclose(inFile);
    
}

//==============================================================================
void CPU_AssemblyStiffnessHexahedron(int ii, int jj, double v, int RowSize, int *Vector_I_h, double *Matrix_K_h)
{
	int k, cont, column;

	// Verifying if the term of the stiffness matrix exists: 
	for(k=0; k<RowSize; k++) {

		column = Vector_I_h[ii*RowSize + k];

		if(column == jj+1) {

			Matrix_K_h[ii*RowSize + k] += v;
			return;

		}
	}

	// Evaluating last column filled on of row:
	cont = 0;
	for(k=0; k<RowSize; k++) {
		if(Vector_I_h[ii*RowSize+k] != 0) cont += 1;
	}
 
	// Adding a new term in the stiffness matrix:
	Matrix_K_h[ii*RowSize+cont] = v;

}

//====================================================================================================
void CPU_ElementStiffnessHexahedron(int ElemNumber, int numnoel, int numel, int numstr, int numdof, double *X_h, double *Y_h, 
									double *Z_h, int *Connect, double dE, double dNu, double *Matrix_k_h)
{
	int i, j, k, ig, jg, kg;
	double jacob, soma;
	double r, s, t;
	double xgaus[2], wgaus[2], De[6][6], gama[3][3];
	double phi_r[8], phi_s[8], phi_t[8];
	double jac[3][3], B[6][24];
	double auxX[8], auxY[8], auxZ[8], aux[6][24];	
	double E, p, sho;

	double *XEL, *YEL, *ZEL;

	XEL = (double *)malloc(numnoel*(sizeof(double)));
	YEL = (double *)malloc(numnoel*(sizeof(double)));
	ZEL = (double *)malloc(numnoel*(sizeof(double)));

	for(i=0; i<numnoel; i++) {  // numnoel = Number of nodes per element

		XEL[i] = X_h[(Connect[ElemNumber + i*numel]-1)];
		YEL[i] = Y_h[(Connect[ElemNumber + i*numel]-1)];
		ZEL[i] = Z_h[(Connect[ElemNumber + i*numel]-1)];

	}

	xgaus[0]=-0.577350269189626;
	xgaus[1]= 0.577350269189626;
	wgaus[0]= 1.;
	wgaus[1]= 1.;

	E=dE;
	p=dNu;

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

	// Cleaning vectors
	for(i=0; i<numnoel*numdof*numnoel*numdof; i++) 
		Matrix_k_h[i] = 0;

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

				jac[0][0]=phi_r[0]*XEL[0]+phi_r[1]*XEL[1]+phi_r[2]*XEL[2]+phi_r[3]*XEL[3]+phi_r[4]*XEL[4]+phi_r[5]*XEL[5]+phi_r[6]*XEL[6]+phi_r[7]*XEL[7];
				jac[0][1]=phi_r[0]*YEL[0]+phi_r[1]*YEL[1]+phi_r[2]*YEL[2]+phi_r[3]*YEL[3]+phi_r[4]*YEL[4]+phi_r[5]*YEL[5]+phi_r[6]*YEL[6]+phi_r[7]*YEL[7];
				jac[0][2]=phi_r[0]*ZEL[0]+phi_r[1]*ZEL[1]+phi_r[2]*ZEL[2]+phi_r[3]*ZEL[3]+phi_r[4]*ZEL[4]+phi_r[5]*ZEL[5]+phi_r[6]*ZEL[6]+phi_r[7]*ZEL[7];

				jac[1][0]=phi_s[0]*XEL[0]+phi_s[1]*XEL[1]+phi_s[2]*XEL[2]+phi_s[3]*XEL[3]+phi_s[4]*XEL[4]+phi_s[5]*XEL[5]+phi_s[6]*XEL[6]+phi_s[7]*XEL[7];
				jac[1][1]=phi_s[0]*YEL[0]+phi_s[1]*YEL[1]+phi_s[2]*YEL[2]+phi_s[3]*YEL[3]+phi_s[4]*YEL[4]+phi_s[5]*YEL[5]+phi_s[6]*YEL[6]+phi_s[7]*YEL[7];
				jac[1][2]=phi_s[0]*ZEL[0]+phi_s[1]*ZEL[1]+phi_s[2]*ZEL[2]+phi_s[3]*ZEL[3]+phi_s[4]*ZEL[4]+phi_s[5]*ZEL[5]+phi_s[6]*ZEL[6]+phi_s[7]*ZEL[7];

				jac[2][0]=phi_t[0]*XEL[0]+phi_t[1]*XEL[1]+phi_t[2]*XEL[2]+phi_t[3]*XEL[3]+phi_t[4]*XEL[4]+phi_t[5]*XEL[5]+phi_t[6]*XEL[6]+phi_t[7]*XEL[7];
				jac[2][1]=phi_t[0]*YEL[0]+phi_t[1]*YEL[1]+phi_t[2]*YEL[2]+phi_t[3]*YEL[3]+phi_t[4]*YEL[4]+phi_t[5]*YEL[5]+phi_t[6]*YEL[6]+phi_t[7]*YEL[7];
				jac[2][2]=phi_t[0]*ZEL[0]+phi_t[1]*ZEL[1]+phi_t[2]*ZEL[2]+phi_t[3]*ZEL[3]+phi_t[4]*ZEL[4]+phi_t[5]*ZEL[5]+phi_t[6]*ZEL[6]+phi_t[7]*ZEL[7];

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

					for(i=0; i<8; i++) { 
						auxX[i] = gama[0][0]*phi_r[i]+gama[0][1]*phi_s[i]+gama[0][2]*phi_t[i];
						auxY[i] = gama[1][0]*phi_r[i]+gama[1][1]*phi_s[i]+gama[1][2]*phi_t[i];
						auxZ[i] = gama[2][0]*phi_r[i]+gama[2][1]*phi_s[i]+gama[2][2]*phi_t[i];
					}
						
					for(i=0; i<numnoel; i++) { 
						B[0][3*i  ] = auxX[i];
						B[1][3*i  ] = 0.; 
						B[2][3*i  ] = 0.;
						B[3][3*i  ] = auxY[i];
						B[4][3*i  ] = 0.;
						B[5][3*i  ] = auxZ[i];

						B[0][3*i+1] = 0.;
						B[1][3*i+1] = auxY[i];
						B[2][3*i+1] = 0.;
						B[3][3*i+1] = auxX[i];
						B[4][3*i+1] = auxZ[i];
						B[5][3*i+1] = 0.;

						B[0][3*i+2] = 0.;
						B[1][3*i+2] = 0.;
						B[2][3*i+2] = auxZ[i];
						B[3][3*i+2] = 0.;
						B[4][3*i+2] = auxY[i];
						B[5][3*i+2] = auxX[i];
					}

					for(i=0; i<numstr; i++) { 
						for(j=0; j<numdof*numnoel; j++) {   
							aux[i][j]=0.;
							for(k=0; k<numstr; k++) {   
								aux[i][j]=aux[i][j]+De[i][k]*B[k][j];
							}
						}
					}

					for(i=0; i<numdof*numnoel; i++) {  

						for(j=0; j<numdof*numnoel; j++) {  

							soma=0.;

							for(k=0; k<numstr; k++)
								soma=soma+B[k][i]*aux[k][j];
							
							Matrix_k_h[i*numdof*numnoel + j] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg]*soma;

						}

					}

				}

			}                                                    // =================== Loop kg ===================

		}                                                        // =================== Loop jg ===================

	}                                                            // =================== Loop ig ===================

	free(XEL); free(YEL); free(ZEL);

}      

//====================================================================================================
void CPU_ElementStiffnessHexahedronConventional(int ElemNumber, int numnoel, int numel, int numstr, int numdof, double *X_h, double *Y_h, 
									double *Z_h, int *Connect, double dE, double dNu, std::vector< std::vector<double> > &KCPULocalMatrix)
{
	int i, j, k, ig, jg, kg;
	double jacob, soma;
	double r, s, t;
	double xgaus[2], wgaus[2], De[6][6], gama[3][3];
	double phi_r[8], phi_s[8], phi_t[8];
	double jac[3][3], B[6][24];
	double auxX[8], auxY[8], auxZ[8], aux[6][24];	
	double E, p, sho;

	double *XEL, *YEL, *ZEL;

	XEL = (double *)malloc(numnoel*(sizeof(double)));
	YEL = (double *)malloc(numnoel*(sizeof(double)));
	ZEL = (double *)malloc(numnoel*(sizeof(double)));

	for(i=0; i<numnoel; i++) {  // numnoel = Number of nodes per element

		XEL[i] = X_h[(Connect[ElemNumber + i*numel]-1)];
		YEL[i] = Y_h[(Connect[ElemNumber + i*numel]-1)];
		ZEL[i] = Z_h[(Connect[ElemNumber + i*numel]-1)];

	}

	xgaus[0]=-0.577350269189626;
	xgaus[1]= 0.577350269189626;
	wgaus[0]= 1.;
	wgaus[1]= 1.;

	E=dE;
	p=dNu;

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

	// Cleaning vectors
	for(j=0; j<numnoel*numdof; j++) 
		for(i=0; i<numnoel*numdof; i++) 
			KCPULocalMatrix[i][j] = 0;

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

				jac[0][0]=phi_r[0]*XEL[0]+phi_r[1]*XEL[1]+phi_r[2]*XEL[2]+phi_r[3]*XEL[3]+phi_r[4]*XEL[4]+phi_r[5]*XEL[5]+phi_r[6]*XEL[6]+phi_r[7]*XEL[7];
				jac[0][1]=phi_r[0]*YEL[0]+phi_r[1]*YEL[1]+phi_r[2]*YEL[2]+phi_r[3]*YEL[3]+phi_r[4]*YEL[4]+phi_r[5]*YEL[5]+phi_r[6]*YEL[6]+phi_r[7]*YEL[7];
				jac[0][2]=phi_r[0]*ZEL[0]+phi_r[1]*ZEL[1]+phi_r[2]*ZEL[2]+phi_r[3]*ZEL[3]+phi_r[4]*ZEL[4]+phi_r[5]*ZEL[5]+phi_r[6]*ZEL[6]+phi_r[7]*ZEL[7];

				jac[1][0]=phi_s[0]*XEL[0]+phi_s[1]*XEL[1]+phi_s[2]*XEL[2]+phi_s[3]*XEL[3]+phi_s[4]*XEL[4]+phi_s[5]*XEL[5]+phi_s[6]*XEL[6]+phi_s[7]*XEL[7];
				jac[1][1]=phi_s[0]*YEL[0]+phi_s[1]*YEL[1]+phi_s[2]*YEL[2]+phi_s[3]*YEL[3]+phi_s[4]*YEL[4]+phi_s[5]*YEL[5]+phi_s[6]*YEL[6]+phi_s[7]*YEL[7];
				jac[1][2]=phi_s[0]*ZEL[0]+phi_s[1]*ZEL[1]+phi_s[2]*ZEL[2]+phi_s[3]*ZEL[3]+phi_s[4]*ZEL[4]+phi_s[5]*ZEL[5]+phi_s[6]*ZEL[6]+phi_s[7]*ZEL[7];

				jac[2][0]=phi_t[0]*XEL[0]+phi_t[1]*XEL[1]+phi_t[2]*XEL[2]+phi_t[3]*XEL[3]+phi_t[4]*XEL[4]+phi_t[5]*XEL[5]+phi_t[6]*XEL[6]+phi_t[7]*XEL[7];
				jac[2][1]=phi_t[0]*YEL[0]+phi_t[1]*YEL[1]+phi_t[2]*YEL[2]+phi_t[3]*YEL[3]+phi_t[4]*YEL[4]+phi_t[5]*YEL[5]+phi_t[6]*YEL[6]+phi_t[7]*YEL[7];
				jac[2][2]=phi_t[0]*ZEL[0]+phi_t[1]*ZEL[1]+phi_t[2]*ZEL[2]+phi_t[3]*ZEL[3]+phi_t[4]*ZEL[4]+phi_t[5]*ZEL[5]+phi_t[6]*ZEL[6]+phi_t[7]*ZEL[7];

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

					for(i=0; i<8; i++) { 
						auxX[i] = gama[0][0]*phi_r[i]+gama[0][1]*phi_s[i]+gama[0][2]*phi_t[i];
						auxY[i] = gama[1][0]*phi_r[i]+gama[1][1]*phi_s[i]+gama[1][2]*phi_t[i];
						auxZ[i] = gama[2][0]*phi_r[i]+gama[2][1]*phi_s[i]+gama[2][2]*phi_t[i];
					}
						
					for(i=0; i<numnoel; i++) { 
						B[0][3*i  ] = auxX[i];
						B[1][3*i  ] = 0.; 
						B[2][3*i  ] = 0.;
						B[3][3*i  ] = auxY[i];
						B[4][3*i  ] = 0.;
						B[5][3*i  ] = auxZ[i];

						B[0][3*i+1] = 0.;
						B[1][3*i+1] = auxY[i];
						B[2][3*i+1] = 0.;
						B[3][3*i+1] = auxX[i];
						B[4][3*i+1] = auxZ[i];
						B[5][3*i+1] = 0.;

						B[0][3*i+2] = 0.;
						B[1][3*i+2] = 0.;
						B[2][3*i+2] = auxZ[i];
						B[3][3*i+2] = 0.;
						B[4][3*i+2] = auxY[i];
						B[5][3*i+2] = auxX[i];
					}

					for(i=0; i<numstr; i++) { 
						for(j=0; j<numdof*numnoel; j++) {   
							aux[i][j]=0.;
							for(k=0; k<numstr; k++) {   
								aux[i][j]=aux[i][j]+De[i][k]*B[k][j];
							}
						}
					}

					for(i=0; i<numdof*numnoel; i++) {  

						for(j=0; j<numdof*numnoel; j++) {  

							soma=0.;

							for(k=0; k<numstr; k++)
								soma=soma+B[k][i]*aux[k][j];
							
							KCPULocalMatrix[i][j] += jacob*wgaus[ig]*wgaus[jg]*wgaus[kg]*soma;

						}

					}

				}

			}                                                    // =================== Loop kg ===================

		}                                                        // =================== Loop jg ===================

	}                                                            // =================== Loop ig ===================

	free(XEL); free(YEL); free(ZEL);

}      

//==============================================================================
void CPU_PreProcesssing(int numel, int *Connect, int numnoel, int numdof, int RowSize, int *Vector_I_h)
{
	int i, j, k, i1, i2, i3, ii, jj;

	std::vector<int> LM;            // Local - Global relation

	LM.resize(numdof*numnoel);

	for(k=0; k<numel; k++) {

		for(i=0; i<numnoel; i++) {  // numnoel = Number of nodes per element

			i1 = 3*i;
			i2 = 3*i+1;
			i3 = 3*i+2;
			LM[i1] = 3*(Connect[k + i*numel]-1)  ;
			LM[i2] = 3*(Connect[k + i*numel]-1)+1;
			LM[i3] = 3*(Connect[k + i*numel]-1)+2;

		}

		for(i=0; i<numdof*numnoel; i++) {
			ii = LM[i];

			for(j=0; j<numdof*numnoel; j++) {
				jj = LM[j];

				CPU_EvaluateNumbering(ii, jj, RowSize, Vector_I_h);

			}

		}

	}

}

//==============================================================================
void CPU_EvaluateNumbering(int ii, int jj, int RowSize, int *Vector_I_h) // ii = rows and jj = columns
{
	int i, cont;
	int LineStep;

	LineStep = ii*RowSize;  // Position of the row

	// Verifying if the term of the stiffness matrix exists: 
	for(i=0; i<RowSize; i++) {
		if(Vector_I_h[LineStep+i] == jj+1) return;
	}

	// Evaluating last column filled on of row:
	cont = 0;
	for(i=0; i<RowSize; i++) {

		if(Vector_I_h[LineStep+i] != 0) cont += 1;
		
	}
 
	// Adding a new term in Vector_I_h:
	Vector_I_h[LineStep+cont] = jj+1;

	return;

}

//==============================================================================
int CPU_ReadInPutFile(int numno, int numel, int nummat, int numprop, int numdof, int numnoel, int numstr, int numbc, int numnl, char GPU_arqaux[80],
double *X_h, double *Y_h, double *Z_h, int *BC_h, double *Material_Param_h, int *Connect, int *Material_Id_h)
{
	char label[80];
	std::string s;
	FILE *inFile;
	int c;
	int i, number, matId, aux, el1, el2, el3, el4, el5, el6, el7, el8;

	int _numno, _numel, _nummat, _numdof, _numnoel, _numstr,  _numbc, _numnl;
	int dx, dy, dz, rx, ry, rz;
	double prop1, prop2, X, Y, Z;
	
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

				if(fscanf(inFile, "%d %lf %lf %lf", &number, &X, &Y, &Z) != 4){
					printf("\nError on reading nodes coordinates, node Id = %d", number);
					return(0);
				}

				X_h[i] = X;
				Y_h[i] = Y;
				Z_h[i] = Z;

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

				Connect[i + 0*numel] = el1;
				Connect[i + 1*numel] = el2;
				Connect[i + 2*numel] = el3;
				Connect[i + 3*numel] = el4;
				Connect[i + 4*numel] = el5;
				Connect[i + 5*numel] = el6;
				Connect[i + 6*numel] = el7;
				Connect[i + 7*numel] = el8;

			}

			break;

		}

	}

	return(1);

}

//==============================================================================
/*void cDrv::PreProcesssingCPU()
{
	/*int i, j, k;
	int i1, i2, i3, ii, jj;
	int NumTerms, size;
	std::vector<int> connect;       // Element conectivity
	std::vector<int> LM;            // Local - Global relation 

 	numdof = Model->NumDofNode();          // Number of freedom of degree (DOF) per node
	numnoel = Elms[0]->Shp->MapNodeNo;    // Number of nodes per element
	numno = Node->NodeNo;               // Number of nodes of the mesh


	numdofel = numdof*numnoel;  // Number of DOF per element

	connect.resize(numnoel);
	LM.resize(numdofel);

	// Maximum row size for hexahedral element:
	rowsize = 96;

	// Stiffness Matrix - Allocating Memory Space:
	NumTerms = numdof*numno*rowsize;  // Size of the stiffness matriz

	size = NumTerms*sizeof(double);
	Matrix_K_h  = (double *)malloc(size);

	size = NumTerms*(sizeof(int));
	Vector_I_h  = (int *)malloc(size);

	for(i=0; i<NumTerms; i++) {
		Matrix_K_h[i] = 0.;
		Vector_I_h[i] = 0;
	}

	numel = Elms.size();  // Number of elements on mesh

	for(k=0; k<numel; k++) {

		// Get element connectivity from element class
		Elms[k]->Shp->Connectivity(connect);

		for(i=0; i<numnoel; i++) {
			i1 = 3*i+2;
			i2 = 3*i+1;
			i3 = 3*i;
			LM[i1] = 3*(connect[i]-1)+2;
			LM[i2] = 3*(connect[i]-1)+1;
			LM[i3] = 3*(connect[i]-1);
		}

		for(i=0; i<numdofel; i++) {
			ii = LM[i];
			for(j=0; j<numdofel; j++) {
				jj = LM[j];
				EvaluateNumberingCPU(ii, jj);
			}
		}

	}
} 

//==============================================================================
void cDrv::EvaluateNumberingCPU(int ii, int jj)
{
	int k, cont;
	int LineStep;

	/*LineStep = ii*rowsize;  // Position of the row

	// Verifying if the term of the stiffness matrix exists: 
	for(k=0; k<rowsize; k++) {
		if(Vector_I_h[LineStep+k] == jj+1) return;
	}

	// Evaluating last column filled on of row:
	cont = 0;
	for(k=0; k<rowsize; k++) {
		if(Vector_I_h[LineStep+k] != 0) cont += 1;
	}
 
	// Adding a new term in Vector_I_h:
	Vector_I_h[LineStep+cont] = jj+1;

	//printf(" %d", Vector_I_h[LineStep+cont]);

}

//====================================================================================================
void cDrv::AssemblyStiffnessCPUTetrahedron(std::vector<sNodeCoord> coord, double **ElemMatrix)
{
	int i, j, k;
	double V, soma;
	double B[6][12], De[6][6], aux[6][12];
	double x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4; 
	double E, p, sho;

	double y12, y13, y14, y21, y24, y31, y32, y34, y42, y43;
	double y23, z23;  // Used for calculating the tetrahedron volume
	double z12, z13, z14, z21, z24, z31, z32, z34, z42, z43;
	double x12, x13, x14, x21, x24, x31, x32, x34, x42, x43;

	std::vector<double>  a, b,c;
	a.resize(4);
	b.resize(4);
	c.resize(4);

	typedef std::vector<double>  iId;
	typedef std::vector<iId>  jMatrix;
	jMatrix  K_Matrix;

	// Get memory for element stiffness matrix
	K_Matrix.resize(12);
	for(j=0; j<12; j++)
		K_Matrix[j].resize(12);

	// ----------------------------------------------------------------------------------------------------------

	x1 = coord[0].x;
	y1 = coord[0].y;
	z1 = coord[0].z;

	x2 = coord[1].x;
	y2 = coord[1].y;
	z2 = coord[1].z;

	x3 = coord[2].x;
	y3 = coord[2].y;	
	z3 = coord[2].z;

	x4 = coord[3].x;
	y4 = coord[3].y;
	z4 = coord[3].z;

	// ----------------------------------------------------------------------------------------------------------

	y42 = y4 - y2;
	y31 = y3 - y1;
	y24 = y2 - y4;
	y13 = y1 - y3;

	z32 = z3 - z2;
	z43 = z4 - z3;
	z14 = z1 - z4;
	z21 = z2 - z1;

	y32 = y3 - y2;
	y34 = y3 - y4;
	y14 = y1 - y4;
	y12 = y1 - y2;

	z42 = z4 - z2;
	z13 = z1 - z3;
	z24 = z2 - z4;
	z31 = z3 - z1;

	// ----------------------------------------------------------------------------------------------------------

	x32 = x3 - x2;
	x43 = x4 - x3;
	x14 = x1 - x4;
	x21 = x2 - x1;

	//z42
	//z31
	//z24
	z13 = z1 - z3;

	x42 = x4 - x2;
	x13 = x1 - x3;
	x24 = x2 - x4;
	x31 = x3 - x1;

	//z32
	z34 = z3 - z4;
	//z14
	z12 = z1 - z2;

	// ----------------------------------------------------------------------------------------------------------

	//x42
	//x31
	//x24
	//x13

	//y32
	y43 = y4 - y3;
	//y14
	y21 = y2 - y1;

	//x32
	x34 = x3 - x4;
	//x14
	x12 = x1 - x2;

	//y42
	//y13
	y24 = y2 - y4;
	//y31

	y23 = y2 - y3;
	z23 = z2 - z3;

	// ----------------------------------------------------------------------------------------------------------

	a[0] = y42*z32 - y32*z42; if(fabs(a[0])<0.00001) a[0] = 0.;
	b[0] = x32*z42 - x42*z32; if(fabs(b[0])<0.00001) b[0] = 0.;
	c[0] = x42*y32 - x32*y42; if(fabs(c[0])<0.00001) c[0] = 0.;

	a[1] = y31*z43 - y34*z13; if(fabs(a[1])<0.00001) a[1] = 0.;
	b[1] = x43*z31 - x13*z34; if(fabs(b[1])<0.00001) b[1] = 0.;
	c[1] = x31*y43 - x34*y13; if(fabs(c[1])<0.00001) c[1] = 0.;

	a[2] = y24*z14 - y14*z24; if(fabs(a[2])<0.00001) a[2] = 0.;
	b[2] = x14*z24 - x24*z14; if(fabs(b[2])<0.00001) b[2] = 0.;
	c[2] = x24*y14 - x14*y24; if(fabs(c[2])<0.00001) c[2] = 0.;

	a[3] = y13*z21 - y12*z31; if(fabs(a[3])<0.00001) a[3] = 0.;
	b[3] = x21*z13 - x31*z12; if(fabs(b[3])<0.00001) b[3] = 0.;
	c[3] = x13*y21 - x12*y31; if(fabs(c[3])<0.00001) c[3] = 0.;

	// ----------------------------------------------------------------------------------------------------------

	// Calculation of tetrahedron volume:

	V = (x21*(y23*z34-y34*z23) + x32*(y34*z12-y12*z34) + x43*(y12*z23-y23*z12)) / 6;

	// ----------------------------------------------------------------------------------------------------------

	E=3000e10;
	p=0.3;

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

	// ----------------------------------------------------------------------------------------------------------

	for(i=0; i<numdofel; i++) { 
		for(j=0; j<numdofel; j++) { 	
			ElemMatrix[i][j] = 0.;
			K_Matrix[i][j] = 0.;
		}
	}

	for(i=0; i<4; i++) { 
		B[0][3*i  ] = a[i];
		B[1][3*i  ] = 0.; 
		B[2][3*i  ] = 0.;
		B[3][3*i  ] = b[i];
		B[4][3*i  ] = 0.;
		B[5][3*i  ] = c[i];

		B[0][3*i+1] = 0.;
		B[1][3*i+1] = b[i];
		B[2][3*i+1] = 0.;
		B[3][3*i+1] = a[i];
		B[4][3*i+1] = c[i];
		B[5][3*i+1] = 0.;

		B[0][3*i+2] = 0.;
		B[1][3*i+2] = 0.;
		B[2][3*i+2] = c[i];
		B[3][3*i+2] = 0.;
		B[4][3*i+2] = b[i];
		B[5][3*i+2] = a[i];
	}

	for(i=0; i<num_stre_comp; i++) { 
		for(j=0; j<numdofel; j++) {   
			aux[i][j]=0.;
			for(k=0; k<num_stre_comp; k++) {   
				aux[i][j]=aux[i][j]+De[i][k]*B[k][j];
			}
		}
	}

	for(i=0; i<numdofel; i++) {  
		for(j=0; j<numdofel; j++) {  
			soma=0.;
			for(k=0; k<num_stre_comp; k++) {
				soma=soma+B[k][i]*aux[k][j];
			}
			ElemMatrix[i][j] = ElemMatrix[i][j] + V*soma;
			K_Matrix[i][j]   = K_Matrix[i][j] + V*soma;
		}

	}

}         */                              
