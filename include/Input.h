/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\
*  Copyright (c) 2019 <GTEP> - All Rights Reserved                            *
*  This file is part of HERMES Project.                                       *
*  Unauthorized copying of this file, via any medium is strictly prohibited.  *
*  Proprietary and confidential.                                              *
*                                                                             *
*  Developers:                                                                *
*     - Nelson Inoue <inoue@puc-rio.br>                                       *
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#ifndef	_Input_H
#define	_Input_H

//==============================================================================
// cInput
//==============================================================================
typedef class cInput *pcInput, &rcInput;

using namespace std;

class cInput
{
public:

	double *coord_h;
	int *supp_h/*, *QuadOrder*/;
	int *MatType, *MatElastIdx;
	double *prop_h;
	int *connect_h;
	int *NumElemByColor;
	int **NumElemByGPUbyColor, *NumColorbyGPU;
	double *GPUData;
	int *off_h, *offfull_h;
	double *KDia;
	double *B_h, *M_full;
	double *Material_Density_h;

	// --------------------------------------------------------------------------------------------------------------------------------------------------------
	int _iNumMeshElem;     // Number of mesh elements
	int _iNumMeshNodes;    // total number of nodes in the mesh
	int _iNumDofNode;      // Number of dof per node
	int _iNumSuppNodes;    // total number of supports
	int _iNumConcLoads;    // total number of nodal forces
	int _iNumQuad;         // Number of quadratures
	int _iNumMeshMat;      // total number of materials
	int _iNumElasMat;      // Number of elastic isotropic materials
	int _iNumStrCmp;       // Number of stress components
	int _iDimBMatrix;      // Dimension of B matrix
	int _iDimBnlMatrix;    // Dimension of Bnl matrix
	int _iNumShpNodes, _iNumMapNodes;
	int _inumDiaFull;
	int _inumDiaPart;      

	int nummat, numprop, numdof, numnoel, numstr, numbc, numnl, numgpu, setgpu;
	int nx, ny, nz, nsi1, nsi2, nsj1, nsj2, nov, nun;

	// imax             Number of nodes in x direction
	// jmax             Number of nodes in y direction
	// kmax             Number of nodes in z direction

	double _dE;            // Young's modulus
	double _dNu;           // Poisson's ratio

	int deviceCount;       // Number of GPUs (total)

	char _anmType[30];

	char GPU_arqaux[80];
	FILE *inFile;

	cInput();
	void ReadInputFile(char* inputPath);
	int ReadAnalysisModel();
	//int ReadNumericalIntegration();
	int ReadMaterialProp();
	int ReadElementProp();
	void ProcessHeaderAnalysis();
	int ReadNodeAttribute();
	int ReadDiagonalFormat();
	void ReadNumberOfGpus();
	int ProcessDiaFormatPartFull();
	int ProcessDiaFormatFullPart();
	int ProcessNodeCoord();
	int ProcessNodeSupport();
	int ProcessNodalForces();
	//int ProcessIntegration();
	int ProcessElasticIsotropic();
	int ProcessQ4();
	int ProcessBRICK8();
	int ProcessNumberbyColor( );
	void ProcessReadNumberOfGpus();
	void ReadGpuSpecification();

	void ReadMeshGeometry();  // For coupling

	int NextLabel2(char label[80]);
	int ReadString(char string[80]);
	//void StrUpper(char *str);
	void StrLower(char *str);

};


#endif