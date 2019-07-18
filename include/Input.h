#ifndef	_Input_H
#define	_Input_H

//==============================================================================
// cInput
//==============================================================================
typedef class cInput *pcInput, &rcInput;

using namespace std;

extern FILE *in;
int  NextLabel   ( char [80] );
int  ReadString  ( char [80] );
void PrintHeader ( void );
void StrUpper    ( char * );
void StrLower    ( char * );

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

	int nummat, numprop, numdof, numnoel, numstr, numbc, numnl, imax, jmax, kmax, numgpu, setgpu;

	double _dE;            // Young's modulus
	double _dNu;           // Poisson's ratio

	int deviceCount;       // Number of GPUs (total)

	char _anmType[30];

	cInput();
	void ReadInputFile();
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

};


#endif