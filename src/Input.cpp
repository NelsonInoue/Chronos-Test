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
#include <time.h>
#include <stdio.h>
#include "input.h"
#include "defs.h"
#include <cuda_runtime.h>

using namespace std;
// -------------------------------------------------------------------------
// Public functions:
//

//=============================================================================
cInput::cInput()
{

}

// ============================  NextLabel  ================================

int cInput::NextLabel2( char label[80] )
{
 int c;

 while( (c = fgetc(inFile)) != '%' )
 {        // finds next % 
  if( c == EOF )
   return( 0 );
 }
 if( fscanf( inFile, "%s", label ) != 1 )       // scan label string 
  return( 0 );
 else
  return( 1 );

}  // End of NextLabel 


// ===========================  ReadString  ================================

int cInput::ReadString( char string[80] )
{
 int i, c;

 while( (c = fgetc( inFile )) != '\'' )
 {
  // finds first '
  if( c == EOF ) return( 0 );
 }
 for( i = 0; (c = fgetc( inFile )) != '\''; i++ )
 {
  // fill string until next '
  if( c == EOF ) return( 0 );
  string[i] = c;
 }
 string[i] = '\0';
 return( 1 );

}  // End of ReadString


// =============================  StrUpper  ================================

/*void cInput::StrUpper( char *str )
{
  int l = strlen(str);

  for (int i = 0; i < l; i++) str[i] = toupper(str[i]);
}*/

// =============================  StrLower  ================================

void cInput::StrLower( char *str )
{
  int l = strlen(str);

  for (int i = 0; i < l; i++) str[i] = tolower(str[i]);
}

// =========================================================== End of publics 


#define streq(s1,s2)	((s1[0]==s2[0]) && strcmp(s1,s2)==0)

//==============================================================================
void cInput::ReadInputFile(char* inputPath)
{
	double time;

	time = clock();

	if(inputPath)
		strcpy(GPU_arqaux, inputPath);

	inFile  = fopen(GPU_arqaux, "r");

	// ========= Reading nanalysis model =========

	if(ReadAnalysisModel() == 0) {
		printf("Invalid neutral file or label END doesn't exist" );
	}

	// ========= Reading nodal attribute =========

	if(ReadNodeAttribute() == 0) {
		printf("Invalid neutral file or label END doesn't exist" );
	}

	// ========= Reading numerical integration data =========

	/*if(ReadNumericalIntegration() == 0) {
		printf("Invalid neutral file or label END doesn't exist" );
	}*/

	// ========= Reading material properties =========

	if(ReadMaterialProp() == 0) {
		printf("Invalid neutral file or label END doesn't exist" );
	}

	// ========= Reading material properties =========

	if(ReadElementProp() == 0) {
		printf("Invalid neutral file or label END doesn't exist" );
	}

	// ========= Reading diagonal format parameters =========

	if(ReadDiagonalFormat() == 0) {
		printf("Invalid neutral file or label END doesn't exist" );
	}

	// ========= Read GPUs Specification =========

	ReadGpuSpecification();  // GPUs specification

	// ========= Read number of GPUs and set GPU =========

	ReadNumberOfGpus();

	printf("         Time: %0.3f s \n", (clock()-time)/CLOCKS_PER_SEC);

}

//==============================================================================
void cInput::ReadMeshGeometry()  // // For coupling
{
	
	char label[80];

	// ========= Reading mesh geometry data =========

	//std::rewind(inFile);
	inFile  = fopen(GPU_arqaux, "r");

	while (1)
	{                
		if (NextLabel2(label) == 0)
		{
			printf("\nNode:\n Invalid neutral file or label END doesn't exist\n\n" );
			exit(0);
		}
		else if (streq(label, "GEOMETRY.SIZES"))
		{

			fscanf(inFile, "%d %d %d %d %d %d %d %d %d", &nx, &ny, &nz, &nsi1, &nsi2, &nsj1, &nsj2, &nov, &nun);
			break;

		}
		else if (streq(label, "END"))
			break;
	}

	// --------------------------------------------------------------------------------------------

}

//==============================================================================
int cInput::ReadAnalysisModel()
{
	char label[80];

	//rewind(inFile);
	inFile  = fopen(GPU_arqaux, "r");

	while( 1 )
	{
		if( NextLabel2( label ) == 0 )
		{
			printf( "\n Invalid neutral file or label END doesn't exist\n\n" );
			exit( 0 );
		}
		else if( strcmp( label, "HEADER.ANALYSIS" ) == 0 )
		{
			ProcessHeaderAnalysis();
		}
		else if( strcmp(label, "REMARK" ) == 0 )
			continue;
		else if( strcmp(label, "END" ) == 0 )
			break;
		else                // Not a known label
			continue;
	}

	return(1);

}

// ======================= ProcessHeaderAnalysis ===========================
void cInput::ProcessHeaderAnalysis()
{
	char analysis[80];

	ReadString(analysis);
	StrLower(analysis);

	if(streq(analysis, "plane_stress"))
	{
		char _anmType_aux[30] = "plane_stress";
		strcpy( _anmType, _anmType_aux);
		_iDimBMatrix   = 3;
		_iDimBnlMatrix = 4;
		_iNumDofNode   = 2;
	}
	else if (streq(analysis, "plane_strain"))
	{
		char _anmType_aux[30] = "plane_strain";
		strcpy( _anmType, _anmType_aux);
		_iDimBMatrix   = 4;
		_iDimBnlMatrix = 5;
		_iNumDofNode   = 2;
	}
	else if (streq(analysis, "axisymmetric"))
	{
		char _anmType_aux[30] = "axisymmetric";
		strcpy( _anmType, _anmType_aux);
		_iDimBMatrix   = 4;
		_iDimBnlMatrix = 5;
		_iNumDofNode   = 2;
	}
	else if (streq(analysis, "solid"))
	{
		char _anmType_aux[30] = "solid";
		strcpy( _anmType, _anmType_aux);
		_iDimBMatrix   = 6;
		_iDimBnlMatrix = 9;
		_iNumDofNode   = 3;
	}
	else
	{
		printf( "\n Error on reading global analysis model type !!!\n\n" );
		exit( 0 );
	}

}  

//==============================================================================
int cInput::ReadNodeAttribute()
{

	char label[80];

	// ========= Reading nodal coordinate data =========

	//rewind(inFile);
	inFile  = fopen(GPU_arqaux, "r");

	while (1)
	{                
		if (NextLabel2(label) == 0)
		{
			printf("\nNode:\n Invalid neutral file or label END doesn't exist\n\n" );
			exit(0);
		}
		else if (streq(label, "NODE.COORD"))
		{
			if(ProcessNodeCoord() == 0) {
				printf("Invalid neutral file or label END doesn't exist" );
			}
		}
		else if (streq(label, "NODE.SUPPORT"))
		{
			ProcessNodeSupport();
		}
		else if (streq(label, "LOAD.CASE.NODAL.FORCE" ) ||
			streq(label, "LOAD.CASE.NODAL.FORCES"))
		{
			ProcessNodalForces();
		}
		else if (streq(label, "LOAD.CASE.NODAL.DISPLACEMENT" ) ||
			streq(label, "LOAD.CASE.NODAL.DISPLACEMENTS"))
		{
			//ProcessNodalDispls( );
		}
		else if (streq(label, "END"))
			break;
	}

	return(1);

}

// ========================== ProcessNodeCoord =============================
int cInput::ProcessNodeCoord()
{
	int i, id, N;
	double x, y, z;

	if(fscanf(inFile, "%d", &_iNumMeshNodes) != 1) {
		printf("\n Error on reading number of nodes !!!\n\n" );
		return(0);
	}

	coord_h = (double *)malloc(sizeof(double)*_iNumDofNode*_iNumMeshNodes);
	//cudaMallocManaged((void **)&coord_h, sizeof(double)*_iNumDofNode*_iNumMeshNodes);

	for(i=0; i<_iNumMeshNodes; i++) {

		if(fscanf(inFile, "%d %lf %lf %lf", &id, &x, &y, &z) != 4){
			printf("\nError on reading nodes coordinates, node Id = %d", id);
			return(0);
		}

		coord_h[id-1                   ] = x;
		coord_h[id-1 +   _iNumMeshNodes] = y;
		coord_h[id-1 + 2*_iNumMeshNodes] = z;

		//printf("%d %f %f %f \n", id, coord_h[i                   ], coord_h[i +   _iNumMeshNodes], coord_h[i +   2*_iNumMeshNodes]);

	}

	return(1);

}

// ======================== ProcessNodeSupport =============================
int cInput::ProcessNodeSupport()
{
	int i, id, dx, dy, dz, rx, ry, rz;
	int N;

	if (fscanf(inFile, "%d", &_iNumSuppNodes) != 1)
	{
		printf("\n Error on reading number of node supports !!!\n\n");
		exit(0);
	}

	if (( _iNumSuppNodes == 0) || (_iNumMeshNodes == 0)) return(0);

	supp_h = (int *)malloc(sizeof(int)*(_iNumDofNode+1)*_iNumSuppNodes);
	//cudaMallocManaged((void **)&supp_h, sizeof(int)*(_iNumDofNode+1)*_iNumSuppNodes);

	for (i=0; i<_iNumSuppNodes; i++)
	{
		if (fscanf(inFile, "%d %d %d %d %d %d %d", &id, &dx, &dy, &dz, &rx, &ry, &rz) != 7)
		{
			printf("\n Error on reading node supports !!!\n\n");
			exit(0);
		}

		supp_h[i                   ] = id;
		supp_h[i + 1*_iNumSuppNodes] = dx;
		supp_h[i + 2*_iNumSuppNodes] = dy;
		supp_h[i + 3*_iNumSuppNodes] = dz;

		//printf("%d %d %d %d \n", supp_h[i                   ], supp_h[i +   _iNumSuppNodes], supp_h[i +   2*_iNumSuppNodes], supp_h[i +   3*_iNumSuppNodes]);

	}

	return(1);

}
// ========================= ProcessNodalForces ============================
int cInput::ProcessNodalForces()
{
	int i, id, N;
	double fx, fy ,fz ,mx ,my ,mz;

	// Read the number of nodal loads

	if (fscanf(inFile, "%d", &_iNumConcLoads) != 1)
	{
		printf("\nError on reading number of nodal loads !!!\n\n");
		exit(0);
	}

	B_h = (double *)malloc(sizeof(double)*_iNumDofNode*_iNumMeshNodes);
	//cudaMallocManaged((void **)&B_h, sizeof(double)*_iNumDofNode*_iNumMeshNodes);
	for(i=0; i<_iNumDofNode*_iNumMeshNodes; i++)
		B_h[i] = 0.;


	// Read the nodal loads

	for (i=0; i<_iNumConcLoads; i++)
	{
		if (fscanf(inFile,"%d %lf %lf %lf %lf %lf %lf", &id, &fx, &fy, &fz, &mx, &my, &mz) != 7)
		{
			printf("\n Error on reading nodal loads (number %d) !!!\n\n", i+1);
			exit(0);
		}

		B_h[3*(id-1)  ] = fx;
		B_h[3*(id-1)+1] = fy;
		B_h[3*(id-1)+2] = fz;

		//printf("%d %f %f %f\n", id, B[3*(id-1)  ], B[3*(id-1)+1], B[3*(id-1)+2]);

	}

	return(1);

} 

//==============================================================================
/*int cInput::ReadNumericalIntegration()
{
	char label[80];

	_iNumQuad = 0;
	rewind(inFile);

	while (1)
	{
		if (NextLabel(label) == 0)
		{
			printf("\n Invalid neutral file or label END doesn't exist.\n\n");
			exit(0);
		}
		else if (streq(label, "INTEGRATION.ORDER"))
		{
			ProcessIntegration( );
		}
		else if (streq(label, "REMARK"))
		{
			continue;
		}
		else if (streq(label, "END"))
		{
			break;
		}
		else
		{
			continue;
		}
	}

return(1);

}*/

// ============================ ProcessIntegration =========================

/*int cInput::ProcessIntegration()
{
	int i, id, r, s, t, rr, ss, tt;
	int N;

	if (fscanf(inFile, "%d", &_iNumQuad) != 1 || _iNumQuad == 0)
	{
		printf("\n Error on reading number of integration orders !!!\n\n");
		exit(0);
	}

	// Get memory for data structure
	N = _iNumQuad;

	cudaMallocManaged((void **)&QuadOrder, sizeof(int) * N);

	for(i = 0; i<_iNumQuad; i++) 
	{
		fscanf(inFile, "%d %d %d %d %d %d %d", &id, &r, &s, &t, &rr, &ss, &tt);

		QuadOrder[i              ] = r;
		QuadOrder[i +   _iNumQuad] = s;
		QuadOrder[i + 2*_iNumQuad] = t;

		//printf("%d %d %d \n", QuadOrder[i                   ], QuadOrder[i +   _iNumQuad], QuadOrder[i +   2*_iNumQuad]);

	}

}*/

// =============================== ReadAll =================================

int cInput::ReadMaterialProp()
{
	int i;
	char label[80];     // Sections labels

	//rewind(inFile);
	inFile  = fopen(GPU_arqaux, "r");

	while( 1 )
	{
		if( NextLabel2( label ) == 0 )
		{
			printf( "\n Invalid neutral file or label END doesn't exist\n\n" );
			exit( 0 );
		}
		else if( streq( label, "MATERIAL" ) )
		{
			fscanf(inFile,"%d", &_iNumMeshMat );
			MatType = (int *)malloc(sizeof(int)*7);
			//cudaMallocManaged((void **)&MatType, sizeof(int)*7);
			for(i=0; i<7; i++) MatType[i] = 0;
		}
		else if( streq( label, "MATERIAL.ISOTROPIC" ) )
		{
			ProcessElasticIsotropic( );
			MatType[1] = 1;
		}
		else if( streq( label, "MATERIAL.ORTHOTROPIC" ) )
		{
			//ProcessElasticOrthotropic( );
			MatType[2] = 1;
		}
		else if( streq( label, "MATERIAL.MISES" ) )
		{
			//ProcessVonMises( );
			MatType[3] = 1;
		}
		else if( streq( label, "MATERIAL.DRUCKER.PRAGER" ) )
		{
			//ProcessDruckerPrager( );
			MatType[4] = 1;
		}
		else if( streq( label, "MATERIAL.TRESCA" ) )
		{
			//ProcessTresca( );
			MatType[5] = 1;
		}
		else if( streq( label, "MATERIAL.MOHR.COULOMB" ) )
		{
			//ProcessMohrCoulomb( );
			MatType[6] = 1;
		}
		else if( streq( label, "REMARK" ) )
		{
			continue;
		}
		else if( streq( label, "END" ) )
		{
			break;
		}
		else                // not a known label
		{
			continue;
		}
	}

	return (1);

}  

// ======================= ProcessElasticIsotropic =========================
int cInput::ProcessElasticIsotropic()
{
	int i, label;
	double prop1, prop2, prop3;

	if( fscanf(inFile, "%d", &_iNumElasMat ) != 1 )
	{
		printf( "\n Error on reading number of elastic isotropic materials !!!\n\n" );
		exit( 0 );
	}

	if( _iNumElasMat == 0 ) return(0);

	MatElastIdx = (int *)malloc(sizeof(int)*_iNumElasMat);
	prop_h = (double *)malloc(sizeof(double)*2*_iNumElasMat);
	Material_Density_h = (double *)malloc(_iNumElasMat*3*(sizeof(double)));

	for(i = 0; i<_iNumElasMat; i++ )
	{
		if( fscanf(inFile, "%d %lf %lf %lf %lf %lf", &label, &_dE, &_dNu, &prop1, &prop2, &prop3) != 6 )
		{
			printf( "\n Error on reading elastic isotropic materials !!!\n\n" );
			exit( 0 );
		}

		MatElastIdx[i] = label;
		prop_h[i             ] = _dE;
		prop_h[i+_iNumElasMat] = _dNu;

		Material_Density_h[i+0*_iNumElasMat] = prop1;
		Material_Density_h[i+1*_iNumElasMat] = prop2;
		Material_Density_h[i+2*_iNumElasMat] = prop3;

		//printf("%d %f %f \n", MatElastIdx [i           ], prop_h[i           ], prop_h[i + num_eiso]);

	}

	return(1);

} 

// =============================== ReadAll =================================
int cInput::ReadElementProp( void )
{
	char label[80];     // sections labels
	int  n;

	// Initialization

	_iNumMeshElem = 0;

	// Read element data

	//rewind(inFile);
	inFile  = fopen(GPU_arqaux, "r");

	while( 1 )
	{
		if( NextLabel2( label ) == 0 )
		{
			printf( "\n Invalid neutral file or label END doesn't exist\n\n" );
			exit( 0 );
		}
		else if( streq( label, "ELEMENT" ))
			fscanf(inFile,"%d", &n );
		/*else if (streq(label, "ELEMENT.PSTRESS.T3")) ProcessPSTRESST3( );
		else if (streq(label, "ELEMENT.PSTRESS.T6")) ProcessPSTRESST6( );
		else if (streq(label, "ELEMENT.PSTRESS.Q4")) ProcessPSTRESSQ4( );
		else if (streq(label, "ELEMENT.PSTRESS.Q8")) ProcessPSTRESSQ8( );
		else if (streq(label, "ELEMENT.PSTRESS.Q9")) ProcessPSTRESSQ9( );
		else if (streq(label, "ELEMENT.PSTRAIN.T3")) ProcessPSTRAINT3( );
		else if (streq(label, "ELEMENT.PSTRAIN.T6")) ProcessPSTRAINT6( );
		else if (streq(label, "ELEMENT.PSTRAIN.Q4")) ProcessPSTRAINQ4( );
		else if (streq(label, "ELEMENT.PSTRAIN.Q8")) ProcessPSTRAINQ8( );
		else if (streq(label, "ELEMENT.PSTRAIN.Q9")) ProcessPSTRAINQ9( );
		else if (streq(label, "ELEMENT.AXISYM.T3" )) ProcessAXISYMT3( );
		else if (streq(label, "ELEMENT.AXISYM.T6" )) ProcessAXISYMT6( );
		else if (streq(label, "ELEMENT.AXISYM.Q4" )) ProcessAXISYMQ4( );
		else if (streq(label, "ELEMENT.AXISYM.Q8" )) ProcessAXISYMQ8( );
		else if (streq(label, "ELEMENT.AXISYM.Q9" )) ProcessAXISYMQ9( );
		else if (streq(label, "ELEMENT.T3"       )) ProcessT3( );
		else if (streq(label, "ELEMENT.T6"       )) ProcessT6( );
		else if (streq(label, "ELEMENT.T10"      )) ProcessT10( );*/
		else if (streq(label, "ELEMENT.Q4"       )) ProcessQ4( );
		/*else if (streq(label, "ELEMENT.Q5"       )) ProcessQ5( );
		else if (streq(label, "ELEMENT.Q6"       )) ProcessQ6( );
		else if (streq(label, "ELEMENT.Q6B"      )) ProcessQ6B( );
		else if (streq(label, "ELEMENT.Q8"       )) ProcessQ8( );
		else if (streq(label, "ELEMENT.Q9"       )) ProcessQ9( );
		else if (streq(label, "ELEMENT.QUAD"     )) ProcessQUAD( );
		else if (streq(label, "ELEMENT.Q12"      )) ProcessQ12( );
		else if (streq(label, "ELEMENT.Q13"      )) ProcessQ13( );
		else if (streq(label, "ELEMENT.INFINITE" )) ProcessINFINITE( );*/
		else if (streq(label, "ELEMENT.BRICK8"   )) ProcessBRICK8( );
		/*else if (streq(label, "ELEMENT.BRICK20"  )) ProcessBRICK20( );
		else if (streq(label, "ELEMENT.TETR4"    )) ProcessTETR4( );
		else if (streq(label, "ELEMENT.TETR10"   )) ProcessTETR10( );*/
		else if (streq(label, "ELEMENT.NUMBER.BY.COLOR"   )) ProcessNumberbyColor( );
		else if (streq(label, "REMARK"))
			continue;
		else if (streq(label, "END"))
			break;
		else 
			continue;
	}

	return (1);

}

// ============================ ProcessBRICK8 ==============================
int cInput::ProcessQ4()
{
	int i, elm, mat, thc, ord;
	int lb1, lb2, lb3, lb4;

	_iNumShpNodes = 4;
	_iNumMapNodes = 4;

	if( fscanf(inFile, "%d", &_iNumMeshElem ) != 1 )
	{
		printf( "\n Error on reading number of BRICK8 elements !!!\n\n" );
		exit( 0 );
	}

	if( _iNumMeshElem == 0 ) return(0);

	connect_h = (int *)malloc(sizeof(int)*_iNumMeshElem*8);
	//cudaMallocManaged((void **)&connect_h, sizeof(int)*_iNumMeshElem*8);

	for( int i = 0; i < _iNumMeshElem; i++ ) {

		if (fscanf(inFile, "%d %d %d %d %d %d %d %d", &elm, &mat, &thc, &ord, &lb1, &lb2, &lb3, &lb4) != 8)
		{
			printf( "\n Error on reading elastic isotropic materials !!!\n\n" );
			exit( 0 );
		}

		connect_h[i            ] = elm;
		connect_h[i +  1*_iNumMeshElem] = mat;
		connect_h[i +  2*_iNumMeshElem] = thc;
		connect_h[i +  3*_iNumMeshElem] = ord;
		connect_h[i +  4*_iNumMeshElem] = lb1;
		connect_h[i +  5*_iNumMeshElem] = lb2;
		connect_h[i +  6*_iNumMeshElem] = lb3;
		connect_h[i +  7*_iNumMeshElem] = lb4;

		//printf("%d %d %d %d %d %d %d %d \n", elm, connect_h[i +  0*num_Q4], connect_h[i +  1*num_Q4], connect_h[i +  2*num_Q4], connect_h[i +  3*num_Q4],
			//connect_h[i +  4*num_Q4], connect_h[i +  5*num_Q4], connect_h[i +  6*num_Q4], connect_h[i +  7*num_Q4]);

	} 

	return(1);

}

// ============================ ProcessBRICK8 ==============================
int cInput::ProcessBRICK8( void )
{
	int i, elm, mat, ord;
	int lb1, lb2, lb3, lb4, lb5, lb6, lb7, lb8;

	_iNumShpNodes = 8;
	_iNumMapNodes = 8;

	if( fscanf(inFile, "%d", &_iNumMeshElem ) != 1 )
	{
		printf( "\n Error on reading number of BRICK8 elements !!!\n\n" );
		exit( 0 );
	}

	if( _iNumMeshElem == 0 ) return(0);

	connect_h = (int *)malloc(sizeof(int)*_iNumMeshElem*11);
	//cudaMallocManaged((void **)&connect_h, sizeof(int)*_iNumMeshElem*11);

	for( int i = 0; i < _iNumMeshElem; i++ ) {

		if (fscanf(inFile, "%d %d %d %d %d %d %d %d %d %d %d", &elm, &mat, &ord, &lb1, &lb2, &lb3, &lb4, &lb5, &lb6, &lb7, &lb8) != 11)
		{
			printf( "\n Error on reading elastic isotropic materials !!!\n\n" );
			exit( 0 );
		}

		connect_h[i                   ] = elm;  // 0
		connect_h[i +  1*_iNumMeshElem] = mat;  // 1
		connect_h[i +  2*_iNumMeshElem] = ord;  // 2
		connect_h[i +  3*_iNumMeshElem] = lb1;  // 3 => Element starts
		connect_h[i +  4*_iNumMeshElem] = lb2;  // 4
		connect_h[i +  5*_iNumMeshElem] = lb3;  // 5
		connect_h[i +  6*_iNumMeshElem] = lb4;  // 6
		connect_h[i +  7*_iNumMeshElem] = lb5;  // 7 => Top starts
		connect_h[i +  8*_iNumMeshElem] = lb6;  // 8
		connect_h[i +  9*_iNumMeshElem] = lb7;  // 9
		connect_h[i + 10*_iNumMeshElem] = lb8;  // 10

	} 

	return(1);

}

// ============================ ProcessNumberbyColor ==============================
int cInput::ProcessNumberbyColor()
{
	int i, j, elms, numcolor;

	if( fscanf(inFile, "%d", &numgpu ) != 1 )
	{
		printf( "\n Error on reading number of BRICK8 elements !!!\n\n" );
		exit( 0 );
	}

	// Get memory for data structure

	NumElemByGPUbyColor = (int **)malloc(sizeof(int)*numgpu);

	NumColorbyGPU = (int *)malloc(sizeof(int) * numgpu);

	for(j=0; j<numgpu; j++) {

		fscanf(inFile, "%d", &numcolor);

		NumColorbyGPU[j] = numcolor;

		NumElemByGPUbyColor[j] = (int *)malloc(sizeof(int) * numcolor);

		for(i=0; i<numcolor; i++) {

			fscanf(inFile, "%d", &elms);
			NumElemByGPUbyColor[j][i] = elms;

		}

	}

	return(1);	

}

//==============================================================================
int cInput::ReadDiagonalFormat()
{
	char label[80];

	_iNumQuad = 0;
	//rewind(inFile);
	inFile  = fopen(GPU_arqaux, "r");

	while (1)
	{
		if (NextLabel2(label) == 0)
		{
			printf("\n Invalid neutral file or label END doesn't exist.\n\n");
			exit(0);
		}
		else if (streq(label, "OFFSETS.PART.FULL"))
		{
			ProcessDiaFormatPartFull( );
		}
		else if (streq(label, "OFFSETS.FULL.PART"))
		{
			ProcessDiaFormatFullPart( );
		}
		else if (streq(label, "REMARK"))
		{
			continue;
		}
		else if (streq(label, "END"))
		{
			break;
		}
		else
		{
			continue;
		}
	}

return(1);

}

// ============================ ProcessDiaFormatPartFull ==============================
int cInput::ProcessDiaFormatPartFull()
{
	int i, elms;

	if( fscanf(inFile, "%d", &_inumDiaPart ) != 1 )
	{
		printf( "\n Error on reading number of BRICK8 elements !!!\n\n" );
		exit( 0 );
	}

	off_h = (int *)malloc(sizeof(int)*_inumDiaPart);
	//cudaMallocManaged((void **)&off_h, sizeof(int)*_inumDiaPart);

	for(i=0; i<_inumDiaPart; i++) {

		fscanf(inFile, "%d", &elms);
		off_h[i] = elms;

	}

	/*for( int i = 0; i < _inumDiaPart; i++ )
		printf("%d  ", off_h[i]);*/

	return(1);	

}

// ============================ ProcessDiaFormatFullPart ==============================
int cInput::ProcessDiaFormatFullPart()
{
	int i, a1, a2;

	if( fscanf(inFile, "%d", &_inumDiaFull ) != 1 )
	{
		printf( "\n Error on reading number of BRICK8 elements !!!\n\n" );
		exit( 0 );
	}

	offfull_h = (int *)malloc(sizeof(int)*_inumDiaFull);
	//cudaMallocManaged((void **)&offfull_h, sizeof(int) * _inumDiaFull);

	for(i=0; i<_inumDiaFull; i++)
		offfull_h[i] = 0;


	for(i=0; i<_inumDiaPart; i++) {

		fscanf(inFile, "%d %d", &a1, &a2);
		offfull_h[a1] = a2;

	}

	/*for( int i = 0; i < _inumDiaFull; i++ )
		if(offfull_h[i] >0)
			printf("%d  %d\n", i, offfull_h[i]);*/

	return(1);	

}

// ============================ ReadGpuSpecification ==============================
void cInput::ReadGpuSpecification()
{
	int dev;
	size_t free_byte, total_byte;
	double auxN;


	// Printing on output file:  ===============================================
	// Create a new file
	ofstream outfile;


	// ========= Getting GPU information =========

	cudaGetDeviceCount(&deviceCount);

	GPUData = (double *)malloc(6*deviceCount*sizeof(double));
   
	// GPUData[dev*6+0] = Total number of multiprocessor
	// GPUData[dev*6+1] = Total amount of shared memory per block
	// GPUData[dev*6+2] = Total number of registers availaHble per block
	// GPUData[dev*6+3] = Warp size
	// GPUData[dev*6+4] = Maximum number of threads per block
	// GPUData[dev*6+5] = Total number of threads per multiprocessor

	
		
	if (DEBUGGING) {
		outfile.open("GPUsInformation.out");

		outfile << "======================================================== " << endl;
		outfile << endl;
		outfile << "Number of devices " << endl;
		outfile << deviceCount << endl;
		outfile << endl;
	}

	// Printing on Display:  ===================================================

	printf("         Number of devices = %d\n", deviceCount);

	// This function call returns 0 if there are no CUDA capable devices.
	if (deviceCount == 0)
		printf("         There is no device supporting CUDA\n");

	// ======================================================================================================================

	for(dev=0; dev<deviceCount; ++dev) {

		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);

		cudaSetDevice(dev);
		GPUData[dev*6+0] = deviceProp.multiProcessorCount;
		GPUData[dev*6+1] = deviceProp.sharedMemPerBlock;
		GPUData[dev*6+2] = deviceProp.regsPerBlock;
		GPUData[dev*6+3] = deviceProp.warpSize;
		GPUData[dev*6+4] = deviceProp.maxThreadsPerBlock;
		GPUData[dev*6+5] = deviceProp.maxThreadsPerMultiProcessor;

	// *********************************** Print GPUs Information: ***********************************

		if (DEBUGGING) {
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

	if (DEBUGGING) 
		outfile.close(); 

}

//==============================================================================
void cInput::ReadNumberOfGpus()
{
	char label[80];

	//rewind(inFile);
	inFile  = fopen(GPU_arqaux, "r");

	while (1)
	{
		if (NextLabel2(label) == 0)
		{
			printf("\n Invalid neutral file or label END doesn't exist.\n\n");
			exit(0);
		}
		else if (streq(label, "NUMBER.GPUS"))
		{
			ProcessReadNumberOfGpus( );
		}
		else if (streq(label, "END"))
		{
			break;
		}
		else
		{
			continue;
		}
	}

}

// ============================ ProcessReadNumberOfGpus ==============================
void cInput::ProcessReadNumberOfGpus()
{
	if(fscanf(inFile, "%d %d", &numgpu, &setgpu) != 2) {
	}

	// numgpu = 100 to use all the GPUs on the computer
	//        = 0   to use GPU "0"
	//        = 1   to use GPU "1" ...
	// setgpu       setted GPU
	
}