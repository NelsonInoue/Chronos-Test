// FastFem.cpp : Defines the entry point for the console application.
//

#include <stdio.h>
#include <cstdlib>
#include <string.h>
#include "defs.h"
#include "GPUInfo.h"
#include "FemOneGPU.h"
#include "time.h"
#include <direct.h>

FILE *in;
FILE *pos;
//FILE *sens;
char  arqinp[80];
char  arqpos[80];


// -------------------------------------------------------------------------
// Local functions:
//
static void   GetFiles   ( int );
static void   OpenFiles  ( void );
static void   CloseFiles ( void );
static double GetTime    ( void );

// =============================  GetFiles  ================================

static void GetFiles( int given_project )
{
	char arqaux[85];

	if( !given_project )
	{
	printf( "\n\n" );
	printf("\t --------------------------------------------------------------------\n");
	printf("\t          PONTIFICAL CATHOLIC UNIVERSITY OF RIO DE JANEIRO           \n");
	printf("\t **               DEPARTMENT OF CIVIL ENGINEERING                  ** \n");
	printf("\t        GTEP - Group of Technology in Petroleum Engineering        \n\n");
	printf("\t                           C H R O N O S                           \n\n");
	printf("\t                   FINITE ELEMENT METHOD ON GPU                      \n");
	printf("\t                                                  June 2019          \n");
	printf("\t --------------------------------------------------------------------\n\n");

		printf("         Enter input file name [.dat]........: " );
		fgets( arqinp, 80, stdin );
		arqinp[strlen(arqinp)-1] = 0;
	}

	strcpy(arqaux, OUTPUT_EXAMPLES);
	strcat(arqaux, arqinp);
	strcat(arqaux, ".dat");
	in = fopen(arqaux, "r");

	/*sprintf( arqaux, "%s.dat", arqinp );
	in = fopen( arqaux, "r" );*/
	if( in == NULL )
	{
		printf( "\n\n\n\t ### Error on reading file %s !!! ### \n", arqaux );
		exit( 0 );
	}

}

// ==============================  OpenFiles  =================================

static void OpenFiles ( void )
{
 // Open input/output data files
   
 strcat( arqpos, OUTPUT_DIR);
 strcat( arqpos, arqinp);
 strcat( arqpos, ".pos" );
 strcat( arqinp, ".dat" );

 pos = fopen( arqpos, "w" );

 if (pos == NULL)
 {
  printf( "\n\n\n\t ### Error on opening output file !!! ### \n" );
  exit( 0 );
 }
}

// =============================  CloseFiles  =================================

static void CloseFiles( void )
{
 // Write END labels on output files

 fseek( pos, 0, 2 );
 fprintf( pos, "\n%%END\n" );

 // Close files

 fclose(  in );
 fclose( pos );
}

// ============================================================================

int main(int argc, char* argv[])
{
	char c;
	clock_t tInitial, tFinal, tTotal;
	
	GPUInfo g();
	cFemOneGPU *pcFemOneGPU;             // GPU class object
	
	mkdir("running");
	_chdir("running");	
	mkdir(OUTPUT_DIR);

	// Initial time:
	tInitial = clock();

	// ----------------------------------------------------------------------------

	// Get input file

	if (argc == 1)
	{
		GetFiles( 0 );
	}
	else
	{
		strcpy( arqinp, argv[1] );
		GetFiles( 1 );
	}

	// Open output files

	OpenFiles( );

	// ----------------------------------------------------------------------------

	
	/*if(argc <= 2) {
		printf("\n");
		printf ( "\t Enter Paralleling Scheme \n");
		printf("\n\n");
		printf ( "\t    1. With GPU Processing Scheme\n\n");
		printf ( "\t    2. With CPU Processing Scheme\n\n");
		printf("\n\n");
        printf ( "\t    Enter with choice........: ");
        scanf("%s", &c); 
		getchar();
	}
	else
		c = *argv[2];*/

	c = '1';

	if(c=='1'){
		printf ("\n\t Analysis using GPU Processing Scheme........\n\n\n");

		if( 1) {
			//pcFemOneGPU = new cFemOneGPU(arqinp);
			//pcFemOneGPU->AnalyzeFemOneGPU();
		}
		delete pcFemOneGPU;

	}else if(c=='2'){
	    printf ("\n\t Analysis using naive CPU Processing Scheme........\n\n\n");
		/*cCpu::Cpu = new cCpu();
		if(cCpu::Cpu->AnalyzeCPUNaive()==0) 
			printf("A problem occurred while the program was running!!");
		delete cCpu::Cpu;*/

	}

	// Final time:
	tFinal = clock();

	tTotal = (tFinal - tInitial) / CLOCKS_PER_SEC;
	printf("\n\         Processing time ................ %0.3f (s)\n\n", (double)tTotal);

	printf("\n");

	system ("pause");

	return 0;

}

