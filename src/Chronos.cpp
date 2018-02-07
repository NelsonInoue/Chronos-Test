// FastFem.cpp : Defines the entry point for the console application.
//

#include <stdio.h>
#include <cstdlib>
#include "GPU.h"
#include "cpu.h"
#include "time.h"

void InitialInformation();

int main(int argc, char* argv[])
{
	char c;
	clock_t tInitial, tFinal, tTotal;

	// Initial time:
	tInitial = clock();

	InitialInformation();
	
	if(argc <= 2) {
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
		c = *argv[2];


	if(c=='1'){
		printf ("\n\t Analysis using GPU Processing Scheme........\n\n\n");
		cGPU::GPU = new cGPU();
		if(cGPU::GPU->AnalyzeMultiGPU()==0) 
			printf("A problem occurred while the program was running!!");
		delete cGPU::GPU;

	}else if(c=='2'){
	    printf ("\n\t Analysis using naive CPU Processing Scheme........\n\n\n");
		cCpu::Cpu = new cCpu();
		if(cCpu::Cpu->AnalyzeCPUNaive()==0) 
			printf("A problem occurred while the program was running!!");
		delete cCpu::Cpu;

	}

	// Final time:
	tFinal = clock();

	tTotal = (tFinal - tInitial) / CLOCKS_PER_SEC;
	printf("\n\         Processing time ................ %0.3f (s)\n\n", (double)tTotal);

	printf("\n");

	system ("pause");

	return 0;

}

