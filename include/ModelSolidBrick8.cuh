#ifndef	_MODELSOLIDBRICK8_H
#define	_MODELSOLIDBRICK8_H

#include <cuda_runtime.h>

//==============================================================================
// cModelSolidBrick8
//==============================================================================
typedef class cModelSolidBrick8 *pcModelSolidBrick8, &rcModelSolidBrick8;

class cModelSolidBrick8
{
public:

	int i, j, k;
	int GPU_N;
	


	cModelSolidBrick8();
	void AssemblySolidBrick8StiffnessMatrix(int, int, int, int, int, int, int, int, double *, int *, double *, double *, int *);
	

};


#endif

