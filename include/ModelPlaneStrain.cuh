#ifndef	_MODELPLANESTRAIN_H
#define	_MODELPLANESTRAIN_H

#include <cuda_runtime.h>

//==============================================================================
// cModelPlaneStrain
//==============================================================================
typedef class cModelPlaneStrain *pcModelPlaneStrain, &rcModelPlaneStrain;

class cModelPlaneStrain
{
public:

	int i, j, k;
	int GPU_N;
	


	cModelPlaneStrain();
	void AssemblyPlaneStrainStiffnessMatrix(int, int, int, int, int, int, int, double *, int *);
	

};


#endif

