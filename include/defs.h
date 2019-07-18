// Definitions file
#ifndef __DEFS__
#define __DEFS__

//#define DEBUG_MODE

#include <direct.h> // mkdir
#define OUTPUT_DIR      "../output/"
#define OUTPUT_STRESS   OUTPUT_DIR "StressState.pos"
#define OUTPUT_STRAIN   OUTPUT_DIR "StrainState.pos"
#define OUTPUT_EXAMPLES "../examples/"

#ifdef DEBUG_MODE
	#define IF_DEBUGGING if(true)
	#define DEBUG_IF(x) if(x)
	#define IF_NOT_DEBUGGING if(false)
#else
	#define IF_DEBUGGING if(false)
	#define DEBUG_IF(x) if(false)
	#define IF_NOT_DEBUGGING if(true)
#endif

#endif