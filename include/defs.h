/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\
*  Copyright (c) 2019 <GTEP> - All Rights Reserved                            *
*  This file is part of HERMES Project.                                       *
*  Unauthorized copying of this file, via any medium is strictly prohibited.  *
*  Proprietary and confidential.                                              *
*                                                                             *
*  Developers:                                                                *
*     - Nelson Inoue <inoue@puc-rio.br>                                       *
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

// Definitions file
#ifndef __DEFS_CHRONOS__
#define __DEFS_CHRONOS__
#include "Error.h"

namespace Chronos {

	#define VERSION "2019.09"

	//Debugging mode
	static bool DEBUGGING = false;
	static char* DEBUG_FILE = "dean_p1";

	#define OUTPUT_DIR			"../output/"
	#define OUTPUT_STRESS   OUTPUT_DIR "StressState.pos"
	#define OUTPUT_STRAIN   OUTPUT_DIR "StrainState.pos"
	#define OUTPUT_EXAMPLES "../examples/"
}

#endif