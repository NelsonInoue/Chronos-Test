/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\
*  Copyright (c) 2019 <GTEP> - All Rights Reserved                            *
*  This file is part of HERMES Project.                                       *
*  Unauthorized copying of this file, via any medium is strictly prohibited.  *
*  Proprietary and confidential.                                              *
*                                                                             *
*  Developers:                                                                *
*     - Bismarck G. Souza Jr <bismarck@puc-rio.br>                            *
*     - Nelson Inoue <inoue@puc-rio.br>                                       *
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#include "Utils.h"
#include <time.h>
#include <stdarg.h>
#include <sstream>

using namespace std;

void Error::print_msg()
{
	stringstream ss;
	ss << endl << endl
	   << "  ######################################################################\n\n"
	   << "   Error " << code << ": " << exception::what() << endl
	   << "   File: " << filename << " [" << line << "]\n\n" 
	   << "  ######################################################################\n\n";
	
	PRINT_LINE(ss.str());

	system("PAUSE");
}

int START_TIME() 
{
	TIMES_.push_back(clock()); 
	return TIMES_.size()-1;
} 

void START_TIME(int i) 
{ 
	TIMES_[i] = clock(); 
}

void START_TIME(vector<int> v) 
{ 
	for(auto it=v.begin(); it!=v.end();++it) 
		TIMES_[*it]=clock(); 
}

double END_TIME(int i) 
{ 
	return (clock()-TIMES_[i])/CLOCKS_PER_SEC; 
}

void END_TIME() 
{
	if (TIMES_[0] != 0.0 ) {
		cout << END_TIME(0) << " s" << endl << endl; 
		TIMES_[0] = 0.0;
	}
}

void PRINT(string fmt, ...) 
{
	va_list args;
	va_start(args, fmt);
	END_TIME();
	printf("  ");
	vprintf(fmt.c_str(), args);
	printf("\n\n");
	va_end(args);
}

void PRINT_TIME(string fmt, ...) 
{
	va_list args;
	va_start(args, fmt);
	END_TIME();
	printf("  ");
	vprintf(fmt.c_str(), args);
	printf("... ");
	va_end(args);
	START_TIME(0);
}

void PRINT_LINE()
{ 
	printf("   ##########################################################################\n\n"); 
}

void PRINT_LINE(string fmt, ...) 
{
	va_list args;
	va_start(args, fmt);
	END_TIME();
	vprintf(fmt.c_str(), args);
	va_end(args);
	TIMES_[0] = 0.0;
}
