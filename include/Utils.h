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
#ifndef __ERROR__
#define __ERROR__

// Error handling
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#define ERROR(code, msg) throw Error(code, msg, __FILE__, __LINE__)
#define EXIT_TRY throw 0;
#define CATCH catch (Error& e){ e.print_msg(); }
#define START_TRY try{
#define END_TRY } CATCH
#define WRITE_ERROR(code, msg) Error(code, msg, __FILE__, __LINE__).print_msg()

class Error : public std::exception
{
public:
	Error(int code_, std::string msg, std::string filename_, int line_) :
	  exception(msg.c_str()), code(code_), filename(filename_), line(line_) {};

	void print_msg();

private:
	int code;
	int line;
	std::string filename;
};

using namespace std;

// Counting time
static vector<double> TIMES_(20);
int START_TIME();
void START_TIME(int i); 
void START_TIME(std::vector<int> v);
void END_TIME(); 
double END_TIME(int i); 

// Print information
void PRINT(string fmt, ...);
void PRINT_LINE();
void PRINT_LINE(string fmt, ...);
void PRINT_TIME(string fmt, ...);


#endif
