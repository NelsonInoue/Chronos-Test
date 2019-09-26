/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\
*  Copyright (c) 2019 <GTEP> - All Rights Reserved                            *
*  This file is part of HERMES Project.                                       *
*  Unauthorized copying of this file, via any medium is strictly prohibited.  *
*  Proprietary and confidential.                                              *
*                                                                             *
*  Developers:                                                                *
*     - Bismarck G. Souza Jr <bismarck@puc-rio.br>                            *
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
// Definitions file
#ifndef __DEFS__
#define __DEFS__

#define VERSION "2019.09"

//#define DEBUG_MODE
#define DEBUG_FILE "dean_p1"

#define OUTPUT_DIR			"../output/"

#define OUTPUT_STRESS   OUTPUT_DIR "StressState.pos"
#define OUTPUT_STRAIN   OUTPUT_DIR "StrainState.pos"
#define OUTPUT_EXAMPLES "../examples/"

#define STRESS_OUTPUT_DIR	OUTPUT_DIR "Stress/"
#define FLOW_OUTPUT_DIR		OUTPUT_DIR "Flow/"
#define INPUT_DIR			"../input/"


#ifdef DEBUG_MODE
	#define IF_DEBUGGING if(true)
	#define DEBUG_IF(x) if(x)
	#define IF_NOT_DEBUGGING if(false)
#else
	#define IF_DEBUGGING if(false)
	#define DEBUG_IF(x) if(false)
	#define IF_NOT_DEBUGGING if(true)
#endif

#define ELSE else

// Enumerations
enum { X, Y, Z };
enum { LEFT, RIGHT, FRONT, BACK, UP, DOWN };


#include <exception>
#include <iostream>
#include <string>

#define ERROR(code, msg) throw Error(code, msg, __FILE__, __LINE__)
#define CATCH catch (Error& e){ e.print_msg(); }
#define TRY(a) try{a} CATCH
#define START_TRY try{
#define END_TRY } catch (Error& e){ e.print_msg(); }
#define WRITE_ERROR(code, msg) Error(code, msg, __FILE__, __LINE__).print_msg()

class Error : public std::exception
{
public:
  Error(int code_, std::string msg, std::string filename_, int line_) :
    exception(msg.c_str()), code(code_), filename(filename_), line(line_) {};

  void print_msg()
  {
    std::string s;
    std::cout << "Error " << code << ": " << exception::what() << std::endl
      << "File: " << filename << " [" << line << "]" << std::endl;
	system("PAUSE");
  }

private:
  int code;
  int line;
  std::string filename;

};

#endif