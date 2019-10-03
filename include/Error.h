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

#define ERROR(code, msg) throw Error(code, msg, __FILE__, __LINE__)
#define CATCH catch (Error& e){ e.print_msg(); }
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
		std::cout << std::endl << std::endl
					<< "######################################################################" << std::endl
					<< "Error " << code << ": " << exception::what() << std::endl
					<< "File: " << filename << " [" << line << "]" << std::endl
					<< "######################################################################" << std::endl
					<< std::endl;
		system("PAUSE");
	}

private:
	int code;
	int line;
	std::string filename;
};

#endif
