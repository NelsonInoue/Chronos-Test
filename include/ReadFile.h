/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\
*  Copyright (c) 2019 <GTEP> - All Rights Reserved                            *
*  This file is part of HERMES Project.                                       *
*  Unauthorized copying of this file, via any medium is strictly prohibited.  *
*  Proprietary and confidential.                                              *
*                                                                             *
*  Developers:                                                                *
*     - Bismarck G. Souza Jr <bismarck@puc-rio.br>                            *
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#ifndef _READFILE_H_
#define _READFILE_H_

#include <string>
#include <fstream>
#include <sstream>
#include "defs.h"

#define MAX_LEN_LINE 2048

using namespace std;

/**
Read a file wiht the pattern:

*KEY1 VALUE1  ** COMMENTS1
*KEY2 VALUE2  ** COMMENTS2
*KEY3 VALUE3  ** COMMENTS3
^    ^        ^
|    |        L str_comments: "**"
|    L chr_delimiter: ' '
L chr_key: '*'
*/
class ReadFile
{

public:
	ReadFile() : filename(""), chr_key('*'), str_comments("**"), finish(false),
		chr_delimiter(' ') {  };

	double read_file(string filename_);

protected:
	bool finish;
	char chr_key;
	char chr_delimiter;
	char key[100];
	char value[1000];
	char* str_comments;
	FILE* fin;
	string path;
	string basename;
	string filename;

	bool read_key_value(char* line);
	virtual bool get_line(char* line);
	virtual void process_file();
	virtual void process_key() {};
	virtual bool process_line(char* line);
	virtual void init() {};
	virtual void finally() {};
};

string get_text(string str);

#endif