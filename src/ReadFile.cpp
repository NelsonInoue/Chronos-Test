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
#include "ReadFile.h"
#include <sstream>
#include <time.h>

string get_text(string str)	{
	return str.substr(str.find("'") + 1, str.rfind("'") - 1);
};

double ReadFile::read_file(string filename_, string path_)
{
	double time = clock();
	auto found = filename_.find_last_of("/\\");
	if (found != string::npos){
		filename = filename_.substr(found + 1);
		path = path_+filename_.substr(0, found);
	}
	else{
		filename = filename_;
		path = path_==""? "./" : path_;
	}
	basename = filename.substr(0, filename.find('.'));

	START_TRY

		if (fopen_s( &fin, (path+"/"+filename).c_str(), "r" ))
			file_not_found();

		init();

		process_file();

		fclose(fin);

		finally();

	END_TRY	

	catch(int e){
		return -1;
	}

	return (clock()-time)/CLOCKS_PER_SEC;
}

void ReadFile::process_file()
{
	char line[MAX_LEN_LINE];

	while (get_line(line) && !finish)
		process_line(line);
}

bool ReadFile::get_line(char* line)
{
	if (!fgets(line, MAX_LEN_LINE, fin))
		return false;

	char* com;
	if (com = strstr(line, str_comments)){
		line[com-line] = '\n';
		line[com-line+1] = 0;
	}

	return true;
}

bool ReadFile::process_line(char* line)
{
	if (!read_key_value(line))
		return false;

	process_key();
	return true;
}

bool ReadFile::read_key_value(char* line)
{
	if ( line[0] == chr_key ){
		char* str_value = strstr(line++, " ");
		if (str_value){
			sscanf_s(line, "%s", key, sizeof(key));
			strcpy(value, ++str_value);
		}
		else{
			strcpy(key, line);
			key[strlen(line)-1]=NULL;
			value[0]= NULL;
		}
	}
	else
		return false;

	return true;
}

void ReadFile::file_not_found()
{
	ERROR(222, "File not found: " + filename);
}

