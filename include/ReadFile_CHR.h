/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\
*  Copyright (c) 2019 <GTEP> - All Rights Reserved                            *
*  This file is part of HERMES Project.                                       *
*  Unauthorized copying of this file, via any medium is strictly prohibited.  *
*  Proprietary and confidential.                                              *
*                                                                             *
*  Developers:                                                                *
*     - Bismarck G. Souza Jr <bismarck@puc-rio.br>                            *
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#ifndef _READFILE_CHR_H_
#define _READFILE_CHR_H_

#include <string>
#include <fstream>
#include <vector>
#include <sstream>
#include "ReadFile.h"
#include "defs.h"

using namespace std;

enum { FIXED_X = 1,
	   FIXED_Y = 2,
	   FIXED_Z = 4};

enum Esides { LEFT, RIGHT, FRONT, BACK, UP, DOWN };

enum Eprops { YOUNG, POISSON, DENSITY, K0X, K0Y, NMATERIALS };

enum Edirections { X, Y, Z };

/**
Read a file with the pattern:

%KEY1
VALUE1  %% COMMENTS 1

%KEY2
VALUE2  %% COMMENTS 2
*/
class ReadFile_CHR : public ReadFile
{

public:
	ReadFile_CHR();
	
	int get_nNodes() { return coords[0].size(); }
	int get_nOffsets() { return offsets.size(); }
	int get_nDofNode() { return nDofNode; }
	int get_nSupports() { return supports_ids.size(); }
	int get_nElements() { return elements_ids.size(); }
	int get_nMaterials() { return nMaterials; }
	int get_dim(int i) { return dim[i]; }
	int get_element_pos(int ele) { return elements_ids[ele]; }
	int get_extension_size(int i) { return extension_sizes[i]; }	
	int get_nColorGroups(int gpu) { return coloring_groups[gpu].size(); }
	int* ptr_offsets() { return &offsets[0]; } 
	int* ptr_supports_ids() { return &supports_ids[0]; }
	int* ptr_elements_ids() { return &elements_ids[0]; } 
	int* ptr_materials_ids() { return &materials_ids2[0]; }
	int* ptr_connects(int pos) { return &connects2[pos][0]; }
	double get_prop(int ele, Eprops prop) { return materials[prop][materials_ids[ele]-1]; } 
	double get_coord(int node, Edirections dir) { return coords[dir][node]; }
	double get_coord(int ele, int pos, Edirections dir) { return coords[dir][connects[pos][ele]-1]; }
	double* ptr_coord(int dir) { return &coords[dir][0]; }
	double* ptr_prop(int prop) { return &materials[prop][0]; }
	vector<int> get_extension_sizes() { return extension_sizes; }	
	vector<int> get_color_group(int gpu, int pos) { return coloring_groups[gpu][pos]; }

	vector<int> get_connects(); 
	vector<int> get_supports();
	

protected:
	int nDofNode;							//< Number of degree of freedom per node
	int nMaterials;							//< Number of materials
	vector<int> dim;						//< Dimensions: nx, ny, nz
	vector<int> gpus;						//< Used GPUs indexes
	vector<int> extension_sizes;			//< Sizes: LEFT, RIGHT, FRONT, BACK, UP, DOWN
	vector<int> offsets;					//< Offsets for sparse matrix (diagonal format)
	vector<int> elements_ids;				//< Elements order in input file
	vector<int> supports_ids;				//< Nodes with restrictions
	vector<int> supports_vals;				//< Restrictions values
	vector<int> materials_ids;				//< Materials indexes
	vector<int> materials_ids2;				//< Materials indexes (read order)
	vector<vector<int>> connects;			//< Connections: [position of node][element]
	vector<vector<int>> connects2;			//< Connections: [position of node][element] (read order)
	vector<vector<double>> loads;			//< Nodal loads: [node][direction]
	vector<vector<double>> coords;			//< Coordinates of nodes: [direction][node]
	vector<vector<double>> materials;		//< Material properties: [prop][element]
	stringstream append_txt;				//< Text to be append of input file

	//Groups of non-neighbors elements: [gpu][color_id][ele]
	vector<vector<vector<int>>> coloring_groups;

	void finally();
	void process_key();
	void process_loads(int size);
	void process_supports(int size);
	void process_coordinates(int size);
	void process_elements(int size, string elem_type);
	void process_materials(int size, string mat_type);
	void create_offsets();
	void create_coloring_groups();
	void rewrite_input_file(bool);
};

#endif