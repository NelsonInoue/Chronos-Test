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
#include "ReadFile_CHR.h"
#include <iomanip>
#include <list>
#include <set>
#include <algorithm>


list<int>::iterator find_and_erase(list<int>::iterator ini, list<int> &mylist, int val){
	auto itt = find(ini, mylist.end(), val);

	if (itt != mylist.end())
		return mylist.erase(itt);
	else
		return ini;
}



ReadFile_CHR::ReadFile_CHR() : ReadFile()
{
	nDofNode = 3;
	reservoir_size.resize(3, 0);
	chr_key = '%';
	str_comments = "%%";
}

vector<int> ReadFile_CHR::get_connects() 
{ 
	int ele, nElements = get_nElements();
	vector<int> connects_out(11*nElements);
	copy(elements_ids.begin(), elements_ids.end(), connects_out.begin());

	for (int i=0; i < nElements; ++i){
		ele = elements_ids[i]-1;

		connects_out[i+1*nElements] = materials_ids[ele];
		connects_out[i+2*nElements] = 1;

		for (int pos=0; pos < 8; ++pos ){
			connects_out[i+(pos+3)*nElements] = connects[pos][ele];
		}
	}

	return connects_out;
}

vector<int> ReadFile_CHR::get_supports()
{
	int nSupports = get_nSupports();

	vector<int> supp(4*nSupports);

	for (int i=0; i < nSupports; ++i){
		supp[i + 0*nSupports] = supports_ids[i];
		supp[i + 1*nSupports] = supports_vals[i] & FIXED_X;
		supp[i + 2*nSupports] = supports_vals[i] & FIXED_Y ? 1 : 0;
		supp[i + 3*nSupports] = supports_vals[i] & FIXED_Z ? 1 : 0;
	}

	return supp;
}


void ReadFile_CHR::process_key()
{
	char line[MAX_LEN_LINE];
	int size;
	get_line(line);

	if (strcmp(key,"NODE.COORD")==0){
		sscanf_s(line, "%d", &size);
		process_coordinates(size);
	}

	else if (strcmp(key, "NODE.SUPPORT")==0 ){
		sscanf_s(line, "%d", &size);
		process_supports(size);
	}

	else if (strncmp(key, "MATERIAL.", 9) ==0){
		sscanf_s(line, "%d", &size);
		process_materials(size, key+9);
	}

	else if (strncmp(key, "ELEMENT.", 8)==0){
		sscanf_s(line, "%d", &size);
		process_elements(size, key+8);
	}

	else if (strcmp(key, "GEOMETRY.SIZES")==0){
		sscanf_s(line, "%d", &size);
		extension_sizes.resize(6,0);
		sscanf_s(line, "%d %d %d %d %d %d %d %d %d", 
			&reservoir_size[X], &reservoir_size[Y], &reservoir_size[Z],
			&extension_sizes[LEFT], &extension_sizes[RIGHT], &extension_sizes[FRONT], 
			&extension_sizes[BACK], &extension_sizes[UP], &extension_sizes[DOWN ]);
	}

	else if (strcmp(key, "OFFSETS")==0){
		sscanf_s(line, "%d", &size);
		offsets.resize(size);

		for (int i=0; i < size; ++i)
			fscanf(fin, "%d", &offsets[i]);
	}

	else if (strcmp(key, "COLORING")==0){
		int size_group;
		auto it = elements_ids.begin();
		vector<vector<int>> coloring_groups_1gpu;
	    fscanf(fin, "%d", &size);
		
		for (int i=0; i < size; ++i) {
			fscanf(fin, "%d", &size_group);

			coloring_groups_1gpu.push_back(vector<int>(it, it+size_group));

			it += size_group;
		}
		coloring_groups.push_back(coloring_groups_1gpu);
	}

	else if (strcmp(key, "LOAD.CASE.NODAL.FORCE")==0) {
		sscanf_s(line, "%d", &size);
		process_loads(size);
	}

	else if (strcmp(key, "NUMBER.GPUS")==0) {
		int aux;
		if (sscanf_s(line, "%d %d", &size, &aux)==1){
			gpus.resize(size);

			for (int i=0; i < size; ++i)
				if (fscanf(fin, "%d", &gpus[i])==0) {
					WRITE_ERROR(103, "Invalid value for NUMBER.GPUS key.")	;
					break;
				}
		}
		else
			gpus.resize(size, aux);
	}
}

void ReadFile_CHR::finally()
{
	// Check adjacent rocks
	if (extension_sizes.size() == 0)
		ERROR(58, "Key GEOMETRY.SIZES not found in file "+filename);
	
	// Check GPUs specifications
	if (gpus.size() == 0)
		gpus.resize(1, 0);
	
	// Check offstes
	if (offsets.size() == 0){
		create_offsets();
	}

	// Check coloring groups
	bool rewrite_elements = false;
	if (coloring_groups.size() == 0){
		create_coloring_groups();
		rewrite_elements = true;
	}

	// Rewrite chr
	if (rewrite_elements || append_txt.rdbuf()->in_avail()!=0) 
		rewrite_input_file(rewrite_elements);
}

void ReadFile_CHR::process_coordinates(int size)
{
	int i;
	double x, y, z;
	char line[MAX_LEN_LINE];
	coords.resize(3, vector<double>(size));

	for (int id=0; id < size; ++id){
		get_line(line);
		sscanf_s(line, "%d %lf %lf %lf", &i, &x, &y, &z);
		i--;
		coords[X][i] = x;
		coords[Y][i] = y;
		coords[Z][i] = z;
	}
}

void ReadFile_CHR::process_supports(int size)
{
	int id, fixed_x, fixed_y, fixed_z;
	char line[MAX_LEN_LINE];

	supports_ids.resize(size);
	supports_vals.resize(size);

	for (int i=0; i < size; ++i){
		get_line(line);
		sscanf_s(line, "%d %d %d %d", &id, &fixed_x, &fixed_y, &fixed_z);

		supports_ids[i] = id;
		supports_vals[i] = fixed_x*FIXED_X+fixed_y*FIXED_Y+fixed_z*FIXED_Z;
	}
}

void ReadFile_CHR::process_materials(int size, string mat_type)
{
	char line[MAX_LEN_LINE];
	int i;
	double E, v, rho, k0x, k0y;

	nMaterials = 0;
	materials.resize(NMATERIALS, vector<double>(size));

	if (mat_type == "ISOTROPIC"){
		for (int id=0; id < size; ++id){
			get_line(line);
			sscanf_s(line, "%d %lf %lf %lf %lf %lf", &i, &E, &v, &rho, &k0x, &k0y);
			i--;
			materials[YOUNG][i] = E;
			materials[POISSON][i] = v;
			materials[DENSITY][i] = rho;
			materials[K0X][i] = k0x;
			materials[K0Y][i] = k0y;
			nMaterials++;
		}
	}
}

void ReadFile_CHR::process_elements(int size, string elem_type)
{
	int i, aux, mat, nNodes, cont;
	char line[MAX_LEN_LINE];

	if (elem_type == "BRICK8"){
		nNodes = 8;
		materials_ids.resize(materials_ids.size()+size);
		materials_ids2.resize(materials_ids2.size()+size);
		connects.resize(nNodes);
		connects2.resize(nNodes);
		elements_ids.reserve(elements_ids.size()+size);
		for (int pos=0; pos < nNodes; ++pos){
			connects[pos].resize(connects[pos].size()+size);
			connects2[pos].resize(connects2[pos].size()+size);
		}

		// Read all elements
		cont = 0;
		for (int id=0; id < size; ++id){
			// Read element number, material id and integration order
			fscanf(fin, "%d %d %d", &i, &mat, &aux);
			elements_ids.push_back(i);

			i--;
			materials_ids[i] = mat;
			materials_ids2[cont] = mat;

			// Read nodes
			for (int node=0; node < nNodes; ++node) {
				fscanf(fin, " %d", &aux);
				connects[node][i] = aux;
				connects2[node][cont] = aux;
			}
			cont++;
		}
	}
}

void ReadFile_CHR::process_loads(int size) 
{
	int i;
	double fx, fy, fz;
	char line[MAX_LEN_LINE];
	loads.resize(size, vector<double>(3));

	for (int id=0; id < size; ++id){
		get_line(line);
		sscanf_s(line, "%d %lf %lf %lf", &i, &fx, &fy, &fz);
		i--;
		loads[i][X] = fx;
		loads[i][Y] = fy;
		loads[i][Z] = fz;
	}
}

void ReadFile_CHR::create_offsets()
{
	
	printf("\n\n         Creating offsets sparse matrix \n");
	printf("         ========================================= \n");

	int pos1, pos2, node1, node2, elem_nNodes, offset;
	vector<int> offsets_base(1, 1);	

	for (int ele=0; ele < get_nElements(); ++ele){
		elem_nNodes = connects.size();
			
		for (pos1=0; pos1 < elem_nNodes; ++pos1){
			node1 = connects[pos1][ele]-1;

			for(pos2=pos1+1; pos2 < elem_nNodes; ++pos2){
				node2 = connects[pos2][ele]-1;

				if (node1 > node2+1)
					offset = node1 - node2;
				else if (node2 > node1+1)
					offset = node2 - node1;
				else
					continue;

				if (find(offsets_base.begin(), offsets_base.end(), offset) == offsets_base.end())
					offsets_base.push_back(offset);
			}
		}
	}

	// Sort vector
	sort(offsets_base.begin(), offsets_base.end());

	// Create positive offset vector
	int size = offsets_base.size();
	vector<int> offsets_positive(2);
	offsets_positive[0] = 1;
	offsets_positive[1] = 2;
	offsets_positive.reserve(5*size);

	int last_offset = 0;
	for (auto it=offsets_base.begin(); it != offsets_base.end(); ++it){
		offset = 3*(*it);
		if (last_offset < *it -1){
			offsets_positive.push_back(offset-2);
			offsets_positive.push_back(offset-1);
		}
		offsets_positive.push_back(offset);
		offsets_positive.push_back(offset+1);
		offsets_positive.push_back(offset+2);
		last_offset = *it;
	}

	// Create offset vector
	size = offsets_positive.size();
	offsets.resize(2*size+1, 0);

	int pos=size+1;
	int neg=size-1;
	for (auto it=offsets_positive.begin(); it != offsets_positive.end(); ++it){
		offsets[pos++] = *it;
		offsets[neg--] = -*it;
	}

	// Prepare to write
	append_txt << "%OFFSETS\n";
	append_txt << offsets.size() << endl;
	for (auto it=offsets.begin(); it != offsets.end();  ++it)
		append_txt << " " << *it << endl;

	append_txt << endl;
}	

void ReadFile_CHR::create_coloring_groups()
{
	printf("\n\n         Creating coloring groups \n");
	printf("         ========================================= \n");

	vector<vector<int>> coloring_groups_1gpu;

	// Create a list of neighbors
	vector<list<int>>list_nodes(get_nNodes()+1);  // node N is a node of elements in list_nodes[N]
	list<int> *list_node, mylist;
	vector<vector<int>>neighbors(get_nElements()+1);
	vector<int> vec;

	for (int ele=0; ele < get_nElements(); ++ele){
		mylist.clear();
		for(auto con=connects.begin(); con!=connects.end(); ++con){
			list_node = &list_nodes[(*con)[ele]];

			// Including all elements that has a node in commum
			mylist.insert(mylist.begin(), list_node->begin(), list_node->end());

			list_node->push_back(ele+1);
		}
		mylist.unique(); //< All neighboring elements
		
		// Add neighbors if is not in list
		for(auto it=mylist.begin(); it != mylist.end(); ++it){
			if (find(neighbors[*it].begin(), neighbors[*it].end(), ele+1) == neighbors[*it].end())
				neighbors[*it].push_back(ele+1);
		}
	}

	// Create non-neighbors groups
	int id, pos;
	list<int> group; 
	vector<bool> used (get_nElements()+1, false);
	list<int>::iterator it_find, it_group;
	vector<int>::iterator it_neigh;

	id = 0;
	while (++id <= get_nElements()){
		if (used[id])
			continue;

		group.clear();

		// Create a list of possibilities for a group
		pos = 0;
		vec = neighbors[id];
		vec.push_back(get_nElements()+1);
		for(auto it=vec.begin(); it != vec.end(); ++it) {
			while (++pos < *it){
				if (!used[pos])
					group.push_back(pos); 
			}
		}
		used[id] = true;

		// Remove all neighbors from group
		it_group = group.begin();
		while ( ++it_group != group.end() ){
			it_neigh = neighbors[*it_group].begin();
			it_find = it_group;

			while ( it_neigh != neighbors[*it_group].end()){
				if (*it_neigh > *it_group)
					it_find = find_and_erase(it_find, group, *it_neigh);

				++it_neigh;
			}

			used[*it_group] = true;
		}
	
		coloring_groups_1gpu.push_back(vector<int>(group.begin(), group.end()));
	}

	coloring_groups.push_back(coloring_groups_1gpu);

	// Prepare to write
	append_txt << "%COLORING\n";
	append_txt << coloring_groups.size() << endl;
	for (int gpu=0; gpu < coloring_groups.size(); ++gpu){
		append_txt << " " << coloring_groups[gpu].size() << " ";
		for (auto it=coloring_groups[gpu].begin(); it != coloring_groups[gpu].end();  ++it)
			append_txt << " " << (*it).size();
		append_txt << endl;
	}
	append_txt << endl;
}

void ReadFile_CHR::rewrite_input_file(bool rewrite_elements)
{
	string tmpName = path+"/"+basename + "_tmp.chr";
	string realName = (path+"/"+filename);
	ofstream fout(tmpName);
	FILE* fin2;
	fopen_s( &fin2, realName.c_str(), "r" );

	printf("\n\n         Rewriting chr input file in coloring format \n");
	printf("         ========================================= \n\n\n");

	char line[MAX_LEN_LINE];

	while (fgets(line, MAX_LEN_LINE, fin2)) {

		if (rewrite_elements && strstr(line, "%ELEMENT.BRICK8") == line) {
			fout << line;
			fout << get_nElements() << endl;
			
			for (auto it=coloring_groups[0].begin(); it != coloring_groups[0].end(); ++it){
				for (auto ele=(*it).begin(); ele != (*it).end(); ++ele) {
					fout << " " << *ele << " " << materials_ids[*ele-1] << " 1";
					for (auto con=connects.begin(); con != connects.end(); ++con) 
						fout << " " << (*con)[*ele-1];
					fout << endl;
				}
			}
			fout << endl;

			while (fgets(line, MAX_LEN_LINE, fin2)) {
				if (line[0] == '%' && line[1] != '%')
					break;
			}
		}

		if (strstr(line, "%END") == line && append_txt.rdbuf()->in_avail()!=0) 
			fout << append_txt.str();

		fout << line;
	}

	fclose(fin2);
	fout.close();

	rename(realName.c_str(), (path+"/"+basename + "_orig.chr").c_str());
	rename(tmpName.c_str(), realName.c_str());
}
