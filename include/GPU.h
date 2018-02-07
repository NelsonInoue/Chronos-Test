#ifndef	_GPU_H
#define	_GPU_H

typedef struct {
	
	int LocalNumTerm, Localnumno, Localnumno_0, Localnumno_1, Localnumno_2, Localnumno_3, DeviceNumber, DeviceId;
	int Localnumel;
	double delta_new;
	double *Matrix_K_h_Pt, *Vector_B_h_Pt, *Vector_X_h_Pt, *Matrix_M_h_Pt, *Delta_New_GPU_h_Pt, *Delta_Aux_GPU_h_Pt;
	double *Vector_D_h_A_Pt, *Vector_D_h_P_Pt;
	int   *Vector_I_h_Pt, *eventId_1_h_Pt, *eventId_2_h_Pt;
	double *CoordElem_h_Pt, *Strain_h_Pt;
	int   GlobalNumRow, GlobalNumTerm, BlockSize, RowSize, NumMultiProcessor;
	double *MatPropQ1_h_Pt, *MatPropQ2_h_Pt, *MatPropQ3_h_Pt, *MatPropQ4_h_Pt, *MatPropQ5_h_Pt, *MatPropQ6_h_Pt, *MatPropQ7_h_Pt, *MatPropQ8_h_Pt;
	double *CoordQ1_h_Pt, *CoordQ2_h_Pt, *CoordQ3_h_Pt, *CoordQ4_h_Pt, *CoordQ5_h_Pt, *CoordQ6_h_Pt, *CoordQ7_h_Pt, *CoordQ8_h_Pt;
	int   *iiPosition_h_Pt, *Connect_h_Pt;

	//Partial sum for this GPU
	double *h_Sum;

} GPU_Struct;

//==============================================================================
// cGPU
//==============================================================================
typedef class cGPU *pcGPU, &rcGPU;

class cGPU
{
public:
	
	cGPU();
	int AnalyzeMultiGPU();
	static pcGPU GPU;

};


#endif

