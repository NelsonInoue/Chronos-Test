#ifndef	_CPU_H
#define	_CPU_H

//==============================================================================
// cCpu
//==============================================================================
typedef class cCpu *pcCpu, &rcCpu;

class cCpu
{
public:
	
	cCpu();
	static pcCpu Cpu;
	static int AnalyzeCPUNaive();

};

#endif