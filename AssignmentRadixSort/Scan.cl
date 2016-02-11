


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Scan_Naive(const __global uint* inArray, __global uint* outArray, uint N, uint offset) 
{
	// TO DO: Kernel implementation
}



// Why did we not have conflicts in the Reduction? Because of the sequential addressing (here we use interleaved => we have conflicts).

#define UNROLL
#define NUM_BANKS			32
#define NUM_BANKS_LOG		5
#define SIMD_GROUP_SIZE		32

// Bank conflicts
#define AVOID_BANK_CONFLICTS
#ifdef AVOID_BANK_CONFLICTS
	// TO DO: define your conflict-free macro here
#else
	#define OFFSET(A) (A)
#endif

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Scan_WorkEfficient(__global uint* array, __global uint* higherLevelArray, __local uint* localBlock) 
{
	// TO DO: Kernel implementation
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Scan_WorkEfficientAdd(__global uint* higherLevelArray, __global uint* array, __local uint* localBlock) 
{
	// TO DO: Kernel implementation (large arrays)
	// Kernel that should add the group PPS to the local PPS (Figure 14)
}