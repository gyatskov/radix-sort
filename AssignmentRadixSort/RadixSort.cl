// 
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void RadixSortReadWrite(__global int* array)
{
	uint GID = get_global_id(0);

	uint readIdx = GID;
	uint writeIdx = readIdx;

	array[writeIdx] *= array[readIdx];
}

__kernel void RadixSort(__global int* input, __global int* output)
{
    uint GID = get_global_id(0);

    uint readIdx = GID;
    uint writeIdx = readIdx;

    int val = input[readIdx];
    output[writeIdx] = val*val;
}