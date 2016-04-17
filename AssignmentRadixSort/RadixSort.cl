//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef DataType
#define DataType int
#endif

#ifndef OFFSET
#define OFFSET (0)
#endif

#ifndef UnsignedDataType
#define UnsignedDataType unsigned int
#endif

// compute the histogram for each radix and each virtual processor for the pass
__kernel void histogram(
            const __global DataType* restrict d_Keys,
			      __global int*      restrict d_Histograms,
			const int pass,
			       __local int* loc_histo,
			const int n) {
  int it = get_local_id(0);  // i local number of the processor
  int ig = get_global_id(0); // global number = i + g I

  int gr = get_group_id(0); // g group number

  const int groups = get_num_groups(0);
  int items  = get_local_size(0);

  // initialize the local histograms to zero
  for(int ir = 0; ir < _RADIX; ir++) {
    loc_histo[ir * items + it] = 0;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  // range of keys that are analyzed by the work item
  int sublist_size  = n/groups/items; // size of the sub-list
  int sublist_start = ig * sublist_size; // beginning of the sub-list

  UnsignedDataType key;
  UnsignedDataType shortkey;
  int k;

  // compute the index
  // the computation depends on the transposition
  for(int j = 0; j < sublist_size; j++) {
    k = j + sublist_start;

    key = d_Keys[k] + OFFSET;

    // extract the group of _BITS bits of the pass
    // the result is in the range 0.._RADIX-1
	// _BITS = size of _RADIX in bits. So basically they
	// represent both the same. 
    shortkey=(( key >> (pass * _BITS)) & (_RADIX-1)); // _RADIX-1 to get #_BITS "ones"

    // increment the local histogram
    loc_histo[shortkey *  items + it ]++;
  }

  // wait for local histogram to finish
  barrier(CLK_LOCAL_MEM_FENCE);

  // copy the local histogram to the global one
  // in this case the global histo is the group histo.
  for(int ir = 0; ir < _RADIX; ir++) {
    d_Histograms[items * (ir * groups + gr) + it] = loc_histo[ir * items + it];
  }

  // TODO: Check if this barrier here is really necessary.
  barrier(CLK_GLOBAL_MEM_FENCE);
}

// each virtual processor reorders its data using the scanned histogram
__kernel void reorder(
    const __global DataType* restrict d_inKeys,
          __global DataType* restrict d_outKeys,
    const __global int* d_Histograms,
    const int pass,
          __global int* d_inPermut,
          __global int* d_outPermut,
          __local  int* loc_histo,
    const int n){

    int it = get_local_id(0);	// 
    int ig = get_global_id(0);	//

    int gr = get_group_id(0);				// 
    const int groups = get_num_groups(0);	// G: group count
    int items = get_local_size(0);			// group size

	int start = ig *(n / groups / items);   // eq. 2.1 : index of first elem this work-item processes
    int size  = n / groups / items;			//			 count of elements this work-item processes

    // take the histogram in the cache
    for (int ir = 0; ir < _RADIX; ir++){
        loc_histo[ir * items + it] =
            d_Histograms[items * (ir * groups + gr) + it];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

	int newpos;			// new position of element
	UnsignedDataType key;		// key element
	UnsignedDataType shortkey;	// key element within cache (cache line)
	int k;				// global position within input elements
	int newpost;		// new position of element (transposed)

    for (int j = 0; j < size; j++) {
        k = j + start;
        key = d_inKeys[k] + OFFSET;
        shortkey = ((key >> (pass * _BITS)) & (_RADIX - 1));	// shift element to relevant bit positions

        newpos = loc_histo[shortkey * items + it];
        newpost = newpos;

        d_outKeys[newpost] = key - OFFSET;

#ifdef PERMUT 
        d_outPermut[newpost] = d_inPermut[k];
#endif

        newpos++;
        loc_histo[shortkey * items + it] = newpos;
    }
}


// perform a parallel prefix sum (a scan) on the local histograms
// (see Blelloch 1990) each workitem worries about two memories
// see also http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
__kernel void scanhistograms(
    __global int* histo, 
    __local int* temp, 
    __global int* globsum) {
    int it = get_local_id(0);
    int ig = get_global_id(0);
    int decale = 1;
    int n = get_local_size(0) << 1;
    int gr = get_group_id(0);

    // load input into local memory
    // up sweep phase
    temp[(it << 1)]     = histo[(ig << 1)];
    temp[(it << 1) + 1] = histo[(ig << 1) + 1];

    // parallel prefix sum (algorithm of Blelloch 1990)
    // This loop runs log2(n) times
    for (int d = n >> 1; d > 0; d >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (it < d) {
            int ai = decale * ((it << 1) + 1) - 1;
            int bi = decale * ((it << 1) + 2) - 1;
            temp[bi] += temp[ai];
        }
        decale <<= 1;
    }

    // store the last element in the global sum vector
    // (maybe used in the next step for constructing the global scan)
    // clear the last element
    if (it == 0) {
        globsum[gr] = temp[n - 1];
        temp[n - 1] = 0;
    }

    // down sweep phase
    // This loop runs log2(n) times
    for (int d = 1; d < n; d <<= 1){
        decale >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);

        if (it < d){
            int ai = decale*((it << 1) + 1) - 1;
            int bi = decale*((it << 1) + 2) - 1;

            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // write results to device memory
    histo[(ig << 1)]       = temp[(it << 1)];
    histo[(ig << 1) + 1]   = temp[(it << 1) + 1];

    // TODO: Check if this barrier here is really necessary.
    barrier(CLK_GLOBAL_MEM_FENCE);
}

// use the global sum for updating the local histograms
// each work item updates two values
__kernel void pastehistograms(
          __global int* restrict histo, 
    const __global int* restrict globsum) {
    int ig = get_global_id(0);
    int gr = get_group_id(0);

    int s = globsum[gr];

    // write results to device memory
    histo[(ig << 1)]     += s;
    histo[(ig << 1) + 1] += s;

    // TODO: Check if this barrier here is really necessary.
    barrier(CLK_GLOBAL_MEM_FENCE);
}