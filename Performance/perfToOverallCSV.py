out = open("performance.csv", 'w')
out.write("NumElements,Datatype,Dataset,avgHistogram,avgScan,avgPaste,avgReorder,avgTotalGPU,avgTotalSTLCPU,avgTotalRDXCPU\n")

for i in range(0,26):
	fname = "input_test_2_"+str(i)+".txt"
	# print("reading file "+fname+"\n")
	f = open(fname)
	lines = f.readlines()
	f.close()
	
	for i in range(0, len(lines)):
		words = lines[i].split(',')
		if words[0] != "NumElements":
			continue
		out.write(lines[i+1])

out.close()

