import numpy
arr = numpy.empty((13000, 2), dtype=numpy.float32)
arr[:2000] = numpy.random.rand(2000, 2) + [0, 0.5]
arr[2000:4000] = numpy.random.rand(2000, 2) + [0, 1.5]
arr[4000:6000] = numpy.random.rand(2000, 2) - [0, 0.5]
arr[6000:8000] = numpy.random.rand(2000, 2) + [0.5, 0]
arr[8000:10000] = numpy.random.rand(2000, 2) - [0.5, 0]
arr[10000:] = numpy.random.rand(3000, 2) * 5 - [2, 2]
myarr = arr.tolist()
f = open("testfile.txt","w")
for i,x in enumerate(myarr):
    f.write(str(i+1)+" ")
    n_features = len(x)
    for j in range(n_features):
        f.write(str(x[j]))
        if(j != n_features-1):
            f.write(" ")
    f.write("\n")
f.close()
