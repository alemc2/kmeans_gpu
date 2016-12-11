#Paste your sample file by name testfile.txt in the same folder as
#the code
import numpy as np
f = open("testfile.txt","r")
a = f.readlines()
y = range(len(a))
np.random.shuffle(y)
f.close()
centroids = []
k = 50
for value in range(k):
    centroids.append(a[y[value]].split()[1:])
f = open("initialclusterfile.txt","w")
for i,each in enumerate(centroids):
    f.write(str(i+1))
    f.write(" ")
    f.write(str(each[0]))
    f.write(" ")
    f.write(str(each[1]))
    f.write("\n")
f.close()
