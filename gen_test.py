#!/usr/bin/python3
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--num_samples', nargs='?', required=True, 
        help='number of samples', type=int)
parser.add_argument('-f', '--num_features', nargs='?', required=True,
        help='number of features', type=int)
parser.add_argument('-o', '--outfile', nargs='?', required=True, 
        help='file to output', type=str)
parser.add_argument('--offset', nargs='?', default=0, help='mean to offset by',
        type=float)
parser.add_argument('--scale', nargs='?', default=10, help='amount to scale by',
        type=float)
args = parser.parse_args()

tst_data = np.random.randn(args.num_samples,args.num_features)
tst_data = tst_data*args.scale + args.offset
with open(args.outfile,'w') as f:
    for i,x in enumerate(tst_data.tolist()):
        f.write(str(i+1)+" ")
        n_features = len(x)
        for j in range(n_features):
            f.write(str(x[j]))
            if(j != n_features-1):
                f.write(" ")
        f.write("\n")
