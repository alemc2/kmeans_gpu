import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import argparse
import timeit

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', nargs='?', required=True, 
        help='file to read data from', type=str)
parser.add_argument('-c', '--num_clusters', nargs='?', required=True,
        help='number of clusters', type=int)
parser.add_argument('-v', '--visualize', action='store_true', default=False, help='Visualize')
args = parser.parse_args()

tst = pd.read_table(args.file,sep=' ',header=None, index_col=0)
mem = pd.read_table(args.file + '.membership',sep=' ',header=None, index_col=0)
clusters = pd.read_table(args.file + '.cluster_centres',sep=' ',header=None, index_col=0)

if args.visualize:
    plt.scatter(tst[1],tst[2], c = mem[1])
    plt.scatter(clusters[1],clusters[2], c = np.arange(clusters.shape[0]), marker='x',s=50)
    plt.pause(0.5)

init = pd.read_table(args.file + '.initcentroids',sep=' ',header=None, index_col=0)

km = KMeans(n_clusters=args.num_clusters, init=init, max_iter=500, n_init=1, tol=0.001)
km_par = KMeans(n_clusters=args.num_clusters, init=init, max_iter=500, n_init=1,
        tol=0.001, n_jobs=-1)
start_time = timeit.default_timer()
km_par.fit(tst)
par_end_time = timeit.default_timer() - start_time
print('parallel kmeans ran for %f seconds' % par_end_time)
start_time = timeit.default_timer()
km.fit(tst)
ser_end_time = timeit.default_timer() - start_time
print('serial kmeans ran for %f seconds' % ser_end_time)
