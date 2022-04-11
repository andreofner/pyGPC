"""
This script comes from the RTRBM code by Ilya Sutskever from
http://www.cs.utoronto.ca/~ilya/code/2008/RTRBM.tar
"""

# usage
# python3 gen_bouncing_ball_video.py num_frames num_seq

from numpy import *
from scipy import *
import pdb
#import cPickle as pickle
import _pickle as pickle
import scipy.io
import sys
import hickle as hkl

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

shape_std=shape
def shape(A):
    if isinstance(A, ndarray):
        return shape_std(A)
    else:
        return A.shape()

size_std = size
def size(A):
    if isinstance(A, ndarray):
        return size_std(A)
    else:
        return A.size()

det = linalg.det

def new_speeds(m1, m2, v1, v2):
    new_v2 = (2*m1*v1 + v2*(m2-m1))/(m1+m2)
    new_v1 = new_v2 + (v2 - v1)
    return new_v1, new_v2


def norm(x): return sqrt((x**2).sum())
def sigmoid(x):        return 1./(1.+exp(-x))

SIZE=10
# size of bounding box: SIZE X SIZE.

def bounce_n(T=128, n=2, r=None, m=None):
    # if r==None: r=array([1.2]*n)
    if m==None: m=array([1]*n)
    # r is to be rather small.
    X=zeros((T, n, 2), dtype='float')
    v = randn(n,2)
    v = v / norm(v)*.5
    good_config=False
    while not good_config:
        x = 2+rand(n,2)*8
        good_config=True
        for i in range(n):
            for z in range(2):
                if x[i][z]-r[i]<0:      good_config=False
                if x[i][z]+r[i]>SIZE:     good_config=False

        # that's the main part.
        for i in range(n):
            for j in range(i):
                if norm(x[i]-x[j])<r[i]+r[j]:
                    good_config=False


    eps = .5
    for t in range(T):
        # for how long do we show small simulation

        for i in range(n):
            X[t,i]=x[i]

        for mu in range(int(1/eps)):

            for i in range(n):
                x[i]+=eps*v[i]

            for i in range(n):
                for z in range(2):
                    if x[i][z]-r[i]<0:  v[i][z]= abs(v[i][z]) # want positive
                    if x[i][z]+r[i]>SIZE: v[i][z]=-abs(v[i][z]) # want negative


            for i in range(n):
                for j in range(i):
                    if norm(x[i]-x[j])<r[i]+r[j]:
                        # the bouncing off part:
                        w    = x[i]-x[j]
                        w    = w / norm(w)

                        v_i  = dot(w.transpose(),v[i])
                        v_j  = dot(w.transpose(),v[j])

                        new_v_i, new_v_j = new_speeds(m[i], m[j], v_i, v_j)

                        v[i]+= w*(new_v_i - v_i)
                        v[j]+= w*(new_v_j - v_j)

    return X

def ar(x,y,z):
    return z/2+arange(x,y,z,dtype='float')

def matricize(X,res,r=None):

    T, n= shape(X)[0:2]
    # if r==None: r=array([1.2]*n)

    A=zeros((T,res,res), dtype='float')

    [I, J]=meshgrid(ar(0,1,1./res)*SIZE, ar(0,1,1./res)*SIZE)

    for t in range(T):
        for i in range(n):
            A[t]+= exp(-(  ((I-X[t,i,0])**2+(J-X[t,i,1])**2)/(r[i]**2)  )**4    )

        A[t][A[t]>1]=1
    return A

def bounce_mat(res, n=2, T=128, r =None):
    if r==None: r=array([1.2]*n)
    x = bounce_n(T,n,r);
    A = matricize(x,res,r)
    return A,X

def bounce_vec(res, n=2, T=128, r =None, m =None):
    if r==None: r=array([1.2]*n)
    x = bounce_n(T,n,r,m);
    V = matricize(x,res,r)
    return V.reshape(T, res**2), x

# make sure you have this folder
logdir = './sample'
def show_sample(V):
    T   = len(V)
    res = int(sqrt(shape(V)[1]))
    for t in range(T):
        plt.imshow(V[t].reshape(res,res),cmap=matplotlib.cm.Greys_r)
        # Save it
        fname = logdir+'/'+str(t)+'.png'
        plt.savefig(fname)

if __name__ == "__main__":

    T = 1000    # 10
    N = 10#0000 # 10000
    balls = 1

    print('T={:d}, N={:d}'.format(T,N))
    res=32

    dat=zeros((N,T,res*res), dtype=float32)
    x = zeros((N,T,balls,2),dtype=float32) # ball locations
    print(dat.shape)
    for i in range(N):
    	# print i
        dat[i,:,:], x[i,:,:,:]=bounce_vec(res=res, n=balls, T=T)
    #hkl.dump(dat, 'training.hkl', mode='w', compression='gzip')
    #hkl.dump(x, 'training_locs.hkl', mode='w', compression='gzip')

    import numpy as np
    np.save("./sample/train.npy", dat)

    if False:
        Nv=int(N/20)
        dat=zeros((Nv,T,res*res), dtype=float32)
        x = zeros((Nv,T,3,2),dtype=float32) # ball locations
        for i in range(Nv):
            # print i
            dat[i,:,:], x[i,:,:,:]=bounce_vec(res=res, n=3, T=T)
        hkl.dump(dat, 'val.hkl', mode='w', compression='gzip')
        hkl.dump(x, 'val_locs.hkl', mode='w', compression='gzip')

        Nt=int(N/20)
        dat=zeros((Nt,T,res*res), dtype=float32)
        x = zeros((Nt,T,3,2),dtype=float32) # ball locations
        for i in range(Nt):
            # print i
            dat[i,:,:], x[i,:,:,:]=bounce_vec(res=res, n=3, T=T)
        hkl.dump(dat, 'test.hkl', mode='w', compression='gzip')
        hkl.dump(x, 'test_locs.hkl', mode='w', compression='gzip')


        # show one video
        #show_sample(dat[1])
        #!ffmpeg -start_number 0 -i %d.png -c:v libx264 -pix_fmt yuv420p -r 30 sample.mp4

