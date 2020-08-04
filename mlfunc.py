# This code was developed by Man Yi Yim (manyi.yim@gmail.com) under Python 2.7.13.
# See https://github.com/myyim/SVM_weight_constraints for more details on SVM.

import numpy as np
import pylab
import itertools
exec(open('mathfunc.py').read())

def cover(n,p):
    """cover(n,p) returns the number of linearly separable pattern sets out of p sets given n inputs in general position based on Cover's counting theorem"""
    temp = 0
    for j in range(np.min([n,p])):
        temp += 2*nCr(p-1,j)
    return temp

def perceptron(u,yloc,usetheta=0):
    """perceptron(u,yloc,usetheta=0) returns 1 if the input u with positive labels yloc is realizable, and 0 otherwise.
        Note that yloc is a list of positive labels."""
    eta = 1 # learning rate
    Nshot = 100 # maximum number of iterations
    Ng = u.shape[0]
    R = u.shape[1]
    y = np.zeros(R)
    y[yloc] = 1
    w = np.zeros(Ng)
    theta = 0.
    sigma = np.zeros(R)
    itrial = 0
    while itrial < Nshot and np.sum(np.abs(y-sigma))>0:
        for ix in range(R):
            sigma[ix] = (np.dot(w,u[:,ix])-theta) > 0
            w += eta*(y[ix]-sigma[ix])*u[:,ix]
            if usetheta == 1:
                theta += -eta*(y[ix]-sigma[ix])
        itrial += 1
        sloc = np.argwhere(np.dot(w,u)-theta>0).T[0]
    if len(sloc)!=len(yloc) or np.sum(sloc==yloc)!=len(yloc):
        return 0
    else:
        return 1

def svm_margin(X,Y):
    """svm_margin(X,Y) return the SVM maximum margin, the corresponding weights and threshold using sklearn function SVC."""
    num = X.shape[0]
    if Y.shape[0] != num:
        print('Dimensions mismatched!')
        return
    dim = X.shape[1]
    w = np.zeros(dim)
    from sklearn import svm
    hyp = svm.SVC(kernel='linear',C=10000,cache_size=20000,tol=1e-5)
    hyp.fit(X,Y)
    for j in range(hyp.support_.size):
        w += hyp.dual_coef_[0][j]*hyp.support_vectors_[j]
    return 2./pylab.norm(w),w,hyp.intercept_[0]

def svm_qp(x,y,is_thre=1,is_wconstrained=1):
    """svm_qp(x,y,is_thre=1,is_wconstrained=1) returns the SVM maximum margin and the corresponding weights (and threshold if any). x is the input matrix with dimension N (number of neurons) by P (number of patterns). y is the desired output vector of dimension P. y is either -1 or 1. """
    import qpsolvers
    R = x.shape[1]
    G = -(x*y).T
    if is_thre:
        N = x.shape[0] + 1
        G = np.append(G.T,y)
        G = G.reshape(N,R)
        G = G.T
        P = np.identity(N)
        P[-1,-1] = 1e-14    # regularization; may have to play around
        #for j in range(N):
        #    P[j,j] += 1e-16
        #P += 1e-10
    else:
        N = x.shape[0]
        P = np.identity(N)
    if is_wconstrained:
        if is_thre:
            G = np.append(G,-np.identity(N)[:N-1,:])
            G = G.reshape(R+N-1,N)
            h = np.array([-1.]*R+[0]*(N-1))
        else:
            G = np.append(G,-np.identity(N))
            G = G.reshape(R+N,N)
            h = np.array([-1.]*R+[0]*N)
    else:
        h = np.array([-1.]*R)
    #print(G)
    w = qpsolvers.solve_qp(P,np.zeros(N),G,h)
    #w = qpsolvers.solve_qp(np.identity(N),np.zeros(N),G,h,np.zeros(N),0) #CVXOPT,qpOASES,quadprog
    if is_thre:
        return 2/pylab.norm(w[:-1]),w[:-1],-w[-1]
    else:
        return 2/pylab.norm(w),w
