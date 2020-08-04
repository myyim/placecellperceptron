import numpy as np
import pylab
import matplotlib as mpl
import scipy.stats
import scipy.optimize
import pickle
import math
import itertools
#import os.path

font_size = 14
mpl.rcParams['axes.titlesize'] = font_size
mpl.rcParams['xtick.labelsize'] = font_size-2
mpl.rcParams['ytick.labelsize'] = font_size-2
mpl.rcParams['axes.labelsize'] = font_size-2
mpl.rcParams['legend.fontsize'] = font_size-7
mpl.rcParams['font.size'] = font_size-1
new_rc_params = {'text.usetex': False,"svg.fonttype": 'none'}
mpl.rcParams.update(new_rc_params)
fs = 16

### Parameters
eps = 1e-8
### Figures
color = ['#1e90ff','#ff8c00','#3cb371']
color4 = ['#DC143C','#3cb371','#9932CC','#FFD700','#1e90ff','#ff8c00','#3cb371','m']

def phase(x,period):
    """phase(x,period) returns the phase of location x with respect to the module with spacing period."""
    return np.mod(x/period,1)

def grid(x,period,prefphase,phsh=0.,gridmodel='gau',sig=0.212):
    """grid(x,period,prefphase,phsh) returns the grid cell activity with prefphase at all location x."""
    if gridmodel == 'gau':
        temp_array = np.array((abs(phase(x-phsh*period,period)-prefphase),1-abs(phase(x-phsh*period,period)-prefphase)))
        temp = np.exp(-np.min(temp_array,axis=0)**2/(2*sig**2))
    elif gridmodel == 'del':
        temp = np.zeros(x.size)
        temp[int(np.round((phsh+prefphase)*period)):x.size:period] = 1
    return temp

def grid1d_orient(x,period,ori,xph=0.,yph=0.,sig=0.106,full=0,mode='exp'):
    r = np.zeros(x.size)
    xv = x*np.cos(ori)
    yv = x*np.sin(ori)
    if mode == 'exp':
        if full == 1:
            for n in range(-1,int(np.ceil(2*yv[-1]/(np.sqrt(3)*period)))+2):
                for m in range(-n/2-1,int(np.ceil(xv[-1]/period-n/2)+1)):
                    r += np.exp((-(xv-(m-xph+(n-yph)/2.)*period)**2-(yv-((n-yph)*np.sqrt(3)/2)*period)**2)/(2*(sig*period)**2))
        elif full == 0:
            for m in range(0,int(np.ceil(xv[-1]/period))+2):
                for n in range(0,m+2):
                    r += np.exp((-(xv-(m-xph+(n-yph)/2.)*period)**2-(yv-((n-yph)*np.sqrt(3)/2)*period)**2)/(2*(sig*period)**2))
    elif mode == 'cos':
        b = np.array([[0,2./np.sqrt(3)],[1.,-1./np.sqrt(3)],[1.,1./np.sqrt(3)]])/period
        for j in range(3):
            r += np.cos(2*np.pi*(b[j,0]*(xv-xph)+b[j,1]*(yv-yph)))/3.
        r += 1
    return r

def detect_field(a,nth,fmerge=10,fwidthmin=0):
    af = np.zeros(a.shape)
    tharr = []
    for j in range(a.shape[0]):
        th = np.mean(a[j,:]) + nth*np.std(a[j,:])
        tharr.append(th)
        idx = pylab.find(a[j,:]-th>=0)  # index of non-zero activity location
        if idx.size > 0:
            di = np.diff(idx)
            ii = idx[np.append([True],di>1)]   # index of the start bump location
            if ii.size > 1:
                bi = pylab.find(di>1)   # bump separating index
                bw = np.append(np.append(bi[0]+1,np.diff(bi)),di.size-bi[-1]) # bump width-1
            else:
                bw = [di.size + 1]
            fc = []
            for l in range(ii.size):
                if bw[l] >= fwidthmin:  # COM
                    fc.append(int(np.round(np.dot(np.arange(ii[l],ii[l]+bw[l]),a[j,ii[l]:ii[l]+bw[l]])/float(np.sum(a[j,ii[l]:ii[l]+bw[l]])))))
            dfc = np.diff(fc)
            if np.any(dfc<fmerge):
                imerge = pylab.find(dfc<fmerge)
                dimage = np.diff(imerge)
                for l in np.flipud(imerge):
                    bw[l] = ii[l+1]-ii[l]+bw[l+1]
                ii[imerge+1] = 0
                ii = ii[ii>0]
                bw[imerge+1] = 0
                bw = bw[bw>0]
                fc = []
                for l in range(ii.size):
                    if bw[l] >= fwidthmin:
                        fc.append(int(np.round(np.dot(np.arange(ii[l],ii[l]+bw[l]),a[j,ii[l]:ii[l]+bw[l]])/float(np.sum(a[j,ii[l]:ii[l]+bw[l]])))))
            af[j,fc] = 1
    return af,tharr

#def poster_ifi():
if 0:
    print 'Poster'
    Ng = 10  # dinstinct for each neuron
    Np = 20
    l = [31,43] #[53,87] #[31,43]
    R = 10000
    sig = 0.16
    bin = 1
    #ori = [0.2153,0.3162] # ori2
    #ori = [0.39,0.46] # ori1
    rng = np.random.RandomState(52)
    x = np.arange(R)
    px = rng.rand(Np,Ng)
    py = rng.rand(Np,Ng)
    w = rng.lognormal(0,0.25,size=[Np,Ng])
    fig = pylab.figure(figsize=[10,4])
    # 1D
    ax = pylab.subplot(121)
    ori = [0,0]
    nth = 2
    v = np.zeros(((Np,Ng,R)))
    a = np.zeros((Np,R))
    acf = np.zeros((Np,2*R/bin-1))
    for iN in range(Np):
        for iM in range(Ng):
            v[iN,iM,:] = grid1d_orient(x,l[iM/(Ng/len(l))],ori[iM/(Ng/len(l))],px[iN,iM],py[iN,iM],sig=sig)
            #v[iN,iM,:] = grid(x,l[iN],float(iM)/l[iN],phsh=0.,gridmodel='del'')    #
            a[iN,:] += w[iN,iM]*v[iN,iM,:]
    print '1D'
    af,tharr = detect_field(a,nth,fmerge=10,fwidthmin=0)
    fid = []
    for iN in range(Np):
        fid.append(list(pylab.find(af[iN,:]==1)))
    ifiall = []
    for iN in range(Np):
        ifiall.extend(np.diff(fid[iN]))
    y = np.histogram(ifiall,np.arange(0.5,np.max(ifiall)+1,bin))
    for j in range(len(l)):
        for k in range(1,10):
            pylab.plot([k*l[j]]*2,[0,max(y[0])+1],'--',c=color[j],lw=0.5)
    pylab.hist(ifiall,np.arange(0.5,np.max(ifiall)+1,bin),color='k')
    pylab.xlim(0,300)
    pylab.xlabel('IFI')
    pylab.ylabel('number')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # slices through 2D
    ax = pylab.subplot(122)
    nth = 2
    ori = [0.39,0.46]
    v = np.zeros(((Np,Ng,R)))
    a = np.zeros((Np,R))
    acf = np.zeros((Np,2*R/bin-1))
    for iN in range(Np):
        for iM in range(Ng):
            v[iN,iM,:] = grid1d_orient(x,l[iM/(Ng/len(l))],ori[iM/(Ng/len(l))],px[iN,iM],py[iN,iM],sig=sig)
            #v[iN,iM,:] = grid(x,l[iN],float(iM)/l[iN],phsh=0.,gridmodel='del'')    #
            a[iN,:] += w[iN,iM]*v[iN,iM,:]
    print '2D'
    af,tharr = detect_field(a,nth,fmerge=10,fwidthmin=0)
    fid = []
    for iN in range(Np):
        fid.append(list(pylab.find(af[iN,:]==1)))
    ifiall = []
    for iN in range(Np):
        ifiall.extend(np.diff(fid[iN]))
    y = np.histogram(ifiall,np.arange(0.5,np.max(ifiall)+1,bin))
    j = 0
    pylab.plot([l[j]*np.sqrt(3)]*2,[0,max(y[0])+1],'--',c=color[j],lw=0.5)
    pylab.plot([l[j]*np.sqrt(7)]*2,[0,max(y[0])+1],'--',c=color[j],lw=0.5)
    pylab.plot([l[j]*np.sqrt(3)+l[j]*np.sqrt(7)]*2,[0,max(y[0])+1],'--',c=color[j],lw=0.5)
    pylab.plot([l[j]*np.sqrt(3)+2*l[j]*np.sqrt(7)]*2,[0,max(y[0])+1],'--',c=color[j],lw=0.5)
    j = 1
    pylab.plot([l[j]*np.sqrt(3)]*2,[0,max(y[0])+1],'--',c=color[j],lw=0.5)
    pylab.plot([l[j]*np.sqrt(3)+l[j]*np.sqrt(7)]*2,[0,max(y[0])+1],'--',c=color[j],lw=0.5)
    pylab.plot([2*l[j]*np.sqrt(3)+l[j]*np.sqrt(7)]*2,[0,max(y[0])+1],'--',c=color[j],lw=0.5)
    pylab.hist(ifiall,np.arange(0.5,np.max(ifiall)+1,bin),color='k')
    pylab.xlim(0,300)
    pylab.xlabel('IFI')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    pylab.subplots_adjust(left=0.1,top=0.9,right=0.95,bottom=0.17,hspace=0.5,wspace=0.35)
    #pylab.savefig('ifi'+str(Np)+'.svg')

if 1:
    pylab.figure(figsize=[8,5])
    rng = np.random.RandomState(52)
    wp1 = 0
    wp2 = 0.25
    bin = 1
    l = [31,43,59]
    Ng = 15
    Np = 10
    sig = 0.16
    ori = [0.39,0.46,0.03]
    R = l[0]
    for j in range(len(l)-1):
        R *= l[j+1]
    x = np.arange(R)
    p1d = []
    for j in range(Np):
        #temp = rng.choice(range(l[0]),Ng/len(l),replace=False)/float(l[0])
        temp = np.arange(0,l[0],int(np.round(l[0]/float(Ng/len(l)))))[:Ng/len(l)]/float(l[0])
        for iM in range(1,len(l)):
            #temp = np.append(temp,rng.choice(range(l[iM]),Ng/len(l),replace=False)/float(l[iM]))
            temp = np.append(temp,np.arange(0,l[iM],int(np.round(l[iM]/float(Ng/len(l)))))[:Ng/len(l)]/float(l[iM]))
        p1d.append(list(temp))
    p1d = np.array(p1d)
    px = rng.rand(Np,Ng)
    py = rng.rand(Np,Ng)
    w = rng.lognormal(wp1,wp2,size=[Np,Ng])
    v = np.zeros(((Np,Ng,R)))
    a = np.zeros((Np,R))
    print '1D'
    for iN in range(Np):
        for iM in range(Ng):
            v[iN,iM,:] = grid(x,l[iM/(Ng/len(l))],p1d[iN,iM],phsh=0.,gridmodel='gau',sig=sig)
        a[iN,:] = np.dot(w[iN,:],v[iN,:,:])
    # 1D
    nth = 1.9
    af,tharr = detect_field(a,nth,fmerge=10,fwidthmin=0)
    print np.sum(af)
    fid = []
    for iN in range(Np):
        fid.append(list(pylab.find(af[iN,:]==1)))
    ifiall = []
    for iN in range(1):
        ifiall.extend(np.diff(fid[iN]))
    y = np.histogram(ifiall,np.arange(0.5,np.max(ifiall)+1,bin))
    ax = pylab.subplot(211)
    for j in range(Ng):
        pylab.plot(range(1000),v[0,j,:1000]*0.8+j,'k')
    for j in fid[0]:
        pylab.plot([j]*2,[0,Ng],'r--',lw=1)
    pylab.xlim(0,1000)
    ax = pylab.subplot(234)
    for j in range(len(l)):
        for k in range(1,10):
            pylab.plot([k*l[j]]*2,[0,max(y[0])+1],'--',c=color[j],lw=0.5)
    pylab.hist(ifiall,np.arange(0.5,np.max(ifiall)+1,bin),color='k')
    pylab.xlim(0,300)
    pylab.xlabel('IFI')
    #pylab.title('mean+'+str(nth)+'*std')
    ax = pylab.subplot(235)
    count = np.zeros(Ng)
    for j in range(Ng):
        temp = pylab.find(v[0,j,:]>0.95)
        for k in fid[0]:
            count[j] += np.any(k==temp)
    pylab.plot(w[0],count/float(len(fid[0])),'+')
    pylab.xlabel('$w$')
    pylab.ylabel('prob of coincidence')
    ax = pylab.subplot(236)
    count = np.zeros((Np,Ng))
    for iN in range(Np):
        for j in range(Ng):
            temp = pylab.find(v[iN,j,:]>0.95)
            for k in fid[iN]:
                count[iN,j] += np.any(k==temp)
    pylab.plot(w.ravel(),count.ravel()/float(len(fid[0])),'+')
    pylab.xlabel('$w$')
    pylab.subplots_adjust(wspace=0.4,hspace=0.4)
    """
    # slices through 2D
    print '2D'
    l = [31,43]
    ori = [0.39,0.46]
    R = 10000
    x = np.arange(R)
    v = np.zeros(((Np,Ng,R)))
    a = np.zeros((Np,R))
    for iN in range(Np):
        for iM in range(Ng):
            v[iN,iM,:] = grid1d_orient(x,l[iM/(Ng/len(l))],ori[iM/(Ng/len(l))],px[iN,iM],py[iN,iM],sig=sig)
        a[iN,:] = np.dot(w[iN,:],v[iN,:,:])
    for m in range(4):
        nth = 1.4+0.2*m
        ax = pylab.subplot(2,4,m+5)
        af,tharr = detect_field(a,nth,fmerge=10,fwidthmin=0)
        print np.sum(af)
        fid = []
        for iN in range(Np):
            fid.append(list(pylab.find(af[iN,:]==1)))
        ifiall = []
        for iN in range(Np):
            ifiall.extend(np.diff(fid[iN]))
        y = np.histogram(ifiall,np.arange(0.5,np.max(ifiall)+1,bin))
        ymax = max(y[0])
        j = 0
        pylab.plot([l[j]]*2,[0,ymax+1],'--',c=color[j],lw=0.5)  # less likely
        #pylab.plot([l[j]*np.sqrt(3)]*2,[0,ymax+1],'--',c=color[j],lw=0.5)
        pylab.plot([l[j]*np.sqrt(7)]*2,[0,ymax+1],'--',c=color[j],lw=0.5)
        #pylab.plot([l[j]+l[j]*np.sqrt(3)]*2,[0,ymax+1],'--',c=color[j],lw=0.5)
        #pylab.plot([l[j]+l[j]*np.sqrt(7)]*2,[0,ymax+1],'--',c=color[j],lw=0.5)
        #pylab.plot([l[j]*np.sqrt(3)+l[j]*np.sqrt(7)]*2,[0,ymax+1],'--',c=color[j],lw=0.5)
        pylab.plot([2*l[j]+l[j]*np.sqrt(3)]*2,[0,ymax+1],'--',c=color[j],lw=0.5)
        pylab.plot([l[j]+2*l[j]*np.sqrt(7)]*2,[0,ymax+1],'--',c=color[j],lw=0.5)
        j = 1
        pylab.plot([l[j]*np.sqrt(3)]*2,[0,ymax+1],'--',c=color[j],lw=0.5)
        pylab.plot([l[j]*np.sqrt(7)]*2,[0,ymax+1],'--',c=color[j],lw=0.5)
        pylab.plot([l[j]+2*l[j]*np.sqrt(3)]*2,[0,ymax+1],'--',c=color[j],lw=0.5)
        pylab.plot([l[j]+2*l[j]*np.sqrt(7)]*2,[0,ymax+1],'--',c=color[j],lw=0.5)
        pylab.hist(ifiall,np.arange(0.5,np.max(ifiall)+1,bin),color='k')
        pylab.xlim(0,300)
        pylab.xlabel('IFI')
        #pylab.title('1D slices through 2D model')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)"""

pylab.show()
